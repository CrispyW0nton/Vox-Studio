#include "secrets/DpapiVault.h"
#include "support/TemporarySecretFile.h"

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <dpapi.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::string readBinaryFile(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

void writeLegacySecret(const std::filesystem::path& path, const std::string& apiKey) {
    const std::string entropyText{"VoxStudio.ElevenLabs.ApiKey.v1"};
    std::vector<unsigned char> entropy{entropyText.begin(), entropyText.end()};
    DATA_BLOB input{
        static_cast<DWORD>(apiKey.size()),
        reinterpret_cast<BYTE*>(const_cast<char*>(apiKey.data()))};
    DATA_BLOB entropyBlob{static_cast<DWORD>(entropy.size()), entropy.data()};
    DATA_BLOB encrypted{};

    REQUIRE(CryptProtectData(
                &input, nullptr, &entropyBlob, nullptr, nullptr, 0, &encrypted) != 0);
    {
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        REQUIRE(output.good());
        output.write(reinterpret_cast<const char*>(encrypted.pbData),
                     static_cast<std::streamsize>(encrypted.cbData));
        REQUIRE(output.good());
    }
    LocalFree(encrypted.pbData);
}

} // namespace

TEST_CASE("DPAPI vault encrypts and restores the ElevenLabs API key", "[secrets][dpapi]") {
    const voxstudio::test::TemporarySecretFile secretFile{"voxstudio_dpapi_test_"};
    const voxstudio::secrets::DpapiVault vault{secretFile.secretPath()};
    const std::string apiKey = "sk_test_key_123";

    auto stored = vault.storeValidatedElevenLabsApiKey(apiKey);
    REQUIRE(stored.hasValue());
    CHECK(stored.value());
    REQUIRE(std::filesystem::exists(secretFile.secretPath()));

    const std::string encryptedBytes = readBinaryFile(secretFile.secretPath());
    CHECK(!encryptedBytes.empty());
    CHECK(encryptedBytes.find(apiKey) == std::string::npos);

    auto loaded = vault.loadElevenLabsApiKey();
    REQUIRE(loaded.hasValue());
    CHECK(loaded.value() == apiKey);

    auto deleted = vault.deleteElevenLabsApiKey();
    REQUIRE(deleted.hasValue());
    CHECK(deleted.value());
    CHECK(!std::filesystem::exists(secretFile.secretPath()));
}

TEST_CASE("DPAPI vault only reports a usable ElevenLabs credential", "[secrets][dpapi]") {
    const voxstudio::test::TemporarySecretFile secretFile{"voxstudio_dpapi_test_"};
    const voxstudio::secrets::DpapiVault vault{secretFile.secretPath()};

    CHECK_FALSE(vault.hasElevenLabsApiKey());

    {
        std::ofstream corruptSecret{secretFile.secretPath(), std::ios::binary};
        corruptSecret << "not a DPAPI payload";
    }
    CHECK_FALSE(vault.hasElevenLabsApiKey());

    auto stored =
        vault.storeValidatedElevenLabsApiKey("test_key_for_current_windows_user");
    REQUIRE(stored.hasValue());
    CHECK(vault.hasElevenLabsApiKey());
}

TEST_CASE("DPAPI vault rejects legacy unvalidated ElevenLabs credentials",
          "[secrets][dpapi]") {
    const voxstudio::test::TemporarySecretFile secretFile{"voxstudio_dpapi_test_"};
    const voxstudio::secrets::DpapiVault vault{secretFile.secretPath()};

    writeLegacySecret(secretFile.secretPath(), "legacy_unvalidated_key");

    CHECK_FALSE(vault.hasElevenLabsApiKey());
    const auto loaded = vault.loadElevenLabsApiKey();
    REQUIRE_FALSE(loaded.hasValue());
    CHECK(loaded.error().message.find("reconnected and validated") != std::string::npos);
}
