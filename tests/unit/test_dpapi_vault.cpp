#include "secrets/DpapiVault.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

class TemporarySecretFile final {
public:
    TemporarySecretFile() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_directory = std::filesystem::temp_directory_path() /
                      ("voxstudio_dpapi_test_" + std::to_string(now));
        std::filesystem::create_directories(m_directory);
        m_secretPath = m_directory / "elevenlabs.bin";
    }

    ~TemporarySecretFile() {
        std::error_code error;
        std::filesystem::remove_all(m_directory, error);
    }

    TemporarySecretFile(const TemporarySecretFile&) = delete;
    TemporarySecretFile& operator=(const TemporarySecretFile&) = delete;
    TemporarySecretFile(TemporarySecretFile&&) = delete;
    TemporarySecretFile& operator=(TemporarySecretFile&&) = delete;

    [[nodiscard]] const std::filesystem::path& secretPath() const noexcept {
        return m_secretPath;
    }

private:
    std::filesystem::path m_directory;
    std::filesystem::path m_secretPath;
};

[[nodiscard]] std::string readBinaryFile(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("DPAPI vault encrypts and restores the ElevenLabs API key", "[secrets][dpapi]") {
    const TemporarySecretFile secretFile;
    const voxstudio::secrets::DpapiVault vault{secretFile.secretPath()};
    const std::string apiKey = "sk_test_key_123";

    auto stored = vault.storeElevenLabsApiKey(apiKey);
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

