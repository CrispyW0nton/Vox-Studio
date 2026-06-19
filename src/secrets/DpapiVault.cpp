#include "secrets/DpapiVault.h"

#include "platform/win/AppPaths.h"

#include <Windows.h>
#include <dpapi.h>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace voxstudio::secrets {
namespace {

constexpr auto kSecretsDirectoryName = "secrets";
constexpr auto kElevenLabsSecretFileName = "elevenlabs.bin";
constexpr auto kEntropyText = "VoxStudio.ElevenLabs.ApiKey.v1";

class LocalBlob final {
public:
    explicit LocalBlob(DATA_BLOB blob) noexcept
        : m_blob(blob) {}

    ~LocalBlob() noexcept {
        if (m_blob.pbData != nullptr) {
            LocalFree(m_blob.pbData);
        }
    }

    LocalBlob(const LocalBlob&) = delete;
    LocalBlob& operator=(const LocalBlob&) = delete;
    LocalBlob(LocalBlob&&) = delete;
    LocalBlob& operator=(LocalBlob&&) = delete;

    [[nodiscard]] const DATA_BLOB& get() const noexcept {
        return m_blob;
    }

private:
    DATA_BLOB m_blob{};
};

[[nodiscard]] std::filesystem::path defaultSecretFilePath() {
    const auto dataPath = platform::win::voxStudioDataPath();
    if (!dataPath.has_value()) {
        return {};
    }

    return *dataPath / kSecretsDirectoryName / kElevenLabsSecretFileName;
}

[[nodiscard]] DATA_BLOB blobFromBytes(std::vector<unsigned char>& bytes) noexcept {
    return DATA_BLOB{static_cast<DWORD>(bytes.size()), bytes.data()};
}

[[nodiscard]] DATA_BLOB blobFromString(const std::string& value) noexcept {
    return DATA_BLOB{static_cast<DWORD>(value.size()),
                     reinterpret_cast<BYTE*>(const_cast<char*>(value.data()))};
}

[[nodiscard]] std::vector<unsigned char> entropyBytes() {
    const std::string entropy{kEntropyText};
    return {entropy.begin(), entropy.end()};
}

[[nodiscard]] core::Error secretError(const std::string& message) {
    return core::makeError(core::ErrorCode::SecretStorageFailed, message);
}

} // namespace

DpapiVault::DpapiVault()
    : m_secretFilePath(defaultSecretFilePath()) {}

DpapiVault::DpapiVault(std::filesystem::path secretFilePath)
    : m_secretFilePath(std::move(secretFilePath)) {}

core::Expected<bool> DpapiVault::storeElevenLabsApiKey(const std::string& apiKey) const {
    if (apiKey.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument, "API key must not be empty.");
    }
    if (m_secretFilePath.empty()) {
        return secretError("Unable to resolve the Vox Studio secrets directory.");
    }

    try {
        std::filesystem::create_directories(m_secretFilePath.parent_path());

        auto entropy = entropyBytes();
        DATA_BLOB input = blobFromString(apiKey);
        DATA_BLOB entropyBlob = blobFromBytes(entropy);
        DATA_BLOB encrypted{};

        if (CryptProtectData(&input, nullptr, &entropyBlob, nullptr, nullptr, 0, &encrypted) == 0) {
            return secretError("CryptProtectData failed.");
        }

        const LocalBlob encryptedBlob{encrypted};
        std::ofstream output{m_secretFilePath, std::ios::binary | std::ios::trunc};
        if (!output) {
            return secretError("Unable to open the ElevenLabs secret file for writing.");
        }

        const auto* data = reinterpret_cast<const char*>(encryptedBlob.get().pbData);
        output.write(data, static_cast<std::streamsize>(encryptedBlob.get().cbData));
        if (!output) {
            return secretError("Unable to write the ElevenLabs secret file.");
        }

        return true;
    } catch (const std::filesystem::filesystem_error& exception) {
        return secretError(exception.what());
    } catch (const std::exception& exception) {
        return secretError(exception.what());
    }
}

core::Expected<std::string> DpapiVault::loadElevenLabsApiKey() const {
    if (m_secretFilePath.empty()) {
        return secretError("Unable to resolve the Vox Studio secrets directory.");
    }

    try {
        std::ifstream input{m_secretFilePath, std::ios::binary};
        if (!input) {
            return core::makeError(core::ErrorCode::ProjectNotFound,
                                   "ElevenLabs API key has not been saved.");
        }

        std::vector<unsigned char> encrypted{std::istreambuf_iterator<char>{input},
                                             std::istreambuf_iterator<char>{}};
        if (encrypted.empty()) {
            return secretError("ElevenLabs secret file is empty.");
        }

        auto entropy = entropyBytes();
        DATA_BLOB inputBlob = blobFromBytes(encrypted);
        DATA_BLOB entropyBlob = blobFromBytes(entropy);
        DATA_BLOB decrypted{};

        if (CryptUnprotectData(&inputBlob, nullptr, &entropyBlob, nullptr, nullptr, 0,
                               &decrypted) == 0) {
            return secretError("CryptUnprotectData failed.");
        }

        const LocalBlob decryptedBlob{decrypted};
        return std::string{reinterpret_cast<const char*>(decryptedBlob.get().pbData),
                           static_cast<std::size_t>(decryptedBlob.get().cbData)};
    } catch (const std::filesystem::filesystem_error& exception) {
        return secretError(exception.what());
    } catch (const std::exception& exception) {
        return secretError(exception.what());
    }
}

bool DpapiVault::hasElevenLabsApiKey() const {
    return !m_secretFilePath.empty() && std::filesystem::exists(m_secretFilePath);
}

core::Expected<bool> DpapiVault::deleteElevenLabsApiKey() const {
    if (m_secretFilePath.empty()) {
        return secretError("Unable to resolve the Vox Studio secrets directory.");
    }

    try {
        if (!std::filesystem::exists(m_secretFilePath)) {
            return true;
        }

        return std::filesystem::remove(m_secretFilePath);
    } catch (const std::filesystem::filesystem_error& exception) {
        return secretError(exception.what());
    }
}

const std::filesystem::path& DpapiVault::secretFilePath() const noexcept {
    return m_secretFilePath;
}

} // namespace voxstudio::secrets

