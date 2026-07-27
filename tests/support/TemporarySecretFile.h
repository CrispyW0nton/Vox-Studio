#pragma once

#include <chrono>
#include <filesystem>
#include <string>

namespace voxstudio::test {

class TemporarySecretFile final {
public:
    explicit TemporarySecretFile(const std::string& directoryPrefix) {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_directory = std::filesystem::temp_directory_path() /
                      (directoryPrefix + std::to_string(now));
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

} // namespace voxstudio::test
