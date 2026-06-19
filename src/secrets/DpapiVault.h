#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <string>

namespace voxstudio::secrets {

class DpapiVault final {
public:
    DpapiVault();
    explicit DpapiVault(std::filesystem::path secretFilePath);

    [[nodiscard]] core::Expected<bool> storeElevenLabsApiKey(const std::string& apiKey) const;
    [[nodiscard]] core::Expected<std::string> loadElevenLabsApiKey() const;
    [[nodiscard]] bool hasElevenLabsApiKey() const;
    [[nodiscard]] core::Expected<bool> deleteElevenLabsApiKey() const;
    [[nodiscard]] const std::filesystem::path& secretFilePath() const noexcept;

private:
    std::filesystem::path m_secretFilePath;
};

} // namespace voxstudio::secrets

