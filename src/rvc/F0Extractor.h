#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <string>

namespace voxstudio::rvc {

struct F0ExtractorConfig final {
    std::filesystem::path rmvpeModelPath;
    int sampleRate{48000};
    int hopLength{160};
};

class F0Extractor final {
public:
    [[nodiscard]] core::Expected<bool> configure(const F0ExtractorConfig& config);
    [[nodiscard]] bool isConfigured() const noexcept;
    [[nodiscard]] const F0ExtractorConfig& config() const noexcept;
    [[nodiscard]] core::Expected<std::string> describe() const;

private:
    F0ExtractorConfig m_config;
    bool m_configured{false};
};

} // namespace voxstudio::rvc
