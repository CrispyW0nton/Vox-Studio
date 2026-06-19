#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <string>

namespace voxstudio::rvc {

struct HubertEncoderConfig final {
    std::filesystem::path hubertModelPath;
    int sampleRate{16000};
    int featureSize{256};
};

class HubertEncoder final {
public:
    [[nodiscard]] core::Expected<bool> configure(const HubertEncoderConfig& config);
    [[nodiscard]] bool isConfigured() const noexcept;
    [[nodiscard]] const HubertEncoderConfig& config() const noexcept;
    [[nodiscard]] core::Expected<std::string> describe() const;

private:
    HubertEncoderConfig m_config;
    bool m_configured{false};
};

} // namespace voxstudio::rvc
