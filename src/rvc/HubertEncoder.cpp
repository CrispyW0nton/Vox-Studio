#include "rvc/HubertEncoder.h"

#include <filesystem>
#include <system_error>

namespace voxstudio::rvc {
namespace {

[[nodiscard]] bool fileExists(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error) && !error;
}

} // namespace

core::Expected<bool> HubertEncoder::configure(const HubertEncoderConfig& config) {
    if (config.hubertModelPath.empty() || config.hubertModelPath.extension() != ".onnx") {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "HuBERT encoder requires an .onnx model.");
    }
    if (!fileExists(config.hubertModelPath)) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "HuBERT model file does not exist.");
    }
    if (config.sampleRate <= 0 || config.featureSize <= 0) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "HuBERT sample rate and feature size must be positive.");
    }

    m_config = config;
    m_configured = true;
    return true;
}

bool HubertEncoder::isConfigured() const noexcept {
    return m_configured;
}

const HubertEncoderConfig& HubertEncoder::config() const noexcept {
    return m_config;
}

core::Expected<std::string> HubertEncoder::describe() const {
    if (!m_configured) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "HuBERT encoder is not configured.");
    }
    return "HuBERT encoder: " + m_config.hubertModelPath.string();
}

} // namespace voxstudio::rvc
