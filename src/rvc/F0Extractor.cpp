#include "rvc/F0Extractor.h"

#include <filesystem>
#include <system_error>

namespace voxstudio::rvc {
namespace {

[[nodiscard]] bool fileExists(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error) && !error;
}

} // namespace

core::Expected<bool> F0Extractor::configure(const F0ExtractorConfig& config) {
    if (config.rmvpeModelPath.empty() || config.rmvpeModelPath.extension() != ".onnx") {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "RMVPE F0 extractor requires an .onnx model.");
    }
    if (!fileExists(config.rmvpeModelPath)) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "RMVPE model file does not exist.");
    }
    if (config.sampleRate <= 0 || config.hopLength <= 0) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "F0 extractor sample rate and hop length must be positive.");
    }

    m_config = config;
    m_configured = true;
    return true;
}

bool F0Extractor::isConfigured() const noexcept {
    return m_configured;
}

const F0ExtractorConfig& F0Extractor::config() const noexcept {
    return m_config;
}

core::Expected<std::string> F0Extractor::describe() const {
    if (!m_configured) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "F0 extractor is not configured.");
    }
    return "RMVPE F0 extractor: " + m_config.rmvpeModelPath.string();
}

} // namespace voxstudio::rvc
