#include "voxcpm/VoxCpmClient.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <string_view>
#include <utility>

namespace voxstudio::voxcpm {
namespace {

constexpr std::chrono::seconds kHealthTimeout{5};
constexpr std::chrono::seconds kRenderTimeout{180};

[[nodiscard]] std::string joinedUrl(const std::string& baseUrl, const std::string& path) {
    if (baseUrl.empty()) {
        return path;
    }
    if (baseUrl.back() == '/' && !path.empty() && path.front() == '/') {
        return baseUrl.substr(0, baseUrl.size() - 1) + path;
    }
    if (baseUrl.back() != '/' && !path.empty() && path.front() != '/') {
        return baseUrl + "/" + path;
    }
    return baseUrl + path;
}

[[nodiscard]] core::Error clientError(const std::string& message) {
    return core::makeError(core::ErrorCode::InvalidArgument, message);
}

[[nodiscard]] std::string headerValue(const cpr::Header& headers, const std::string& name) {
    const auto found = headers.find(name);
    return found == headers.end() ? std::string{} : found->second;
}

[[nodiscard]] int integerHeader(const cpr::Header& headers, const std::string& name,
                                const int fallback) {
    const auto value = headerValue(headers, name);
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

} // namespace

CprVoxCpmHttpTransport::CprVoxCpmHttpTransport(std::string baseUrl)
    : m_baseUrl(std::move(baseUrl)) {}

core::Expected<VoxCpmHttpResponse> CprVoxCpmHttpTransport::getJson(const std::string& path) const {
    try {
        const auto response =
            cpr::Get(cpr::Url{joinedUrl(m_baseUrl, path)},
                     cpr::Header{{"Accept", "application/json"}}, cpr::Timeout{kHealthTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return core::makeError(core::ErrorCode::FileSystemFailure, response.error.message);
        }
        return VoxCpmHttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

core::Expected<VoxCpmHttpResponse>
CprVoxCpmHttpTransport::postPerformance(const std::string& path,
                                        const VoxCpmRenderRequest& request) const {
    try {
        cpr::Multipart multipart{{"audio",
                                  cpr::Buffer{request.pcm16Audio.begin(), request.pcm16Audio.end(),
                                              cpr::fs::path{"voxstudio-performance.pcm"}},
                                  "application/octet-stream"},
                                 {"voice_id", request.voiceId},
                                 {"sample_rate", std::to_string(request.sampleRate)},
                                 {"channels", std::to_string(request.channels)},
                                 {"transcript", request.transcript}};

        const auto response =
            cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)}, cpr::Header{{"Accept", "audio/L16"}},
                      std::move(multipart), cpr::Timeout{kRenderTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return core::makeError(core::ErrorCode::FileSystemFailure, response.error.message);
        }

        VoxCpmHttpResponse result;
        result.statusCode = static_cast<int>(response.status_code);
        result.body = response.text;
        result.transcript = headerValue(response.header, "x-vox-transcript");
        result.characterName = headerValue(response.header, "x-vox-character");
        result.sampleRate = integerHeader(response.header, "x-vox-sample-rate", 48000);
        result.latencyMs = integerHeader(response.header, "x-vox-latency-ms", 0);
        result.delivery = headerValue(response.header, "x-vox-delivery");
        result.pronunciations = headerValue(response.header, "x-vox-pronunciations");
        return result;
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

VoxCpmClient::VoxCpmClient(std::string endpoint)
    : VoxCpmClient(endpoint, std::make_unique<CprVoxCpmHttpTransport>(endpoint)) {}

VoxCpmClient::VoxCpmClient(std::string endpoint, std::unique_ptr<IVoxCpmHttpTransport> transport)
    : m_endpoint(std::move(endpoint)), m_transport(std::move(transport)) {}

core::Expected<VoxCpmHealth> VoxCpmClient::health() const {
    if (m_transport == nullptr) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "VoxCPM2 HTTP transport is not configured.");
    }

    auto response = m_transport->getJson(voxCpmHealthPath());
    if (!response) {
        return response.error();
    }
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return core::makeError(core::ErrorCode::FileSystemFailure, response.value().body);
    }

    try {
        const auto json = nlohmann::json::parse(
            response.value().body.empty() ? std::string{"{}"} : response.value().body);
        VoxCpmHealth health;
        health.ok = json.value("ok", true);
        health.cudaAvailable = json.value("cuda_available", false);
        health.modelLoaded = json.value("model_loaded", false);
        health.transcriberLoaded = json.value("transcriber_loaded", false);
        health.profileCount = json.value("profile_count", 0);
        health.engine = json.value("engine", std::string{"VoxCPM2"});
        health.message = json.value("message", std::string{"VoxCPM2 sidecar healthy."});
        return health;
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

core::Expected<VoxCpmRenderResult>
VoxCpmClient::renderPerformance(const VoxCpmRenderRequest& request) const {
    if (request.voiceId.empty()) {
        return clientError("Select a VoxCPM2 character profile first.");
    }
    if (request.pcm16Audio.empty() || (request.pcm16Audio.size() % 2U) != 0U) {
        return clientError("The recorded performance contains no complete audio samples.");
    }
    if (request.sampleRate <= 0 || request.channels != 1) {
        return clientError("VoxCPM2 performance audio must be mono with a valid sample rate.");
    }
    if (m_transport == nullptr) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "VoxCPM2 HTTP transport is not configured.");
    }

    auto response = m_transport->postPerformance(voxCpmRenderPath(), request);
    if (!response) {
        return response.error();
    }
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return core::makeError(core::ErrorCode::FileSystemFailure, response.value().body);
    }
    if (response.value().body.empty()) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "VoxCPM2 returned no character audio.");
    }

    const auto& body = response.value().body;
    VoxCpmRenderResult result;
    result.pcm16Audio.assign(reinterpret_cast<const std::uint8_t*>(body.data()),
                             reinterpret_cast<const std::uint8_t*>(body.data() + body.size()));
    result.transcript = response.value().transcript;
    result.characterName = response.value().characterName;
    result.sampleRate = response.value().sampleRate;
    result.latencyMs = response.value().latencyMs;
    result.delivery = response.value().delivery;
    result.pronunciations = response.value().pronunciations;
    return result;
}

std::string voxCpmHealthPath() {
    return "/health";
}

std::string voxCpmRenderPath() {
    return "/render_performance";
}

} // namespace voxstudio::voxcpm
