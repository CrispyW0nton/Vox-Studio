#include "voxcpm/VoxCpmClient.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <string_view>
#include <utility>

namespace voxstudio::voxcpm {
namespace {

constexpr std::chrono::seconds kHealthTimeout{5};
constexpr std::chrono::seconds kStoryTimeout{15};
constexpr std::chrono::seconds kRenderTimeout{180};
constexpr std::chrono::seconds kTextRenderTimeout{600};

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

[[nodiscard]] std::size_t utf8CodePointCount(const std::string& value) {
    return static_cast<std::size_t>(
        std::count_if(value.begin(), value.end(), [](const unsigned char byte) {
            return (byte & 0xC0U) != 0x80U;
        }));
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
        result.adapter = headerValue(response.header, "x-vox-adapter");
        result.sectionCount = integerHeader(response.header, "x-vox-section-count", 1);
        return result;
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

core::Expected<VoxCpmHttpResponse>
CprVoxCpmHttpTransport::postText(const std::string& path, const VoxCpmTextRequest& request) const {
    try {
        cpr::Multipart multipart{{"voice_id", request.voiceId},
                                 {"text", request.text},
                                 {"delivery", request.delivery},
                                 {"mode", request.mode}};

        const auto response =
            cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)}, cpr::Header{{"Accept", "audio/L16"}},
                      std::move(multipart), cpr::Timeout{kTextRenderTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return core::makeError(core::ErrorCode::FileSystemFailure, response.error.message);
        }

        VoxCpmHttpResponse result;
        result.statusCode = static_cast<int>(response.status_code);
        result.body = response.text;
        result.characterName = headerValue(response.header, "x-vox-character");
        result.sampleRate = integerHeader(response.header, "x-vox-sample-rate", 48000);
        result.latencyMs = integerHeader(response.header, "x-vox-latency-ms", 0);
        result.delivery = headerValue(response.header, "x-vox-delivery");
        result.pronunciations = headerValue(response.header, "x-vox-pronunciations");
        result.adapter = headerValue(response.header, "x-vox-adapter");
        result.performanceMode = headerValue(response.header, "x-vox-performance-mode");
        result.sectionCount = integerHeader(response.header, "x-vox-section-count", 1);
        return result;
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

core::Expected<VoxCpmHttpResponse>
CprVoxCpmHttpTransport::postStoryPlan(const std::string& path,
                                      const VoxCpmStoryRequest& request) const {
    try {
        cpr::Multipart multipart{{"text", request.text}, {"delivery", request.delivery}};
        const auto response = cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)},
                                        cpr::Header{{"Accept", "application/json"}},
                                        std::move(multipart), cpr::Timeout{kStoryTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return core::makeError(core::ErrorCode::FileSystemFailure, response.error.message);
        }
        return VoxCpmHttpResponse{static_cast<int>(response.status_code), response.text};
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
    result.adapter = response.value().adapter;
    return result;
}

core::Expected<VoxCpmTextResult> VoxCpmClient::renderText(const VoxCpmTextRequest& request) const {
    if (request.voiceId.empty()) {
        return clientError("Select a VoxCPM2 character profile first.");
    }
    if (request.text.empty()) {
        return clientError("Enter text to synthesize.");
    }
    if (utf8CodePointCount(request.text) > 10000U) {
        return clientError("Text-to-speech captures are limited to 10,000 characters.");
    }
    if (m_transport == nullptr) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "VoxCPM2 HTTP transport is not configured.");
    }

    auto response = m_transport->postText(voxCpmTextPath(), request);
    if (!response) {
        return response.error();
    }
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return core::makeError(core::ErrorCode::FileSystemFailure, response.value().body);
    }
    if (response.value().body.empty()) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "VoxCPM2 returned no text-to-speech audio.");
    }
    if (response.value().performanceMode.empty()) {
        return clientError(
            "The installed VoxCPM2 service is outdated. Refresh the local voice engine and try "
            "again.");
    }

    const auto& body = response.value().body;
    VoxCpmTextResult result;
    result.pcm16Audio.assign(reinterpret_cast<const std::uint8_t*>(body.data()),
                             reinterpret_cast<const std::uint8_t*>(body.data() + body.size()));
    result.characterName = response.value().characterName;
    result.sampleRate = response.value().sampleRate;
    result.latencyMs = response.value().latencyMs;
    result.delivery = response.value().delivery;
    result.pronunciations = response.value().pronunciations;
    result.adapter = response.value().adapter;
    result.performanceMode = response.value().performanceMode;
    result.sectionCount = response.value().sectionCount;
    return result;
}

core::Expected<VoxCpmStoryResult>
VoxCpmClient::analyzeStory(const VoxCpmStoryRequest& request) const {
    if (request.text.empty()) {
        return clientError("Enter a story to analyze.");
    }
    if (utf8CodePointCount(request.text) > 10000U) {
        return clientError("Storytelling captures are limited to 10,000 characters.");
    }
    if (m_transport == nullptr) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "VoxCPM2 HTTP transport is not configured.");
    }

    auto response = m_transport->postStoryPlan(voxCpmStoryPath(), request);
    if (!response) {
        return response.error();
    }
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return core::makeError(core::ErrorCode::FileSystemFailure, response.value().body);
    }

    try {
        const auto json = nlohmann::json::parse(
            response.value().body.empty() ? std::string{"{}"} : response.value().body);
        VoxCpmStoryResult result;
        result.mode = json.value("mode", std::string{"storytelling"});
        result.summary = json.value("summary", std::string{});
        for (const auto& beat : json.value("beats", nlohmann::json::array())) {
            VoxCpmStoryBeat value;
            value.index = beat.value("index", 0);
            value.role = beat.value("role", std::string{});
            value.text = beat.value("text", std::string{});
            value.delivery = beat.value("delivery", std::string{"natural"});
            value.direction = beat.value("direction", std::string{});
            value.emphasis = beat.value("emphasis", std::vector<std::string>{});
            value.pauseAfter = beat.value("pause_after", 0.0);
            result.beats.push_back(std::move(value));
        }
        if (result.beats.empty()) {
            return core::makeError(core::ErrorCode::FileSystemFailure,
                                   "VoxCPM2 returned an empty story direction.");
        }
        return result;
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

std::string voxCpmHealthPath() {
    return "/health";
}

std::string voxCpmRenderPath() {
    return "/render_performance";
}

std::string voxCpmTextPath() {
    return "/render_text";
}

std::string voxCpmStoryPath() {
    return "/analyze_story";
}

} // namespace voxstudio::voxcpm
