#include "rvc/RvcClient.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <string_view>
#include <utility>

namespace voxstudio::rvc {
namespace {

constexpr std::chrono::seconds kHealthTimeout{5};
constexpr std::chrono::seconds kConvertTimeout{30};

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

[[nodiscard]] core::Error rvcError(const std::string& message) {
    return core::makeError(core::ErrorCode::InvalidArgument, message);
}

[[nodiscard]] core::Expected<bool> validateConvertRequest(const RvcConvertRequest& request) {
    if (request.modelId.empty()) {
        return rvcError("RVC model id must not be empty.");
    }
    if (request.pcm16Audio.empty()) {
        return rvcError("RVC audio chunk must not be empty.");
    }
    if ((request.pcm16Audio.size() % 2U) != 0U) {
        return rvcError("RVC audio chunk must contain whole 16-bit samples.");
    }
    if (request.sampleRate <= 0 || request.channels <= 0) {
        return rvcError("RVC audio chunk metadata is invalid.");
    }
    return true;
}

[[nodiscard]] std::string stringValue(const nlohmann::json& json,
                                      const char* first,
                                      const char* fallback) {
    if (json.contains(first) && json.at(first).is_string()) {
        return json.at(first).get<std::string>();
    }
    if (json.contains(fallback) && json.at(fallback).is_string()) {
        return json.at(fallback).get<std::string>();
    }
    return {};
}

} // namespace

CprRvcHttpTransport::CprRvcHttpTransport(std::string baseUrl)
    : m_baseUrl(std::move(baseUrl)) {}

core::Expected<RvcHttpResponse> CprRvcHttpTransport::getJson(const std::string& path) const {
    try {
        const auto response = cpr::Get(cpr::Url{joinedUrl(m_baseUrl, path)},
                                       cpr::Header{{"Accept", "application/json"}},
                                       cpr::Timeout{kHealthTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return core::makeError(core::ErrorCode::FileSystemFailure,
                                   response.error.message);
        }
        return RvcHttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

core::Expected<RvcHttpResponse> CprRvcHttpTransport::postPcmStream(
    const std::string& path,
    const RvcConvertRequest& request,
    const RvcAudioChunkCallback& onChunk) const {
    try {
        const auto callback = cpr::WriteCallback{[&onChunk](const std::string_view data, intptr_t) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.data());
            return !onChunk ||
                   onChunk(std::span<const std::uint8_t>{bytes, data.size()});
        }};

        cpr::Multipart multipart{
            {"audio",
             cpr::Buffer{request.pcm16Audio.begin(),
                         request.pcm16Audio.end(),
                         cpr::fs::path{"voxstudio-rvc-frame.pcm"}},
             "application/octet-stream"},
            {"model_id", request.modelId},
            {"sample_rate", std::to_string(request.sampleRate)},
            {"channels", std::to_string(request.channels)},
            {"pitch_shift", std::to_string(request.pitchShiftSemitones)}};

        const auto response = cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)},
                                        cpr::Header{{"Accept", "audio/*"}},
                                        std::move(multipart),
                                        callback,
                                        cpr::Timeout{kConvertTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return core::makeError(core::ErrorCode::FileSystemFailure,
                                   response.error.message);
        }
        return RvcHttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

RvcClient::RvcClient(std::string endpoint)
    : RvcClient(endpoint, std::make_unique<CprRvcHttpTransport>(endpoint)) {}

RvcClient::RvcClient(std::string endpoint, std::unique_ptr<IRvcHttpTransport> transport)
    : m_endpoint(std::move(endpoint))
    , m_transport(std::move(transport)) {}

core::Expected<RvcHealth> RvcClient::health() const {
    if (m_transport == nullptr) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "RVC HTTP transport is not configured.");
    }

    auto response = m_transport->getJson(rvcHealthPath());
    if (!response) {
        return response.error();
    }
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return core::makeError(core::ErrorCode::FileSystemFailure, response.value().body);
    }

    try {
        const auto json = nlohmann::json::parse(response.value().body.empty()
                                                   ? std::string{"{}"}
                                                   : response.value().body);
        RvcHealth health;
        health.ok = json.value("ok", true);
        health.engine = stringValue(json, "engine", "server");
        health.cudaAvailable = json.value("cuda_available", false);
        health.cudaVersion = stringValue(json, "cuda_version", "cuda");
        health.loadedModelId = stringValue(json, "loaded_model_id", "model_id");
        health.lastLatencyMs = json.value("last_latency_ms", -1);
        health.message = json.value("message", std::string{"RVC sidecar healthy."});
        return health;
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

core::Expected<RvcConvertResult> RvcClient::convertChunk(
    const RvcConvertRequest& request,
    const RvcAudioChunkCallback& onChunk) const {
    auto valid = validateConvertRequest(request);
    if (!valid) {
        return valid.error();
    }
    if (m_transport == nullptr) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "RVC HTTP transport is not configured.");
    }

    RvcConvertResult result;
    result.sampleRate = request.sampleRate;
    result.channels = request.channels;
    const auto startedAt = std::chrono::steady_clock::now();
    const auto collectingCallback = [&result, &onChunk](std::span<const std::uint8_t> chunk) {
        result.pcm16Audio.insert(result.pcm16Audio.end(), chunk.begin(), chunk.end());
        return !onChunk || onChunk(chunk);
    };

    auto response = m_transport->postPcmStream(rvcConvertChunkPath(), request, collectingCallback);
    const auto finishedAt = std::chrono::steady_clock::now();
    result.latencyMs = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(finishedAt - startedAt).count());
    if (!response) {
        return response.error();
    }
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return core::makeError(core::ErrorCode::FileSystemFailure, response.value().body);
    }
    if (result.pcm16Audio.empty()) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "RVC sidecar returned no converted audio.");
    }
    return result;
}

std::string rvcHealthPath() {
    return "/health";
}

std::string rvcConvertChunkPath() {
    return "/convert_chunk";
}

} // namespace voxstudio::rvc
