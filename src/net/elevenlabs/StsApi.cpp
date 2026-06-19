#include "net/elevenlabs/StsApi.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cctype>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

namespace voxstudio::net::elevenlabs {
namespace {

constexpr auto kDefaultBaseUrl = "https://api.elevenlabs.io";
constexpr int kPcm16InputSampleRate = 16000;
constexpr std::chrono::seconds kRequestTimeout{90};

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

[[nodiscard]] bool isUnreservedUrlCharacter(const unsigned char value) noexcept {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
           (value >= '0' && value <= '9') || value == '-' || value == '_' || value == '.' ||
           value == '~';
}

[[nodiscard]] std::string urlEncode(const std::string& value) {
    std::ostringstream stream;
    stream << std::uppercase << std::hex;
    for (const unsigned char character : value) {
        if (isUnreservedUrlCharacter(character)) {
            stream << static_cast<char>(character);
        } else {
            stream << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(character);
        }
    }
    return stream.str();
}

[[nodiscard]] core::Expected<bool, ApiError> validateRequest(const std::string& apiKey,
                                                             const StsRequest& request) {
    if (apiKey.empty()) {
        return makeApiError(ApiErrorCode::MissingApiKey, "ElevenLabs API key is missing.");
    }
    if (request.voiceId.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Voice id must not be empty.");
    }
    if (request.pcm16Audio.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "STS audio must not be empty.");
    }
    if ((request.pcm16Audio.size() % 2U) != 0U) {
        return makeApiError(ApiErrorCode::UnexpectedResponse,
                            "STS PCM audio must contain whole 16-bit samples.");
    }
    if (request.inputFileFormat != "pcm_s16le_16" && request.inputFileFormat != "other") {
        return makeApiError(ApiErrorCode::UnexpectedResponse,
                            "STS input file format is not supported.");
    }
    return true;
}

} // namespace

CprStsHttpTransport::CprStsHttpTransport()
    : CprStsHttpTransport(kDefaultBaseUrl) {}

CprStsHttpTransport::CprStsHttpTransport(std::string baseUrl)
    : m_baseUrl(std::move(baseUrl)) {}

core::Expected<HttpResponse, ApiError>
CprStsHttpTransport::postMultipartStream(const std::string& path,
                                         const std::string& apiKey,
                                         const StsMultipartRequest& request,
                                         const AudioChunkCallback& onChunk) const {
    try {
        const auto callback = cpr::WriteCallback{[&onChunk](const std::string_view data, intptr_t) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.data());
            return onChunk(std::span<const std::uint8_t>{bytes, data.size()});
        }};

        std::vector<cpr::Part> parts;
        parts.reserve(5U);
        parts.emplace_back("audio",
                           cpr::Buffer{request.audioBytes.begin(),
                                       request.audioBytes.end(),
                                       cpr::fs::path{request.audioFileName}},
                           "application/octet-stream");
        parts.emplace_back("model_id", request.modelId);
        parts.emplace_back("voice_settings", request.voiceSettingsJson);
        parts.emplace_back("file_format", request.fileFormat);
        parts.emplace_back("remove_background_noise",
                           request.removeBackgroundNoise ? "true" : "false");

        const auto response =
            cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)},
                      cpr::Header{{"Accept", "audio/*"}, {"xi-api-key", apiKey}},
                      cpr::Multipart{std::move(parts)},
                      callback,
                      cpr::Timeout{kRequestTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return makeApiError(ApiErrorCode::TransportFailure, response.error.message);
        }
        return HttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::TransportFailure, exception.what());
    }
}

StsApi::StsApi(std::string apiKey)
    : StsApi(std::move(apiKey), std::make_unique<CprStsHttpTransport>()) {}

StsApi::StsApi(std::string apiKey, std::unique_ptr<IStsHttpTransport> transport)
    : m_apiKey(std::move(apiKey))
    , m_transport(std::move(transport)) {}

core::Expected<StsStreamResult, ApiError>
StsApi::streamSpeech(const StsRequest& request, const AudioChunkCallback& onChunk) const {
    auto valid = validateRequest(m_apiKey, request);
    if (!valid) {
        return valid.error();
    }
    if (m_transport == nullptr) {
        return makeApiError(ApiErrorCode::TransportFailure, "HTTP transport is not configured.");
    }

    StsStreamResult result;
    result.inputSeconds = pcm16MonoDurationSeconds(request.pcm16Audio, kPcm16InputSampleRate);
    const auto collectingCallback = [&result, &onChunk](std::span<const std::uint8_t> chunk) {
        result.audioBytes.insert(result.audioBytes.end(), chunk.begin(), chunk.end());
        return !onChunk || onChunk(chunk);
    };

    auto response = m_transport->postMultipartStream(stsStreamPath(request),
                                                     m_apiKey,
                                                     stsMultipartRequest(request),
                                                     collectingCallback);
    if (!response) {
        return response.error();
    }

    result.statusCode = response.value().statusCode;
    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return makeApiError(ApiErrorCode::HttpError,
                            response.value().body,
                            response.value().statusCode);
    }
    if (result.audioBytes.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "STS stream returned no audio.");
    }
    return result;
}

std::string stsStreamPath(const StsRequest& request) {
    auto path = "/v1/speech-to-speech/" + urlEncode(request.voiceId) + "/stream";
    bool hasQuery = false;
    if (!request.outputFormat.empty()) {
        path += "?output_format=" + urlEncode(request.outputFormat);
        hasQuery = true;
    }
    if (request.optimizeStreamingLatency >= 0) {
        path += hasQuery ? "&" : "?";
        path += "optimize_streaming_latency=" +
                std::to_string(request.optimizeStreamingLatency);
    }
    return path;
}

std::string stsVoiceSettingsJson(const core::VoiceSettings& settings) {
    nlohmann::json json;
    json["stability"] = settings.stability;
    json["similarity_boost"] = settings.similarityBoost;
    json["style"] = settings.style;
    json["use_speaker_boost"] = settings.useSpeakerBoost;
    return json.dump();
}

StsMultipartRequest stsMultipartRequest(const StsRequest& request) {
    return StsMultipartRequest{request.pcm16Audio,
                               "voxstudio-mic.pcm",
                               request.modelId,
                               stsVoiceSettingsJson(request.voiceSettings),
                               request.inputFileFormat,
                               request.removeBackgroundNoise};
}

double pcm16MonoDurationSeconds(std::span<const std::uint8_t> bytes, const int sampleRate) {
    if (sampleRate <= 0 || (bytes.size() % 2U) != 0U) {
        return 0.0;
    }
    return static_cast<double>(bytes.size() / 2U) / static_cast<double>(sampleRate);
}

} // namespace voxstudio::net::elevenlabs
