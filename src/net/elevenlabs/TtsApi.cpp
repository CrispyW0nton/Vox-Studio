#include "net/elevenlabs/TtsApi.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <algorithm>
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
constexpr std::chrono::seconds kRequestTimeout{60};

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
                                                             const TtsRequest& request) {
    if (apiKey.empty()) {
        return makeApiError(ApiErrorCode::MissingApiKey, "ElevenLabs API key is missing.");
    }
    if (request.voiceId.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Voice id must not be empty.");
    }
    if (request.text.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "TTS text must not be empty.");
    }
    return true;
}

} // namespace

CprTtsHttpTransport::CprTtsHttpTransport()
    : CprTtsHttpTransport(kDefaultBaseUrl) {}

CprTtsHttpTransport::CprTtsHttpTransport(std::string baseUrl)
    : m_baseUrl(std::move(baseUrl)) {}

core::Expected<HttpResponse, ApiError>
CprTtsHttpTransport::postJsonStream(const std::string& path,
                                    const std::string& apiKey,
                                    const std::string& bodyJson,
                                    const AudioChunkCallback& onChunk) const {
    try {
        const auto callback = cpr::WriteCallback{[&onChunk](const std::string_view data, intptr_t) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.data());
            return onChunk(std::span<const std::uint8_t>{bytes, data.size()});
        }};

        const auto response =
            cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)},
                      cpr::Header{{"Accept", "audio/*"},
                                  {"Content-Type", "application/json"},
                                  {"xi-api-key", apiKey}},
                      cpr::Body{bodyJson},
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

TtsApi::TtsApi(std::string apiKey)
    : TtsApi(std::move(apiKey), std::make_unique<CprTtsHttpTransport>()) {}

TtsApi::TtsApi(std::string apiKey, std::unique_ptr<ITtsHttpTransport> transport)
    : m_apiKey(std::move(apiKey))
    , m_transport(std::move(transport)) {}

core::Expected<TtsStreamResult, ApiError>
TtsApi::streamSpeech(const TtsRequest& request, const AudioChunkCallback& onChunk) const {
    auto valid = validateRequest(m_apiKey, request);
    if (!valid) {
        return valid.error();
    }
    if (m_transport == nullptr) {
        return makeApiError(ApiErrorCode::TransportFailure, "HTTP transport is not configured.");
    }

    TtsStreamResult result;
    const auto collectingCallback = [&result, &onChunk](std::span<const std::uint8_t> chunk) {
        result.audioBytes.insert(result.audioBytes.end(), chunk.begin(), chunk.end());
        return !onChunk || onChunk(chunk);
    };

    auto response =
        m_transport->postJsonStream(ttsStreamPath(request),
                                    m_apiKey,
                                    ttsRequestBodyJson(request),
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
        return makeApiError(ApiErrorCode::UnexpectedResponse, "TTS stream returned no audio.");
    }
    return result;
}

std::string ttsRequestBodyJson(const TtsRequest& request) {
    nlohmann::json settings;
    settings["stability"] = request.voiceSettings.stability;
    settings["similarity_boost"] = request.voiceSettings.similarityBoost;
    settings["style"] = request.voiceSettings.style;
    settings["use_speaker_boost"] = request.voiceSettings.useSpeakerBoost;

    nlohmann::json body;
    body["text"] = request.text;
    body["model_id"] = request.modelId;
    body["voice_settings"] = settings;
    return body.dump();
}

std::string ttsStreamPath(const TtsRequest& request) {
    auto path = "/v1/text-to-speech/" + urlEncode(request.voiceId) + "/stream";
    if (!request.outputFormat.empty()) {
        path += "?output_format=" + urlEncode(request.outputFormat);
    }
    return path;
}

int pcmSampleRateFromOutputFormat(const std::string& outputFormat) {
    constexpr std::string_view kPrefix = "pcm_";
    if (outputFormat.rfind(kPrefix, 0) != 0) {
        return 0;
    }

    const auto sampleRateText = outputFormat.substr(kPrefix.size());
    if (sampleRateText.empty() ||
        !std::ranges::all_of(sampleRateText, [](const unsigned char character) {
            return std::isdigit(character) != 0;
        })) {
        return 0;
    }
    return std::stoi(sampleRateText);
}

} // namespace voxstudio::net::elevenlabs
