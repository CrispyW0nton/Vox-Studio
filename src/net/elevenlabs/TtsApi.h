#pragma once

#include "core/Expected.h"
#include "core/TakeManager.h"
#include "net/elevenlabs/Client.h"
#include "net/elevenlabs/Models.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace voxstudio::net::elevenlabs {

using AudioChunkCallback = std::function<bool(std::span<const std::uint8_t>)>;

class ITtsHttpTransport {
public:
    virtual ~ITtsHttpTransport() = default;

    [[nodiscard]] virtual core::Expected<HttpResponse, ApiError>
    postJsonStream(const std::string& path,
                   const std::string& apiKey,
                   const std::string& bodyJson,
                   const AudioChunkCallback& onChunk) const = 0;
};

class CprTtsHttpTransport final : public ITtsHttpTransport {
public:
    CprTtsHttpTransport();
    explicit CprTtsHttpTransport(std::string baseUrl);

    [[nodiscard]] core::Expected<HttpResponse, ApiError>
    postJsonStream(const std::string& path,
                   const std::string& apiKey,
                   const std::string& bodyJson,
                   const AudioChunkCallback& onChunk) const override;

private:
    std::string m_baseUrl;
};

struct TtsRequest final {
    std::string voiceId;
    std::string text;
    std::string modelId{"eleven_multilingual_v2"};
    std::string outputFormat{"pcm_44100"};
    core::VoiceSettings voiceSettings;
};

struct TtsStreamResult final {
    std::vector<std::uint8_t> audioBytes;
    int statusCode{0};
};

class TtsApi final {
public:
    explicit TtsApi(std::string apiKey);
    TtsApi(std::string apiKey, std::unique_ptr<ITtsHttpTransport> transport);

    [[nodiscard]] core::Expected<TtsStreamResult, ApiError>
    streamSpeech(const TtsRequest& request, const AudioChunkCallback& onChunk) const;

private:
    std::string m_apiKey;
    std::unique_ptr<ITtsHttpTransport> m_transport;
};

[[nodiscard]] std::string ttsRequestBodyJson(const TtsRequest& request);
[[nodiscard]] std::string ttsStreamPath(const TtsRequest& request);
[[nodiscard]] int pcmSampleRateFromOutputFormat(const std::string& outputFormat);

} // namespace voxstudio::net::elevenlabs
