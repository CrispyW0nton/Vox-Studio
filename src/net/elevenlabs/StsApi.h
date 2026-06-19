#pragma once

#include "core/Expected.h"
#include "core/TakeManager.h"
#include "net/elevenlabs/Client.h"
#include "net/elevenlabs/TtsApi.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace voxstudio::net::elevenlabs {

struct StsMultipartRequest final {
    std::vector<std::uint8_t> audioBytes;
    std::string audioFileName{"voxstudio-mic.pcm"};
    std::string modelId;
    std::string voiceSettingsJson;
    std::string fileFormat;
    bool removeBackgroundNoise{false};
};

class IStsHttpTransport {
public:
    virtual ~IStsHttpTransport() = default;

    [[nodiscard]] virtual core::Expected<HttpResponse, ApiError>
    postMultipartStream(const std::string& path,
                        const std::string& apiKey,
                        const StsMultipartRequest& request,
                        const AudioChunkCallback& onChunk) const = 0;
};

class CprStsHttpTransport final : public IStsHttpTransport {
public:
    CprStsHttpTransport();
    explicit CprStsHttpTransport(std::string baseUrl);

    [[nodiscard]] core::Expected<HttpResponse, ApiError>
    postMultipartStream(const std::string& path,
                        const std::string& apiKey,
                        const StsMultipartRequest& request,
                        const AudioChunkCallback& onChunk) const override;

private:
    std::string m_baseUrl;
};

struct StsRequest final {
    std::string voiceId;
    std::vector<std::uint8_t> pcm16Audio;
    std::string modelId{"eleven_multilingual_sts_v2"};
    std::string outputFormat{"pcm_44100"};
    std::string inputFileFormat{"pcm_s16le_16"};
    core::VoiceSettings voiceSettings;
    bool removeBackgroundNoise{true};
    int optimizeStreamingLatency{3};
};

struct StsStreamResult final {
    std::vector<std::uint8_t> audioBytes;
    int statusCode{0};
    double inputSeconds{0.0};
};

class StsApi final {
public:
    explicit StsApi(std::string apiKey);
    StsApi(std::string apiKey, std::unique_ptr<IStsHttpTransport> transport);

    [[nodiscard]] core::Expected<StsStreamResult, ApiError>
    streamSpeech(const StsRequest& request, const AudioChunkCallback& onChunk) const;

private:
    std::string m_apiKey;
    std::unique_ptr<IStsHttpTransport> m_transport;
};

[[nodiscard]] std::string stsStreamPath(const StsRequest& request);
[[nodiscard]] std::string stsVoiceSettingsJson(const core::VoiceSettings& settings);
[[nodiscard]] StsMultipartRequest stsMultipartRequest(const StsRequest& request);
[[nodiscard]] double pcm16MonoDurationSeconds(std::span<const std::uint8_t> bytes,
                                              int sampleRate);

} // namespace voxstudio::net::elevenlabs
