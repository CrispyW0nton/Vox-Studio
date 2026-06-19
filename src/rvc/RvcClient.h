#pragma once

#include "core/Expected.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace voxstudio::rvc {

using RvcAudioChunkCallback = std::function<bool(std::span<const std::uint8_t>)>;

struct RvcHttpResponse final {
    int statusCode{0};
    std::string body;
};

struct RvcConvertRequest final {
    std::string modelId;
    std::vector<std::uint8_t> pcm16Audio;
    int sampleRate{48000};
    int channels{1};
    int pitchShiftSemitones{0};
};

struct RvcConvertResult final {
    std::vector<std::uint8_t> pcm16Audio;
    int sampleRate{48000};
    int channels{1};
    int latencyMs{0};
};

struct RvcHealth final {
    bool ok{false};
    std::string engine;
    bool cudaAvailable{false};
    std::string cudaVersion;
    std::string loadedModelId;
    int lastLatencyMs{-1};
    std::string message;
};

class IRvcHttpTransport {
public:
    virtual ~IRvcHttpTransport() = default;

    [[nodiscard]] virtual core::Expected<RvcHttpResponse> getJson(
        const std::string& path) const = 0;

    [[nodiscard]] virtual core::Expected<RvcHttpResponse> postPcmStream(
        const std::string& path,
        const RvcConvertRequest& request,
        const RvcAudioChunkCallback& onChunk) const = 0;
};

class CprRvcHttpTransport final : public IRvcHttpTransport {
public:
    explicit CprRvcHttpTransport(std::string baseUrl);

    [[nodiscard]] core::Expected<RvcHttpResponse> getJson(
        const std::string& path) const override;

    [[nodiscard]] core::Expected<RvcHttpResponse> postPcmStream(
        const std::string& path,
        const RvcConvertRequest& request,
        const RvcAudioChunkCallback& onChunk) const override;

private:
    std::string m_baseUrl;
};

class RvcClient final {
public:
    explicit RvcClient(std::string endpoint = "http://127.0.0.1:18888");
    RvcClient(std::string endpoint, std::unique_ptr<IRvcHttpTransport> transport);

    [[nodiscard]] core::Expected<RvcHealth> health() const;
    [[nodiscard]] core::Expected<RvcConvertResult> convertChunk(
        const RvcConvertRequest& request,
        const RvcAudioChunkCallback& onChunk) const;

private:
    std::string m_endpoint;
    std::unique_ptr<IRvcHttpTransport> m_transport;
};

[[nodiscard]] std::string rvcHealthPath();
[[nodiscard]] std::string rvcConvertChunkPath();

} // namespace voxstudio::rvc
