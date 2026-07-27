#pragma once

#include "core/Expected.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace voxstudio::voxcpm {

struct VoxCpmHttpResponse final {
    int statusCode{0};
    std::string body;
    std::string transcript;
    std::string characterName;
    int sampleRate{48000};
    int latencyMs{0};
    std::string delivery;
    std::string pronunciations;
};

struct VoxCpmRenderRequest final {
    std::string voiceId;
    std::vector<std::uint8_t> pcm16Audio;
    int sampleRate{16000};
    int channels{1};
    std::string transcript;
};

struct VoxCpmRenderResult final {
    std::vector<std::uint8_t> pcm16Audio;
    std::string transcript;
    std::string characterName;
    int sampleRate{48000};
    int channels{1};
    int latencyMs{0};
    std::string delivery;
    std::string pronunciations;
};

struct VoxCpmHealth final {
    bool ok{false};
    bool cudaAvailable{false};
    bool modelLoaded{false};
    bool transcriberLoaded{false};
    int profileCount{0};
    std::string engine;
    std::string message;
};

class IVoxCpmHttpTransport {
public:
    virtual ~IVoxCpmHttpTransport() = default;

    [[nodiscard]] virtual core::Expected<VoxCpmHttpResponse>
    getJson(const std::string& path) const = 0;

    [[nodiscard]] virtual core::Expected<VoxCpmHttpResponse>
    postPerformance(const std::string& path, const VoxCpmRenderRequest& request) const = 0;
};

class CprVoxCpmHttpTransport final : public IVoxCpmHttpTransport {
public:
    explicit CprVoxCpmHttpTransport(std::string baseUrl);

    [[nodiscard]] core::Expected<VoxCpmHttpResponse>
    getJson(const std::string& path) const override;

    [[nodiscard]] core::Expected<VoxCpmHttpResponse>
    postPerformance(const std::string& path, const VoxCpmRenderRequest& request) const override;

private:
    std::string m_baseUrl;
};

class VoxCpmClient final {
public:
    explicit VoxCpmClient(std::string endpoint = "http://127.0.0.1:18990");
    VoxCpmClient(std::string endpoint, std::unique_ptr<IVoxCpmHttpTransport> transport);

    [[nodiscard]] core::Expected<VoxCpmHealth> health() const;
    [[nodiscard]] core::Expected<VoxCpmRenderResult>
    renderPerformance(const VoxCpmRenderRequest& request) const;

private:
    std::string m_endpoint;
    std::unique_ptr<IVoxCpmHttpTransport> m_transport;
};

[[nodiscard]] std::string voxCpmHealthPath();
[[nodiscard]] std::string voxCpmRenderPath();

} // namespace voxstudio::voxcpm
