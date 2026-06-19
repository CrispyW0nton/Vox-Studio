#include "net/elevenlabs/StsApi.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeStsTransport final : public voxstudio::net::elevenlabs::IStsHttpTransport {
public:
    explicit FakeStsTransport(std::vector<std::vector<std::uint8_t>> chunks,
                              const int statusCode = 200,
                              std::string body = {})
        : m_chunks(std::move(chunks))
        , m_statusCode(statusCode)
        , m_body(std::move(body)) {}

    [[nodiscard]] voxstudio::core::Expected<voxstudio::net::elevenlabs::HttpResponse,
                                            voxstudio::net::elevenlabs::ApiError>
    postMultipartStream(
        const std::string& path,
        const std::string& apiKey,
        const voxstudio::net::elevenlabs::StsMultipartRequest& request,
        const voxstudio::net::elevenlabs::AudioChunkCallback& onChunk) const override {
        m_path = path;
        m_apiKey = apiKey;
        m_request = request;
        for (const auto& chunk : m_chunks) {
            if (!onChunk(std::span<const std::uint8_t>{chunk.data(), chunk.size()})) {
                return voxstudio::net::elevenlabs::makeApiError(
                    voxstudio::net::elevenlabs::ApiErrorCode::TransportFailure,
                    "callback stopped");
            }
        }
        return voxstudio::net::elevenlabs::HttpResponse{m_statusCode, m_body};
    }

    [[nodiscard]] const std::string& path() const noexcept {
        return m_path;
    }

    [[nodiscard]] const std::string& apiKey() const noexcept {
        return m_apiKey;
    }

    [[nodiscard]] const voxstudio::net::elevenlabs::StsMultipartRequest& request() const noexcept {
        return m_request;
    }

private:
    std::vector<std::vector<std::uint8_t>> m_chunks;
    int m_statusCode{200};
    std::string m_body;
    mutable std::string m_path;
    mutable std::string m_apiKey;
    mutable voxstudio::net::elevenlabs::StsMultipartRequest m_request;
};

} // namespace

TEST_CASE("STS API streams converted audio and builds multipart request",
          "[net][elevenlabs][sts]") {
    auto transport = std::make_unique<FakeStsTransport>(
        std::vector<std::vector<std::uint8_t>>{{0x10, 0x20}, {0x30, 0x40}});
    const auto* transportView = transport.get();

    voxstudio::net::elevenlabs::StsRequest request;
    request.voiceId = "voice/id";
    request.pcm16Audio = {0x01, 0x00, 0x02, 0x00};
    request.outputFormat = "pcm_44100";
    request.voiceSettings.stability = 0.25;
    request.voiceSettings.useSpeakerBoost = false;

    const voxstudio::net::elevenlabs::StsApi api{"test-key", std::move(transport)};
    std::vector<std::uint8_t> callbackBytes;
    auto streamed =
        api.streamSpeech(request, [&callbackBytes](std::span<const std::uint8_t> chunk) {
            callbackBytes.insert(callbackBytes.end(), chunk.begin(), chunk.end());
            return true;
        });

    REQUIRE(streamed.hasValue());
    CHECK(streamed.value().audioBytes == std::vector<std::uint8_t>{0x10, 0x20, 0x30, 0x40});
    CHECK(callbackBytes == streamed.value().audioBytes);
    CHECK(transportView->apiKey() == "test-key");
    CHECK(transportView->path() ==
          "/v1/speech-to-speech/voice%2Fid/stream?output_format=pcm_44100&"
          "optimize_streaming_latency=3");
    CHECK(transportView->request().audioBytes == request.pcm16Audio);
    CHECK(transportView->request().modelId == "eleven_multilingual_sts_v2");
    CHECK(transportView->request().fileFormat == "pcm_s16le_16");
    CHECK(transportView->request().removeBackgroundNoise);

    const auto settings = nlohmann::json::parse(transportView->request().voiceSettingsJson);
    CHECK(settings.at("stability").get<double>() == 0.25);
    CHECK(settings.at("use_speaker_boost").get<bool>() == false);
    CHECK(voxstudio::net::elevenlabs::pcm16MonoDurationSeconds(request.pcm16Audio, 16000) ==
          0.000125);
}

TEST_CASE("STS API reports HTTP and validation errors", "[net][elevenlabs][sts]") {
    voxstudio::net::elevenlabs::StsRequest request;
    request.voiceId = "voice_001";
    request.pcm16Audio = {0x01, 0x00};

    auto transport = std::make_unique<FakeStsTransport>(
        std::vector<std::vector<std::uint8_t>>{{0x01, 0x02}}, 402, "no credits");
    const voxstudio::net::elevenlabs::StsApi api{"test-key", std::move(transport)};
    auto streamed = api.streamSpeech(request, {});
    REQUIRE_FALSE(streamed.hasValue());
    CHECK(streamed.error().statusCode == 402);

    const voxstudio::net::elevenlabs::StsApi missingKeyApi{""};
    streamed = missingKeyApi.streamSpeech(request, {});
    REQUIRE_FALSE(streamed.hasValue());
    CHECK(streamed.error().code == voxstudio::net::elevenlabs::ApiErrorCode::MissingApiKey);

    request.pcm16Audio = {0x01};
    const voxstudio::net::elevenlabs::StsApi malformedApi{"test-key"};
    streamed = malformedApi.streamSpeech(request, {});
    REQUIRE_FALSE(streamed.hasValue());
    CHECK(streamed.error().code ==
          voxstudio::net::elevenlabs::ApiErrorCode::UnexpectedResponse);
}
