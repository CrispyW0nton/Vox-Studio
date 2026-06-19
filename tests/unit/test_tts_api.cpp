#include "net/elevenlabs/TtsApi.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeTtsTransport final : public voxstudio::net::elevenlabs::ITtsHttpTransport {
public:
    explicit FakeTtsTransport(std::vector<std::vector<std::uint8_t>> chunks,
                              const int statusCode = 200,
                              std::string body = {})
        : m_chunks(std::move(chunks))
        , m_statusCode(statusCode)
        , m_body(std::move(body)) {}

    [[nodiscard]] voxstudio::core::Expected<voxstudio::net::elevenlabs::HttpResponse,
                                            voxstudio::net::elevenlabs::ApiError>
    postJsonStream(const std::string& path,
                   const std::string& apiKey,
                   const std::string& bodyJson,
                   const voxstudio::net::elevenlabs::AudioChunkCallback& onChunk) const override {
        m_path = path;
        m_apiKey = apiKey;
        m_bodyJson = bodyJson;
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

    [[nodiscard]] const std::string& bodyJson() const noexcept {
        return m_bodyJson;
    }

private:
    std::vector<std::vector<std::uint8_t>> m_chunks;
    int m_statusCode{200};
    std::string m_body;
    mutable std::string m_path;
    mutable std::string m_apiKey;
    mutable std::string m_bodyJson;
};

} // namespace

TEST_CASE("TTS API streams PCM chunks and builds request body", "[net][elevenlabs][tts]") {
    auto transport = std::make_unique<FakeTtsTransport>(
        std::vector<std::vector<std::uint8_t>>{{0x01, 0x02}, {0x03, 0x04}});
    const auto* transportView = transport.get();

    voxstudio::net::elevenlabs::TtsRequest request;
    request.voiceId = "voice/id";
    request.text = "Line text";
    request.outputFormat = "pcm_44100";
    request.voiceSettings.stability = 0.3;
    request.voiceSettings.similarityBoost = 0.8;
    request.voiceSettings.style = 0.1;
    request.voiceSettings.useSpeakerBoost = false;

    const voxstudio::net::elevenlabs::TtsApi api{"test-key", std::move(transport)};
    std::vector<std::uint8_t> callbackBytes;
    auto streamed =
        api.streamSpeech(request, [&callbackBytes](std::span<const std::uint8_t> chunk) {
            callbackBytes.insert(callbackBytes.end(), chunk.begin(), chunk.end());
            return true;
        });

    REQUIRE(streamed.hasValue());
    CHECK(streamed.value().audioBytes == std::vector<std::uint8_t>{0x01, 0x02, 0x03, 0x04});
    CHECK(callbackBytes == streamed.value().audioBytes);
    CHECK(transportView->apiKey() == "test-key");
    CHECK(transportView->path() ==
          "/v1/text-to-speech/voice%2Fid/stream?output_format=pcm_44100");

    const auto body = nlohmann::json::parse(transportView->bodyJson());
    CHECK(body.at("text").get<std::string>() == "Line text");
    CHECK(body.at("voice_settings").at("stability").get<double>() == 0.3);
    CHECK(body.at("voice_settings").at("use_speaker_boost").get<bool>() == false);
    CHECK(voxstudio::net::elevenlabs::pcmSampleRateFromOutputFormat("pcm_44100") == 44100);
}

TEST_CASE("TTS API reports HTTP and validation errors", "[net][elevenlabs][tts]") {
    voxstudio::net::elevenlabs::TtsRequest request;
    request.voiceId = "voice_001";
    request.text = "Line text";

    auto transport = std::make_unique<FakeTtsTransport>(
        std::vector<std::vector<std::uint8_t>>{{0x01, 0x02}}, 402, "no credits");
    const voxstudio::net::elevenlabs::TtsApi api{"test-key", std::move(transport)};
    auto streamed = api.streamSpeech(request, {});
    REQUIRE_FALSE(streamed.hasValue());
    CHECK(streamed.error().statusCode == 402);

    const voxstudio::net::elevenlabs::TtsApi missingKeyApi{""};
    streamed = missingKeyApi.streamSpeech(request, {});
    REQUIRE_FALSE(streamed.hasValue());
    CHECK(streamed.error().code == voxstudio::net::elevenlabs::ApiErrorCode::MissingApiKey);
}
