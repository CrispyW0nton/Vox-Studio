#include "voxcpm/VoxCpmClient.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <utility>

namespace {

class FakeVoxCpmTransport final : public voxstudio::voxcpm::IVoxCpmHttpTransport {
public:
    [[nodiscard]] voxstudio::core::Expected<voxstudio::voxcpm::VoxCpmHttpResponse>
    getJson(const std::string& path) const override {
        healthPath = path;
        return voxstudio::voxcpm::VoxCpmHttpResponse{
            200,
            R"({"ok":true,"engine":"VoxCPM2","cuda_available":true,"model_loaded":true,"transcriber_loaded":true,"profile_count":5})"};
    }

    [[nodiscard]] voxstudio::core::Expected<voxstudio::voxcpm::VoxCpmHttpResponse>
    postPerformance(const std::string& path,
                    const voxstudio::voxcpm::VoxCpmRenderRequest& value) const override {
        renderPath = path;
        request = value;
        voxstudio::voxcpm::VoxCpmHttpResponse response;
        response.statusCode = 200;
        response.body = std::string{"\x01\x00\x02\x00", 4};
        response.transcript = "Trust me.";
        response.characterName = "Carth";
        response.sampleRate = 48000;
        response.latencyMs = 731;
        response.delivery = "calm";
        response.pronunciations = "Telos";
        response.adapter = "trained";
        return response;
    }

    [[nodiscard]] voxstudio::core::Expected<voxstudio::voxcpm::VoxCpmHttpResponse>
    postText(const std::string& path,
             const voxstudio::voxcpm::VoxCpmTextRequest& value) const override {
        textPath = path;
        textRequest = value;
        voxstudio::voxcpm::VoxCpmHttpResponse response;
        response.statusCode = 200;
        response.body = std::string{"\x03\x00\x04\x00", 4};
        response.characterName = "Carth";
        response.sampleRate = 24000;
        response.latencyMs = 1200;
        response.delivery = "reflective";
        response.pronunciations = "Rodian";
        response.adapter = "trained";
        response.sectionCount = 3;
        return response;
    }

    mutable std::string healthPath;
    mutable std::string renderPath;
    mutable std::string textPath;
    mutable voxstudio::voxcpm::VoxCpmRenderRequest request;
    mutable voxstudio::voxcpm::VoxCpmTextRequest textRequest;
};

} // namespace

TEST_CASE("VoxCPM2 client reports local engine readiness", "[voxcpm][client]") {
    auto transport = std::make_unique<FakeVoxCpmTransport>();
    const auto* view = transport.get();
    const voxstudio::voxcpm::VoxCpmClient client{"http://127.0.0.1:18990", std::move(transport)};

    auto health = client.health();

    REQUIRE(health.hasValue());
    CHECK(health.value().engine == "VoxCPM2");
    CHECK(health.value().cudaAvailable);
    CHECK(health.value().modelLoaded);
    CHECK(health.value().transcriberLoaded);
    CHECK(health.value().profileCount == 5);
    CHECK(view->healthPath == "/health");
}

TEST_CASE("VoxCPM2 client sends delivery audio and returns character PCM", "[voxcpm][client]") {
    auto transport = std::make_unique<FakeVoxCpmTransport>();
    const auto* view = transport.get();
    const voxstudio::voxcpm::VoxCpmClient client{"http://127.0.0.1:18990", std::move(transport)};
    voxstudio::voxcpm::VoxCpmRenderRequest request;
    request.voiceId = "carth";
    request.pcm16Audio = {0x10, 0x00, 0x20, 0x00};

    auto rendered = client.renderPerformance(request);

    REQUIRE(rendered.hasValue());
    CHECK(rendered.value().pcm16Audio == std::vector<std::uint8_t>{0x01, 0x00, 0x02, 0x00});
    CHECK(rendered.value().transcript == "Trust me.");
    CHECK(rendered.value().characterName == "Carth");
    CHECK(rendered.value().latencyMs == 731);
    CHECK(rendered.value().delivery == "calm");
    CHECK(rendered.value().pronunciations == "Telos");
    CHECK(rendered.value().adapter == "trained");
    CHECK(view->renderPath == "/render_performance");
    CHECK(view->request.voiceId == "carth");
}

TEST_CASE("VoxCPM2 client rejects an empty performance", "[voxcpm][client]") {
    const voxstudio::voxcpm::VoxCpmClient client{"http://127.0.0.1:18990",
                                                 std::make_unique<FakeVoxCpmTransport>()};
    voxstudio::voxcpm::VoxCpmRenderRequest request;
    request.voiceId = "carth";

    auto rendered = client.renderPerformance(request);

    REQUIRE_FALSE(rendered.hasValue());
}

TEST_CASE("VoxCPM2 client renders emotional long-form text", "[voxcpm][client][text]") {
    auto transport = std::make_unique<FakeVoxCpmTransport>();
    const auto* view = transport.get();
    const voxstudio::voxcpm::VoxCpmClient client{
        "http://127.0.0.1:18990", std::move(transport)};
    const voxstudio::voxcpm::VoxCpmTextRequest request{
        "carth", "The Rodian remembered Telos.", "reflective"};

    auto rendered = client.renderText(request);

    REQUIRE(rendered.hasValue());
    CHECK(rendered.value().pcm16Audio ==
          std::vector<std::uint8_t>{0x03, 0x00, 0x04, 0x00});
    CHECK(rendered.value().sampleRate == 24000);
    CHECK(rendered.value().delivery == "reflective");
    CHECK(rendered.value().pronunciations == "Rodian");
    CHECK(rendered.value().adapter == "trained");
    CHECK(rendered.value().sectionCount == 3);
    CHECK(view->textPath == "/render_text");
    CHECK(view->textRequest.text == "The Rodian remembered Telos.");
}
