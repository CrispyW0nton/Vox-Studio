#include "audio/LatencyProbe.h"
#include "audio/RingBuffer.h"
#include "dsp/NoiseSuppressor.h"
#include "dsp/Vad.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <string>

namespace {

[[nodiscard]] voxstudio::audio::AudioFrame frameWithRms(const float amplitude) {
    voxstudio::audio::AudioFrame frame;
    frame.frameCount = 480;
    for (std::size_t index = 0; index < frame.frameCount; ++index) {
        frame.samples[index] = amplitude;
    }
    frame.rms = std::abs(amplitude);
    return frame;
}

} // namespace

TEST_CASE("audio ring buffer preserves order and reports full queue", "[audio][ring]") {
    voxstudio::audio::AudioRingBuffer ring{2};

    auto first = frameWithRms(0.1F);
    first.capturedAtSeconds = 1.0;
    auto second = frameWithRms(0.2F);
    second.capturedAtSeconds = 2.0;
    auto third = frameWithRms(0.3F);

    CHECK(ring.tryPush(first));
    CHECK(ring.tryPush(second));
    CHECK_FALSE(ring.tryPush(third));
    CHECK(ring.sizeApprox() == 2);

    voxstudio::audio::AudioFrame popped;
    REQUIRE(ring.tryPop(popped));
    CHECK(popped.capturedAtSeconds == 1.0);
    REQUIRE(ring.tryPop(popped));
    CHECK(popped.capturedAtSeconds == 2.0);
    CHECK_FALSE(ring.tryPop(popped));
}

TEST_CASE("noise suppressor attenuates quiet frames and preserves speech frames", "[dsp]") {
    voxstudio::dsp::NoiseSuppressor suppressor;

    auto quiet = suppressor.processFrame(frameWithRms(0.001F));
    REQUIRE(quiet.hasValue());
    CHECK(quiet.value().rms < 0.001F);

    auto speech = suppressor.processFrame(frameWithRms(0.1F));
    REQUIRE(speech.hasValue());
    CHECK(speech.value().rms == 0.1F);
}

TEST_CASE("vad activates with hangover and rejects missing Silero model", "[dsp][vad]") {
    voxstudio::dsp::Vad vad;

    auto missing = vad.loadSileroModel(std::filesystem::path{"missing-silero-vad.onnx"});
    REQUIRE_FALSE(missing.hasValue());
    CHECK_FALSE(vad.hasModel());

    const auto speech = vad.analyze(frameWithRms(0.1F));
    CHECK(speech.speechActive);
    CHECK(speech.speechProbability > 0.9F);

    const auto quiet = vad.analyze(frameWithRms(0.0F));
    CHECK(quiet.speechActive);
}

TEST_CASE("latency probe measures impulse offset and shared-mode estimate", "[audio][latency]") {
    constexpr int kSampleRate = 48000;
    std::array<float, 2048> output{};
    std::array<float, 2048> input{};
    output[100] = 1.0F;
    input[1060] = 1.0F;

    const voxstudio::audio::LatencyProbe probe;
    auto measured = probe.measureImpulseLatency(output, input, kSampleRate);
    REQUIRE(measured.hasValue());
    CHECK(measured.value().sampleOffset == 960);
    CHECK(measured.value().latencyMs == 20);
    CHECK(measured.value().withinTarget);

    const auto estimated = probe.estimateSharedModeLatency(10);
    CHECK(estimated.latencyMs == 40);
    CHECK(estimated.withinTarget);
}
