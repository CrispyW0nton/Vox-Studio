#include "audio/LatencyProbe.h"

#include <algorithm>
#include <cmath>

namespace voxstudio::audio {

core::Expected<LatencyProbeResult>
LatencyProbe::measureImpulseLatency(std::span<const float> output,
                                    std::span<const float> input,
                                    const int sampleRate) const {
    if (output.empty() || input.empty() || sampleRate <= 0) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Impulse latency probe requires non-empty buffers.");
    }

    const auto outputPeak = std::max_element(output.begin(), output.end(), [](float left,
                                                                             float right) {
        return std::abs(left) < std::abs(right);
    });
    const auto inputPeak = std::max_element(input.begin(), input.end(), [](float left,
                                                                          float right) {
        return std::abs(left) < std::abs(right);
    });
    const auto outputIndex = static_cast<int>(std::distance(output.begin(), outputPeak));
    const auto inputIndex = static_cast<int>(std::distance(input.begin(), inputPeak));
    const auto offset = std::max(0, inputIndex - outputIndex);
    const auto latencyMs =
        static_cast<int>(std::lround((static_cast<double>(offset) * 1000.0) /
                                     static_cast<double>(sampleRate)));
    return LatencyProbeResult{latencyMs, offset, latencyMs <= 60};
}

LatencyProbeResult LatencyProbe::estimateSharedModeLatency(const int frameMs) const noexcept {
    const auto clampedFrameMs = std::clamp(frameMs, 10, kMaxRealtimeFrameMs);
    const auto latencyMs = (clampedFrameMs * 3) + 10;
    const auto offset = (kRealtimeSampleRate * latencyMs) / 1000;
    return LatencyProbeResult{latencyMs, offset, latencyMs <= 60};
}

} // namespace voxstudio::audio
