#pragma once

#include "audio/RingBuffer.h"
#include "core/Expected.h"

#include <span>

namespace voxstudio::audio {

struct LatencyProbeResult final {
    int latencyMs{0};
    int sampleOffset{0};
    bool withinTarget{false};
};

class LatencyProbe final {
public:
    [[nodiscard]] core::Expected<LatencyProbeResult>
    measureImpulseLatency(std::span<const float> output,
                          std::span<const float> input,
                          int sampleRate) const;

    [[nodiscard]] LatencyProbeResult estimateSharedModeLatency(int frameMs) const noexcept;
};

} // namespace voxstudio::audio
