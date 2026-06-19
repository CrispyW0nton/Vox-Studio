#pragma once

#include "audio/RingBuffer.h"
#include "core/Expected.h"

namespace voxstudio::dsp {

struct NoiseSuppressorConfig final {
    float gateThreshold{0.012F};
    float attenuation{0.15F};
};

class NoiseSuppressor final {
public:
    explicit NoiseSuppressor(NoiseSuppressorConfig config = {});

    [[nodiscard]] core::Expected<audio::AudioFrame>
    processFrame(const audio::AudioFrame& input) noexcept;

private:
    NoiseSuppressorConfig m_config;
    float m_noiseFloor{0.004F};
};

} // namespace voxstudio::dsp
