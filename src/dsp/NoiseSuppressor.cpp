#include "dsp/NoiseSuppressor.h"

#include <algorithm>

namespace voxstudio::dsp {

NoiseSuppressor::NoiseSuppressor(NoiseSuppressorConfig config)
    : m_config(config) {}

core::Expected<audio::AudioFrame>
NoiseSuppressor::processFrame(const audio::AudioFrame& input) noexcept {
    if (input.empty() || input.sampleRate != audio::kRealtimeSampleRate ||
        input.channels != audio::kRealtimeChannels) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Noise suppressor expects non-empty 48 kHz mono frames.");
    }

    auto output = input;
    m_noiseFloor = (0.98F * m_noiseFloor) + (0.02F * std::min(input.rms, m_config.gateThreshold));
    const auto threshold = std::max(m_config.gateThreshold, m_noiseFloor * 2.5F);
    const auto gain = input.rms < threshold ? m_config.attenuation : 1.0F;
    for (std::size_t index = 0; index < output.sampleCount(); ++index) {
        output.samples[index] *= gain;
    }
    output.rms *= gain;
    return output;
}

} // namespace voxstudio::dsp
