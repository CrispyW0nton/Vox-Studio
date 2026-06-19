#include "dsp/Vad.h"

#include <algorithm>
#include <cmath>

namespace voxstudio::dsp {

Vad::Vad(VadConfig config)
    : m_config(config) {}

core::Expected<bool> Vad::loadSileroModel(const std::filesystem::path& modelPath) {
    if (modelPath.empty() || !std::filesystem::exists(modelPath)) {
        return core::makeError(core::ErrorCode::FileSystemFailure,
                               "Silero VAD model file was not found.");
    }

    m_hasModel = true;
    return true;
}

VadResult Vad::analyze(const audio::AudioFrame& frame) noexcept {
    const auto denominator = std::max(m_config.activationThreshold * 2.0F, 0.001F);
    const auto probability = std::clamp(frame.rms / denominator, 0.0F, 1.0F);

    if (frame.rms >= m_config.activationThreshold) {
        m_active = true;
        m_hangoverRemaining = m_config.hangoverFrames;
    } else if (frame.rms < m_config.releaseThreshold) {
        if (m_hangoverRemaining > 0) {
            --m_hangoverRemaining;
        } else {
            m_active = false;
        }
    }

    return VadResult{m_active, probability};
}

bool Vad::hasModel() const noexcept {
    return m_hasModel;
}

} // namespace voxstudio::dsp
