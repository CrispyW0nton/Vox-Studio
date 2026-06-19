#pragma once

#include "audio/RingBuffer.h"
#include "core/Expected.h"

#include <filesystem>

namespace voxstudio::dsp {

struct VadResult final {
    bool speechActive{false};
    float speechProbability{0.0F};
};

struct VadConfig final {
    float activationThreshold{0.025F};
    float releaseThreshold{0.015F};
    int hangoverFrames{8};
};

class Vad final {
public:
    explicit Vad(VadConfig config = {});

    [[nodiscard]] core::Expected<bool> loadSileroModel(const std::filesystem::path& modelPath);
    [[nodiscard]] VadResult analyze(const audio::AudioFrame& frame) noexcept;
    [[nodiscard]] bool hasModel() const noexcept;

private:
    VadConfig m_config;
    bool m_hasModel{false};
    bool m_active{false};
    int m_hangoverRemaining{0};
};

} // namespace voxstudio::dsp
