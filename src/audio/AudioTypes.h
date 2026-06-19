#pragma once

#include <cstddef>
#include <vector>

namespace voxstudio::audio {

struct PcmAudioBuffer final {
    int sampleRate{0};
    int channels{0};
    std::vector<float> samples;

    [[nodiscard]] std::size_t frameCount() const noexcept {
        if (channels <= 0) {
            return 0;
        }
        return samples.size() / static_cast<std::size_t>(channels);
    }

    [[nodiscard]] bool empty() const noexcept {
        return samples.empty() || sampleRate <= 0 || channels <= 0;
    }
};

} // namespace voxstudio::audio
