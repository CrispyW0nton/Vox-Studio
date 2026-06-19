#include "core/Sequence.h"

#include <cmath>

namespace voxstudio::core {

int sequenceDurationMs(const audio::PcmAudioBuffer& audio) noexcept {
    if (audio.sampleRate <= 0) {
        return 0;
    }

    const auto seconds = static_cast<double>(audio.frameCount()) /
                         static_cast<double>(audio.sampleRate);
    return static_cast<int>(std::lround(seconds * 1000.0));
}

} // namespace voxstudio::core
