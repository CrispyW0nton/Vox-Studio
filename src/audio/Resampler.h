#pragma once

#include "audio/AudioTypes.h"
#include "core/Expected.h"

namespace voxstudio::audio {

[[nodiscard]] core::Expected<PcmAudioBuffer>
resamplePcm(const PcmAudioBuffer& input, int outputSampleRate);

} // namespace voxstudio::audio
