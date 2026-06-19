#include "audio/Resampler.h"

#include <soxr.h>

#include <cmath>
#include <string>

namespace voxstudio::audio {
namespace {

[[nodiscard]] core::Error resampleError(const std::string& message) {
    return core::makeError(core::ErrorCode::InvalidArgument, message);
}

} // namespace

core::Expected<PcmAudioBuffer> resamplePcm(const PcmAudioBuffer& input,
                                           const int outputSampleRate) {
    if (input.empty()) {
        return resampleError("PCM input is empty.");
    }
    if (outputSampleRate <= 0) {
        return resampleError("Output sample rate must be positive.");
    }
    if (input.sampleRate == outputSampleRate) {
        return input;
    }

    const auto inputFrames = input.frameCount();
    const auto ratio = static_cast<double>(outputSampleRate) /
                       static_cast<double>(input.sampleRate);
    const auto outputFrames =
        static_cast<std::size_t>(std::ceil(static_cast<double>(inputFrames) * ratio)) + 8U;

    PcmAudioBuffer output;
    output.sampleRate = outputSampleRate;
    output.channels = input.channels;
    output.samples.resize(outputFrames * static_cast<std::size_t>(output.channels));

    const auto ioSpec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    const auto qualitySpec = soxr_quality_spec(SOXR_HQ, 0);
    std::size_t inputDone = 0;
    std::size_t outputDone = 0;
    const auto* error = soxr_oneshot(static_cast<double>(input.sampleRate),
                                     static_cast<double>(outputSampleRate),
                                     static_cast<unsigned>(input.channels),
                                     input.samples.data(),
                                     inputFrames,
                                     &inputDone,
                                     output.samples.data(),
                                     outputFrames,
                                     &outputDone,
                                     &ioSpec,
                                     &qualitySpec,
                                     nullptr);
    if (error != nullptr) {
        return resampleError(soxr_strerror(error));
    }

    output.samples.resize(outputDone * static_cast<std::size_t>(output.channels));
    return output;
}

} // namespace voxstudio::audio
