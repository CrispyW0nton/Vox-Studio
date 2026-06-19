#pragma once

#include "core/Expected.h"
#include "rvc/OnnxRvcEngine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace voxstudio::rvc {

struct NativeRvcAudioTensor final {
    std::string name;
    int sampleRate{0};
    std::vector<std::int64_t> dimensions;
    std::vector<float> values;
};

struct NativeRvcPreparedAudio final {
    NativeRvcAudioTensor hubertAudio;
    NativeRvcAudioTensor f0Audio;
};

struct NativeRvcAuxiliaryTensor final {
    std::string name;
    std::string elementType;
    std::vector<std::int64_t> dimensions;
    std::vector<float> floatValues;
    std::vector<std::int64_t> int64Values;
};

[[nodiscard]] core::Expected<NativeRvcPreparedAudio> prepareNativeRvcAudio(
    const OnnxRvcRequest& request,
    const OnnxRvcModelBundle& bundle,
    const OnnxRvcPipelinePlan& pipeline);

[[nodiscard]] core::Expected<std::vector<NativeRvcAuxiliaryTensor>>
prepareGeneratorAuxiliaryInputs(const OnnxRvcPipelinePlan& pipeline,
                                std::int64_t contentFrameCount);

} // namespace voxstudio::rvc
