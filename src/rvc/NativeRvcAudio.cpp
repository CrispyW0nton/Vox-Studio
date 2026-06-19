#include "rvc/NativeRvcAudio.h"

#include "audio/AudioFile.h"
#include "audio/Resampler.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <utility>

namespace voxstudio::rvc {
namespace {

constexpr int kHubertInputSampleRate = 16000;
constexpr auto kElementFloat32 = "float32";
constexpr auto kElementInt64 = "int64";
constexpr auto kRoleLength = "length";
constexpr auto kRoleNoise = "noise";
constexpr auto kRoleSpeaker = "speaker";

[[nodiscard]] core::Error nativeAudioError(const std::string& message) {
    return core::makeError(core::ErrorCode::InvalidArgument, message);
}

[[nodiscard]] audio::PcmAudioBuffer mixToMono(const audio::PcmAudioBuffer& input) {
    if (input.channels == 1) {
        return input;
    }

    audio::PcmAudioBuffer output;
    output.sampleRate = input.sampleRate;
    output.channels = 1;
    output.samples.reserve(input.frameCount());
    for (std::size_t frame = 0; frame < input.frameCount(); ++frame) {
        float mixed = 0.0F;
        for (int channel = 0; channel < input.channels; ++channel) {
            const auto offset = (frame * static_cast<std::size_t>(input.channels)) +
                                static_cast<std::size_t>(channel);
            mixed += input.samples[offset];
        }
        output.samples.push_back(mixed / static_cast<float>(input.channels));
    }
    return output;
}

[[nodiscard]] core::Expected<audio::PcmAudioBuffer> prepareMonoAtSampleRate(
    const audio::PcmAudioBuffer& mono,
    const int sampleRate) {
    if (sampleRate <= 0) {
        return nativeAudioError("Native RVC tensor sample rate must be positive.");
    }
    if (mono.sampleRate == sampleRate) {
        return mono;
    }
    return audio::resamplePcm(mono, sampleRate);
}

[[nodiscard]] core::Expected<std::vector<std::int64_t>> shapeForAudioTensor(
    const OnnxTensorDescription& tensor,
    const std::size_t sampleCount) {
    if (sampleCount == 0U) {
        return nativeAudioError("Native RVC audio tensor cannot be empty.");
    }
    if (tensor.dimensions.empty()) {
        return std::vector<std::int64_t>{1, static_cast<std::int64_t>(sampleCount)};
    }

    auto dimensions = tensor.dimensions;
    const auto dynamicAxes = std::ranges::count_if(
        dimensions,
        [](const std::int64_t dimension) { return dimension < 0; });
    if (dynamicAxes > 1) {
        return nativeAudioError(tensor.name + " audio tensor has multiple dynamic axes.");
    }
    if (dynamicAxes == 1) {
        for (auto& dimension : dimensions) {
            if (dimension < 0) {
                dimension = static_cast<std::int64_t>(sampleCount);
            }
        }
        return dimensions;
    }

    const auto elementCount = std::accumulate(
        dimensions.begin(),
        dimensions.end(),
        std::int64_t{1},
        [](const std::int64_t total, const std::int64_t dimension) {
            return total * dimension;
        });
    if (elementCount != static_cast<std::int64_t>(sampleCount)) {
        return nativeAudioError(tensor.name + " audio tensor fixed shape expects " +
                                std::to_string(elementCount) + " samples, got " +
                                std::to_string(sampleCount) + ".");
    }
    return dimensions;
}

[[nodiscard]] std::string normalizedRole(std::string role) {
    std::ranges::transform(role, role.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return role;
}

[[nodiscard]] core::Expected<std::vector<std::int64_t>> defaultAuxiliaryShape(
    const OnnxTensorDescription& tensor,
    const std::int64_t contentFrameCount) {
    if (contentFrameCount <= 0) {
        return nativeAudioError("Generator content frame count must be positive.");
    }
    if (tensor.dimensions.empty()) {
        return std::vector<std::int64_t>{1};
    }

    auto dimensions = tensor.dimensions;
    for (auto& dimension : dimensions) {
        if (dimension < 0) {
            dimension = contentFrameCount;
        }
        if (dimension <= 0) {
            return nativeAudioError(tensor.name + " auxiliary tensor shape is invalid.");
        }
    }
    return dimensions;
}

[[nodiscard]] core::Expected<std::int64_t> elementCountForShape(
    const std::vector<std::int64_t>& dimensions,
    const std::string& tensorName) {
    if (dimensions.empty()) {
        return nativeAudioError(tensorName + " auxiliary tensor shape is missing.");
    }
    std::int64_t elementCount = 1;
    for (const auto dimension : dimensions) {
        if (dimension <= 0) {
            return nativeAudioError(tensorName + " auxiliary tensor shape is invalid.");
        }
        elementCount *= dimension;
    }
    return elementCount;
}

[[nodiscard]] core::Expected<NativeRvcAudioTensor> tensorFromAudio(
    const OnnxRvcTensorBinding& binding,
    audio::PcmAudioBuffer audio) {
    auto dimensions = shapeForAudioTensor(binding.tensor, audio.samples.size());
    if (!dimensions) {
        return dimensions.error();
    }

    NativeRvcAudioTensor tensor;
    tensor.name = binding.tensor.name;
    tensor.sampleRate = audio.sampleRate;
    tensor.dimensions = std::move(dimensions.value());
    tensor.values = std::move(audio.samples);
    return tensor;
}

} // namespace

core::Expected<NativeRvcPreparedAudio> prepareNativeRvcAudio(
    const OnnxRvcRequest& request,
    const OnnxRvcModelBundle& bundle,
    const OnnxRvcPipelinePlan& pipeline) {
    if (request.pcm16Audio.empty() || (request.pcm16Audio.size() % 2U) != 0U ||
        request.sampleRate <= 0 || request.channels <= 0) {
        return nativeAudioError("Native RVC request audio is invalid.");
    }
    if (bundle.sampleRate <= 0) {
        return nativeAudioError("Native RVC model sample rate must be positive.");
    }

    const auto bytes = std::span<const std::uint8_t>{request.pcm16Audio.data(),
                                                    request.pcm16Audio.size()};
    auto decoded = audio::pcm16LittleEndianToPcm(bytes, request.sampleRate, request.channels);
    if (!decoded) {
        return decoded.error();
    }

    const auto mono = mixToMono(decoded.value());
    auto hubertAudio = prepareMonoAtSampleRate(mono, kHubertInputSampleRate);
    if (!hubertAudio) {
        return hubertAudio.error();
    }
    auto f0Audio = prepareMonoAtSampleRate(mono, bundle.sampleRate);
    if (!f0Audio) {
        return f0Audio.error();
    }

    auto hubertTensor =
        tensorFromAudio(pipeline.hubertAudioInput, std::move(hubertAudio.value()));
    if (!hubertTensor) {
        return hubertTensor.error();
    }
    auto f0Tensor = tensorFromAudio(pipeline.f0AudioInput, std::move(f0Audio.value()));
    if (!f0Tensor) {
        return f0Tensor.error();
    }

    return NativeRvcPreparedAudio{std::move(hubertTensor.value()),
                                  std::move(f0Tensor.value())};
}

core::Expected<std::vector<NativeRvcAuxiliaryTensor>> prepareGeneratorAuxiliaryInputs(
    const OnnxRvcPipelinePlan& pipeline,
    const std::int64_t contentFrameCount) {
    std::vector<NativeRvcAuxiliaryTensor> tensors;
    tensors.reserve(pipeline.generatorAuxiliaryInputs.size());
    for (const auto& binding : pipeline.generatorAuxiliaryInputs) {
        const auto role = normalizedRole(binding.tensor.semanticRole);
        auto dimensions = defaultAuxiliaryShape(binding.tensor, contentFrameCount);
        if (!dimensions) {
            return dimensions.error();
        }
        auto elementCount = elementCountForShape(dimensions.value(), binding.tensor.name);
        if (!elementCount) {
            return elementCount.error();
        }

        NativeRvcAuxiliaryTensor tensor;
        tensor.name = binding.tensor.name;
        tensor.elementType = binding.tensor.elementType;
        tensor.dimensions = std::move(dimensions.value());
        if (role == kRoleSpeaker) {
            if (binding.tensor.elementType != kElementInt64) {
                return nativeAudioError("Speaker tensor must be int64.");
            }
            tensor.int64Values.assign(static_cast<std::size_t>(elementCount.value()), 0);
        } else if (role == kRoleLength) {
            if (binding.tensor.elementType != kElementInt64) {
                return nativeAudioError("Length tensor must be int64.");
            }
            tensor.int64Values.assign(static_cast<std::size_t>(elementCount.value()),
                                      contentFrameCount);
        } else if (role == kRoleNoise) {
            if (binding.tensor.elementType != kElementFloat32) {
                return nativeAudioError("Noise tensor must be float32.");
            }
            tensor.floatValues.assign(static_cast<std::size_t>(elementCount.value()), 0.0F);
        } else {
            return nativeAudioError("Unsupported generator auxiliary tensor role: " + role + ".");
        }
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

} // namespace voxstudio::rvc
