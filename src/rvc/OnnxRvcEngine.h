#pragma once

#include "core/Expected.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace voxstudio::rvc {

struct OnnxRuntimeInfo final {
    bool available{false};
    std::filesystem::path runtimeDllPath;
    std::string version;
    std::string message;
};

struct OnnxTensorDescription final {
    std::string name;
    std::string elementType;
    std::vector<std::int64_t> dimensions;
    std::vector<std::string> symbolicDimensions;
    std::string semanticRole;
};

struct OnnxSessionDescription final {
    std::string label;
    std::vector<OnnxTensorDescription> inputs;
    std::vector<OnnxTensorDescription> outputs;
};

struct OnnxRvcGraphContract final {
    bool required{false};
    OnnxSessionDescription generator;
    OnnxSessionDescription hubert;
    OnnxSessionDescription f0;
};

struct OnnxRvcModelBundle final {
    std::string modelId;
    std::filesystem::path rootPath;
    std::filesystem::path generatorModelPath;
    std::filesystem::path hubertModelPath;
    std::filesystem::path f0ModelPath;
    int sampleRate{48000};
    int hopLength{160};
    OnnxRvcGraphContract graphContract;
};

struct OnnxRvcRequest final {
    std::vector<std::uint8_t> pcm16Audio;
    int sampleRate{48000};
    int channels{1};
    int pitchShiftSemitones{0};
};

struct OnnxRvcResult final {
    std::vector<std::uint8_t> pcm16Audio;
    int sampleRate{48000};
    int channels{1};
    int latencyMs{0};
};

struct OnnxRvcModelDescription final {
    OnnxRuntimeInfo runtime;
    OnnxRvcModelBundle bundle;
    OnnxSessionDescription generator;
    OnnxSessionDescription hubert;
    OnnxSessionDescription f0;
};

struct OnnxRvcTensorBinding final {
    std::string sessionLabel;
    OnnxTensorDescription tensor;
};

struct OnnxRvcPipelinePlan final {
    OnnxRvcTensorBinding hubertAudioInput;
    OnnxRvcTensorBinding hubertContentOutput;
    OnnxRvcTensorBinding f0AudioInput;
    OnnxRvcTensorBinding f0PitchOutput;
    OnnxRvcTensorBinding generatorContentInput;
    OnnxRvcTensorBinding generatorF0Input;
    OnnxRvcTensorBinding generatorAudioOutput;
    std::vector<OnnxRvcTensorBinding> generatorAuxiliaryInputs;
};

class OnnxRvcEngine final {
public:
    OnnxRvcEngine();
    explicit OnnxRvcEngine(std::filesystem::path runtimeDllPath);
    ~OnnxRvcEngine();

    OnnxRvcEngine(const OnnxRvcEngine&) = delete;
    OnnxRvcEngine& operator=(const OnnxRvcEngine&) = delete;
    OnnxRvcEngine(OnnxRvcEngine&&) noexcept;
    OnnxRvcEngine& operator=(OnnxRvcEngine&&) noexcept;

    [[nodiscard]] static std::vector<std::filesystem::path> defaultRuntimeDllCandidates();
    [[nodiscard]] static std::filesystem::path defaultNativeModelRoot();
    [[nodiscard]] static core::Expected<bool> validateGraphContract(
        const OnnxRvcGraphContract& contract,
        const OnnxRvcModelDescription& description);
    [[nodiscard]] static core::Expected<OnnxRvcPipelinePlan> resolvePipelinePlan(
        const OnnxRvcGraphContract& contract);

    [[nodiscard]] core::Expected<OnnxRuntimeInfo> probeRuntime() const;
    [[nodiscard]] core::Expected<OnnxRvcModelBundle> loadModelBundle(
        const std::filesystem::path& bundleRoot) const;
    [[nodiscard]] core::Expected<bool> configureModelBundle(OnnxRvcModelBundle bundle);
    [[nodiscard]] core::Expected<OnnxRvcModelDescription> describeConfiguredModel() const;
    [[nodiscard]] core::Expected<OnnxRvcPipelinePlan> describeConfiguredPipeline() const;
    [[nodiscard]] core::Expected<OnnxRvcResult> convertChunk(
        const OnnxRvcRequest& request) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voxstudio::rvc
