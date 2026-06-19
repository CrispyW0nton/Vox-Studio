#include "rvc/F0Extractor.h"
#include "rvc/HubertEncoder.h"
#include "rvc/NativeRvcAudio.h"
#include "rvc/OnnxRvcEngine.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_onnx_rvc_test_" + std::to_string(now));
        std::filesystem::create_directories(m_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

void writeText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output{path, std::ios::trunc};
    output << text;
}

void appendPcm16(std::vector<std::uint8_t>& bytes, const std::int16_t sample) {
    const auto value = static_cast<std::uint16_t>(sample);
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

[[nodiscard]] voxstudio::rvc::OnnxRvcPipelinePlan basicPipelinePlan(
    std::vector<std::int64_t> hubertDimensions,
    std::vector<std::int64_t> f0Dimensions) {
    voxstudio::rvc::OnnxRvcPipelinePlan plan;
    plan.hubertAudioInput = voxstudio::rvc::OnnxRvcTensorBinding{
        "HuBERT encoder",
        voxstudio::rvc::OnnxTensorDescription{"waveform", "float32",
                                              std::move(hubertDimensions), {}, "audio"}};
    plan.f0AudioInput = voxstudio::rvc::OnnxRvcTensorBinding{
        "RMVPE F0 extractor",
        voxstudio::rvc::OnnxTensorDescription{"f0_waveform", "float32",
                                              std::move(f0Dimensions), {}, "audio"}};
    return plan;
}

} // namespace

TEST_CASE("ONNX RVC engine validates native model bundles", "[rvc][onnx]") {
    const TemporaryDirectory directory;
    const auto bundleRoot = directory.path() / "hero";
    std::filesystem::create_directories(bundleRoot);
    writeText(bundleRoot / "generator.onnx", "generator");
    writeText(bundleRoot / "hubert.onnx", "hubert");
    writeText(bundleRoot / "rmvpe.onnx", "rmvpe");
    writeText(bundleRoot / "native_manifest.json",
              R"({"model_id":"hero","generator_onnx":"generator.onnx",)"
              R"("hubert_onnx":"hubert.onnx","f0_onnx":"rmvpe.onnx",)"
              R"("sample_rate":48000,"hop_length":160})");

    const voxstudio::rvc::OnnxRvcEngine engine{directory.path() / "missing_ort.dll"};
    auto bundle = engine.loadModelBundle(bundleRoot);
    REQUIRE(bundle.hasValue());
    CHECK(bundle.value().modelId == "hero");
    CHECK(bundle.value().sampleRate == 48000);
    CHECK(bundle.value().hopLength == 160);

    voxstudio::rvc::OnnxRvcRequest request;
    request.pcm16Audio = {0x01, 0x00};
    auto converted = engine.convertChunk(request);
    REQUIRE_FALSE(converted.hasValue());
}

TEST_CASE("ONNX RVC engine parses native graph contracts", "[rvc][onnx]") {
    const TemporaryDirectory directory;
    const auto bundleRoot = directory.path() / "hero";
    std::filesystem::create_directories(bundleRoot);
    writeText(bundleRoot / "generator.onnx", "generator");
    writeText(bundleRoot / "hubert.onnx", "hubert");
    writeText(bundleRoot / "rmvpe.onnx", "rmvpe");
    writeText(bundleRoot / "native_manifest.json",
              R"({"model_id":"hero","generator_onnx":"generator.onnx",)"
              R"("hubert_onnx":"hubert.onnx","f0_onnx":"rmvpe.onnx",)"
              R"("sample_rate":48000,"hop_length":160,"graph_contract":)"
              R"({"required":true,"generator":{"inputs":[)"
              R"({"name":"content","element_type":"float32","role":"content"},)"
              R"({"name":"pitch","element_type":"float32","role":"f0"}],)"
              R"("outputs":[{"name":"wave","element_type":"float32","role":"audio"}]},)"
              R"("hubert":{"inputs":[],"outputs":[]},"f0":{"inputs":[],"outputs":[]}}})");

    const voxstudio::rvc::OnnxRvcEngine engine{directory.path() / "missing_ort.dll"};
    auto bundle = engine.loadModelBundle(bundleRoot);
    REQUIRE(bundle.hasValue());
    CHECK(bundle.value().graphContract.required);
    REQUIRE(bundle.value().graphContract.generator.inputs.size() == 2U);
    CHECK(bundle.value().graphContract.generator.inputs.front().name == "content");
    CHECK(bundle.value().graphContract.generator.inputs.front().semanticRole == "content");
    CHECK(bundle.value().graphContract.generator.outputs.front().semanticRole == "audio");
}

TEST_CASE("ONNX RVC pipeline plans resolve semantic tensor roles", "[rvc][onnx]") {
    voxstudio::rvc::OnnxRvcGraphContract contract;
    contract.required = true;
    contract.generator.label = "RVC generator";
    contract.generator.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"phone", "float32", {1, -1, 256}, {},
                                              "content"});
    contract.generator.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"pitchf", "float32", {1, -1}, {}, "f0"});
    contract.generator.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"sid", "int64", {1}, {}, "speaker"});
    contract.generator.outputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"wave", "float32", {1, -1}, {}, "audio"});
    contract.hubert.label = "HuBERT encoder";
    contract.hubert.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"waveform", "float32", {1, -1}, {}, "audio"});
    contract.hubert.outputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"features", "float32", {1, -1, 256}, {},
                                              "content"});
    contract.f0.label = "RMVPE F0 extractor";
    contract.f0.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"waveform", "float32", {1, -1}, {}, "audio"});
    contract.f0.outputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"pitchf", "float32", {1, -1}, {}, "f0"});

    auto plan = voxstudio::rvc::OnnxRvcEngine::resolvePipelinePlan(contract);
    REQUIRE(plan.hasValue());
    CHECK(plan.value().hubertAudioInput.tensor.name == "waveform");
    CHECK(plan.value().hubertContentOutput.tensor.name == "features");
    CHECK(plan.value().f0PitchOutput.tensor.name == "pitchf");
    CHECK(plan.value().generatorContentInput.tensor.name == "phone");
    CHECK(plan.value().generatorAudioOutput.tensor.name == "wave");
    REQUIRE(plan.value().generatorAuxiliaryInputs.size() == 1U);
    CHECK(plan.value().generatorAuxiliaryInputs.front().tensor.semanticRole == "speaker");

    contract.generator.inputs.at(1).semanticRole.clear();
    plan = voxstudio::rvc::OnnxRvcEngine::resolvePipelinePlan(contract);
    REQUIRE_FALSE(plan.hasValue());
    CHECK(plan.error().message.find("RVC generator input contract needs tensor role 'f0'") !=
          std::string::npos);
}

TEST_CASE("Native RVC audio preparation mixes PCM16 into tensor inputs", "[rvc][onnx]") {
    voxstudio::rvc::OnnxRvcRequest request;
    request.sampleRate = 16000;
    request.channels = 2;
    appendPcm16(request.pcm16Audio, 16384);
    appendPcm16(request.pcm16Audio, 0);
    appendPcm16(request.pcm16Audio, 0);
    appendPcm16(request.pcm16Audio, -16384);

    voxstudio::rvc::OnnxRvcModelBundle bundle;
    bundle.sampleRate = 16000;
    auto plan = basicPipelinePlan({1, -1}, {-1});

    auto prepared = voxstudio::rvc::prepareNativeRvcAudio(request, bundle, plan);
    REQUIRE(prepared.hasValue());
    CHECK(prepared.value().hubertAudio.name == "waveform");
    CHECK(prepared.value().hubertAudio.sampleRate == 16000);
    CHECK(prepared.value().hubertAudio.dimensions == std::vector<std::int64_t>{1, 2});
    REQUIRE(prepared.value().hubertAudio.values.size() == 2U);
    CHECK(std::fabs(prepared.value().hubertAudio.values.at(0) - 0.25F) < 0.0001F);
    CHECK(std::fabs(prepared.value().hubertAudio.values.at(1) + 0.25F) < 0.0001F);
    CHECK(prepared.value().f0Audio.dimensions == std::vector<std::int64_t>{2});
}

TEST_CASE("Native RVC audio preparation rejects incompatible tensor shapes", "[rvc][onnx]") {
    voxstudio::rvc::OnnxRvcRequest request;
    request.sampleRate = 16000;
    request.channels = 1;
    appendPcm16(request.pcm16Audio, 1024);
    appendPcm16(request.pcm16Audio, 2048);

    voxstudio::rvc::OnnxRvcModelBundle bundle;
    bundle.sampleRate = 16000;
    auto plan = basicPipelinePlan({1, 3}, {-1});

    auto prepared = voxstudio::rvc::prepareNativeRvcAudio(request, bundle, plan);
    REQUIRE_FALSE(prepared.hasValue());
    CHECK(prepared.error().message.find("fixed shape expects 3 samples") != std::string::npos);
}

TEST_CASE("Native RVC auxiliary generator defaults are prepared by role", "[rvc][onnx]") {
    auto plan = basicPipelinePlan({1, -1}, {-1});
    plan.generatorAuxiliaryInputs.push_back(voxstudio::rvc::OnnxRvcTensorBinding{
        "RVC generator",
        voxstudio::rvc::OnnxTensorDescription{"sid", "int64", {1}, {}, "speaker"}});
    plan.generatorAuxiliaryInputs.push_back(voxstudio::rvc::OnnxRvcTensorBinding{
        "RVC generator",
        voxstudio::rvc::OnnxTensorDescription{"length", "int64", {1}, {}, "length"}});
    plan.generatorAuxiliaryInputs.push_back(voxstudio::rvc::OnnxRvcTensorBinding{
        "RVC generator",
        voxstudio::rvc::OnnxTensorDescription{"noise", "float32", {1, -1}, {}, "noise"}});

    auto auxiliaries = voxstudio::rvc::prepareGeneratorAuxiliaryInputs(plan, 4);
    REQUIRE(auxiliaries.hasValue());
    REQUIRE(auxiliaries.value().size() == 3U);
    CHECK(auxiliaries.value().at(0).int64Values == std::vector<std::int64_t>{0});
    CHECK(auxiliaries.value().at(1).int64Values == std::vector<std::int64_t>{4});
    CHECK(auxiliaries.value().at(2).dimensions == std::vector<std::int64_t>{1, 4});
    CHECK(auxiliaries.value().at(2).floatValues == std::vector<float>{0.0F, 0.0F, 0.0F,
                                                                       0.0F});
}

TEST_CASE("Native RVC auxiliary generator defaults reject unknown roles", "[rvc][onnx]") {
    auto plan = basicPipelinePlan({1, -1}, {-1});
    plan.generatorAuxiliaryInputs.push_back(voxstudio::rvc::OnnxRvcTensorBinding{
        "RVC generator",
        voxstudio::rvc::OnnxTensorDescription{"mystery", "float32", {1}, {}, "style"}});

    auto auxiliaries = voxstudio::rvc::prepareGeneratorAuxiliaryInputs(plan, 4);
    REQUIRE_FALSE(auxiliaries.hasValue());
    CHECK(auxiliaries.error().message.find("Unsupported generator auxiliary tensor role") !=
          std::string::npos);
}

TEST_CASE("ONNX RVC graph contracts validate model descriptions", "[rvc][onnx]") {
    voxstudio::rvc::OnnxRvcGraphContract contract;
    contract.required = true;
    contract.generator.label = "RVC generator";
    contract.generator.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"audio", "float32", {1, -1}, {}});
    contract.generator.outputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"wave", "float32", {1, -1}, {}});
    contract.hubert.label = "HuBERT encoder";
    contract.f0.label = "RMVPE F0 extractor";

    voxstudio::rvc::OnnxRvcModelDescription description;
    description.generator.label = "RVC generator";
    description.generator.inputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"audio", "float32", {1, 3200}, {}});
    description.generator.outputs.push_back(
        voxstudio::rvc::OnnxTensorDescription{"wave", "float32", {1, 3200}, {}});
    description.hubert.label = "HuBERT encoder";
    description.f0.label = "RMVPE F0 extractor";

    auto valid = voxstudio::rvc::OnnxRvcEngine::validateGraphContract(contract, description);
    REQUIRE_FALSE(valid.hasValue());
    CHECK(valid.error().message.find("HuBERT encoder graph contract") != std::string::npos);

    contract.required = false;
    valid = voxstudio::rvc::OnnxRvcEngine::validateGraphContract(contract, description);
    REQUIRE(valid.hasValue());

    contract.generator.inputs.front().elementType = "int64";
    valid = voxstudio::rvc::OnnxRvcEngine::validateGraphContract(contract, description);
    REQUIRE_FALSE(valid.hasValue());
    CHECK(valid.error().message.find("expected audio type int64") != std::string::npos);
}

TEST_CASE("ONNX RVC engine requires a loadable runtime before configuration", "[rvc][onnx]") {
    const TemporaryDirectory directory;
    const auto bundleRoot = directory.path() / "hero";
    const auto invalidRuntime = directory.path() / "not_onnxruntime.dll";
    std::filesystem::create_directories(bundleRoot);
    writeText(invalidRuntime, "not a dll");
    writeText(bundleRoot / "generator.onnx", "generator");
    writeText(bundleRoot / "hubert.onnx", "hubert");
    writeText(bundleRoot / "rmvpe.onnx", "rmvpe");
    writeText(bundleRoot / "native_manifest.json",
              R"({"model_id":"hero","generator_onnx":"generator.onnx",)"
              R"("hubert_onnx":"hubert.onnx","f0_onnx":"rmvpe.onnx",)"
              R"("sample_rate":48000,"hop_length":160})");

    voxstudio::rvc::OnnxRvcEngine engine{invalidRuntime};
    auto bundle = engine.loadModelBundle(bundleRoot);
    REQUIRE(bundle.hasValue());

    auto configured = engine.configureModelBundle(std::move(bundle.value()));
    REQUIRE_FALSE(configured.hasValue());
    CHECK(configured.error().message.find("Failed to load") != std::string::npos);
}

TEST_CASE("ONNX RVC engine describes only configured native sessions", "[rvc][onnx]") {
    const voxstudio::rvc::OnnxRvcEngine engine;

    auto description = engine.describeConfiguredModel();
    REQUIRE_FALSE(description.hasValue());
    CHECK(description.error().message.find("not configured") != std::string::npos);
}

TEST_CASE("ONNX RVC feature extractors validate model files", "[rvc][onnx]") {
    const TemporaryDirectory directory;
    const auto rmvpe = directory.path() / "rmvpe.onnx";
    const auto hubert = directory.path() / "hubert.onnx";
    writeText(rmvpe, "rmvpe");
    writeText(hubert, "hubert");

    voxstudio::rvc::F0Extractor f0;
    auto f0Configured = f0.configure(voxstudio::rvc::F0ExtractorConfig{rmvpe, 48000, 160});
    REQUIRE(f0Configured.hasValue());
    CHECK(f0.isConfigured());
    CHECK(f0.describe().hasValue());

    voxstudio::rvc::HubertEncoder encoder;
    auto encoderConfigured =
        encoder.configure(voxstudio::rvc::HubertEncoderConfig{hubert, 16000, 256});
    REQUIRE(encoderConfigured.hasValue());
    CHECK(encoder.isConfigured());
    CHECK(encoder.describe().hasValue());
}

TEST_CASE("ONNX Runtime probe reports availability without throwing", "[rvc][onnx]") {
    const voxstudio::rvc::OnnxRvcEngine engine;
    auto runtime = engine.probeRuntime();
    REQUIRE(runtime.hasValue());
    CHECK(!runtime.value().message.empty());
}
