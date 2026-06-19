#include "audio/AudioFile.h"
#include "rvc/OnnxRvcEngine.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options final {
    std::filesystem::path runtimeDll;
    std::filesystem::path bundleRoot;
    std::filesystem::path inputWav;
    std::filesystem::path outputWav;
    std::filesystem::path reportJson;
    int pitchShiftSemitones{0};
};

[[nodiscard]] bool readInt(std::string_view text, int& value) {
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} && result.ptr == last;
}

[[nodiscard]] std::string usage() {
    return "Usage: VoxStudioRvcNativeSmoke --bundle <dir> --input <wav> --output <wav> "
           "--report <json> [--runtime <onnxruntime.dll>] [--pitch <semitones>]";
}

[[nodiscard]] voxstudio::core::Expected<Options> parseOptions(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view key{argv[index]};
        const auto nextValue = [&]() -> voxstudio::core::Expected<std::string_view> {
            if ((index + 1) >= argc) {
                return voxstudio::core::makeError(voxstudio::core::ErrorCode::InvalidArgument,
                                                  std::string{key} + " needs a value.");
            }
            ++index;
            return std::string_view{argv[index]};
        };

        if (key == "--runtime") {
            auto value = nextValue();
            if (!value) {
                return value.error();
            }
            options.runtimeDll = std::filesystem::path{std::string{value.value()}};
        } else if (key == "--bundle") {
            auto value = nextValue();
            if (!value) {
                return value.error();
            }
            options.bundleRoot = std::filesystem::path{std::string{value.value()}};
        } else if (key == "--input") {
            auto value = nextValue();
            if (!value) {
                return value.error();
            }
            options.inputWav = std::filesystem::path{std::string{value.value()}};
        } else if (key == "--output") {
            auto value = nextValue();
            if (!value) {
                return value.error();
            }
            options.outputWav = std::filesystem::path{std::string{value.value()}};
        } else if (key == "--report") {
            auto value = nextValue();
            if (!value) {
                return value.error();
            }
            options.reportJson = std::filesystem::path{std::string{value.value()}};
        } else if (key == "--pitch") {
            auto value = nextValue();
            if (!value) {
                return value.error();
            }
            if (!readInt(value.value(), options.pitchShiftSemitones)) {
                return voxstudio::core::makeError(voxstudio::core::ErrorCode::InvalidArgument,
                                                  "--pitch must be an integer.");
            }
        } else if (key == "--help" || key == "-h") {
            return voxstudio::core::makeError(voxstudio::core::ErrorCode::InvalidArgument,
                                              usage());
        } else {
            return voxstudio::core::makeError(voxstudio::core::ErrorCode::InvalidArgument,
                                              "Unknown argument: " + std::string{key});
        }
    }

    if (options.bundleRoot.empty() || options.inputWav.empty() || options.outputWav.empty() ||
        options.reportJson.empty()) {
        return voxstudio::core::makeError(voxstudio::core::ErrorCode::InvalidArgument,
                                          usage());
    }
    return options;
}

[[nodiscard]] double durationSeconds(const voxstudio::audio::PcmAudioBuffer& audio) {
    if (audio.sampleRate <= 0) {
        return 0.0;
    }
    return static_cast<double>(audio.frameCount()) / static_cast<double>(audio.sampleRate);
}

[[nodiscard]] voxstudio::core::Expected<nlohmann::json> renderNative(
    const Options& options) {
    auto inputAudio = voxstudio::audio::decodeAudioFile(options.inputWav);
    if (!inputAudio) {
        return inputAudio.error();
    }

    voxstudio::rvc::OnnxRvcEngine engine{options.runtimeDll};
    auto bundle = engine.loadModelBundle(options.bundleRoot);
    if (!bundle) {
        return bundle.error();
    }

    auto configured = engine.configureModelBundle(bundle.value());
    if (!configured) {
        return configured.error();
    }

    voxstudio::rvc::OnnxRvcRequest request;
    request.pcm16Audio = voxstudio::audio::pcmToPcm16LittleEndian(inputAudio.value());
    request.sampleRate = inputAudio.value().sampleRate;
    request.channels = inputAudio.value().channels;
    request.pitchShiftSemitones = options.pitchShiftSemitones;

    auto converted = engine.convertChunk(request);
    if (!converted) {
        return converted.error();
    }

    auto outputAudio = voxstudio::audio::pcm16LittleEndianToPcm(converted.value().pcm16Audio,
                                                               converted.value().sampleRate,
                                                               converted.value().channels);
    if (!outputAudio) {
        return outputAudio.error();
    }

    auto written = voxstudio::audio::writeWavFile(options.outputWav, outputAudio.value());
    if (!written) {
        return written.error();
    }

    auto description = engine.describeConfiguredModel();
    if (!description) {
        return description.error();
    }

    nlohmann::json report;
    report["bundle"] = options.bundleRoot.string();
    report["input_wav"] = options.inputWav.string();
    report["output_wav"] = options.outputWav.string();
    report["model_id"] = bundle.value().modelId;
    report["input_sample_rate"] = inputAudio.value().sampleRate;
    report["input_channels"] = inputAudio.value().channels;
    report["input_seconds"] = durationSeconds(inputAudio.value());
    report["output_sample_rate"] = outputAudio.value().sampleRate;
    report["output_channels"] = outputAudio.value().channels;
    report["output_seconds"] = durationSeconds(outputAudio.value());
    report["latency_ms"] = converted.value().latencyMs;
    report["pitch_shift_semitones"] = options.pitchShiftSemitones;
    report["native_runtime"] = description.value().runtime.runtimeDllPath.string();
    report["native_runtime_version"] = description.value().runtime.version;
    return report;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc == 2) {
            const std::string_view argument{argv[1]};
            if (argument == "--help" || argument == "-h") {
                std::fprintf(stdout, "%s\n", usage().c_str());
                return 0;
            }
        }

        auto options = parseOptions(argc, argv);
        if (!options) {
            std::fprintf(stderr, "%s\n", options.error().message.c_str());
            return 2;
        }

        auto report = renderNative(options.value());
        if (!report) {
            std::fprintf(stderr, "%s\n", report.error().message.c_str());
            return 1;
        }

        const auto reportDirectory = options.value().reportJson.parent_path();
        if (!reportDirectory.empty()) {
            std::filesystem::create_directories(reportDirectory);
        }
        std::ofstream output{options.value().reportJson, std::ios::trunc};
        output << report.value().dump(2) << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "%s\n", exception.what());
        return 1;
    }
}
