#include "core/SequenceRenderer.h"

#include "audio/AudioFile.h"
#include "audio/Resampler.h"

#include <algorithm>
#include <future>
#include <optional>
#include <string>

namespace voxstudio::core {
namespace {

[[nodiscard]] Expected<audio::PcmAudioBuffer>
convertChannels(const audio::PcmAudioBuffer& input, const int channels) {
    if (channels <= 0 || input.channels <= 0) {
        return makeError(ErrorCode::InvalidArgument, "Audio channel count is invalid.");
    }
    if (input.channels == channels) {
        return input;
    }

    audio::PcmAudioBuffer output;
    output.sampleRate = input.sampleRate;
    output.channels = channels;
    output.samples.reserve(input.frameCount() * static_cast<std::size_t>(channels));

    for (std::size_t frame = 0; frame < input.frameCount(); ++frame) {
        if (channels == 1) {
            float sum = 0.0F;
            for (int channel = 0; channel < input.channels; ++channel) {
                const auto index = (frame * static_cast<std::size_t>(input.channels)) +
                                   static_cast<std::size_t>(channel);
                sum += input.samples[index];
            }
            output.samples.push_back(sum / static_cast<float>(input.channels));
            continue;
        }

        for (int channel = 0; channel < channels; ++channel) {
            const auto sourceChannel = input.channels == 1 ? 0 : channel % input.channels;
            const auto index = (frame * static_cast<std::size_t>(input.channels)) +
                               static_cast<std::size_t>(sourceChannel);
            output.samples.push_back(input.samples[index]);
        }
    }

    return output;
}

[[nodiscard]] Expected<audio::PcmAudioBuffer>
prepareClip(const audio::PcmAudioBuffer& input, const SequenceRenderOptions& options) {
    if (input.empty()) {
        return makeError(ErrorCode::InvalidArgument, "Sequence line audio is empty.");
    }

    auto audio = input;
    if (audio.sampleRate != options.sampleRate) {
        auto resampled = audio::resamplePcm(audio, options.sampleRate);
        if (!resampled) {
            return resampled.error();
        }
        audio = std::move(resampled).value();
    }

    return convertChannels(audio, options.channels);
}

[[nodiscard]] bool isSafeRelativePath(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) {
        return false;
    }
    for (const auto& part : path) {
        if (part == "..") {
            return false;
        }
    }
    return true;
}

[[nodiscard]] Expected<audio::PcmAudioBuffer>
loadLineAudio(const std::filesystem::path& projectRoot, const SequenceLine& line) {
    if (line.takeFilePath.empty()) {
        return makeError(ErrorCode::InvalidArgument,
                         "Sequence line has no active take: " + line.lineId);
    }

    const std::filesystem::path relativePath{line.takeFilePath};
    if (!isSafeRelativePath(relativePath)) {
        return makeError(ErrorCode::InvalidArgument, "Take path is not project-relative.");
    }

    const auto absolutePath = projectRoot / relativePath;
    const auto extension = absolutePath.extension().string();
    if (extension == ".opus" || extension == ".ogg") {
        return audio::readOpusFile(absolutePath);
    }
    return audio::decodeAudioFile(absolutePath);
}

void appendSilence(audio::PcmAudioBuffer& output, const int gapMs) {
    if (gapMs <= 0 || output.sampleRate <= 0 || output.channels <= 0) {
        return;
    }

    const auto frames =
        static_cast<std::size_t>((static_cast<long long>(output.sampleRate) * gapMs) / 1000);
    output.samples.insert(output.samples.end(),
                          frames * static_cast<std::size_t>(output.channels),
                          0.0F);
}

void appendClip(audio::PcmAudioBuffer& output,
                const audio::PcmAudioBuffer& clip,
                const int crossfadeFrames) {
    auto overlap = std::min<std::size_t>(clip.frameCount(), output.frameCount());
    overlap = std::min(overlap, static_cast<std::size_t>(std::max(0, crossfadeFrames)));
    if (overlap == 0U) {
        output.samples.insert(output.samples.end(), clip.samples.begin(), clip.samples.end());
        return;
    }

    const auto channels = static_cast<std::size_t>(output.channels);
    const auto outputStartFrame = output.frameCount() - overlap;
    for (std::size_t frame = 0; frame < overlap; ++frame) {
        const auto mix = static_cast<float>(frame + 1U) / static_cast<float>(overlap + 1U);
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto outputIndex = ((outputStartFrame + frame) * channels) + channel;
            const auto clipIndex = (frame * channels) + channel;
            const auto oldSample = output.samples[outputIndex];
            const auto newSample = clip.samples[clipIndex];
            output.samples[outputIndex] = (oldSample * (1.0F - mix)) + (newSample * mix);
        }
    }

    const auto remainingStart = overlap * channels;
    output.samples.insert(output.samples.end(),
                          clip.samples.begin() + static_cast<std::ptrdiff_t>(remainingStart),
                          clip.samples.end());
}

[[nodiscard]] Expected<GeneratedLineAudio>
generateOneLine(const SequenceLine& line, const SequenceLineAudioProvider& provider) {
    auto audio = provider(line);
    if (!audio) {
        return audio.error();
    }
    return GeneratedLineAudio{line.lineId, std::move(audio).value()};
}

} // namespace

Expected<RenderedSequence>
SequenceRenderer::render(const std::filesystem::path& projectRoot,
                         const Sequence& sequence,
                         const SequenceRenderOptions& options) const {
    if (projectRoot.empty() || sequence.lines.empty()) {
        return makeError(ErrorCode::InvalidArgument,
                         "Project root and at least one sequence line are required.");
    }
    if (options.sampleRate <= 0 || options.channels <= 0) {
        return makeError(ErrorCode::InvalidArgument, "Sequence render options are invalid.");
    }

    RenderedSequence rendered;
    rendered.audio.sampleRate = options.sampleRate;
    rendered.audio.channels = options.channels;

    const auto crossfadeFrames =
        static_cast<int>((static_cast<long long>(options.sampleRate) *
                          std::max(0, options.crossfadeMs)) /
                         1000);
    bool canCrossfade = false;
    for (std::size_t index = 0; index < sequence.lines.size(); ++index) {
        auto lineAudio = loadLineAudio(projectRoot, sequence.lines[index]);
        if (!lineAudio) {
            return lineAudio.error();
        }

        auto clip = prepareClip(lineAudio.value(), options);
        if (!clip) {
            return clip.error();
        }

        const auto startFrame = rendered.audio.frameCount();
        appendClip(rendered.audio, clip.value(), canCrossfade ? crossfadeFrames : 0);
        rendered.segments.push_back(SequenceRenderSegment{
            sequence.lines[index].lineId, startFrame, clip.value().frameCount(),
            sequence.lines[index].gapMs});

        const bool hasNext = index + 1U < sequence.lines.size();
        if (hasNext && sequence.lines[index].gapMs > 0) {
            appendSilence(rendered.audio, sequence.lines[index].gapMs);
            canCrossfade = false;
        } else {
            canCrossfade = hasNext;
        }
    }

    return rendered;
}

Expected<SequenceExportResult>
SequenceRenderer::exportWav(const std::filesystem::path& projectRoot,
                            const Sequence& sequence,
                            const std::filesystem::path& outputPath,
                            const SequenceRenderOptions& options) const {
    auto rendered = render(projectRoot, sequence, options);
    if (!rendered) {
        return rendered.error();
    }

    auto written = audio::writeWavFile(outputPath, rendered.value().audio);
    if (!written) {
        return written.error();
    }
    return SequenceExportResult{outputPath,
                                sequenceDurationMs(rendered.value().audio),
                                rendered.value().audio.frameCount()};
}

Expected<SequenceExportResult>
SequenceRenderer::exportOpus(const std::filesystem::path& projectRoot,
                             const Sequence& sequence,
                             const std::filesystem::path& outputPath,
                             const SequenceRenderOptions& options) const {
    auto rendered = render(projectRoot, sequence, options);
    if (!rendered) {
        return rendered.error();
    }

    auto written = audio::writeOpusFile(outputPath, rendered.value().audio);
    if (!written) {
        return written.error();
    }
    return SequenceExportResult{outputPath,
                                sequenceDurationMs(rendered.value().audio),
                                rendered.value().audio.frameCount()};
}

Expected<std::vector<GeneratedLineAudio>>
SequenceRenderer::generateLineAudio(const Sequence& sequence,
                                    const SequenceLineAudioProvider& provider,
                                    const SequenceGenerationOptions& options) const {
    if (sequence.lines.empty()) {
        return makeError(ErrorCode::InvalidArgument, "Sequence must contain lines.");
    }
    if (!provider) {
        return makeError(ErrorCode::InvalidArgument, "Sequence audio provider is missing.");
    }
    if (options.maxConcurrency == 0U) {
        return makeError(ErrorCode::InvalidArgument, "Concurrency cap must be at least one.");
    }

    std::vector<std::optional<GeneratedLineAudio>> ordered(sequence.lines.size());
    std::vector<std::future<Expected<GeneratedLineAudio>>> running;
    std::vector<std::size_t> runningIndexes;
    std::size_t nextIndex = 0;

    while (nextIndex < sequence.lines.size() || !running.empty()) {
        while (nextIndex < sequence.lines.size() &&
               running.size() < options.maxConcurrency) {
            const auto lineIndex = nextIndex;
            runningIndexes.push_back(lineIndex);
            running.push_back(std::async(std::launch::async, [&, lineIndex]() {
                return generateOneLine(sequence.lines[lineIndex], provider);
            }));
            ++nextIndex;
        }

        auto generated = running.front().get();
        const auto lineIndex = runningIndexes.front();
        running.erase(running.begin());
        runningIndexes.erase(runningIndexes.begin());
        if (!generated) {
            return generated.error();
        }
        ordered[lineIndex] = std::move(generated).value();
    }

    std::vector<GeneratedLineAudio> result;
    result.reserve(ordered.size());
    for (auto& generated : ordered) {
        if (!generated.has_value()) {
            return makeError(ErrorCode::InvalidArgument, "Sequence generation was incomplete.");
        }
        result.push_back(std::move(generated).value());
    }
    return result;
}

} // namespace voxstudio::core
