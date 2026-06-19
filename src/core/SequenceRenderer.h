#pragma once

#include "audio/AudioTypes.h"
#include "core/Expected.h"
#include "core/Sequence.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <vector>

namespace voxstudio::core {

struct SequenceRenderOptions final {
    int sampleRate{48000};
    int channels{2};
    int crossfadeMs{10};
};

struct SequenceGenerationOptions final {
    std::size_t maxConcurrency{3};
};

struct GeneratedLineAudio final {
    std::string lineId;
    audio::PcmAudioBuffer audio;
};

struct SequenceExportResult final {
    std::filesystem::path path;
    int durationMs{0};
    std::size_t frameCount{0};
};

using SequenceLineAudioProvider =
    std::function<Expected<audio::PcmAudioBuffer>(const SequenceLine&)>;

class SequenceRenderer final {
public:
    [[nodiscard]] Expected<RenderedSequence>
    render(const std::filesystem::path& projectRoot,
           const Sequence& sequence,
           const SequenceRenderOptions& options = {}) const;

    [[nodiscard]] Expected<SequenceExportResult>
    exportWav(const std::filesystem::path& projectRoot,
              const Sequence& sequence,
              const std::filesystem::path& outputPath,
              const SequenceRenderOptions& options = {}) const;

    [[nodiscard]] Expected<SequenceExportResult>
    exportOpus(const std::filesystem::path& projectRoot,
               const Sequence& sequence,
               const std::filesystem::path& outputPath,
               const SequenceRenderOptions& options = {}) const;

    [[nodiscard]] Expected<std::vector<GeneratedLineAudio>>
    generateLineAudio(const Sequence& sequence,
                      const SequenceLineAudioProvider& provider,
                      const SequenceGenerationOptions& options = {}) const;
};

} // namespace voxstudio::core
