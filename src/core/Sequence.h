#pragma once

#include "audio/AudioTypes.h"

#include <cstddef>
#include <string>
#include <vector>

namespace voxstudio::core {

struct SequenceLine final {
    int order{0};
    std::string lineId;
    std::string characterName;
    std::string voiceId;
    std::string text;
    std::string activeTakeId;
    std::string takeFilePath;
    int gapMs{250};
};

struct Sequence final {
    std::string id;
    std::string name;
    std::string createdAt;
    std::vector<SequenceLine> lines;
};

struct SequenceRenderSegment final {
    std::string lineId;
    std::size_t startFrame{0};
    std::size_t frameCount{0};
    int gapMs{0};
};

struct RenderedSequence final {
    audio::PcmAudioBuffer audio;
    std::vector<SequenceRenderSegment> segments;
};

[[nodiscard]] int sequenceDurationMs(const audio::PcmAudioBuffer& audio) noexcept;

} // namespace voxstudio::core
