#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace voxstudio::audio {

struct SpeechPhraseBufferConfig final {
    std::size_t preRollBytes{3840};
    std::size_t trailingSilenceBytes{9600};
    std::size_t retainedTrailingSilenceBytes{3200};
    std::size_t minimumSpeechBytes{2560};
    std::size_t maximumPhraseBytes{192000};
};

class SpeechPhraseBuffer final {
public:
    explicit SpeechPhraseBuffer(SpeechPhraseBufferConfig config = {});

    [[nodiscard]] std::optional<std::vector<std::uint8_t>>
    append(std::span<const std::uint8_t> pcmBytes, bool speechActive);
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> flush();
    void reset() noexcept;

private:
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> finishPhrase();
    void appendPreRoll(std::span<const std::uint8_t> pcmBytes);

    SpeechPhraseBufferConfig m_config;
    std::vector<std::uint8_t> m_preRoll;
    std::vector<std::uint8_t> m_phrase;
    std::size_t m_speechBytes{0};
    std::size_t m_trailingSilenceBytes{0};
    bool m_phraseActive{false};
};

} // namespace voxstudio::audio
