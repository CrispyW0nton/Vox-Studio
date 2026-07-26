#include "audio/SpeechPhraseBuffer.h"

#include <algorithm>
#include <utility>

namespace voxstudio::audio {

SpeechPhraseBuffer::SpeechPhraseBuffer(SpeechPhraseBufferConfig config)
    : m_config(config) {
    m_preRoll.reserve(m_config.preRollBytes);
    m_phrase.reserve(m_config.maximumPhraseBytes);
}

std::optional<std::vector<std::uint8_t>>
SpeechPhraseBuffer::append(const std::span<const std::uint8_t> pcmBytes,
                           const bool speechActive) {
    if (pcmBytes.empty()) {
        return std::nullopt;
    }

    if (speechActive) {
        if (!m_phraseActive) {
            m_phraseActive = true;
            m_phrase = std::move(m_preRoll);
            m_preRoll.clear();
        }
        m_phrase.insert(m_phrase.end(), pcmBytes.begin(), pcmBytes.end());
        m_speechBytes += pcmBytes.size();
        m_trailingSilenceBytes = 0;
    } else if (m_phraseActive) {
        m_phrase.insert(m_phrase.end(), pcmBytes.begin(), pcmBytes.end());
        m_trailingSilenceBytes += pcmBytes.size();
    } else {
        appendPreRoll(pcmBytes);
        return std::nullopt;
    }

    if (m_phrase.size() >= m_config.maximumPhraseBytes ||
        m_trailingSilenceBytes >= m_config.trailingSilenceBytes) {
        return finishPhrase();
    }
    return std::nullopt;
}

std::optional<std::vector<std::uint8_t>> SpeechPhraseBuffer::flush() {
    if (!m_phraseActive) {
        m_preRoll.clear();
        return std::nullopt;
    }
    return finishPhrase();
}

void SpeechPhraseBuffer::reset() noexcept {
    m_preRoll.clear();
    m_phrase.clear();
    m_speechBytes = 0;
    m_trailingSilenceBytes = 0;
    m_phraseActive = false;
}

std::optional<std::vector<std::uint8_t>> SpeechPhraseBuffer::finishPhrase() {
    if (m_trailingSilenceBytes > m_config.retainedTrailingSilenceBytes) {
        const auto trimBytes =
            m_trailingSilenceBytes - m_config.retainedTrailingSilenceBytes;
        if (trimBytes <= m_phrase.size()) {
            m_phrase.resize(m_phrase.size() - trimBytes);
        }
    }

    std::optional<std::vector<std::uint8_t>> result;
    if (m_speechBytes >= m_config.minimumSpeechBytes) {
        result = std::move(m_phrase);
    }

    m_phrase.clear();
    m_speechBytes = 0;
    m_trailingSilenceBytes = 0;
    m_phraseActive = false;
    return result;
}

void SpeechPhraseBuffer::appendPreRoll(const std::span<const std::uint8_t> pcmBytes) {
    m_preRoll.insert(m_preRoll.end(), pcmBytes.begin(), pcmBytes.end());
    if (m_preRoll.size() <= m_config.preRollBytes) {
        return;
    }

    const auto excess = m_preRoll.size() - m_config.preRollBytes;
    m_preRoll.erase(m_preRoll.begin(),
                    m_preRoll.begin() + static_cast<std::ptrdiff_t>(excess));
}

} // namespace voxstudio::audio
