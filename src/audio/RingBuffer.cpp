#include "audio/RingBuffer.h"

#include <algorithm>

namespace voxstudio::audio {

AudioRingBuffer::AudioRingBuffer(const std::size_t capacityFrames)
    : m_frames(std::max<std::size_t>(capacityFrames + 1U, 2U)) {}

bool AudioRingBuffer::tryPush(const AudioFrame& frame) noexcept {
    const auto write = m_writeIndex.load(std::memory_order_relaxed);
    const auto next = nextIndex(write);
    if (next == m_readIndex.load(std::memory_order_acquire)) {
        return false;
    }

    m_frames[write] = frame;
    m_writeIndex.store(next, std::memory_order_release);
    return true;
}

bool AudioRingBuffer::tryPop(AudioFrame& frame) noexcept {
    const auto read = m_readIndex.load(std::memory_order_relaxed);
    if (read == m_writeIndex.load(std::memory_order_acquire)) {
        return false;
    }

    frame = m_frames[read];
    m_readIndex.store(nextIndex(read), std::memory_order_release);
    return true;
}

void AudioRingBuffer::clear() noexcept {
    m_readIndex.store(0, std::memory_order_release);
    m_writeIndex.store(0, std::memory_order_release);
}

std::size_t AudioRingBuffer::capacity() const noexcept {
    return m_frames.size() - 1U;
}

std::size_t AudioRingBuffer::sizeApprox() const noexcept {
    const auto read = m_readIndex.load(std::memory_order_acquire);
    const auto write = m_writeIndex.load(std::memory_order_acquire);
    if (write >= read) {
        return write - read;
    }
    return (m_frames.size() - read) + write;
}

std::size_t AudioRingBuffer::nextIndex(const std::size_t index) const noexcept {
    const auto next = index + 1U;
    return next == m_frames.size() ? 0U : next;
}

} // namespace voxstudio::audio
