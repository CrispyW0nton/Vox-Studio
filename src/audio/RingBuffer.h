#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <vector>

namespace voxstudio::audio {

constexpr int kRealtimeSampleRate = 48000;
constexpr int kRealtimeChannels = 1;
constexpr int kMaxRealtimeFrameMs = 20;
constexpr std::size_t kMaxRealtimeFrameFrames =
    static_cast<std::size_t>((kRealtimeSampleRate * kMaxRealtimeFrameMs) / 1000);
constexpr std::size_t kMaxRealtimeFrameSamples =
    kMaxRealtimeFrameFrames * static_cast<std::size_t>(kRealtimeChannels);

struct AudioFrame final {
    int sampleRate{kRealtimeSampleRate};
    int channels{kRealtimeChannels};
    std::size_t frameCount{0};
    double capturedAtSeconds{0.0};
    float rms{0.0F};
    std::array<float, kMaxRealtimeFrameSamples> samples{};

    [[nodiscard]] bool empty() const noexcept {
        return frameCount == 0U || sampleRate <= 0 || channels <= 0;
    }

    [[nodiscard]] std::size_t sampleCount() const noexcept {
        return frameCount * static_cast<std::size_t>(channels);
    }
};

class AudioRingBuffer final {
public:
    explicit AudioRingBuffer(std::size_t capacityFrames);

    AudioRingBuffer(const AudioRingBuffer&) = delete;
    AudioRingBuffer& operator=(const AudioRingBuffer&) = delete;
    AudioRingBuffer(AudioRingBuffer&&) = delete;
    AudioRingBuffer& operator=(AudioRingBuffer&&) = delete;

    [[nodiscard]] bool tryPush(const AudioFrame& frame) noexcept;
    [[nodiscard]] bool tryPop(AudioFrame& frame) noexcept;
    void clear() noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t sizeApprox() const noexcept;

private:
    [[nodiscard]] std::size_t nextIndex(std::size_t index) const noexcept;

    std::vector<AudioFrame> m_frames;
    std::atomic<std::size_t> m_readIndex{0};
    std::atomic<std::size_t> m_writeIndex{0};
};

} // namespace voxstudio::audio
