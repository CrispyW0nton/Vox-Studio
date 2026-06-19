#include "audio/AudioEngine.h"

#include "audio/AudioFile.h"
#include "audio/Resampler.h"

#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace voxstudio::audio {
namespace {

constexpr int kPlaybackSampleRate = 48000;
constexpr int kPlaybackChannels = 2;
constexpr std::size_t kRingSeconds = 12;
constexpr std::size_t kRingSamples =
    static_cast<std::size_t>(kPlaybackSampleRate * kPlaybackChannels) * kRingSeconds;

[[nodiscard]] core::Error audioEngineError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] PcmAudioBuffer stereoCopy(const PcmAudioBuffer& input) {
    if (input.channels == kPlaybackChannels) {
        return input;
    }

    PcmAudioBuffer output;
    output.sampleRate = input.sampleRate;
    output.channels = kPlaybackChannels;
    output.samples.reserve(input.frameCount() * static_cast<std::size_t>(kPlaybackChannels));
    for (std::size_t frame = 0; frame < input.frameCount(); ++frame) {
        if (input.channels == 1) {
            const auto sample = input.samples[frame];
            output.samples.push_back(sample);
            output.samples.push_back(sample);
        } else {
            float mixed = 0.0F;
            for (int channel = 0; channel < input.channels; ++channel) {
                mixed += input.samples[(frame * static_cast<std::size_t>(input.channels)) +
                                       static_cast<std::size_t>(channel)];
            }
            mixed /= static_cast<float>(input.channels);
            output.samples.push_back(mixed);
            output.samples.push_back(mixed);
        }
    }
    return output;
}

} // namespace

class AudioEngine::Impl final {
public:
    Impl()
        : m_ring(kRingSamples + 1U, 0.0F) {
        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = kPlaybackChannels;
        config.sampleRate = kPlaybackSampleRate;
        config.dataCallback = &Impl::dataCallback;
        config.pUserData = this;

        m_initialized = ma_device_init(nullptr, &config, &m_device) == MA_SUCCESS;
        if (m_initialized) {
            m_initialized = ma_device_start(&m_device) == MA_SUCCESS;
        }
    }

    ~Impl() {
        if (m_initialized) {
            ma_device_uninit(&m_device);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    [[nodiscard]] core::Expected<bool> queuePcm(const PcmAudioBuffer& audio) {
        if (!m_initialized) {
            return audioEngineError("Unable to initialize the audio output device.");
        }
        if (audio.empty()) {
            return audioEngineError("Cannot play an empty audio buffer.");
        }

        auto stereo = stereoCopy(audio);
        if (stereo.sampleRate != kPlaybackSampleRate) {
            auto resampled = resamplePcm(stereo, kPlaybackSampleRate);
            if (!resampled) {
                return resampled.error();
            }
            stereo = std::move(resampled).value();
        }

        pushSamples(stereo.samples);
        return true;
    }

    [[nodiscard]] core::Expected<bool>
    queuePcm16LittleEndian(std::span<const std::uint8_t> bytes,
                           const int sampleRate,
                           const int channels) {
        auto audio = pcm16LittleEndianToPcm(bytes, sampleRate, channels);
        if (!audio) {
            return audio.error();
        }
        return queuePcm(audio.value());
    }

    [[nodiscard]] core::Expected<bool> playFile(const std::filesystem::path& path) {
        auto audio = decodeAudioFile(path);
        if (!audio) {
            return audio.error();
        }
        return queuePcm(audio.value());
    }

    void clear() noexcept {
        m_readIndex.store(0, std::memory_order_release);
        m_writeIndex.store(0, std::memory_order_release);
    }

private:
    static void dataCallback(ma_device* device,
                             void* output,
                             const void*,
                             const ma_uint32 frameCount) noexcept {
        auto* self = static_cast<Impl*>(device->pUserData);
        if (self == nullptr) {
            std::memset(output,
                        0,
                        static_cast<std::size_t>(frameCount * kPlaybackChannels) *
                            sizeof(float));
            return;
        }
        self->readSamples(static_cast<float*>(output),
                          static_cast<std::size_t>(frameCount * kPlaybackChannels));
    }

    [[nodiscard]] std::size_t nextIndex(const std::size_t index) const noexcept {
        const auto next = index + 1U;
        return next == m_ring.size() ? 0U : next;
    }

    void readSamples(float* output, const std::size_t sampleCount) noexcept {
        auto read = m_readIndex.load(std::memory_order_acquire);
        const auto write = m_writeIndex.load(std::memory_order_acquire);
        for (std::size_t index = 0; index < sampleCount; ++index) {
            if (read == write) {
                output[index] = 0.0F;
                continue;
            }
            output[index] = m_ring[read];
            read = nextIndex(read);
        }
        m_readIndex.store(read, std::memory_order_release);
    }

    void pushSamples(const std::vector<float>& samples) noexcept {
        auto write = m_writeIndex.load(std::memory_order_acquire);
        auto read = m_readIndex.load(std::memory_order_acquire);
        for (const float sample : samples) {
            const auto next = nextIndex(write);
            if (next == read) {
                read = nextIndex(read);
                m_readIndex.store(read, std::memory_order_release);
            }
            m_ring[write] = sample;
            write = next;
        }
        m_writeIndex.store(write, std::memory_order_release);
    }

    ma_device m_device{};
    std::vector<float> m_ring;
    std::atomic<std::size_t> m_readIndex{0};
    std::atomic<std::size_t> m_writeIndex{0};
    bool m_initialized{false};
};

AudioEngine::AudioEngine()
    : m_impl(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine() = default;
AudioEngine::AudioEngine(AudioEngine&&) noexcept = default;
AudioEngine& AudioEngine::operator=(AudioEngine&&) noexcept = default;

core::Expected<bool> AudioEngine::queuePcm(const PcmAudioBuffer& audio) {
    return m_impl->queuePcm(audio);
}

core::Expected<bool>
AudioEngine::queuePcm16LittleEndian(std::span<const std::uint8_t> bytes,
                                    const int sampleRate,
                                    const int channels) {
    return m_impl->queuePcm16LittleEndian(bytes, sampleRate, channels);
}

core::Expected<bool> AudioEngine::playFile(const std::filesystem::path& path) {
    return m_impl->playFile(path);
}

void AudioEngine::clear() noexcept {
    m_impl->clear();
}

} // namespace voxstudio::audio
