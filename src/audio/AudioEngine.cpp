#include "audio/AudioEngine.h"

#include "audio/AudioFile.h"
#include "audio/Resampler.h"

#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
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
constexpr float kMinPitchFactor = 0.25F;
constexpr float kMaxPitchFactor = 4.0F;

[[nodiscard]] core::Error audioEngineError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] float decibelsToLinear(const float decibels) noexcept {
    return std::pow(10.0F, decibels / 20.0F);
}

[[nodiscard]] float filterCoefficient(const float cutoffHz, const int sampleRate) noexcept {
    const auto dt = 1.0F / static_cast<float>(sampleRate);
    const auto rc = 1.0F / (2.0F * 3.1415926535F * cutoffHz);
    return dt / (rc + dt);
}

[[nodiscard]] ma_device_id* selectedDeviceId(std::vector<ma_device_info>& devices,
                                             const int index) noexcept {
    if (index < 0 || index >= static_cast<int>(devices.size())) {
        return nullptr;
    }
    return &devices[static_cast<std::size_t>(index)].id;
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

[[nodiscard]] PcmAudioBuffer applyOutputFx(const PcmAudioBuffer& input,
                                           const OutputFxSettings& settings) {
    PcmAudioBuffer output = input;
    if (output.empty()) {
        return output;
    }

    const auto pitchFactor = std::clamp(
        std::pow(2.0F, static_cast<float>(settings.pitchShiftSemitones) / 12.0F),
        kMinPitchFactor,
        kMaxPitchFactor);
    output.sampleRate = std::max(1, static_cast<int>(
                                        std::lround(static_cast<float>(output.sampleRate) *
                                                    pitchFactor)));

    const auto bassGain = decibelsToLinear(settings.bassDb);
    const auto midGain = decibelsToLinear(settings.midDb);
    const auto trebleGain = decibelsToLinear(settings.trebleDb);
    const auto volume = std::clamp(settings.volume, 0.0F, 2.0F);
    const bool flatEq = std::abs(settings.bassDb) < 0.001F &&
                        std::abs(settings.midDb) < 0.001F &&
                        std::abs(settings.trebleDb) < 0.001F;

    if (flatEq) {
        for (auto& sample : output.samples) {
            sample = std::clamp(sample * volume, -1.0F, 1.0F);
        }
        return output;
    }

    const auto lowAlpha = filterCoefficient(240.0F, input.sampleRate);
    const auto highAlpha = filterCoefficient(4200.0F, input.sampleRate);
    std::vector<float> lowState(static_cast<std::size_t>(input.channels), 0.0F);
    std::vector<float> highLowState(static_cast<std::size_t>(input.channels), 0.0F);

    const auto frameCount = output.frameCount();
    for (std::size_t frame = 0; frame < frameCount; ++frame) {
        for (int channel = 0; channel < output.channels; ++channel) {
            const auto sampleIndex =
                (frame * static_cast<std::size_t>(output.channels)) +
                static_cast<std::size_t>(channel);
            const auto sample = output.samples[sampleIndex];
            auto& low = lowState[static_cast<std::size_t>(channel)];
            auto& highLow = highLowState[static_cast<std::size_t>(channel)];
            low += lowAlpha * (sample - low);
            highLow += highAlpha * (sample - highLow);
            const auto high = sample - highLow;
            const auto mid = sample - low - high;
            output.samples[sampleIndex] =
                std::clamp((low * bassGain + mid * midGain + high * trebleGain) * volume,
                           -1.0F,
                           1.0F);
        }
    }

    return output;
}

} // namespace

class AudioEngine::Impl final {
public:
    Impl()
        : m_ring(kRingSamples + 1U, 0.0F) {
        const auto initialized = initializeDevice(-1);
        (void)initialized;
    }

    ~Impl() {
        uninitializeDevice();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    [[nodiscard]] core::Expected<bool> queuePcm(const PcmAudioBuffer& audio) {
        std::lock_guard lock{m_controlMutex};
        if (!m_initialized) {
            return audioEngineError("Unable to initialize the audio output device.");
        }
        if (audio.empty()) {
            return audioEngineError("Cannot play an empty audio buffer.");
        }

        auto stereo = stereoCopy(applyOutputFx(audio, m_fxSettings));
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

    [[nodiscard]] core::Expected<bool> setOutputDeviceIndex(const int outputDeviceIndex) {
        std::lock_guard lock{m_controlMutex};
        return initializeDevice(outputDeviceIndex);
    }

    void setOutputFxSettings(const OutputFxSettings& settings) noexcept {
        std::lock_guard lock{m_controlMutex};
        m_fxSettings = OutputFxSettings{std::clamp(settings.volume, 0.0F, 2.0F),
                                        std::clamp(settings.bassDb, -12.0F, 12.0F),
                                        std::clamp(settings.midDb, -12.0F, 12.0F),
                                        std::clamp(settings.trebleDb, -12.0F, 12.0F),
                                        std::clamp(settings.pitchShiftSemitones, -24, 24)};
    }

private:
    [[nodiscard]] core::Expected<bool> initializeDevice(const int outputDeviceIndex) {
        uninitializeDevice();
        clear();
        m_outputDeviceIndex = outputDeviceIndex;

        if (ma_context_init(nullptr, 0, nullptr, &m_context) != MA_SUCCESS) {
            return audioEngineError("Unable to initialize the audio output context.");
        }
        m_contextInitialized = true;

        ma_device_info* playbackDevices = nullptr;
        ma_uint32 playbackCount = 0;
        ma_device_info* captureDevices = nullptr;
        ma_uint32 captureCount = 0;
        if (ma_context_get_devices(&m_context,
                                   &playbackDevices,
                                   &playbackCount,
                                   &captureDevices,
                                   &captureCount) != MA_SUCCESS) {
            uninitializeDevice();
            return audioEngineError("Unable to enumerate audio output devices.");
        }
        if (playbackDevices != nullptr && playbackCount > 0) {
            m_playbackDeviceIds.assign(playbackDevices, playbackDevices + playbackCount);
        }

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = kPlaybackChannels;
        config.playback.pDeviceID = selectedDeviceId(m_playbackDeviceIds, m_outputDeviceIndex);
        config.sampleRate = kPlaybackSampleRate;
        config.dataCallback = &Impl::dataCallback;
        config.pUserData = this;

        if (ma_device_init(&m_context, &config, &m_device) != MA_SUCCESS) {
            uninitializeDevice();
            return audioEngineError("Unable to initialize the selected audio output device.");
        }
        m_initialized = true;
        if (ma_device_start(&m_device) != MA_SUCCESS) {
            uninitializeDevice();
            return audioEngineError("Unable to start the selected audio output device.");
        }
        return true;
    }

    void uninitializeDevice() noexcept {
        if (m_initialized) {
            ma_device_uninit(&m_device);
            m_initialized = false;
        }
        if (m_contextInitialized) {
            ma_context_uninit(&m_context);
            m_contextInitialized = false;
        }
        m_playbackDeviceIds.clear();
    }

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

    ma_context m_context{};
    ma_device m_device{};
    std::vector<ma_device_info> m_playbackDeviceIds;
    std::vector<float> m_ring;
    std::atomic<std::size_t> m_readIndex{0};
    std::atomic<std::size_t> m_writeIndex{0};
    std::mutex m_controlMutex;
    OutputFxSettings m_fxSettings;
    int m_outputDeviceIndex{-1};
    bool m_initialized{false};
    bool m_contextInitialized{false};
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

core::Expected<bool> AudioEngine::setOutputDeviceIndex(const int outputDeviceIndex) {
    return m_impl->setOutputDeviceIndex(outputDeviceIndex);
}

void AudioEngine::setOutputFxSettings(const OutputFxSettings& settings) noexcept {
    m_impl->setOutputFxSettings(settings);
}

void AudioEngine::clear() noexcept {
    m_impl->clear();
}

} // namespace voxstudio::audio
