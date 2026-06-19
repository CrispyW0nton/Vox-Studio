#include "audio/Capture.h"

#include <miniaudio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

namespace voxstudio::audio {
namespace {

constexpr std::size_t kCaptureQueueFrames = 256;
constexpr std::size_t kMonitorQueueFrames = 256;
constexpr int kPlaybackChannels = 2;

[[nodiscard]] core::Error captureError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] float clampedGain(const float gain) noexcept {
    return std::clamp(gain, 0.0F, 4.0F);
}

[[nodiscard]] float frameRms(const float* samples, const std::size_t count) noexcept {
    if (samples == nullptr || count == 0U) {
        return 0.0F;
    }

    double sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        sum += static_cast<double>(samples[index]) * static_cast<double>(samples[index]);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(count)));
}

[[nodiscard]] ma_device_id* selectedDeviceId(std::vector<ma_device_info>& devices,
                                             const int index) noexcept {
    if (index < 0 || index >= static_cast<int>(devices.size())) {
        return nullptr;
    }
    return &devices[static_cast<std::size_t>(index)].id;
}

[[nodiscard]] core::Expected<std::vector<AudioDeviceInfo>>
listDevices(const ma_device_type deviceType) {
    ma_context context{};
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) {
        return captureError("Unable to initialize audio device enumeration.");
    }

    ma_device_info* playbackDevices = nullptr;
    ma_uint32 playbackCount = 0;
    ma_device_info* captureDevices = nullptr;
    ma_uint32 captureCount = 0;
    const auto result = ma_context_get_devices(&context,
                                               &playbackDevices,
                                               &playbackCount,
                                               &captureDevices,
                                               &captureCount);
    if (result != MA_SUCCESS) {
        ma_context_uninit(&context);
        return captureError("Unable to enumerate audio devices.");
    }

    const auto* source = deviceType == ma_device_type_capture ? captureDevices : playbackDevices;
    const auto count = deviceType == ma_device_type_capture ? captureCount : playbackCount;

    std::vector<AudioDeviceInfo> devices;
    devices.reserve(count);
    for (ma_uint32 index = 0; index < count; ++index) {
        devices.push_back(AudioDeviceInfo{static_cast<int>(index),
                                          source[index].name,
                                          source[index].isDefault != 0});
    }

    ma_context_uninit(&context);
    return devices;
}

} // namespace

class Capture::Impl final {
public:
    Impl()
        : m_captureFrames(kCaptureQueueFrames)
        , m_monitorFrames(kMonitorQueueFrames) {}

    ~Impl() {
        stop();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    [[nodiscard]] core::Expected<bool> start(const CaptureConfig& config) {
        stop();

        m_config = config;
        m_config.frameMs = std::clamp(m_config.frameMs, 10, kMaxRealtimeFrameMs);
        m_gain.store(clampedGain(m_config.gain), std::memory_order_release);
        m_captureFrames.clear();
        m_monitorFrames.clear();
        m_inputRms.store(0.0F, std::memory_order_release);
        m_droppedCaptureFrames.store(0, std::memory_order_release);
        m_droppedMonitorFrames.store(0, std::memory_order_release);
        m_captureClockSeconds = 0.0;

        if (ma_context_init(nullptr, 0, nullptr, &m_context) != MA_SUCCESS) {
            return captureError("Unable to initialize audio context.");
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
            stop();
            return captureError("Unable to enumerate audio devices.");
        }

        if (captureDevices != nullptr && captureCount > 0) {
            m_captureDeviceIds.assign(captureDevices, captureDevices + captureCount);
        }
        if (playbackDevices != nullptr && playbackCount > 0) {
            m_playbackDeviceIds.assign(playbackDevices, playbackDevices + playbackCount);
        }

        auto deviceConfig = ma_device_config_init(ma_device_type_duplex);
        deviceConfig.capture.format = ma_format_f32;
        deviceConfig.capture.channels = kRealtimeChannels;
        deviceConfig.capture.pDeviceID =
            selectedDeviceId(m_captureDeviceIds, m_config.inputDeviceIndex);
        deviceConfig.playback.format = ma_format_f32;
        deviceConfig.playback.channels = kPlaybackChannels;
        deviceConfig.playback.pDeviceID =
            selectedDeviceId(m_playbackDeviceIds, m_config.outputDeviceIndex);
        deviceConfig.sampleRate = kRealtimeSampleRate;
        deviceConfig.periodSizeInFrames =
            static_cast<ma_uint32>((kRealtimeSampleRate * m_config.frameMs) / 1000);
        deviceConfig.dataCallback = &Impl::dataCallback;
        deviceConfig.pUserData = this;

        if (ma_device_init(&m_context, &deviceConfig, &m_device) != MA_SUCCESS) {
            stop();
            return captureError("Unable to initialize audio capture device.");
        }
        m_deviceInitialized = true;

        if (ma_device_start(&m_device) != MA_SUCCESS) {
            stop();
            return captureError("Unable to start audio capture device.");
        }

        m_running.store(true, std::memory_order_release);
        return true;
    }

    void stop() noexcept {
        m_running.store(false, std::memory_order_release);
        if (m_deviceInitialized) {
            ma_device_uninit(&m_device);
            m_deviceInitialized = false;
        }
        if (m_contextInitialized) {
            ma_context_uninit(&m_context);
            m_contextInitialized = false;
        }
        m_captureDeviceIds.clear();
        m_playbackDeviceIds.clear();
    }

    void setMonitorEnabled(const bool enabled) noexcept {
        m_monitorEnabled.store(enabled, std::memory_order_release);
    }

    void setGain(const float gain) noexcept {
        m_gain.store(clampedGain(gain), std::memory_order_release);
    }

    [[nodiscard]] bool tryPopCapturedFrame(AudioFrame& frame) noexcept {
        return m_captureFrames.tryPop(frame);
    }

    [[nodiscard]] bool tryPushMonitorFrame(const AudioFrame& frame) noexcept {
        if (m_monitorFrames.tryPush(frame)) {
            return true;
        }
        m_droppedMonitorFrames.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    [[nodiscard]] CaptureStats stats() const noexcept {
        return CaptureStats{m_running.load(std::memory_order_acquire),
                            m_monitorEnabled.load(std::memory_order_acquire),
                            m_inputRms.load(std::memory_order_acquire),
                            m_captureFrames.sizeApprox(),
                            m_monitorFrames.sizeApprox(),
                            m_droppedCaptureFrames.load(std::memory_order_acquire),
                            m_droppedMonitorFrames.load(std::memory_order_acquire)};
    }

private:
    static void dataCallback(ma_device* device,
                             void* output,
                             const void* input,
                             const ma_uint32 frameCount) noexcept {
        auto* self = static_cast<Impl*>(device->pUserData);
        if (self == nullptr) {
            std::memset(output,
                        0,
                        static_cast<std::size_t>(frameCount * kPlaybackChannels) *
                            sizeof(float));
            return;
        }

        self->captureInput(static_cast<const float*>(input), static_cast<std::size_t>(frameCount));
        self->fillOutput(static_cast<float*>(output), static_cast<std::size_t>(frameCount));
    }

    void captureInput(const float* input, const std::size_t frames) noexcept {
        if (input == nullptr || frames == 0U) {
            return;
        }

        const auto gain = m_gain.load(std::memory_order_acquire);
        std::size_t offset = 0;
        while (offset < frames) {
            const auto frameCount = std::min(kMaxRealtimeFrameFrames, frames - offset);
            AudioFrame frame;
            frame.frameCount = frameCount;
            frame.capturedAtSeconds = m_captureClockSeconds;
            for (std::size_t index = 0; index < frameCount; ++index) {
                frame.samples[index] = std::clamp(input[offset + index] * gain, -1.0F, 1.0F);
            }
            frame.rms = frameRms(frame.samples.data(), frameCount);
            m_inputRms.store(frame.rms, std::memory_order_release);
            if (!m_captureFrames.tryPush(frame)) {
                m_droppedCaptureFrames.fetch_add(1, std::memory_order_relaxed);
            }
            m_captureClockSeconds += static_cast<double>(frameCount) /
                                     static_cast<double>(kRealtimeSampleRate);
            offset += frameCount;
        }
    }

    void fillOutput(float* output, const std::size_t frames) noexcept {
        if (output == nullptr || frames == 0U) {
            return;
        }

        std::memset(output, 0, frames * static_cast<std::size_t>(kPlaybackChannels) *
                                   sizeof(float));
        if (!m_monitorEnabled.load(std::memory_order_acquire)) {
            return;
        }

        std::size_t outputFrame = 0;
        while (outputFrame < frames) {
            AudioFrame frame;
            if (!m_monitorFrames.tryPop(frame)) {
                return;
            }

            for (std::size_t index = 0; index < frame.frameCount && outputFrame < frames;
                 ++index, ++outputFrame) {
                const auto sample = frame.samples[index];
                output[(outputFrame * kPlaybackChannels)] = sample;
                output[(outputFrame * kPlaybackChannels) + 1U] = sample;
            }
        }
    }

    ma_context m_context{};
    ma_device m_device{};
    std::vector<ma_device_info> m_captureDeviceIds;
    std::vector<ma_device_info> m_playbackDeviceIds;
    AudioRingBuffer m_captureFrames;
    AudioRingBuffer m_monitorFrames;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_monitorEnabled{false};
    std::atomic<float> m_gain{1.0F};
    std::atomic<float> m_inputRms{0.0F};
    std::atomic<std::size_t> m_droppedCaptureFrames{0};
    std::atomic<std::size_t> m_droppedMonitorFrames{0};
    CaptureConfig m_config;
    double m_captureClockSeconds{0.0};
    bool m_contextInitialized{false};
    bool m_deviceInitialized{false};
};

Capture::Capture()
    : m_impl(std::make_unique<Impl>()) {}

Capture::~Capture() = default;
Capture::Capture(Capture&&) noexcept = default;
Capture& Capture::operator=(Capture&&) noexcept = default;

core::Expected<std::vector<AudioDeviceInfo>> Capture::listInputDevices() {
    return listDevices(ma_device_type_capture);
}

core::Expected<std::vector<AudioDeviceInfo>> Capture::listOutputDevices() {
    return listDevices(ma_device_type_playback);
}

core::Expected<bool> Capture::start(const CaptureConfig& config) {
    return m_impl->start(config);
}

void Capture::stop() noexcept {
    m_impl->stop();
}

void Capture::setMonitorEnabled(const bool enabled) noexcept {
    m_impl->setMonitorEnabled(enabled);
}

void Capture::setGain(const float gain) noexcept {
    m_impl->setGain(gain);
}

bool Capture::tryPopCapturedFrame(AudioFrame& frame) noexcept {
    return m_impl->tryPopCapturedFrame(frame);
}

bool Capture::tryPushMonitorFrame(const AudioFrame& frame) noexcept {
    return m_impl->tryPushMonitorFrame(frame);
}

CaptureStats Capture::stats() const noexcept {
    return m_impl->stats();
}

} // namespace voxstudio::audio
