#pragma once

#include "audio/RingBuffer.h"
#include "core/Expected.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace voxstudio::audio {

struct AudioDeviceInfo final {
    int index{-1};
    std::string name;
    bool isDefault{false};
};

struct CaptureConfig final {
    int inputDeviceIndex{-1};
    int outputDeviceIndex{-1};
    int frameMs{10};
    float gain{1.0F};
};

struct CaptureStats final {
    bool running{false};
    bool monitorEnabled{false};
    float inputRms{0.0F};
    std::size_t capturedFramesQueued{0};
    std::size_t monitorFramesQueued{0};
    std::size_t droppedCaptureFrames{0};
    std::size_t droppedMonitorFrames{0};
};

class Capture final {
public:
    Capture();
    ~Capture();

    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
    Capture(Capture&&) noexcept;
    Capture& operator=(Capture&&) noexcept;

    [[nodiscard]] static core::Expected<std::vector<AudioDeviceInfo>> listInputDevices();
    [[nodiscard]] static core::Expected<std::vector<AudioDeviceInfo>> listOutputDevices();

    [[nodiscard]] core::Expected<bool> start(const CaptureConfig& config);
    void stop() noexcept;
    void setMonitorEnabled(bool enabled) noexcept;
    void setGain(float gain) noexcept;
    [[nodiscard]] bool tryPopCapturedFrame(AudioFrame& frame) noexcept;
    [[nodiscard]] bool tryPushMonitorFrame(const AudioFrame& frame) noexcept;
    [[nodiscard]] CaptureStats stats() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voxstudio::audio
