#pragma once

#include "audio/AudioTypes.h"
#include "core/Expected.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace voxstudio::audio {

class AudioEngine final {
public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    AudioEngine(AudioEngine&&) noexcept;
    AudioEngine& operator=(AudioEngine&&) noexcept;

    [[nodiscard]] core::Expected<bool> queuePcm(const PcmAudioBuffer& audio);
    [[nodiscard]] core::Expected<bool>
    queuePcm16LittleEndian(std::span<const std::uint8_t> bytes, int sampleRate, int channels);
    [[nodiscard]] core::Expected<bool> playFile(const std::filesystem::path& path);
    void clear() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voxstudio::audio
