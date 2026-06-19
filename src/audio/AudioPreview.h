#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <memory>

namespace voxstudio::audio {

class AudioPreviewPlayer final {
public:
    AudioPreviewPlayer();
    ~AudioPreviewPlayer();

    AudioPreviewPlayer(const AudioPreviewPlayer&) = delete;
    AudioPreviewPlayer& operator=(const AudioPreviewPlayer&) = delete;
    AudioPreviewPlayer(AudioPreviewPlayer&&) noexcept;
    AudioPreviewPlayer& operator=(AudioPreviewPlayer&&) noexcept;

    [[nodiscard]] core::Expected<bool> playFile(const std::filesystem::path& filePath);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

[[nodiscard]] core::Expected<double> audioDurationSeconds(const std::filesystem::path& filePath);

} // namespace voxstudio::audio
