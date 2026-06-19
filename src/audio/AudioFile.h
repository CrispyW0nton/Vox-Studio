#pragma once

#include "audio/AudioTypes.h"
#include "core/Expected.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace voxstudio::audio {

[[nodiscard]] core::Expected<PcmAudioBuffer> decodeAudioFile(const std::filesystem::path& path);
[[nodiscard]] core::Expected<PcmAudioBuffer> readOpusFile(const std::filesystem::path& path);
[[nodiscard]] core::Expected<bool> writeOpusFile(const std::filesystem::path& path,
                                                 const PcmAudioBuffer& audio);
[[nodiscard]] core::Expected<bool> writeWavFile(const std::filesystem::path& path,
                                                const PcmAudioBuffer& audio);
[[nodiscard]] core::Expected<PcmAudioBuffer>
pcm16LittleEndianToPcm(std::span<const std::uint8_t> bytes, int sampleRate, int channels);
[[nodiscard]] std::vector<std::uint8_t> pcmToPcm16LittleEndian(const PcmAudioBuffer& audio);

} // namespace voxstudio::audio
