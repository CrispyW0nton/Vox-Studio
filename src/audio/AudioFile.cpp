#include "audio/AudioFile.h"

#include <miniaudio.h>
#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

namespace voxstudio::audio {
namespace {

[[nodiscard]] core::Error audioFileError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

using SndFileHandle = std::unique_ptr<SNDFILE, decltype(&sf_close)>;

[[nodiscard]] SndFileHandle openSndFile(const std::filesystem::path& path,
                                        const int mode,
                                        SF_INFO& info) {
    return SndFileHandle{sf_open(path.string().c_str(), mode, &info), &sf_close};
}

} // namespace

core::Expected<PcmAudioBuffer> decodeAudioFile(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return audioFileError("Audio file does not exist.");
    }

    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
    ma_decoder decoder{};
    const auto initResult = ma_decoder_init_file(path.string().c_str(), &config, &decoder);
    if (initResult != MA_SUCCESS) {
        return audioFileError("Audio file could not be decoded.");
    }

    ma_uint64 frameCount = 0;
    const auto lengthResult = ma_decoder_get_length_in_pcm_frames(&decoder, &frameCount);
    if (lengthResult != MA_SUCCESS || decoder.outputSampleRate == 0 ||
        decoder.outputChannels == 0) {
        ma_decoder_uninit(&decoder);
        return audioFileError("Audio file length could not be measured.");
    }

    PcmAudioBuffer audio;
    audio.sampleRate = static_cast<int>(decoder.outputSampleRate);
    audio.channels = static_cast<int>(decoder.outputChannels);
    audio.samples.resize(static_cast<std::size_t>(frameCount) *
                         static_cast<std::size_t>(audio.channels));

    ma_uint64 framesRead = 0;
    const auto readResult =
        ma_decoder_read_pcm_frames(&decoder, audio.samples.data(), frameCount, &framesRead);
    ma_decoder_uninit(&decoder);
    if (readResult != MA_SUCCESS) {
        return audioFileError("Audio file samples could not be read.");
    }

    audio.samples.resize(static_cast<std::size_t>(framesRead) *
                         static_cast<std::size_t>(audio.channels));
    return audio;
}

core::Expected<PcmAudioBuffer> readOpusFile(const std::filesystem::path& path) {
    SF_INFO info{};
    auto file = openSndFile(path, SFM_READ, info);
    if (file == nullptr || info.frames < 0 || info.samplerate <= 0 || info.channels <= 0) {
        return audioFileError("Opus take could not be opened.");
    }

    PcmAudioBuffer audio;
    audio.sampleRate = info.samplerate;
    audio.channels = info.channels;
    audio.samples.resize(static_cast<std::size_t>(info.frames) *
                         static_cast<std::size_t>(audio.channels));

    const auto framesRead = sf_readf_float(file.get(), audio.samples.data(), info.frames);
    if (framesRead < 0) {
        return audioFileError("Opus take could not be decoded.");
    }
    audio.samples.resize(static_cast<std::size_t>(framesRead) *
                         static_cast<std::size_t>(audio.channels));
    return audio;
}

core::Expected<bool> writeOpusFile(const std::filesystem::path& path,
                                   const PcmAudioBuffer& audio) {
    if (audio.empty()) {
        return audioFileError("Cannot write an empty Opus take.");
    }

    try {
        std::filesystem::create_directories(path.parent_path());
    } catch (const std::filesystem::filesystem_error& exception) {
        return audioFileError(exception.what());
    }

    SF_INFO info{};
    info.samplerate = audio.sampleRate;
    info.channels = audio.channels;
    info.format = SF_FORMAT_OGG | SF_FORMAT_OPUS;

    auto file = openSndFile(path, SFM_WRITE, info);
    if (file == nullptr) {
        return audioFileError("Opus take could not be opened for writing.");
    }

    const auto written = sf_writef_float(file.get(), audio.samples.data(),
                                         static_cast<sf_count_t>(audio.frameCount()));
    if (written != static_cast<sf_count_t>(audio.frameCount())) {
        return audioFileError("Opus take could not be fully written.");
    }
    return true;
}

core::Expected<bool> writeWavFile(const std::filesystem::path& path,
                                  const PcmAudioBuffer& audio) {
    if (audio.empty()) {
        return audioFileError("Cannot write an empty WAV file.");
    }

    try {
        std::filesystem::create_directories(path.parent_path());
    } catch (const std::filesystem::filesystem_error& exception) {
        return audioFileError(exception.what());
    }

    SF_INFO info{};
    info.samplerate = audio.sampleRate;
    info.channels = audio.channels;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    auto file = openSndFile(path, SFM_WRITE, info);
    if (file == nullptr) {
        return audioFileError("WAV file could not be opened for writing.");
    }

    const auto written = sf_writef_float(file.get(), audio.samples.data(),
                                         static_cast<sf_count_t>(audio.frameCount()));
    if (written != static_cast<sf_count_t>(audio.frameCount())) {
        return audioFileError("WAV file could not be fully written.");
    }
    return true;
}

core::Expected<PcmAudioBuffer>
pcm16LittleEndianToPcm(std::span<const std::uint8_t> bytes,
                       const int sampleRate,
                       const int channels) {
    if (sampleRate <= 0 || channels <= 0) {
        return audioFileError("PCM stream metadata is invalid.");
    }
    if ((bytes.size() % 2U) != 0U) {
        return audioFileError("PCM stream has an odd byte count.");
    }

    PcmAudioBuffer audio;
    audio.sampleRate = sampleRate;
    audio.channels = channels;
    audio.samples.reserve(bytes.size() / 2U);
    for (std::size_t index = 0; index < bytes.size(); index += 2U) {
        const auto low = static_cast<std::uint16_t>(bytes[index]);
        const auto high = static_cast<std::uint16_t>(bytes[index + 1U]) << 8U;
        const auto sample = static_cast<std::int16_t>(low | high);
        audio.samples.push_back(static_cast<float>(sample) / 32768.0F);
    }
    return audio;
}

std::vector<std::uint8_t> pcmToPcm16LittleEndian(const PcmAudioBuffer& audio) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(audio.samples.size() * 2U);
    for (const float sample : audio.samples) {
        const auto clamped = std::clamp(sample, -1.0F, 1.0F);
        const auto scaled = static_cast<int>(std::lround(clamped * 32767.0F));
        const auto value = static_cast<std::int16_t>(
            std::clamp(scaled,
                       static_cast<int>(std::numeric_limits<std::int16_t>::min()),
                       static_cast<int>(std::numeric_limits<std::int16_t>::max())));
        bytes.push_back(static_cast<std::uint8_t>(value & 0xFF));
        bytes.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
    }
    return bytes;
}

} // namespace voxstudio::audio
