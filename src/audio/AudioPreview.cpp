#include "audio/AudioPreview.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <filesystem>
#include <string>
#include <utility>

namespace voxstudio::audio {
namespace {

[[nodiscard]] core::Error audioError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

} // namespace

class AudioPreviewPlayer::Impl final {
public:
    Impl() {
        m_initialized = ma_engine_init(nullptr, &m_engine) == MA_SUCCESS;
    }

    ~Impl() {
        if (m_initialized) {
            ma_engine_uninit(&m_engine);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    [[nodiscard]] core::Expected<bool> playFile(const std::filesystem::path& filePath) {
        if (!m_initialized) {
            return audioError("Unable to initialize the audio preview engine.");
        }
        if (!std::filesystem::exists(filePath)) {
            return audioError("Audio preview file does not exist.");
        }

        const auto result = ma_engine_play_sound(&m_engine, filePath.string().c_str(), nullptr);
        if (result != MA_SUCCESS) {
            return audioError("Unable to play the selected audio preview.");
        }

        return true;
    }

private:
    ma_engine m_engine{};
    bool m_initialized{false};
};

AudioPreviewPlayer::AudioPreviewPlayer()
    : m_impl(std::make_unique<Impl>()) {}

AudioPreviewPlayer::~AudioPreviewPlayer() = default;
AudioPreviewPlayer::AudioPreviewPlayer(AudioPreviewPlayer&&) noexcept = default;
AudioPreviewPlayer& AudioPreviewPlayer::operator=(AudioPreviewPlayer&&) noexcept = default;

core::Expected<bool> AudioPreviewPlayer::playFile(const std::filesystem::path& filePath) {
    return m_impl->playFile(filePath);
}

core::Expected<double> audioDurationSeconds(const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) {
        return audioError("Audio sample file does not exist.");
    }

    ma_decoder decoder{};
    const auto initResult = ma_decoder_init_file(filePath.string().c_str(), nullptr, &decoder);
    if (initResult != MA_SUCCESS) {
        return audioError("Audio sample could not be decoded.");
    }

    ma_uint64 frameCount = 0;
    const auto lengthResult = ma_decoder_get_length_in_pcm_frames(&decoder, &frameCount);
    const auto sampleRate = decoder.outputSampleRate;
    ma_decoder_uninit(&decoder);

    if (lengthResult != MA_SUCCESS || sampleRate == 0) {
        return audioError("Audio sample duration could not be measured.");
    }

    return static_cast<double>(frameCount) / static_cast<double>(sampleRate);
}

} // namespace voxstudio::audio
