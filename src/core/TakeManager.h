#pragma once

#include "audio/AudioTypes.h"
#include "core/Expected.h"
#include "db/TakeRepository.h"

#include <filesystem>
#include <string>

namespace voxstudio::core {

struct VoiceSettings final {
    double stability{0.5};
    double similarityBoost{0.75};
    double style{0.0};
    bool useSpeakerBoost{true};
};

struct SavedTake final {
    db::TakeRecord take;
    std::filesystem::path absolutePath;
};

[[nodiscard]] VoiceSettings defaultVoiceSettings() noexcept;
[[nodiscard]] std::string voiceSettingsToJson(const VoiceSettings& settings);
[[nodiscard]] Expected<VoiceSettings> voiceSettingsFromJson(const std::string& jsonText);

class TakeManager final {
public:
    [[nodiscard]] Expected<SavedTake>
    saveTtsTake(const std::filesystem::path& projectRoot,
                const std::string& lineId,
                const std::string& voiceId,
                const audio::PcmAudioBuffer& audio,
                const VoiceSettings& settings) const;

    [[nodiscard]] Expected<SavedTake>
    saveStsTake(const std::filesystem::path& projectRoot,
                const std::string& lineId,
                const std::string& voiceId,
                const audio::PcmAudioBuffer& audio,
                const VoiceSettings& settings) const;

    [[nodiscard]] Expected<SavedTake>
    saveRvcLocalTake(const std::filesystem::path& projectRoot,
                     const std::string& lineId,
                     const std::string& rvcModelId,
                     const audio::PcmAudioBuffer& audio) const;

private:
    db::TakeRepository m_repository;
};

} // namespace voxstudio::core
