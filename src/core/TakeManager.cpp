#include "core/TakeManager.h"

#include "audio/AudioFile.h"
#include "audio/Resampler.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <string>

namespace voxstudio::core {
namespace {

constexpr int kTakeSampleRate = 48000;

[[nodiscard]] std::string safePathSegment(std::string value) {
    for (char& character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte) == 0 && character != '-' && character != '_') {
            character = '_';
        }
    }
    return value.empty() ? "line" : value;
}

[[nodiscard]] audio::PcmAudioBuffer monoIfNeeded(const audio::PcmAudioBuffer& input) {
    if (input.channels <= 2) {
        return input;
    }

    audio::PcmAudioBuffer output;
    output.sampleRate = input.sampleRate;
    output.channels = 2;
    output.samples.reserve(input.frameCount() * 2U);
    for (std::size_t frame = 0; frame < input.frameCount(); ++frame) {
        float left = 0.0F;
        float right = 0.0F;
        for (int channel = 0; channel < input.channels; ++channel) {
            const auto sampleIndex =
                (frame * static_cast<std::size_t>(input.channels)) +
                static_cast<std::size_t>(channel);
            if ((channel % 2) == 0) {
                left += input.samples[sampleIndex];
            } else {
                right += input.samples[sampleIndex];
            }
        }
        output.samples.push_back(left / static_cast<float>((input.channels + 1) / 2));
        output.samples.push_back(right / static_cast<float>(input.channels / 2));
    }
    return output;
}

[[nodiscard]] Expected<audio::PcmAudioBuffer>
preparedTakeAudio(const audio::PcmAudioBuffer& audio) {
    if (audio.empty()) {
        return makeError(ErrorCode::InvalidArgument, "Cannot save an empty take.");
    }

    auto prepared = monoIfNeeded(audio);
    if (prepared.sampleRate == kTakeSampleRate) {
        return prepared;
    }
    return audio::resamplePcm(prepared, kTakeSampleRate);
}

[[nodiscard]] int durationMs(const audio::PcmAudioBuffer& audio) {
    if (audio.sampleRate <= 0) {
        return 0;
    }
    const auto seconds = static_cast<double>(audio.frameCount()) /
                         static_cast<double>(audio.sampleRate);
    return static_cast<int>(std::lround(seconds * 1000.0));
}

} // namespace

VoiceSettings defaultVoiceSettings() noexcept {
    return VoiceSettings{};
}

std::string voiceSettingsToJson(const VoiceSettings& settings) {
    nlohmann::json json;
    json["stability"] = settings.stability;
    json["similarity_boost"] = settings.similarityBoost;
    json["style"] = settings.style;
    json["use_speaker_boost"] = settings.useSpeakerBoost;
    return json.dump();
}

Expected<VoiceSettings> voiceSettingsFromJson(const std::string& jsonText) {
    if (jsonText.empty()) {
        return defaultVoiceSettings();
    }

    try {
        const auto json = nlohmann::json::parse(jsonText);
        if (!json.is_object()) {
            return defaultVoiceSettings();
        }

        auto settings = defaultVoiceSettings();
        if (json.contains("stability") && json.at("stability").is_number()) {
            settings.stability = std::clamp(json.at("stability").get<double>(), 0.0, 1.0);
        }
        if (json.contains("similarity_boost") && json.at("similarity_boost").is_number()) {
            settings.similarityBoost =
                std::clamp(json.at("similarity_boost").get<double>(), 0.0, 1.0);
        }
        if (json.contains("style") && json.at("style").is_number()) {
            settings.style = std::clamp(json.at("style").get<double>(), 0.0, 1.0);
        }
        if (json.contains("use_speaker_boost") && json.at("use_speaker_boost").is_boolean()) {
            settings.useSpeakerBoost = json.at("use_speaker_boost").get<bool>();
        }
        return settings;
    } catch (const nlohmann::json::exception& exception) {
        return makeError(ErrorCode::InvalidArgument, exception.what());
    }
}

namespace {

Expected<SavedTake> saveVoiceTake(const db::TakeRepository& repository,
                                  const std::filesystem::path& projectRoot,
                                  const std::string& lineId,
                                  const std::string& voiceId,
                                  const std::string& rvcModelId,
                                  const audio::PcmAudioBuffer& audio,
                                  const VoiceSettings& settings,
                                  const std::string& source) {
    if (projectRoot.empty() || lineId.empty() || source.empty()) {
        return makeError(ErrorCode::InvalidArgument,
                         "Project root, line id, and take source are required.");
    }
    if (voiceId.empty() && rvcModelId.empty()) {
        return makeError(ErrorCode::InvalidArgument,
                         "Voice id or RVC model id is required.");
    }

    auto takeId = repository.createTakeId();
    if (!takeId) {
        return takeId.error();
    }

    auto prepared = preparedTakeAudio(audio);
    if (!prepared) {
        return prepared.error();
    }

    const auto safeLineId = safePathSegment(lineId);
    const auto safeTakeId = safePathSegment(takeId.value());
    const auto relativePath =
        std::filesystem::path{"takes"} / safeLineId / (safeTakeId + ".opus");
    const auto absolutePath = projectRoot / relativePath;

    auto written = audio::writeOpusFile(absolutePath, prepared.value());
    if (!written) {
        return written.error();
    }

    nlohmann::json metadata;
    metadata["voice_settings"] = nlohmann::json::parse(voiceSettingsToJson(settings));
    metadata["codec"] = "opus";
    if (source == "sts") {
        metadata["engine"] = "elevenlabs_sts";
    } else if (source == "rvc_local") {
        metadata["engine"] = "rvc_sidecar";
    } else {
        metadata["engine"] = "elevenlabs_tts";
    }

    db::NewTakeRecord record;
    record.id = takeId.value();
    record.lineId = lineId;
    record.source = source;
    record.voiceId = voiceId;
    record.rvcModelId = rvcModelId;
    record.filePath = relativePath.generic_string();
    record.durationMs = durationMs(prepared.value());
    record.starred = true;
    record.metadataJson = metadata.dump();

    auto inserted = repository.insertTake(projectRoot, record);
    if (!inserted) {
        std::error_code error;
        std::filesystem::remove(absolutePath, error);
        return inserted.error();
    }

    return SavedTake{inserted.value(), absolutePath};
}

} // namespace

Expected<SavedTake>
TakeManager::saveTtsTake(const std::filesystem::path& projectRoot,
                         const std::string& lineId,
                         const std::string& voiceId,
                         const audio::PcmAudioBuffer& audio,
                         const VoiceSettings& settings) const {
    return saveVoiceTake(m_repository, projectRoot, lineId, voiceId, {}, audio, settings, "tts");
}

Expected<SavedTake>
TakeManager::saveStsTake(const std::filesystem::path& projectRoot,
                         const std::string& lineId,
                         const std::string& voiceId,
                         const audio::PcmAudioBuffer& audio,
                         const VoiceSettings& settings) const {
    return saveVoiceTake(m_repository, projectRoot, lineId, voiceId, {}, audio, settings, "sts");
}

Expected<SavedTake>
TakeManager::saveRvcLocalTake(const std::filesystem::path& projectRoot,
                              const std::string& lineId,
                              const std::string& rvcModelId,
                              const audio::PcmAudioBuffer& audio) const {
    return saveVoiceTake(m_repository,
                         projectRoot,
                         lineId,
                         {},
                         rvcModelId,
                         audio,
                         defaultVoiceSettings(),
                         "rvc_local");
}

} // namespace voxstudio::core
