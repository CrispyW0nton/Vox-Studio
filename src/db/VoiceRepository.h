#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <string>
#include <vector>

namespace voxstudio::db {

struct VoiceRecord final {
    std::string id;
    std::string name;
    std::string origin;
    std::string labelsJson;
    std::string defaultSettingsJson;
    std::string consentConfirmedAt;
    std::string lastSyncedAt;
};

class VoiceRepository final {
public:
    [[nodiscard]] core::Expected<std::vector<VoiceRecord>>
    listVoices(const std::filesystem::path& projectRoot) const;

    [[nodiscard]] core::Expected<bool> replaceVoices(const std::filesystem::path& projectRoot,
                                                     const std::vector<VoiceRecord>& voices) const;

    [[nodiscard]] core::Expected<bool> upsertVoice(const std::filesystem::path& projectRoot,
                                                   const VoiceRecord& voice) const;

    [[nodiscard]] core::Expected<bool> deleteVoice(const std::filesystem::path& projectRoot,
                                                   const std::string& voiceId) const;

    [[nodiscard]] core::Expected<bool>
    setConsentConfirmed(const std::filesystem::path& projectRoot,
                        const std::string& voiceId,
                        const std::string& consentConfirmedAt) const;
};

} // namespace voxstudio::db
