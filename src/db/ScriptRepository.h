#pragma once

#include "core/Expected.h"
#include "io/scripts/ScriptTypes.h"

#include <filesystem>
#include <string>
#include <vector>

namespace voxstudio::db {

struct CharacterRecord final {
    std::string id;
    std::string name;
    std::string voiceId;
    std::string rvcModelId;
};

struct CharacterAssignment final {
    std::string speaker;
    std::string voiceId;
};

struct ScriptRecord final {
    std::string id;
    std::string sourcePath;
    std::string format;
    std::string importedAt;
};

struct ScriptLineRecord final {
    std::string id;
    std::string scriptId;
    int order{0};
    std::string characterId;
    std::string characterName;
    std::string voiceId;
    std::string text;
    std::string sceneTag;
    std::string voiceSettingsJson;
    std::string activeTakeId;
};

struct ImportedScriptRecord final {
    ScriptRecord script;
    std::vector<ScriptLineRecord> lines;
};

class ScriptRepository final {
public:
    [[nodiscard]] core::Expected<std::vector<CharacterRecord>>
    listCharacters(const std::filesystem::path& projectRoot) const;

    [[nodiscard]] core::Expected<std::vector<ScriptRecord>>
    listScripts(const std::filesystem::path& projectRoot) const;

    [[nodiscard]] core::Expected<std::vector<ScriptLineRecord>>
    listLines(const std::filesystem::path& projectRoot, const std::string& scriptId) const;

    [[nodiscard]] core::Expected<ImportedScriptRecord>
    importScript(const std::filesystem::path& projectRoot,
                 const io::scripts::ParsedScript& script,
                 const std::vector<CharacterAssignment>& assignments) const;

    [[nodiscard]] core::Expected<bool>
    updateLineVoiceSettings(const std::filesystem::path& projectRoot,
                            const std::string& lineId,
                            const std::string& voiceSettingsJson) const;

    [[nodiscard]] core::Expected<bool>
    updateCharacterRvcModel(const std::filesystem::path& projectRoot,
                            const std::string& characterId,
                            const std::string& rvcModelId) const;
};

} // namespace voxstudio::db
