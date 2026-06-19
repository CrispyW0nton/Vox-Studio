#include "db/ScriptRepository.h"

#include "db/Database.h"
#include "db/Migrations.h"
#include "db/ProjectRepository.h"

#include <SQLiteCpp/SQLiteCpp.h>
#include <Objbase.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <utility>

namespace voxstudio::db {
namespace {

constexpr auto kDefaultCharacterColor = "#8AA4FF";
constexpr auto kDefaultVoiceSettingsJson = "{}";

[[nodiscard]] core::Expected<Database>
openScriptDatabase(const std::filesystem::path& projectRoot) {
    auto database = Database::open(projectDatabasePath(projectRoot), SQLite::OPEN_READWRITE);
    if (!database) {
        return database.error();
    }

    auto migration = applyMigrations(database.value());
    if (!migration) {
        return migration.error();
    }

    return std::move(database).value();
}

[[nodiscard]] std::string columnString(const SQLite::Column& column) {
    if (column.isNull()) {
        return {};
    }
    return column.getString();
}

[[nodiscard]] std::string utcTimestampNow() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);

    std::tm utcTime{};
    gmtime_s(&utcTime, &time);

    std::ostringstream stream;
    stream << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

[[nodiscard]] core::Expected<std::string> generateId(const std::string& prefix) {
    GUID guid{};
    if (CoCreateGuid(&guid) != S_OK) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, "Unable to create UUID.");
    }

    wchar_t buffer[39]{};
    if (StringFromGUID2(guid, buffer, static_cast<int>(std::size(buffer))) == 0) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, "Unable to format UUID.");
    }

    std::wstring wideGuid{buffer};
    std::string id;
    id.reserve(prefix.size() + wideGuid.size());
    id += prefix;
    for (const wchar_t character : wideGuid) {
        if (character != L'{' && character != L'}') {
            id.push_back(static_cast<char>(character));
        }
    }
    return id;
}

[[nodiscard]] std::map<std::string, std::string>
assignmentMap(const std::vector<CharacterAssignment>& assignments) {
    std::map<std::string, std::string> result;
    for (const auto& assignment : assignments) {
        if (!assignment.speaker.empty()) {
            result[assignment.speaker] = assignment.voiceId;
        }
    }
    return result;
}

[[nodiscard]] core::Expected<CharacterRecord>
upsertCharacter(SQLite::Database& connection, const std::string& name, const std::string& voiceId) {
    SQLite::Statement existingQuery{
        connection, "SELECT id, name, voice_id, rvc_model_id FROM characters WHERE name = ?;"};
    existingQuery.bind(1, name);
    if (existingQuery.executeStep()) {
        const CharacterRecord existing{existingQuery.getColumn(0).getString(),
                                       existingQuery.getColumn(1).getString(),
                                       columnString(existingQuery.getColumn(2)),
                                       columnString(existingQuery.getColumn(3))};
        if (!voiceId.empty() && voiceId != existing.voiceId) {
            SQLite::Statement update{
                connection, "UPDATE characters SET voice_id = ? WHERE id = ?;"};
            update.bind(1, voiceId);
            update.bind(2, existing.id);
            update.exec();
            return CharacterRecord{existing.id, existing.name, voiceId, existing.rvcModelId};
        }
        return existing;
    }

    auto characterId = generateId("character_");
    if (!characterId) {
        return characterId.error();
    }

    SQLite::Statement insert{
        connection,
        "INSERT INTO characters(id, name, color, voice_id, rvc_model_id, notes) "
        "VALUES (?, ?, ?, ?, NULL, '');"};
    insert.bind(1, characterId.value());
    insert.bind(2, name);
    insert.bind(3, kDefaultCharacterColor);
    if (voiceId.empty()) {
        insert.bind(4);
    } else {
        insert.bind(4, voiceId);
    }
    insert.exec();
    return CharacterRecord{characterId.value(), name, voiceId, {}};
}

void bindNullableString(SQLite::Statement& statement, const int index, const std::string& value) {
    if (value.empty()) {
        statement.bind(index);
    } else {
        statement.bind(index, value);
    }
}

} // namespace

core::Expected<std::vector<CharacterRecord>>
ScriptRepository::listCharacters(const std::filesystem::path& projectRoot) const {
    auto database = openScriptDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement query{
            database.value().connection(),
            "SELECT id, name, voice_id, rvc_model_id FROM characters "
            "ORDER BY name COLLATE NOCASE;"};

        std::vector<CharacterRecord> characters;
        while (query.executeStep()) {
            characters.push_back(CharacterRecord{query.getColumn(0).getString(),
                                                 query.getColumn(1).getString(),
                                                 columnString(query.getColumn(2)),
                                                 columnString(query.getColumn(3))});
        }
        return characters;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<std::vector<ScriptRecord>>
ScriptRepository::listScripts(const std::filesystem::path& projectRoot) const {
    auto database = openScriptDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement query{
            database.value().connection(),
            "SELECT id, source_path, format, imported_at FROM scripts ORDER BY imported_at DESC;"};

        std::vector<ScriptRecord> scripts;
        while (query.executeStep()) {
            scripts.push_back(ScriptRecord{query.getColumn(0).getString(),
                                           columnString(query.getColumn(1)),
                                           query.getColumn(2).getString(),
                                           query.getColumn(3).getString()});
        }
        return scripts;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<std::vector<ScriptLineRecord>>
ScriptRepository::listLines(const std::filesystem::path& projectRoot,
                            const std::string& scriptId) const {
    auto database = openScriptDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement query{
            database.value().connection(),
            "SELECT lines.id, lines.script_id, lines.ord, lines.character_id, "
            "COALESCE(characters.name, ''), COALESCE(characters.voice_id, ''), "
            "lines.text, lines.scene_tag, lines.voice_settings_json, lines.active_take_id "
            "FROM lines LEFT JOIN characters ON characters.id = lines.character_id "
            "WHERE lines.script_id = ? ORDER BY lines.ord;"};
        query.bind(1, scriptId);

        std::vector<ScriptLineRecord> lines;
        while (query.executeStep()) {
            lines.push_back(ScriptLineRecord{query.getColumn(0).getString(),
                                             query.getColumn(1).getString(),
                                             query.getColumn(2).getInt(),
                                             columnString(query.getColumn(3)),
                                             columnString(query.getColumn(4)),
                                             columnString(query.getColumn(5)),
                                             query.getColumn(6).getString(),
                                             columnString(query.getColumn(7)),
                                             columnString(query.getColumn(8)),
                                             columnString(query.getColumn(9))});
        }
        return lines;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<ImportedScriptRecord>
ScriptRepository::importScript(const std::filesystem::path& projectRoot,
                               const io::scripts::ParsedScript& script,
                               const std::vector<CharacterAssignment>& assignments) const {
    if (script.lines.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Imported script must contain at least one line.");
    }

    auto database = openScriptDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};

        auto scriptId = generateId("script_");
        if (!scriptId) {
            return scriptId.error();
        }

        const auto importedAt = utcTimestampNow();
        const ScriptRecord scriptRecord{scriptId.value(),
                                        script.sourcePath.string(),
                                        io::scripts::scriptFormatStorageName(script.format),
                                        importedAt};

        SQLite::Statement insertScript{
            connection,
            "INSERT INTO scripts(id, source_path, format, imported_at) "
            "VALUES (?, ?, ?, ?);"};
        insertScript.bind(1, scriptRecord.id);
        insertScript.bind(2, scriptRecord.sourcePath);
        insertScript.bind(3, scriptRecord.format);
        insertScript.bind(4, scriptRecord.importedAt);
        insertScript.exec();

        const auto assignedVoices = assignmentMap(assignments);
        std::map<std::string, CharacterRecord> charactersBySpeaker;
        for (const auto& line : script.lines) {
            if (line.speaker.empty() || charactersBySpeaker.contains(line.speaker)) {
                continue;
            }

            const auto voice = assignedVoices.find(line.speaker);
            const auto voiceId = voice == assignedVoices.end() ? std::string{} : voice->second;
            auto character = upsertCharacter(connection, line.speaker, voiceId);
            if (!character) {
                return character.error();
            }
            charactersBySpeaker.emplace(line.speaker, character.value());
        }

        std::vector<ScriptLineRecord> storedLines;
        storedLines.reserve(script.lines.size());
        for (std::size_t index = 0; index < script.lines.size(); ++index) {
            auto lineId = generateId("line_");
            if (!lineId) {
                return lineId.error();
            }

            std::string characterId;
            std::string characterName;
            std::string voiceId;
            if (!script.lines[index].speaker.empty()) {
                const auto character = charactersBySpeaker.find(script.lines[index].speaker);
                if (character != charactersBySpeaker.end()) {
                    characterId = character->second.id;
                    characterName = character->second.name;
                    voiceId = character->second.voiceId;
                }
            }

            SQLite::Statement insertLine{
                connection,
                "INSERT INTO lines(id, script_id, ord, character_id, text, scene_tag, "
                "voice_settings_json, active_take_id) VALUES (?, ?, ?, ?, ?, ?, ?, NULL);"};
            insertLine.bind(1, lineId.value());
            insertLine.bind(2, scriptRecord.id);
            insertLine.bind(3, static_cast<int>(index));
            bindNullableString(insertLine, 4, characterId);
            insertLine.bind(5, script.lines[index].text);
            bindNullableString(insertLine, 6, script.lines[index].sceneTag);
            insertLine.bind(7, kDefaultVoiceSettingsJson);
            insertLine.exec();

            storedLines.push_back(ScriptLineRecord{lineId.value(),
                                                   scriptRecord.id,
                                                   static_cast<int>(index),
                                                   characterId,
                                                   characterName,
                                                   voiceId,
                                                   script.lines[index].text,
                                                   script.lines[index].sceneTag,
                                                   kDefaultVoiceSettingsJson,
                                                   {}});
        }

        transaction.commit();
        return ImportedScriptRecord{scriptRecord, storedLines};
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool>
ScriptRepository::updateLineVoiceSettings(const std::filesystem::path& projectRoot,
                                          const std::string& lineId,
                                          const std::string& voiceSettingsJson) const {
    if (lineId.empty() || voiceSettingsJson.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Line id and voice settings must not be empty.");
    }

    auto database = openScriptDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement statement{
            database.value().connection(),
            "UPDATE lines SET voice_settings_json = ? WHERE id = ?;"};
        statement.bind(1, voiceSettingsJson);
        statement.bind(2, lineId);
        statement.exec();
        return true;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool>
ScriptRepository::updateCharacterRvcModel(const std::filesystem::path& projectRoot,
                                          const std::string& characterId,
                                          const std::string& rvcModelId) const {
    if (characterId.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Character id must not be empty.");
    }

    auto database = openScriptDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement statement{
            database.value().connection(),
            "UPDATE characters SET rvc_model_id = ? WHERE id = ?;"};
        bindNullableString(statement, 1, rvcModelId);
        statement.bind(2, characterId);
        statement.exec();
        return true;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

} // namespace voxstudio::db
