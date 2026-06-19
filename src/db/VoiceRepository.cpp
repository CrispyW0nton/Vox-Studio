#include "db/VoiceRepository.h"

#include "db/Database.h"
#include "db/Migrations.h"
#include "db/ProjectRepository.h"

#include <SQLiteCpp/SQLiteCpp.h>

#include <exception>

namespace voxstudio::db {
namespace {

[[nodiscard]] core::Expected<Database> openVoiceDatabase(const std::filesystem::path& projectRoot) {
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

void bindVoice(SQLite::Statement& statement, const VoiceRecord& voice) {
    statement.bind(1, voice.id);
    statement.bind(2, voice.name);
    statement.bind(3, voice.origin);
    statement.bind(4, voice.labelsJson);
    statement.bind(5, voice.defaultSettingsJson);
    statement.bind(6, voice.consentConfirmedAt);
    statement.bind(7, voice.lastSyncedAt);
}

[[nodiscard]] core::Expected<bool> upsertVoiceRecord(SQLite::Database& connection,
                                                     const VoiceRecord& voice) {
    if (voice.id.empty() || voice.name.empty() || voice.origin.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Voice cache entries require id, name, and origin.");
    }

    SQLite::Statement statement{
        connection,
        "INSERT INTO voices(id, name, origin, labels_json, default_settings_json, "
        "consent_confirmed_at, last_synced_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "name = excluded.name, "
        "origin = excluded.origin, "
        "labels_json = excluded.labels_json, "
        "default_settings_json = excluded.default_settings_json, "
        "consent_confirmed_at = COALESCE(NULLIF(excluded.consent_confirmed_at, ''), "
        "voices.consent_confirmed_at), "
        "last_synced_at = excluded.last_synced_at;"};
    bindVoice(statement, voice);
    statement.exec();
    return true;
}

} // namespace

core::Expected<std::vector<VoiceRecord>>
VoiceRepository::listVoices(const std::filesystem::path& projectRoot) const {
    auto database = openVoiceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement query{
            database.value().connection(),
            "SELECT id, name, origin, labels_json, default_settings_json, "
            "consent_confirmed_at, last_synced_at FROM voices ORDER BY name COLLATE NOCASE;"};

        std::vector<VoiceRecord> voices;
        while (query.executeStep()) {
            voices.push_back(VoiceRecord{query.getColumn(0).getString(),
                                         query.getColumn(1).getString(),
                                         query.getColumn(2).getString(),
                                         columnString(query.getColumn(3)),
                                         columnString(query.getColumn(4)),
                                         columnString(query.getColumn(5)),
                                         columnString(query.getColumn(6))});
        }

        return voices;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool> VoiceRepository::replaceVoices(const std::filesystem::path& projectRoot,
                                                    const std::vector<VoiceRecord>& voices) const {
    auto database = openVoiceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        for (const auto& voice : voices) {
            auto upserted = upsertVoiceRecord(connection, voice);
            if (!upserted) {
                return upserted.error();
            }
        }
        transaction.commit();
        return true;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool> VoiceRepository::upsertVoice(const std::filesystem::path& projectRoot,
                                                  const VoiceRecord& voice) const {
    auto database = openVoiceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        return upsertVoiceRecord(database.value().connection(), voice);
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool> VoiceRepository::deleteVoice(const std::filesystem::path& projectRoot,
                                                  const std::string& voiceId) const {
    if (voiceId.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument, "Voice id must not be empty.");
    }

    auto database = openVoiceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement statement{database.value().connection(),
                                    "DELETE FROM voices WHERE id = ?;"};
        statement.bind(1, voiceId);
        statement.exec();
        return true;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool>
VoiceRepository::setConsentConfirmed(const std::filesystem::path& projectRoot,
                                     const std::string& voiceId,
                                     const std::string& consentConfirmedAt) const {
    if (voiceId.empty() || consentConfirmedAt.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Voice id and consent timestamp must not be empty.");
    }

    auto database = openVoiceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement statement{
            database.value().connection(),
            "UPDATE voices SET consent_confirmed_at = ? WHERE id = ?;"};
        statement.bind(1, consentConfirmedAt);
        statement.bind(2, voiceId);
        statement.exec();
        return true;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

} // namespace voxstudio::db
