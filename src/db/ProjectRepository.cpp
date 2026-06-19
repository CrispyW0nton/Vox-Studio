#include "db/ProjectRepository.h"

#include "db/Database.h"
#include "db/Migrations.h"

#include <SQLiteCpp/SQLiteCpp.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace voxstudio::db {
namespace {

constexpr auto kProjectDatabaseName = "project.db";
constexpr auto kMediaDirectoryName = "media";
constexpr auto kVoxExtension = ".vox";
constexpr auto kDefaultVoiceSettingsJson = "{}";

[[nodiscard]] std::string utcTimestampNow() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);

    std::tm utcTime{};
    gmtime_s(&utcTime, &time);

    std::ostringstream stream;
    stream << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

[[nodiscard]] core::Expected<std::filesystem::path>
canonicalProjectRoot(const std::filesystem::path& projectDirectory) {
    if (projectDirectory.empty()) {
        return core::makeError(core::ErrorCode::InvalidProjectPath,
                               "Project directory must not be empty.");
    }

    if (projectDirectory.extension() != kVoxExtension) {
        return core::makeError(core::ErrorCode::InvalidProjectPath,
                               "Project directory must use the .vox extension.");
    }

    try {
        return std::filesystem::absolute(projectDirectory);
    } catch (const std::filesystem::filesystem_error& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

[[nodiscard]] core::Expected<Database>
openProjectDatabase(const std::filesystem::path& projectRoot, const int flags) {
    return Database::open(projectDatabasePath(projectRoot), flags);
}

[[nodiscard]] core::Expected<std::string> readMeta(SQLite::Database& connection,
                                                   const std::string& key) {
    try {
        SQLite::Statement query{connection, "SELECT value FROM project_meta WHERE key = ?;"};
        query.bind(1, key);

        if (!query.executeStep()) {
            return core::makeError(core::ErrorCode::DatabaseQueryFailed,
                                   "Project metadata key is missing: " + key);
        }

        return query.getColumn(0).getString();
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

[[nodiscard]] core::Expected<int> countRows(SQLite::Database& connection, const char* tableName) {
    try {
        const std::string sql = std::string{"SELECT COUNT(*) FROM "} + tableName + ";";
        SQLite::Statement query{connection, sql};

        if (!query.executeStep()) {
            return 0;
        }

        return query.getColumn(0).getInt();
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

[[nodiscard]] core::Expected<core::ProjectCounts> readCounts(SQLite::Database& connection) {
    auto characterCount = countRows(connection, "characters");
    if (!characterCount) {
        return characterCount.error();
    }

    auto voiceCount = countRows(connection, "voices");
    if (!voiceCount) {
        return voiceCount.error();
    }

    auto lineCount = countRows(connection, "lines");
    if (!lineCount) {
        return lineCount.error();
    }

    return core::ProjectCounts{characterCount.value(), voiceCount.value(), lineCount.value()};
}

void upsertMeta(SQLite::Database& connection, const std::string& key, const std::string& value) {
    SQLite::Statement statement{
        connection,
        "INSERT INTO project_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value;"};
    statement.bind(1, key);
    statement.bind(2, value);
    statement.exec();
}

[[nodiscard]] core::Expected<core::Project>
loadProjectFromDatabase(const std::filesystem::path& projectRoot, SQLite::Database& connection) {
    auto name = readMeta(connection, "name");
    if (!name) {
        return name.error();
    }

    auto createdAt = readMeta(connection, "created_at");
    if (!createdAt) {
        return createdAt.error();
    }

    auto updatedAt = readMeta(connection, "updated_at");
    if (!updatedAt) {
        return updatedAt.error();
    }

    auto counts = readCounts(connection);
    if (!counts) {
        return counts.error();
    }

    return core::Project{projectRoot, name.value(), createdAt.value(), updatedAt.value(),
                         counts.value()};
}

} // namespace

std::filesystem::path projectDatabasePath(const std::filesystem::path& projectRoot) {
    return projectRoot / kProjectDatabaseName;
}

core::Expected<core::Project>
ProjectRepository::createProject(const std::filesystem::path& projectDirectory,
                                 const std::string& name) const {
    if (name.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument, "Project name must not be empty.");
    }

    auto projectRoot = canonicalProjectRoot(projectDirectory);
    if (!projectRoot) {
        return projectRoot.error();
    }

    try {
        if (std::filesystem::exists(projectDatabasePath(projectRoot.value()))) {
            return core::makeError(core::ErrorCode::ProjectAlreadyExists,
                                   "Project database already exists.");
        }

        std::filesystem::create_directories(projectRoot.value() / kMediaDirectoryName);
    } catch (const std::filesystem::filesystem_error& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }

    auto database = openProjectDatabase(projectRoot.value(), SQLite::OPEN_READWRITE |
                                                             SQLite::OPEN_CREATE);
    if (!database) {
        return database.error();
    }

    auto migration = applyMigrations(database.value());
    if (!migration) {
        return migration.error();
    }

    try {
        const auto timestamp = utcTimestampNow();
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        upsertMeta(connection, "name", name);
        upsertMeta(connection, "created_at", timestamp);
        upsertMeta(connection, "updated_at", timestamp);
        upsertMeta(connection, "vox_schema_version", std::to_string(migration.value()));
        upsertMeta(connection, "target_engine", "");
        upsertMeta(connection, "default_voice_settings_json", kDefaultVoiceSettingsJson);
        transaction.commit();

        return loadProjectFromDatabase(projectRoot.value(), connection);
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<core::Project>
ProjectRepository::openProject(const std::filesystem::path& projectDirectory) const {
    auto projectRoot = canonicalProjectRoot(projectDirectory);
    if (!projectRoot) {
        return projectRoot.error();
    }

    if (!std::filesystem::exists(projectDatabasePath(projectRoot.value()))) {
        return core::makeError(core::ErrorCode::ProjectNotFound,
                               "Project database was not found.");
    }

    auto database = openProjectDatabase(projectRoot.value(), SQLite::OPEN_READWRITE);
    if (!database) {
        return database.error();
    }

    auto migration = applyMigrations(database.value());
    if (!migration) {
        return migration.error();
    }

    return loadProjectFromDatabase(projectRoot.value(), database.value().connection());
}

core::Expected<core::Project> ProjectRepository::saveProject(const core::Project& project) const {
    auto projectRoot = canonicalProjectRoot(project.rootPath());
    if (!projectRoot) {
        return projectRoot.error();
    }

    if (!std::filesystem::exists(projectDatabasePath(projectRoot.value()))) {
        return core::makeError(core::ErrorCode::ProjectNotFound,
                               "Project database was not found.");
    }

    auto database = openProjectDatabase(projectRoot.value(), SQLite::OPEN_READWRITE);
    if (!database) {
        return database.error();
    }

    try {
        const auto timestamp = utcTimestampNow();
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        upsertMeta(connection, "name", project.name());
        upsertMeta(connection, "updated_at", timestamp);
        transaction.commit();

        return loadProjectFromDatabase(projectRoot.value(), connection);
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

} // namespace voxstudio::db

