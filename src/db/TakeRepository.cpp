#include "db/TakeRepository.h"

#include "db/Database.h"
#include "db/Migrations.h"
#include "db/ProjectRepository.h"

#include <SQLiteCpp/SQLiteCpp.h>
#include <Objbase.h>

#include <array>
#include <chrono>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace voxstudio::db {
namespace {

[[nodiscard]] core::Expected<Database> openTakeDatabase(const std::filesystem::path& projectRoot) {
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

    std::array<wchar_t, 39> buffer{};
    if (StringFromGUID2(guid, buffer.data(), static_cast<int>(buffer.size())) == 0) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, "Unable to format UUID.");
    }

    std::wstring wideGuid{buffer.data()};
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

void bindNullableString(SQLite::Statement& statement, const int index, const std::string& value) {
    if (value.empty()) {
        statement.bind(index);
    } else {
        statement.bind(index, value);
    }
}

[[nodiscard]] TakeRecord takeFromQuery(SQLite::Statement& query) {
    return TakeRecord{query.getColumn(0).getString(),
                      query.getColumn(1).getString(),
                      query.getColumn(2).getString(),
                      columnString(query.getColumn(3)),
                      columnString(query.getColumn(4)),
                      query.getColumn(5).getString(),
                      query.getColumn(6).getInt(),
                      query.getColumn(7).isNull() ? 0.0 : query.getColumn(7).getDouble(),
                      query.getColumn(8).getInt() != 0,
                      query.getColumn(9).getString(),
                      columnString(query.getColumn(10))};
}

[[nodiscard]] core::Expected<TakeRecord>
loadTake(SQLite::Database& connection, const std::string& lineId, const std::string& takeId) {
    SQLite::Statement query{
        connection,
        "SELECT id, line_id, source, voice_id, rvc_model_id, file_path, duration_ms, "
        "lufs, starred, created_at, metadata_json FROM takes WHERE line_id = ? AND id = ?;"};
    query.bind(1, lineId);
    query.bind(2, takeId);
    if (!query.executeStep()) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, "Take was not found.");
    }
    return takeFromQuery(query);
}

[[nodiscard]] bool isRelativeSafePath(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) {
        return false;
    }

    for (const auto& part : path) {
        if (part == "..") {
            return false;
        }
    }
    return true;
}

} // namespace

core::Expected<std::string> TakeRepository::createTakeId() const {
    return generateId("take_");
}

core::Expected<std::vector<TakeRecord>>
TakeRepository::listTakes(const std::filesystem::path& projectRoot,
                          const std::string& lineId) const {
    auto database = openTakeDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement query{
            database.value().connection(),
            "SELECT id, line_id, source, voice_id, rvc_model_id, file_path, duration_ms, "
            "lufs, starred, created_at, metadata_json FROM takes WHERE line_id = ? "
            "ORDER BY starred DESC, created_at DESC;"};
        query.bind(1, lineId);

        std::vector<TakeRecord> takes;
        while (query.executeStep()) {
            takes.push_back(takeFromQuery(query));
        }
        return takes;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<TakeRecord>
TakeRepository::insertTake(const std::filesystem::path& projectRoot,
                           const NewTakeRecord& take) const {
    if (take.id.empty() || take.lineId.empty() || take.source.empty() || take.filePath.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Take requires id, line id, source, and file path.");
    }

    auto database = openTakeDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};

        if (take.starred) {
            SQLite::Statement clearStarred{connection,
                                           "UPDATE takes SET starred = 0 WHERE line_id = ?;"};
            clearStarred.bind(1, take.lineId);
            clearStarred.exec();
        }

        SQLite::Statement insert{
            connection,
            "INSERT INTO takes(id, line_id, source, voice_id, rvc_model_id, file_path, "
            "duration_ms, lufs, starred, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"};
        insert.bind(1, take.id);
        insert.bind(2, take.lineId);
        insert.bind(3, take.source);
        bindNullableString(insert, 4, take.voiceId);
        bindNullableString(insert, 5, take.rvcModelId);
        insert.bind(6, take.filePath);
        insert.bind(7, take.durationMs);
        insert.bind(8, take.lufs);
        insert.bind(9, take.starred ? 1 : 0);
        insert.bind(10, utcTimestampNow());
        bindNullableString(insert, 11, take.metadataJson);
        insert.exec();

        if (take.starred) {
            SQLite::Statement updateLine{connection,
                                         "UPDATE lines SET active_take_id = ? WHERE id = ?;"};
            updateLine.bind(1, take.id);
            updateLine.bind(2, take.lineId);
            updateLine.exec();
        }

        auto stored = loadTake(connection, take.lineId, take.id);
        if (!stored) {
            return stored.error();
        }

        transaction.commit();
        return stored.value();
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<TakeRecord>
TakeRepository::setActiveTake(const std::filesystem::path& projectRoot,
                              const std::string& lineId,
                              const std::string& takeId) const {
    if (lineId.empty() || takeId.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Line id and take id must not be empty.");
    }

    auto database = openTakeDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        auto take = loadTake(connection, lineId, takeId);
        if (!take) {
            return take.error();
        }

        SQLite::Statement clearStarred{
            connection, "UPDATE takes SET starred = 0 WHERE line_id = ?;"};
        clearStarred.bind(1, lineId);
        clearStarred.exec();

        SQLite::Statement star{connection, "UPDATE takes SET starred = 1 WHERE id = ?;"};
        star.bind(1, takeId);
        star.exec();

        SQLite::Statement updateLine{
            connection, "UPDATE lines SET active_take_id = ? WHERE id = ?;"};
        updateLine.bind(1, takeId);
        updateLine.bind(2, lineId);
        updateLine.exec();

        transaction.commit();
        return loadTake(connection, lineId, takeId);
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<bool>
TakeRepository::deleteTake(const std::filesystem::path& projectRoot,
                           const std::string& lineId,
                           const std::string& takeId) const {
    auto database = openTakeDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        auto take = loadTake(connection, lineId, takeId);
        if (!take) {
            return take.error();
        }

        SQLite::Statement clearLine{
            connection,
            "UPDATE lines SET active_take_id = NULL WHERE id = ? AND active_take_id = ?;"};
        clearLine.bind(1, lineId);
        clearLine.bind(2, takeId);
        clearLine.exec();

        SQLite::Statement remove{connection, "DELETE FROM takes WHERE id = ? AND line_id = ?;"};
        remove.bind(1, takeId);
        remove.bind(2, lineId);
        remove.exec();
        transaction.commit();

        const std::filesystem::path relativePath{take.value().filePath};
        if (isRelativeSafePath(relativePath)) {
            const auto absolutePath = projectRoot / relativePath;
            std::error_code error;
            std::filesystem::remove(absolutePath, error);
        }
        return true;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

} // namespace voxstudio::db
