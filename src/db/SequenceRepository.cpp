#include "db/SequenceRepository.h"

#include "db/Database.h"
#include "db/Migrations.h"
#include "db/ProjectRepository.h"

#include <SQLiteCpp/SQLiteCpp.h>
#include <Objbase.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>

namespace voxstudio::db {
namespace {

[[nodiscard]] core::Expected<Database>
openSequenceDatabase(const std::filesystem::path& projectRoot) {
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

void insertSequenceItems(SQLite::Database& connection,
                         const std::string& sequenceId,
                         const std::vector<NewSequenceItemRecord>& items) {
    SQLite::Statement insert{
        connection,
        "INSERT INTO sequence_items(sequence_id, ord, line_id, gap_ms) "
        "VALUES (?, ?, ?, ?);"};
    for (std::size_t index = 0; index < items.size(); ++index) {
        insert.reset();
        insert.bind(1, sequenceId);
        insert.bind(2, static_cast<int>(index));
        insert.bind(3, items[index].lineId);
        insert.bind(4, std::max(0, items[index].gapMs));
        insert.exec();
    }
}

[[nodiscard]] core::Expected<core::Sequence>
loadSequenceFromConnection(SQLite::Database& connection, const std::string& sequenceId) {
    SQLite::Statement sequenceQuery{
        connection, "SELECT id, name, created_at FROM sequences WHERE id = ?;"};
    sequenceQuery.bind(1, sequenceId);
    if (!sequenceQuery.executeStep()) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, "Sequence was not found.");
    }

    core::Sequence sequence;
    sequence.id = sequenceQuery.getColumn(0).getString();
    sequence.name = sequenceQuery.getColumn(1).getString();
    sequence.createdAt = sequenceQuery.getColumn(2).getString();

    SQLite::Statement itemQuery{
        connection,
        "SELECT sequence_items.ord, sequence_items.line_id, sequence_items.gap_ms, "
        "lines.text, COALESCE(characters.name, ''), COALESCE(characters.voice_id, ''), "
        "COALESCE(lines.active_take_id, ''), COALESCE(takes.file_path, '') "
        "FROM sequence_items "
        "JOIN lines ON lines.id = sequence_items.line_id "
        "LEFT JOIN characters ON characters.id = lines.character_id "
        "LEFT JOIN takes ON takes.id = lines.active_take_id "
        "WHERE sequence_items.sequence_id = ? ORDER BY sequence_items.ord;"};
    itemQuery.bind(1, sequenceId);
    while (itemQuery.executeStep()) {
        sequence.lines.push_back(core::SequenceLine{itemQuery.getColumn(0).getInt(),
                                                    itemQuery.getColumn(1).getString(),
                                                    columnString(itemQuery.getColumn(4)),
                                                    columnString(itemQuery.getColumn(5)),
                                                    itemQuery.getColumn(3).getString(),
                                                    columnString(itemQuery.getColumn(6)),
                                                    columnString(itemQuery.getColumn(7)),
                                                    itemQuery.getColumn(2).getInt()});
    }

    return sequence;
}

[[nodiscard]] core::Expected<bool>
validateSequenceItems(const std::vector<NewSequenceItemRecord>& items) {
    if (items.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "A sequence must contain at least one line.");
    }

    for (const auto& item : items) {
        if (item.lineId.empty()) {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Sequence items require line ids.");
        }
    }
    return true;
}

} // namespace

core::Expected<std::vector<SequenceSummaryRecord>>
SequenceRepository::listSequences(const std::filesystem::path& projectRoot) const {
    auto database = openSequenceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        SQLite::Statement query{
            database.value().connection(),
            "SELECT id, name, created_at FROM sequences ORDER BY created_at DESC;"};
        std::vector<SequenceSummaryRecord> sequences;
        while (query.executeStep()) {
            sequences.push_back(SequenceSummaryRecord{query.getColumn(0).getString(),
                                                      query.getColumn(1).getString(),
                                                      query.getColumn(2).getString()});
        }
        return sequences;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<core::Sequence>
SequenceRepository::createSequence(const std::filesystem::path& projectRoot,
                                   const std::string& name,
                                   const std::vector<NewSequenceItemRecord>& items) const {
    if (name.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Sequence name must not be empty.");
    }
    auto valid = validateSequenceItems(items);
    if (!valid) {
        return valid.error();
    }

    auto database = openSequenceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    auto sequenceId = generateId("sequence_");
    if (!sequenceId) {
        return sequenceId.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        SQLite::Statement insert{
            connection, "INSERT INTO sequences(id, name, created_at) VALUES (?, ?, ?);"};
        insert.bind(1, sequenceId.value());
        insert.bind(2, name);
        insert.bind(3, utcTimestampNow());
        insert.exec();

        insertSequenceItems(connection, sequenceId.value(), items);
        auto sequence = loadSequenceFromConnection(connection, sequenceId.value());
        if (!sequence) {
            return sequence.error();
        }

        transaction.commit();
        return sequence.value();
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<core::Sequence>
SequenceRepository::loadSequence(const std::filesystem::path& projectRoot,
                                 const std::string& sequenceId) const {
    auto database = openSequenceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        return loadSequenceFromConnection(database.value().connection(), sequenceId);
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<core::Sequence>
SequenceRepository::updateSequenceItems(
    const std::filesystem::path& projectRoot,
    const std::string& sequenceId,
    const std::vector<NewSequenceItemRecord>& items) const {
    if (sequenceId.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Sequence id must not be empty.");
    }
    auto valid = validateSequenceItems(items);
    if (!valid) {
        return valid.error();
    }

    auto database = openSequenceDatabase(projectRoot);
    if (!database) {
        return database.error();
    }

    try {
        auto& connection = database.value().connection();
        SQLite::Transaction transaction{connection};
        SQLite::Statement remove{connection,
                                 "DELETE FROM sequence_items WHERE sequence_id = ?;"};
        remove.bind(1, sequenceId);
        remove.exec();
        insertSequenceItems(connection, sequenceId, items);

        auto sequence = loadSequenceFromConnection(connection, sequenceId);
        if (!sequence) {
            return sequence.error();
        }

        transaction.commit();
        return sequence.value();
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

} // namespace voxstudio::db
