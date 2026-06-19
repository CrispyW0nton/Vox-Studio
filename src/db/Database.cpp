#include "db/Database.h"

#include <SQLiteCpp/SQLiteCpp.h>

#include <exception>
#include <utility>

namespace voxstudio::db {

Database::Database(std::unique_ptr<SQLite::Database> database) noexcept
    : m_database(std::move(database)) {}

Database::~Database() = default;
Database::Database(Database&&) noexcept = default;
Database& Database::operator=(Database&&) noexcept = default;

core::Expected<Database> Database::open(const std::filesystem::path& databasePath,
                                        const int flags) {
    try {
        auto database = std::make_unique<SQLite::Database>(databasePath.string(), flags);
        database->exec("PRAGMA foreign_keys = ON;");
        return Database{std::move(database)};
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseOpenFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseOpenFailed, exception.what());
    }
}

SQLite::Database& Database::connection() noexcept {
    return *m_database;
}

const SQLite::Database& Database::connection() const noexcept {
    return *m_database;
}

} // namespace voxstudio::db

