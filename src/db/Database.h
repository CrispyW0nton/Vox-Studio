#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <memory>

namespace SQLite {
class Database;
}

namespace voxstudio::db {

class Database final {
public:
    ~Database();

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&&) noexcept;
    Database& operator=(Database&&) noexcept;

    [[nodiscard]] static core::Expected<Database> open(const std::filesystem::path& databasePath,
                                                       int flags);

    [[nodiscard]] SQLite::Database& connection() noexcept;
    [[nodiscard]] const SQLite::Database& connection() const noexcept;

private:
    explicit Database(std::unique_ptr<SQLite::Database> database) noexcept;

    std::unique_ptr<SQLite::Database> m_database;
};

} // namespace voxstudio::db

