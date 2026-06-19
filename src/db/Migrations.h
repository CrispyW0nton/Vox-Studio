#pragma once

#include "core/Expected.h"

namespace voxstudio::db {

class Database;

[[nodiscard]] core::Expected<int> applyMigrations(Database& database);
[[nodiscard]] core::Expected<bool> hasMigration(Database& database, int version);

} // namespace voxstudio::db

