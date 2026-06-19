#include "db/Migrations.h"

#include "db/Database.h"

#include <SQLiteCpp/SQLiteCpp.h>

#include <exception>

namespace voxstudio::db {
namespace {

constexpr int kSchemaVersion = 3;

constexpr const char* kCreateMigrationTableSql = R"sql(
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
)sql";

constexpr const char* kSchemaV1Sql = R"sql(
CREATE TABLE IF NOT EXISTS project_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voices (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  origin TEXT NOT NULL,
  labels_json TEXT,
  default_settings_json TEXT,
  consent_confirmed_at TEXT,
  last_synced_at TEXT
);

CREATE TABLE IF NOT EXISTS characters (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  color TEXT,
  voice_id TEXT REFERENCES voices(id) ON DELETE SET NULL,
  rvc_model_id TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS scripts (
  id TEXT PRIMARY KEY,
  source_path TEXT,
  format TEXT NOT NULL,
  imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lines (
  id TEXT PRIMARY KEY,
  script_id TEXT NOT NULL REFERENCES scripts(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  character_id TEXT REFERENCES characters(id),
  text TEXT NOT NULL,
  scene_tag TEXT,
  voice_settings_json TEXT,
  active_take_id TEXT REFERENCES takes(id),
  UNIQUE(script_id, ord)
);

CREATE TABLE IF NOT EXISTS takes (
  id TEXT PRIMARY KEY,
  line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  voice_id TEXT REFERENCES voices(id),
  rvc_model_id TEXT,
  file_path TEXT NOT NULL,
  duration_ms INTEGER,
  lufs REAL,
  starred INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_lines_script ON lines(script_id, ord);
CREATE INDEX IF NOT EXISTS idx_takes_line ON takes(line_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
)sql";

constexpr const char* kSchemaV2Sql = R"sql(
CREATE TABLE IF NOT EXISTS sequences (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequence_items (
  sequence_id TEXT NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
  gap_ms INTEGER NOT NULL DEFAULT 250,
  PRIMARY KEY (sequence_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_sequence_items_line ON sequence_items(line_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (2);
)sql";

constexpr const char* kSchemaV3Sql = R"sql(
CREATE TABLE IF NOT EXISTS rvc_models (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  pth_path TEXT NOT NULL,
  index_path TEXT,
  sample_rate INTEGER NOT NULL,
  notes TEXT,
  imported_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (3);
)sql";

[[nodiscard]] core::Expected<int> latestMigration(SQLite::Database& connection) {
    try {
        SQLite::Statement query{connection,
                                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations;"};
        if (!query.executeStep()) {
            return 0;
        }

        return query.getColumn(0).getInt();
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

} // namespace

core::Expected<bool> hasMigration(Database& database, const int version) {
    try {
        SQLite::Statement query{database.connection(),
                                "SELECT COUNT(*) FROM schema_migrations WHERE version = ?;"};
        query.bind(1, version);

        if (!query.executeStep()) {
            return false;
        }

        return query.getColumn(0).getInt() > 0;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::DatabaseQueryFailed, exception.what());
    }
}

core::Expected<int> applyMigrations(Database& database) {
    try {
        auto& connection = database.connection();
        connection.exec(kCreateMigrationTableSql);

        auto currentVersion = latestMigration(connection);
        if (!currentVersion) {
            return currentVersion.error();
        }

        SQLite::Transaction transaction{connection};
        if (currentVersion.value() < 1) {
            connection.exec(kSchemaV1Sql);
        }
        if (currentVersion.value() < 2) {
            connection.exec(kSchemaV2Sql);
        }
        if (currentVersion.value() < 3) {
            connection.exec(kSchemaV3Sql);
        }
        transaction.commit();
        return kSchemaVersion;
    } catch (const SQLite::Exception& exception) {
        return core::makeError(core::ErrorCode::MigrationFailed, exception.what());
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::MigrationFailed, exception.what());
    }
}

} // namespace voxstudio::db
