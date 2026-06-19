#include "db/Database.h"
#include "db/Migrations.h"
#include "db/ProjectRepository.h"

#include <SQLiteCpp/SQLiteCpp.h>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_project_repository_test_" + std::to_string(now));
        std::filesystem::create_directories(m_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

[[nodiscard]] int scalarInt(SQLite::Database& database, const std::string& sql) {
    SQLite::Statement statement{database, sql};
    REQUIRE(statement.executeStep());
    return statement.getColumn(0).getInt();
}

} // namespace

TEST_CASE("project repository creates schema v3 projects", "[db][project]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "TestVN.vox";

    const voxstudio::db::ProjectRepository repository;
    auto project = repository.createProject(projectRoot, "TestVN");

    REQUIRE(project.hasValue());
    CHECK(project.value().name() == "TestVN");
    CHECK(project.value().rootPath() == std::filesystem::absolute(projectRoot));
    CHECK(project.value().counts().characterCount == 0);
    CHECK(project.value().counts().voiceCount == 0);
    CHECK(project.value().counts().lineCount == 0);
    CHECK(std::filesystem::exists(projectRoot / "project.db"));
    CHECK(std::filesystem::is_directory(projectRoot / "media"));

    auto database =
        voxstudio::db::Database::open(projectRoot / "project.db", SQLite::OPEN_READWRITE);
    REQUIRE(database.hasValue());

    auto hasV1 = voxstudio::db::hasMigration(database.value(), 1);
    REQUIRE(hasV1.hasValue());
    CHECK(hasV1.value());
    auto hasV2 = voxstudio::db::hasMigration(database.value(), 2);
    REQUIRE(hasV2.hasValue());
    CHECK(hasV2.value());
    auto hasV3 = voxstudio::db::hasMigration(database.value(), 3);
    REQUIRE(hasV3.hasValue());
    CHECK(hasV3.value());
    CHECK(scalarInt(database.value().connection(),
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('schema_migrations', 'project_meta', 'characters', 'voices', "
                    "'rvc_models', 'scripts', 'lines', 'takes', 'sequences', "
                    "'sequence_items');") == 10);
}

TEST_CASE("project repository round-trips metadata and counts", "[db][project]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "RoundTrip.vox";

    const voxstudio::db::ProjectRepository repository;
    auto createdProject = repository.createProject(projectRoot, "RoundTrip");
    REQUIRE(createdProject.hasValue());

    auto openedProject = repository.openProject(projectRoot);
    REQUIRE(openedProject.hasValue());
    CHECK(openedProject.value().name() == "RoundTrip");
    CHECK(openedProject.value().createdAt() == createdProject.value().createdAt());
    CHECK(openedProject.value().counts().characterCount == 0);
    CHECK(openedProject.value().counts().voiceCount == 0);
    CHECK(openedProject.value().counts().lineCount == 0);

    auto renamedProject = openedProject.value();
    renamedProject.setName("RenamedVN");

    auto savedProject = repository.saveProject(renamedProject);
    REQUIRE(savedProject.hasValue());
    CHECK(savedProject.value().name() == "RenamedVN");

    auto reopenedProject = repository.openProject(projectRoot);
    REQUIRE(reopenedProject.hasValue());
    CHECK(reopenedProject.value().name() == "RenamedVN");
}
