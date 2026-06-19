#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/SequenceRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_sequence_repository_test_" + std::to_string(now));
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

[[nodiscard]] std::filesystem::path fixturePath(const std::string& relativePath) {
    return std::filesystem::path{VOXSTUDIO_TEST_FIXTURE_DIR} / relativePath;
}

} // namespace

TEST_CASE("sequence repository persists order and per-line gaps", "[db][sequence]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Sequence.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Sequence");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_alice", "Alice Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    REQUIRE(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    const std::vector<voxstudio::db::CharacterAssignment> assignments{
        {"Alice", "voice_alice"},
        {"Bob", ""},
    };
    auto imported = scriptRepository.importScript(projectRoot, parsed.value(), assignments);
    REQUIRE(imported.hasValue());
    REQUIRE(imported.value().lines.size() >= 2);

    const voxstudio::db::SequenceRepository sequenceRepository;
    const std::vector<voxstudio::db::NewSequenceItemRecord> items{
        {imported.value().lines[0].id, 100},
        {imported.value().lines[1].id, 350},
    };
    auto sequence = sequenceRepository.createSequence(projectRoot, "Opening", items);
    REQUIRE(sequence.hasValue());
    REQUIRE(sequence.value().lines.size() == 2);
    CHECK(sequence.value().lines[0].characterName == "Alice");
    CHECK(sequence.value().lines[0].voiceId == "voice_alice");
    CHECK(sequence.value().lines[0].gapMs == 100);

    const std::vector<voxstudio::db::NewSequenceItemRecord> reordered{
        {imported.value().lines[1].id, 0},
        {imported.value().lines[0].id, 500},
    };
    auto updated =
        sequenceRepository.updateSequenceItems(projectRoot, sequence.value().id, reordered);
    REQUIRE(updated.hasValue());
    REQUIRE(updated.value().lines.size() == 2);
    CHECK(updated.value().lines[0].lineId == imported.value().lines[1].id);
    CHECK(updated.value().lines[1].gapMs == 500);

    auto summaries = sequenceRepository.listSequences(projectRoot);
    REQUIRE(summaries.hasValue());
    REQUIRE(summaries.value().size() == 1);
    CHECK(summaries.value().front().name == "Opening");
}
