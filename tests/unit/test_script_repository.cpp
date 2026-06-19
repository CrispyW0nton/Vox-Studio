#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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
                 ("voxstudio_script_repository_test_" + std::to_string(now));
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

TEST_CASE("script repository persists imported lines and speaker mappings", "[db][scripts]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Scripts.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Scripts");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord aliceVoice{
        "voice_alice", "Alice Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    auto voiceSaved = voiceRepository.upsertVoice(projectRoot, aliceVoice);
    REQUIRE(voiceSaved.hasValue());
    CHECK(voiceSaved.value());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    const std::vector<voxstudio::db::CharacterAssignment> assignments{
        {"Alice", "voice_alice"},
        {"Bob", ""},
    };
    auto imported = scriptRepository.importScript(projectRoot, parsed.value(), assignments);
    REQUIRE(imported.hasValue());
    CHECK(imported.value().script.format == "txt");
    REQUIRE(imported.value().lines.size() == 3);
    CHECK(imported.value().lines[0].characterName == "Alice");
    CHECK(imported.value().lines[1].characterName == "Bob");
    CHECK(imported.value().lines[2].characterName.empty());

    auto scripts = scriptRepository.listScripts(projectRoot);
    REQUIRE(scripts.hasValue());
    REQUIRE(scripts.value().size() == 1);

    auto lines = scriptRepository.listLines(projectRoot, scripts.value().front().id);
    REQUIRE(lines.hasValue());
    REQUIRE(lines.value().size() == 3);
    CHECK(lines.value()[0].order == 0);
    CHECK(lines.value()[0].sceneTag == "Village Gate");
    CHECK(lines.value()[2].characterId.empty());

    auto characters = scriptRepository.listCharacters(projectRoot);
    REQUIRE(characters.hasValue());
    REQUIRE(characters.value().size() == 2);
    const auto alice = std::ranges::find_if(characters.value(), [](const auto& character) {
        return character.name == "Alice";
    });
    REQUIRE(alice != characters.value().end());
    CHECK(alice->voiceId == "voice_alice");

    auto assignedRvc =
        scriptRepository.updateCharacterRvcModel(projectRoot, alice->id, "hero_rvc");
    REQUIRE(assignedRvc.hasValue());
    characters = scriptRepository.listCharacters(projectRoot);
    REQUIRE(characters.hasValue());
    const auto refreshedAlice = std::ranges::find_if(characters.value(), [](const auto& character) {
        return character.name == "Alice";
    });
    REQUIRE(refreshedAlice != characters.value().end());
    CHECK(refreshedAlice->rvcModelId == "hero_rvc");

    auto refreshedProject = projectRepository.openProject(projectRoot);
    REQUIRE(refreshedProject.hasValue());
    CHECK(refreshedProject.value().counts().characterCount == 2);
    CHECK(refreshedProject.value().counts().lineCount == 3);
    CHECK(refreshedProject.value().counts().voiceCount == 1);
}
