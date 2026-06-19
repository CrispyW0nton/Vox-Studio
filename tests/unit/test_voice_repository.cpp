#include "db/Database.h"
#include "db/ProjectRepository.h"
#include "db/VoiceRepository.h"

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
        m_path =
            std::filesystem::temp_directory_path() / ("voxstudio_voice_repository_test_" +
                                                      std::to_string(now));
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

} // namespace

TEST_CASE("voice repository mirrors voices and preserves consent", "[db][voices]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Voices.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Voices");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const std::vector<voxstudio::db::VoiceRecord> firstSync{
        {"voice_001", "Narrator", "premade", R"({"accent":"neutral"})", "{}",
         "", "2026-01-01T00:00:00Z"},
        {"voice_002", "Clone Hero", "ivc", R"({"age":"adult"})", "{}", "2026-01-01T00:00:01Z",
         "2026-01-01T00:00:02Z"}};

    auto replaced = voiceRepository.replaceVoices(projectRoot, firstSync);
    REQUIRE(replaced.hasValue());
    CHECK(replaced.value());

    auto listed = voiceRepository.listVoices(projectRoot);
    REQUIRE(listed.hasValue());
    REQUIRE(listed.value().size() == 2);
    const auto firstClone = std::ranges::find_if(listed.value(), [](const auto& voice) {
        return voice.id == "voice_002";
    });
    REQUIRE(firstClone != listed.value().end());
    CHECK(firstClone->consentConfirmedAt == "2026-01-01T00:00:01Z");

    const std::vector<voxstudio::db::VoiceRecord> secondSync{
        {"voice_002", "Clone Hero Renamed", "ivc", R"({"age":"adult"})", "{}", "",
         "2026-01-02T00:00:00Z"}};
    replaced = voiceRepository.replaceVoices(projectRoot, secondSync);
    REQUIRE(replaced.hasValue());

    listed = voiceRepository.listVoices(projectRoot);
    REQUIRE(listed.hasValue());
    REQUIRE(listed.value().size() == 2);
    const auto secondClone = std::ranges::find_if(listed.value(), [](const auto& voice) {
        return voice.id == "voice_002";
    });
    REQUIRE(secondClone != listed.value().end());
    CHECK(secondClone->name == "Clone Hero Renamed");
    CHECK(secondClone->consentConfirmedAt == "2026-01-01T00:00:01Z");

    auto refreshedProject = projectRepository.openProject(projectRoot);
    REQUIRE(refreshedProject.hasValue());
    CHECK(refreshedProject.value().counts().voiceCount == 2);

    auto deleted = voiceRepository.deleteVoice(projectRoot, "voice_001");
    REQUIRE(deleted.hasValue());
    listed = voiceRepository.listVoices(projectRoot);
    REQUIRE(listed.hasValue());
    REQUIRE(listed.value().size() == 1);
    CHECK(listed.value().front().id == "voice_002");
}
