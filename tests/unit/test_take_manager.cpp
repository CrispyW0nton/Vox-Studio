#include "audio/AudioFile.h"
#include "core/TakeManager.h"
#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/TakeRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_take_manager_test_" + std::to_string(now));
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

[[nodiscard]] voxstudio::audio::PcmAudioBuffer sinePcm() {
    voxstudio::audio::PcmAudioBuffer audio;
    audio.sampleRate = 24000;
    audio.channels = 1;
    constexpr int kFrames = 2400;
    audio.samples.reserve(kFrames);
    for (int frame = 0; frame < kFrames; ++frame) {
        const auto radians = (static_cast<double>(frame) / 24000.0) * 440.0 * 6.283185307179586;
        audio.samples.push_back(static_cast<float>(std::sin(radians) * 0.2));
    }
    return audio;
}

} // namespace

TEST_CASE("take manager stores Opus takes and restores active take", "[core][takes]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Takes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Takes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_alice", "Alice Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    auto voiceSaved = voiceRepository.upsertVoice(projectRoot, voice);
    REQUIRE(voiceSaved.hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    auto imported =
        scriptRepository.importScript(projectRoot, parsed.value(), {{"Alice", "voice_alice"}});
    REQUIRE(imported.hasValue());
    const auto lineId = imported.value().lines.front().id;

    voxstudio::core::TakeManager manager;
    voxstudio::core::VoiceSettings settings;
    settings.stability = 0.25;
    auto savedTake = manager.saveTtsTake(projectRoot, lineId, "voice_alice", sinePcm(), settings);
    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.starred);
    CHECK(savedTake.value().take.source == "tts");
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));

    auto decoded = voxstudio::audio::readOpusFile(savedTake.value().absolutePath);
    REQUIRE(decoded.hasValue());
    CHECK(decoded.value().sampleRate == 48000);
    CHECK(decoded.value().frameCount() > 0);

    const voxstudio::db::TakeRepository takeRepository;
    auto takes = takeRepository.listTakes(projectRoot, lineId);
    REQUIRE(takes.hasValue());
    REQUIRE(takes.value().size() == 1);
    CHECK(takes.value().front().id == savedTake.value().take.id);
    CHECK(takes.value().front().starred);

    auto lines = scriptRepository.listLines(projectRoot, imported.value().script.id);
    REQUIRE(lines.hasValue());
    CHECK(lines.value().front().activeTakeId == savedTake.value().take.id);

    auto deleted = takeRepository.deleteTake(projectRoot, lineId, savedTake.value().take.id);
    REQUIRE(deleted.hasValue());
    CHECK_FALSE(std::filesystem::exists(savedTake.value().absolutePath));
}

TEST_CASE("take manager stores STS takes as active Opus takes", "[core][takes][sts]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "StsTakes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "StsTakes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_bob", "Bob Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    auto voiceSaved = voiceRepository.upsertVoice(projectRoot, voice);
    REQUIRE(voiceSaved.hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    auto imported =
        scriptRepository.importScript(projectRoot, parsed.value(), {{"Alice", "voice_bob"}});
    REQUIRE(imported.hasValue());
    const auto lineId = imported.value().lines.front().id;

    voxstudio::core::TakeManager manager;
    auto savedTake = manager.saveStsTake(
        projectRoot, lineId, "voice_bob", sinePcm(), voxstudio::core::defaultVoiceSettings());
    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.starred);
    CHECK(savedTake.value().take.source == "sts");
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));

    const voxstudio::db::TakeRepository takeRepository;
    auto takes = takeRepository.listTakes(projectRoot, lineId);
    REQUIRE(takes.hasValue());
    REQUIRE(takes.value().size() == 1);
    CHECK(takes.value().front().source == "sts");
}

TEST_CASE("take manager stores local RVC takes with model id", "[core][takes][rvc]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "RvcTakes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "RvcTakes");
    REQUIRE(project.hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    auto imported = scriptRepository.importScript(projectRoot, parsed.value(), {});
    REQUIRE(imported.hasValue());
    const auto lineId = imported.value().lines.front().id;

    voxstudio::core::TakeManager manager;
    auto savedTake = manager.saveRvcLocalTake(projectRoot, lineId, "hero_rvc", sinePcm());
    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.starred);
    CHECK(savedTake.value().take.source == "rvc_local");
    CHECK(savedTake.value().take.rvcModelId == "hero_rvc");
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));

    const voxstudio::db::TakeRepository takeRepository;
    auto takes = takeRepository.listTakes(projectRoot, lineId);
    REQUIRE(takes.hasValue());
    REQUIRE(takes.value().size() == 1);
    CHECK(takes.value().front().source == "rvc_local");
    CHECK(takes.value().front().rvcModelId == "hero_rvc");
}

TEST_CASE("voice settings JSON round trips with defaults", "[core][takes]") {
    voxstudio::core::VoiceSettings settings;
    settings.stability = 0.2;
    settings.similarityBoost = 0.9;
    settings.style = 0.4;
    settings.useSpeakerBoost = false;

    auto parsed = voxstudio::core::voiceSettingsFromJson(
        voxstudio::core::voiceSettingsToJson(settings));
    REQUIRE(parsed.hasValue());
    CHECK(parsed.value().stability == 0.2);
    CHECK(parsed.value().similarityBoost == 0.9);
    CHECK(parsed.value().style == 0.4);
    CHECK_FALSE(parsed.value().useSpeakerBoost);

    parsed = voxstudio::core::voiceSettingsFromJson("{}");
    REQUIRE(parsed.hasValue());
    CHECK(parsed.value().useSpeakerBoost);
}
