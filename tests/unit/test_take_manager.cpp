#include "audio/AudioFile.h"
#include "core/TakeManager.h"
#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/TakeRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"

#include <catch2/catch_test_macros.hpp>
#include <sndfile.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
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

[[nodiscard]] bool isMp3File(const std::filesystem::path& path) {
    SF_INFO info{};
    const std::unique_ptr<SNDFILE, decltype(&sf_close)> file{
        sf_open(path.string().c_str(), SFM_READ, &info),
        &sf_close,
    };
    return file != nullptr && (info.format & SF_FORMAT_TYPEMASK) == SF_FORMAT_MPEG &&
           (info.format & SF_FORMAT_SUBMASK) == SF_FORMAT_MPEG_LAYER_III;
}

} // namespace

TEST_CASE("take manager stores MP3 takes and restores active take", "[core][takes]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Takes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Takes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_alice",          "Alice Clone",         "ivc", "{}", "{}",
        "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"};
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
    const auto saveError = savedTake ? std::string{} : savedTake.error().message;
    INFO(saveError);
    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.starred);
    CHECK(savedTake.value().take.source == "tts");
    CHECK(savedTake.value().take.filePath.ends_with(".mp3"));
    CHECK(savedTake.value().absolutePath.extension() == ".mp3");
    CHECK(savedTake.value().take.metadataJson.find("\"codec\":\"mp3\"") != std::string::npos);
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));
    CHECK(isMp3File(savedTake.value().absolutePath));

    auto playbackDecoded = voxstudio::audio::decodeAudioFile(savedTake.value().absolutePath);
    REQUIRE(playbackDecoded.hasValue());
    CHECK(playbackDecoded.value().sampleRate == 48000);
    CHECK(playbackDecoded.value().frameCount() > 0);

    const voxstudio::db::TakeRepository takeRepository;
    auto takes = takeRepository.listTakes(projectRoot, lineId);
    REQUIRE(takes.hasValue());
    REQUIRE(takes.value().size() == 1);
    CHECK(takes.value().front().id == savedTake.value().take.id);
    CHECK(takes.value().front().starred);
    CHECK(takes.value().front().characterName == "Alice");
    CHECK(takes.value().front().lineText == imported.value().lines.front().text);

    auto recentTakes = takeRepository.listRecentTakes(projectRoot);
    REQUIRE(recentTakes.hasValue());
    REQUIRE(recentTakes.value().size() == 1);
    CHECK(recentTakes.value().front().id == savedTake.value().take.id);

    auto lines = scriptRepository.listLines(projectRoot, imported.value().script.id);
    REQUIRE(lines.hasValue());
    CHECK(lines.value().front().activeTakeId == savedTake.value().take.id);

    auto deleted = takeRepository.deleteTake(projectRoot, lineId, savedTake.value().take.id);
    REQUIRE(deleted.hasValue());
    CHECK_FALSE(std::filesystem::exists(savedTake.value().absolutePath));
}

TEST_CASE("take manager exports selected takes as collision-safe MP3 files",
          "[core][takes][export]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "ExportTakes.vox";
    const auto exportFolder = directory.path() / "Exports";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "ExportTakes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_atton", "Atton", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    REQUIRE(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    auto line = scriptRepository.createPerformanceLine(
        projectRoot, "Atton", "voice_atton", "Pure pazaak. The table is ours.");
    REQUIRE(line.hasValue());

    voxstudio::core::TakeManager manager;
    REQUIRE(manager
                .saveVoxCpmTextTake(projectRoot, line.value().id, "voice_atton",
                                    sinePcm(), "sarcastic", "storytelling")
                .hasValue());
    REQUIRE(manager
                .saveVoxCpmTextTake(projectRoot, line.value().id, "voice_atton",
                                    sinePcm(), "natural", "standard")
                .hasValue());

    const voxstudio::db::TakeRepository takeRepository;
    auto takes = takeRepository.listTakes(projectRoot, line.value().id);
    REQUIRE(takes.hasValue());
    REQUIRE(takes.value().size() == 2);

    auto exported = manager.exportTakesAsMp3(projectRoot, exportFolder, takes.value());
    REQUIRE(exported.hasValue());
    REQUIRE(exported.value().size() == 2);
    for (std::size_t index = 0; index < exported.value().size(); ++index) {
        CAPTURE(exported.value()[index]);
        CHECK(exported.value()[index].extension() == ".mp3");
        CHECK(exported.value()[index].filename().string().starts_with("Atton - Pure pazaak"));
        CHECK(isMp3File(exported.value()[index]));
        CHECK(std::filesystem::file_size(exported.value()[index]) ==
              std::filesystem::file_size(projectRoot / takes.value()[index].filePath));
    }

    auto exportedAgain = manager.exportTakesAsMp3(projectRoot, exportFolder, takes.value());
    REQUIRE(exportedAgain.hasValue());
    REQUIRE(exportedAgain.value().size() == 2);
    CHECK(exportedAgain.value().front() != exported.value().front());
    CHECK(exportedAgain.value().front().stem().string().ends_with("(2)"));
}

TEST_CASE("take manager converts legacy takes and rolls back failed batch exports",
          "[core][takes][export]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "LegacyExport.vox";
    const auto takeFolder = projectRoot / "takes" / "legacy";
    const auto exportFolder = directory.path() / "LegacyExports";
    std::filesystem::create_directories(takeFolder);

    const auto legacyPath = takeFolder / "old-take.opus";
    REQUIRE(voxstudio::audio::writeOpusFile(legacyPath, sinePcm()).hasValue());

    voxstudio::db::TakeRecord legacyTake;
    legacyTake.id = "old-take";
    legacyTake.filePath = "takes/legacy/old-take.opus";
    legacyTake.characterName = "Kreia";
    legacyTake.lineText = "A lesson from the past.";

    voxstudio::core::TakeManager manager;
    const std::vector legacyTakes{legacyTake};
    auto exported = manager.exportTakesAsMp3(projectRoot, exportFolder, legacyTakes);
    REQUIRE(exported.hasValue());
    REQUIRE(exported.value().size() == 1);
    CHECK(isMp3File(exported.value().front()));

    auto invalidTake = legacyTake;
    invalidTake.id = "unsafe";
    invalidTake.filePath = "../outside.mp3";
    const std::vector mixedTakes{legacyTake, invalidTake};
    const auto rollbackFolder = directory.path() / "RolledBack";
    auto failed = manager.exportTakesAsMp3(projectRoot, rollbackFolder, mixedTakes);
    REQUIRE_FALSE(failed.hasValue());
    REQUIRE(std::filesystem::exists(rollbackFolder));
    CHECK(std::filesystem::is_empty(rollbackFolder));
}

TEST_CASE("take manager stores STS takes as active MP3 takes", "[core][takes][sts]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "StsTakes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "StsTakes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_bob",           "Bob Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
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
    auto savedTake = manager.saveStsTake(projectRoot, lineId, "voice_bob", sinePcm(),
                                         voxstudio::core::defaultVoiceSettings());
    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.starred);
    CHECK(savedTake.value().take.source == "sts");
    CHECK(savedTake.value().absolutePath.extension() == ".mp3");
    CHECK(savedTake.value().take.metadataJson.find("\"codec\":\"mp3\"") != std::string::npos);
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));

    const voxstudio::db::TakeRepository takeRepository;
    auto takes = takeRepository.listTakes(projectRoot, lineId);
    REQUIRE(takes.hasValue());
    REQUIRE(takes.value().size() == 1);
    CHECK(takes.value().front().source == "sts");
}

TEST_CASE("take manager labels VoxCPM2 performance takes", "[core][takes][voxcpm]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "VoxCpmTakes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "VoxCpmTakes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_carth", "Carth", "ivc", "{}", "{}", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"};
    REQUIRE(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());
    const voxstudio::db::ScriptRepository scriptRepository;
    auto imported =
        scriptRepository.importScript(projectRoot, parsed.value(), {{"Alice", "voice_carth"}});
    REQUIRE(imported.hasValue());

    voxstudio::core::TakeManager manager;
    auto savedTake = manager.saveVoxCpmTake(projectRoot, imported.value().lines.front().id,
                                            "voice_carth", sinePcm());

    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.source == "voxcpm2");
    CHECK(savedTake.value().take.metadataJson.find("\"engine\":\"voxcpm2\"") != std::string::npos);
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));
}

TEST_CASE("take manager stores VoxCPM2 text delivery tags", "[core][takes][voxcpm][tts]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "VoxCpmTextTakes.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "VoxCpmTextTakes");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_carth", "Carth", "ivc", "{}", "{}", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"};
    REQUIRE(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    auto line = scriptRepository.createPerformanceLine(projectRoot, "Carth", "voice_carth",
                                                       "The Rodian remembered Telos.");
    REQUIRE(line.hasValue());

    voxstudio::core::TakeManager manager;
    auto savedTake = manager.saveVoxCpmTextTake(projectRoot, line.value().id, "voice_carth",
                                                sinePcm(), "reflective", "storytelling");

    REQUIRE(savedTake.hasValue());
    CHECK(savedTake.value().take.source == "voxcpm2_tts");
    CHECK(savedTake.value().take.metadataJson.find("\"delivery\":\"reflective\"") !=
          std::string::npos);
    CHECK(savedTake.value().take.metadataJson.find("\"performance_mode\":\"storytelling\"") !=
          std::string::npos);
    CHECK(std::filesystem::exists(savedTake.value().absolutePath));
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

    auto parsed =
        voxstudio::core::voiceSettingsFromJson(voxstudio::core::voiceSettingsToJson(settings));
    REQUIRE(parsed.hasValue());
    CHECK(parsed.value().stability == 0.2);
    CHECK(parsed.value().similarityBoost == 0.9);
    CHECK(parsed.value().style == 0.4);
    CHECK_FALSE(parsed.value().useSpeakerBoost);

    parsed = voxstudio::core::voiceSettingsFromJson("{}");
    REQUIRE(parsed.hasValue());
    CHECK(parsed.value().useSpeakerBoost);
}
