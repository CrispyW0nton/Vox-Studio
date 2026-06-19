#include "audio/AudioFile.h"
#include "core/SequenceRenderer.h"
#include "core/TakeManager.h"
#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/SequenceRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_sequence_renderer_test_" + std::to_string(now));
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

[[nodiscard]] voxstudio::audio::PcmAudioBuffer tone(const float frequency) {
    constexpr int kSampleRate = 48000;
    constexpr int kFrames = 2400;

    voxstudio::audio::PcmAudioBuffer audio;
    audio.sampleRate = kSampleRate;
    audio.channels = 1;
    audio.samples.reserve(kFrames);
    for (int frame = 0; frame < kFrames; ++frame) {
        const auto phase = (2.0 * 3.14159265358979323846 * frequency *
                            static_cast<double>(frame)) /
                           static_cast<double>(kSampleRate);
        audio.samples.push_back(static_cast<float>(std::sin(phase) * 0.2));
    }
    return audio;
}

} // namespace

TEST_CASE("sequence renderer concatenates active takes and exports files", "[core][sequence]") {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Render.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Render");
    REQUIRE(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord aliceVoice{
        "voice_alice", "Alice Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    const voxstudio::db::VoiceRecord bobVoice{
        "voice_bob", "Bob Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    REQUIRE(voiceRepository.upsertVoice(projectRoot, aliceVoice).hasValue());
    REQUIRE(voiceRepository.upsertVoice(projectRoot, bobVoice).hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    REQUIRE(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    const std::vector<voxstudio::db::CharacterAssignment> assignments{
        {"Alice", "voice_alice"},
        {"Bob", "voice_bob"},
    };
    auto imported = scriptRepository.importScript(projectRoot, parsed.value(), assignments);
    REQUIRE(imported.hasValue());
    REQUIRE(imported.value().lines.size() >= 2);

    voxstudio::core::TakeManager takeManager;
    REQUIRE(takeManager
                .saveTtsTake(projectRoot,
                             imported.value().lines[0].id,
                             "voice_alice",
                             tone(220.0F),
                             voxstudio::core::defaultVoiceSettings())
                .hasValue());
    REQUIRE(takeManager
                .saveTtsTake(projectRoot,
                             imported.value().lines[1].id,
                             "voice_bob",
                             tone(330.0F),
                             voxstudio::core::defaultVoiceSettings())
                .hasValue());

    const voxstudio::db::SequenceRepository sequenceRepository;
    const std::vector<voxstudio::db::NewSequenceItemRecord> items{
        {imported.value().lines[0].id, 100},
        {imported.value().lines[1].id, 0},
    };
    auto sequence = sequenceRepository.createSequence(projectRoot, "Two Lines", items);
    REQUIRE(sequence.hasValue());

    const voxstudio::core::SequenceRenderer renderer;
    auto rendered = renderer.render(projectRoot, sequence.value());
    REQUIRE(rendered.hasValue());
    CHECK(rendered.value().audio.sampleRate == 48000);
    CHECK(rendered.value().audio.channels == 2);
    CHECK(rendered.value().audio.frameCount() >= 9600);

    const auto wavPath = directory.path() / "sequence.wav";
    auto wav = renderer.exportWav(projectRoot, sequence.value(), wavPath);
    REQUIRE(wav.hasValue());
    CHECK(std::filesystem::exists(wavPath));
    CHECK(voxstudio::audio::decodeAudioFile(wavPath).hasValue());

    const auto opusPath = directory.path() / "sequence.opus";
    auto opus = renderer.exportOpus(projectRoot, sequence.value(), opusPath);
    REQUIRE(opus.hasValue());
    CHECK(std::filesystem::exists(opusPath));
    CHECK(voxstudio::audio::readOpusFile(opusPath).hasValue());
}

TEST_CASE("sequence renderer honors generation concurrency cap", "[core][sequence]") {
    voxstudio::core::Sequence sequence;
    sequence.id = "sequence_test";
    sequence.name = "Concurrency";
    for (int index = 0; index < 6; ++index) {
        sequence.lines.push_back(voxstudio::core::SequenceLine{
            index, "line_" + std::to_string(index), "Actor", "voice", "Hello", "", "", 0});
    }

    std::atomic<int> inFlight{0};
    std::atomic<int> maxInFlight{0};
    const voxstudio::core::SequenceRenderer renderer;
    auto generated = renderer.generateLineAudio(
        sequence,
        [&inFlight, &maxInFlight](const voxstudio::core::SequenceLine&) {
            const auto current = inFlight.fetch_add(1) + 1;
            int observed = maxInFlight.load();
            while (current > observed &&
                   !maxInFlight.compare_exchange_weak(observed, current)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
            inFlight.fetch_sub(1);
            return voxstudio::core::Expected<voxstudio::audio::PcmAudioBuffer>{tone(440.0F)};
        },
        voxstudio::core::SequenceGenerationOptions{2});

    REQUIRE(generated.hasValue());
    CHECK(generated.value().size() == 6);
    CHECK(maxInFlight.load() <= 2);
}
