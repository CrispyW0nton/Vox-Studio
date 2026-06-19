#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"
#include "ui/TimelinePanel.h"

#include <QComboBox>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QtTest/QtTest>

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
                 ("voxstudio_timeline_panel_test_" + std::to_string(now));
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

class TimelinePanelTest final : public QObject {
    Q_OBJECT

private slots:
    void buildsSequenceAndSwitchesActiveCharacter();
};

void TimelinePanelTest::buildsSequenceAndSwitchesActiveCharacter() {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Timeline.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Timeline");
    QVERIFY(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_alice", "Alice Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    QVERIFY(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    QVERIFY(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    const std::vector<voxstudio::db::CharacterAssignment> assignments{{"Alice", "voice_alice"},
                                                                      {"Bob", ""}};
    auto imported = scriptRepository.importScript(projectRoot, parsed.value(), assignments);
    QVERIFY(imported.hasValue());

    voxstudio::ui::TimelinePanel panel;
    panel.setProject(project.value());
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* scriptSelector = panel.findChild<QComboBox*>(QStringLiteral("TimelineScriptSelector"));
    QVERIFY(scriptSelector != nullptr);
    QCOMPARE(scriptSelector->count(), 1);

    auto* sourceList = panel.findChild<QListWidget*>(QStringLiteral("TimelineSourceList"));
    QVERIFY(sourceList != nullptr);
    QCOMPARE(sourceList->count(), 3);

    auto* buildButton = panel.findChild<QPushButton*>(QStringLiteral("TimelineBuildButton"));
    QVERIFY(buildButton != nullptr);
    QTest::mouseClick(buildButton, Qt::LeftButton);

    auto* timelineList = panel.findChild<QListWidget*>(QStringLiteral("TimelineList"));
    QVERIFY(timelineList != nullptr);
    QCOMPARE(timelineList->count(), 3);

    auto* gapSpin = panel.findChild<QSpinBox*>(QStringLiteral("TimelineGapSpin"));
    QVERIFY(gapSpin != nullptr);
    gapSpin->setValue(750);
    QVERIFY(timelineList->item(0)->text().contains(QStringLiteral("750 ms")));

    panel.setFocus();
    QTest::keyClick(&panel, Qt::Key_1);
    auto* activeLabel =
        panel.findChild<QLabel*>(QStringLiteral("TimelineActiveCharacterLabel"));
    QVERIFY(activeLabel != nullptr);
    QVERIFY(activeLabel->text().contains(QStringLiteral("Alice")));
}

QTEST_MAIN(TimelinePanelTest)

#include "test_timeline_panel.moc"
