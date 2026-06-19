#include "db/ProjectRepository.h"
#include "db/ScriptRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptImporter.h"
#include "ui/ScriptViewerPanel.h"

#include <QCheckBox>
#include <QColor>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
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
                 ("voxstudio_script_viewer_test_" + std::to_string(now));
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

class ScriptViewerPanelTest final : public QObject {
    Q_OBJECT

private slots:
    void loadsLinesAndHandlesKeyboardNavigation();
};

void ScriptViewerPanelTest::loadsLinesAndHandlesKeyboardNavigation() {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "Viewer.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "Viewer");
    QVERIFY(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_alice", "Alice Clone", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    auto voiceSaved = voiceRepository.upsertVoice(projectRoot, voice);
    QVERIFY(voiceSaved.hasValue());

    auto parsed = voxstudio::io::scripts::importScriptFile(fixturePath("scripts/sample.txt"));
    QVERIFY(parsed.hasValue());

    const voxstudio::db::ScriptRepository scriptRepository;
    const std::vector<voxstudio::db::CharacterAssignment> assignments{{"Alice", "voice_alice"},
                                                                      {"Bob", ""}};
    auto imported = scriptRepository.importScript(projectRoot, parsed.value(), assignments);
    QVERIFY(imported.hasValue());

    voxstudio::ui::ScriptViewerPanel panel;
    panel.setProject(project.value());
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* lineList = panel.findChild<QListWidget*>(QStringLiteral("ScriptLineList"));
    QVERIFY(lineList != nullptr);
    QCOMPARE(lineList->count(), 3);
    QCOMPARE(lineList->currentRow(), 0);

    const auto unknownColor = lineList->item(2)->foreground().color();
    QCOMPARE(unknownColor, QColor(185, 45, 45));
    QVERIFY(!lineList->item(2)->toolTip().isEmpty());

    auto* generateButton = panel.findChild<QPushButton*>(QStringLiteral("ScriptGenerateButton"));
    QVERIFY(generateButton != nullptr);
    QVERIFY(generateButton->isEnabled());

    auto* stabilitySpin = panel.findChild<QDoubleSpinBox*>(QStringLiteral("TtsStabilitySpin"));
    QVERIFY(stabilitySpin != nullptr);
    QCOMPARE(stabilitySpin->value(), 0.5);

    auto* speakerBoostCheck =
        panel.findChild<QCheckBox*>(QStringLiteral("TtsSpeakerBoostCheck"));
    QVERIFY(speakerBoostCheck != nullptr);
    QVERIFY(speakerBoostCheck->isChecked());

    auto* takeList = panel.findChild<QListWidget*>(QStringLiteral("TakeList"));
    QVERIFY(takeList != nullptr);
    QCOMPARE(takeList->count(), 0);

    lineList->setFocus();
    QTest::keyClick(lineList, Qt::Key_Down);
    QCOMPARE(lineList->currentRow(), 1);

    QTest::keyClick(lineList, Qt::Key_Enter);
    auto* statusLabel = panel.findChild<QLabel*>(QStringLiteral("ScriptStatusLabel"));
    QVERIFY(statusLabel != nullptr);
    QVERIFY(statusLabel->text().contains(QStringLiteral("Confirmed line 2 of 3.")));
}

QTEST_MAIN(ScriptViewerPanelTest)

#include "test_script_viewer_panel.moc"
