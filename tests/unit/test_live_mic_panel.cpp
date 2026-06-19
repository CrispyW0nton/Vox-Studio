#include "ui/LiveMicPanel.h"

#include "db/ProjectRepository.h"
#include "db/VoiceRepository.h"

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QtTest/QtTest>

#include <chrono>
#include <filesystem>
#include <string>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_live_mic_test_" + std::to_string(now));
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

class LiveMicPanelTest final : public QObject {
    Q_OBJECT

private slots:
    void exposesLiveMicControlsAndLatencyProbe();
    void enablesCloudConversionWhenProjectHasCachedVoice();
};

void LiveMicPanelTest::exposesLiveMicControlsAndLatencyProbe() {
    voxstudio::ui::LiveMicPanel panel;
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* inputCombo = panel.findChild<QComboBox*>(QStringLiteral("LiveMicInputCombo"));
    QVERIFY(inputCombo != nullptr);
    auto* outputCombo = panel.findChild<QComboBox*>(QStringLiteral("LiveMicOutputCombo"));
    QVERIFY(outputCombo != nullptr);

    auto* modeCombo = panel.findChild<QComboBox*>(QStringLiteral("LiveMicModeCombo"));
    QVERIFY(modeCombo != nullptr);
    QCOMPARE(modeCombo->currentText(), QStringLiteral("Monitor"));

    auto* voiceCombo = panel.findChild<QComboBox*>(QStringLiteral("LiveMicVoiceCombo"));
    QVERIFY(voiceCombo != nullptr);
    QVERIFY(!voiceCombo->isEnabled());

    auto* rvcModelCombo = panel.findChild<QComboBox*>(QStringLiteral("LiveMicRvcModelCombo"));
    QVERIFY(rvcModelCombo != nullptr);

    auto* monitorToggle = panel.findChild<QCheckBox*>(QStringLiteral("LiveMicMonitorToggle"));
    QVERIFY(monitorToggle != nullptr);
    QVERIFY(monitorToggle->isChecked());

    auto* powerButton = panel.findChild<QPushButton*>(QStringLiteral("LiveMicPowerButton"));
    QVERIFY(powerButton != nullptr);
    QVERIFY(!powerButton->isChecked());

    auto* hearButton = panel.findChild<QPushButton*>(QStringLiteral("LiveMicHearButton"));
    QVERIFY(hearButton != nullptr);
    QVERIFY(hearButton->isChecked());

    auto* gainSlider = panel.findChild<QSlider*>(QStringLiteral("LiveMicGainSlider"));
    QVERIFY(gainSlider != nullptr);
    QCOMPARE(gainSlider->value(), 100);

    auto* voiceVolumeSlider =
        panel.findChild<QSlider*>(QStringLiteral("LiveMicVoiceVolumeSlider"));
    QVERIFY(voiceVolumeSlider != nullptr);
    QCOMPARE(voiceVolumeSlider->value(), 100);

    auto* pitchSlider = panel.findChild<QSlider*>(QStringLiteral("LiveMicPitchSlider"));
    QVERIFY(pitchSlider != nullptr);
    QCOMPARE(pitchSlider->value(), 0);

    auto* frameSpin = panel.findChild<QSpinBox*>(QStringLiteral("LiveMicFrameMsSpin"));
    QVERIFY(frameSpin != nullptr);
    QCOMPARE(frameSpin->value(), 10);

    auto* levelMeter = panel.findChild<QProgressBar*>(QStringLiteral("LiveMicLevelMeter"));
    QVERIFY(levelMeter != nullptr);
    QCOMPARE(levelMeter->value(), 0);

    auto* latencyButton = panel.findChild<QPushButton*>(QStringLiteral("LiveMicLatencyButton"));
    QVERIFY(latencyButton != nullptr);
    QTest::mouseClick(latencyButton, Qt::LeftButton);

    auto* cloudButton = panel.findChild<QPushButton*>(QStringLiteral("LiveMicCloudButton"));
    QVERIFY(cloudButton != nullptr);
    QVERIFY(!cloudButton->isEnabled());

    auto* cancelButton =
        panel.findChild<QPushButton*>(QStringLiteral("LiveMicCancelCloudButton"));
    QVERIFY(cancelButton != nullptr);
    QVERIFY(!cancelButton->isEnabled());

    auto* localButton = panel.findChild<QPushButton*>(QStringLiteral("LiveMicLocalRvcButton"));
    QVERIFY(localButton != nullptr);

    auto* cancelLocalButton =
        panel.findChild<QPushButton*>(QStringLiteral("LiveMicCancelLocalRvcButton"));
    QVERIFY(cancelLocalButton != nullptr);
    QVERIFY(!cancelLocalButton->isEnabled());

    auto* manageRvcButton =
        panel.findChild<QPushButton*>(QStringLiteral("LiveMicManageRvcModelsButton"));
    QVERIFY(manageRvcButton != nullptr);

    auto* recordTake = panel.findChild<QCheckBox*>(QStringLiteral("LiveMicRecordTakeCheck"));
    QVERIFY(recordTake != nullptr);
    QVERIFY(!recordTake->isChecked());

    auto* lineIdEdit = panel.findChild<QLineEdit*>(QStringLiteral("LiveMicLineIdEdit"));
    QVERIFY(lineIdEdit != nullptr);

    auto* costLabel = panel.findChild<QLabel*>(QStringLiteral("LiveMicCostLabel"));
    QVERIFY(costLabel != nullptr);
    QVERIFY(costLabel->text().contains(QStringLiteral("Cloud cost")));

    auto* selectedVoice =
        panel.findChild<QLabel*>(QStringLiteral("LiveMicSelectedVoiceName"));
    QVERIFY(selectedVoice != nullptr);
    QVERIFY(selectedVoice->text().contains(QStringLiteral("No voice")));

    auto* statusLabel = panel.findChild<QLabel*>(QStringLiteral("LiveMicStatusLabel"));
    QVERIFY(statusLabel != nullptr);
    QVERIFY(statusLabel->text().contains(QStringLiteral("Estimated monitor latency")));
}

void LiveMicPanelTest::enablesCloudConversionWhenProjectHasCachedVoice() {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "LiveMic.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "LiveMic");
    QVERIFY(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_live", "Live Character", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    auto voiceSaved = voiceRepository.upsertVoice(projectRoot, voice);
    QVERIFY(voiceSaved.hasValue());

    voxstudio::ui::LiveMicPanel panel;
    panel.setProject(project.value());
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* voiceCombo = panel.findChild<QComboBox*>(QStringLiteral("LiveMicVoiceCombo"));
    QVERIFY(voiceCombo != nullptr);
    QVERIFY(voiceCombo->isEnabled());
    QCOMPARE(voiceCombo->currentData().toString(), QStringLiteral("voice_live"));

    auto* selectedVoice =
        panel.findChild<QLabel*>(QStringLiteral("LiveMicSelectedVoiceName"));
    QVERIFY(selectedVoice != nullptr);
    QCOMPARE(selectedVoice->text(), QStringLiteral("Live Character"));

    auto* quickSlot = panel.findChild<QPushButton*>(QStringLiteral("LiveMicQuickVoiceSlot1"));
    QVERIFY(quickSlot != nullptr);
    QVERIFY(quickSlot->isEnabled());

    auto* cloudButton = panel.findChild<QPushButton*>(QStringLiteral("LiveMicCloudButton"));
    QVERIFY(cloudButton != nullptr);
    QVERIFY(cloudButton->isEnabled());
}

QTEST_MAIN(LiveMicPanelTest)

#include "test_live_mic_panel.moc"
