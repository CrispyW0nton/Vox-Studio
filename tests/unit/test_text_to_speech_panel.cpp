#include "ui/TextToSpeechPanel.h"

#include "db/ProjectRepository.h"
#include "db/VoiceRepository.h"

#include <QComboBox>
#include <QPlainTextEdit>
#include <QPushButton>
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
                 ("voxstudio_tts_panel_test_" + std::to_string(now));
        std::filesystem::create_directories(m_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

} // namespace

class TextToSpeechPanelTest final : public QObject {
    Q_OBJECT

private slots:
    void exposesLongFormEditorAndExclusiveDeliveryTags();
    void enablesGenerationForProjectVoice();
};

void TextToSpeechPanelTest::exposesLongFormEditorAndExclusiveDeliveryTags() {
    voxstudio::ui::TextToSpeechPanel panel;
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* editor =
        panel.findChild<QPlainTextEdit*>(QStringLiteral("TextToSpeechEditor"));
    QVERIFY(editor != nullptr);
    QVERIFY(editor->minimumHeight() >= 250);

    auto* natural = panel.findChild<QPushButton*>(
        QStringLiteral("TextToSpeechDelivery_natural"));
    auto* reflective = panel.findChild<QPushButton*>(
        QStringLiteral("TextToSpeechDelivery_reflective"));
    auto* sarcastic = panel.findChild<QPushButton*>(
        QStringLiteral("TextToSpeechDelivery_sarcastic"));
    QVERIFY(natural != nullptr);
    QVERIFY(reflective != nullptr);
    QVERIFY(sarcastic != nullptr);
    QVERIFY(natural->isChecked());
    reflective->click();
    QVERIFY(reflective->isChecked());
    QVERIFY(!natural->isChecked());

    auto* output =
        panel.findChild<QComboBox*>(QStringLiteral("TextToSpeechOutputCombo"));
    QVERIFY(output != nullptr);
    auto* stop =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechStopButton"));
    QVERIFY(stop != nullptr);
    QVERIFY(!stop->toolTip().isEmpty());
}

void TextToSpeechPanelTest::enablesGenerationForProjectVoice() {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "TextToSpeech.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "TextToSpeech");
    QVERIFY(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_carth", "Carth", "ivc", "{}", "{}", "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z"};
    QVERIFY(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    voxstudio::ui::TextToSpeechPanel panel;
    panel.setProject(project.value());
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* voiceCombo =
        panel.findChild<QComboBox*>(QStringLiteral("TextToSpeechVoiceCombo"));
    QVERIFY(voiceCombo != nullptr);
    QCOMPARE(voiceCombo->currentData().toString(), QStringLiteral("voice_carth"));

    auto* generate = panel.findChild<QPushButton*>(
        QStringLiteral("TextToSpeechGenerateButton"));
    QVERIFY(generate != nullptr);
    QVERIFY(generate->isEnabled());
}

QTEST_MAIN(TextToSpeechPanelTest)

#include "test_text_to_speech_panel.moc"
