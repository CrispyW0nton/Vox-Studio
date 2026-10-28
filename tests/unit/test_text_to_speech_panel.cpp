#include "ui/TextToSpeechPanel.h"

#include "db/ProjectRepository.h"
#include "db/VoiceRepository.h"
#include "ui/TakeListWidget.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QtTest/QtTest>

#include <chrono>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

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
    void exposesStorytellingModeAndDirectionPreview();
    void invalidatesDirectionPreviewWhenHiddenInputsChange();
    void enablesGenerationForProjectVoice();
    void supportsSelectingAndExportingMultipleGeneratedTakes();
};

void TextToSpeechPanelTest::exposesLongFormEditorAndExclusiveDeliveryTags() {
    voxstudio::ui::TextToSpeechPanel panel;
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* editor = panel.findChild<QPlainTextEdit*>(QStringLiteral("TextToSpeechEditor"));
    QVERIFY(editor != nullptr);
    QVERIFY(editor->minimumHeight() >= 250);
    QVERIFY(editor->placeholderText().contains(QStringLiteral("*sighs*")));

    auto* natural = panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechDelivery_natural"));
    auto* reflective =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechDelivery_reflective"));
    auto* sarcastic =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechDelivery_sarcastic"));
    QVERIFY(natural != nullptr);
    QVERIFY(reflective != nullptr);
    QVERIFY(sarcastic != nullptr);
    QVERIFY(natural->isChecked());
    reflective->click();
    QVERIFY(reflective->isChecked());
    QVERIFY(!natural->isChecked());

    auto* output = panel.findChild<QComboBox*>(QStringLiteral("TextToSpeechOutputCombo"));
    QVERIFY(output != nullptr);
    auto* stop = panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechStopButton"));
    QVERIFY(stop != nullptr);
    QVERIFY(!stop->toolTip().isEmpty());
}

void TextToSpeechPanelTest::exposesStorytellingModeAndDirectionPreview() {
    voxstudio::ui::TextToSpeechPanel panel;
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* standard = panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechMode_standard"));
    auto* storytelling =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechMode_storytelling"));
    auto* previewButton =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechPreviewDirectionButton"));
    auto* preview =
        panel.findChild<QPlainTextEdit*>(QStringLiteral("TextToSpeechDirectionPreview"));
    QVERIFY(standard != nullptr);
    QVERIFY(storytelling != nullptr);
    QVERIFY(previewButton != nullptr);
    QVERIFY(preview != nullptr);
    QVERIFY(standard->isChecked());
    QVERIFY(!preview->isVisible());

    storytelling->click();

    QVERIFY(storytelling->isChecked());
    QVERIFY(!standard->isChecked());
    QVERIFY(preview->isVisible());
    QVERIFY(previewButton->isVisible());
    QVERIFY(preview->isReadOnly());
}

void TextToSpeechPanelTest::invalidatesDirectionPreviewWhenHiddenInputsChange() {
    voxstudio::ui::TextToSpeechPanel panel;
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* standard = panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechMode_standard"));
    auto* storytelling =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechMode_storytelling"));
    auto* editor = panel.findChild<QPlainTextEdit*>(QStringLiteral("TextToSpeechEditor"));
    auto* preview =
        panel.findChild<QPlainTextEdit*>(QStringLiteral("TextToSpeechDirectionPreview"));
    auto* reflective =
        panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechDelivery_reflective"));
    QVERIFY(standard != nullptr);
    QVERIFY(storytelling != nullptr);
    QVERIFY(editor != nullptr);
    QVERIFY(preview != nullptr);
    QVERIFY(reflective != nullptr);

    storytelling->click();
    preview->setPlainText(QStringLiteral("Old direction"));
    standard->click();
    editor->setPlainText(QStringLiteral("A changed story."));
    QVERIFY(preview->toPlainText().isEmpty());

    storytelling->click();
    preview->setPlainText(QStringLiteral("Another old direction"));
    standard->click();
    reflective->click();
    QVERIFY(preview->toPlainText().isEmpty());
}

void TextToSpeechPanelTest::enablesGenerationForProjectVoice() {
    const TemporaryDirectory directory;
    const auto projectRoot = directory.path() / "TextToSpeech.vox";

    const voxstudio::db::ProjectRepository projectRepository;
    auto project = projectRepository.createProject(projectRoot, "TextToSpeech");
    QVERIFY(project.hasValue());

    const voxstudio::db::VoiceRepository voiceRepository;
    const voxstudio::db::VoiceRecord voice{
        "voice_carth", "Carth", "ivc", "{}", "{}", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"};
    QVERIFY(voiceRepository.upsertVoice(projectRoot, voice).hasValue());

    voxstudio::ui::TextToSpeechPanel panel;
    panel.setProject(project.value());
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* voiceCombo = panel.findChild<QComboBox*>(QStringLiteral("TextToSpeechVoiceCombo"));
    QVERIFY(voiceCombo != nullptr);
    QCOMPARE(voiceCombo->currentData().toString(), QStringLiteral("voice_carth"));

    auto* generate = panel.findChild<QPushButton*>(QStringLiteral("TextToSpeechGenerateButton"));
    QVERIFY(generate != nullptr);
    QVERIFY(generate->isEnabled());
}

void TextToSpeechPanelTest::supportsSelectingAndExportingMultipleGeneratedTakes() {
    voxstudio::ui::TextToSpeechPanel panel;
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));

    auto* takesWidget = panel.findChild<voxstudio::ui::TakeListWidget*>(
        QStringLiteral("TextToSpeechTakes"));
    auto* list = panel.findChild<QListWidget*>(QStringLiteral("TakeList"));
    auto* selectAll =
        panel.findChild<QPushButton*>(QStringLiteral("TakeSelectAllButton"));
    auto* exportMp3 =
        panel.findChild<QPushButton*>(QStringLiteral("TakeExportMp3Button"));
    auto* play = panel.findChild<QPushButton*>(QStringLiteral("TakePlayButton"));
    QVERIFY(takesWidget != nullptr);
    QVERIFY(list != nullptr);
    QVERIFY(selectAll != nullptr);
    QVERIFY(exportMp3 != nullptr);
    QVERIFY(play != nullptr);

    QCOMPARE(list->selectionMode(), QAbstractItemView::MultiSelection);
    QVERIFY(selectAll->isVisible());
    QVERIFY(exportMp3->isVisible());

    std::vector<voxstudio::db::TakeRecord> takes(3);
    takes[0].id = "take-one";
    takes[1].id = "take-two";
    takes[2].id = "take-three";
    takesWidget->setTakes(takes);

    QCOMPARE(list->selectedItems().size(), 0);
    QVERIFY(!exportMp3->isEnabled());
    QVERIFY(!play->isEnabled());

    std::vector<voxstudio::db::TakeRecord> requested;
    connect(takesWidget, &voxstudio::ui::TakeListWidget::exportTakesRequested,
            this, [&requested](std::vector<voxstudio::db::TakeRecord> selected) {
                requested = std::move(selected);
            });

    selectAll->click();
    QCOMPARE(list->selectedItems().size(), 3);
    QVERIFY(!play->isEnabled());
    QCOMPARE(selectAll->text(), QStringLiteral("Clear Selection"));

    exportMp3->click();
    QCOMPARE(requested.size(), 3);
    QCOMPARE(requested[0].id, std::string{"take-one"});
    QCOMPARE(requested[1].id, std::string{"take-two"});
    QCOMPARE(requested[2].id, std::string{"take-three"});

    selectAll->click();
    QCOMPARE(list->selectedItems().size(), 0);
    QCOMPARE(selectAll->text(), QStringLiteral("Select All"));
    QVERIFY(!exportMp3->isEnabled());
}

QTEST_MAIN(TextToSpeechPanelTest)

#include "test_text_to_speech_panel.moc"
