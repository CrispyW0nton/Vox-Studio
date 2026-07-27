#pragma once

#include "audio/AudioEngine.h"
#include "audio/Capture.h"
#include "core/Project.h"
#include "db/ScriptRepository.h"
#include "db/TakeRepository.h"
#include "db/VoiceRepository.h"
#include "voxcpm/VoxCpmSidecar.h"

#include <QByteArray>
#include <QFutureWatcher>
#include <QWidget>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class QButtonGroup;
class QComboBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;

namespace voxstudio::ui {

class TakeListWidget;

struct TextSynthesisResult final {
    bool success{false};
    QString message;
    QByteArray pcm16Audio;
    int sampleRate{24000};
    int latencyMs{0};
    int sectionCount{1};
    QString delivery;
    QString pronunciations;
    QString performanceMode;
};

struct StoryDirectionResult final {
    bool success{false};
    QString message;
    QString preview;
};

class TextToSpeechPanel final : public QWidget {
    Q_OBJECT

public:
    explicit TextToSpeechPanel(QWidget* parent = nullptr);
    ~TextToSpeechPanel() override;

    void setProject(std::optional<core::Project> project);

private:
    void refreshVoices();
    void refreshOutputs();
    void refreshTakes();
    void generateSpeech();
    void finishGeneration();
    void previewDirection();
    void finishDirectionPreview();
    void updateModeControls();
    void stopPlayback();
    void updateOutputDevice(int index);
    void playTake(db::TakeRecord take);
    void starTake(db::TakeRecord take);
    void revealTake(db::TakeRecord take);
    void deleteTake(db::TakeRecord take);
    void setBusy(bool busy);
    void setStatus(const QString& text);
    [[nodiscard]] std::string selectedVoiceId() const;
    [[nodiscard]] QString selectedVoiceName() const;
    [[nodiscard]] std::string selectedDelivery() const;
    [[nodiscard]] std::string selectedMode() const;

    audio::AudioEngine m_audioEngine;
    db::ScriptRepository m_scriptRepository;
    db::TakeRepository m_takeRepository;
    db::VoiceRepository m_voiceRepository;
    voxcpm::VoxCpmSidecar m_sidecar;
    std::optional<core::Project> m_project;
    std::vector<audio::AudioDeviceInfo> m_outputDevices;
    std::unique_ptr<QFutureWatcher<TextSynthesisResult>> m_generationWatcher;
    std::unique_ptr<QFutureWatcher<StoryDirectionResult>> m_analysisWatcher;
    std::string m_activeLineId;
    std::string m_activeVoiceId;
    std::string m_activeText;
    QString m_activeVoiceName;
    std::string m_activeDelivery;
    std::string m_activeMode;
    std::filesystem::path m_activeProjectRoot;
    QComboBox* m_voiceCombo{nullptr};
    QComboBox* m_outputCombo{nullptr};
    QPlainTextEdit* m_textEdit{nullptr};
    QPlainTextEdit* m_directionPreview{nullptr};
    QButtonGroup* m_modeGroup{nullptr};
    QButtonGroup* m_deliveryGroup{nullptr};
    QPushButton* m_previewButton{nullptr};
    QPushButton* m_generateButton{nullptr};
    QPushButton* m_stopButton{nullptr};
    QLabel* m_directionLabel{nullptr};
    QLabel* m_statusLabel{nullptr};
    TakeListWidget* m_takesWidget{nullptr};
};

} // namespace voxstudio::ui
