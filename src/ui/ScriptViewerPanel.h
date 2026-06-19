#pragma once

#include "audio/AudioEngine.h"
#include "core/Expected.h"
#include "core/Project.h"
#include "core/TakeManager.h"
#include "db/ScriptRepository.h"
#include "db/TakeRepository.h"
#include "db/VoiceRepository.h"
#include "io/scripts/ScriptTypes.h"

#include <QFutureWatcher>
#include <QString>
#include <QWidget>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QListWidget;
class QPushButton;

namespace voxstudio::ui {

struct ScriptImportResult final {
    bool success{false};
    QString message;
    db::ImportedScriptRecord importedScript;
};

struct TtsGenerationResult final {
    bool success{false};
    QString message;
    QString playbackWarning;
    db::TakeRecord take;
};

class TakeListWidget;

class ScriptViewerPanel final : public QWidget {
    Q_OBJECT

public:
    explicit ScriptViewerPanel(QWidget* parent = nullptr);

    void setProject(std::optional<core::Project> project);

signals:
    void scriptImported();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void importScript();
    void finishParseScript();
    void finishImportScript();
    void generateCurrentLine();
    void finishGeneration();
    void loadScripts();
    void loadSelectedScriptLines();
    void handleLineSelectionChanged();
    void loadCurrentLineTakes();
    void setBusy(bool isBusy);
    void setLines(std::vector<db::ScriptLineRecord> lines);
    void selectLine(int row);
    void confirmCurrentLine();
    void playTake(db::TakeRecord take);
    void starTake(db::TakeRecord take);
    void deleteTake(db::TakeRecord take);
    void applyLineSettingsToControls(const db::ScriptLineRecord& line);
    [[nodiscard]] core::VoiceSettings currentVoiceSettings() const;
    [[nodiscard]] db::ScriptLineRecord* currentLine();
    [[nodiscard]] const db::ScriptLineRecord* currentLine() const;
    [[nodiscard]] std::vector<std::string>
    detectedSpeakers(const io::scripts::ParsedScript& script) const;
    [[nodiscard]] std::vector<db::VoiceRecord> cachedVoices() const;

    db::ScriptRepository m_scriptRepository;
    db::TakeRepository m_takeRepository;
    db::VoiceRepository m_voiceRepository;
    core::TakeManager m_takeManager;
    std::optional<core::Project> m_project;
    std::vector<db::ScriptRecord> m_scripts;
    std::vector<db::ScriptLineRecord> m_lines;
    audio::AudioEngine m_audioEngine;
    QComboBox* m_scriptSelector{nullptr};
    QListWidget* m_lineList{nullptr};
    TakeListWidget* m_takeListWidget{nullptr};
    QLabel* m_statusLabel{nullptr};
    QPushButton* m_importButton{nullptr};
    QPushButton* m_confirmButton{nullptr};
    QPushButton* m_generateButton{nullptr};
    QDoubleSpinBox* m_stabilitySpin{nullptr};
    QDoubleSpinBox* m_similaritySpin{nullptr};
    QDoubleSpinBox* m_styleSpin{nullptr};
    QCheckBox* m_speakerBoostCheck{nullptr};
    std::unique_ptr<QFutureWatcher<core::Expected<io::scripts::ParsedScript>>> m_parseWatcher;
    std::unique_ptr<QFutureWatcher<ScriptImportResult>> m_importWatcher;
    std::unique_ptr<QFutureWatcher<TtsGenerationResult>> m_generationWatcher;
};

} // namespace voxstudio::ui
