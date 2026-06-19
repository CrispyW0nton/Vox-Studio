#pragma once

#include "audio/AudioEngine.h"
#include "core/Project.h"
#include "core/Sequence.h"
#include "core/SequenceRenderer.h"
#include "db/ScriptRepository.h"
#include "db/SequenceRepository.h"
#include "ui/HotkeyManager.h"

#include <QFutureWatcher>
#include <QString>
#include <QWidget>

#include <memory>
#include <optional>
#include <vector>

class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QSpinBox;

namespace voxstudio::ui {

struct TimelineGenerationResult final {
    bool success{false};
    QString message;
    core::Sequence sequence;
};

class TimelinePanel final : public QWidget {
    Q_OBJECT

public:
    explicit TimelinePanel(QWidget* parent = nullptr);

    void setProject(std::optional<core::Project> project);

signals:
    void sequenceChanged();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void loadScripts();
    void loadSelectedScriptLines();
    void buildSequenceFromSelection();
    void generateSequence();
    void finishSequenceGeneration();
    void previewSequence();
    void exportSequenceWav();
    void exportSequenceOpus();
    void updateSelectedGap(int gapMs);
    void updateActiveCharacterLabel(const CharacterHotkeySlot& slot);
    void refreshTimelineList();
    void refreshHotkeys();
    void setBusy(bool isBusy);
    [[nodiscard]] std::vector<db::NewSequenceItemRecord> selectedSourceItems() const;
    [[nodiscard]] std::vector<db::NewSequenceItemRecord> timelineItems() const;
    [[nodiscard]] std::optional<core::Sequence> currentTimelineSequence() const;

    db::ScriptRepository m_scriptRepository;
    db::SequenceRepository m_sequenceRepository;
    core::SequenceRenderer m_renderer;
    audio::AudioEngine m_audioEngine;
    HotkeyManager m_hotkeyManager;
    std::optional<core::Project> m_project;
    std::vector<db::ScriptRecord> m_scripts;
    std::vector<db::ScriptLineRecord> m_sourceLines;
    std::optional<core::Sequence> m_sequence;
    QComboBox* m_scriptSelector{nullptr};
    QListWidget* m_sourceList{nullptr};
    QListWidget* m_timelineList{nullptr};
    QLabel* m_activeCharacterLabel{nullptr};
    QLabel* m_statusLabel{nullptr};
    QPushButton* m_buildButton{nullptr};
    QPushButton* m_generateButton{nullptr};
    QPushButton* m_previewButton{nullptr};
    QPushButton* m_exportWavButton{nullptr};
    QPushButton* m_exportOpusButton{nullptr};
    QSpinBox* m_gapSpin{nullptr};
    QSpinBox* m_concurrencySpin{nullptr};
    std::unique_ptr<QFutureWatcher<TimelineGenerationResult>> m_generationWatcher;
};

} // namespace voxstudio::ui
