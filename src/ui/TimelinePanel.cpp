#include "ui/TimelinePanel.h"

#include "audio/AudioFile.h"
#include "core/TakeManager.h"
#include "net/elevenlabs/TtsApi.h"
#include "secrets/DpapiVault.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace voxstudio::ui {
namespace {

using GenerationWatcher = QFutureWatcher<TimelineGenerationResult>;

template <typename TWidget, typename... TArgs>
[[nodiscard]] TWidget* addOwnedWidget(QLayout& layout, TArgs&&... args) {
    auto widget = std::make_unique<TWidget>(std::forward<TArgs>(args)...);
    auto* widgetPointer = widget.get();
    layout.addWidget(widget.release());
    return widgetPointer;
}

[[nodiscard]] QString sourceLineText(const db::ScriptLineRecord& line) {
    const auto speaker = line.characterName.empty()
                             ? QStringLiteral("Unknown")
                             : QString::fromStdString(line.characterName);
    return QStringLiteral("%1: %2").arg(speaker, QString::fromStdString(line.text));
}

[[nodiscard]] QString sequenceLineText(const core::SequenceLine& line) {
    const auto takeStatus =
        line.takeFilePath.empty() ? QStringLiteral("missing take") : QStringLiteral("take ready");
    const auto speaker = line.characterName.empty()
                             ? QStringLiteral("Unknown")
                             : QString::fromStdString(line.characterName);
    return QStringLiteral("%1: %2 [%3 ms, %4]")
        .arg(speaker, QString::fromStdString(line.text))
        .arg(line.gapMs)
        .arg(takeStatus);
}

[[nodiscard]] QString apiErrorText(const net::elevenlabs::ApiError& error) {
    if (error.statusCode > 0) {
        return QStringLiteral("%1 (HTTP %2)")
            .arg(QString::fromStdString(error.message))
            .arg(error.statusCode);
    }
    return QString::fromStdString(error.message);
}

[[nodiscard]] core::Expected<std::string> loadApiKey() {
    const secrets::DpapiVault vault;
    return vault.loadElevenLabsApiKey();
}

[[nodiscard]] core::Expected<audio::PcmAudioBuffer>
generateLineTts(const std::string& apiKey, const core::SequenceLine& line) {
    if (line.voiceId.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Sequence line has no voice: " + line.lineId);
    }

    net::elevenlabs::TtsRequest request;
    request.voiceId = line.voiceId;
    request.text = line.text;
    request.outputFormat = "pcm_44100";

    const net::elevenlabs::TtsApi api{apiKey};
    auto streamed = api.streamSpeech(request, {});
    if (!streamed) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               apiErrorText(streamed.error()).toStdString());
    }

    const auto sampleRate =
        net::elevenlabs::pcmSampleRateFromOutputFormat(request.outputFormat);
    return audio::pcm16LittleEndianToPcm(streamed.value().audioBytes, sampleRate, 1);
}

[[nodiscard]] TimelineGenerationResult
generateSequenceTakes(const core::Project& project,
                      core::Sequence sequence,
                      const std::size_t maxConcurrency) {
    auto apiKey = loadApiKey();
    if (!apiKey) {
        return TimelineGenerationResult{
            false, QString::fromStdString(apiKey.error().message), sequence};
    }

    const core::SequenceRenderer renderer;
    auto generated = renderer.generateLineAudio(
        sequence,
        [&apiKey](const core::SequenceLine& line) {
            return generateLineTts(apiKey.value(), line);
        },
        core::SequenceGenerationOptions{maxConcurrency});
    if (!generated) {
        return TimelineGenerationResult{
            false, QString::fromStdString(generated.error().message), sequence};
    }

    core::TakeManager takeManager;
    for (const auto& audio : generated.value()) {
        const auto line = std::ranges::find_if(sequence.lines, [&audio](const auto& candidate) {
            return candidate.lineId == audio.lineId;
        });
        if (line == sequence.lines.end()) {
            continue;
        }

        auto saved = takeManager.saveTtsTake(project.rootPath(),
                                             line->lineId,
                                             line->voiceId,
                                             audio.audio,
                                             core::defaultVoiceSettings());
        if (!saved) {
            return TimelineGenerationResult{
                false, QString::fromStdString(saved.error().message), sequence};
        }

        line->activeTakeId = saved.value().take.id;
        line->takeFilePath = saved.value().take.filePath;
    }

    return TimelineGenerationResult{
        true, QStringLiteral("Generated sequence takes."), std::move(sequence)};
}

} // namespace

TimelinePanel::TimelinePanel(QWidget* parent)
    : QWidget(parent)
    , m_hotkeyManager(this)
    , m_generationWatcher(std::make_unique<QFutureWatcher<TimelineGenerationResult>>()) {
    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto title = std::make_unique<QLabel>(QStringLiteral("Sequence Builder"));
    title->setObjectName(QStringLiteral("TimelineTitle"));
    rootLayout->addWidget(title.release());

    auto topLayout = std::make_unique<QHBoxLayout>();
    m_scriptSelector = addOwnedWidget<QComboBox>(*topLayout);
    m_scriptSelector->setObjectName(QStringLiteral("TimelineScriptSelector"));
    m_buildButton = addOwnedWidget<QPushButton>(*topLayout, QStringLiteral("Build Sequence"));
    m_buildButton->setObjectName(QStringLiteral("TimelineBuildButton"));
    m_generateButton = addOwnedWidget<QPushButton>(*topLayout, QStringLiteral("Generate Sequence"));
    m_generateButton->setObjectName(QStringLiteral("TimelineGenerateButton"));
    m_previewButton = addOwnedWidget<QPushButton>(*topLayout, QStringLiteral("Preview"));
    m_previewButton->setObjectName(QStringLiteral("TimelinePreviewButton"));
    rootLayout->addLayout(topLayout.release());

    auto listsLayout = std::make_unique<QHBoxLayout>();
    m_sourceList = addOwnedWidget<QListWidget>(*listsLayout);
    m_sourceList->setObjectName(QStringLiteral("TimelineSourceList"));
    m_sourceList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_timelineList = addOwnedWidget<QListWidget>(*listsLayout);
    m_timelineList->setObjectName(QStringLiteral("TimelineList"));
    m_timelineList->setDragDropMode(QAbstractItemView::InternalMove);
    m_timelineList->setSelectionMode(QAbstractItemView::SingleSelection);
    rootLayout->addLayout(listsLayout.release());

    auto controlsLayout = std::make_unique<QHBoxLayout>();
    controlsLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Gap after")).release());
    m_gapSpin = addOwnedWidget<QSpinBox>(*controlsLayout);
    m_gapSpin->setObjectName(QStringLiteral("TimelineGapSpin"));
    m_gapSpin->setRange(0, 3000);
    m_gapSpin->setSingleStep(50);
    m_gapSpin->setSuffix(QStringLiteral(" ms"));

    controlsLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Parallel")).release());
    m_concurrencySpin = addOwnedWidget<QSpinBox>(*controlsLayout);
    m_concurrencySpin->setObjectName(QStringLiteral("TimelineConcurrencySpin"));
    m_concurrencySpin->setRange(1, 6);
    m_concurrencySpin->setValue(3);

    m_exportWavButton = addOwnedWidget<QPushButton>(*controlsLayout, QStringLiteral("Export WAV"));
    m_exportWavButton->setObjectName(QStringLiteral("TimelineExportWavButton"));
    m_exportOpusButton =
        addOwnedWidget<QPushButton>(*controlsLayout, QStringLiteral("Export Opus"));
    m_exportOpusButton->setObjectName(QStringLiteral("TimelineExportOpusButton"));
    rootLayout->addLayout(controlsLayout.release());

    m_activeCharacterLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_activeCharacterLabel->setObjectName(QStringLiteral("TimelineActiveCharacterLabel"));
    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("TimelineStatusLabel"));
    m_statusLabel->setWordWrap(true);

    setLayout(rootLayout.release());
    setFocusPolicy(Qt::StrongFocus);
    installEventFilter(this);
    m_sourceList->installEventFilter(this);
    m_timelineList->installEventFilter(this);

    connect(m_scriptSelector, &QComboBox::currentIndexChanged, this,
            &TimelinePanel::loadSelectedScriptLines);
    connect(m_buildButton, &QPushButton::clicked, this,
            &TimelinePanel::buildSequenceFromSelection);
    connect(m_generateButton, &QPushButton::clicked, this, &TimelinePanel::generateSequence);
    connect(m_previewButton, &QPushButton::clicked, this, &TimelinePanel::previewSequence);
    connect(m_exportWavButton, &QPushButton::clicked, this, &TimelinePanel::exportSequenceWav);
    connect(m_exportOpusButton, &QPushButton::clicked, this,
            &TimelinePanel::exportSequenceOpus);
    connect(m_gapSpin, &QSpinBox::valueChanged, this, &TimelinePanel::updateSelectedGap);
    connect(m_generationWatcher.get(), &GenerationWatcher::finished, this,
            &TimelinePanel::finishSequenceGeneration);
    connect(&m_hotkeyManager, &HotkeyManager::activeCharacterChanged, this,
            &TimelinePanel::updateActiveCharacterLabel);

    setBusy(false);
    m_activeCharacterLabel->setText(QStringLiteral("No active character"));
    m_statusLabel->setText(QStringLiteral("Open a project and import a script."));
}

void TimelinePanel::setProject(std::optional<core::Project> project) {
    m_project = std::move(project);
    m_sequence.reset();
    refreshHotkeys();
    loadScripts();
}

bool TimelinePanel::eventFilter(QObject* watched, QEvent* event) {
    if ((watched == this || watched == m_sourceList || watched == m_timelineList) &&
        event->type() == QEvent::KeyPress) {
        const auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (m_hotkeyManager.handleKeyPress(*keyEvent)) {
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

void TimelinePanel::loadScripts() {
    m_scriptSelector->blockSignals(true);
    m_scriptSelector->clear();
    m_scripts.clear();
    m_sourceLines.clear();
    m_sourceList->clear();
    m_sequence.reset();
    refreshTimelineList();

    if (!m_project.has_value()) {
        m_scriptSelector->setEnabled(false);
        m_scriptSelector->blockSignals(false);
        setBusy(false);
        return;
    }

    auto scripts = m_scriptRepository.listScripts(m_project->rootPath());
    if (!scripts) {
        m_statusLabel->setText(QString::fromStdString(scripts.error().message));
        m_scriptSelector->blockSignals(false);
        return;
    }

    m_scripts = std::move(scripts).value();
    for (const auto& script : m_scripts) {
        const auto label = std::filesystem::path{script.sourcePath}.filename().empty()
                               ? QString::fromStdString(script.id)
                               : QString::fromStdString(
                                     std::filesystem::path{script.sourcePath}
                                         .filename()
                                         .string());
        m_scriptSelector->addItem(label, QString::fromStdString(script.id));
    }
    m_scriptSelector->setEnabled(!m_scripts.empty());
    m_scriptSelector->blockSignals(false);
    loadSelectedScriptLines();
}

void TimelinePanel::loadSelectedScriptLines() {
    m_sourceLines.clear();
    m_sourceList->clear();
    if (!m_project.has_value() || m_scriptSelector->currentIndex() < 0) {
        setBusy(false);
        return;
    }

    const auto scriptId = m_scriptSelector->currentData().toString().toStdString();
    auto lines = m_scriptRepository.listLines(m_project->rootPath(), scriptId);
    if (!lines) {
        m_statusLabel->setText(QString::fromStdString(lines.error().message));
        return;
    }

    m_sourceLines = std::move(lines).value();
    for (const auto& line : m_sourceLines) {
        auto item = std::make_unique<QListWidgetItem>(sourceLineText(line));
        item->setData(Qt::UserRole, QString::fromStdString(line.id));
        m_sourceList->addItem(item.release());
    }
    m_statusLabel->setText(QStringLiteral("Select lines or build from the full script."));
    setBusy(false);
}

void TimelinePanel::buildSequenceFromSelection() {
    if (!m_project.has_value()) {
        return;
    }

    const auto items = selectedSourceItems();
    const auto name =
        m_scriptSelector->currentText().isEmpty()
            ? QStringLiteral("Dialogue Sequence")
            : QStringLiteral("%1 Sequence").arg(m_scriptSelector->currentText());
    auto sequence = m_sequenceRepository.createSequence(m_project->rootPath(),
                                                        name.toStdString(),
                                                        items);
    if (!sequence) {
        m_statusLabel->setText(QString::fromStdString(sequence.error().message));
        return;
    }

    m_sequence = std::move(sequence).value();
    refreshTimelineList();
    emit sequenceChanged();
    m_statusLabel->setText(QStringLiteral("Sequence built."));
}

void TimelinePanel::generateSequence() {
    if (!m_project.has_value() || m_generationWatcher->isRunning()) {
        return;
    }

    auto sequence = currentTimelineSequence();
    if (!sequence.has_value()) {
        m_statusLabel->setText(QStringLiteral("Build a sequence first."));
        return;
    }

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Generating sequence takes..."));
    const auto project = *m_project;
    const auto maxConcurrency = static_cast<std::size_t>(m_concurrencySpin->value());
    m_generationWatcher->setFuture(QtConcurrent::run(
        [project, sequence = std::move(sequence).value(), maxConcurrency]() mutable {
            return generateSequenceTakes(project, std::move(sequence), maxConcurrency);
        }));
}

void TimelinePanel::finishSequenceGeneration() {
    const auto result = m_generationWatcher->result();
    setBusy(false);
    if (!result.success) {
        m_statusLabel->setText(result.message);
        return;
    }

    m_sequence = result.sequence;
    refreshTimelineList();
    m_statusLabel->setText(result.message);
    emit sequenceChanged();
}

void TimelinePanel::previewSequence() {
    auto sequence = currentTimelineSequence();
    if (!m_project.has_value() || !sequence.has_value()) {
        m_statusLabel->setText(QStringLiteral("Build a sequence first."));
        return;
    }

    auto rendered = m_renderer.render(m_project->rootPath(), sequence.value());
    if (!rendered) {
        m_statusLabel->setText(QString::fromStdString(rendered.error().message));
        return;
    }

    auto queued = m_audioEngine.queuePcm(rendered.value().audio);
    if (!queued) {
        m_statusLabel->setText(QString::fromStdString(queued.error().message));
        return;
    }

    m_statusLabel->setText(QStringLiteral("Playing sequence preview."));
}

void TimelinePanel::exportSequenceWav() {
    auto sequence = currentTimelineSequence();
    if (!m_project.has_value() || !sequence.has_value()) {
        m_statusLabel->setText(QStringLiteral("Build a sequence first."));
        return;
    }

    const auto path = QFileDialog::getSaveFileName(
        this, QStringLiteral("Export Sequence WAV"), QString{}, QStringLiteral("WAV (*.wav)"));
    if (path.isEmpty()) {
        return;
    }

    auto exported =
        m_renderer.exportWav(m_project->rootPath(), sequence.value(),
                             std::filesystem::path{path.toStdString()});
    if (!exported) {
        m_statusLabel->setText(QString::fromStdString(exported.error().message));
        return;
    }
    m_statusLabel->setText(QStringLiteral("Sequence WAV exported."));
}

void TimelinePanel::exportSequenceOpus() {
    auto sequence = currentTimelineSequence();
    if (!m_project.has_value() || !sequence.has_value()) {
        m_statusLabel->setText(QStringLiteral("Build a sequence first."));
        return;
    }

    const auto path = QFileDialog::getSaveFileName(
        this, QStringLiteral("Export Sequence Opus"), QString{}, QStringLiteral("Opus (*.opus)"));
    if (path.isEmpty()) {
        return;
    }

    auto exported =
        m_renderer.exportOpus(m_project->rootPath(), sequence.value(),
                              std::filesystem::path{path.toStdString()});
    if (!exported) {
        m_statusLabel->setText(QString::fromStdString(exported.error().message));
        return;
    }
    m_statusLabel->setText(QStringLiteral("Sequence Opus exported."));
}

void TimelinePanel::updateSelectedGap(const int gapMs) {
    auto* item = m_timelineList->currentItem();
    if (item == nullptr || !m_sequence.has_value()) {
        return;
    }

    item->setData(Qt::UserRole + 1, gapMs);
    for (auto& line : m_sequence->lines) {
        if (line.lineId == item->data(Qt::UserRole).toString().toStdString()) {
            line.gapMs = gapMs;
            item->setText(sequenceLineText(line));
            break;
        }
    }
}

void TimelinePanel::updateActiveCharacterLabel(const CharacterHotkeySlot& slot) {
    m_activeCharacterLabel->setText(
        QStringLiteral("Active character %1: %2")
            .arg(slot.number)
            .arg(QString::fromStdString(slot.character.name)));
}

void TimelinePanel::refreshTimelineList() {
    m_timelineList->clear();
    if (!m_sequence.has_value()) {
        setBusy(false);
        return;
    }

    for (const auto& line : m_sequence->lines) {
        auto item = std::make_unique<QListWidgetItem>(sequenceLineText(line));
        item->setData(Qt::UserRole, QString::fromStdString(line.lineId));
        item->setData(Qt::UserRole + 1, line.gapMs);
        m_timelineList->addItem(item.release());
    }
    if (m_timelineList->count() > 0) {
        m_timelineList->setCurrentRow(0);
        m_gapSpin->setValue(m_timelineList->item(0)->data(Qt::UserRole + 1).toInt());
    }
    setBusy(false);
}

void TimelinePanel::refreshHotkeys() {
    if (!m_project.has_value()) {
        m_hotkeyManager.setCharacters({});
        m_activeCharacterLabel->setText(QStringLiteral("No active character"));
        return;
    }

    auto characters = m_scriptRepository.listCharacters(m_project->rootPath());
    if (!characters) {
        m_activeCharacterLabel->setText(QString::fromStdString(characters.error().message));
        return;
    }
    m_hotkeyManager.setCharacters(std::move(characters).value());
}

void TimelinePanel::setBusy(const bool isBusy) {
    const bool hasProject = m_project.has_value();
    const bool hasSourceLines = !m_sourceLines.empty();
    const bool hasSequence = m_sequence.has_value() && !m_sequence->lines.empty();
    m_scriptSelector->setEnabled(hasProject && !isBusy && !m_scripts.empty());
    m_sourceList->setEnabled(hasProject && !isBusy && hasSourceLines);
    m_timelineList->setEnabled(hasProject && !isBusy && hasSequence);
    m_buildButton->setEnabled(hasProject && !isBusy && hasSourceLines);
    m_generateButton->setEnabled(hasProject && !isBusy && hasSequence);
    m_previewButton->setEnabled(hasProject && !isBusy && hasSequence);
    m_exportWavButton->setEnabled(hasProject && !isBusy && hasSequence);
    m_exportOpusButton->setEnabled(hasProject && !isBusy && hasSequence);
    m_gapSpin->setEnabled(hasProject && !isBusy && hasSequence);
    m_concurrencySpin->setEnabled(hasProject && !isBusy);
}

std::vector<db::NewSequenceItemRecord> TimelinePanel::selectedSourceItems() const {
    QList<QListWidgetItem*> selectedItems = m_sourceList->selectedItems();
    if (selectedItems.empty()) {
        for (int row = 0; row < m_sourceList->count(); ++row) {
            selectedItems.push_back(m_sourceList->item(row));
        }
    }

    std::ranges::sort(selectedItems, [this](const auto* left, const auto* right) {
        return m_sourceList->row(left) < m_sourceList->row(right);
    });

    std::vector<db::NewSequenceItemRecord> items;
    items.reserve(static_cast<std::size_t>(selectedItems.size()));
    for (const auto* item : selectedItems) {
        items.push_back(
            db::NewSequenceItemRecord{item->data(Qt::UserRole).toString().toStdString(), 250});
    }
    return items;
}

std::vector<db::NewSequenceItemRecord> TimelinePanel::timelineItems() const {
    std::vector<db::NewSequenceItemRecord> items;
    items.reserve(static_cast<std::size_t>(m_timelineList->count()));
    for (int row = 0; row < m_timelineList->count(); ++row) {
        const auto* item = m_timelineList->item(row);
        items.push_back(db::NewSequenceItemRecord{
            item->data(Qt::UserRole).toString().toStdString(),
            item->data(Qt::UserRole + 1).toInt()});
    }
    return items;
}

std::optional<core::Sequence> TimelinePanel::currentTimelineSequence() const {
    if (!m_project.has_value() || !m_sequence.has_value()) {
        return std::nullopt;
    }

    auto updated = m_sequenceRepository.updateSequenceItems(m_project->rootPath(),
                                                            m_sequence->id,
                                                            timelineItems());
    if (!updated) {
        return std::nullopt;
    }
    return updated.value();
}

} // namespace voxstudio::ui
