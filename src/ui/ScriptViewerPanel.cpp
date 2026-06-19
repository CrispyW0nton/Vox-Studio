#include "ui/ScriptViewerPanel.h"

#include "audio/AudioFile.h"
#include "io/scripts/ScriptImporter.h"
#include "net/elevenlabs/TtsApi.h"
#include "secrets/DpapiVault.h"
#include "ui/CharacterAssignmentDialog.h"
#include "ui/TakeListWidget.h"

#include <QAbstractItemView>
#include <QBrush>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace voxstudio::ui {
namespace {

using ParsedScriptWatcher = QFutureWatcher<core::Expected<io::scripts::ParsedScript>>;
using ScriptImportWatcher = QFutureWatcher<ScriptImportResult>;
using TtsGenerationWatcher = QFutureWatcher<TtsGenerationResult>;

template <typename TWidget, typename... TArgs>
[[nodiscard]] TWidget* addOwnedWidget(QLayout& layout, TArgs&&... args) {
    auto widget = std::make_unique<TWidget>(std::forward<TArgs>(args)...);
    auto* widgetPointer = widget.get();
    layout.addWidget(widget.release());
    return widgetPointer;
}

[[nodiscard]] QString lineText(const db::ScriptLineRecord& line) {
    const auto prefix = line.characterName.empty()
                            ? QStringLiteral("Unknown")
                            : QString::fromStdString(line.characterName);
    return QStringLiteral("%1: %2").arg(prefix, QString::fromStdString(line.text));
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

[[nodiscard]] ScriptImportResult importParsedScript(const std::filesystem::path& projectRoot,
                                                    const io::scripts::ParsedScript& script,
                                                    const std::vector<db::CharacterAssignment>&
                                                        assignments) {
    const db::ScriptRepository repository;
    auto imported = repository.importScript(projectRoot, script, assignments);
    if (!imported) {
        return ScriptImportResult{false, QString::fromStdString(imported.error().message), {}};
    }

    return ScriptImportResult{
        true,
        QStringLiteral("Imported %1 lines.").arg(static_cast<int>(imported.value().lines.size())),
        std::move(imported).value()};
}

[[nodiscard]] TtsGenerationResult generateTtsTake(const core::Project& project,
                                                  const db::ScriptLineRecord& line,
                                                  const core::VoiceSettings& settings,
                                                  audio::AudioEngine* audioEngine) {
    auto apiKey = loadApiKey();
    if (!apiKey) {
        return TtsGenerationResult{false, QString::fromStdString(apiKey.error().message), {}, {}};
    }

    const auto settingsJson = core::voiceSettingsToJson(settings);
    const db::ScriptRepository scriptRepository;
    auto savedSettings =
        scriptRepository.updateLineVoiceSettings(project.rootPath(), line.id, settingsJson);
    if (!savedSettings) {
        return TtsGenerationResult{
            false, QString::fromStdString(savedSettings.error().message), {}, {}};
    }

    net::elevenlabs::TtsRequest request;
    request.voiceId = line.voiceId;
    request.text = line.text;
    request.outputFormat = "pcm_44100";
    request.voiceSettings = settings;

    const auto sampleRate = net::elevenlabs::pcmSampleRateFromOutputFormat(request.outputFormat);
    std::vector<std::uint8_t> pendingBytes;
    QString playbackWarning;

    const auto onChunk = [audioEngine, sampleRate, &pendingBytes, &playbackWarning](
                             std::span<const std::uint8_t> chunk) {
        pendingBytes.insert(pendingBytes.end(), chunk.begin(), chunk.end());
        const auto playableByteCount = pendingBytes.size() - (pendingBytes.size() % 2U);
        if (playableByteCount == 0U || audioEngine == nullptr) {
            return true;
        }

        const auto playableBytes =
            std::span<const std::uint8_t>{pendingBytes.data(), playableByteCount};
        auto queued = audioEngine->queuePcm16LittleEndian(playableBytes, sampleRate, 1);
        if (!queued && playbackWarning.isEmpty()) {
            playbackWarning = QString::fromStdString(queued.error().message);
        }
        pendingBytes.erase(pendingBytes.begin(),
                           pendingBytes.begin() + static_cast<std::ptrdiff_t>(playableByteCount));
        return true;
    };

    const net::elevenlabs::TtsApi api{std::move(apiKey).value()};
    auto streamed = api.streamSpeech(request, onChunk);
    if (!streamed) {
        return TtsGenerationResult{false, apiErrorText(streamed.error()), playbackWarning, {}};
    }

    auto audio =
        audio::pcm16LittleEndianToPcm(streamed.value().audioBytes, sampleRate, 1);
    if (!audio) {
        return TtsGenerationResult{
            false, QString::fromStdString(audio.error().message), playbackWarning, {}};
    }

    core::TakeManager takeManager;
    auto savedTake =
        takeManager.saveTtsTake(project.rootPath(), line.id, line.voiceId, audio.value(), settings);
    if (!savedTake) {
        return TtsGenerationResult{
            false, QString::fromStdString(savedTake.error().message), playbackWarning, {}};
    }

    return TtsGenerationResult{true, QStringLiteral("Generated and saved take."),
                               playbackWarning, savedTake.value().take};
}

} // namespace

ScriptViewerPanel::ScriptViewerPanel(QWidget* parent)
    : QWidget(parent)
    , m_parseWatcher(std::make_unique<QFutureWatcher<core::Expected<io::scripts::ParsedScript>>>())
    , m_importWatcher(std::make_unique<QFutureWatcher<ScriptImportResult>>())
    , m_generationWatcher(std::make_unique<QFutureWatcher<TtsGenerationResult>>()) {
    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto* titleLabel = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Script Viewer"));
    titleLabel->setObjectName(QStringLiteral("ScriptViewerTitle"));

    auto actionsLayout = std::make_unique<QHBoxLayout>();
    m_importButton = addOwnedWidget<QPushButton>(*actionsLayout, QStringLiteral("Import Script"));
    m_importButton->setObjectName(QStringLiteral("ScriptImportButton"));
    m_confirmButton = addOwnedWidget<QPushButton>(*actionsLayout, QStringLiteral("Confirm Line"));
    m_confirmButton->setObjectName(QStringLiteral("ScriptConfirmButton"));
    m_generateButton = addOwnedWidget<QPushButton>(*actionsLayout, QStringLiteral("Generate"));
    m_generateButton->setObjectName(QStringLiteral("ScriptGenerateButton"));
    rootLayout->addLayout(actionsLayout.release());

    m_scriptSelector = addOwnedWidget<QComboBox>(*rootLayout);
    m_scriptSelector->setObjectName(QStringLiteral("ScriptSelector"));
    m_scriptSelector->setEnabled(false);

    m_lineList = addOwnedWidget<QListWidget>(*rootLayout);
    m_lineList->setObjectName(QStringLiteral("ScriptLineList"));
    m_lineList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_lineList->installEventFilter(this);

    auto settingsLayout = std::make_unique<QGridLayout>();
    settingsLayout->addWidget(
        std::make_unique<QLabel>(QStringLiteral("Stability")).release(), 0, 0);
    m_stabilitySpin = addOwnedWidget<QDoubleSpinBox>(*settingsLayout);
    m_stabilitySpin->setObjectName(QStringLiteral("TtsStabilitySpin"));
    m_stabilitySpin->setRange(0.0, 1.0);
    m_stabilitySpin->setSingleStep(0.05);

    settingsLayout->addWidget(
        std::make_unique<QLabel>(QStringLiteral("Similarity")).release(), 0, 2);
    m_similaritySpin = addOwnedWidget<QDoubleSpinBox>(*settingsLayout);
    m_similaritySpin->setObjectName(QStringLiteral("TtsSimilaritySpin"));
    m_similaritySpin->setRange(0.0, 1.0);
    m_similaritySpin->setSingleStep(0.05);

    settingsLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Style")).release(), 1, 0);
    m_styleSpin = addOwnedWidget<QDoubleSpinBox>(*settingsLayout);
    m_styleSpin->setObjectName(QStringLiteral("TtsStyleSpin"));
    m_styleSpin->setRange(0.0, 1.0);
    m_styleSpin->setSingleStep(0.05);

    m_speakerBoostCheck =
        std::make_unique<QCheckBox>(QStringLiteral("Speaker boost")).release();
    m_speakerBoostCheck->setObjectName(QStringLiteral("TtsSpeakerBoostCheck"));
    settingsLayout->addWidget(m_speakerBoostCheck, 1, 2);
    rootLayout->addLayout(settingsLayout.release());

    auto takeList = std::make_unique<TakeListWidget>(this);
    m_takeListWidget = takeList.get();
    rootLayout->addWidget(takeList.release());

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("ScriptStatusLabel"));
    m_statusLabel->setWordWrap(true);

    setLayout(rootLayout.release());
    setFocusPolicy(Qt::StrongFocus);

    connect(m_importButton, &QPushButton::clicked, this, &ScriptViewerPanel::importScript);
    connect(m_confirmButton, &QPushButton::clicked, this, &ScriptViewerPanel::confirmCurrentLine);
    connect(m_generateButton, &QPushButton::clicked, this,
            &ScriptViewerPanel::generateCurrentLine);
    connect(m_scriptSelector, &QComboBox::currentIndexChanged, this,
            &ScriptViewerPanel::loadSelectedScriptLines);
    connect(m_lineList, &QListWidget::currentRowChanged, this,
            &ScriptViewerPanel::handleLineSelectionChanged);
    connect(m_parseWatcher.get(), &ParsedScriptWatcher::finished, this,
            &ScriptViewerPanel::finishParseScript);
    connect(m_importWatcher.get(), &ScriptImportWatcher::finished, this,
            &ScriptViewerPanel::finishImportScript);
    connect(m_generationWatcher.get(), &TtsGenerationWatcher::finished, this,
            &ScriptViewerPanel::finishGeneration);
    connect(m_takeListWidget, &TakeListWidget::playTakeRequested, this,
            &ScriptViewerPanel::playTake);
    connect(m_takeListWidget, &TakeListWidget::starTakeRequested, this,
            &ScriptViewerPanel::starTake);
    connect(m_takeListWidget, &TakeListWidget::deleteTakeRequested, this,
            &ScriptViewerPanel::deleteTake);

    setBusy(false);
    m_statusLabel->setText(QStringLiteral("Open or create a project to import scripts."));
}

void ScriptViewerPanel::setProject(std::optional<core::Project> project) {
    m_project = std::move(project);
    loadScripts();
}

bool ScriptViewerPanel::eventFilter(QObject* watched, QEvent* event) {
    if (watched != m_lineList || event->type() != QEvent::KeyPress) {
        return QWidget::eventFilter(watched, event);
    }

    const auto* keyEvent = static_cast<QKeyEvent*>(event);
    if (keyEvent->key() == Qt::Key_Up) {
        selectLine(std::max(0, m_lineList->currentRow() - 1));
        return true;
    }
    if (keyEvent->key() == Qt::Key_Down) {
        selectLine(std::min(m_lineList->count() - 1, m_lineList->currentRow() + 1));
        return true;
    }
    if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter) {
        confirmCurrentLine();
        return true;
    }

    return QWidget::eventFilter(watched, event);
}

void ScriptViewerPanel::importScript() {
    if (!m_project.has_value() || m_parseWatcher->isRunning() || m_importWatcher->isRunning()) {
        return;
    }

    const auto file = QFileDialog::getOpenFileName(
        this, QStringLiteral("Import Script"), QString{},
        QStringLiteral("Scripts (*.txt *.fountain *.rpy *.json *.csv)"));
    if (file.isEmpty()) {
        return;
    }

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Parsing script..."));
    const std::filesystem::path scriptPath{file.toStdString()};
    m_parseWatcher->setFuture(QtConcurrent::run([scriptPath]() {
        return io::scripts::importScriptFile(scriptPath);
    }));
}

void ScriptViewerPanel::finishParseScript() {
    auto parsed = m_parseWatcher->result();
    if (!parsed) {
        setBusy(false);
        m_statusLabel->setText(QString::fromStdString(parsed.error().message));
        return;
    }

    const auto speakers = detectedSpeakers(parsed.value());
    std::vector<db::CharacterAssignment> assignments;
    if (!speakers.empty()) {
        CharacterAssignmentDialog dialog{speakers, cachedVoices(), this};
        if (dialog.exec() != QDialog::Accepted) {
            setBusy(false);
            m_statusLabel->setText(QStringLiteral("Script import cancelled."));
            return;
        }
        assignments = dialog.assignments();
    }

    if (!m_project.has_value()) {
        setBusy(false);
        m_statusLabel->setText(QStringLiteral("No project is open."));
        return;
    }

    m_statusLabel->setText(QStringLiteral("Saving script..."));
    const auto projectRoot = m_project->rootPath();
    const auto parsedScript = std::move(parsed).value();
    m_importWatcher->setFuture(QtConcurrent::run([projectRoot, parsedScript, assignments]() {
        return importParsedScript(projectRoot, parsedScript, assignments);
    }));
}

void ScriptViewerPanel::finishImportScript() {
    const auto result = m_importWatcher->result();
    setBusy(false);
    if (!result.success) {
        m_statusLabel->setText(result.message);
        return;
    }

    loadScripts();
    for (int index = 0; index < m_scriptSelector->count(); ++index) {
        if (m_scriptSelector->itemData(index).toString().toStdString() ==
            result.importedScript.script.id) {
            m_scriptSelector->setCurrentIndex(index);
            break;
        }
    }
    setLines(result.importedScript.lines);
    m_statusLabel->setText(result.message);
    emit scriptImported();
}

void ScriptViewerPanel::generateCurrentLine() {
    const auto* line = currentLine();
    if (!m_project.has_value() || line == nullptr || m_generationWatcher->isRunning()) {
        return;
    }
    if (line->voiceId.empty()) {
        m_statusLabel->setText(QStringLiteral("Assign a voice to this line's character first."));
        return;
    }

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Generating TTS..."));
    const auto project = *m_project;
    const auto lineCopy = *line;
    const auto settings = currentVoiceSettings();
    m_generationWatcher->setFuture(QtConcurrent::run([this, project, lineCopy, settings]() {
        return generateTtsTake(project, lineCopy, settings, &m_audioEngine);
    }));
}

void ScriptViewerPanel::finishGeneration() {
    const auto result = m_generationWatcher->result();
    setBusy(false);
    if (!result.success) {
        m_statusLabel->setText(result.message);
        loadCurrentLineTakes();
        return;
    }

    if (auto* line = currentLine(); line != nullptr) {
        line->activeTakeId = result.take.id;
        line->voiceSettingsJson = core::voiceSettingsToJson(currentVoiceSettings());
    }
    loadCurrentLineTakes();
    m_statusLabel->setText(result.playbackWarning.isEmpty()
                               ? result.message
                               : QStringLiteral("%1 Playback: %2")
                                     .arg(result.message, result.playbackWarning));
}

void ScriptViewerPanel::loadScripts() {
    m_scriptSelector->blockSignals(true);
    m_scriptSelector->clear();
    m_scripts.clear();

    if (!m_project.has_value()) {
        setLines({});
        m_scriptSelector->setEnabled(false);
        m_scriptSelector->blockSignals(false);
        m_statusLabel->setText(QStringLiteral("Open or create a project to import scripts."));
        setBusy(false);
        return;
    }

    auto scripts = m_scriptRepository.listScripts(m_project->rootPath());
    if (!scripts) {
        setLines({});
        m_scriptSelector->blockSignals(false);
        m_statusLabel->setText(QString::fromStdString(scripts.error().message));
        return;
    }

    m_scripts = std::move(scripts).value();
    for (const auto& script : m_scripts) {
        const auto displayName =
            std::filesystem::path{script.sourcePath}.filename().empty()
                ? QString::fromStdString(script.id)
                : QString::fromStdString(
                      std::filesystem::path{script.sourcePath}.filename().string());
        m_scriptSelector->addItem(displayName, QString::fromStdString(script.id));
    }

    m_scriptSelector->setEnabled(!m_scripts.empty());
    m_scriptSelector->blockSignals(false);
    loadSelectedScriptLines();
    setBusy(false);
}

void ScriptViewerPanel::loadSelectedScriptLines() {
    if (!m_project.has_value() || m_scriptSelector->currentIndex() < 0) {
        setLines({});
        return;
    }

    const auto scriptId = m_scriptSelector->currentData().toString().toStdString();
    auto lines = m_scriptRepository.listLines(m_project->rootPath(), scriptId);
    if (!lines) {
        m_statusLabel->setText(QString::fromStdString(lines.error().message));
        return;
    }
    setLines(std::move(lines).value());
}

void ScriptViewerPanel::handleLineSelectionChanged() {
    const auto* line = currentLine();
    if (line != nullptr) {
        applyLineSettingsToControls(*line);
    }
    loadCurrentLineTakes();
    setBusy(false);
}

void ScriptViewerPanel::loadCurrentLineTakes() {
    if (m_takeListWidget == nullptr) {
        return;
    }

    const auto* line = currentLine();
    if (!m_project.has_value() || line == nullptr) {
        m_takeListWidget->setTakes({});
        return;
    }

    auto takes = m_takeRepository.listTakes(m_project->rootPath(), line->id);
    if (!takes) {
        m_takeListWidget->setTakes({});
        m_statusLabel->setText(QString::fromStdString(takes.error().message));
        return;
    }
    m_takeListWidget->setTakes(std::move(takes).value());
}

void ScriptViewerPanel::setBusy(const bool isBusy) {
    const bool hasProject = m_project.has_value();
    const auto* line = currentLine();
    m_importButton->setEnabled(hasProject && !isBusy);
    m_confirmButton->setEnabled(hasProject && !isBusy && !m_lines.empty());
    m_generateButton->setEnabled(hasProject && !isBusy && line != nullptr &&
                                 !line->voiceId.empty());
    m_scriptSelector->setEnabled(hasProject && !isBusy && !m_scripts.empty());
    m_stabilitySpin->setEnabled(hasProject && !isBusy && line != nullptr);
    m_similaritySpin->setEnabled(hasProject && !isBusy && line != nullptr);
    m_styleSpin->setEnabled(hasProject && !isBusy && line != nullptr);
    m_speakerBoostCheck->setEnabled(hasProject && !isBusy && line != nullptr);
}

void ScriptViewerPanel::setLines(std::vector<db::ScriptLineRecord> lines) {
    m_lines = std::move(lines);
    m_lineList->clear();
    for (const auto& line : m_lines) {
        auto item = std::make_unique<QListWidgetItem>(lineText(line));
        item->setData(Qt::UserRole, QString::fromStdString(line.id));
        if (line.characterName.empty()) {
            item->setForeground(QBrush{QColor{185, 45, 45}});
            item->setToolTip(QStringLiteral("No character is assigned to this line."));
        } else if (!line.sceneTag.empty()) {
            item->setToolTip(QString::fromStdString(line.sceneTag));
        }
        m_lineList->addItem(item.release());
    }

    if (!m_lines.empty()) {
        selectLine(0);
    } else {
        loadCurrentLineTakes();
    }
    setBusy(false);
}

void ScriptViewerPanel::selectLine(const int row) {
    if (row < 0 || row >= m_lineList->count()) {
        return;
    }
    m_lineList->setCurrentRow(row);
    m_lineList->scrollToItem(m_lineList->item(row), QAbstractItemView::PositionAtCenter);
}

void ScriptViewerPanel::confirmCurrentLine() {
    const int row = m_lineList->currentRow();
    if (row < 0 || row >= static_cast<int>(m_lines.size())) {
        return;
    }

    m_statusLabel->setText(QStringLiteral("Confirmed line %1 of %2.")
                               .arg(row + 1)
                               .arg(static_cast<int>(m_lines.size())));
}

void ScriptViewerPanel::playTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }

    const auto absolutePath = m_project->rootPath() / std::filesystem::path{take.filePath};
    auto played = m_audioEngine.playFile(absolutePath);
    if (!played) {
        m_statusLabel->setText(QString::fromStdString(played.error().message));
        return;
    }
    m_statusLabel->setText(QStringLiteral("Playing take."));
}

void ScriptViewerPanel::starTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }

    auto starred = m_takeRepository.setActiveTake(m_project->rootPath(), take.lineId, take.id);
    if (!starred) {
        m_statusLabel->setText(QString::fromStdString(starred.error().message));
        return;
    }
    if (auto* line = currentLine(); line != nullptr) {
        line->activeTakeId = take.id;
    }
    loadCurrentLineTakes();
    m_statusLabel->setText(QStringLiteral("Active take updated."));
}

void ScriptViewerPanel::deleteTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }

    const auto answer =
        QMessageBox::question(this,
                              QStringLiteral("Delete Take"),
                              QStringLiteral("Delete this take from the project?"));
    if (answer != QMessageBox::Yes) {
        return;
    }

    auto deleted = m_takeRepository.deleteTake(m_project->rootPath(), take.lineId, take.id);
    if (!deleted) {
        m_statusLabel->setText(QString::fromStdString(deleted.error().message));
        return;
    }
    if (auto* line = currentLine(); line != nullptr && line->activeTakeId == take.id) {
        line->activeTakeId.clear();
    }
    loadCurrentLineTakes();
    m_statusLabel->setText(QStringLiteral("Take deleted."));
}

void ScriptViewerPanel::applyLineSettingsToControls(const db::ScriptLineRecord& line) {
    auto settings = core::voiceSettingsFromJson(line.voiceSettingsJson);
    const auto values = settings ? settings.value() : core::defaultVoiceSettings();
    m_stabilitySpin->setValue(values.stability);
    m_similaritySpin->setValue(values.similarityBoost);
    m_styleSpin->setValue(values.style);
    m_speakerBoostCheck->setChecked(values.useSpeakerBoost);
}

core::VoiceSettings ScriptViewerPanel::currentVoiceSettings() const {
    core::VoiceSettings settings;
    settings.stability = m_stabilitySpin->value();
    settings.similarityBoost = m_similaritySpin->value();
    settings.style = m_styleSpin->value();
    settings.useSpeakerBoost = m_speakerBoostCheck->isChecked();
    return settings;
}

db::ScriptLineRecord* ScriptViewerPanel::currentLine() {
    const int row = m_lineList->currentRow();
    if (row < 0 || row >= static_cast<int>(m_lines.size())) {
        return nullptr;
    }
    return &m_lines[static_cast<std::size_t>(row)];
}

const db::ScriptLineRecord* ScriptViewerPanel::currentLine() const {
    return const_cast<ScriptViewerPanel*>(this)->currentLine();
}

std::vector<std::string>
ScriptViewerPanel::detectedSpeakers(const io::scripts::ParsedScript& script) const {
    std::set<std::string> speakers;
    for (const auto& line : script.lines) {
        if (!line.speaker.empty()) {
            speakers.insert(line.speaker);
        }
    }
    return {speakers.begin(), speakers.end()};
}

std::vector<db::VoiceRecord> ScriptViewerPanel::cachedVoices() const {
    if (!m_project.has_value()) {
        return {};
    }

    auto voices = m_voiceRepository.listVoices(m_project->rootPath());
    if (!voices) {
        return {};
    }
    return std::move(voices).value();
}

} // namespace voxstudio::ui
