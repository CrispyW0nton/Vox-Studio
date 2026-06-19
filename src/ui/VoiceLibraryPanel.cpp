#include "ui/VoiceLibraryPanel.h"

#include "secrets/DpapiVault.h"
#include "ui/CloneVoiceDialog.h"

#include <QAbstractItemView>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

namespace voxstudio::ui {
namespace {

template <typename TWidget, typename... TArgs>
[[nodiscard]] TWidget* addOwnedWidget(QLayout& layout, TArgs&&... args) {
    auto widget = std::make_unique<TWidget>(std::forward<TArgs>(args)...);
    auto* widgetPointer = widget.get();
    layout.addWidget(widget.release());
    return widgetPointer;
}

[[nodiscard]] QString apiErrorText(const net::elevenlabs::ApiError& error) {
    if (error.statusCode > 0) {
        return QStringLiteral("%1 (HTTP %2)")
            .arg(QString::fromStdString(error.message))
            .arg(error.statusCode);
    }

    return QString::fromStdString(error.message);
}

[[nodiscard]] std::string utcTimestampNow() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);

    std::tm utcTime{};
    gmtime_s(&utcTime, &time);

    std::ostringstream stream;
    stream << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

[[nodiscard]] std::string originFromCategory(const std::string& category) {
    if (category == "premade") {
        return "premade";
    }
    if (category == "cloned" || category == "generated") {
        return "ivc";
    }
    if (category == "professional") {
        return "pvc";
    }
    return "shared";
}

[[nodiscard]] std::string labelsJson(const std::map<std::string, std::string>& labels) {
    nlohmann::json labelsObject = nlohmann::json::object();
    for (const auto& [key, value] : labels) {
        labelsObject[key] = value;
    }
    return labelsObject.dump();
}

[[nodiscard]] std::map<std::string, std::string> parseLabelsJson(const std::string& labelsText) {
    if (labelsText.empty()) {
        return {};
    }

    try {
        const auto json = nlohmann::json::parse(labelsText);
        if (!json.is_object()) {
            return {};
        }

        std::map<std::string, std::string> labels;
        for (const auto& item : json.items()) {
            if (item.value().is_string()) {
                labels.emplace(item.key(), item.value().get<std::string>());
            }
        }
        return labels;
    } catch (const nlohmann::json::exception&) {
        return {};
    }
}

[[nodiscard]] db::VoiceRecord voiceRecordFromInfo(const net::elevenlabs::VoiceInfo& voice,
                                                  const std::string& consentConfirmedAt) {
    return db::VoiceRecord{voice.voiceId,
                           voice.name,
                           originFromCategory(voice.category),
                           labelsJson(voice.labels),
                           voice.defaultSettingsJson.empty() ? "{}" : voice.defaultSettingsJson,
                           consentConfirmedAt,
                           utcTimestampNow()};
}

[[nodiscard]] std::vector<db::VoiceRecord>
voiceRecordsFromInfos(const std::vector<net::elevenlabs::VoiceInfo>& voices) {
    std::vector<db::VoiceRecord> records;
    records.reserve(voices.size());
    for (const auto& voice : voices) {
        records.push_back(voiceRecordFromInfo(voice, {}));
    }
    return records;
}

[[nodiscard]] net::elevenlabs::VoiceInfo voiceInfoFromRecord(const db::VoiceRecord& record) {
    net::elevenlabs::VoiceInfo voice;
    voice.voiceId = record.id;
    voice.name = record.name;
    voice.category = record.origin;
    voice.labels = parseLabelsJson(record.labelsJson);
    voice.defaultSettingsJson = record.defaultSettingsJson;
    return voice;
}

[[nodiscard]] QString voiceDisplayText(const net::elevenlabs::VoiceInfo& voice) {
    QString category = QString::fromStdString(voice.category.empty() ? "unknown" : voice.category);
    return QStringLiteral("%1  [%2]").arg(QString::fromStdString(voice.name), category);
}

[[nodiscard]] core::Expected<std::string> loadApiKey() {
    const secrets::DpapiVault vault;
    return vault.loadElevenLabsApiKey();
}

[[nodiscard]] VoiceLibraryTaskResult refreshCloudVoices(const core::Project& project) {
    auto apiKey = loadApiKey();
    if (!apiKey) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Refresh, false,
                                      QString::fromStdString(apiKey.error().message), {}};
    }

    const net::elevenlabs::VoicesApi voicesApi{std::move(apiKey).value()};
    auto voices = voicesApi.listVoices();
    if (!voices) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Refresh, false,
                                      apiErrorText(voices.error()), {}};
    }

    const db::VoiceRepository repository;
    auto cached = repository.replaceVoices(project.rootPath(),
                                           voiceRecordsFromInfos(voices.value().voices));
    if (!cached) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Refresh, false,
                                      QString::fromStdString(cached.error().message), {}};
    }

    return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Refresh,
                                  true,
                                  QStringLiteral("Voice library refreshed."),
                                  std::move(voices).value().voices};
}

[[nodiscard]] VoiceLibraryTaskResult cloneCloudVoice(
    const core::Project& project, const net::elevenlabs::VoiceCloneRequest& request) {
    auto apiKey = loadApiKey();
    if (!apiKey) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Clone, false,
                                      QString::fromStdString(apiKey.error().message), {}};
    }

    const net::elevenlabs::VoicesApi voicesApi{std::move(apiKey).value()};
    auto created = voicesApi.addVoice(request);
    if (!created) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Clone, false,
                                      apiErrorText(created.error()), {}};
    }

    net::elevenlabs::VoiceInfo createdVoice;
    auto fetchedVoice = voicesApi.getVoice(created.value().voiceId);
    if (fetchedVoice) {
        createdVoice = std::move(fetchedVoice).value();
    } else {
        createdVoice.voiceId = created.value().voiceId;
        createdVoice.name = request.name;
        createdVoice.category = "cloned";
        createdVoice.description = request.description;
        createdVoice.labels = request.labels;
    }

    const db::VoiceRepository repository;
    auto cached = repository.upsertVoice(project.rootPath(),
                                         voiceRecordFromInfo(createdVoice, utcTimestampNow()));
    if (!cached) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Clone, false,
                                      QString::fromStdString(cached.error().message), {}};
    }

    return refreshCloudVoices(project);
}

[[nodiscard]] VoiceLibraryTaskResult editCloudVoice(
    const core::Project& project,
    const std::string& voiceId,
    const net::elevenlabs::VoiceEditRequest& request) {
    auto apiKey = loadApiKey();
    if (!apiKey) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Edit, false,
                                      QString::fromStdString(apiKey.error().message), {}};
    }

    const net::elevenlabs::VoicesApi voicesApi{std::move(apiKey).value()};
    auto edited = voicesApi.editVoice(voiceId, request);
    if (!edited) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Edit, false,
                                      apiErrorText(edited.error()), {}};
    }

    auto voice = voicesApi.getVoice(voiceId);
    if (voice) {
        const db::VoiceRepository repository;
        auto cached =
            repository.upsertVoice(project.rootPath(), voiceRecordFromInfo(voice.value(), {}));
        if (!cached) {
            return VoiceLibraryTaskResult{
                VoiceLibraryTaskKind::Edit, false,
                QString::fromStdString(cached.error().message), {}};
        }
    }

    return refreshCloudVoices(project);
}

[[nodiscard]] VoiceLibraryTaskResult deleteCloudVoice(const core::Project& project,
                                                      const std::string& voiceId) {
    auto apiKey = loadApiKey();
    if (!apiKey) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Delete, false,
                                      QString::fromStdString(apiKey.error().message), {}};
    }

    const net::elevenlabs::VoicesApi voicesApi{std::move(apiKey).value()};
    auto deleted = voicesApi.deleteVoice(voiceId);
    if (!deleted) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Delete, false,
                                      apiErrorText(deleted.error()), {}};
    }

    const db::VoiceRepository repository;
    auto removed = repository.deleteVoice(project.rootPath(), voiceId);
    if (!removed) {
        return VoiceLibraryTaskResult{VoiceLibraryTaskKind::Delete, false,
                                      QString::fromStdString(removed.error().message), {}};
    }

    return refreshCloudVoices(project);
}

[[nodiscard]] std::string safePreviewFileName(std::string voiceId) {
    for (char& character : voiceId) {
        const auto value = static_cast<unsigned char>(character);
        if (!std::isalnum(value) && character != '-' && character != '_') {
            character = '_';
        }
    }
    return voiceId.empty() ? "voice_preview" : voiceId;
}

[[nodiscard]] VoicePreviewTaskResult downloadPreview(const net::elevenlabs::VoiceInfo& voice) {
    if (voice.previewUrl.empty()) {
        return VoicePreviewTaskResult{false, QStringLiteral("No preview URL is available."), {}};
    }

    const auto response =
        cpr::Get(cpr::Url{voice.previewUrl}, cpr::Timeout{std::chrono::seconds{20}});
    if (response.error.code != cpr::ErrorCode::OK) {
        return VoicePreviewTaskResult{false, QString::fromStdString(response.error.message), {}};
    }
    if (response.status_code < 200 || response.status_code >= 300) {
        return VoicePreviewTaskResult{
            false,
            QStringLiteral("Preview download failed (HTTP %1).").arg(response.status_code), {}};
    }

    try {
        const auto directory = std::filesystem::temp_directory_path() / "VoxStudioPreviews";
        std::filesystem::create_directories(directory);
        const auto filePath = directory / (safePreviewFileName(voice.voiceId) + ".mp3");
        std::ofstream output{filePath, std::ios::binary | std::ios::trunc};
        output.write(response.text.data(), static_cast<std::streamsize>(response.text.size()));
        if (!output) {
            return VoicePreviewTaskResult{
                false, QStringLiteral("Preview file could not be written."), {}};
        }
        return VoicePreviewTaskResult{true, QString{}, filePath};
    } catch (const std::exception& exception) {
        return VoicePreviewTaskResult{false, QString::fromStdString(exception.what()), {}};
    }
}

} // namespace

VoiceLibraryPanel::VoiceLibraryPanel(QWidget* parent)
    : QWidget(parent)
    , m_voiceTaskWatcher(std::make_unique<QFutureWatcher<VoiceLibraryTaskResult>>())
    , m_previewTaskWatcher(std::make_unique<QFutureWatcher<VoicePreviewTaskResult>>()) {
    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto* titleLabel = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Voice Library"));
    titleLabel->setObjectName(QStringLiteral("VoiceLibraryTitle"));

    auto buttonLayout = std::make_unique<QHBoxLayout>();
    m_refreshButton = addOwnedWidget<QPushButton>(*buttonLayout, QStringLiteral("Refresh"));
    m_cloneButton = addOwnedWidget<QPushButton>(*buttonLayout, QStringLiteral("Clone New Voice"));
    m_previewButton = addOwnedWidget<QPushButton>(*buttonLayout, QStringLiteral("Preview"));
    m_editButton = addOwnedWidget<QPushButton>(*buttonLayout, QStringLiteral("Edit"));
    m_deleteButton = addOwnedWidget<QPushButton>(*buttonLayout, QStringLiteral("Delete"));
    rootLayout->addLayout(buttonLayout.release());

    m_voiceList = addOwnedWidget<QListWidget>(*rootLayout);
    m_voiceList->setSelectionMode(QAbstractItemView::SingleSelection);

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setWordWrap(true);

    setLayout(rootLayout.release());

    connect(m_refreshButton, &QPushButton::clicked, this, &VoiceLibraryPanel::refreshVoices);
    connect(m_cloneButton, &QPushButton::clicked, this, &VoiceLibraryPanel::cloneVoice);
    connect(m_previewButton, &QPushButton::clicked, this, &VoiceLibraryPanel::previewSelectedVoice);
    connect(m_editButton, &QPushButton::clicked, this, &VoiceLibraryPanel::editSelectedVoice);
    connect(m_deleteButton, &QPushButton::clicked, this, &VoiceLibraryPanel::deleteSelectedVoice);
    connect(m_voiceTaskWatcher.get(), &QFutureWatcher<VoiceLibraryTaskResult>::finished, this,
            &VoiceLibraryPanel::finishVoiceTask);
    connect(m_previewTaskWatcher.get(), &QFutureWatcher<VoicePreviewTaskResult>::finished, this,
            &VoiceLibraryPanel::finishPreviewTask);

    setBusy(false);
    m_statusLabel->setText(QStringLiteral("Open or create a project to manage voices."));
}

void VoiceLibraryPanel::setProject(std::optional<core::Project> project) {
    m_project = std::move(project);
    loadCachedVoices();
    refreshVoices();
}

void VoiceLibraryPanel::refreshVoices() {
    if (!m_project.has_value()) {
        setVoices({});
        m_statusLabel->setText(QStringLiteral("Open or create a project to manage voices."));
        setBusy(false);
        return;
    }
    if (m_voiceTaskWatcher->isRunning()) {
        return;
    }

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Refreshing voice library..."));
    const auto project = *m_project;
    m_voiceTaskWatcher->setFuture(QtConcurrent::run([project]() {
        return refreshCloudVoices(project);
    }));
}

void VoiceLibraryPanel::cloneVoice() {
    if (!m_project.has_value() || m_voiceTaskWatcher->isRunning()) {
        return;
    }

    CloneVoiceDialog dialog{this};
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    const auto dialogResult = dialog.result();
    net::elevenlabs::VoiceCloneRequest request;
    request.name = dialogResult.name;
    request.description = dialogResult.description;
    request.labels = dialogResult.labels;
    request.sampleFiles = dialogResult.sampleFiles;
    request.removeBackgroundNoise = dialogResult.removeBackgroundNoise;

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Cloning voice..."));
    const auto project = *m_project;
    m_voiceTaskWatcher->setFuture(QtConcurrent::run([project, request]() {
        return cloneCloudVoice(project, request);
    }));
}

void VoiceLibraryPanel::editSelectedVoice() {
    const auto voice = selectedVoice();
    if (!m_project.has_value() || !voice.has_value() || m_voiceTaskWatcher->isRunning()) {
        return;
    }

    QDialog dialog{this};
    dialog.setWindowTitle(QStringLiteral("Edit Voice"));
    auto layout = std::make_unique<QVBoxLayout>();
    auto* nameEdit = addOwnedWidget<QLineEdit>(*layout);
    nameEdit->setText(QString::fromStdString(voice->name));
    auto* descriptionEdit = addOwnedWidget<QTextEdit>(*layout);
    descriptionEdit->setPlainText(QString::fromStdString(voice->description));
    descriptionEdit->setMaximumHeight(90);
    auto buttons =
        std::make_unique<QDialogButtonBox>(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons.get(), &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons.get(), &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons.release());
    dialog.setLayout(layout.release());
    if (dialog.exec() != QDialog::Accepted || nameEdit->text().trimmed().isEmpty()) {
        return;
    }

    net::elevenlabs::VoiceEditRequest request;
    request.name = nameEdit->text().trimmed().toStdString();
    request.description = descriptionEdit->toPlainText().trimmed().toStdString();
    request.labels = voice->labels;

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Updating voice..."));
    const auto project = *m_project;
    const auto voiceId = voice->voiceId;
    m_voiceTaskWatcher->setFuture(QtConcurrent::run([project, voiceId, request]() {
        return editCloudVoice(project, voiceId, request);
    }));
}

void VoiceLibraryPanel::deleteSelectedVoice() {
    const auto voice = selectedVoice();
    if (!m_project.has_value() || !voice.has_value() || m_voiceTaskWatcher->isRunning()) {
        return;
    }

    const auto answer = QMessageBox::question(
        this, QStringLiteral("Delete Voice"),
        QStringLiteral("Delete %1 from ElevenLabs and the local cache?")
            .arg(QString::fromStdString(voice->name)));
    if (answer != QMessageBox::Yes) {
        return;
    }

    setBusy(true);
    m_statusLabel->setText(QStringLiteral("Deleting voice..."));
    const auto project = *m_project;
    const auto voiceId = voice->voiceId;
    m_voiceTaskWatcher->setFuture(QtConcurrent::run([project, voiceId]() {
        return deleteCloudVoice(project, voiceId);
    }));
}

void VoiceLibraryPanel::previewSelectedVoice() {
    const auto voice = selectedVoice();
    if (!voice.has_value() || m_previewTaskWatcher->isRunning()) {
        return;
    }

    m_statusLabel->setText(QStringLiteral("Loading preview..."));
    m_previewTaskWatcher->setFuture(QtConcurrent::run([voice]() {
        return downloadPreview(*voice);
    }));
}

void VoiceLibraryPanel::finishVoiceTask() {
    const auto result = m_voiceTaskWatcher->result();
    setBusy(false);

    if (!result.success) {
        m_statusLabel->setText(result.message);
        loadCachedVoices();
        return;
    }

    setVoices(result.voices);
    m_statusLabel->setText(result.message);
    emit voiceCacheChanged();
}

void VoiceLibraryPanel::finishPreviewTask() {
    const auto result = m_previewTaskWatcher->result();
    if (!result.success) {
        m_statusLabel->setText(result.message);
        return;
    }

    auto played = m_previewPlayer.playFile(result.filePath);
    if (!played) {
        m_statusLabel->setText(QString::fromStdString(played.error().message));
        return;
    }
    m_statusLabel->setText(QStringLiteral("Preview playing."));
}

void VoiceLibraryPanel::setBusy(const bool isBusy) {
    const bool hasProject = m_project.has_value();
    m_refreshButton->setEnabled(hasProject && !isBusy);
    m_cloneButton->setEnabled(hasProject && !isBusy);
    m_previewButton->setEnabled(hasProject && !isBusy);
    m_editButton->setEnabled(hasProject && !isBusy);
    m_deleteButton->setEnabled(hasProject && !isBusy);
}

void VoiceLibraryPanel::setVoices(std::vector<net::elevenlabs::VoiceInfo> voices) {
    m_voices = std::move(voices);
    m_voiceList->clear();
    for (const auto& voice : m_voices) {
        auto item = std::make_unique<QListWidgetItem>(voiceDisplayText(voice));
        item->setData(Qt::UserRole, QString::fromStdString(voice.voiceId));
        m_voiceList->addItem(item.release());
    }
}

void VoiceLibraryPanel::loadCachedVoices() {
    if (!m_project.has_value()) {
        setVoices({});
        return;
    }

    auto cachedVoices = m_voiceRepository.listVoices(m_project->rootPath());
    if (!cachedVoices) {
        m_statusLabel->setText(QString::fromStdString(cachedVoices.error().message));
        return;
    }

    std::vector<net::elevenlabs::VoiceInfo> voices;
    voices.reserve(cachedVoices.value().size());
    for (const auto& record : cachedVoices.value()) {
        voices.push_back(voiceInfoFromRecord(record));
    }
    setVoices(std::move(voices));
    m_statusLabel->setText(
        QStringLiteral("%1 cached voices.").arg(static_cast<int>(m_voices.size())));
}

std::optional<net::elevenlabs::VoiceInfo> VoiceLibraryPanel::selectedVoice() const {
    const auto selectedItems = m_voiceList->selectedItems();
    if (selectedItems.isEmpty()) {
        return std::nullopt;
    }

    const auto voiceId = selectedItems.front()->data(Qt::UserRole).toString().toStdString();
    const auto voice = std::ranges::find_if(m_voices, [&voiceId](const auto& item) {
        return item.voiceId == voiceId;
    });
    if (voice == m_voices.end()) {
        return std::nullopt;
    }
    return *voice;
}

} // namespace voxstudio::ui
