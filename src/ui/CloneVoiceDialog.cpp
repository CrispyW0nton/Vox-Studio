#include "ui/CloneVoiceDialog.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDragEnterEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMimeData>
#include <QPushButton>
#include <QTextEdit>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <sstream>
#include <utility>

namespace voxstudio::ui {
namespace {

constexpr int kMaxSampleFiles = 25;
constexpr double kMaxTotalDurationSeconds = 11.0 * 60.0;

template <typename TWidget, typename... TArgs>
[[nodiscard]] TWidget* addOwnedWidget(QLayout& layout, TArgs&&... args) {
    auto widget = std::make_unique<TWidget>(std::forward<TArgs>(args)...);
    auto* widgetPointer = widget.get();
    layout.addWidget(widget.release());
    return widgetPointer;
}

[[nodiscard]] std::string lowerExtension(const std::filesystem::path& filePath) {
    auto extension = filePath.extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

[[nodiscard]] bool isSupportedAudioFile(const std::filesystem::path& filePath) {
    const auto extension = lowerExtension(filePath);
    return extension == ".wav" || extension == ".mp3" || extension == ".flac" ||
           extension == ".ogg";
}

[[nodiscard]] QString durationText(const double seconds) {
    const auto roundedSeconds = static_cast<int>(std::lround(seconds));
    return QStringLiteral("%1:%2")
        .arg(roundedSeconds / 60)
        .arg(roundedSeconds % 60, 2, 10, QLatin1Char('0'));
}

[[nodiscard]] std::string trimCopy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

} // namespace

CloneVoiceDialog::CloneVoiceDialog(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(QStringLiteral("Clone New Voice"));
    setModal(true);
    setAcceptDrops(true);

    auto rootLayout = std::make_unique<QVBoxLayout>();

    m_nameEdit = addOwnedWidget<QLineEdit>(*rootLayout);
    m_nameEdit->setPlaceholderText(QStringLiteral("Voice name"));

    m_descriptionEdit = addOwnedWidget<QTextEdit>(*rootLayout);
    m_descriptionEdit->setPlaceholderText(QStringLiteral("Description"));
    m_descriptionEdit->setMaximumHeight(90);

    m_labelsEdit = addOwnedWidget<QLineEdit>(*rootLayout);
    m_labelsEdit->setPlaceholderText(QStringLiteral("Labels, e.g. accent=neutral, age=adult"));

    auto fileButtonLayout = std::make_unique<QHBoxLayout>();
    auto browseButton = std::make_unique<QPushButton>(QStringLiteral("Add Audio"));
    connect(browseButton.get(), &QPushButton::clicked, this, &CloneVoiceDialog::browseFiles);
    fileButtonLayout->addWidget(browseButton.release());

    auto previewButton = std::make_unique<QPushButton>(QStringLiteral("Preview"));
    connect(previewButton.get(), &QPushButton::clicked, this,
            &CloneVoiceDialog::previewSelectedFile);
    fileButtonLayout->addWidget(previewButton.release());

    auto removeButton = std::make_unique<QPushButton>(QStringLiteral("Remove"));
    connect(removeButton.get(), &QPushButton::clicked, this,
            &CloneVoiceDialog::removeSelectedFiles);
    fileButtonLayout->addWidget(removeButton.release());
    rootLayout->addLayout(fileButtonLayout.release());

    m_filesList = addOwnedWidget<QListWidget>(*rootLayout);
    m_filesList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    m_removeNoiseCheck =
        addOwnedWidget<QCheckBox>(*rootLayout, QStringLiteral("Remove background noise"));
    m_consentCheck = addOwnedWidget<QCheckBox>(
        *rootLayout, QStringLiteral("I confirm I have the right to clone this voice"));

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setWordWrap(true);

    auto buttons =
        std::make_unique<QDialogButtonBox>(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    m_cloneButton = buttons->button(QDialogButtonBox::Ok);
    m_cloneButton->setText(QStringLiteral("Clone"));
    m_cloneButton->setEnabled(false);
    connect(buttons.get(), &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons.get(), &QDialogButtonBox::rejected, this, &QDialog::reject);
    rootLayout->addWidget(buttons.release());

    connect(m_nameEdit, &QLineEdit::textChanged, this, &CloneVoiceDialog::updateValidation);
    connect(m_labelsEdit, &QLineEdit::textChanged, this, &CloneVoiceDialog::updateValidation);
    connect(m_consentCheck, &QCheckBox::toggled, this, &CloneVoiceDialog::updateValidation);

    setLayout(rootLayout.release());
    resize(540, 520);
    updateValidation();
}

CloneVoiceDialogResult CloneVoiceDialog::result() const {
    return CloneVoiceDialogResult{m_nameEdit->text().trimmed().toStdString(),
                                  m_descriptionEdit->toPlainText().trimmed().toStdString(),
                                  parsedLabels(),
                                  sampleFiles(),
                                  m_removeNoiseCheck->isChecked()};
}

void CloneVoiceDialog::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    }
}

void CloneVoiceDialog::dropEvent(QDropEvent* event) {
    std::vector<std::filesystem::path> files;
    for (const auto& url : event->mimeData()->urls()) {
        if (url.isLocalFile()) {
            files.push_back(std::filesystem::path{url.toLocalFile().toStdString()});
        }
    }
    addFiles(files);
    event->acceptProposedAction();
}

void CloneVoiceDialog::browseFiles() {
    const auto files = QFileDialog::getOpenFileNames(
        this, QStringLiteral("Add Voice Samples"), QString{},
        QStringLiteral("Audio Files (*.wav *.mp3 *.flac *.ogg)"));
    std::vector<std::filesystem::path> filePaths;
    filePaths.reserve(static_cast<std::size_t>(files.size()));
    for (const auto& file : files) {
        filePaths.push_back(std::filesystem::path{file.toStdString()});
    }
    addFiles(filePaths);
}

void CloneVoiceDialog::removeSelectedFiles() {
    const auto selectedItems = m_filesList->selectedItems();
    for (auto* item : selectedItems) {
        const std::unique_ptr<QListWidgetItem> removedItem{
            m_filesList->takeItem(m_filesList->row(item))};
    }
    updateValidation();
}

void CloneVoiceDialog::previewSelectedFile() {
    const auto selectedItems = m_filesList->selectedItems();
    if (selectedItems.isEmpty()) {
        m_statusLabel->setText(QStringLiteral("Select a sample to preview."));
        return;
    }

    const auto filePath =
        std::filesystem::path{selectedItems.front()->data(Qt::UserRole).toString().toStdString()};
    auto played = m_previewPlayer.playFile(filePath);
    if (!played) {
        m_statusLabel->setText(QString::fromStdString(played.error().message));
    }
}

void CloneVoiceDialog::updateValidation() {
    QString errorMessage;
    m_totalDurationSeconds = 0.0;

    if (m_nameEdit->text().trimmed().isEmpty()) {
        errorMessage = QStringLiteral("Voice name is required.");
    }

    if (m_filesList->count() == 0 && errorMessage.isEmpty()) {
        errorMessage = QStringLiteral("Add at least one audio sample.");
    }

    if (m_filesList->count() > kMaxSampleFiles && errorMessage.isEmpty()) {
        errorMessage = QStringLiteral("Use 25 or fewer audio samples.");
    }

    for (int index = 0; index < m_filesList->count() && errorMessage.isEmpty(); ++index) {
        const auto fileData = m_filesList->item(index)->data(Qt::UserRole).toString();
        const auto filePath = std::filesystem::path{fileData.toStdString()};
        if (!isSupportedAudioFile(filePath)) {
            errorMessage = QStringLiteral("Unsupported audio file type.");
            break;
        }

        auto duration = audio::audioDurationSeconds(filePath);
        if (!duration) {
            errorMessage = QString::fromStdString(duration.error().message);
            break;
        }
        m_totalDurationSeconds += duration.value();
    }

    if (m_totalDurationSeconds > kMaxTotalDurationSeconds && errorMessage.isEmpty()) {
        errorMessage = QStringLiteral("Total sample duration must be 11 minutes or less.");
    }

    if (m_consentCheck != nullptr && !m_consentCheck->isChecked() && errorMessage.isEmpty()) {
        errorMessage = QStringLiteral("Consent is required before upload.");
    }

    if (errorMessage.isEmpty()) {
        m_statusLabel->setText(
            QStringLiteral("%1 files, %2 total.")
                .arg(m_filesList->count())
                .arg(durationText(m_totalDurationSeconds)));
        m_cloneButton->setEnabled(true);
    } else {
        m_statusLabel->setText(errorMessage);
        m_cloneButton->setEnabled(false);
    }
}

void CloneVoiceDialog::addFiles(const std::vector<std::filesystem::path>& files) {
    for (const auto& file : files) {
        const auto absoluteFile = std::filesystem::absolute(file);
        bool alreadyAdded = false;
        for (int index = 0; index < m_filesList->count(); ++index) {
            const auto existingData =
                m_filesList->item(index)->data(Qt::UserRole).toString();
            const auto existing = std::filesystem::path{existingData.toStdString()};
            if (existing == absoluteFile) {
                alreadyAdded = true;
                break;
            }
        }
        if (alreadyAdded) {
            continue;
        }

        auto item = std::make_unique<QListWidgetItem>(
            QString::fromStdString(absoluteFile.filename().string()));
        item->setData(Qt::UserRole, QString::fromStdString(absoluteFile.string()));
        m_filesList->addItem(item.release());
    }

    updateValidation();
}

std::map<std::string, std::string> CloneVoiceDialog::parsedLabels() const {
    std::map<std::string, std::string> labels;
    const auto text = m_labelsEdit->text().toStdString();
    std::stringstream stream{text};
    std::string segment;
    while (std::getline(stream, segment, ',')) {
        const auto equals = segment.find('=');
        if (equals == std::string::npos) {
            continue;
        }

        const auto key = trimCopy(segment.substr(0, equals));
        const auto value = trimCopy(segment.substr(equals + 1));
        if (!key.empty() && !value.empty()) {
            labels.emplace(key, value);
        }
    }
    return labels;
}

std::vector<std::filesystem::path> CloneVoiceDialog::sampleFiles() const {
    std::vector<std::filesystem::path> files;
    files.reserve(static_cast<std::size_t>(m_filesList->count()));
    for (int index = 0; index < m_filesList->count(); ++index) {
        const auto fileData = m_filesList->item(index)->data(Qt::UserRole).toString();
        files.push_back(std::filesystem::path{fileData.toStdString()});
    }
    return files;
}

} // namespace voxstudio::ui
