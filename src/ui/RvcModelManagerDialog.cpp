#include "ui/RvcModelManagerDialog.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QVBoxLayout>

#include <filesystem>
#include <memory>
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

[[nodiscard]] QString modelLabel(const rvc::RvcModelRecord& model) {
    return QStringLiteral("%1 (%2)")
        .arg(QString::fromStdString(model.displayName), QString::fromStdString(model.id));
}

} // namespace

RvcModelManagerDialog::RvcModelManagerDialog(QWidget* parent)
    : RvcModelManagerDialog(std::nullopt, parent) {}

RvcModelManagerDialog::RvcModelManagerDialog(std::optional<core::Project> project,
                                             QWidget* parent)
    : QDialog(parent)
    , m_project(std::move(project)) {
    setWindowTitle(QStringLiteral("RVC Models"));
    setModal(true);

    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto title = std::make_unique<QLabel>(QStringLiteral("Local RVC Models"));
    title->setObjectName(QStringLiteral("RvcModelManagerTitle"));
    rootLayout->addWidget(title.release());

    m_modelList = addOwnedWidget<QListWidget>(*rootLayout);
    m_modelList->setObjectName(QStringLiteral("RvcModelList"));

    auto assignmentLayout = std::make_unique<QHBoxLayout>();
    assignmentLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Character")).release());
    m_characterCombo = addOwnedWidget<QComboBox>(*assignmentLayout);
    m_characterCombo->setObjectName(QStringLiteral("RvcCharacterCombo"));
    m_assignButton = addOwnedWidget<QPushButton>(*assignmentLayout, QStringLiteral("Assign"));
    m_assignButton->setObjectName(QStringLiteral("RvcAssignButton"));
    rootLayout->addLayout(assignmentLayout.release());

    auto actionLayout = std::make_unique<QHBoxLayout>();
    m_importButton = addOwnedWidget<QPushButton>(*actionLayout, QStringLiteral("Import"));
    m_importButton->setObjectName(QStringLiteral("RvcImportButton"));
    m_deleteButton = addOwnedWidget<QPushButton>(*actionLayout, QStringLiteral("Delete"));
    m_deleteButton->setObjectName(QStringLiteral("RvcDeleteButton"));
    m_refreshButton = addOwnedWidget<QPushButton>(*actionLayout, QStringLiteral("Refresh"));
    m_refreshButton->setObjectName(QStringLiteral("RvcRefreshButton"));
    rootLayout->addLayout(actionLayout.release());

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("RvcModelStatusLabel"));
    m_statusLabel->setWordWrap(true);

    auto buttons = std::make_unique<QDialogButtonBox>(QDialogButtonBox::Close);
    connect(buttons->button(QDialogButtonBox::Close), &QPushButton::clicked, this,
            &QDialog::reject);
    rootLayout->addWidget(buttons.release());

    connect(m_importButton, &QPushButton::clicked, this,
            &RvcModelManagerDialog::importModel);
    connect(m_deleteButton, &QPushButton::clicked, this,
            &RvcModelManagerDialog::deleteSelectedModel);
    connect(m_refreshButton, &QPushButton::clicked, this, &RvcModelManagerDialog::refresh);
    connect(m_assignButton, &QPushButton::clicked, this,
            &RvcModelManagerDialog::assignSelectedModel);

    setLayout(rootLayout.release());
    resize(620, 420);
    refresh();
}

void RvcModelManagerDialog::refresh() {
    m_modelList->clear();
    m_characterCombo->clear();
    m_models.clear();
    m_characters.clear();

    auto models = m_registry.listModels();
    if (!models) {
        setStatusText(QString::fromStdString(models.error().message));
        return;
    }

    m_models = std::move(models).value();
    for (const auto& model : m_models) {
        auto item = std::make_unique<QListWidgetItem>(modelLabel(model));
        item->setData(Qt::UserRole, QString::fromStdString(model.id));
        m_modelList->addItem(item.release());
    }

    if (m_project.has_value()) {
        auto characters = m_scriptRepository.listCharacters(m_project->rootPath());
        if (!characters) {
            setStatusText(QString::fromStdString(characters.error().message));
            return;
        }
        m_characters = std::move(characters).value();
        for (const auto& character : m_characters) {
            m_characterCombo->addItem(QString::fromStdString(character.name),
                                      QString::fromStdString(character.id));
        }
    }

    const bool hasModels = !m_models.empty();
    const bool hasCharacters = !m_characters.empty();
    m_deleteButton->setEnabled(hasModels);
    m_assignButton->setEnabled(hasModels && hasCharacters);
    m_characterCombo->setEnabled(hasCharacters);
    setStatusText(QStringLiteral("%1 model(s) in %2.")
                      .arg(static_cast<int>(m_models.size()))
                      .arg(QString::fromStdString(m_registry.modelRoot().string())));
}

void RvcModelManagerDialog::importModel() {
    const auto pthPath = QFileDialog::getOpenFileName(
        this, QStringLiteral("Import RVC Model"), QString{}, QStringLiteral("RVC (*.pth)"));
    if (pthPath.isEmpty()) {
        return;
    }

    const auto indexPath = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Optional RVC Index"),
        QString{},
        QStringLiteral("RVC Index (*.index);;Skip Index (*)"));
    const auto defaultName =
        QFileInfo{pthPath}.completeBaseName().replace(QLatin1Char('_'), QLatin1Char(' '));
    bool accepted = false;
    const auto displayName = QInputDialog::getText(this,
                                                   QStringLiteral("RVC Model Name"),
                                                   QStringLiteral("Name"),
                                                   QLineEdit::Normal,
                                                   defaultName,
                                                   &accepted);
    if (!accepted || displayName.trimmed().isEmpty()) {
        return;
    }

    const int sampleRate = QInputDialog::getInt(this,
                                                QStringLiteral("RVC Sample Rate"),
                                                QStringLiteral("Hz"),
                                                48000,
                                                8000,
                                                192000,
                                                1000,
                                                &accepted);
    if (!accepted) {
        return;
    }

    rvc::RvcModelImportRequest request;
    request.displayName = displayName.trimmed().toStdString();
    request.pthPath = std::filesystem::path{pthPath.toStdString()};
    request.indexPath = indexPath.isEmpty() ? std::filesystem::path{}
                                            : std::filesystem::path{indexPath.toStdString()};
    request.sampleRate = sampleRate;

    auto imported = m_registry.importModel(request);
    if (!imported) {
        setStatusText(QString::fromStdString(imported.error().message));
        return;
    }

    refresh();
    setStatusText(QStringLiteral("Imported %1.")
                      .arg(QString::fromStdString(imported.value().displayName)));
}

void RvcModelManagerDialog::deleteSelectedModel() {
    const auto modelId = selectedModelId();
    if (modelId.empty()) {
        setStatusText(QStringLiteral("Select a model first."));
        return;
    }

    auto deleted = m_registry.deleteModel(modelId);
    if (!deleted) {
        setStatusText(QString::fromStdString(deleted.error().message));
        return;
    }

    refresh();
    setStatusText(QStringLiteral("Deleted RVC model."));
}

void RvcModelManagerDialog::assignSelectedModel() {
    if (!m_project.has_value()) {
        setStatusText(QStringLiteral("Open a project before assigning an RVC model."));
        return;
    }

    const auto characterId = selectedCharacterId();
    const auto modelId = selectedModelId();
    if (characterId.empty() || modelId.empty()) {
        setStatusText(QStringLiteral("Select both a character and a model."));
        return;
    }

    auto assigned =
        m_scriptRepository.updateCharacterRvcModel(m_project->rootPath(), characterId, modelId);
    if (!assigned) {
        setStatusText(QString::fromStdString(assigned.error().message));
        return;
    }

    emit rvcAssignmentsChanged();
    refresh();
    setStatusText(QStringLiteral("Assigned RVC model to character."));
}

void RvcModelManagerDialog::setStatusText(const QString& text) {
    m_statusLabel->setText(text);
}

std::string RvcModelManagerDialog::selectedModelId() const {
    const auto* item = m_modelList->currentItem();
    if (item == nullptr) {
        return {};
    }
    return item->data(Qt::UserRole).toString().toStdString();
}

std::string RvcModelManagerDialog::selectedCharacterId() const {
    if (m_characterCombo == nullptr || m_characterCombo->currentIndex() < 0) {
        return {};
    }
    return m_characterCombo->currentData().toString().toStdString();
}

} // namespace voxstudio::ui
