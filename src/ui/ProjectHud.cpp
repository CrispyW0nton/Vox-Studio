#include "ui/ProjectHud.h"

#include <QAbstractItemModel>
#include <QFormLayout>
#include <QFrame>
#include <QLabel>
#include <QListView>
#include <QPushButton>
#include <QVBoxLayout>

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

[[nodiscard]] QLabel* createValueLabel() {
    auto label = std::make_unique<QLabel>(QStringLiteral("-"));
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setWordWrap(true);
    return label.release();
}

} // namespace

ProjectHud::ProjectHud(QWidget* parent)
    : QWidget(parent) {
    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto* actionsLabel = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Project"));
    actionsLabel->setObjectName(QStringLiteral("ProjectHudTitle"));

    auto newButton = std::make_unique<QPushButton>(QStringLiteral("New Project"));
    connect(newButton.get(), &QPushButton::clicked, this, &ProjectHud::newProjectRequested);
    rootLayout->addWidget(newButton.release());

    auto openButton = std::make_unique<QPushButton>(QStringLiteral("Open Project"));
    connect(openButton.get(), &QPushButton::clicked, this, &ProjectHud::openProjectRequested);
    rootLayout->addWidget(openButton.release());

    auto saveButton = std::make_unique<QPushButton>(QStringLiteral("Save"));
    m_saveButton = saveButton.get();
    m_saveButton->setEnabled(false);
    connect(saveButton.get(), &QPushButton::clicked, this, &ProjectHud::saveProjectRequested);
    rootLayout->addWidget(saveButton.release());

    auto separator = std::make_unique<QFrame>();
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);
    rootLayout->addWidget(separator.release());

    auto formLayout = std::make_unique<QFormLayout>();
    m_nameValue = createValueLabel();
    m_pathValue = createValueLabel();
    m_createdValue = createValueLabel();
    m_updatedValue = createValueLabel();
    m_countsValue = createValueLabel();

    formLayout->addRow(QStringLiteral("Name"), m_nameValue);
    formLayout->addRow(QStringLiteral("Path"), m_pathValue);
    formLayout->addRow(QStringLiteral("Created"), m_createdValue);
    formLayout->addRow(QStringLiteral("Updated"), m_updatedValue);
    formLayout->addRow(QStringLiteral("Counts"), m_countsValue);
    rootLayout->addLayout(formLayout.release());

    auto* recentLabel = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Recent"));
    recentLabel->setObjectName(QStringLiteral("RecentProjectsTitle"));

    auto recentView = std::make_unique<QListView>();
    m_recentProjectsView = recentView.get();
    m_recentProjectsView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_recentProjectsView->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(m_recentProjectsView, &QListView::doubleClicked, this,
            &ProjectHud::recentProjectActivated);
    rootLayout->addWidget(recentView.release(), 1);

    setLayout(rootLayout.release());
    updateProjectText(nullptr);
}

void ProjectHud::setRecentProjectsModel(QAbstractItemModel* model) {
    m_recentProjectsView->setModel(model);
}

void ProjectHud::setProject(const core::Project* project) {
    updateProjectText(project);
}

void ProjectHud::setDirty(const bool isDirty) {
    m_saveButton->setEnabled(isDirty);
}

void ProjectHud::updateProjectText(const core::Project* project) {
    if (project == nullptr) {
        m_nameValue->setText(QStringLiteral("-"));
        m_pathValue->setText(QStringLiteral("-"));
        m_createdValue->setText(QStringLiteral("-"));
        m_updatedValue->setText(QStringLiteral("-"));
        m_countsValue->setText(QStringLiteral("-"));
        return;
    }

    const auto counts = project->counts();
    m_nameValue->setText(QString::fromStdString(project->name()));
    m_pathValue->setText(QString::fromStdString(project->rootPath().string()));
    m_createdValue->setText(QString::fromStdString(project->createdAt()));
    m_updatedValue->setText(QString::fromStdString(project->updatedAt()));
    m_countsValue->setText(QStringLiteral("%1 characters, %2 voices, %3 lines")
                               .arg(counts.characterCount)
                               .arg(counts.voiceCount)
                               .arg(counts.lineCount));
}

} // namespace voxstudio::ui
