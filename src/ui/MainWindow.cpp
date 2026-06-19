#include "ui/MainWindow.h"

#include "ui/NewProjectWizard.h"
#include "ui/LiveMicPanel.h"
#include "ui/ProjectHud.h"
#include "ui/ScriptViewerPanel.h"
#include "ui/SettingsDialog.h"
#include "ui/TimelinePanel.h"
#include "ui/VoiceLibraryPanel.h"

#include <QAction>
#include <QDockWidget>
#include <QFileDialog>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QString>
#include <QStatusBar>
#include <QTabWidget>
#include <QTimer>

#include <memory>

namespace voxstudio::ui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    m_recentProjectsModel = std::make_unique<RecentProjectsModel>();

    createMenus();
    createWorkspace();
    createProjectHud();
    updateWindowTitle();
    QTimer::singleShot(0, this, &MainWindow::promptForApiKeyIfMissing);
    resize(1280, 800);
    setMinimumSize(960, 640);
}

MainWindow::~MainWindow() = default;

void MainWindow::createMenus() {
    QMenu* fileMenu = menuBar()->addMenu(QStringLiteral("&File"));

    QAction* newAction = fileMenu->addAction(QStringLiteral("&New Project..."));
    connect(newAction, &QAction::triggered, this, &MainWindow::createProject);

    QAction* openAction = fileMenu->addAction(QStringLiteral("&Open Project..."));
    connect(openAction, &QAction::triggered, this, &MainWindow::openProject);

    m_saveAction = fileMenu->addAction(QStringLiteral("&Save"));
    m_saveAction->setEnabled(false);
    connect(m_saveAction, &QAction::triggered, this, &MainWindow::saveProject);

    QAction* settingsAction = fileMenu->addAction(QStringLiteral("Se&ttings..."));
    connect(settingsAction, &QAction::triggered, this, &MainWindow::showSettings);

    fileMenu->addSeparator();

    QAction* exitAction = fileMenu->addAction(QStringLiteral("E&xit"));
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
}

void MainWindow::createProjectHud() {
    auto dock = std::make_unique<QDockWidget>(QStringLiteral("Project HUD"), this);
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);

    auto hud = std::make_unique<ProjectHud>(dock.get());
    m_projectHud = hud.get();
    m_projectHud->setRecentProjectsModel(m_recentProjectsModel.get());

    connect(m_projectHud, &ProjectHud::newProjectRequested, this, &MainWindow::createProject);
    connect(m_projectHud, &ProjectHud::openProjectRequested, this, &MainWindow::openProject);
    connect(m_projectHud, &ProjectHud::saveProjectRequested, this, &MainWindow::saveProject);
    connect(m_projectHud, &ProjectHud::recentProjectActivated, this,
            &MainWindow::openRecentProject);

    dock->setWidget(hud.release());
    addDockWidget(Qt::LeftDockWidgetArea, dock.get());
    dock.release();
}

void MainWindow::createWorkspace() {
    auto tabs = std::make_unique<QTabWidget>(this);
    m_workspaceTabs = tabs.get();
    tabs->setTabPosition(QTabWidget::West);
    tabs->setDocumentMode(true);

    auto label = std::make_unique<QLabel>(QStringLiteral("No project open"), this);
    label->setAlignment(Qt::AlignCenter);
    m_workspaceLabel = label.get();
    tabs->addTab(label.release(), QStringLiteral("Project"));

    auto voiceLibrary = std::make_unique<VoiceLibraryPanel>(tabs.get());
    m_voiceLibraryPanel = voiceLibrary.get();
    connect(m_voiceLibraryPanel, &VoiceLibraryPanel::voiceCacheChanged, this,
            &MainWindow::refreshCurrentProjectFromDisk);
    tabs->addTab(voiceLibrary.release(), QStringLiteral("Voices"));

    auto scriptViewer = std::make_unique<ScriptViewerPanel>(tabs.get());
    m_scriptViewerPanel = scriptViewer.get();
    connect(m_scriptViewerPanel, &ScriptViewerPanel::scriptImported, this,
            &MainWindow::refreshCurrentProjectFromDisk);
    tabs->addTab(scriptViewer.release(), QStringLiteral("Scripts"));

    auto timeline = std::make_unique<TimelinePanel>(tabs.get());
    m_timelinePanel = timeline.get();
    connect(m_timelinePanel, &TimelinePanel::sequenceChanged, this,
            &MainWindow::refreshCurrentProjectFromDisk);
    tabs->addTab(timeline.release(), QStringLiteral("Timeline"));

    auto liveMic = std::make_unique<LiveMicPanel>(tabs.get());
    m_liveMicPanel = liveMic.get();
    tabs->addTab(liveMic.release(), QStringLiteral("Live Mic"));

    setCentralWidget(tabs.release());
}

void MainWindow::setCurrentProject(core::Project project) {
    m_currentProject = std::move(project);
    m_recentProjectsModel->addOrUpdate(*m_currentProject);
    setDirty(false);
    updateProjectHud();
    updateWorkspaceProjectText();
    if (m_voiceLibraryPanel != nullptr) {
        m_voiceLibraryPanel->setProject(m_currentProject);
    }
    if (m_scriptViewerPanel != nullptr) {
        m_scriptViewerPanel->setProject(m_currentProject);
    }
    if (m_timelinePanel != nullptr) {
        m_timelinePanel->setProject(m_currentProject);
    }
    if (m_liveMicPanel != nullptr) {
        m_liveMicPanel->setProject(m_currentProject);
    }
}

void MainWindow::setDirty(const bool isDirty) {
    m_isDirty = isDirty;
    const bool canSave = m_currentProject.has_value() && m_isDirty;
    if (m_saveAction != nullptr) {
        m_saveAction->setEnabled(canSave);
    }
    if (m_projectHud != nullptr) {
        m_projectHud->setDirty(canSave);
    }
    updateWindowTitle();
}

void MainWindow::updateWindowTitle() {
    QString title = QStringLiteral("Vox Studio");
    if (m_currentProject.has_value()) {
        title += QStringLiteral(" - ");
        title += QString::fromStdString(m_currentProject->name());
        if (m_isDirty) {
            title += QStringLiteral("*");
        }
    }
    setWindowTitle(title);
}

void MainWindow::updateProjectHud() {
    if (m_projectHud == nullptr) {
        return;
    }

    m_projectHud->setProject(m_currentProject.has_value() ? &(*m_currentProject) : nullptr);
    m_projectHud->setDirty(m_currentProject.has_value() && m_isDirty);
}

void MainWindow::updateWorkspaceProjectText() {
    if (m_workspaceLabel == nullptr) {
        return;
    }

    if (!m_currentProject.has_value()) {
        m_workspaceLabel->setText(QStringLiteral("No project open"));
        return;
    }

    m_workspaceLabel->setText(
        QStringLiteral("%1\n%2")
            .arg(QString::fromStdString(m_currentProject->name()),
                 QString::fromStdString(m_currentProject->rootPath().string())));
}

void MainWindow::showError(const QString& title, const core::Error& error) {
    QMessageBox::critical(this, title, QString::fromStdString(error.message));
}

void MainWindow::refreshCurrentProjectFromDisk() {
    if (!m_currentProject.has_value()) {
        return;
    }

    auto refreshedProject = m_projectRepository.openProject(m_currentProject->rootPath());
    if (!refreshedProject) {
        showError(QStringLiteral("Refresh Project"), refreshedProject.error());
        return;
    }

    m_currentProject = std::move(refreshedProject).value();
    updateProjectHud();
    updateWorkspaceProjectText();
    updateWindowTitle();
    if (m_scriptViewerPanel != nullptr) {
        m_scriptViewerPanel->setProject(m_currentProject);
    }
    if (m_timelinePanel != nullptr) {
        m_timelinePanel->setProject(m_currentProject);
    }
    if (m_liveMicPanel != nullptr) {
        m_liveMicPanel->setProject(m_currentProject);
    }
}

void MainWindow::createProject() {
    NewProjectWizard wizard{this};
    if (wizard.exec() != QDialog::Accepted) {
        return;
    }

    auto project = m_projectRepository.createProject(wizard.projectDirectory(),
                                                     wizard.projectName().toStdString());
    if (!project) {
        showError(QStringLiteral("New Project"), project.error());
        return;
    }

    setCurrentProject(std::move(project).value());
    statusBar()->showMessage(QStringLiteral("Project created"), 3000);
}

void MainWindow::openProject() {
    const QString directory =
        QFileDialog::getExistingDirectory(this, QStringLiteral("Open Vox Project"));
    if (directory.isEmpty()) {
        return;
    }

    auto project = m_projectRepository.openProject(std::filesystem::path{directory.toStdString()});
    if (!project) {
        showError(QStringLiteral("Open Project"), project.error());
        return;
    }

    setCurrentProject(std::move(project).value());
    statusBar()->showMessage(QStringLiteral("Project opened"), 3000);
}

void MainWindow::openRecentProject(const QModelIndex& index) {
    const auto recentProject = m_recentProjectsModel->projectAt(index);
    if (!recentProject.has_value()) {
        return;
    }

    auto project = m_projectRepository.openProject(
        std::filesystem::path{recentProject->path.toStdString()});
    if (!project) {
        showError(QStringLiteral("Open Recent Project"), project.error());
        return;
    }

    setCurrentProject(std::move(project).value());
    statusBar()->showMessage(QStringLiteral("Project opened"), 3000);
}

void MainWindow::saveProject() {
    if (!m_currentProject.has_value()) {
        return;
    }

    auto savedProject = m_projectRepository.saveProject(*m_currentProject);
    if (!savedProject) {
        showError(QStringLiteral("Save Project"), savedProject.error());
        return;
    }

    setCurrentProject(std::move(savedProject).value());
    statusBar()->showMessage(QStringLiteral("Project saved"), 3000);
}

void MainWindow::showSettings() {
    SettingsDialog dialog{this};
    dialog.exec();
}

void MainWindow::promptForApiKeyIfMissing() {
    if (m_promptedForApiKey) {
        return;
    }

    m_promptedForApiKey = true;
    const secrets::DpapiVault vault;
    if (vault.hasElevenLabsApiKey()) {
        return;
    }

    SettingsDialog dialog{this};
    dialog.setIntroMessage(
        QStringLiteral("Enter an ElevenLabs API key to enable cloud voice features."));
    dialog.exec();
}

} // namespace voxstudio::ui
