#pragma once

#include "core/Project.h"
#include "db/ProjectRepository.h"
#include "ui/RecentProjectsModel.h"

#include <QMainWindow>

#include <memory>
#include <optional>

class QAction;
class QLabel;
class QModelIndex;
class QTabWidget;

namespace voxstudio::ui {

class ProjectHud;
class LiveMicPanel;
class ScriptViewerPanel;
class TimelinePanel;
class VoiceLibraryPanel;

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    void createMenus();
    void createProjectHud();
    void createWorkspace();
    void setCurrentProject(core::Project project);
    void setDirty(bool isDirty);
    void updateWindowTitle();
    void updateProjectHud();
    void updateWorkspaceProjectText();
    void showError(const QString& title, const core::Error& error);
    void refreshCurrentProjectFromDisk();

    void createProject();
    void openProject();
    void openRecentProject(const QModelIndex& index);
    void saveProject();
    void showSettings();
    void promptForApiKeyIfMissing();

    db::ProjectRepository m_projectRepository;
    std::unique_ptr<RecentProjectsModel> m_recentProjectsModel;
    std::optional<core::Project> m_currentProject;
    ProjectHud* m_projectHud{nullptr};
    LiveMicPanel* m_liveMicPanel{nullptr};
    ScriptViewerPanel* m_scriptViewerPanel{nullptr};
    TimelinePanel* m_timelinePanel{nullptr};
    VoiceLibraryPanel* m_voiceLibraryPanel{nullptr};
    QLabel* m_workspaceLabel{nullptr};
    QTabWidget* m_workspaceTabs{nullptr};
    QAction* m_saveAction{nullptr};
    bool m_isDirty{false};
    bool m_promptedForApiKey{false};
};

} // namespace voxstudio::ui
