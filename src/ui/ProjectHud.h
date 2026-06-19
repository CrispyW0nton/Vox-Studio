#pragma once

#include "core/Project.h"

#include <QWidget>

class QLabel;
class QListView;
class QModelIndex;
class QPushButton;
class QAbstractItemModel;

namespace voxstudio::ui {

class ProjectHud final : public QWidget {
    Q_OBJECT

public:
    explicit ProjectHud(QWidget* parent = nullptr);

    void setRecentProjectsModel(QAbstractItemModel* model);
    void setProject(const core::Project* project);
    void setDirty(bool isDirty);

signals:
    void newProjectRequested();
    void openProjectRequested();
    void saveProjectRequested();
    void recentProjectActivated(const QModelIndex& index);

private:
    void updateProjectText(const core::Project* project);

    QLabel* m_nameValue{nullptr};
    QLabel* m_pathValue{nullptr};
    QLabel* m_createdValue{nullptr};
    QLabel* m_updatedValue{nullptr};
    QLabel* m_countsValue{nullptr};
    QListView* m_recentProjectsView{nullptr};
    QPushButton* m_saveButton{nullptr};
};

} // namespace voxstudio::ui

