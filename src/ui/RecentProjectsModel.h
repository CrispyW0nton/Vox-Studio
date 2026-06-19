#pragma once

#include "core/Project.h"

#include <QAbstractListModel>
#include <QDateTime>
#include <QString>
#include <QVector>

#include <optional>

namespace voxstudio::ui {

struct RecentProjectEntry final {
    QString name;
    QString path;
    QString lastOpenedIso;
};

class RecentProjectsModel final : public QAbstractListModel {
    Q_OBJECT

public:
    enum Role {
        NameRole = Qt::UserRole + 1,
        PathRole,
        LastOpenedRole,
    };

    explicit RecentProjectsModel(QObject* parent = nullptr);

    [[nodiscard]] int rowCount(const QModelIndex& parent = QModelIndex{}) const override;
    [[nodiscard]] QVariant data(const QModelIndex& index, int role) const override;
    [[nodiscard]] QHash<int, QByteArray> roleNames() const override;

    void load();
    void save() const;
    void addOrUpdate(const core::Project& project);

    [[nodiscard]] std::optional<RecentProjectEntry> projectAt(const QModelIndex& index) const;

private:
    QVector<RecentProjectEntry> m_projects;
};

} // namespace voxstudio::ui

