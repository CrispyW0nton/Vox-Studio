#include "ui/RecentProjectsModel.h"

#include <QSettings>

#include <utility>

namespace voxstudio::ui {
namespace {

constexpr auto kSettingsGroup = "recentProjects";
constexpr auto kNameKey = "name";
constexpr auto kPathKey = "path";
constexpr auto kLastOpenedKey = "lastOpened";
constexpr int kMaxRecentProjects = 10;

[[nodiscard]] QString currentTimestamp() {
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}

} // namespace

RecentProjectsModel::RecentProjectsModel(QObject* parent)
    : QAbstractListModel(parent) {
    load();
}

int RecentProjectsModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) {
        return 0;
    }

    return static_cast<int>(m_projects.size());
}

QVariant RecentProjectsModel::data(const QModelIndex& index, const int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= m_projects.size()) {
        return {};
    }

    const auto& project = m_projects.at(index.row());
    switch (role) {
    case Qt::DisplayRole:
    case NameRole:
        return project.name;
    case PathRole:
        return project.path;
    case LastOpenedRole:
        return project.lastOpenedIso;
    default:
        return {};
    }
}

QHash<int, QByteArray> RecentProjectsModel::roleNames() const {
    return {{NameRole, "name"}, {PathRole, "path"}, {LastOpenedRole, "lastOpened"}};
}

void RecentProjectsModel::load() {
    QSettings settings;
    const int count = settings.beginReadArray(kSettingsGroup);

    QVector<RecentProjectEntry> loadedProjects;
    loadedProjects.reserve(count);
    for (int index = 0; index < count; ++index) {
        settings.setArrayIndex(index);
        RecentProjectEntry project{settings.value(kNameKey).toString(),
                                   settings.value(kPathKey).toString(),
                                   settings.value(kLastOpenedKey).toString()};
        if (!project.name.isEmpty() && !project.path.isEmpty()) {
            loadedProjects.push_back(project);
        }
    }
    settings.endArray();

    beginResetModel();
    m_projects = std::move(loadedProjects);
    endResetModel();
}

void RecentProjectsModel::save() const {
    QSettings settings;
    settings.beginWriteArray(kSettingsGroup);
    for (qsizetype index = 0; index < m_projects.size(); ++index) {
        settings.setArrayIndex(static_cast<int>(index));
        settings.setValue(kNameKey, m_projects.at(index).name);
        settings.setValue(kPathKey, m_projects.at(index).path);
        settings.setValue(kLastOpenedKey, m_projects.at(index).lastOpenedIso);
    }
    settings.endArray();
}

void RecentProjectsModel::addOrUpdate(const core::Project& project) {
    RecentProjectEntry entry{QString::fromStdString(project.name()),
                             QString::fromStdString(project.rootPath().string()),
                             currentTimestamp()};

    beginResetModel();
    for (qsizetype index = 0; index < m_projects.size(); ++index) {
        if (m_projects.at(index).path == entry.path) {
            m_projects.removeAt(index);
            break;
        }
    }

    m_projects.prepend(entry);
    while (m_projects.size() > kMaxRecentProjects) {
        m_projects.removeLast();
    }
    endResetModel();

    save();
}

std::optional<RecentProjectEntry> RecentProjectsModel::projectAt(const QModelIndex& index) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= m_projects.size()) {
        return std::nullopt;
    }

    return m_projects.at(index.row());
}

} // namespace voxstudio::ui
