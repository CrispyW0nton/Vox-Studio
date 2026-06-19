#include "core/Project.h"

#include <utility>

namespace voxstudio::core {

Project::Project(std::filesystem::path rootPath,
                 std::string name,
                 std::string createdAt,
                 std::string updatedAt,
                 ProjectCounts counts)
    : m_rootPath(std::move(rootPath))
    , m_name(std::move(name))
    , m_createdAt(std::move(createdAt))
    , m_updatedAt(std::move(updatedAt))
    , m_counts(counts) {}

const std::filesystem::path& Project::rootPath() const noexcept {
    return m_rootPath;
}

const std::string& Project::name() const noexcept {
    return m_name;
}

const std::string& Project::createdAt() const noexcept {
    return m_createdAt;
}

const std::string& Project::updatedAt() const noexcept {
    return m_updatedAt;
}

ProjectCounts Project::counts() const noexcept {
    return m_counts;
}

void Project::setName(std::string name) {
    m_name = std::move(name);
}

void Project::setUpdatedAt(std::string updatedAt) {
    m_updatedAt = std::move(updatedAt);
}

void Project::setCounts(ProjectCounts counts) noexcept {
    m_counts = counts;
}

} // namespace voxstudio::core

