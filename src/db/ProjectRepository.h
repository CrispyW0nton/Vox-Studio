#pragma once

#include "core/Expected.h"
#include "core/Project.h"

#include <filesystem>
#include <string>

namespace voxstudio::db {

class ProjectRepository final {
public:
    [[nodiscard]] core::Expected<core::Project>
    createProject(const std::filesystem::path& projectDirectory, const std::string& name) const;

    [[nodiscard]] core::Expected<core::Project>
    openProject(const std::filesystem::path& projectDirectory) const;

    [[nodiscard]] core::Expected<core::Project> saveProject(const core::Project& project) const;
};

[[nodiscard]] std::filesystem::path projectDatabasePath(const std::filesystem::path& projectRoot);

} // namespace voxstudio::db

