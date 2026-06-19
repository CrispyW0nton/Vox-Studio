#pragma once

#include <filesystem>
#include <string>

namespace voxstudio::core {

struct ProjectCounts final {
    int characterCount{0};
    int voiceCount{0};
    int lineCount{0};
};

class Project final {
public:
    Project(std::filesystem::path rootPath,
            std::string name,
            std::string createdAt,
            std::string updatedAt,
            ProjectCounts counts);

    [[nodiscard]] const std::filesystem::path& rootPath() const noexcept;
    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] const std::string& createdAt() const noexcept;
    [[nodiscard]] const std::string& updatedAt() const noexcept;
    [[nodiscard]] ProjectCounts counts() const noexcept;

    void setName(std::string name);
    void setUpdatedAt(std::string updatedAt);
    void setCounts(ProjectCounts counts) noexcept;

private:
    std::filesystem::path m_rootPath;
    std::string m_name;
    std::string m_createdAt;
    std::string m_updatedAt;
    ProjectCounts m_counts;
};

} // namespace voxstudio::core

