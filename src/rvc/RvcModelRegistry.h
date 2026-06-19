#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <string>
#include <vector>

namespace voxstudio::rvc {

struct RvcModelRecord final {
    std::string id;
    std::string displayName;
    std::filesystem::path pthPath;
    std::filesystem::path indexPath;
    int sampleRate{48000};
    std::string notes;
    std::string importedAt;
};

struct RvcModelImportRequest final {
    std::string displayName;
    std::filesystem::path pthPath;
    std::filesystem::path indexPath;
    int sampleRate{48000};
    std::string notes;
};

class RvcModelRegistry final {
public:
    RvcModelRegistry();
    explicit RvcModelRegistry(std::filesystem::path modelRoot);

    [[nodiscard]] static std::filesystem::path defaultModelRoot();

    [[nodiscard]] const std::filesystem::path& modelRoot() const noexcept;
    [[nodiscard]] core::Expected<std::vector<RvcModelRecord>> listModels() const;
    [[nodiscard]] core::Expected<RvcModelRecord> importModel(
        const RvcModelImportRequest& request) const;
    [[nodiscard]] core::Expected<bool> deleteModel(const std::string& modelId) const;
    [[nodiscard]] core::Expected<std::filesystem::path> modelDirectory(
        const std::string& modelId) const;

private:
    std::filesystem::path m_modelRoot;
};

} // namespace voxstudio::rvc
