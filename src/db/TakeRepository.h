#pragma once

#include "core/Expected.h"

#include <filesystem>
#include <string>
#include <vector>

namespace voxstudio::db {

struct TakeRecord final {
    std::string id;
    std::string lineId;
    std::string source;
    std::string voiceId;
    std::string rvcModelId;
    std::string filePath;
    int durationMs{0};
    double lufs{0.0};
    bool starred{false};
    std::string createdAt;
    std::string metadataJson;
};

struct NewTakeRecord final {
    std::string id;
    std::string lineId;
    std::string source;
    std::string voiceId;
    std::string rvcModelId;
    std::string filePath;
    int durationMs{0};
    double lufs{0.0};
    bool starred{false};
    std::string metadataJson;
};

class TakeRepository final {
public:
    [[nodiscard]] core::Expected<std::string> createTakeId() const;

    [[nodiscard]] core::Expected<std::vector<TakeRecord>>
    listTakes(const std::filesystem::path& projectRoot, const std::string& lineId) const;

    [[nodiscard]] core::Expected<TakeRecord>
    insertTake(const std::filesystem::path& projectRoot, const NewTakeRecord& take) const;

    [[nodiscard]] core::Expected<TakeRecord>
    setActiveTake(const std::filesystem::path& projectRoot,
                  const std::string& lineId,
                  const std::string& takeId) const;

    [[nodiscard]] core::Expected<bool>
    deleteTake(const std::filesystem::path& projectRoot,
               const std::string& lineId,
               const std::string& takeId) const;
};

} // namespace voxstudio::db
