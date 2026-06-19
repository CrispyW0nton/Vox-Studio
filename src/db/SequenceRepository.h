#pragma once

#include "core/Expected.h"
#include "core/Sequence.h"

#include <filesystem>
#include <string>
#include <vector>

namespace voxstudio::db {

struct SequenceSummaryRecord final {
    std::string id;
    std::string name;
    std::string createdAt;
};

struct NewSequenceItemRecord final {
    std::string lineId;
    int gapMs{250};
};

class SequenceRepository final {
public:
    [[nodiscard]] core::Expected<std::vector<SequenceSummaryRecord>>
    listSequences(const std::filesystem::path& projectRoot) const;

    [[nodiscard]] core::Expected<core::Sequence>
    createSequence(const std::filesystem::path& projectRoot,
                   const std::string& name,
                   const std::vector<NewSequenceItemRecord>& items) const;

    [[nodiscard]] core::Expected<core::Sequence>
    loadSequence(const std::filesystem::path& projectRoot,
                 const std::string& sequenceId) const;

    [[nodiscard]] core::Expected<core::Sequence>
    updateSequenceItems(const std::filesystem::path& projectRoot,
                        const std::string& sequenceId,
                        const std::vector<NewSequenceItemRecord>& items) const;
};

} // namespace voxstudio::db
