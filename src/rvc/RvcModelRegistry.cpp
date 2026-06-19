#include "rvc/RvcModelRegistry.h"

#include "platform/win/AppPaths.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>

namespace voxstudio::rvc {
namespace {

constexpr auto kManifestFileName = "model.json";

[[nodiscard]] core::Error registryError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] std::string utcTimestampNow() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utcTime{};
    gmtime_s(&utcTime, &time);

    std::ostringstream stream;
    stream << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

[[nodiscard]] std::string sanitizedModelId(const std::string& displayName) {
    std::string id;
    id.reserve(displayName.size() + 24U);
    for (const char character : displayName) {
        const auto value = static_cast<unsigned char>(character);
        if (std::isalnum(value)) {
            id.push_back(static_cast<char>(std::tolower(value)));
        } else if (character == '-' || character == '_') {
            id.push_back(character);
        } else if (!id.empty() && id.back() != '_') {
            id.push_back('_');
        }
    }
    if (id.empty()) {
        id = "rvc_model";
    }

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return id + "_" + std::to_string(millis);
}

[[nodiscard]] bool isSafeModelId(const std::string& modelId) {
    return !modelId.empty() &&
           std::ranges::all_of(modelId, [](const char character) {
               const auto value = static_cast<unsigned char>(character);
               return std::isalnum(value) || character == '-' || character == '_';
           });
}

[[nodiscard]] core::Expected<RvcModelRecord> recordFromManifest(
    const std::filesystem::path& manifestPath) {
    try {
        std::ifstream input{manifestPath};
        if (!input) {
            return registryError("RVC model manifest could not be opened.");
        }

        const auto json = nlohmann::json::parse(input);
        return RvcModelRecord{json.at("id").get<std::string>(),
                              json.at("display_name").get<std::string>(),
                              json.at("pth_path").get<std::string>(),
                              json.value("index_path", std::string{}),
                              json.value("sample_rate", 48000),
                              json.value("notes", std::string{}),
                              json.value("imported_at", std::string{})};
    } catch (const std::exception& exception) {
        return registryError(exception.what());
    }
}

[[nodiscard]] nlohmann::json manifestFromRecord(const RvcModelRecord& record) {
    nlohmann::json json;
    json["id"] = record.id;
    json["display_name"] = record.displayName;
    json["pth_path"] = record.pthPath.string();
    json["index_path"] = record.indexPath.empty() ? std::string{} : record.indexPath.string();
    json["sample_rate"] = record.sampleRate;
    json["notes"] = record.notes;
    json["imported_at"] = record.importedAt;
    return json;
}

[[nodiscard]] core::Expected<bool> validateImportRequest(
    const RvcModelImportRequest& request) {
    if (request.displayName.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "RVC model display name must not be empty.");
    }
    if (request.sampleRate <= 0) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "RVC model sample rate is invalid.");
    }
    if (request.pthPath.extension() != ".pth") {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "RVC model weights must use the .pth extension.");
    }
    if (!std::filesystem::exists(request.pthPath)) {
        return registryError("RVC model weights file does not exist.");
    }
    if (!request.indexPath.empty()) {
        if (request.indexPath.extension() != ".index") {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "RVC index file must use the .index extension.");
        }
        if (!std::filesystem::exists(request.indexPath)) {
            return registryError("RVC index file does not exist.");
        }
    }
    return true;
}

} // namespace

RvcModelRegistry::RvcModelRegistry()
    : RvcModelRegistry(defaultModelRoot()) {}

RvcModelRegistry::RvcModelRegistry(std::filesystem::path modelRoot)
    : m_modelRoot(std::move(modelRoot)) {}

std::filesystem::path RvcModelRegistry::defaultModelRoot() {
    const auto appData = platform::win::voxStudioDataPath();
    if (appData.has_value()) {
        return *appData / "rvc_models";
    }
    return std::filesystem::temp_directory_path() / "VoxStudio" / "rvc_models";
}

const std::filesystem::path& RvcModelRegistry::modelRoot() const noexcept {
    return m_modelRoot;
}

core::Expected<std::vector<RvcModelRecord>> RvcModelRegistry::listModels() const {
    std::vector<RvcModelRecord> records;
    if (!std::filesystem::exists(m_modelRoot)) {
        return records;
    }

    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{m_modelRoot, error}) {
        if (error) {
            return registryError(error.message());
        }
        if (!entry.is_directory()) {
            continue;
        }
        const auto manifestPath = entry.path() / kManifestFileName;
        if (!std::filesystem::exists(manifestPath)) {
            continue;
        }
        auto record = recordFromManifest(manifestPath);
        if (!record) {
            return record.error();
        }
        records.push_back(std::move(record).value());
    }

    std::ranges::sort(records, {}, &RvcModelRecord::displayName);
    return records;
}

core::Expected<RvcModelRecord> RvcModelRegistry::importModel(
    const RvcModelImportRequest& request) const {
    auto valid = validateImportRequest(request);
    if (!valid) {
        return valid.error();
    }

    const auto id = sanitizedModelId(request.displayName);
    const auto modelDirectoryPath = m_modelRoot / id;
    const auto targetPth = modelDirectoryPath / request.pthPath.filename();
    const auto targetIndex = request.indexPath.empty()
                                 ? std::filesystem::path{}
                                 : modelDirectoryPath / request.indexPath.filename();

    std::error_code error;
    std::filesystem::create_directories(modelDirectoryPath, error);
    if (error) {
        return registryError(error.message());
    }

    std::filesystem::copy_file(request.pthPath,
                               targetPth,
                               std::filesystem::copy_options::overwrite_existing,
                               error);
    if (error) {
        return registryError(error.message());
    }
    if (!request.indexPath.empty()) {
        std::filesystem::copy_file(request.indexPath,
                                   targetIndex,
                                   std::filesystem::copy_options::overwrite_existing,
                                   error);
        if (error) {
            return registryError(error.message());
        }
    }

    const RvcModelRecord record{id,
                                request.displayName,
                                targetPth,
                                targetIndex,
                                request.sampleRate,
                                request.notes,
                                utcTimestampNow()};
    try {
        std::ofstream output{modelDirectoryPath / kManifestFileName, std::ios::trunc};
        output << manifestFromRecord(record).dump(2);
    } catch (const std::exception& exception) {
        return registryError(exception.what());
    }

    return record;
}

core::Expected<bool> RvcModelRegistry::deleteModel(const std::string& modelId) const {
    if (!isSafeModelId(modelId)) {
        return core::makeError(core::ErrorCode::InvalidArgument, "RVC model id is invalid.");
    }

    auto directory = modelDirectory(modelId);
    if (!directory) {
        return directory.error();
    }

    std::error_code error;
    std::filesystem::remove_all(directory.value(), error);
    if (error) {
        return registryError(error.message());
    }
    return true;
}

core::Expected<std::filesystem::path> RvcModelRegistry::modelDirectory(
    const std::string& modelId) const {
    if (!isSafeModelId(modelId)) {
        return core::makeError(core::ErrorCode::InvalidArgument, "RVC model id is invalid.");
    }
    return m_modelRoot / modelId;
}

} // namespace voxstudio::rvc
