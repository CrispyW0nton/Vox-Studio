#pragma once

#include <string>
#include <utility>

namespace voxstudio::core {

enum class ErrorCode {
    InvalidArgument,
    InvalidProjectPath,
    ProjectAlreadyExists,
    ProjectNotFound,
    FileSystemFailure,
    DatabaseOpenFailed,
    DatabaseQueryFailed,
    MigrationFailed,
    SecretStorageFailed,
};

struct Error final {
    ErrorCode code{ErrorCode::InvalidArgument};
    std::string message;
};

[[nodiscard]] inline Error makeError(const ErrorCode code, std::string message) {
    return Error{code, std::move(message)};
}

} // namespace voxstudio::core
