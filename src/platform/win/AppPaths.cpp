#include "platform/win/AppPaths.h"

#include <Windows.h>
#include <ShlObj.h>

namespace voxstudio::platform::win {
namespace {

[[nodiscard]] std::optional<std::filesystem::path>
knownFolderPath(const KNOWNFOLDERID& folderId) noexcept {
    PWSTR rawPath = nullptr;
    const HRESULT result = SHGetKnownFolderPath(folderId, KF_FLAG_DEFAULT, nullptr, &rawPath);
    if (FAILED(result) || rawPath == nullptr) {
        return std::nullopt;
    }

    try {
        std::filesystem::path path{rawPath};
        CoTaskMemFree(rawPath);
        return path;
    } catch (...) {
        CoTaskMemFree(rawPath);
        return std::nullopt;
    }
}

} // namespace

std::optional<std::filesystem::path> localAppDataPath() noexcept {
    return knownFolderPath(FOLDERID_LocalAppData);
}

std::optional<std::filesystem::path> voxStudioDataPath() noexcept {
    const auto localAppData = localAppDataPath();
    if (!localAppData.has_value()) {
        return std::nullopt;
    }

    try {
        return *localAppData / "VoxStudio";
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::filesystem::path> voxStudioLogPath() noexcept {
    const auto appData = voxStudioDataPath();
    if (!appData.has_value()) {
        return std::nullopt;
    }

    try {
        return *appData / "logs";
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace voxstudio::platform::win
