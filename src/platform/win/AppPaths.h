#pragma once

#include <filesystem>
#include <optional>

namespace voxstudio::platform::win {

[[nodiscard]] std::optional<std::filesystem::path> localAppDataPath() noexcept;
[[nodiscard]] std::optional<std::filesystem::path> voxStudioDataPath() noexcept;
[[nodiscard]] std::optional<std::filesystem::path> voxStudioLogPath() noexcept;

} // namespace voxstudio::platform::win
