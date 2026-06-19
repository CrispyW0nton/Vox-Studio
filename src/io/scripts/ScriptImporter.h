#pragma once

#include "core/Expected.h"
#include "io/scripts/ScriptTypes.h"

#include <filesystem>

namespace voxstudio::io::scripts {

[[nodiscard]] core::Expected<ScriptFormat> scriptFormatFromPath(const std::filesystem::path& path);
[[nodiscard]] core::Expected<ParsedScript> importScriptFile(const std::filesystem::path& path);

} // namespace voxstudio::io::scripts
