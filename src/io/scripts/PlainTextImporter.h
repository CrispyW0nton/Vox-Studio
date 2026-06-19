#pragma once

#include "core/Expected.h"
#include "io/scripts/ScriptTypes.h"

#include <filesystem>
#include <string>

namespace voxstudio::io::scripts {

[[nodiscard]] core::Expected<ParsedScript>
parsePlainTextScript(const std::string& text, const std::filesystem::path& sourcePath);

} // namespace voxstudio::io::scripts
