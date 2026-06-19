#include "io/scripts/ScriptImporter.h"

#include "io/scripts/CsvImporter.h"
#include "io/scripts/FountainImporter.h"
#include "io/scripts/PlainTextImporter.h"
#include "io/scripts/RenpyImporter.h"
#include "io/scripts/YarnImporter.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iterator>

namespace voxstudio::io::scripts {
namespace {

[[nodiscard]] std::string lowerExtension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension;
}

[[nodiscard]] core::Expected<std::string> readTextFile(const std::filesystem::path& path) {
    try {
        std::ifstream input{path, std::ios::binary};
        if (!input) {
            return core::makeError(core::ErrorCode::FileSystemFailure,
                                   "Script file could not be opened.");
        }

        return std::string{std::istreambuf_iterator<char>{input},
                           std::istreambuf_iterator<char>{}};
    } catch (const std::exception& exception) {
        return core::makeError(core::ErrorCode::FileSystemFailure, exception.what());
    }
}

} // namespace

core::Expected<ScriptFormat> scriptFormatFromPath(const std::filesystem::path& path) {
    const auto extension = lowerExtension(path);
    if (extension == ".txt") {
        return ScriptFormat::PlainText;
    }
    if (extension == ".fountain") {
        return ScriptFormat::Fountain;
    }
    if (extension == ".rpy") {
        return ScriptFormat::Renpy;
    }
    if (extension == ".yarn.json" || extension == ".json") {
        return ScriptFormat::Yarn;
    }
    if (extension == ".csv") {
        return ScriptFormat::Csv;
    }

    return core::makeError(core::ErrorCode::InvalidArgument,
                           "Unsupported script format: " + extension);
}

core::Expected<ParsedScript> importScriptFile(const std::filesystem::path& path) {
    auto format = scriptFormatFromPath(path);
    if (!format) {
        return format.error();
    }

    auto text = readTextFile(path);
    if (!text) {
        return text.error();
    }

    switch (format.value()) {
    case ScriptFormat::PlainText:
        return parsePlainTextScript(text.value(), path);
    case ScriptFormat::Fountain:
        return parseFountainScript(text.value(), path);
    case ScriptFormat::Renpy:
        return parseRenpyScript(text.value(), path);
    case ScriptFormat::Yarn:
        return parseYarnScript(text.value(), path);
    case ScriptFormat::Csv:
        return parseCsvScript(text.value(), path);
    }

    return core::makeError(core::ErrorCode::InvalidArgument, "Unsupported script format.");
}

} // namespace voxstudio::io::scripts
