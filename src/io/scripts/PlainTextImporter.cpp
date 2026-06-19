#include "io/scripts/PlainTextImporter.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace voxstudio::io::scripts {
namespace {

[[nodiscard]] std::string trimCopy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

[[nodiscard]] bool isLikelySpeaker(const std::string& value) {
    if (value.empty() || value.size() > 48) {
        return false;
    }

    bool hasAlpha = false;
    for (const unsigned char character : value) {
        if (std::isalpha(character) != 0) {
            hasAlpha = true;
            continue;
        }
        if (std::isspace(character) != 0 || character == '_' || character == '-' ||
            character == '\'' || character == '.') {
            continue;
        }
        return false;
    }

    return hasAlpha;
}

} // namespace

core::Expected<ParsedScript> parsePlainTextScript(const std::string& text,
                                                  const std::filesystem::path& sourcePath) {
    ParsedScript script;
    script.sourcePath = sourcePath;
    script.format = ScriptFormat::PlainText;

    std::istringstream stream{text};
    std::string rawLine;
    std::string currentScene;
    while (std::getline(stream, rawLine)) {
        const auto line = trimCopy(rawLine);
        if (line.empty()) {
            continue;
        }

        if (line.rfind("# ", 0) == 0 || line.rfind("## ", 0) == 0) {
            currentScene = trimCopy(line.substr(line.find_first_not_of("# ")));
            continue;
        }

        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            const auto speaker = trimCopy(line.substr(0, colon));
            const auto dialogue = trimCopy(line.substr(colon + 1));
            if (isLikelySpeaker(speaker) && !dialogue.empty()) {
                script.lines.push_back(ParsedScriptLine{speaker, dialogue, currentScene});
                continue;
            }
        }

        script.lines.push_back(ParsedScriptLine{{}, line, currentScene});
    }

    if (script.lines.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Plain text script did not contain importable lines.");
    }

    return script;
}

} // namespace voxstudio::io::scripts
