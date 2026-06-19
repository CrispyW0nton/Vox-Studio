#include "io/scripts/RenpyImporter.h"

#include <map>
#include <regex>
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

[[nodiscard]] std::string unescapeRenpyText(std::string value) {
    std::string result;
    result.reserve(value.size());
    bool escaping = false;
    for (const char character : value) {
        if (escaping) {
            result.push_back(character);
            escaping = false;
            continue;
        }
        if (character == '\\') {
            escaping = true;
            continue;
        }
        result.push_back(character);
    }
    return result;
}

} // namespace

core::Expected<ParsedScript> parseRenpyScript(const std::string& text,
                                              const std::filesystem::path& sourcePath) {
    ParsedScript script;
    script.sourcePath = sourcePath;
    script.format = ScriptFormat::Renpy;

    const std::regex defineRegex{
        R"renpy(^\s*define\s+([A-Za-z_]\w*)\s*=\s*Character\("([^"]+)"\))renpy"};
    const std::regex labelRegex{R"renpy(^\s*label\s+([A-Za-z_]\w*)\s*:)renpy"};
    const std::regex speakerRegex{R"renpy(^\s*([A-Za-z_]\w*)\s+"((?:\\"|[^"])*)")renpy"};
    const std::regex narrationRegex{R"renpy(^\s*"((?:\\"|[^"])*)")renpy"};

    std::map<std::string, std::string> aliases;
    std::istringstream stream{text};
    std::string rawLine;
    std::string currentScene;

    while (std::getline(stream, rawLine)) {
        std::smatch match;
        if (std::regex_search(rawLine, match, defineRegex)) {
            aliases[match[1].str()] = match[2].str();
            continue;
        }

        if (std::regex_search(rawLine, match, labelRegex)) {
            currentScene = match[1].str();
            continue;
        }

        if (std::regex_search(rawLine, match, speakerRegex)) {
            const auto alias = match[1].str();
            const auto aliasMatch = aliases.find(alias);
            const auto speaker = aliasMatch == aliases.end() ? alias : aliasMatch->second;
            script.lines.push_back(ParsedScriptLine{speaker, unescapeRenpyText(match[2].str()),
                                                    currentScene});
            continue;
        }

        if (std::regex_search(rawLine, match, narrationRegex)) {
            const auto dialogue = trimCopy(unescapeRenpyText(match[1].str()));
            if (!dialogue.empty()) {
                script.lines.push_back(ParsedScriptLine{{}, dialogue, currentScene});
            }
        }
    }

    if (script.lines.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Ren'Py script did not contain importable dialogue lines.");
    }

    return script;
}

} // namespace voxstudio::io::scripts
