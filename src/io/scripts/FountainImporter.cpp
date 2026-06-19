#include "io/scripts/FountainImporter.h"

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

[[nodiscard]] bool isSceneHeading(const std::string& line) {
    return line.rfind("INT.", 0) == 0 || line.rfind("EXT.", 0) == 0 ||
           line.rfind("EST.", 0) == 0 || line.rfind("I/E.", 0) == 0 ||
           line.rfind("#", 0) == 0;
}

[[nodiscard]] bool isCharacterCue(const std::string& line) {
    if (line.empty() || line.size() > 48 || isSceneHeading(line)) {
        return false;
    }
    if (line.find(':') != std::string::npos || line.front() == '(') {
        return false;
    }

    bool hasAlpha = false;
    for (const unsigned char character : line) {
        if (std::isalpha(character) != 0) {
            hasAlpha = true;
            if (std::toupper(character) != character) {
                return false;
            }
            continue;
        }
        if (std::isspace(character) != 0 || character == '-' || character == '\'' ||
            character == '.' || character == '(' || character == ')') {
            continue;
        }
        return false;
    }

    return hasAlpha;
}

} // namespace

core::Expected<ParsedScript> parseFountainScript(const std::string& text,
                                                 const std::filesystem::path& sourcePath) {
    ParsedScript script;
    script.sourcePath = sourcePath;
    script.format = ScriptFormat::Fountain;

    std::istringstream stream{text};
    std::string rawLine;
    std::string currentScene;
    std::string currentSpeaker;
    std::string dialogue;

    auto flushDialogue = [&]() {
        const auto textLine = trimCopy(dialogue);
        if (!currentSpeaker.empty() && !textLine.empty()) {
            script.lines.push_back(ParsedScriptLine{currentSpeaker, textLine, currentScene});
        }
        dialogue.clear();
    };

    while (std::getline(stream, rawLine)) {
        const auto line = trimCopy(rawLine);
        if (line.empty()) {
            flushDialogue();
            currentSpeaker.clear();
            continue;
        }

        if (isSceneHeading(line)) {
            flushDialogue();
            currentSpeaker.clear();
            currentScene = line.front() == '#' ? trimCopy(line.substr(1)) : line;
            continue;
        }

        if (isCharacterCue(line)) {
            flushDialogue();
            currentSpeaker = line;
            continue;
        }

        if (!currentSpeaker.empty() && line.front() != '(') {
            if (!dialogue.empty()) {
                dialogue += ' ';
            }
            dialogue += line;
        }
    }

    flushDialogue();

    if (script.lines.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Fountain script did not contain importable dialogue lines.");
    }

    return script;
}

} // namespace voxstudio::io::scripts
