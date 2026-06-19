#include "io/scripts/CsvImporter.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

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

[[nodiscard]] std::string lowerCopy(std::string value) {
    std::ranges::transform(value, value.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

[[nodiscard]] std::vector<std::string> parseCsvRecord(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
            continue;
        }
        if (character == ',' && !quoted) {
            fields.push_back(trimCopy(field));
            field.clear();
            continue;
        }
        field.push_back(character);
    }
    fields.push_back(trimCopy(field));
    return fields;
}

[[nodiscard]] int findColumn(const std::vector<std::string>& headers,
                             const std::vector<std::string>& names) {
    for (std::size_t index = 0; index < headers.size(); ++index) {
        const auto header = lowerCopy(trimCopy(headers[index]));
        if (std::ranges::find(names, header) != names.end()) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

[[nodiscard]] std::string fieldAt(const std::vector<std::string>& fields, const int index) {
    if (index < 0 || static_cast<std::size_t>(index) >= fields.size()) {
        return {};
    }
    return fields[static_cast<std::size_t>(index)];
}

} // namespace

core::Expected<ParsedScript> parseCsvScript(const std::string& text,
                                            const std::filesystem::path& sourcePath) {
    ParsedScript script;
    script.sourcePath = sourcePath;
    script.format = ScriptFormat::Csv;

    std::istringstream stream{text};
    std::string rawLine;
    if (!std::getline(stream, rawLine)) {
        return core::makeError(core::ErrorCode::InvalidArgument, "CSV script is empty.");
    }

    const auto headers = parseCsvRecord(rawLine);
    const int speakerColumn = findColumn(headers, {"speaker", "character", "name"});
    const int textColumn = findColumn(headers, {"text", "line", "dialogue"});
    const int sceneColumn = findColumn(headers, {"scene", "scene_tag", "scene tag"});
    if (textColumn < 0) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "CSV script requires a text, line, or dialogue column.");
    }

    while (std::getline(stream, rawLine)) {
        const auto fields = parseCsvRecord(rawLine);
        const auto textLine = fieldAt(fields, textColumn);
        if (textLine.empty()) {
            continue;
        }
        script.lines.push_back(ParsedScriptLine{fieldAt(fields, speakerColumn),
                                                textLine,
                                                fieldAt(fields, sceneColumn)});
    }

    if (script.lines.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "CSV script did not contain importable dialogue lines.");
    }

    return script;
}

} // namespace voxstudio::io::scripts
