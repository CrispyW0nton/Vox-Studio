#include "io/scripts/YarnImporter.h"

#include "io/scripts/PlainTextImporter.h"

#include <nlohmann/json.hpp>

#include <sstream>

namespace voxstudio::io::scripts {
namespace {

[[nodiscard]] std::string optionalString(const nlohmann::json& json, const char* key) {
    if (json.contains(key) && json.at(key).is_string()) {
        return json.at(key).get<std::string>();
    }
    return {};
}

void appendBodyLines(ParsedScript& target, const std::string& title, const std::string& body) {
    auto parsedBody = parsePlainTextScript(body, target.sourcePath);
    if (!parsedBody) {
        return;
    }

    for (auto line : parsedBody.value().lines) {
        if (line.sceneTag.empty()) {
            line.sceneTag = title;
        }
        target.lines.push_back(std::move(line));
    }
}

} // namespace

core::Expected<ParsedScript> parseYarnScript(const std::string& text,
                                             const std::filesystem::path& sourcePath) {
    ParsedScript script;
    script.sourcePath = sourcePath;
    script.format = ScriptFormat::Yarn;

    try {
        const auto json = nlohmann::json::parse(text);
        if (json.is_object() && json.contains("nodes") && json.at("nodes").is_array()) {
            for (const auto& node : json.at("nodes")) {
                if (!node.is_object()) {
                    continue;
                }
                appendBodyLines(script,
                                optionalString(node, "title"),
                                optionalString(node, "body"));
            }
        } else if (json.is_object() && json.contains("lines") && json.at("lines").is_array()) {
            for (const auto& lineJson : json.at("lines")) {
                if (!lineJson.is_object()) {
                    continue;
                }

                const auto textLine = optionalString(lineJson, "text");
                if (textLine.empty()) {
                    continue;
                }

                script.lines.push_back(ParsedScriptLine{optionalString(lineJson, "speaker"),
                                                        textLine,
                                                        optionalString(lineJson, "scene")});
            }
        } else {
            return core::makeError(core::ErrorCode::InvalidArgument,
                                   "Yarn JSON must contain nodes or lines.");
        }
    } catch (const nlohmann::json::exception& exception) {
        return core::makeError(core::ErrorCode::InvalidArgument, exception.what());
    }

    if (script.lines.empty()) {
        return core::makeError(core::ErrorCode::InvalidArgument,
                               "Yarn script did not contain importable dialogue lines.");
    }

    return script;
}

} // namespace voxstudio::io::scripts
