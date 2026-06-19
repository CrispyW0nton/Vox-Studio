#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace voxstudio::io::scripts {

enum class ScriptFormat {
    PlainText,
    Fountain,
    Renpy,
    Yarn,
    Csv,
};

struct ParsedScriptLine final {
    std::string speaker;
    std::string text;
    std::string sceneTag;
};

struct ParsedScript final {
    std::filesystem::path sourcePath;
    ScriptFormat format{ScriptFormat::PlainText};
    std::vector<ParsedScriptLine> lines;
};

[[nodiscard]] std::string scriptFormatName(ScriptFormat format);
[[nodiscard]] std::string scriptFormatStorageName(ScriptFormat format);

} // namespace voxstudio::io::scripts
