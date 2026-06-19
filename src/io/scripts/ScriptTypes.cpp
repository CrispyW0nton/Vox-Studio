#include "io/scripts/ScriptTypes.h"

namespace voxstudio::io::scripts {

std::string scriptFormatName(const ScriptFormat format) {
    switch (format) {
    case ScriptFormat::PlainText:
        return "Plain Text";
    case ScriptFormat::Fountain:
        return "Fountain";
    case ScriptFormat::Renpy:
        return "Ren'Py";
    case ScriptFormat::Yarn:
        return "Yarn";
    case ScriptFormat::Csv:
        return "CSV";
    }

    return "Plain Text";
}

std::string scriptFormatStorageName(const ScriptFormat format) {
    switch (format) {
    case ScriptFormat::PlainText:
        return "txt";
    case ScriptFormat::Fountain:
        return "fountain";
    case ScriptFormat::Renpy:
        return "rpy";
    case ScriptFormat::Yarn:
        return "yarn";
    case ScriptFormat::Csv:
        return "csv";
    }

    return "txt";
}

} // namespace voxstudio::io::scripts
