#include "io/scripts/CsvImporter.h"
#include "io/scripts/ScriptImporter.h"
#include "io/scripts/YarnImporter.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <string>

namespace {

[[nodiscard]] std::filesystem::path fixturePath(const std::string& relativePath) {
    return std::filesystem::path{VOXSTUDIO_TEST_FIXTURE_DIR} / relativePath;
}

} // namespace

TEST_CASE("script importer dispatches all supported fixture formats", "[scripts][import]") {
    using voxstudio::io::scripts::ScriptFormat;

    struct Fixture final {
        std::string path;
        ScriptFormat format;
        std::size_t expectedLines;
        std::string firstSpeaker;
        std::string firstScene;
    };

    const std::array fixtures{
        Fixture{"scripts/sample.txt", ScriptFormat::PlainText, 3, "Alice", "Village Gate"},
        Fixture{"scripts/sample.fountain", ScriptFormat::Fountain, 3, "ALICE", "Village Gate"},
        Fixture{"scripts/sample.rpy", ScriptFormat::Renpy, 3, "Alice", "village_gate"},
        Fixture{"scripts/sample.yarn.json", ScriptFormat::Yarn, 3, "Alice", "Village Gate"},
        Fixture{"scripts/sample.csv", ScriptFormat::Csv, 3, "Alice", "Village Gate"},
    };

    for (const auto& fixture : fixtures) {
        auto script = voxstudio::io::scripts::importScriptFile(fixturePath(fixture.path));
        REQUIRE(script.hasValue());
        CHECK(script.value().format == fixture.format);
        REQUIRE(script.value().lines.size() == fixture.expectedLines);
        CHECK(script.value().lines.front().speaker == fixture.firstSpeaker);
        CHECK(script.value().lines.front().text == "The caravan is late.");
        CHECK(script.value().lines.front().sceneTag == fixture.firstScene);
        CHECK((script.value().lines.back().speaker.empty() ||
               script.value().lines.back().speaker == "ALICE"));
    }
}

TEST_CASE("script importers reject unsupported or malformed input", "[scripts][import]") {
    auto unsupported =
        voxstudio::io::scripts::scriptFormatFromPath(fixturePath("scripts/sample.docx"));
    CHECK_FALSE(unsupported.hasValue());

    auto invalidCsv = voxstudio::io::scripts::parseCsvScript("speaker,scene\nAlice,Intro\n",
                                                             "memory.csv");
    CHECK_FALSE(invalidCsv.hasValue());

    auto invalidYarn = voxstudio::io::scripts::parseYarnScript(R"({"unknown":[]})",
                                                               "memory.json");
    CHECK_FALSE(invalidYarn.hasValue());
}
