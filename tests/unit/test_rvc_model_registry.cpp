#include "rvc/RvcModelRegistry.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_rvc_registry_test_" + std::to_string(now));
        std::filesystem::create_directories(m_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

void writeBytes(const std::filesystem::path& path) {
    std::ofstream output{path, std::ios::binary};
    output << "test";
}

} // namespace

TEST_CASE("RVC model registry imports, lists, and deletes models", "[rvc][models]") {
    const TemporaryDirectory directory;
    const auto sourcePth = directory.path() / "hero.pth";
    const auto sourceIndex = directory.path() / "hero.index";
    writeBytes(sourcePth);
    writeBytes(sourceIndex);

    const voxstudio::rvc::RvcModelRegistry registry{directory.path() / "registry"};
    voxstudio::rvc::RvcModelImportRequest request;
    request.displayName = "Hero Voice";
    request.pthPath = sourcePth;
    request.indexPath = sourceIndex;
    request.sampleRate = 48000;

    auto imported = registry.importModel(request);
    REQUIRE(imported.hasValue());
    CHECK(imported.value().displayName == "Hero Voice");
    CHECK(imported.value().sampleRate == 48000);
    CHECK(std::filesystem::exists(imported.value().pthPath));
    CHECK(std::filesystem::exists(imported.value().indexPath));

    auto models = registry.listModels();
    REQUIRE(models.hasValue());
    REQUIRE(models.value().size() == 1);
    CHECK(models.value().front().id == imported.value().id);

    auto deleted = registry.deleteModel(imported.value().id);
    REQUIRE(deleted.hasValue());
    CHECK(deleted.value());
    models = registry.listModels();
    REQUIRE(models.hasValue());
    CHECK(models.value().empty());
}

TEST_CASE("RVC model registry rejects unsafe inputs", "[rvc][models]") {
    const TemporaryDirectory directory;
    const auto sourceOnnx = directory.path() / "wrong.onnx";
    writeBytes(sourceOnnx);

    const voxstudio::rvc::RvcModelRegistry registry{directory.path() / "registry"};
    voxstudio::rvc::RvcModelImportRequest request;
    request.displayName = "Wrong";
    request.pthPath = sourceOnnx;

    auto imported = registry.importModel(request);
    REQUIRE_FALSE(imported.hasValue());

    auto deleted = registry.deleteModel("..\\escape");
    REQUIRE_FALSE(deleted.hasValue());
}
