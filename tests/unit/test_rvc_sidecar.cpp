#include "rvc/RvcSidecar.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace {

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 ("voxstudio_rvc_sidecar_test_" + std::to_string(now));
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

void writeText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output{path, std::ios::trunc};
    output << text;
}

} // namespace

TEST_CASE("RVC sidecar installs launcher metadata from bundle", "[rvc][sidecar]") {
    const TemporaryDirectory directory;
    const auto bundleRoot = directory.path() / "bundle";
    const auto installRoot = directory.path() / "installed";
    std::filesystem::create_directories(bundleRoot);
    writeText(bundleRoot / "launch_rvc_sidecar.cmd", "@echo off\r\nexit /b 0\r\n");
    writeText(bundleRoot / "sidecar_manifest.json", R"({"name":"w-okada/voice-changer"})");

    auto installed = voxstudio::rvc::RvcSidecar::installFromBundle(bundleRoot, installRoot);
    REQUIRE(installed.hasValue());
    CHECK(installed.value());
    CHECK(std::filesystem::exists(installRoot / "launch_rvc_sidecar.cmd"));
    CHECK(std::filesystem::exists(installRoot / "sidecar_manifest.json"));

    voxstudio::rvc::RvcSidecarConfig config;
    config.sidecarRoot = installRoot;
    const voxstudio::rvc::RvcSidecar sidecar{std::move(config)};
    const auto status = sidecar.status();
    CHECK(status.installed);
    CHECK_FALSE(status.running);
    CHECK(status.endpoint == "http://127.0.0.1:18888");
}
