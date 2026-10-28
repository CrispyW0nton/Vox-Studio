#include "performance_mirror/PerformanceMirrorSidecar.h"

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
                 ("voxstudio_performance_mirror_test_" + std::to_string(now));
        std::filesystem::create_directories(m_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

void writeText(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream{path, std::ios::binary};
    stream << text;
}

} // namespace

TEST_CASE("Performance Mirror installs its lightweight sidecar bundle") {
    const TemporaryDirectory directory;
    const auto bundleRoot = directory.path() / "bundle";
    const auto installRoot = directory.path() / "installed";
    writeText(bundleRoot / "launch_performance_mirror.cmd", "@echo off\r\nexit /b 0\r\n");
    writeText(bundleRoot / "setup_performance_mirror.ps1", "Write-Host ready\r\n");
    writeText(bundleRoot / "sidecar_manifest.json", R"({"version":1})");

    auto installed = voxstudio::performance_mirror::PerformanceMirrorSidecar::installFromBundle(
        bundleRoot, installRoot);

    REQUIRE(installed.hasValue());
    CHECK(std::filesystem::exists(installRoot / "launch_performance_mirror.cmd"));
    CHECK(std::filesystem::exists(installRoot / "setup_performance_mirror.ps1"));
}

TEST_CASE("Performance Mirror status distinguishes sidecar and engine installation") {
    const TemporaryDirectory directory;
    const auto sidecarRoot = directory.path() / "sidecar";
    const auto engineRoot = directory.path() / "engine";
    writeText(sidecarRoot / "launch_performance_mirror.cmd", "@echo off\r\nexit /b 0\r\n");
    writeText(sidecarRoot / "setup_performance_mirror.ps1", "Write-Host ready\r\n");
    writeText(sidecarRoot / "sidecar_manifest.json", R"({"version":1})");

    voxstudio::performance_mirror::PerformanceMirrorSidecarConfig config;
    config.sidecarRoot = sidecarRoot;
    config.engineRoot = engineRoot;
    const voxstudio::performance_mirror::PerformanceMirrorSidecar sidecar{std::move(config)};

    auto status = sidecar.status();
    CHECK(status.installed);
    CHECK_FALSE(status.engineInstalled);
    CHECK(status.endpoint == "http://127.0.0.1:18910");

    writeText(engineRoot / "real-time-gui.py", "# Seed-VC");
    writeText(engineRoot / ".venv/Scripts/python.exe", "");
    status = sidecar.status();
    CHECK(status.engineInstalled);
}
