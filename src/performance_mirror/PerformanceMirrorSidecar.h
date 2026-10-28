#pragma once

#include "core/Expected.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace voxstudio::performance_mirror {

struct PerformanceMirrorSidecarConfig final {
    std::filesystem::path sidecarRoot;
    std::filesystem::path engineRoot;
    std::string host{"127.0.0.1"};
    std::uint16_t port{18910};
};

struct PerformanceMirrorSidecarStatus final {
    bool installed{false};
    bool engineInstalled{false};
    bool running{false};
    std::filesystem::path sidecarRoot;
    std::filesystem::path engineRoot;
    std::filesystem::path launcherPath;
    std::filesystem::path setupPath;
    std::string endpoint;
    std::string message;
};

class PerformanceMirrorSidecar final {
public:
    PerformanceMirrorSidecar();
    explicit PerformanceMirrorSidecar(PerformanceMirrorSidecarConfig config);
    ~PerformanceMirrorSidecar();

    PerformanceMirrorSidecar(const PerformanceMirrorSidecar&) = delete;
    PerformanceMirrorSidecar& operator=(const PerformanceMirrorSidecar&) = delete;
    PerformanceMirrorSidecar(PerformanceMirrorSidecar&&) noexcept;
    PerformanceMirrorSidecar& operator=(PerformanceMirrorSidecar&&) noexcept;

    [[nodiscard]] static PerformanceMirrorSidecarConfig defaultConfig();
    [[nodiscard]] static std::filesystem::path defaultSidecarRoot();
    [[nodiscard]] static std::filesystem::path defaultEngineRoot();
    [[nodiscard]] static std::filesystem::path
    launcherPathForRoot(const std::filesystem::path& sidecarRoot);
    [[nodiscard]] static std::filesystem::path
    setupPathForRoot(const std::filesystem::path& sidecarRoot);
    [[nodiscard]] static core::Expected<bool>
    installFromBundle(const std::filesystem::path& bundleRoot,
                      const std::filesystem::path& sidecarRoot);

    [[nodiscard]] core::Expected<PerformanceMirrorSidecarStatus> start();
    [[nodiscard]] PerformanceMirrorSidecarStatus status() const;
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voxstudio::performance_mirror
