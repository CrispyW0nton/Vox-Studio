#pragma once

#include "core/Expected.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace voxstudio::rvc {

struct RvcSidecarConfig final {
    std::filesystem::path sidecarRoot;
    std::string host{"127.0.0.1"};
    std::uint16_t port{18888};
    std::chrono::milliseconds startupTimeout{std::chrono::seconds{20}};
};

struct RvcSidecarStatus final {
    bool installed{false};
    bool running{false};
    std::filesystem::path sidecarRoot;
    std::filesystem::path launcherPath;
    std::string endpoint;
    std::string message;
};

class RvcSidecar final {
public:
    RvcSidecar();
    explicit RvcSidecar(RvcSidecarConfig config);
    ~RvcSidecar();

    RvcSidecar(const RvcSidecar&) = delete;
    RvcSidecar& operator=(const RvcSidecar&) = delete;
    RvcSidecar(RvcSidecar&&) noexcept;
    RvcSidecar& operator=(RvcSidecar&&) noexcept;

    [[nodiscard]] static RvcSidecarConfig defaultConfig();
    [[nodiscard]] static std::filesystem::path defaultSidecarRoot();
    [[nodiscard]] static std::filesystem::path launcherPathForRoot(
        const std::filesystem::path& sidecarRoot);
    [[nodiscard]] static core::Expected<bool> installFromBundle(
        const std::filesystem::path& bundleRoot,
        const std::filesystem::path& sidecarRoot);

    [[nodiscard]] core::Expected<RvcSidecarStatus> start();
    [[nodiscard]] core::Expected<RvcSidecarStatus> restartIfNeeded();
    [[nodiscard]] RvcSidecarStatus status() const;
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voxstudio::rvc
