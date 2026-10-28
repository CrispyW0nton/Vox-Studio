#pragma once

#include "core/Expected.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace voxstudio::voxcpm {

struct VoxCpmSidecarStatus final {
    bool installed{false};
    bool running{false};
    std::filesystem::path root;
    std::string endpoint;
    std::string message;
};

class VoxCpmSidecar final {
public:
    VoxCpmSidecar();
    ~VoxCpmSidecar();

    VoxCpmSidecar(const VoxCpmSidecar&) = delete;
    VoxCpmSidecar& operator=(const VoxCpmSidecar&) = delete;

    [[nodiscard]] static std::filesystem::path defaultRoot();
    [[nodiscard]] core::Expected<VoxCpmSidecarStatus> start();
    [[nodiscard]] VoxCpmSidecarStatus status() const;
    void stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voxstudio::voxcpm
