#include "rvc/RvcSidecar.h"

#include "platform/win/AppPaths.h"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace voxstudio::rvc {
namespace {

constexpr auto kLauncherFileName = "launch_rvc_sidecar.cmd";
constexpr auto kManifestFileName = "sidecar_manifest.json";

[[nodiscard]] core::Error sidecarError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] std::string endpointText(const RvcSidecarConfig& config) {
    return "http://" + config.host + ":" + std::to_string(config.port);
}

[[nodiscard]] std::wstring quoted(const std::filesystem::path& path) {
    return L"\"" + path.wstring() + L"\"";
}

[[nodiscard]] bool isCopyableSidecarFile(const std::filesystem::path& path) {
    const auto fileName = path.filename().wstring();
    return fileName != L".gitkeep";
}

[[nodiscard]] std::vector<std::filesystem::path> bundledSidecarCandidates() {
    std::vector<std::filesystem::path> candidates;
    std::error_code error;
    const auto current = std::filesystem::current_path(error);
    if (!error) {
        candidates.push_back(current / "rvc_sidecar");
        candidates.push_back(current / "third_party" / "rvc_sidecar");
    }
    return candidates;
}

[[nodiscard]] bool processStillRunning(HANDLE process) noexcept {
    if (process == nullptr) {
        return false;
    }

    DWORD exitCode = 0;
    if (GetExitCodeProcess(process, &exitCode) == 0) {
        return false;
    }
    return exitCode == STILL_ACTIVE;
}

} // namespace

class RvcSidecar::Impl final {
public:
    explicit Impl(RvcSidecarConfig config)
        : m_config(std::move(config)) {
        if (m_config.sidecarRoot.empty()) {
            m_config.sidecarRoot = defaultSidecarRoot();
        }
    }

    ~Impl() {
        stop();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    [[nodiscard]] core::Expected<RvcSidecarStatus> start() {
        if (isRunning()) {
            return status("RVC sidecar is already running.");
        }

        auto installed = ensureInstalled();
        if (!installed) {
            return installed.error();
        }

        const auto launcherPath = launcherPathForRoot(m_config.sidecarRoot);
        if (!std::filesystem::exists(launcherPath)) {
            return sidecarError("RVC sidecar launcher is missing: " + launcherPath.string());
        }

        closeProcessHandles();

        auto command = L"cmd.exe /d /c " + quoted(launcherPath) + L" --host " +
                       std::wstring{m_config.host.begin(), m_config.host.end()} + L" --port " +
                       std::to_wstring(m_config.port);

        STARTUPINFOW startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        PROCESS_INFORMATION processInfo{};

        std::vector<wchar_t> commandBuffer(command.begin(), command.end());
        commandBuffer.push_back(L'\0');

        const BOOL created = CreateProcessW(nullptr,
                                            commandBuffer.data(),
                                            nullptr,
                                            nullptr,
                                            FALSE,
                                            CREATE_NO_WINDOW,
                                            nullptr,
                                            m_config.sidecarRoot.wstring().c_str(),
                                            &startupInfo,
                                            &processInfo);
        if (created == 0) {
            return sidecarError("Unable to launch RVC sidecar process.");
        }

        m_process = processInfo.hProcess;
        m_thread = processInfo.hThread;
        const auto waitMs = static_cast<DWORD>(
            std::min<std::int64_t>(m_config.startupTimeout.count(), 1000));
        const DWORD waitResult = WaitForSingleObject(m_process, waitMs);
        if (waitResult == WAIT_OBJECT_0) {
            closeProcessHandles();
            return sidecarError("RVC sidecar exited during startup.");
        }

        return status("RVC sidecar process started.");
    }

    [[nodiscard]] core::Expected<RvcSidecarStatus> restartIfNeeded() {
        if (isRunning()) {
            return status("RVC sidecar is running.");
        }
        stop();
        return start();
    }

    [[nodiscard]] RvcSidecarStatus status(std::string message = {}) const {
        const auto launcherPath = launcherPathForRoot(m_config.sidecarRoot);
        const bool installed = std::filesystem::exists(launcherPath) &&
                               std::filesystem::exists(m_config.sidecarRoot /
                                                       kManifestFileName);
        if (message.empty()) {
            message = installed ? "RVC sidecar is installed." : "RVC sidecar is not installed.";
        }

        return RvcSidecarStatus{installed,
                                isRunning(),
                                m_config.sidecarRoot,
                                launcherPath,
                                endpointText(m_config),
                                std::move(message)};
    }

    void stop() noexcept {
        if (processStillRunning(m_process)) {
            TerminateProcess(m_process, 0);
            WaitForSingleObject(m_process, 3000);
        }
        closeProcessHandles();
    }

private:
    [[nodiscard]] bool isRunning() const noexcept {
        return processStillRunning(m_process);
    }

    [[nodiscard]] core::Expected<bool> ensureInstalled() const {
        const auto launcherPath = launcherPathForRoot(m_config.sidecarRoot);
        if (std::filesystem::exists(launcherPath)) {
            return true;
        }

        for (const auto& candidate : bundledSidecarCandidates()) {
            if (std::filesystem::exists(candidate / kLauncherFileName)) {
                return installFromBundle(candidate, m_config.sidecarRoot);
            }
        }

        return sidecarError("RVC sidecar bundle was not found.");
    }

    void closeProcessHandles() noexcept {
        if (m_thread != nullptr) {
            CloseHandle(m_thread);
            m_thread = nullptr;
        }
        if (m_process != nullptr) {
            CloseHandle(m_process);
            m_process = nullptr;
        }
    }

    RvcSidecarConfig m_config;
    HANDLE m_process{nullptr};
    HANDLE m_thread{nullptr};
};

RvcSidecar::RvcSidecar()
    : RvcSidecar(defaultConfig()) {}

RvcSidecar::RvcSidecar(RvcSidecarConfig config)
    : m_impl(std::make_unique<Impl>(std::move(config))) {}

RvcSidecar::~RvcSidecar() = default;
RvcSidecar::RvcSidecar(RvcSidecar&&) noexcept = default;
RvcSidecar& RvcSidecar::operator=(RvcSidecar&&) noexcept = default;

RvcSidecarConfig RvcSidecar::defaultConfig() {
    RvcSidecarConfig config;
    config.sidecarRoot = defaultSidecarRoot();
    return config;
}

std::filesystem::path RvcSidecar::defaultSidecarRoot() {
    const auto appData = platform::win::voxStudioDataPath();
    if (appData.has_value()) {
        return *appData / "rvc_sidecar";
    }
    return std::filesystem::temp_directory_path() / "VoxStudio" / "rvc_sidecar";
}

std::filesystem::path RvcSidecar::launcherPathForRoot(
    const std::filesystem::path& sidecarRoot) {
    return sidecarRoot / kLauncherFileName;
}

core::Expected<bool> RvcSidecar::installFromBundle(
    const std::filesystem::path& bundleRoot,
    const std::filesystem::path& sidecarRoot) {
    if (!std::filesystem::exists(bundleRoot / kLauncherFileName)) {
        return sidecarError("RVC sidecar bundle is missing its launcher.");
    }

    std::error_code error;
    std::filesystem::create_directories(sidecarRoot, error);
    if (error) {
        return sidecarError(error.message());
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator{bundleRoot, error}) {
        if (error) {
            return sidecarError(error.message());
        }
        if (!isCopyableSidecarFile(entry.path())) {
            continue;
        }

        const auto relative = std::filesystem::relative(entry.path(), bundleRoot, error);
        if (error) {
            return sidecarError(error.message());
        }
        const auto target = sidecarRoot / relative;
        if (entry.is_directory()) {
            std::filesystem::create_directories(target, error);
        } else if (entry.is_regular_file()) {
            std::filesystem::create_directories(target.parent_path(), error);
            if (!error) {
                std::filesystem::copy_file(entry.path(),
                                           target,
                                           std::filesystem::copy_options::overwrite_existing,
                                           error);
            }
        }
        if (error) {
            return sidecarError(error.message());
        }
    }

    return true;
}

core::Expected<RvcSidecarStatus> RvcSidecar::start() {
    return m_impl->start();
}

core::Expected<RvcSidecarStatus> RvcSidecar::restartIfNeeded() {
    return m_impl->restartIfNeeded();
}

RvcSidecarStatus RvcSidecar::status() const {
    return m_impl->status();
}

void RvcSidecar::stop() noexcept {
    m_impl->stop();
}

} // namespace voxstudio::rvc
