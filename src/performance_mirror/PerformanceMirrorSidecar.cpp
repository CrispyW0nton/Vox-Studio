#include "performance_mirror/PerformanceMirrorSidecar.h"

#include "platform/win/AppPaths.h"
#include "rvc/RvcClient.h"

#include <Windows.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace voxstudio::performance_mirror {
namespace {

constexpr auto kLauncherName = "launch_performance_mirror.cmd";
constexpr auto kSetupName = "setup_performance_mirror.ps1";
constexpr auto kManifestName = "sidecar_manifest.json";

[[nodiscard]] core::Error mirrorError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] bool processRunning(HANDLE process) noexcept {
    DWORD exitCode = 0;
    return process != nullptr && GetExitCodeProcess(process, &exitCode) != 0 &&
           exitCode == STILL_ACTIVE;
}

[[nodiscard]] std::filesystem::path executableDirectory() {
    std::vector<wchar_t> buffer(32768U, L'\0');
    const auto length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0U || length >= buffer.size()) {
        return {};
    }
    return std::filesystem::path{std::wstring_view{buffer.data(), static_cast<std::size_t>(length)}}
        .parent_path();
}

[[nodiscard]] std::vector<std::filesystem::path> bundleCandidates() {
    std::vector<std::filesystem::path> candidates;
    const auto executable = executableDirectory();
    if (!executable.empty()) {
        candidates.push_back(executable / "performance_mirror_sidecar");
        candidates.push_back(executable / "third_party" / "performance_mirror_sidecar");
    }
    std::error_code error;
    const auto current = std::filesystem::current_path(error);
    if (!error) {
        candidates.push_back(current / "performance_mirror_sidecar");
        candidates.push_back(current / "third_party" / "performance_mirror_sidecar");
    }
    return candidates;
}

[[nodiscard]] bool filesMatch(const std::filesystem::path& first,
                              const std::filesystem::path& second) {
    std::ifstream firstStream{first, std::ios::binary};
    std::ifstream secondStream{second, std::ios::binary};
    if (!firstStream || !secondStream) {
        return false;
    }
    return std::equal(std::istreambuf_iterator<char>{firstStream}, std::istreambuf_iterator<char>{},
                      std::istreambuf_iterator<char>{secondStream},
                      std::istreambuf_iterator<char>{});
}

[[nodiscard]] std::string endpointText(const PerformanceMirrorSidecarConfig& config) {
    return "http://" + config.host + ":" + std::to_string(config.port);
}

} // namespace

class PerformanceMirrorSidecar::Impl final {
public:
    explicit Impl(PerformanceMirrorSidecarConfig config) : m_config(std::move(config)) {
        if (m_config.sidecarRoot.empty()) {
            m_config.sidecarRoot = defaultSidecarRoot();
        }
        if (m_config.engineRoot.empty()) {
            m_config.engineRoot = defaultEngineRoot();
        }
    }

    ~Impl() {
        stop();
    }

    [[nodiscard]] core::Expected<PerformanceMirrorSidecarStatus> start() {
        auto installed = ensureInstalled();
        if (!installed) {
            return installed.error();
        }

        const rvc::RvcClient existing{endpointText(m_config)};
        auto health = existing.health();
        if (health && health.value().ok && health.value().engine == "performance-mirror") {
            m_externalRunning = true;
            return status("Connected to the running Performance Mirror.");
        }
        m_externalRunning = false;

        const auto currentStatus = status();
        if (!currentStatus.engineInstalled) {
            return mirrorError("Performance Mirror needs its one-time local engine setup.");
        }
        if (processRunning(m_process)) {
            return status("Performance Mirror is warming up.");
        }

        closeHandles();
        const auto launcher = launcherPathForRoot(m_config.sidecarRoot);
        auto command = L"cmd.exe /d /c \"" + launcher.wstring() + L"\" --host " +
                       std::wstring{m_config.host.begin(), m_config.host.end()} + L" --port " +
                       std::to_wstring(m_config.port);
        std::vector<wchar_t> commandBuffer(command.begin(), command.end());
        commandBuffer.push_back(L'\0');

        STARTUPINFOW startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        PROCESS_INFORMATION processInfo{};
        const auto job = CreateJobObjectW(nullptr, nullptr);
        if (job == nullptr) {
            return mirrorError("Unable to create the Performance Mirror process group.");
        }
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jobInfo{};
        jobInfo.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jobInfo,
                                    sizeof(jobInfo)) == 0) {
            CloseHandle(job);
            return mirrorError("Unable to configure the Performance Mirror process group.");
        }
        const BOOL created =
            CreateProcessW(nullptr, commandBuffer.data(), nullptr, nullptr, FALSE,
                           CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                           m_config.sidecarRoot.wstring().c_str(), &startupInfo, &processInfo);
        if (created == 0) {
            CloseHandle(job);
            return mirrorError("Unable to start Performance Mirror.");
        }
        if (AssignProcessToJobObject(job, processInfo.hProcess) == 0) {
            TerminateProcess(processInfo.hProcess, 1);
            CloseHandle(processInfo.hThread);
            CloseHandle(processInfo.hProcess);
            CloseHandle(job);
            return mirrorError("Unable to own the Performance Mirror process group.");
        }
        if (ResumeThread(processInfo.hThread) == static_cast<DWORD>(-1)) {
            TerminateJobObject(job, 1);
            CloseHandle(processInfo.hThread);
            CloseHandle(processInfo.hProcess);
            CloseHandle(job);
            return mirrorError("Unable to start the Performance Mirror process group.");
        }

        m_job = job;
        m_process = processInfo.hProcess;
        m_thread = processInfo.hThread;
        const DWORD waitResult = WaitForSingleObject(m_process, 800);
        if (waitResult == WAIT_OBJECT_0) {
            closeHandles();
            return mirrorError("Performance Mirror exited during startup. Run its setup again.");
        }
        return status("Performance Mirror is warming up on the GPU.");
    }

    [[nodiscard]] PerformanceMirrorSidecarStatus status(std::string message = {}) const {
        const auto launcher = launcherPathForRoot(m_config.sidecarRoot);
        const auto setup = setupPathForRoot(m_config.sidecarRoot);
        const bool installed = std::filesystem::exists(launcher) &&
                               std::filesystem::exists(m_config.sidecarRoot / kManifestName);
        const bool engineInstalled =
            std::filesystem::exists(m_config.engineRoot / "real-time-gui.py") &&
            std::filesystem::exists(m_config.engineRoot / ".venv/Scripts/python.exe");
        if (message.empty()) {
            if (!installed) {
                message = "Performance Mirror sidecar is not installed.";
            } else if (!engineInstalled) {
                message = "Performance Mirror needs its one-time local engine setup.";
            } else {
                message = "Performance Mirror is installed.";
            }
        }
        return PerformanceMirrorSidecarStatus{
            installed,
            engineInstalled,
            processRunning(m_process) || m_externalRunning,
            m_config.sidecarRoot,
            m_config.engineRoot,
            launcher,
            setup,
            endpointText(m_config),
            std::move(message),
        };
    }

    void stop() noexcept {
        if (processRunning(m_process)) {
            if (m_job != nullptr) {
                TerminateJobObject(m_job, 0);
            } else {
                TerminateProcess(m_process, 0);
            }
            WaitForSingleObject(m_process, 3000);
        }
        closeHandles();
        m_externalRunning = false;
    }

private:
    [[nodiscard]] core::Expected<bool> ensureInstalled() const {
        const auto manifest = m_config.sidecarRoot / kManifestName;
        const bool installed = std::filesystem::exists(launcherPathForRoot(m_config.sidecarRoot)) &&
                               std::filesystem::exists(manifest);
        for (const auto& candidate : bundleCandidates()) {
            const auto candidateManifest = candidate / kManifestName;
            if (!std::filesystem::exists(candidate / kLauncherName) ||
                !std::filesystem::exists(candidateManifest)) {
                continue;
            }
            if (!installed || !filesMatch(candidateManifest, manifest)) {
                return installFromBundle(candidate, m_config.sidecarRoot);
            }
            break;
        }
        if (installed) {
            return true;
        }
        return mirrorError("Performance Mirror sidecar bundle was not found.");
    }

    void closeHandles() noexcept {
        if (m_thread != nullptr) {
            CloseHandle(m_thread);
            m_thread = nullptr;
        }
        if (m_process != nullptr) {
            CloseHandle(m_process);
            m_process = nullptr;
        }
        if (m_job != nullptr) {
            CloseHandle(m_job);
            m_job = nullptr;
        }
    }

    PerformanceMirrorSidecarConfig m_config;
    HANDLE m_job{nullptr};
    HANDLE m_process{nullptr};
    HANDLE m_thread{nullptr};
    bool m_externalRunning{false};
};

PerformanceMirrorSidecar::PerformanceMirrorSidecar() : PerformanceMirrorSidecar(defaultConfig()) {}

PerformanceMirrorSidecar::PerformanceMirrorSidecar(PerformanceMirrorSidecarConfig config)
    : m_impl(std::make_unique<Impl>(std::move(config))) {}

PerformanceMirrorSidecar::~PerformanceMirrorSidecar() = default;
PerformanceMirrorSidecar::PerformanceMirrorSidecar(PerformanceMirrorSidecar&&) noexcept = default;
PerformanceMirrorSidecar&
PerformanceMirrorSidecar::operator=(PerformanceMirrorSidecar&&) noexcept = default;

PerformanceMirrorSidecarConfig PerformanceMirrorSidecar::defaultConfig() {
    PerformanceMirrorSidecarConfig config;
    config.sidecarRoot = defaultSidecarRoot();
    config.engineRoot = defaultEngineRoot();
    return config;
}

std::filesystem::path PerformanceMirrorSidecar::defaultSidecarRoot() {
    const auto appData = platform::win::voxStudioDataPath();
    if (appData.has_value()) {
        return *appData / "performance_mirror_sidecar";
    }
    return std::filesystem::temp_directory_path() / "VoxStudio" / "performance_mirror_sidecar";
}

std::filesystem::path PerformanceMirrorSidecar::defaultEngineRoot() {
    const auto appData = platform::win::voxStudioDataPath();
    if (appData.has_value()) {
        return *appData / "engines" / "seed-vc";
    }
    return std::filesystem::temp_directory_path() / "VoxStudio" / "engines" / "seed-vc";
}

std::filesystem::path
PerformanceMirrorSidecar::launcherPathForRoot(const std::filesystem::path& sidecarRoot) {
    return sidecarRoot / kLauncherName;
}

std::filesystem::path
PerformanceMirrorSidecar::setupPathForRoot(const std::filesystem::path& sidecarRoot) {
    return sidecarRoot / kSetupName;
}

core::Expected<bool>
PerformanceMirrorSidecar::installFromBundle(const std::filesystem::path& bundleRoot,
                                            const std::filesystem::path& sidecarRoot) {
    if (!std::filesystem::exists(bundleRoot / kLauncherName) ||
        !std::filesystem::exists(bundleRoot / kManifestName)) {
        return mirrorError("Performance Mirror bundle is incomplete.");
    }

    std::error_code error;
    std::filesystem::create_directories(sidecarRoot, error);
    if (error) {
        return mirrorError(error.message());
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator{bundleRoot, error}) {
        if (error) {
            return mirrorError(error.message());
        }
        const auto relative = std::filesystem::relative(entry.path(), bundleRoot, error);
        if (error) {
            return mirrorError(error.message());
        }
        const auto destination = sidecarRoot / relative;
        if (entry.is_directory()) {
            std::filesystem::create_directories(destination, error);
        } else if (entry.is_regular_file()) {
            std::filesystem::create_directories(destination.parent_path(), error);
            if (!error) {
                std::filesystem::copy_file(entry.path(), destination,
                                           std::filesystem::copy_options::overwrite_existing,
                                           error);
            }
        }
        if (error) {
            return mirrorError(error.message());
        }
    }
    return true;
}

core::Expected<PerformanceMirrorSidecarStatus> PerformanceMirrorSidecar::start() {
    return m_impl->start();
}

PerformanceMirrorSidecarStatus PerformanceMirrorSidecar::status() const {
    return m_impl->status();
}

void PerformanceMirrorSidecar::stop() noexcept {
    m_impl->stop();
}

} // namespace voxstudio::performance_mirror
