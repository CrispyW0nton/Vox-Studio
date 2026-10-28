#include "voxcpm/VoxCpmSidecar.h"

#include "platform/win/AppPaths.h"
#include "voxcpm/VoxCpmClient.h"

#include <Windows.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace voxstudio::voxcpm {
namespace {

constexpr auto kLauncherName = "launch_voxcpm_sidecar.cmd";
constexpr auto kManifestName = "sidecar_manifest.json";
constexpr auto kEndpoint = "http://127.0.0.1:18990";

[[nodiscard]] core::Error sidecarError(const std::string& message) {
    return core::makeError(core::ErrorCode::FileSystemFailure, message);
}

[[nodiscard]] bool processRunning(HANDLE process) noexcept {
    DWORD exitCode = 0;
    return process != nullptr && GetExitCodeProcess(process, &exitCode) != 0 &&
           exitCode == STILL_ACTIVE;
}

[[nodiscard]] std::filesystem::path executableDirectory() {
    std::vector<wchar_t> pathBuffer(32768U, L'\0');
    const auto length = GetModuleFileNameW(
        nullptr,
        pathBuffer.data(),
        static_cast<DWORD>(pathBuffer.size()));
    if (length == 0U || length >= pathBuffer.size()) {
        return {};
    }
    return std::filesystem::path{
               std::wstring_view{pathBuffer.data(), static_cast<std::size_t>(length)}}
        .parent_path();
}

[[nodiscard]] std::vector<std::filesystem::path> bundleCandidates() {
    std::vector<std::filesystem::path> candidates;
    const auto executable = executableDirectory();
    if (!executable.empty()) {
        candidates.push_back(executable / "voxcpm_sidecar");
        candidates.push_back(executable / "third_party" / "voxcpm_sidecar");
    }
    std::error_code error;
    const auto current = std::filesystem::current_path(error);
    if (!error) {
        candidates.push_back(current / "voxcpm_sidecar");
        candidates.push_back(current / "third_party" / "voxcpm_sidecar");
    }
    return candidates;
}

[[nodiscard]] core::Expected<bool> installBundle(const std::filesystem::path& source,
                                                 const std::filesystem::path& target) {
    std::error_code error;
    std::filesystem::create_directories(target, error);
    if (error) {
        return sidecarError(error.message());
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator{source, error}) {
        if (error) {
            return sidecarError(error.message());
        }
        const auto relative = std::filesystem::relative(entry.path(), source, error);
        if (error) {
            return sidecarError(error.message());
        }
        const auto destination = target / relative;
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
            return sidecarError(error.message());
        }
    }
    return true;
}

} // namespace

class VoxCpmSidecar::Impl final {
public:
    ~Impl() {
        stop();
    }

    [[nodiscard]] core::Expected<VoxCpmSidecarStatus> start() {
        if (processRunning(m_process)) {
            return status("VoxCPM2 is ready.");
        }
        const auto root = defaultRoot();
        for (const auto& candidate : bundleCandidates()) {
            std::error_code equivalentError;
            const bool sameBundle =
                std::filesystem::equivalent(candidate, root, equivalentError);
            if (!std::filesystem::exists(candidate / kLauncherName) ||
                (!equivalentError && sameBundle)) {
                continue;
            }
            auto copied = installBundle(candidate, root);
            if (!copied) {
                return copied.error();
            }
            break;
        }

        const VoxCpmClient existingService{kEndpoint};
        auto health = existingService.health();
        if (health && health.value().ok) {
            m_externalRunning = true;
            return status("Connected to the running VoxCPM2 service.");
        }
        m_externalRunning = false;

        const auto launcher = root / kLauncherName;
        const auto manifest = root / kManifestName;
        if (!std::filesystem::exists(launcher) || !std::filesystem::exists(manifest)) {
            bool installed = false;
            for (const auto& candidate : bundleCandidates()) {
                if (!std::filesystem::exists(candidate / kLauncherName)) {
                    continue;
                }
                auto copied = installBundle(candidate, root);
                if (!copied) {
                    return copied.error();
                }
                installed = true;
                break;
            }
            if (!installed) {
                return sidecarError(
                    "VoxCPM2 service is not installed. Re-run the Vox Studio setup.");
            }
        }

        stop();
        auto command =
            L"cmd.exe /d /c \"" + launcher.wstring() + L"\" --host 127.0.0.1 --port 18990";
        std::vector<wchar_t> commandBuffer(command.begin(), command.end());
        commandBuffer.push_back(L'\0');

        STARTUPINFOW startupInfo{};
        startupInfo.cb = sizeof(startupInfo);
        PROCESS_INFORMATION processInfo{};
        const BOOL created =
            CreateProcessW(nullptr, commandBuffer.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                           nullptr, root.wstring().c_str(), &startupInfo, &processInfo);
        if (created == 0) {
            return sidecarError("Unable to start the local VoxCPM2 service.");
        }

        m_process = processInfo.hProcess;
        m_thread = processInfo.hThread;
        const DWORD waitResult = WaitForSingleObject(m_process, 800);
        if (waitResult == WAIT_OBJECT_0) {
            stop();
            return sidecarError(
                "VoxCPM2 exited during startup. Check the local engine installation.");
        }
        return status("VoxCPM2 is loading. The first phrase can take longer.");
    }

    [[nodiscard]] VoxCpmSidecarStatus status(std::string message = {}) const {
        const auto root = defaultRoot();
        const bool installed = std::filesystem::exists(root / kLauncherName) &&
                               std::filesystem::exists(root / kManifestName);
        if (message.empty()) {
            message = installed ? "VoxCPM2 is installed." : "VoxCPM2 is not installed.";
        }
        return VoxCpmSidecarStatus{installed, processRunning(m_process) || m_externalRunning, root,
                                   kEndpoint, std::move(message)};
    }

    void stop() noexcept {
        if (processRunning(m_process)) {
            TerminateProcess(m_process, 0);
            WaitForSingleObject(m_process, 3000);
        }
        if (m_thread != nullptr) {
            CloseHandle(m_thread);
            m_thread = nullptr;
        }
        if (m_process != nullptr) {
            CloseHandle(m_process);
            m_process = nullptr;
        }
        m_externalRunning = false;
    }

private:
    HANDLE m_process{nullptr};
    HANDLE m_thread{nullptr};
    bool m_externalRunning{false};
};

VoxCpmSidecar::VoxCpmSidecar() : m_impl(std::make_unique<Impl>()) {}

VoxCpmSidecar::~VoxCpmSidecar() = default;

std::filesystem::path VoxCpmSidecar::defaultRoot() {
    const auto appData = platform::win::voxStudioDataPath();
    if (appData.has_value()) {
        return *appData / "voxcpm_sidecar";
    }
    return std::filesystem::temp_directory_path() / "VoxStudio" / "voxcpm_sidecar";
}

core::Expected<VoxCpmSidecarStatus> VoxCpmSidecar::start() {
    return m_impl->start();
}

VoxCpmSidecarStatus VoxCpmSidecar::status() const {
    return m_impl->status();
}

void VoxCpmSidecar::stop() noexcept {
    m_impl->stop();
}

} // namespace voxstudio::voxcpm
