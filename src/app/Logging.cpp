#include "app/Logging.h"

#include "platform/win/AppPaths.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <utility>

namespace voxstudio::app {
namespace {

constexpr auto kLoggerName = "voxstudio";
constexpr auto kLogFileName = "voxstudio.log";

} // namespace

bool initializeLogging() noexcept {
    try {
        const auto logDirectory = platform::win::voxStudioLogPath();
        if (!logDirectory.has_value()) {
            return false;
        }

        std::filesystem::create_directories(*logDirectory);
        auto logger =
            spdlog::basic_logger_mt(kLoggerName, (*logDirectory / kLogFileName).string(), true);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

#if defined(NDEBUG)
        logger->set_level(spdlog::level::info);
#else
        logger->set_level(spdlog::level::debug);
#endif

        spdlog::set_default_logger(std::move(logger));
        spdlog::flush_on(spdlog::level::warn);
        spdlog::info("Vox Studio logging initialized.");
        return true;
    } catch (const spdlog::spdlog_ex&) {
        return false;
    } catch (const std::filesystem::filesystem_error&) {
        return false;
    } catch (...) {
        return false;
    }
}

void shutdownLogging() noexcept {
    spdlog::shutdown();
}

} // namespace voxstudio::app
