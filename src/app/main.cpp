#include "app/Logging.h"
#include "platform/win/SingleInstanceGuard.h"
#include "ui/MainWindow.h"

#include <QApplication>
#include <QString>

#include <cstdlib>

#include <spdlog/spdlog.h>

namespace {

constexpr auto kOrganizationName = "Vox Studio";
constexpr auto kApplicationName = "VoxStudio";
constexpr auto kDisplayName = "Vox Studio";
constexpr auto kSingleInstanceName = L"Local\\VoxStudio.SingleInstance";

} // namespace

int main(int argc, char* argv[]) {
    QApplication application(argc, argv);
    QApplication::setOrganizationName(kOrganizationName);
    QApplication::setApplicationName(kApplicationName);
    QApplication::setApplicationDisplayName(kDisplayName);
    QApplication::setApplicationVersion(QStringLiteral("0.1.0"));

    const bool loggingReady = voxstudio::app::initializeLogging();

    const voxstudio::platform::win::SingleInstanceGuard singleInstanceGuard{kSingleInstanceName};
    if (!singleInstanceGuard.isPrimaryInstance()) {
        if (loggingReady) {
            spdlog::warn("A second Vox Studio instance was started and will exit.");
        }
        voxstudio::app::shutdownLogging();
        return EXIT_SUCCESS;
    }

    voxstudio::ui::MainWindow mainWindow;
    mainWindow.show();

    const int exitCode = application.exec();

    if (loggingReady) {
        spdlog::info("Vox Studio exiting with code {}.", exitCode);
    }
    voxstudio::app::shutdownLogging();
    return exitCode;
}
