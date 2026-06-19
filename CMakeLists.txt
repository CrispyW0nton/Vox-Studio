cmake_minimum_required(VERSION 3.27)

project(
    VoxStudio
    VERSION 0.1.0
    DESCRIPTION "Native Windows AI voice acting studio for indie game developers"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(QtSetup)
include(Warnings)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Qt6 6.7 REQUIRED COMPONENTS Concurrent Core Test Widgets)
find_package(SQLiteCpp CONFIG REQUIRED)
find_package(Opus CONFIG REQUIRED)
find_package(cpr CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_path(MINIAUDIO_INCLUDE_DIR miniaudio.h REQUIRED)
find_path(SNDFILE_INCLUDE_DIR sndfile.h REQUIRED)
find_path(SOXR_INCLUDE_DIR soxr.h REQUIRED)
find_library(SNDFILE_LIBRARY sndfile REQUIRED)
find_library(SOXR_LIBRARY soxr REQUIRED)

qt_standard_project_setup(REQUIRES 6.7)

set(VOXSTUDIO_CORE_SOURCES
    src/audio/AudioEngine.cpp
    src/audio/AudioEngine.h
    src/audio/Capture.cpp
    src/audio/Capture.h
    src/audio/AudioFile.cpp
    src/audio/AudioFile.h
    src/audio/LatencyProbe.cpp
    src/audio/LatencyProbe.h
    src/audio/AudioPreview.cpp
    src/audio/AudioPreview.h
    src/audio/RingBuffer.cpp
    src/audio/RingBuffer.h
    src/audio/AudioTypes.h
    src/audio/Resampler.cpp
    src/audio/Resampler.h
    src/core/Error.h
    src/core/Expected.h
    src/core/Project.cpp
    src/core/Project.h
    src/core/Sequence.cpp
    src/core/Sequence.h
    src/core/SequenceRenderer.cpp
    src/core/SequenceRenderer.h
    src/core/TakeManager.cpp
    src/core/TakeManager.h
    src/db/Database.cpp
    src/db/Database.h
    src/db/Migrations.cpp
    src/db/Migrations.h
    src/db/ProjectRepository.cpp
    src/db/ProjectRepository.h
    src/db/ScriptRepository.cpp
    src/db/ScriptRepository.h
    src/db/SequenceRepository.cpp
    src/db/SequenceRepository.h
    src/db/TakeRepository.cpp
    src/db/TakeRepository.h
    src/db/VoiceRepository.cpp
    src/db/VoiceRepository.h
    src/dsp/NoiseSuppressor.cpp
    src/dsp/NoiseSuppressor.h
    src/dsp/Vad.cpp
    src/dsp/Vad.h
    src/io/scripts/CsvImporter.cpp
    src/io/scripts/CsvImporter.h
    src/io/scripts/FountainImporter.cpp
    src/io/scripts/FountainImporter.h
    src/io/scripts/PlainTextImporter.cpp
    src/io/scripts/PlainTextImporter.h
    src/io/scripts/RenpyImporter.cpp
    src/io/scripts/RenpyImporter.h
    src/io/scripts/ScriptImporter.cpp
    src/io/scripts/ScriptImporter.h
    src/io/scripts/ScriptTypes.cpp
    src/io/scripts/ScriptTypes.h
    src/io/scripts/YarnImporter.cpp
    src/io/scripts/YarnImporter.h
    src/net/elevenlabs/Client.cpp
    src/net/elevenlabs/Client.h
    src/net/elevenlabs/Models.cpp
    src/net/elevenlabs/Models.h
    src/net/elevenlabs/StsApi.cpp
    src/net/elevenlabs/StsApi.h
    src/net/elevenlabs/TtsApi.cpp
    src/net/elevenlabs/TtsApi.h
    src/net/elevenlabs/VoicesApi.cpp
    src/net/elevenlabs/VoicesApi.h
    src/platform/win/AppPaths.cpp
    src/platform/win/AppPaths.h
    src/rvc/F0Extractor.cpp
    src/rvc/F0Extractor.h
    src/rvc/HubertEncoder.cpp
    src/rvc/HubertEncoder.h
    src/rvc/NativeRvcAudio.cpp
    src/rvc/NativeRvcAudio.h
    src/rvc/OnnxRvcEngine.cpp
    src/rvc/OnnxRvcEngine.h
    src/rvc/RvcClient.cpp
    src/rvc/RvcClient.h
    src/rvc/RvcModelRegistry.cpp
    src/rvc/RvcModelRegistry.h
    src/rvc/RvcSidecar.cpp
    src/rvc/RvcSidecar.h
    src/secrets/DpapiVault.cpp
    src/secrets/DpapiVault.h
)

add_library(VoxStudioCore STATIC ${VOXSTUDIO_CORE_SOURCES})
target_include_directories(VoxStudioCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(
    VoxStudioCore
    SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include"
)
target_include_directories(
    VoxStudioCore
    PRIVATE "${MINIAUDIO_INCLUDE_DIR}"
            "${SNDFILE_INCLUDE_DIR}"
            "${SOXR_INCLUDE_DIR}"
)
target_link_libraries(
    VoxStudioCore
    PUBLIC SQLiteCpp
           Opus::opus
           cpr::cpr
           nlohmann_json::nlohmann_json
           "${SNDFILE_LIBRARY}"
           "${SOXR_LIBRARY}"
           Crypt32
           Ole32
           Shell32
)
target_compile_definitions(VoxStudioCore PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
voxstudio_apply_project_warnings(VoxStudioCore)

set(VOXSTUDIO_APP_SOURCES
    src/app/Logging.cpp
    src/app/Logging.h
    src/app/main.cpp
    src/platform/win/SingleInstanceGuard.cpp
    src/platform/win/SingleInstanceGuard.h
    src/ui/MainWindow.cpp
    src/ui/MainWindow.h
    src/ui/CharacterAssignmentDialog.cpp
    src/ui/CharacterAssignmentDialog.h
    src/ui/CloneVoiceDialog.cpp
    src/ui/CloneVoiceDialog.h
    src/ui/HotkeyManager.cpp
    src/ui/HotkeyManager.h
    src/ui/LiveAudioProcessor.cpp
    src/ui/LiveAudioProcessor.h
    src/ui/LiveMicPanel.cpp
    src/ui/LiveMicPanel.h
    src/ui/NewProjectWizard.cpp
    src/ui/NewProjectWizard.h
    src/ui/ProjectHud.cpp
    src/ui/ProjectHud.h
    src/ui/RecentProjectsModel.cpp
    src/ui/RecentProjectsModel.h
    src/ui/RvcModelManagerDialog.cpp
    src/ui/RvcModelManagerDialog.h
    src/ui/SettingsDialog.cpp
    src/ui/SettingsDialog.h
    src/ui/ScriptViewerPanel.cpp
    src/ui/ScriptViewerPanel.h
    src/ui/TakeListWidget.cpp
    src/ui/TakeListWidget.h
    src/ui/TimelinePanel.cpp
    src/ui/TimelinePanel.h
    src/ui/VoiceLibraryPanel.cpp
    src/ui/VoiceLibraryPanel.h
)

qt_add_executable(VoxStudio MANUAL_FINALIZATION WIN32 ${VOXSTUDIO_APP_SOURCES})

target_include_directories(VoxStudio PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(
    VoxStudio
    PRIVATE Qt6::Concurrent
            Qt6::Core
            Qt6::Widgets
            VoxStudioCore
            spdlog::spdlog
            Ole32
            Shell32
)

target_compile_definitions(
    VoxStudio
    PRIVATE UNICODE
            _UNICODE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
)

voxstudio_apply_project_warnings(VoxStudio)

add_executable(VoxStudioRvcNativeSmoke tools/rvc_native_smoke.cpp)
target_include_directories(VoxStudioRvcNativeSmoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(VoxStudioRvcNativeSmoke PRIVATE VoxStudioCore nlohmann_json::nlohmann_json)
target_compile_definitions(
    VoxStudioRvcNativeSmoke
    PRIVATE NOMINMAX
            WIN32_LEAN_AND_MEAN
)
voxstudio_apply_project_warnings(VoxStudioRvcNativeSmoke)

if(WIN32)
    if(NOT TARGET Qt6::QWindowsIntegrationPlugin)
        message(FATAL_ERROR "Qt Windows platform plugin target is required for deployment.")
    endif()

    set(QT_WINDOWS_PLUGIN_NAME "$<TARGET_FILE_NAME:Qt6::QWindowsIntegrationPlugin>")
    set(VOXSTUDIO_QT_PLATFORM_PLUGIN
        "$<TARGET_FILE_DIR:VoxStudio>/platforms/${QT_WINDOWS_PLUGIN_NAME}"
    )
    add_custom_command(
        TARGET VoxStudio
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:VoxStudio>/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:Qt6::QWindowsIntegrationPlugin>"
                "${VOXSTUDIO_QT_PLATFORM_PLUGIN}"
        COMMENT "Deploying Qt Windows platform plugin for VoxStudio"
        VERBATIM
    )
endif()

set_target_properties(
    VoxStudio
    PROPERTIES OUTPUT_NAME "VoxStudio"
               VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:VoxStudio>"
)

if(BUILD_TESTING)
    find_package(Catch2 3 REQUIRED CONFIG)
    find_package(Python3 COMPONENTS Interpreter)

    add_executable(
        VoxStudioUnitTests
        tests/unit/test_dpapi_vault.cpp
        tests/unit/test_elevenlabs_client.cpp
        tests/unit/test_elevenlabs_models.cpp
        tests/unit/test_live_audio.cpp
        tests/unit/test_onnx_rvc_engine.cpp
        tests/unit/test_project_repository.cpp
        tests/unit/test_rvc_client.cpp
        tests/unit/test_rvc_model_registry.cpp
        tests/unit/test_rvc_sidecar.cpp
        tests/unit/test_script_importers.cpp
        tests/unit/test_script_repository.cpp
        tests/unit/test_sequence_renderer.cpp
        tests/unit/test_sequence_repository.cpp
        tests/unit/test_sts_api.cpp
        tests/unit/test_take_manager.cpp
        tests/unit/test_tts_api.cpp
        tests/unit/test_smoke.cpp
        tests/unit/test_voice_repository.cpp
        tests/unit/test_voices_api.cpp
    )
    target_link_libraries(VoxStudioUnitTests PRIVATE Catch2::Catch2WithMain VoxStudioCore)
    target_compile_definitions(
        VoxStudioUnitTests
        PRIVATE VOXSTUDIO_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )
    voxstudio_apply_project_warnings(VoxStudioUnitTests)

    add_test(NAME VoxStudioUnitTests COMMAND VoxStudioUnitTests)

    add_executable(
        VoxStudioQtTests
        tests/unit/test_script_viewer_panel.cpp
        src/ui/CharacterAssignmentDialog.cpp
        src/ui/CharacterAssignmentDialog.h
        src/ui/ScriptViewerPanel.cpp
        src/ui/ScriptViewerPanel.h
        src/ui/TakeListWidget.cpp
        src/ui/TakeListWidget.h
    )
    target_include_directories(VoxStudioQtTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        VoxStudioQtTests
        PRIVATE Qt6::Concurrent
                Qt6::Core
                Qt6::Test
                Qt6::Widgets
                VoxStudioCore
    )
    target_compile_definitions(
        VoxStudioQtTests
        PRIVATE VOXSTUDIO_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )
    voxstudio_apply_project_warnings(VoxStudioQtTests)
    if(WIN32)
        set(VOXSTUDIO_QT_TEST_PLATFORM_PLUGIN
            "$<TARGET_FILE_DIR:VoxStudioQtTests>/platforms/${QT_WINDOWS_PLUGIN_NAME}"
        )
        add_custom_command(
            TARGET VoxStudioQtTests
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:VoxStudioQtTests>/platforms"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:Qt6::QWindowsIntegrationPlugin>"
                    "${VOXSTUDIO_QT_TEST_PLATFORM_PLUGIN}"
            COMMENT "Deploying Qt Windows platform plugin for VoxStudioQtTests"
            VERBATIM
        )
    endif()
    add_test(NAME VoxStudioQtTests COMMAND VoxStudioQtTests)

    add_executable(
        VoxStudioTimelineQtTests
        tests/unit/test_timeline_panel.cpp
        src/ui/HotkeyManager.cpp
        src/ui/HotkeyManager.h
        src/ui/TimelinePanel.cpp
        src/ui/TimelinePanel.h
    )
    target_include_directories(VoxStudioTimelineQtTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        VoxStudioTimelineQtTests
        PRIVATE Qt6::Concurrent
                Qt6::Core
                Qt6::Test
                Qt6::Widgets
                VoxStudioCore
    )
    target_compile_definitions(
        VoxStudioTimelineQtTests
        PRIVATE VOXSTUDIO_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )
    voxstudio_apply_project_warnings(VoxStudioTimelineQtTests)
    if(WIN32)
        set(VOXSTUDIO_TIMELINE_QT_TEST_PLATFORM_PLUGIN
            "$<TARGET_FILE_DIR:VoxStudioTimelineQtTests>/platforms/${QT_WINDOWS_PLUGIN_NAME}"
        )
        add_custom_command(
            TARGET VoxStudioTimelineQtTests
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:VoxStudioTimelineQtTests>/platforms"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:Qt6::QWindowsIntegrationPlugin>"
                    "${VOXSTUDIO_TIMELINE_QT_TEST_PLATFORM_PLUGIN}"
            COMMENT "Deploying Qt Windows platform plugin for VoxStudioTimelineQtTests"
            VERBATIM
        )
    endif()
    add_test(NAME VoxStudioTimelineQtTests COMMAND VoxStudioTimelineQtTests)

    add_executable(
        VoxStudioLiveMicQtTests
        tests/unit/test_live_mic_panel.cpp
        src/ui/LiveAudioProcessor.cpp
        src/ui/LiveAudioProcessor.h
        src/ui/LiveMicPanel.cpp
        src/ui/LiveMicPanel.h
        src/ui/RvcModelManagerDialog.cpp
        src/ui/RvcModelManagerDialog.h
    )
    target_include_directories(VoxStudioLiveMicQtTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(
        VoxStudioLiveMicQtTests
        PRIVATE Qt6::Concurrent
                Qt6::Core
                Qt6::Test
                Qt6::Widgets
                VoxStudioCore
    )
    voxstudio_apply_project_warnings(VoxStudioLiveMicQtTests)
    if(WIN32)
        set(VOXSTUDIO_LIVE_MIC_QT_TEST_PLATFORM_PLUGIN
            "$<TARGET_FILE_DIR:VoxStudioLiveMicQtTests>/platforms/${QT_WINDOWS_PLUGIN_NAME}"
        )
        add_custom_command(
            TARGET VoxStudioLiveMicQtTests
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:VoxStudioLiveMicQtTests>/platforms"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:Qt6::QWindowsIntegrationPlugin>"
                    "${VOXSTUDIO_LIVE_MIC_QT_TEST_PLATFORM_PLUGIN}"
            COMMENT "Deploying Qt Windows platform plugin for VoxStudioLiveMicQtTests"
            VERBATIM
        )
    endif()
    add_test(NAME VoxStudioLiveMicQtTests COMMAND VoxStudioLiveMicQtTests)

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME VoxStudioRvcToolTests
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_rvc_onnx_tools.py"
        )
    else()
        message(WARNING "Python3 interpreter not found; RVC ONNX tool tests are disabled.")
    endif()

    add_test(NAME VoxStudioRvcNativeSmokeHelp COMMAND VoxStudioRvcNativeSmoke --help)
endif()

qt_finalize_executable(VoxStudio)
