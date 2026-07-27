@echo off
setlocal

set "VOX_RVC_ROOT=%~dp0"
set "VOX_RVC_HOST=127.0.0.1"
set "VOX_RVC_PORT=18888"

:parse
if "%~1"=="" goto run
if "%~1"=="--host" (
  set "VOX_RVC_HOST=%~2"
  shift
  shift
  goto parse
)
if "%~1"=="--port" (
  set "VOX_RVC_PORT=%~2"
  shift
  shift
  goto parse
)
shift
goto parse

:run
set "VOX_RVC_RUNTIME=%LOCALAPPDATA%\VoxStudio\training\RVC-WebUI"
set "VOX_RVC_PYTHON=%VOX_RVC_RUNTIME%\.venv\Scripts\python.exe"

if exist "%VOX_RVC_PYTHON%" if exist "%VOX_RVC_ROOT%vox_rvc_server.py" (
  "%VOX_RVC_PYTHON%" "%VOX_RVC_ROOT%vox_rvc_server.py" ^
    --host "%VOX_RVC_HOST%" --port "%VOX_RVC_PORT%" ^
    --rvc-root "%VOX_RVC_RUNTIME%"
  exit /b %ERRORLEVEL%
)

if exist "%VOX_RVC_ROOT%VCClient.exe" (
  "%VOX_RVC_ROOT%VCClient.exe" --host "%VOX_RVC_HOST%" --port "%VOX_RVC_PORT%"
  exit /b %ERRORLEVEL%
)

echo Vox Studio RVC sidecar runtime payload is missing.
echo Expected the Vox Studio RVC training runtime at %VOX_RVC_RUNTIME%.
exit /b 2
