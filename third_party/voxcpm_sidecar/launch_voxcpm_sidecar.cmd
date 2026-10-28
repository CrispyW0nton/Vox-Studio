@echo off
setlocal

set "VOX_VOXCPM_ROOT=%~dp0"
set "VOX_VOXCPM_HOST=127.0.0.1"
set "VOX_VOXCPM_PORT=18990"

:parse
if "%~1"=="" goto run
if "%~1"=="--host" (
  set "VOX_VOXCPM_HOST=%~2"
  shift
  shift
  goto parse
)
if "%~1"=="--port" (
  set "VOX_VOXCPM_PORT=%~2"
  shift
  shift
  goto parse
)
shift
goto parse

:run
set "VOX_VOXCPM_ENGINE_ROOT=%LOCALAPPDATA%\VoxStudio\engines\voxcpm2"
set "VOX_VOXCPM_PYTHON=%VOX_VOXCPM_ENGINE_ROOT%\.venv\Scripts\python.exe"

if exist "%VOX_VOXCPM_PYTHON%" if exist "%VOX_VOXCPM_ROOT%vox_voxcpm_server.py" (
  "%VOX_VOXCPM_PYTHON%" "%VOX_VOXCPM_ROOT%vox_voxcpm_server.py" ^
    --host "%VOX_VOXCPM_HOST%" --port "%VOX_VOXCPM_PORT%"
  exit /b %ERRORLEVEL%
)

echo Vox Studio VoxCPM2 runtime is missing.
echo Expected Python at %VOX_VOXCPM_PYTHON%.
exit /b 2
