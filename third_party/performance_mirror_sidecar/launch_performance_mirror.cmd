@echo off
setlocal

set "VOX_MIRROR_ROOT=%~dp0"
set "VOX_MIRROR_HOST=127.0.0.1"
set "VOX_MIRROR_PORT=18910"

:parse
if "%~1"=="" goto run
if "%~1"=="--host" (
  set "VOX_MIRROR_HOST=%~2"
  shift
  shift
  goto parse
)
if "%~1"=="--port" (
  set "VOX_MIRROR_PORT=%~2"
  shift
  shift
  goto parse
)
shift
goto parse

:run
set "VOX_SEED_VC_ROOT=%LOCALAPPDATA%\VoxStudio\engines\seed-vc"
set "VOX_MIRROR_PYTHON=%VOX_SEED_VC_ROOT%\.venv\Scripts\python.exe"

if exist "%VOX_MIRROR_PYTHON%" if exist "%VOX_SEED_VC_ROOT%\real-time-gui.py" (
  "%VOX_MIRROR_PYTHON%" "%VOX_MIRROR_ROOT%vox_performance_mirror_server.py" ^
    --host "%VOX_MIRROR_HOST%" --port "%VOX_MIRROR_PORT%" ^
    --seed-vc-root "%VOX_SEED_VC_ROOT%"
  exit /b %ERRORLEVEL%
)

echo Vox Studio Performance Mirror runtime is missing.
echo Run setup_performance_mirror.ps1 once before using Performance Mirror.
exit /b 2
