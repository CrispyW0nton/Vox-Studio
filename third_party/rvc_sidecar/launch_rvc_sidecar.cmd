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
if exist "%VOX_RVC_ROOT%VCClient.exe" (
  "%VOX_RVC_ROOT%VCClient.exe" --host "%VOX_RVC_HOST%" --port "%VOX_RVC_PORT%"
  exit /b %ERRORLEVEL%
)

if exist "%VOX_RVC_ROOT%python\python.exe" if exist "%VOX_RVC_ROOT%server\MMVCServerSIO.py" (
  "%VOX_RVC_ROOT%python\python.exe" "%VOX_RVC_ROOT%server\MMVCServerSIO.py" ^
    --host "%VOX_RVC_HOST%" --port "%VOX_RVC_PORT%"
  exit /b %ERRORLEVEL%
)

echo Vox Studio RVC sidecar runtime payload is missing.
echo Expected VCClient.exe or python\python.exe plus server\MMVCServerSIO.py in %VOX_RVC_ROOT%.
exit /b 2
