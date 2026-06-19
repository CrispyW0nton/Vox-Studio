# Vox Studio

Native Windows AI voice acting studio for indie game developers.

Vox Studio is a C++20/Qt desktop app for producing game dialogue: project
management, script import, voice library tools, ElevenLabs integration, take
management, dialogue sequencing, and experimental local RVC voice-conversion
plumbing.

> Current status: early test build. The app can be built and launched locally,
> and the unit/Qt test suite is active. The full signed installer, production
> RVC sidecar payload, auto-updater, and release packaging are still future
> work.

## What It Does

- Stores game voice projects locally with SQLite-backed `.vox` project data.
- Saves ElevenLabs API keys with Windows DPAPI, not plaintext config files.
- Imports scripts from plain text, CSV, Fountain, Ren'Py, and Yarn-style JSON.
- Manages voices, character assignments, takes, and dialogue timelines.
- Provides TTS, STS, live microphone, and local RVC UI paths.
- Includes native ONNX RVC plumbing for future in-process inference validation.

## Requirements

- Windows 10/11 x64.
- Visual Studio 2022 with the `Desktop development with C++` workload.
- CMake 3.27 or newer.
- vcpkg, available through `VCPKG_ROOT`.
- Git.

The project uses vcpkg manifest mode. The CI workflow pins vcpkg to baseline
`e6ed7c5be05eaedc2d06ee3e51bd059409429c5e`.

## Quick Start

Clone the repo and set up vcpkg:

```powershell
git clone https://github.com/CrispyW0nton/Vox-Studio.git
cd Vox-Studio

$env:VCPKG_ROOT = 'C:\vcpkg'
if (!(Test-Path "$env:VCPKG_ROOT\.git")) {
  git clone https://github.com/microsoft/vcpkg.git $env:VCPKG_ROOT
}
& "$env:VCPKG_ROOT\bootstrap-vcpkg.bat"
```

Configure, build, and test Release:

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release --output-on-failure
```

Run the app:

```powershell
& 'C:\_b\voxstudio\r\Release\VoxStudio.exe'
```

Debug builds use the `windows-msvc-debug` preset and output to
`C:\_b\voxstudio\d\Debug\`.

## Local Runtime Data

Vox Studio stores user/runtime data under:

```text
%LOCALAPPDATA%\VoxStudio\
```

Important subfolders:

- `logs\` - app logs.
- `secrets\` - DPAPI-protected ElevenLabs API key material.
- `rvc_models\` - user-imported RVC `.pth` and `.index` files.
- `rvc_sidecar\` - installed local RVC sidecar payload.
- `onnxruntime\` - optional `onnxruntime.dll` runtime location.
- `rvc_onnx_models\` - native ONNX RVC model bundles.

Do not commit API keys, voice models, ONNX models, signing keys, or generated
project data. `.gitignore` is set up to keep those out of source control.

## ElevenLabs Setup

Open **Settings**, enter your ElevenLabs API key, click **Save**, then use
**Test Connection**. The key is stored with Windows DPAPI in the current user's
profile.

## RVC Status

There are two local RVC modes in the UI:

- **Sidecar** starts a local process on `http://127.0.0.1:18888` and talks to
  `/health` and `/convert_chunk`.
- **Native ONNX** loads `onnxruntime.dll` dynamically and expects model bundles
  under `%LOCALAPPDATA%\VoxStudio\rvc_onnx_models\<model_id>\`.

The repository intentionally does not include the large W-Okada/VCClient runtime
payload, ONNX Runtime DLLs, `.pth` model weights, `.index` files, or `.onnx`
graphs. Installer packaging will own those runtime artifacts.

For app-pipeline testing, the repo includes a small compatibility sidecar source
at `tools/rvc_compat_sidecar/`. It implements the Vox Studio sidecar HTTP shape
and passes audio through, but it does not perform real voice conversion.

More detail is in [docs/RVC_NATIVE.md](docs/RVC_NATIVE.md).

## Project Layout

```text
cmake/                 CMake helpers for Qt, warnings, and packaging
docs/                  development plan, architecture notes, RVC docs, ADRs
installer/             Inno Setup snippets and future installer work
resources/             icons, models, QML, translations placeholders
src/app/               app entry point and logging
src/audio/             miniaudio, files, preview, capture, buffers, resampling
src/core/              domain models and rendering/take logic
src/db/                SQLite repositories and migrations
src/io/scripts/        script importers
src/net/elevenlabs/    ElevenLabs REST/TTS/STS/voices clients
src/platform/win/      Windows app paths and single-instance guard
src/rvc/               sidecar, model registry, native ONNX RVC plumbing
src/secrets/           DPAPI secret storage
src/ui/                Qt widgets and dialogs
tests/                 Catch2, Qt, fixture, and tool tests
third_party/           vendored headers/metadata only, not heavyweight payloads
tools/                 smoke tools, RVC utilities, compatibility sidecar source
```

## Useful Commands

Run all Release tests:

```powershell
ctest --preset windows-msvc-release --output-on-failure
```

Run the native RVC smoke tool help:

```powershell
& 'C:\_b\voxstudio\r\Release\VoxStudioRvcNativeSmoke.exe' --help
```

Inspect a set of ONNX model contracts:

```powershell
python tools\scripts\inspect_onnx_contract.py `
  --generator C:\models\hero\generator.onnx `
  --hubert C:\models\hero\hubert.onnx `
  --f0 C:\models\hero\rmvpe.onnx `
  --out C:\models\hero\graph_contract.json
```

## Development Notes

- Keep audio callbacks lock-free and avoid UI-thread blocking.
- Prefer focused tests around repositories, importers, network parsing, RVC
  contracts, and Qt panels.
- Keep LGPL dependencies dynamically linked.
- Large runtime payloads belong in installer/runtime delivery, not git.
- See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the pass-based
  roadmap.

## License

Project license is not finalized yet. Third-party notices live with their
respective vendored metadata where present, and release packaging must include
the required notices for dynamically linked LGPL components.
