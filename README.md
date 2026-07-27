# Vox Studio

Native Windows AI voice acting studio for indie game developers.

Vox Studio is a C++20/Qt desktop app for producing game dialogue: project
management, script import, voice library tools, ElevenLabs integration,
VoxCPM2 character performance rendering, take management, dialogue sequencing,
and local RVC voice conversion.

> Current status: early test build. The app can be built and launched locally,
> the unit/Qt test suite is active, and Performance mode runs locally through
> VoxCPM2. A self-contained signed installer, auto-updater, and release
> packaging are still future work.

## What It Does

- Stores game voice projects locally with SQLite-backed `.vox` project data.
- Saves ElevenLabs API keys with Windows DPAPI, not plaintext config files.
- Imports scripts from plain text, CSV, Fountain, Ren'Py, and Yarn-style JSON.
- Manages voices, character assignments, takes, and dialogue timelines.
- Uses microphone wording and delivery cues to choose a clean, transcribed
  VoxCPM2 style reference that remains in the selected character's voice.
- Provides TTS, phrase-live VoxCPM2, live microphone, and local RVC UI paths.
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
- `engines\voxcpm2\` - isolated VoxCPM2 Python/GPU runtime and model cache.
- `voxcpm_profiles\` - local character identity references and profile metadata.
- `voxcpm_sidecar\` - phrase-live transcription and rendering service.

Do not commit API keys, voice models, ONNX models, signing keys, or generated
project data. `.gitignore` is set up to keep those out of source control.

## ElevenLabs Setup

Open **Settings**, enter your ElevenLabs API key, click **Save**, then use
**Test Connection**. The key is stored with Windows DPAPI in the current user's
profile.

Use **Voices** to sync or clone a voice, then select **Assign to Character** to
attach it to a character imported from a script. Vox Studio stores the voice
identity and project assignment locally; ElevenLabs does not make its trained
voice model weights exportable.

In **Live Mic**, use **Mic Check** to verify the selected input and headphone
output with your unchanged microphone. **Performance** captures one phrase at a
time. Local Whisper transcription supplies the words. A smoothed delivery
detector classifies each phrase as calm, measured, neutral, emphatic, urgent,
questioning, or sarcastic from its language, pace, dynamics, pauses, and pitch
contour. It then selects a compatible in-character reference and explicitly
prevents the character profile from adding emotion that was not present in the
performance. Live Mic shows the detected delivery after each phrase. This is
delivery-style matching, not sample-exact prosody transfer. **Hear Result** is
armed automatically and plays the character phrase through the selected
headphones. **Live Input** is independent: turn it off to hear only the
character result, or on to hear the unchanged mic while performing.

Typing the line is optional. When present, it bypasses transcription for exact
script wording. Saved results appear immediately under **Recent Takes**, where
they can be played, starred, revealed in Explorer, or deleted. **Broadcast**
sends the same character result to a selected virtual audio line for OBS, games,
or chat software.

VoxCPM2 Performance mode is phrase-live rather than zero-latency waveform
conversion: the result begins after a natural pause and local inference delay.
The model remains loaded between phrases. Use **Local** RVC when immediate
low-latency feedback matters more than character fidelity.

## VoxCPM2 Profiles

The local service listens on `http://127.0.0.1:18990`. Each profile lives under:

```text
%LOCALAPPDATA%\VoxStudio\voxcpm_profiles\<voice_id>\
```

`reference.wav` contains a curated identity reference. The `styles\` folder
contains clean transcribed delivery anchors, and `profile.json` records their
source coverage, acoustic features, and inference settings. Rebuild the local
character profiles from available source libraries with:

```powershell
& "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2\.venv\Scripts\python.exe" `
  tools\prepare_voxcpm_profiles.py
```

Pass `--voice "Bao-Dur"` (or another character name) to rebuild one profile.
Bao-Dur uses the decoded `GBL\BAODUR` conversation library so the live matcher
can choose among his quiet, reflective, urgent, and technical delivery styles
instead of relying on a single stitched reference recording.

`third_party\voxcpm_sidecar\pronunciations.json` contains local synthesis-only
respellings for character and place names such as Carth, Atton, Bao-Dur, Kreia,
Telos, and Nar Shaddaa. The saved script and take transcript keep their original
spelling.

The source tree contains the service and profile tooling, not model weights or
licensed voice audio. Those remain local user data.

On a new Windows machine, install the isolated GPU runtime and model cache once:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_voxcpm_runtime.ps1
```

## RVC Status

There are two local RVC modes in the UI:

- **Sidecar** starts the Vox Studio real-time bridge on
  `http://127.0.0.1:18888`, talks to `/health` and `/convert_chunk`, and runs
  imported `.pth` plus `.index` models through the official RVC real-time
  pipeline. Audio is processed in stateful 250 ms blocks with RMVPE pitch
  extraction and SOLA crossfading.
- **Native ONNX** loads `onnxruntime.dll` dynamically and expects model bundles
  under `%LOCALAPPDATA%\VoxStudio\rvc_onnx_models\<model_id>\`.

The sidecar expects the official RVC runtime under
`%LOCALAPPDATA%\VoxStudio\training\RVC-WebUI\`. Model weights and indexes remain
user-owned runtime artifacts under `%LOCALAPPDATA%\VoxStudio\rvc_models\`; they
are not committed to this repository. ONNX Runtime DLLs and `.onnx` graphs are
also external runtime artifacts.

The older compatibility sidecar source remains under
`tools/rvc_compat_sidecar/` for protocol-only tests. It passes audio through and
must not be used for character conversion.

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
src/voxcpm/            VoxCPM2 service process and HTTP client
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

Build a clean RVC dataset from KOTOR dialogue:

```powershell
python tools\scripts\build_kotor_rvc_dataset.py `
  --source-root 'C:\Games\swkotor\streamwaves' `
  --selection-dir 'C:\VoiceSelections\Carth' `
  --output-dir 'C:\RvcDatasets\CarthConversational' `
  --exclude-name-glob 'nm01aacart*.wav' `
  --exclude-name-glob 'n_m1bncart*.wav'
```

KOTOR can store MP3 dialogue behind a misleading WAV header. The dataset tool
detects and removes the game wrapper before decoding, writes mono PCM training
audio, rejects noise-like output, and records a manifest beside the dataset.
The exclusion globs above omit Carth's Endar Spire communicator lines so the
model learns his conversational recording rather than the in-game radio effect.

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
