# Vox Studio

Native Windows AI voice acting studio for indie game developers.

Vox Studio is a C++20/Qt desktop app for producing game dialogue: project
management, script import, voice library tools, ElevenLabs integration,
VoxCPM2 character performance rendering, take management, dialogue sequencing,
Performance Mirror live voice conversion, and imported RVC voice conversion.

> Current status: early test build. The app can be built and launched locally,
> the unit/Qt test suite is active, Performance Mirror prefers locally trained
> character RVC checkpoints with Seed-VC as a reference fallback, and HQ
> phrase/monologue rendering runs locally through VoxCPM2.
> A self-contained signed installer, auto-updater, and release packaging are
> still future work.

## What It Does

- Stores game voice projects locally with SQLite-backed `.vox` project data.
- Saves ElevenLabs API keys with Windows DPAPI, not plaintext config files.
- Imports scripts from plain text, CSV, Fountain, Ren'Py, and Yarn-style JSON.
- Manages voices, character assignments, takes, and dialogue timelines.
- Uses local VoxCPM2 character adapters to keep a selected voice stable while
  applying mic-derived timing, emphasis, and emotional controls.
- Provides waveform-level Performance Mirror monitoring that follows the
  performer's live timing, pauses, energy, and pitch movement while using a
  character-trained local identity model when one is installed.
- Provides local emotional TTS, HQ phrase and monologue VoxCPM2 capture,
  live microphone, and imported RVC UI paths.
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
- `engines\seed-vc\` - isolated official Seed-VC runtime for Performance Mirror.
- `performance_mirror_sidecar\` - lightweight local audio bridge and setup tool.
- `voxcpm_profiles\` - local character identity references and profile metadata.
- `voxcpm_training\` - prepared 16 kHz manifests and local LoRA checkpoints.
- `voxcpm_sidecar\` - phrase-live transcription and rendering service.
- `generated_vocal_actions\` - cached character laughs, sighs, and other
  nonverbal performances generated locally for profile use.

Do not commit API keys, voice models, ONNX models, signing keys, or generated
project data. `.gitignore` is set up to keep those out of source control.

## ElevenLabs Setup

The first launch requires each user to enter their own ElevenLabs API key.
Vox Studio does not ship with a shared or developer key. The supplied key is
verified with ElevenLabs before it is saved. It is then stored only on that PC
with Windows DPAPI, encrypted for the current Windows account, and is never
copied into the repository or app bundle.

To replace a saved key later, open **Settings**, enter the new key, click
**Save**, then use **Test Connection**.

Use **Voices** to sync or clone a voice, then select **Assign to Character** to
attach it to a character imported from a script. Vox Studio stores the voice
identity and project assignment locally; ElevenLabs does not make its trained
voice model weights exportable.

In **Live Mic**, use **Mic Check** to verify the selected input and headphone
output with your unchanged microphone. **Performance Mirror** is the default
live character mode: it converts the microphone waveform continuously, so the
selected character follows the performer's timing, pauses, emphasis, energy,
and pitch movement without transcription or phrase regeneration. Character
slots switch the active local model directly. A character-trained RVC checkpoint
is preferred because it holds identity substantially better than a short
zero-shot reference. Characters without a trained checkpoint use Reference
Mirror as a fallback, and the active engine is named beside the selected voice.
**Hear Result** monitors only the converted character; **Live Input**
independently adds the unchanged mic when explicitly enabled.

**HQ Phrase** captures one phrase at a time and plays it after a natural pause.
Local Whisper transcription supplies the words. A smoothed delivery
detector classifies each phrase as calm, measured, neutral, emphatic, urgent,
questioning, reflective, or sarcastic from its language, pace, dynamics,
pauses, and pitch contour. For a trained local character, the selected LoRA
adapter supplies a stable identity while those measurements produce detailed
delivery controls.
Legacy profiles fall back to matching a compatible in-character reference.
Live Mic shows the detected delivery after each HQ phrase. **Hear Result** is
armed automatically and plays the character phrase through the selected
headphones.

Carth adds a character-specific palette derived from his game dialogue: dry
wry skepticism, guarded distrust, contained vulnerability, understated warmth,
moral disapproval, and protective resolve. These refine his manner without
replacing phrase-level pace, pitch, pauses, or intensity.

Live transcription uses Whisper Turbo on supported GPUs and supplies a local
KOTOR vocabulary for character, place, and ship names. If Turbo cannot load,
Vox Studio falls back to the smaller cached English recognizer. Entering the
script line remains the most accurate option because it bypasses transcription.

Typing the line is optional. When present, it bypasses transcription for exact
script wording. Saved results appear immediately under **Recent Takes**, where
they can be played, starred, revealed in Explorer, or deleted. **Broadcast**
sends the same character result to a selected virtual audio line for OBS, games,
or chat software.

Use **Monologue** for an uninterrupted long-form performance. Vox Studio keeps
recording while you speak, divides the performance only at natural pauses,
renders every section after you stop, and then plays the complete character
result. An optional exact script is allocated across those sections at sentence
boundaries. Choose a capture name and folder before recording; the finished
changed-voice performance is saved there as MP3 and can be opened directly with
**Open in Explorer**. When take saving is enabled, it is also stored in the
project.

The **Generated Takes** list supports multi-selection. Click individual takes or
use **Select All**, then choose **Export as .mp3** and select a folder.

The **Text to Speech** workspace generates typed dialogue with the same local
VoxCPM2 character profiles. Long scripts are split and stitched at sentence
boundaries. Select one delivery tag from Natural, Calm, Measured, Reflective,
Warm, Wry, Guarded, Wounded, Resolute, Urgent, Questioning, or Sarcastic.
Generated speech plays through the selected output and is saved under
**Generated Takes** for replay, starring, Explorer access, or deletion.
Vocal stage directions wrapped in asterisks are rendered as in-character
nonverbal beats instead of spoken labels. Examples include `*sighs*`,
`*laughs*`, `*gasps*`, `*coughs*`, `*groans*`, `*sobs*`, and `*dies*`.
Normal emphasis such as `*very*` remains spoken dialogue.

Vocal reactions use a separate performance bank rather than asking the speech
model to pronounce an action. Clean licensed reaction clips are used directly
for effort, pain, gasps, screams, and death sounds. Missing reactions can be
generated offline with Dia from a character profile, reviewed once, and cached;
they are never generated in the middle of playback. Direct cached audio is
preferred over prompt-based fallbacks so a weak synthetic sigh cannot replace a
verified performance.

Choose **Storytelling** for monologues and narrative scenes. Vox Studio analyzes
the pasted text as changing thought beats, preserves the exact wording, and
directs the hook, setup, escalation, turns, climax, and ending independently.
**Preview Flow** shows the planned intent, emotional color, and operative words
before synthesis. The renderer keeps one stable character identity across the
story while varying pacing, emphasis, energy, and pauses only when the thought
or dramatic situation changes.

VoxCPM2 HQ Phrase mode is phrase-live rather than waveform conversion: the
result begins after a natural pause and local inference delay. The model remains
loaded between phrases. Use **Performance Mirror** for live acting and
**Imported RVC** for existing `.pth`/`.index` models.

## Performance Mirror

Performance Mirror is the user-facing live mode. For a character with an
installed trained checkpoint, Vox Studio routes 240 ms microphone blocks through
the official RVC real-time pipeline. This changes speaker identity while
retaining the performed timing and pitch contour. Shared speech and pitch
components stay loaded when character slots change. Low-level room noise is
gated before playback, and converted audio can be sent simultaneously to
headphones, a selected virtual microphone, and a saved MP3 project take.

When no trained character model is installed, Reference Mirror listens on
`http://127.0.0.1:18910` and runs the official
[Seed-VC](https://github.com/Plachtaa/seed-vc) real-time tiny model in a separate
local process. It uses 360 ms blocks and a profile reference. Use **Install
Mirror Engine** once on a new PC to enable that fallback. The setup creates an
isolated CUDA environment under `%LOCALAPPDATA%\VoxStudio\engines\seed-vc\`.
Vox Studio does not redistribute upstream source, model weights, or character
recordings.

For a command-line setup:

```powershell
powershell -ExecutionPolicy Bypass -File `
  third_party\performance_mirror_sidecar\setup_performance_mirror.ps1
```

Run a file through the same streaming endpoint:

```powershell
& "$env:LOCALAPPDATA\VoxStudio\engines\seed-vc\.venv\Scripts\python.exe" `
  tools\smoke_performance_mirror.py --source input.wav `
  --voice-id YOUR_VOICE_ID --output mirror-test.wav
```

## VoxCPM2 Profiles

The local service listens on `http://127.0.0.1:18990`. Each profile lives under:

```text
%LOCALAPPDATA%\VoxStudio\voxcpm_profiles\<voice_id>\
```

`reference.wav` contains one clean identity reference for trained characters.
The optional `lora\` folder contains the character adapter, `styles\` contains
fallback delivery anchors, and `profile.json` records source coverage and
inference settings.

Prepare exact Carth or Bao-Dur dialogue/audio pairs from an installed game:

```powershell
& "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2\.venv\Scripts\python.exe" `
  tools\prepare_kotor_character_corpus.py `
  --game-root "C:\Program Files (x86)\Steam\steamapps\common\swkotor" `
  --character Carth --voiceover-token cart `
  --output "$HOME\Documents\KotorMods\Voices\RvcDatasets\CarthExact"

& "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2\.venv\Scripts\python.exe" `
  tools\prepare_voxcpm_finetune.py --voice carth
```

The generated manifest and training config stay under
`%LOCALAPPDATA%\VoxStudio\voxcpm_training\`. Run the official VoxCPM2 trainer
from a local VoxCPM 2.0.3 source checkout:

```powershell
$voxCpmSource = "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2\source-2.0.3"
& "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2\.venv\Scripts\python.exe" `
  "$voxCpmSource\scripts\train_voxcpm_finetune.py" `
  --config_path "$env:LOCALAPPDATA\VoxStudio\voxcpm_training\carth\train_lora.yaml"
```

Then rebuild the profiles:

```powershell
& "$env:LOCALAPPDATA\VoxStudio\engines\voxcpm2\.venv\Scripts\python.exe" `
  tools\prepare_voxcpm_profiles.py
```

To prepare local character-specific laughs, sighs, and chuckles, install the
isolated Dia action runtime and generate a reviewable cache before rebuilding:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_dia_actions.ps1

& "$env:LOCALAPPDATA\VoxStudio\engines\dia-actions\.venv\Scripts\python.exe" `
  tools\prepare_paralinguistic_actions.py `
  --voice-id YOUR_VOICE_ID `
  --action sigh --action laugh --action chuckle --variants 2
```

Dia and VoxCPM2 are used from their upstream projects and downloaded into local
runtime folders. Generated reaction WAVs, licensed source recordings, manifests,
and character adapters remain local and are not distributed by Vox Studio.

Pass `--voice "Carth"` or `--voice "Bao-Dur"` to rebuild one profile. The
builder installs a completed local adapter only when its training summary
identifies the profile's current corpus. This prevents an older mixed or
superseded dataset from silently changing the character voice.
Licensed game audio, prepared manifests, and adapter weights remain local and
are never included in the repository.

`third_party\voxcpm_sidecar\pronunciations.json` contains local synthesis-only
respellings for character and place names such as Carth, Atton, Bao-Dur, Kreia,
Telos, Rodian, Nar Shaddaa, Pazaak, Sabacc, and Gamorreans. The saved script and
take transcript keep their original spelling.

The source tree contains the service and profile tooling, not model weights or
licensed voice audio. Those remain local user data.

On a new Windows machine, install the isolated GPU runtime and model cache once:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_voxcpm_runtime.ps1
```

## RVC Status

There are two ways RVC is selected in the UI:

- **Performance Mirror** automatically selects the trained model assigned to
  the current character.
- **Imported RVC** lets the performer select an installed model directly.

Both use the **Sidecar** runtime by default. It starts the Vox Studio real-time
bridge on `http://127.0.0.1:18888`, talks to `/health` and `/convert_chunk`, and
runs imported `.pth` plus `.index` models through the official RVC real-time
pipeline. Audio is processed in stateful 240 ms blocks with RMVPE pitch
extraction and SOLA crossfading.

The optional **Native ONNX** runtime loads `onnxruntime.dll` dynamically and
expects model bundles under
`%LOCALAPPDATA%\VoxStudio\rvc_onnx_models\<model_id>\`.

The sidecar expects the official RVC runtime under
`%LOCALAPPDATA%\VoxStudio\training\RVC-WebUI\`. Model weights and indexes remain
user-owned runtime artifacts under `%LOCALAPPDATA%\VoxStudio\rvc_models\`; they
are not committed to this repository. ONNX Runtime DLLs and `.onnx` graphs are
also external runtime artifacts.

To train and install a character checkpoint from the locally selected,
licensed dialogue recorded in an existing VoxCPM profile:

```powershell
& "$env:LOCALAPPDATA\VoxStudio\training\RVC-WebUI\.venv\Scripts\python.exe" `
  tools\train_character_rvc.py `
  --voice-id <voice-id> `
  --experiment <training-name> `
  --model-id <model-id> `
  --display-name <display-name>
```

The tool runs preprocessing, RMVPE and HuBERT extraction, RVC v2 training, and
index generation directly. It then writes a model manifest with the matching
character voice ID, allowing Performance Mirror to discover it automatically.

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
src/performance_mirror/ local Seed-VC sidecar process management
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
