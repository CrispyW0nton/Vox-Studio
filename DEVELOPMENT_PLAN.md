# Vox Studio — Development Plan

> **An AI voice acting studio for indie game developers.**
> Native Windows C++ desktop application. Clones character voices via ElevenLabs, displays scripts during recording, hot-switches voices to build full dialogue sequences, and converts your microphone performance into a cloned character's voice in real time using a local RVC engine.

---

## Table of Contents

1. [Vision & Target Users](#1-vision--target-users)
2. [Tech Stack Summary](#2-tech-stack-summary)
3. [Architecture Overview](#3-architecture-overview)
4. [Threading Model & Latency Budget](#4-threading-model--latency-budget)
5. [Component Breakdown](#5-component-breakdown)
6. [Repository Layout](#6-repository-layout)
7. [Coding Standards](#7-coding-standards)
8. [Pass-Based Development Roadmap](#8-pass-based-development-roadmap)
9. [Dependency Table](#9-dependency-table)
10. [SQLite Project Schema](#10-sqlite-project-schema)
11. [Risk Register](#11-risk-register)
12. [Glossary](#12-glossary)
13. [Primary Source Index](#13-primary-source-index)

---

## 1. Vision & Target Users

**Vision.** Voicemod-grade real-time voice transformation crossed with a lightweight DAW, purpose-built for dialogue production. Indie studios load a script, clone the voice cast once, then perform every character themselves — either by typing lines (TTS) or speaking them (real-time voice conversion preserving the actor's prosody and emotion).

**Primary user.** Solo or small-team indie game developers (Unity, Unreal, Godot, Ren'Py) who need hundreds-to-thousands of dialogue lines but cannot afford a traditional voice acting cast.

**Secondary users.** Audio drama producers, animation studios, indie film prototypers, accessibility teams generating narrated content.

**Non-goals (v1).**
- Not a full DAW (no multi-track mixing, no MIDI, no VST hosting).
- Not a real-time game runtime audio engine (we render *assets*, not stream into engines).
- Not a translation/dubbing tool (ElevenLabs Dubbing API is out of scope for v1).

---

## 2. Tech Stack Summary

| Layer | Choice | Version | License | Rationale |
|---|---|---|---|---|
| Language | C++ | 20 | — | RAII, concepts, ranges, modern threading |
| UI | Qt | 6.7 LTS | LGPLv3 (dynamic) | Mature, theming, audio widgets, OBS-class reference |
| Build | CMake | ≥ 3.27 | BSD | Qt 6 first-class support |
| Deps | vcpkg | manifest mode | MIT | Reproducible Windows builds |
| Project DB | SQLite | 3.45+ | Public Domain | Embedded, zero-config |
| HTTP | libcurl + cpr | 8.x / 1.11 | MIT/MIT | ElevenLabs REST + multipart uploads |
| WebSocket | IXWebSocket | 11.x | BSD-3 | ElevenLabs TTS streaming |
| JSON | nlohmann/json | 3.11+ | MIT | API serialization |
| Audio I/O | miniaudio | 0.11+ | Public Domain / MIT-0 | Single-header WASAPI wrapper |
| DSP | dr_libs (dr_wav, dr_flac, dr_mp3) | latest | Public Domain | Decode/encode without FFmpeg dep |
| Resampler | soxr | 0.1.3 | LGPLv2.1 (dynamic) | High-quality SR conversion |
| Codec | libopus + libsndfile | 1.4 / 1.2 | BSD / LGPLv2.1 (dynamic) | Compact archival of takes |
| Noise suppression | RNNoise | latest | BSD | Pre-clean mic input before RVC |
| VAD | Silero VAD (ONNX) | v4 | MIT | Endpointing, line auto-advance |
| Local RVC engine | w-okada/voice-changer (bundled Python sidecar) | v.1.5.x | MIT | Battle-tested real-time RVC |
| ML runtime (Pass 9b) | ONNX Runtime | 1.18+ | MIT | Native C++ RVC inference path |
| Secrets | Win32 DPAPI | OS | — | API key at-rest encryption |
| Logging | spdlog | 1.13+ | MIT | Async, rotating sinks |
| Tests | Catch2 + Qt Test | v3 / Qt 6 | BSL-1.0 / LGPL | Unit + GUI tests |
| Installer | Inno Setup | 6.x | Modified BSD | De facto Windows indie installer |
| CI | GitHub Actions | — | — | windows-latest runners |
| Code signing | EV Code Signing Cert | — | — | SmartScreen reputation |

**License posture.** All LGPL components (Qt, soxr, libsndfile) are **dynamically linked**. No GPL components. No JUCE (GPL/commercial dual-license — incompatible with our model). VB-CABLE is **not bundled** — installed by the user on first run.

---

## 3. Architecture Overview

```
+---------------------------------------------------------------+
|                      Vox Studio (Qt 6 GUI)                    |
|                                                               |
|  +-----------+  +-----------+  +-----------+  +-----------+   |
|  | Project   |  |  Voice    |  |  Script   |  |  Session  |   |
|  |  HUD      |  | Library   |  |  Viewer   |  |  Timeline |   |
|  +-----+-----+  +-----+-----+  +-----+-----+  +-----+-----+   |
|        |              |              |              |         |
|        v              v              v              v         |
|  +---------------------------------------------------------+  |
|  |                  Core Domain Layer                      |  |
|  |   Project / Voice / Character / Line / Take models      |  |
|  |          SQLite (SQLiteCpp) project database            |  |
|  +-----+---------------+---------------+-------------------+  |
|        |               |               |                      |
|        v               v               v                      |
|  +-----------+  +------------+  +------------------+          |
|  | ElevenLabs|  | RealTime   |  |  AudioEngine     |          |
|  |  Client   |  | RVC Bridge |  |  (miniaudio)     |          |
|  | (libcurl+ |  | (HTTP/WS   |  |  WASAPI shared   |          |
|  |  IXWS)    |  | to sidecar)|  |  + ring buffers  |          |
|  +-----+-----+  +-----+------+  +---------+--------+          |
+--------|--------------|-------------------|-------------------+
         |              |                   |
         v              v                   v
  +-----------+   +-----------+      +--------------+
  | ElevenLabs|   |  w-okada  |      |  WASAPI HAL  |
  |   Cloud   |   |  Python   |      |   (Windows)  |
  |           |   |  sidecar  |      +--------------+
  | TTS / IVC |   |  (PyTorch |
  | PVC / STS |   |   /ONNX)  |
  +-----------+   +-----------+
```

**Two paths to a cloned voice in the user's ear:**

1. **TTS path (typed lines, perfect prosody control via ElevenLabs).**
   `Script line → ElevenLabs TTS Streaming → AudioEngine playback → render to take WAV`

2. **Live performance path (microphone, real-time, low latency).**
   `Mic → WASAPI capture → RNNoise → Silero VAD → ring buffer → RVC sidecar (HTTP /convert_chunk) → soxr resample → AudioEngine playback → optional take capture`

The user can A/B between cloud Speech-to-Speech (higher quality, ~600-1500ms latency, costs credits) and local RVC (lower quality, ~150-300ms latency, free) for the same performance.

---

## 4. Threading Model & Latency Budget

| Thread | Purpose | Priority |
|---|---|---|
| UI (main) | Qt event loop only. No blocking work. | Normal |
| Audio capture | WASAPI input callback, fills lock-free ring buffer | TIME_CRITICAL |
| Audio playback | WASAPI output callback, drains lock-free ring buffer | TIME_CRITICAL |
| DSP worker | RNNoise + VAD + resampling between rings | Above normal |
| RVC bridge | HTTP/WS to local sidecar, blocking allowed | Normal |
| Network | ElevenLabs REST/WS via Qt's threadpool | Normal |
| DB | SQLite serialized access on a dedicated worker | Normal |

**Lock-free queues:** `moodycamel::ReaderWriterQueue` between audio threads and DSP worker.

**Live-mic latency budget (target ≤ 300ms end-to-end):**

| Stage | Budget | Notes |
|---|---|---|
| WASAPI capture frame | 10ms | Shared-mode period |
| Ring → DSP hop | 5ms | |
| RNNoise + VAD | 15ms | Frame-based |
| HTTP chunk POST to sidecar | 20ms | localhost |
| RVC inference (RTX 3060+) | 120ms | RMVPE f0 |
| Resample + crossfade | 10ms | soxr VHQ |
| Playback ring → speaker | 20ms | |
| **Total** | **~200ms** | Headroom for jitter |

CPU fallback (no CUDA) target: ≤ 500ms with a smaller model.

---

## 5. Component Breakdown

- **`core/`** — Pure C++ domain models, no Qt deps. `Project`, `Voice`, `Character`, `Script`, `Line`, `Take`, `Session`.
- **`db/`** — SQLiteCpp wrapper, migrations, project file (`.vox` is a folder containing `project.db` + media).
- **`net/elevenlabs/`** — REST client, TTS streaming WS client, STS client, voice CRUD, voice cloning multipart.
- **`audio/`** — `AudioEngine` (miniaudio), `RingBuffer`, `AudioFile` (dr_libs + libsndfile + opus), `Resampler` (soxr).
- **`dsp/`** — `NoiseSuppressor` (RNNoise), `Vad` (Silero ONNX), `LevelMeter`, `CrossfadeRouter`.
- **`rvc/`** — `RvcSidecar` (process supervisor for bundled Python), `RvcClient` (HTTP/WS), `RvcModelRegistry`. Pass 9b adds `OnnxRvcEngine` for the native path.
- **`secrets/`** — `DpapiVault` for API keys + (optional) voice sample encryption at rest.
- **`ui/`** — Qt widgets/QML. `MainWindow`, `ProjectHud`, `VoiceLibraryPanel`, `ScriptViewerPanel`, `TimelinePanel`, `LiveMicPanel`, `SettingsDialog`.
- **`io/scripts/`** — Importers for `.txt`, Fountain, `.rpy` (Ren'Py), Yarn Spinner JSON, generic CSV.
- **`export/`** — Render queue, batch export with engine-friendly naming (Unity AudioClip, Unreal SoundWave, Ren'Py voice file conventions).
- **`platform/win/`** — DPAPI, Credential Manager, registry helpers, single-instance mutex.

---

## 6. Repository Layout

```
voxstudio/
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── vcpkg-configuration.json
├── .clang-format
├── .clang-tidy
├── .editorconfig
├── .github/
│   └── workflows/
│       ├── ci-windows.yml
│       └── release.yml
├── cmake/
│   ├── QtSetup.cmake
│   ├── Warnings.cmake
│   └── Packaging.cmake
├── docs/
│   ├── DEVELOPMENT_PLAN.md   <-- this file
│   ├── ARCHITECTURE.md
│   ├── CODING_STANDARDS.md
│   └── adr/                   <-- Architecture Decision Records
├── src/
│   ├── app/                   <-- main(), bootstrap
│   ├── core/
│   ├── db/
│   ├── net/elevenlabs/
│   ├── audio/
│   ├── dsp/
│   ├── rvc/
│   ├── secrets/
│   ├── ui/
│   ├── io/scripts/
│   ├── export/
│   └── platform/win/
├── third_party/
│   └── rvc_sidecar/           <-- pinned w-okada release + launcher
├── resources/
│   ├── icons/
│   ├── qml/
│   ├── translations/
│   └── models/                <-- Silero VAD, RNNoise weights
├── tests/
│   ├── unit/
│   └── integration/
├── installer/
│   └── voxstudio.iss          <-- Inno Setup
└── tools/
    └── scripts/               <-- dev tooling, model converters
```

---

## 7. Coding Standards

- **C++20.** Use `std::span`, `std::optional`, `std::expected` (or `tl::expected` polyfill until MSVC ships full C++23), concepts where they clarify intent.
- **RAII everywhere.** No raw `new`/`delete`. Smart pointers: `std::unique_ptr` by default, `std::shared_ptr` only when shared ownership is real.
- **No exceptions across DLL boundaries.** Use `std::expected<T, ErrorCode>` for fallible operations.
- **Qt signals/slots** for cross-thread UI updates. Use `Qt::QueuedConnection` explicitly when crossing threads.
- **No raw threads in the audio path** — only the miniaudio callback runs on the audio thread; communicate via lock-free queues.
- **`const`-correct.** All getters const, all non-mutating member functions const.
- **`[[nodiscard]]`** on any function returning a status or owned resource.
- **clang-format** (project `.clang-format`, LLVM-derived, 100 col).
- **clang-tidy** with `bugprone-*`, `cert-*`, `cppcoreguidelines-*`, `modernize-*`, `performance-*`, `readability-*`.
- **Naming.** `PascalCase` types, `camelCase` functions/variables, `m_` prefix for member fields, `k` prefix for constants, `SCREAMING_SNAKE` for macros only.
- **Headers.** `#pragma once`. Include-what-you-use. Forward declare in headers when possible.
- **Tests.** Every module ships with Catch2 unit tests. UI tests via Qt Test. Coverage gate ≥ 70% on `core/`, `db/`, `dsp/`, `net/`.
- **Logging.** spdlog only. No `std::cout`, no `qDebug()` outside debug-only blocks.
- **No TODO without a GitHub issue link.** `// TODO(#123): ...`
- **Secrets never committed.** `.env*` and `secrets/` in `.gitignore`. Use DPAPI at runtime.

---

## 8. Pass-Based Development Roadmap

Every pass below has a fixed shape: **Goal → Result → Delivered → Verification → Codex Prompt**. Codex MUST report back in the Result / Delivered / Verification format after completing each pass.

---

### Pass 0 — Repo Scaffolding, CMake, Qt 6, vcpkg, CI

**Goal.** Stand up a buildable, lintable, testable empty Qt 6 application skeleton on Windows.

**Result.** `cmake --build` produces `VoxStudio.exe` that opens an empty `QMainWindow` titled "Vox Studio". CI passes on `windows-latest`.

**Delivered.**
- `CMakeLists.txt`, `CMakePresets.json` (`windows-msvc-debug`, `windows-msvc-release`).
- `vcpkg.json` manifest pinning Qt 6.7, SQLiteCpp, libcurl, cpr, IXWebSocket, nlohmann-json, spdlog, catch2, soxr, libsndfile, opus.
- `.clang-format`, `.clang-tidy`, `.editorconfig`, `.gitignore`, `.gitattributes`.
- `src/app/main.cpp` with Qt initialization, single-instance guard, spdlog bootstrap.
- `src/ui/MainWindow.{h,cpp}` empty shell.
- `.github/workflows/ci-windows.yml` (configure, build Debug+Release, run tests).
- `docs/CODING_STANDARDS.md` (extracted from §7).
- `docs/adr/0001-record-architecture-decisions.md`.

**Verification.**
1. `cmake --preset windows-msvc-debug && cmake --build --preset windows-msvc-debug` succeeds with zero warnings on `/W4`.
2. `ctest --preset windows-msvc-debug` runs the placeholder Catch2 test and passes.
3. CI green on a fresh PR.
4. Launch `VoxStudio.exe` — empty main window appears, closes cleanly, log file written to `%LOCALAPPDATA%/VoxStudio/logs/`.

**Codex Prompt.**
> Execute Pass 0 from `docs/DEVELOPMENT_PLAN.md`. Scaffold the Vox Studio repo exactly per the layout in §6 and the standards in §7. Use vcpkg manifest mode. Target Qt 6.7 LTS, MSVC 2022, C++20. Produce a buildable empty Qt application, a passing placeholder Catch2 test, and a green GitHub Actions workflow on `windows-latest`. When finished, reply with the **Result / Delivered / Verification** triplet and list every file you created.

---

### Pass 1 — Project HUD: Create / Open / Save (SQLite-backed)

**Goal.** Users can create a new Vox project, open an existing one, and see project metadata in the HUD.

**Result.** Main window shows a project HUD sidebar listing recent projects. "New Project" wizard creates a `.vox/` folder with a `project.db` (SQLite). "Open Project" loads it. Dirty-state tracking and "Save" wired up.

**Delivered.**
- `src/core/Project.{h,cpp}` — domain model.
- `src/db/Database.{h,cpp}`, `src/db/Migrations.{h,cpp}` — schema v1 (see §10).
- `src/db/ProjectRepository.{h,cpp}`.
- `src/ui/ProjectHud.{h,cpp}`, `src/ui/NewProjectWizard.{h,cpp}`.
- `src/ui/RecentProjectsModel.{h,cpp}` — backed by `QSettings`.
- Unit tests for migrations and round-trip serialization.

**Verification.**
1. Create a project named "TestVN" at a chosen path → folder `TestVN.vox/` appears with `project.db`.
2. Close and re-open the app → "TestVN" appears in Recent Projects.
3. Open it → HUD shows project name, created date, character/voice/line counts (all zero).
4. `tests/unit/test_project_repository.cpp` passes.
5. Manually inspect `project.db` with `sqlite3` CLI — `schema_migrations` table contains v1.

**Codex Prompt.**
> Execute Pass 1 from `docs/DEVELOPMENT_PLAN.md`. Implement the `Project` domain model, SQLite schema v1 per §10 (only the tables needed for v1: `schema_migrations`, `project_meta`, `characters`, `voices`, `scripts`, `lines`, `takes`), `ProjectRepository`, the Project HUD sidebar widget, and the New Project wizard. Wire dirty-state tracking into `MainWindow`. Write Catch2 unit tests for the repository and migration runner. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 2 — Secure API Key Storage (DPAPI) + ElevenLabs HTTP Client

**Goal.** Store the user's ElevenLabs API key encrypted at rest and expose a typed C++ client for the API surface we need.

**Result.** Settings dialog accepts an API key, stores it via DPAPI in `%LOCALAPPDATA%/VoxStudio/secrets/elevenlabs.bin`, and the client can hit `GET /v1/user` and `GET /v1/voices` successfully.

**Delivered.**
- `src/secrets/DpapiVault.{h,cpp}` — `CryptProtectData` / `CryptUnprotectData` wrappers using `CRYPTPROTECT_LOCAL_MACHINE = OFF` (user-scoped).
- `src/net/elevenlabs/Client.{h,cpp}` (cpr-based).
- `src/net/elevenlabs/Models.{h,cpp}` — strongly-typed DTOs.
- Endpoints implemented: `GET /v1/user`, `GET /v1/user/subscription`, `GET /v1/voices`, `GET /v1/models`.
- `src/ui/SettingsDialog.{h,cpp}` with API key field (masked), "Test Connection" button.
- Mock HTTP fixture for tests.

**Verification.**
1. Enter a valid API key → "Test Connection" shows subscription tier + remaining characters.
2. Enter an invalid key → clear error toast, no crash.
3. Inspect the secrets file — it is binary blob, not plaintext.
4. Delete the file → app prompts for key on next launch.
5. Unit tests cover JSON deserialization for all four endpoints with golden fixtures.

**Codex Prompt.**
> Execute Pass 2 from `docs/DEVELOPMENT_PLAN.md`. Implement DPAPI-backed secret storage (user scope, no machine scope), build the typed ElevenLabs HTTP client (cpr + nlohmann::json) for `/v1/user`, `/v1/user/subscription`, `/v1/voices`, `/v1/models`, and the Settings dialog with "Test Connection". All errors flow through `std::expected<T, ApiError>`. Add golden-fixture-based unit tests. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 3 — Voice Library: List, Import, Clone (Instant Voice Cloning)

**Goal.** Users can browse their ElevenLabs voice library and create new cloned voices from local audio samples.

**Result.** Voice Library panel shows all voices (premade + cloned). "Clone New Voice" dialog accepts 1–25 audio files (WAV/MP3/FLAC/OGG, total ≤ 11 minutes), name, description, labels → `POST /v1/voices/add` → new voice appears in the list. Voices are mirrored into local `voices` table for offline display.

**Delivered.**
- `src/net/elevenlabs/VoicesApi.{h,cpp}` — `addVoice`, `editVoice`, `deleteVoice`, `getVoice`.
- Multipart upload via cpr.
- `src/ui/VoiceLibraryPanel.{h,cpp}`.
- `src/ui/CloneVoiceDialog.{h,cpp}` with drag-drop, sample preview playback (miniaudio mini-instance), file validation.
- Local mirroring: `voices` table caches voice metadata; cache invalidated on app start + manual refresh.
- Consent checkbox: "I confirm I have the right to clone this voice" — required, persisted per-voice.

**Verification.**
1. Voice list populates within 2s on connect.
2. Drag 3 WAV samples (≥ 1 min total) of a public-domain voice → clone succeeds, new voice card appears.
3. Click new voice → preview button plays the default ElevenLabs sample line in that voice.
4. Edit voice description → persists round-trip.
5. Delete voice → confirmation dialog → removed from cloud and local cache.
6. Cloning without checking the consent box is blocked.

**Codex Prompt.**
> Execute Pass 3 from `docs/DEVELOPMENT_PLAN.md`. Implement the ElevenLabs Voices API surface (add/edit/delete/get) including multipart upload, the Voice Library panel, the Clone Voice dialog with sample preview, and local mirroring into the `voices` table. Enforce the consent checkbox and persist consent per voice in `voices.consent_confirmed_at`. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 4 — Script Viewer: Load Scripts, Line-by-Line Tracking

**Goal.** Users can import a script and the viewer tracks the current line, with per-line character assignment.

**Result.** Script Viewer panel loads `.txt`, Fountain, Ren'Py `.rpy`, Yarn Spinner JSON, and CSV. Lines are parsed into the `lines` table with character speaker tags. Current line is highlighted; ↑/↓ navigate, Enter confirms.

**Delivered.**
- `src/io/scripts/ScriptImporter.{h,cpp}` (dispatch).
- `src/io/scripts/PlainTextImporter.{h,cpp}` — heuristic `CHARACTER: line` parsing.
- `src/io/scripts/FountainImporter.{h,cpp}`.
- `src/io/scripts/RenpyImporter.{h,cpp}`.
- `src/io/scripts/YarnImporter.{h,cpp}`.
- `src/io/scripts/CsvImporter.{h,cpp}`.
- `src/ui/ScriptViewerPanel.{h,cpp}` (rich text rendering, keyboard nav).
- `src/ui/CharacterAssignmentDialog.{h,cpp}` (map detected speakers → voices).
- Fixture scripts in `tests/fixtures/scripts/`.

**Verification.**
1. Import each of the 5 supported formats from `tests/fixtures/scripts/` — line counts match expected.
2. Speakers auto-detected, dialog prompts for unmapped ones, mappings persist to `characters` table.
3. Keyboard navigation works; current line is visually highlighted.
4. Unknown character lines flagged in red with a tooltip.
5. Importer unit tests achieve > 90% coverage.

**Codex Prompt.**
> Execute Pass 4 from `docs/DEVELOPMENT_PLAN.md`. Implement the script import pipeline (plain text, Fountain, Ren'Py, Yarn, CSV), the Script Viewer panel with line tracking and keyboard navigation, and the Character Assignment dialog. Persist parsed lines into the `lines` table and speaker mappings into the `characters` table. Cover importers with fixture-based unit tests. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 5 — TTS Playback & Take Management

**Goal.** Generate a line's audio via ElevenLabs TTS, play it back, save it as a take.

**Result.** With a line selected and a character→voice mapping in place, pressing "Generate" produces audio via streaming TTS, plays it immediately, and saves the audio under `takes/<line_id>/<uuid>.opus` plus a row in the `takes` table. Each line can have multiple takes; user picks the active one.

**Delivered.**
- `src/net/elevenlabs/TtsApi.{h,cpp}` — `POST /v1/text-to-speech/{voice_id}/stream` with chunked playback.
- `src/audio/AudioEngine.{h,cpp}` (miniaudio init, default output device, shared-mode WASAPI).
- `src/audio/AudioFile.{h,cpp}` — Opus encode/decode via libopus + libsndfile.
- `src/audio/Resampler.{h,cpp}` — soxr wrapper.
- `src/core/TakeManager.{h,cpp}`.
- `src/ui/TakeListWidget.{h,cpp}` per-line takes with play/star/delete.
- Voice settings sliders on the line (stability, similarity_boost, style, speaker_boost).

**Verification.**
1. Generate a line → first audio arrives < 1s on a 100Mbps connection (streaming).
2. Take file appears under `takes/<line_id>/` and `takes` row references it.
3. Generate 3 more takes → all listed, can be A/B'd.
4. Star one take → it becomes the line's `active_take_id`.
5. Re-open project → active takes restored.
6. Disconnect network mid-stream → graceful error, no crash, partial file cleaned up.

**Codex Prompt.**
> Execute Pass 5 from `docs/DEVELOPMENT_PLAN.md`. Implement ElevenLabs streaming TTS (`/v1/text-to-speech/{voice_id}/stream`) with chunked playback through a miniaudio-based `AudioEngine`, Opus-encoded take storage via libsndfile+libopus, `TakeManager`, and the per-line takes UI. Expose stability / similarity_boost / style / speaker_boost as per-line overrides with project-level defaults. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 6 — Multi-Character Dialogue Mode (Hot Voice Switching)

**Goal.** Build a dialogue sequence by hot-switching the active voice line by line, producing a contiguous render.

**Result.** Sequence Builder mode: select a range of lines, hit "Generate Sequence" — each line is rendered with its character's voice, concatenated into a sequence preview with crossfades, and exportable as a single WAV. Hotkeys 1–9 hot-switch the "currently performing" voice for manual line-by-line work.

**Delivered.**
- `src/core/Sequence.{h,cpp}`, `src/core/SequenceRenderer.{h,cpp}`.
- `src/ui/TimelinePanel.{h,cpp}` — horizontal lane view, drag to reorder, per-line gap/pause control.
- Hotkey manager `src/ui/HotkeyManager.{h,cpp}`.
- Sequence export to single WAV/Opus.
- Concurrent generation pool (configurable, default 3 parallel ElevenLabs requests respecting their concurrency limits).

**Verification.**
1. 20-line scene with 4 characters renders sequentially → final WAV plays back coherently.
2. Per-line pause slider (0–3s) audibly inserts silence.
3. Hotkeys 1–9 swap the "active character" indicator; if the user types and hits Generate, the active voice is used.
4. Cancelling mid-render leaves no orphan takes; DB is consistent.
5. Parallelism honored (verify with HTTP logs).

**Codex Prompt.**
> Execute Pass 6 from `docs/DEVELOPMENT_PLAN.md`. Implement `Sequence`, `SequenceRenderer`, the Timeline panel with per-line gaps and reordering, hot-switch hotkeys (1–9), and parallel TTS rendering bounded by a configurable concurrency cap. Export a sequence as a single WAV/Opus with deterministic crossfades. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 7 — Real-Time Audio Engine: WASAPI Capture, VAD, Lock-Free Ring Buffer

**Goal.** Get the user's microphone into a clean, low-latency, voice-activity-gated stream ready for downstream voice conversion.

**Result.** Live Mic panel shows input device picker, level meter, gain, VAD indicator, and a "Monitor" toggle that loops the mic back to the output through RNNoise. End-to-end loopback latency under 60ms on shared-mode WASAPI.

**Delivered.**
- `src/audio/Capture.{h,cpp}` — miniaudio duplex device, configurable buffer (10/20ms frames).
- `src/audio/RingBuffer.{h,cpp}` — `moodycamel::ReaderWriterQueue<AudioFrame>` wrapper.
- `src/dsp/NoiseSuppressor.{h,cpp}` — RNNoise 48k path (resample down from 44.1/48k as needed).
- `src/dsp/Vad.{h,cpp}` — Silero VAD ONNX via ONNX Runtime.
- `src/ui/LiveMicPanel.{h,cpp}` — meter, device picker, monitor toggle, "Test Latency" button.
- `src/audio/LatencyProbe.{h,cpp}` — measures round-trip via impulse.

**Verification.**
1. Pick mic + speakers → enable Monitor → speak into mic → hear yourself with noise suppressed.
2. "Test Latency" reports ≤ 60ms on a typical USB mic + WASAPI shared.
3. VAD indicator flashes green when speaking, idle when silent.
4. Drop input device mid-stream → graceful recovery, no crash, error surfaced to UI.
5. Audio thread never holds a mutex (verified via `clang::no_thread_safety_analysis` audit + targeted test).

**Codex Prompt.**
> Execute Pass 7 from `docs/DEVELOPMENT_PLAN.md`. Implement WASAPI capture+playback via miniaudio with a lock-free ring buffer (moodycamel RWQ), RNNoise noise suppression, Silero VAD via ONNX Runtime, the Live Mic panel UI, and a round-trip latency probe. Hard requirement: no locks on the audio thread. Reply with the **Result / Delivered / Verification** triplet, and include the measured loopback latency.

---

### Pass 8 — ElevenLabs Speech-to-Speech (Cloud Real-Time Path)

**Goal.** Add cloud-based mic→character voice conversion as the high-quality, easy-to-ship path before tackling local RVC.

**Result.** While in Live Mic mode, the user selects a target voice and hits "Convert (Cloud)" — captured audio is buffered, sent to `POST /v1/speech-to-speech/{voice_id}/stream`, and the converted audio plays back. Also supports recording a take from a STS pass.

**Delivered.**
- `src/net/elevenlabs/StsApi.{h,cpp}` — streaming STS endpoint.
- `src/ui/LiveMicPanel` extended with cloud/local mode switch.
- Chunked upload + chunked playback pipeline.
- Cost meter (estimated characters/credits consumed) shown live.

**Verification.**
1. Speak a 5-second line → converted audio plays back within ~1.5s.
2. Record a take from STS → saved as Opus under the active line.
3. Cancel mid-conversion → upstream request cancelled, no orphan files.
4. Subscription with no remaining credits → clear blocking dialog, no silent failure.

**Codex Prompt.**
> Execute Pass 8 from `docs/DEVELOPMENT_PLAN.md`. Implement streaming Speech-to-Speech against `/v1/speech-to-speech/{voice_id}/stream`, including chunked upload of captured audio and chunked playback of converted audio. Wire it into the Live Mic panel as the "Cloud" mode. Surface a live cost meter. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 9a — Local RVC via Bundled w-okada Python Sidecar (MVP Real-Time Path)

**Goal.** Ship a working local real-time RVC mic-to-character voice path without writing a native inference engine.

**Result.** On first run, the app verifies/installs a pinned w-okada/voice-changer release into `%LOCALAPPDATA%/VoxStudio/rvc_sidecar/` (Python embeddable + pip wheels, offline-bundled). The Live Mic panel's "Local" mode starts the sidecar on a local port, streams 20ms frames over HTTP (or WS), receives converted audio, and plays it back with ≤ 300ms total latency on a CUDA-capable GPU.

**Delivered.**
- `third_party/rvc_sidecar/` — pinned w-okada release with a small launcher wrapper.
- `src/rvc/RvcSidecar.{h,cpp}` — process supervisor (start, health-check, graceful shutdown, auto-restart).
- `src/rvc/RvcClient.{h,cpp}` — typed HTTP/WS client to the sidecar.
- `src/rvc/RvcModelRegistry.{h,cpp}` — local catalogue of `.pth` / `.index` files in `%LOCALAPPDATA%/VoxStudio/rvc_models/`.
- `src/ui/RvcModelManagerDialog.{h,cpp}` — import, delete, set per-character RVC model.
- `installer/` snippets for shipping the sidecar.
- Health/diagnostics panel in Settings (GPU detected, CUDA version, model loaded, last latency).

**Verification.**
1. Cold start: sidecar boots within 20s on first launch, 5s on subsequent runs.
2. Speak with a loaded RVC model selected → converted voice plays back with measured end-to-end latency ≤ 300ms on RTX 3060, ≤ 500ms on CPU fallback.
3. Kill the sidecar process externally → supervisor restarts it within 3s, UI shows transient "Reconnecting…" state.
4. Switch RVC model mid-session → seamless changeover in < 1s.
5. Uninstall the app → sidecar processes terminated and folder cleaned.

**Codex Prompt.**
> Execute Pass 9a from `docs/DEVELOPMENT_PLAN.md`. Bundle the pinned w-okada/voice-changer release under `third_party/rvc_sidecar/`. Implement `RvcSidecar` (process supervisor with health checks and auto-restart), `RvcClient` (HTTP/WS streaming), `RvcModelRegistry`, and the RVC Model Manager dialog. Wire "Local" mode in the Live Mic panel. Show GPU + latency diagnostics in Settings. Reply with the **Result / Delivered / Verification** triplet and include measured latency on your test hardware.

---

### Pass 9b — Native ONNX RVC Engine (Post-MVP Optimization)

**Goal.** Replace the Python sidecar with a fully native ONNX Runtime C++ inference engine for smaller install footprint and lower overhead.

**Result.** New `OnnxRvcEngine` runs in-process. Local mode can be switched between "Sidecar" and "Native" in Settings. Native path matches or beats sidecar latency, installs ~400MB smaller.

**Delivered.**
- `src/rvc/OnnxRvcEngine.{h,cpp}` — ONNX Runtime C++ session manager.
- `src/rvc/F0Extractor.{h,cpp}` — RMVPE ONNX path (preferred), Crepe fallback.
- `src/rvc/HubertEncoder.{h,cpp}` — content encoder ONNX.
- `tools/scripts/export_rvc_to_onnx.py` — documented conversion pipeline (acknowledging RVC issue #2684 quirks).
- Model compatibility matrix in `docs/RVC_NATIVE.md`.

**Verification.**
1. Convert a reference RVC model to ONNX and run it natively → output A/B-indistinguishable from sidecar within ±2dB spectral.
2. Native end-to-end latency ≤ sidecar latency on identical hardware.
3. Installer size reduced by ≥ 300MB.
4. Settings toggle "Sidecar / Native" switches engines without restart.

**Codex Prompt.**
> Execute Pass 9b from `docs/DEVELOPMENT_PLAN.md`. Implement the native ONNX Runtime RVC engine (`OnnxRvcEngine`, `F0Extractor` with RMVPE, `HubertEncoder`), a documented `.pth → ONNX` conversion script, and a Settings toggle to switch between sidecar and native. Do not remove the sidecar path — it remains the supported fallback. Reply with the **Result / Delivered / Verification** triplet plus A/B spectral comparison.

---

### Pass 10 — Export Pipeline (Game-Engine-Friendly Output)

**Goal.** Batch-export all active takes with naming and folder conventions that drop cleanly into Unity, Unreal, Ren'Py, and Godot.

**Result.** "Export" wizard: pick destination, target engine, format (WAV 16-bit PCM / WAV 24-bit / Opus / FLAC), normalization (LUFS target), naming template, optional per-character subfolders. Runs as a background job with progress, generates a manifest CSV/JSON.

**Delivered.**
- `src/export/ExportJob.{h,cpp}`.
- `src/export/Normalizer.{h,cpp}` — EBU R128 LUFS via libsndfile + custom integrator (or vendored `libebur128`).
- `src/export/NamingTemplate.{h,cpp}` — tokens: `{character}`, `{line_id}`, `{scene}`, `{index}`, `{slug}`, `{lang}`.
- Engine presets: Unity (16-bit WAV, per-character folder), Unreal (24-bit WAV, flat), Ren'Py (Opus, `voice/<character>/<id>.opus`), Godot (Ogg Vorbis).
- `src/ui/ExportWizard.{h,cpp}`.
- Manifest emitter (CSV + JSON).

**Verification.**
1. Export a 50-line project at Unity preset → folder structure matches expected, all files at -23 LUFS ±0.5.
2. Manifest CSV columns: `file`, `character`, `line_id`, `original_text`, `duration_ms`, `lufs`, `voice_id`, `model`, `created_at`.
3. Cancel mid-export → cleanup leaves no partial files.
4. Re-export skips unchanged takes (incremental mode).

**Codex Prompt.**
> Execute Pass 10 from `docs/DEVELOPMENT_PLAN.md`. Implement the export wizard with Unity / Unreal / Ren'Py / Godot presets, EBU R128 LUFS normalization, the token-based naming template, manifest emission (CSV + JSON), and incremental re-export. Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 11 — Installer, Code Signing, Auto-Updater, Telemetry (Opt-In)

**Goal.** Ship a real, signed, updatable Windows installer.

**Result.** Inno Setup installer produces a signed `VoxStudio-Setup-x.y.z.exe`. SmartScreen-clean after EV cert + reputation. In-app auto-updater checks GitHub Releases, downloads delta, verifies signature, prompts for restart. Telemetry is opt-in (off by default), anonymized, and documented.

**Delivered.**
- `installer/voxstudio.iss`.
- `cmake/Packaging.cmake` — `cpack` wrapper invoking ISCC.
- Sign step in `.github/workflows/release.yml` using EV cert from secrets.
- `src/app/Updater.{h,cpp}` — GitHub Releases poller with Authenticode signature verification.
- `src/app/Telemetry.{h,cpp}` — opt-in, documented schema in `docs/PRIVACY.md`.
- `docs/PRIVACY.md`, `docs/EULA.md`, `docs/THIRD_PARTY_LICENSES.md`.

**Verification.**
1. Install on a fresh Win11 VM → no SmartScreen "Unknown Publisher" warning.
2. Uninstaller removes all program files + bundled sidecar + Start Menu shortcuts. User data under `%LOCALAPPDATA%/VoxStudio/` is preserved (opt-in delete).
3. Bump version, publish GitHub Release → running app detects update, downloads, restarts cleanly.
4. Telemetry off by default; toggle on → privacy dialog shown listing every field sent.

**Codex Prompt.**
> Execute Pass 11 from `docs/DEVELOPMENT_PLAN.md`. Author `installer/voxstudio.iss`, the release workflow with EV code signing, the in-app updater (GitHub Releases + Authenticode verification), and opt-in telemetry. Write `PRIVACY.md`, `EULA.md`, and `THIRD_PARTY_LICENSES.md` (auto-generated from vcpkg). Reply with the **Result / Delivered / Verification** triplet.

---

### Pass 12 — Polish, Accessibility, Localization Scaffold (Bonus)

**Goal.** Bring the app to a shippable polish bar.

**Result.** Dark/light themes, keyboard accessibility (full tab order, screen reader labels), Qt Linguist translation scaffold (English ships, French/Spanish/Japanese stubs), in-app onboarding tour, crash reporter (opt-in, sentry-native or Crashpad), comprehensive logging viewer.

**Delivered.**
- Theming via Qt stylesheets, persisted in `QSettings`.
- Accessibility audit pass with `AccessibilityInsights for Windows`.
- `resources/translations/voxstudio_*.ts` files, `lupdate` integrated into CMake.
- `src/ui/OnboardingTour.{h,cpp}`.
- `src/app/CrashReporter.{h,cpp}` — Crashpad integration, opt-in.
- `src/ui/LogViewerDialog.{h,cpp}`.

**Verification.**
1. Full keyboard nav from app launch to render → no mouse needed.
2. NVDA reads every interactive element.
3. Switch language to French at runtime → all visible strings update.
4. Trigger a controlled crash via debug menu → minidump uploaded with user consent.

**Codex Prompt.**
> Execute Pass 12 from `docs/DEVELOPMENT_PLAN.md`. Add dark/light theming, full keyboard accessibility, Qt Linguist scaffolding with English + 3 stub locales, an onboarding tour, opt-in Crashpad crash reporting, and an in-app log viewer. Reply with the **Result / Delivered / Verification** triplet.

---

## 9. Dependency Table

| Dependency | Version | License | Purpose | Linking |
|---|---|---|---|---|
| Qt 6 Base / Widgets / Network / Concurrent | 6.7 LTS | LGPLv3 | UI, networking, threadpool | Dynamic |
| SQLiteCpp | 3.3.x | MIT | C++ wrapper over sqlite3 | Static |
| sqlite3 | 3.45+ | Public Domain | Embedded DB | Static |
| libcurl | 8.x | curl | HTTP transport | Dynamic |
| cpr | 1.11.x | MIT | C++ wrapper over libcurl | Static |
| IXWebSocket | 11.x | BSD-3 | WebSocket client | Static |
| nlohmann/json | 3.11+ | MIT | JSON | Header-only |
| spdlog | 1.13+ | MIT | Logging | Static |
| miniaudio | 0.11+ | Public Domain / MIT-0 | WASAPI capture/playback | Header-only |
| dr_wav / dr_flac / dr_mp3 | latest | Public Domain | Audio decode | Header-only |
| libsndfile | 1.2 | LGPLv2.1 | Robust audio I/O | Dynamic |
| libopus | 1.4 | BSD-3 | Take compression | Static |
| soxr | 0.1.3 | LGPLv2.1 | Sample rate conversion | Dynamic |
| RNNoise | latest | BSD-3 | Mic noise suppression | Static |
| ONNX Runtime | 1.18+ | MIT | Silero VAD, native RVC (Pass 9b) | Dynamic |
| Silero VAD model | v4 | MIT | VAD weights | Asset |
| Catch2 | v3 | BSL-1.0 | Unit tests | Static |
| moodycamel ReaderWriterQueue | latest | BSD-2 / Boost | Lock-free SPSC queue | Header-only |
| libebur128 (Pass 10) | 1.2+ | MIT | LUFS normalization | Static |
| Crashpad (Pass 12) | latest | Apache-2.0 | Crash reporting | Static |
| w-okada/voice-changer | pinned release | MIT | Local RVC sidecar | Bundled |
| Inno Setup | 6.x | Modified BSD | Installer | Build-time only |

**License compliance.** Qt 6, libsndfile, and soxr are LGPL — dynamic linking only, with `THIRD_PARTY_LICENSES.md` and required notices shipped in the installer. No GPL, AGPL, or "non-commercial only" components.

---

## 10. SQLite Project Schema

```sql
-- v1 migration
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE project_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- keys: name, created_at, updated_at, vox_schema_version, target_engine, default_voice_settings_json

CREATE TABLE characters (
  id TEXT PRIMARY KEY,             -- uuid
  name TEXT NOT NULL UNIQUE,
  color TEXT,                       -- #RRGGBB
  voice_id TEXT,                    -- FK voices.id, nullable
  rvc_model_id TEXT,                -- FK rvc_models.id, nullable
  notes TEXT
);

CREATE TABLE voices (
  id TEXT PRIMARY KEY,              -- ElevenLabs voice_id
  name TEXT NOT NULL,
  origin TEXT NOT NULL,             -- 'premade' | 'ivc' | 'pvc' | 'shared'
  labels_json TEXT,
  default_settings_json TEXT,       -- stability, similarity_boost, style, use_speaker_boost
  consent_confirmed_at TEXT,
  last_synced_at TEXT
);

CREATE TABLE rvc_models (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  pth_path TEXT NOT NULL,
  index_path TEXT,
  sample_rate INTEGER NOT NULL,
  notes TEXT,
  imported_at TEXT NOT NULL
);

CREATE TABLE scripts (
  id TEXT PRIMARY KEY,
  source_path TEXT,
  format TEXT NOT NULL,             -- 'txt'|'fountain'|'rpy'|'yarn'|'csv'
  imported_at TEXT NOT NULL
);

CREATE TABLE lines (
  id TEXT PRIMARY KEY,
  script_id TEXT NOT NULL REFERENCES scripts(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  character_id TEXT REFERENCES characters(id),
  text TEXT NOT NULL,
  scene_tag TEXT,
  voice_settings_json TEXT,         -- per-line override
  active_take_id TEXT REFERENCES takes(id),
  UNIQUE(script_id, ord)
);

CREATE TABLE takes (
  id TEXT PRIMARY KEY,
  line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
  source TEXT NOT NULL,             -- 'tts'|'sts'|'rvc_local'|'manual'
  voice_id TEXT,
  rvc_model_id TEXT,
  file_path TEXT NOT NULL,          -- relative to project root
  duration_ms INTEGER,
  lufs REAL,
  starred INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  metadata_json TEXT
);

CREATE TABLE sequences (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE sequence_items (
  sequence_id TEXT NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
  gap_ms INTEGER NOT NULL DEFAULT 250,
  PRIMARY KEY (sequence_id, ord)
);

CREATE INDEX idx_lines_script ON lines(script_id, ord);
CREATE INDEX idx_takes_line ON takes(line_id);
```

---

## 11. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | ElevenLabs API change/breakage | M | H | Versioned client, contract tests against fixtures, feature-flag new endpoints |
| R2 | RVC `.pth → ONNX` export quirks (RVC issue #2684) | H | M | Ship sidecar-first (Pass 9a); native ONNX is Pass 9b, treat as optimization |
| R3 | Qt LGPL compliance miss | L | H | Dynamic linking only; auto-generated THIRD_PARTY_LICENSES.md gated in CI |
| R4 | Voice cloning consent abuse | M | H | Mandatory consent checkbox + ElevenLabs ToS enforcement + watermarking optional |
| R5 | SmartScreen blocks unsigned installer | H (early) | M | EV code signing cert from day one; budget the 1–2 week procurement window |
| R6 | Latency target missed on weak GPUs | M | M | CPU fallback model + clear hardware spec on the download page |
| R7 | Python sidecar conflicts with user's system Python | M | M | Use embeddable Python under `%LOCALAPPDATA%`, never touch system PATH |
| R8 | VB-CABLE not installed → user can't pipe to Discord/games | M | L | Detect on first run, link to vendor, do not bundle |
| R9 | Audio thread accidentally taking a lock → glitches | M | H | Static analysis + targeted thread-safety tests, code review checklist |
| R10 | ElevenLabs credit exhaustion mid-render | M | M | Pre-flight cost estimate before sequence render, hard cap configurable |
| R11 | Disk full during large export | L | M | Pre-flight free-space check, graceful pause-and-resume |
| R12 | Sample at-rest privacy leak | L | M | Optional DPAPI encryption of `voices/samples/` |

---

## 12. Glossary

- **IVC** — Instant Voice Cloning (ElevenLabs). ~1 minute of audio, available in ~seconds.
- **PVC** — Professional Voice Cloning (ElevenLabs). ≥ 30 minutes of audio, higher fidelity, asynchronous training.
- **TTS** — Text-to-Speech.
- **STS** — Speech-to-Speech (ElevenLabs voice conversion).
- **RVC** — Retrieval-based Voice Conversion. Open-source model family for mic-driven voice conversion.
- **RMVPE** — Robust Model for Vocal Pitch Estimation. Preferred f0 extractor for RVC.
- **VAD** — Voice Activity Detection.
- **WASAPI** — Windows Audio Session API.
- **DPAPI** — Data Protection API. Windows-native at-rest encryption tied to the user account.
- **LUFS** — Loudness Units Full Scale. Broadcast loudness standard (EBU R128).
- **Take** — A single rendered audio file for a script line. A line can have many takes; one is "active".
- **Sequence** — An ordered list of lines rendered into a single continuous audio file.

---

## 13. Primary Source Index

- ElevenLabs API: <https://elevenlabs.io/docs/api-reference/introduction>
- ElevenLabs pricing: <https://elevenlabs.io/pricing>
- ElevenLabs ToS: <https://elevenlabs.io/terms-of-use>
- w-okada/voice-changer: <https://github.com/w-okada/voice-changer>
- RVC-Project (Retrieval-based-Voice-Conversion-WebUI): <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>
- RVC ONNX export issue #2684: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2684>
- Applio (modern RVC fork): <https://github.com/IAHispano/Applio>
- so-vits-svc: <https://github.com/svc-develop-team/so-vits-svc>
- DDSP-SVC: <https://github.com/yxlllc/DDSP-SVC>
- Qt 6 LGPL FAQ: <https://www.qt.io/licensing/open-source-lgpl-obligations>
- ONNX Runtime C++ API: <https://onnxruntime.ai/docs/api/c/>
- Microsoft WASAPI low-latency guide: <https://learn.microsoft.com/windows-hardware/drivers/audio/low-latency-audio>
- miniaudio: <https://miniaud.io/>
- PortAudio: <https://www.portaudio.com/>
- RtAudio: <https://www.music.mcgill.ca/~gary/rtaudio/>
- JUCE licensing: <https://juce.com/legal/juce-8-licence/>
- RNNoise: <https://github.com/xiph/rnnoise>
- Silero VAD: <https://github.com/snakers4/silero-vad>
- VB-CABLE: <https://vb-audio.com/Cable/>
- VoiceMeeter: <https://vb-audio.com/Voicemeeter/>
- OBS Studio (Qt + real-time audio reference): <https://github.com/obsproject/obs-studio>
- Audacity: <https://github.com/audacity/audacity>
- Inno Setup: <https://jrsoftware.org/isinfo.php>
- WiX Toolset: <https://wixtoolset.org/>
- Qt Installer Framework: <https://doc.qt.io/qtinstallerframework/>
- vcpkg: <https://vcpkg.io/>
- libsndfile: <https://libsndfile.github.io/libsndfile/>
- soxr: <https://sourceforge.net/projects/soxr/>
- libopus: <https://opus-codec.org/>
- dr_libs: <https://github.com/mackron/dr_libs>
- Crashpad: <https://chromium.googlesource.com/crashpad/crashpad/>
- moodycamel ReaderWriterQueue: <https://github.com/cameron314/readerwriterqueue>
- DPAPI (CryptProtectData): <https://learn.microsoft.com/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata>
- EV Code Signing on Windows: <https://learn.microsoft.com/windows/win32/seccrypto/cryptography-tools>

---

*Codex must read this file before starting any pass and must respond using the **Result / Delivered / Verification** report shape defined at the top of §8.*
