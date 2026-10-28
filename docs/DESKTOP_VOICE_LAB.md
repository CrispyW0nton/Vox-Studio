# Native Voice Lab consolidation — October 9, 2026

The **Voice Lab** tab in Vox Studio discovers the local engine installations,
character labs, trained checkpoints, desktop profiles and RVC models. Selecting
a character exposes its source decisions, performance bible, takes, mastering
files and listening pack. Files remain in their existing locations; the desktop
does not duplicate model weights or replace trained profiles.

The inventory refreshes when first opened. **Refresh inventory** picks up new
checkpoints and profiles. “Present” means files exist; it does not certify an
engine, a voice or a performance.

## Render workflows

Choose a recipe, choose its JSON script, optionally select comma-separated line
IDs and 1–6 seeds, then **Preview render**. VoxCPM recipes also expose the chosen
LoRA checkpoint. A successful preview unlocks **Render preview**. The workers use
the existing lab's provenance, speaker-screening and held-out-reference checks.
Editing inputs or changing referenced files requires another preview.

Supported desktop launchers:

- VoxCPM2 lab synthesis with original audio, selected LoRA and character recipe.
- GPT-SoVITS: Bao-Dur's recorded g15/s8/aux2 selection.
- IndexTTS 2.5: original-reference performance auditions for configured characters.
- Qwen3-TTS and Chatterbox: import an existing `narration_audition.py` recipe.
  The imported recipe retains its reference hash, model snapshot, texts, seeds,
  Qwen options and Chatterbox watermarking. Its outputs stay in `VoiceAuditionV1`.
- IndexTTS-2.5 direct auditions: import an `indextts25` recipe to keep a fixed
  authorized voice reference while supplying bounded emotion vectors per phrase.
  This uses the installed external engine, not a configured character's corpus.
  Source/model/terms hashes, output claims and all admission limits remain in
  effect. The model has its own Bilibili terms; see [audition details](NARRATION_AUDITIONS.md).

New lab jobs and receipts live in `%LOCALAPPDATA%/VoxStudio/desktop_jobs/`.
Outputs are raw/trimmed auditions, **not mastered, approved game replacements**.
The character's linked mastering and listening files remain the evidence for
the complete sound chain. No game files are installed by this workspace.

RVC, live Seed-VC Performance Mirror, Dia reactions, lip-sync, music and evaluation
runtimes are also inventoried. Their existing specialized workflows remain in
Live Mic or their local tooling; they are not relabeled as text-to-speech engines.

## Character selections

The October 8 director results supersede the earlier continuation-for-everyone
proposal. Evidence is linked in each character row:
`Voices/VOICE_DELIVERY02/pilot_gpu_v001/scores/DIRECTOR_RESULTS_v001.md`.

| Character | Desktop starting recipe | Qualification |
|---|---|---|
| Atris | LoRA v2 step 400, cfg 1.5, `idref_instr`, instruction B, original identity reference | Two mastered pilot takes rated 82/78; new lines still need listening. Uses the exact v005 KJA route script. |
| Atton | LoRA v2 step 800, cfg 1.5, `idref_instr`, instruction B | One favored pilot take; provisional across other lines. |
| Bao-Dur | LoRA v2 step 400, cfg 2.0, `ref_prompt`, original pilot reference map | Preset excludes lines using rejected prompt `gblbaodur227`; GPT-SoVITS remains separately available. |

The plain spelling “Echani” is preserved for Atris. The native legacy TTS and live
voice profiles are not silently replaced by these audition selections.

## Local paths and resource handling

Default discovery uses `%LOCALAPPDATA%/VoxStudio`,
`~/Documents/GDeveloper/External/voice-lab` and `~/Documents/KotorMods/Voices`.
Override the last two with `labRoot` and `voicesRoot` in
`%LOCALAPPDATA%/VoxStudio/voice_lab.json` if the library moves.

Render admits one non-display GPU by UUID with at least 18,000 MiB free, keeps
4 GiB free on every display GPU, and requires 10 GiB physical RAM and 17 GiB
commit free before loading. The worker's stricter existing checks also apply.
The shared `Voices/gpu_train.lock` is exclusive and never reclaimed from another
session. A 3 GiB host-memory floor stops only the owned worker. **Stop** preserves
partial output and does not automatically retry; create a new job after review.

## Build and validation

The `VoxStudioVoiceLabBundle` target refreshes the Python bridge, narration
worker and IndexTTS adapter on every build, including builds with no C++ changes. Models remain local.
The desktop shortcut targets `C:/_b/voxstudio/r/Release/VoxStudio.exe`.

Run `ctest --preset windows-msvc-release --output-on-failure`. The suite includes
native Voice Lab UI tests and bridge tests for changed references, invalid inputs,
single-use jobs, shared locks and resource admission. Actual engine dry runs are
separate from GPU synthesis or human listening acceptance.

### Verification of this update

- Release build succeeded; the existing desktop shortcut targets that executable.
- All 18 CTest suites passed, including the 10 new bridge contract tests.
- All 11 narration audition tests passed.
- Native Qt verification passed all five checks with the real local inventory,
  including an Atris E198 preview launched from the panel and a saved receipt.
- Actual worker previews passed for Atris, Atton, Bao-Dur VoxCPM2, Bao-Dur
  GPT-SoVITS, Bao-Dur/Jolee IndexTTS, Qwen and Chatterbox.
- Bundled bridge and narration worker hashes match their current source files.

Local evidence: `build/desktop-consolidation-ctest.log`,
`build/desktop-narration-tests.log`, `build/voice-lab-qt-visual.txt`,
`build/desktop-voice-lab.bmp`, and `build/desktop-previews/`.
This update did not run GPU synthesis, train models, master audio or install
game dialogue. The external Computer Use launcher failed; native UI validation
was performed with the Qt test harness.
