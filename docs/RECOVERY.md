# Recovering Vox Studio on another machine

This repository preserves the application, its local emotion and performance
controls, and the supporting source tools. It is a source backup. Private voice
recordings, trained character models, projects and credentials are not stored in
this public repository.

## What is consolidated

| Capability | Source |
| --- | --- |
| Native Voice Lab inventory, previews and render ownership | `src/ui/VoiceLabPanel.*`, `tools/voice_studio_lab.py` |
| IndexTTS-2.5 eight-component emotion control | `tools/indextts_audition.py`, `tools/narration_audition.py` |
| Chatterbox expression controls and Qwen audition options | `tools/narration_audition.py` |
| VoxCPM delivery, pronunciation, LoRA performance and vocal actions | `third_party/voxcpm_sidecar/`, the existing preparation and audition tools |
| Performance Mirror and RVC | `third_party/`, `src/`, `tools/train_character_rvc.py` |
| Character corpus, delivery analysis, training, rendering and evaluation tools | `tools/character_lab/` |
| Narration assembly and timing review | `tools/assemble_narration_chunks.py`, `tools/review_narration_timing.py` |

The character-lab snapshot contains 63 Python files, copied byte for byte from
the working laboratory on October 26, 2026. `SOURCE_MANIFEST.json` records their
relative paths, sizes and SHA256 hashes. Those copies are recovery sources; the
desktop continues using the configured external lab. This avoids changing a
working installation or invalidating its recorded previews during a backup.

## Rebuild the application

Clone the repository and check out the backup commit or branch. Follow the
README's Visual Studio, Qt/vcpkg and CMake setup. The vcpkg baseline is recorded
in `vcpkg-configuration.json`. Python tool tests require `numpy` and `soundfile`
in the Python interpreter selected by CMake. Install the pinned CPU test
dependencies with `python -m pip install -r tests/requirements.txt` in a dedicated
test environment. Select it using `-DPython3_EXECUTABLE=<path-to-python>` when
configuring CMake. GPU model packages belong in their own engine environments.

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel 1
ctest --preset windows-msvc-release --output-on-failure --parallel 1
python tests/tools/test_character_lab_snapshot.py
```

The build bundles the Voice Lab, narration and IndexTTS workers. Build success
does not install model weights or establish that an audition sounds convincing.

## Restore the character-lab tools

Verify the source manifest with the test above. Into a **new, empty** recovery
directory, copy `tools/character_lab/baodur-lab` and
`tools/character_lab/characters` as sibling folders. The directory should become
the external `voice-lab` root. Keep that layout: the shared tools import each
other using their relative locations. Do not copy over a surviving lab.

Set `labRoot` and `voicesRoot` in
`%LOCALAPPDATA%/VoxStudio/voice_lab.json` to the restored external lab and private
voice-data directories. The character loader reads private
`characters/<slug>.json` files, whose `lab`, `source_audio` and `exact_manifest`
paths must point to their restored data. Restore each character's corpus,
delivery/content, recipes, selections and trained checkpoints separately.

For rebuilding a training configuration, copy
`baodur-lab/voxcpm/train_lora_v2.example.yaml` to `train_lora_v2.yaml` and replace
its model path. The manifest builder supplies the dataset and output paths.
The example records settings only; it contains no character data or training
approval. Existing manifest/config/speaker-screening checks still apply.

## Restore the emotion engine and local runtimes

`tools/character_lab/UPSTREAM_REVISIONS.json` records public source/model
identities recovered from the installed IndexTTS and evaluation runtimes. The
IndexTTS code commit is `d9e41aac89fd00b3d71497fddb287b7f24613712`; its main
IndexTTS-2.5 model revision is `c39ce5ba981572cb187443877ff559dfb246ce63`.
Keep the upstream lockfile and model licenses. Model acquisition remains a
separate installation step under those terms. No third-party engine weights or
private installation provenance are vendored here.

Restore or recreate the isolated engine environments at their documented
locations. Do not copy a Windows virtual environment between machines and
assume it works. Restore the original private installation records or record a
fresh installation provenance with the actual model revisions and file hashes.
IndexTTS's adapter requires that provenance, model configuration, upstream
sources, licenses and checkpoints before preview. Re-preview all recipes after
moving files; old previews are intentionally bound to the previous paths and
hashes. Preserve resource admission, offline loading, single-use render claims,
reference consent and watermark behavior.

VoxCPM2 installation starts with `tools/install_voxcpm_runtime.ps1` and
`third_party/voxcpm_sidecar/runtime_requirements.txt`. The working installation
used VoxCPM 2.0.3 and model revision
`32279effe8c19989596f05d353d1447f51d9e915`. The installer does not guarantee that
an unqualified future download selects that same model snapshot; restore or
acquire the recorded revision explicitly for a matching installation.
Qwen, Chatterbox, GPT-SoVITS, Seed-VC and evaluation engines retain their separate
installation requirements. This backup has not demonstrated a clean-machine
GPU installation of every engine.

## Private data needed for a complete recovery

Keep an encrypted backup on storage independent of this machine of:

- `%LOCALAPPDATA%/VoxStudio/voxcpm_profiles`, `voxcpm_training`, `rvc_models`,
  `rvc_onnx_models`, generated vocal actions, project/take data and relevant
  private engine installation records.
- The external lab's private character configuration and recipe data, plus
  the directory configured as `voicesRoot`, including selected checkpoints,
  mastering and selection records.
- Authorized original voice references, `.vox` projects and generated dialogue
  masters stored outside those folders.

Keep recording permissions and model licenses with that private backup. Obtain
or restore third-party assets under their terms. Enter the ElevenLabs key again
on the replacement machine; DPAPI-protected credentials from the old Windows
account are not a portable key backup. Never commit that private backup to this
repository.
