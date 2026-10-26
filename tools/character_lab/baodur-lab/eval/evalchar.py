"""Which character the eval harness scores, and that character's evaluation content (stdlib only).

The character is VOICE_LAB_CHARACTER=<slug> (default "baodur"), read through the shared voice-lab module
characters/character.py: display name, lab folder, ASR prompt. Every file the harness reads or writes lives under
that character's lab:
  <lab>/corpus.jsonl, <lab>/restored/, <lab>/restore_report.json            corpus task (restored is optional)
  <lab>/delivery/takes_script.json, delivery_map.json, exclude_other_speakers.json   delivery task (optional)
  <lab>/eval/<run>/, <lab>/eval/_cache/, <lab>/eval/_selftest/, <lab>/eval/speaker_purity_<variant>.json
Character-specific evaluation CONTENT lives in <lab>/eval/eval_character.json (FILE_NAME), one section each:
  speaker_screen  impostor voices: folders of other characters' clips (absolute, or relative to the Voices folder
                  that holds the labs) and how the scorecard names them; in-set impostor speakers (delivery-task
                  groups of another voice inside the character's own clip set, used as a channel-matched floor);
                  the other-speaker screen thresholds (train clips scored as another voice) and the shortest clip
                  averaged into the speaker centroid;
  asr             proper nouns for the prompt-free pronunciation probe (normalised stems; the last token matches
                  as a prefix); default: derived from the character's asr_prompt;
  prosody         pyin F0 search range, creak threshold and octave-fold passes for the character's voice;
  provenance      Vox Studio's own adapter for the character (voxcpm adapter_kind "baseline"): its
                  %LOCALAPPDATA%/VoxStudio/voxcpm_training/<folder> and the label the scorecard uses;
  selftest        the clips make_selftest.py builds its CPU self-test fixtures from.
A missing file or key falls back to DEFAULTS (generic starting points, not tuned to anyone) or a derived value; no
impostor voices are assumed. Bao-Dur's file holds the exact values the harness used before it served several
characters, so his scorecards, caches and self-test are unchanged. A file whose "character" differs from the
active slug is refused (a copied file would silently mix two characters).
Legacy overrides for scratch tests: BAODUR_LAB (lab folder) and BAODUR_VOICES (Voices folder; the lab becomes
<voices>/<the character's lab folder name>).
Optional character-config keys read here (characters/<slug>.json; derived when absent): "pronouns"
("he/him/his" when absent; used in scorecard wording) and "vox_training_dir" (Vox Studio training folder; default
the first word of the display name, lower case: "bao-dur", "jolee").
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
CHARACTERS_DIR = CODE_DIR.parents[1] / "characters"   # voice-lab/characters (shared by every lab recipe)
if str(CHARACTERS_DIR) not in sys.path:
    sys.path.insert(0, str(CHARACTERS_DIR))
import character  # noqa: E402

CHAR = character.load()
SLUG = CHAR["slug"]
NAME = CHAR["display_name"]
SPEAKER = CHAR.get("speaker") or SLUG

_pronouns = [p.strip() for p in str(CHAR.get("pronouns") or "he/him/his").split("/")]
HE, HIM, HIS = _pronouns if len(_pronouns) == 3 and all(_pronouns) else ("he", "him", "his")

_voices_env = os.environ.get("BAODUR_VOICES")
VOICES = Path(_voices_env) if _voices_env else Path(CHAR["lab"]).parent
_lab_env = os.environ.get("BAODUR_LAB")
LAB = Path(_lab_env) if _lab_env else (VOICES / Path(CHAR["lab"]).name if _voices_env else Path(CHAR["lab"]))

FILE_NAME = "eval_character.json"
# Generic starting points (the values the harness was first calibrated with); a section of None is free-form.
DEFAULTS: dict = {
    "speaker_screen": {
        "impostor_dirs": [],             # no impostor voices assumed: list them per character
        "impostor_label": "",            # e.g. "other male game voices (Atton, Carth)"; derived when empty
        "impostors_per_dir": 15,
        "inset_impostor_speakers": [],   # delivery-task groups (exclude_other_speakers.json "confirmed" keys)
        "inset_impostor_min_s": 2.0,
        "suspect_identity": 0.35,        # train clips below this leave-one-out identity: another voice
        "min_centroid_s": 2.0,           # shorter clips are scored but not averaged into the centroid
        "likely_other_identity": 0.35,   # aggregate speaker-purity report, clips >= short_clip_s
        "possible_other_identity": 0.0,  # ... shorter clips
        "short_clip_s": 2.5,
    },
    "asr": {
        "probe_names": None,             # None: derived from the character's asr_prompt
        "probe_skip": ["general", "jedi", "sith"],   # prompt words Whisper spells without help (derivation only)
    },
    "prosody": {
        "fmin_hz": 55.0,
        "fmax_hz": 420.0,
        "creak_max_hz": 85.0,
        "creak_rel": 0.65,
        "fold_passes": [[75, 2.0], [25, 3.0]],
    },
    "provenance": {
        "voxcpm_baseline_dir": None,     # None: character config "vox_training_dir", else derived from the name
        "voxcpm_baseline_label": "voxcpm baseline adapter train.jsonl",
    },
    "selftest": None,
}


def _load() -> tuple[Path | None, dict]:
    for path in dict.fromkeys([LAB / "eval" / FILE_NAME, Path(CHAR["lab"]) / "eval" / FILE_NAME]):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise SystemExit(f"{path}: not valid JSON ({exc})")
            if not isinstance(data, dict):
                raise SystemExit(f"{path}: expected a JSON object")
            owner = data.get("character")
            if owner and owner != SLUG:
                raise SystemExit(f"{path} belongs to character {owner!r}, but VOICE_LAB_CHARACTER selects {SLUG!r}")
            return path, data
    return None, {}


def _merge(defaults: dict, raw: dict) -> dict:
    out: dict = {}
    for section, spec in defaults.items():
        given = raw.get(section)
        if spec is None:
            out[section] = given
            continue
        given = given if isinstance(given, dict) else {}
        merged = {}
        for key, default in spec.items():
            value = given.get(key, default)
            if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)   # 2 and 2.0 must give the same cache keys and report text
            merged[key] = value
        merged.update({k: v for k, v in given.items() if k not in merged})
        out[section] = merged
    return out


FILE, _RAW = _load()
SETTINGS = _merge(DEFAULTS, _RAW)
VOX_TRAINING_DIR = (SETTINGS["provenance"]["voxcpm_baseline_dir"] or CHAR.get("vox_training_dir")
                    or NAME.split()[0].lower())


def section(name: str):
    return SETTINGS.get(name)


def voices_path(value) -> Path:
    """A folder named in the eval character file: absolute, or relative to the Voices folder."""
    path = Path(str(value))
    return path if path.is_absolute() else VOICES / path


def describe() -> str:
    return (f"{NAME} ({SLUG}): lab {LAB}; eval content "
            + (str(FILE) if FILE else f"defaults (no {FILE_NAME} in {LAB / 'eval'})"))
