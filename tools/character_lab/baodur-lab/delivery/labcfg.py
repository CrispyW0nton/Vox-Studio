"""Which character the delivery recipe works on, and that character's paths, names and content (stdlib only).

VOICE_LAB_CHARACTER=<slug> picks the character (default "baodur", so every Bao-Dur command and output is unchanged).
Paths and names come from voice-lab/characters/<slug>.json (character.load()). Everything written for one character
(label set, curated evidence, VoxCPM2 wording, new lines, lexicon, speaker-screen thresholds and attributions, pitch
range, bible prose) lives in that character's lab, <lab>/delivery/content/*.json, never in this code:
  labels.json   build_delivery_map.py, validate.py   label set, curated evidence, TLK rules, VoxCPM2 text, craft notes
  lines.json    make_takes_script.py, probe_voxcpm.py new lines, held-out deliveries, probe lines, new-line limits
  lexicon.json  analyze.py, build_delivery_map.py    syllable exceptions, G2P-risky tokens, VoxCPM2 text rewrites
  screen.json   speaker_check.py, exclusions.py      screen thresholds, other-speaker attributions, near-duplicates
  voice.json    analyze.py                           pitch-tracking range
  bible.json    make_bible.py                        the bible's character-specific prose (templates)
BaoDurLab/delivery/content/ holds Bao-Dur's (moved verbatim from the code): copy it as the template for a new
character. Keys that start with "_" are documentation; every reader skips them.
"""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "characters"))
import character  # noqa: E402

CHARACTER = character.load()
SLUG: str = CHARACTER["slug"]
IS_DEFAULT = SLUG == character.DEFAULT  # Bao-Dur: commands need no VOICE_LAB_CHARACTER
NAME: str = CHARACTER["display_name"]
LAB = Path(CHARACTER["lab"])
DELIVERY = LAB / "delivery"
CONTENT = DELIVERY / "content"
NUMBA_CACHE = DELIVERY / ".cache" / "numba"  # numba's JIT cache for librosa (never inside a read-only venv)
ADDRESS: str = CHARACTER.get("address_term") or ""
# The Vox adapter Vox trained for this character before the lab existed (read only; absent for most characters).
VOX_TRAINING = (Path(os.environ.get("LOCALAPPDATA", "")) / "VoxStudio" / "voxcpm_training"
                / (CHARACTER.get("vox_training_slug") or re.sub(r"\s+", "-", NAME.strip().casefold())))
CONTENT_FILES = ("labels", "lines", "lexicon", "screen", "voice", "bible")


@lru_cache(maxsize=None)
def content(name: str) -> dict:
    """<lab>/delivery/content/<name>.json (required: a missing file stops with a pointer to the template)."""
    if name not in CONTENT_FILES:
        raise ValueError(f"unknown content file {name!r}; one of {CONTENT_FILES}")
    path = CONTENT / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"{NAME} ({SLUG}): {path} is missing. Character content lives in the character's lab; copy "
                         f"BaoDurLab/delivery/content/{name}.json as the template and rewrite it for {NAME}.")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------- address term
def address_word() -> str:
    """Regex for the address term itself: a capitalised title ('General') matches exactly; a lowercase word ('kid')
    also matches with a capital first letter (sentence start). Matches nothing when the config has no term."""
    if not ADDRESS:
        return r"(?!x)x"
    first, rest = ADDRESS[0], re.escape(ADDRESS[1:])
    head = f"[{first.upper()}{first}]" if first.islower() else re.escape(first)
    return head + rest


def address_regex() -> str:
    return rf"\b{address_word()}\b"


def address_key() -> str:
    """Name part for features/facts about the address term ('general' -> text_addresses_general, general_lines)."""
    return re.sub(r"[^a-z0-9]+", "_", ADDRESS.casefold()).strip("_") or "address_term"


# ------------------------------------------------------------------------------------------- clip ids
@lru_cache(maxsize=None)
def corpus_ids() -> tuple[str, ...]:
    path = LAB / "corpus.jsonl"
    if not path.is_file():
        return ()
    return tuple(json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


@lru_cache(maxsize=None)
def id_prefix() -> str:
    """The prefix every corpus id shares, without trailing digits ('gblbaodur' for gblbaodur001..351); '' when the
    ids share fewer than 4 characters (KOTOR 1 ids such as nm35aajole09428_ keep their full form)."""
    ids = corpus_ids()
    prefix = os.path.commonprefix(list(ids)).rstrip("0123456789") if ids else ""
    return prefix if len(prefix) >= 4 else ""


def short_id(clip_id) -> str:
    """Clip id without the shared corpus prefix ('gblbaodur248' -> '248'); other ids and None unchanged (str)."""
    prefix = id_prefix()
    if clip_id and prefix and str(clip_id).startswith(prefix):
        return str(clip_id)[len(prefix):]
    return str(clip_id)


def id_glob() -> str:
    """How the bible names his conversation files ('gblbaodur*'), or '' when the ids share no prefix."""
    prefix = id_prefix()
    return f"{prefix}*" if prefix else ""
