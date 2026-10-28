"""Shared paths and helpers for the VoxCPM2 LoRA v2 recipe, for any voice-lab character.

The character comes from voice-lab\\characters\\character.py: VOICE_LAB_CHARACTER=<slug> (unset = "baodur", so
every Bao-Dur command behaves exactly as before; "jolee" = Jolee Bindo). Its config's `lab` folder (<lab> below:
KotorMods\\Voices\\BaoDurLab, JoleeLab, ...) holds the corpus, the delivery task's files and this recipe's data.

Run everything with the Vox VoxCPM2 runtime (read-only; never pip install into it):
  %LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe

Data written by this recipe lives under <lab>\\voxcpm\\ (cache, manifests, runs, selection, listening,
fixtures, smoke) and rendered takes under <lab>\\takes\\voxcpm\\<run>\\. Nothing here writes to the Vox
profile, the Vox repo, the VoxCPM2 venv or the game install (entry scripts set sys.dont_write_bytecode, so
importing the official trainer or the voxcpm package leaves no __pycache__ in Vox's engine tree).
Held-out TEST clips are never read as audio.

Character-specific content lives with the character's data, never in this code:
  <lab>\\voxcpm\\recipe.json (required; every key optional, see RECIPE_DEFAULTS): text rewrites, the map from
      the character's TLK performance notes to VoxCPM control text, per-label fallback instructions, the neutral
      delivery label, the label of the installed Vox adapter (render.py --baseline) and extra respellings.
  <lab>\\voxcpm\\train_lora_v2.template.yaml (optional): the character's training template; without it
      build_manifest.py uses the code folder's train_lora_v2.yaml.
From the character config: `lab` (every path), `display_name` (messages, the Vox training folder), `address_term`
(a stopword: it names the listener and carries no line identity) and `vox_profile_id` (the installed Vox adapter;
null = none).

User decision 2026-09-30 (<lab>\\DECISIONS.md; it applies to every character): train on the ORIGINAL game
audio and use original clips as references; the ClearerVoice-restored copies are never training or reference
input. The --audio/--ref-audio restored paths therefore refuse to run without --allow-restored-audio.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------------- character
VOICE_LAB = Path(__file__).resolve().parents[2]  # voice-lab\baodur-lab\voxcpm\common.py -> voice-lab
if str(VOICE_LAB / "characters") not in sys.path:
    sys.path.insert(0, str(VOICE_LAB / "characters"))
import character as _character  # noqa: E402

CHARACTER = _character.load()  # VOICE_LAB_CHARACTER, default "baodur"
SLUG = CHARACTER["slug"]
DISPLAY_NAME = CHARACTER["display_name"]
ADDRESS_TERM = CHARACTER.get("address_term") or ""
VOX_PROFILE_ID = CHARACTER.get("vox_profile_id") or None

# --------------------------------------------------------------------------------------------- paths
LAB = Path(CHARACTER["lab"])
CORPUS = LAB / "corpus.jsonl"
RESTORED_DIR = LAB / "restored"
DATA_ROOT = LAB / "voxcpm"
DELIVERY_MAP = LAB / "delivery" / "delivery_map.json"
TAKES_SCRIPT = LAB / "delivery" / "takes_script.json"
# Delivery task: per-label VoxCPM instructions, variant A (descriptive, the map's default) and B (compact,
# positive-only). Its probe writes the per-label winner back into delivery_map.json.
INSTRUCTION_VARIANTS = LAB / "delivery" / "voxcpm_instruction_variants.json"
# Written by the delivery task (baodur-lab\delivery\exclusions.py): other characters' lines inside the
# character's voice files (Bao-Dur's gblbaodur* files also hold HK-47, G0-T0 and Atton lines, all ASR-only text,
# so no TLK filter caught them) plus held-out near-duplicates.
SPEAKER_SCREEN = LAB / "delivery" / "exclude_other_speakers.json"
TAKES_ROOT = LAB / "takes" / "voxcpm"
DECISIONS = LAB / "DECISIONS.md"
RECIPE_FILE = DATA_ROOT / "recipe.json"
CODE_DIR = Path(__file__).resolve().parent
CHARACTER_TEMPLATE = DATA_ROOT / "train_lora_v2.template.yaml"
TEMPLATE_YAML = CHARACTER_TEMPLATE if CHARACTER_TEMPLATE.is_file() else CODE_DIR / "train_lora_v2.yaml"
BUILD_MANIFEST = CODE_DIR / "build_manifest.py"
# Bumped whenever build_manifest.py changes what a manifest row means; train_lora_v2.py refuses manifests whose
# manifest_report.json was written by another schema or by another revision of build_manifest.py.
MANIFEST_SCHEMA = 3
RESTORED_DECISION = (f"user decision 2026-09-30 ({LAB.name}\\DECISIONS.md): training and reference audio stay ORIGINAL "
                     "game audio; restored copies are only an optional post-processing experiment")

LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA") or r"C:\Users\NewAdmin\AppData\Local")
VOX_ENGINE = LOCALAPPDATA / "VoxStudio" / "engines" / "voxcpm2"
VOX_PYTHON = VOX_ENGINE / ".venv" / "Scripts" / "python.exe"
SNAPSHOT_ID = "32279effe8c19989596f05d353d1447f51d9e915"
SNAPSHOT = VOX_ENGINE / "cache" / "models--openbmb--VoxCPM2" / "snapshots" / SNAPSHOT_ID
OLD_SNAPSHOT_ID = "bffb3df5a29440629464e5e839f4d214c8714c3d"
OFFICIAL_SRC = VOX_ENGINE / "source-2.0.3"
OFFICIAL_TRAINER = OFFICIAL_SRC / "scripts" / "train_voxcpm_finetune.py"


def vox_folder_name(name: str) -> str:
    """Vox's training-folder spelling of a voice name: 'Bao-Dur' -> 'bao-dur', 'Jolee Bindo' -> 'jolee-bindo'."""
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


# The character's installed Vox adapter (render.py --baseline, read in place, never written) and the train.jsonl
# Vox trained it on; None when the character config has no vox_profile_id.
BASELINE_LORA = (LOCALAPPDATA / "VoxStudio" / "voxcpm_profiles" / VOX_PROFILE_ID / "lora") if VOX_PROFILE_ID else None
BASELINE_TRAIN_MANIFEST = (LOCALAPPDATA / "VoxStudio" / "voxcpm_training" / vox_folder_name(DISPLAY_NAME)
                           / "train.jsonl") if VOX_PROFILE_ID else None
# Vox's shared KOTOR respelling lexicon (both games); recipe.json "pronunciations" adds or overrides entries.
VOX_PRONUNCIATIONS = Path(
    r"C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\AI-VoiceActing-GameDevelopment"
    r"\third_party\voxcpm_sidecar\pronunciations.json"
)
HF_DATASETS_CACHE = DATA_ROOT / "hf-datasets-cache"

ENCODER_SR = 16_000  # AudioVAE V2 encoder input rate (asserted by the official trainer)
OUTPUT_SR = 48_000  # AudioVAE V2 decoder output rate
AUDIO_VARIANTS = ("original", "restored")

# --------------------------------------------------------------------------------------------- recipe
# Generic (character-neutral) values for every recipe.json key. A character's recipe.json replaces a key as a
# whole, except fallback_instructions, which is merged per label over these defaults.
RECIPE_DEFAULTS: dict = {
    # [regex, replacement] pairs applied by clean_text to training, prompt and target text alike, so text and
    # audio agree (Bao-Dur: the game writes "G0-T0", the actor says "Goto")
    "text_rewrites": [],
    # TLK performance note (stripped, casefolded) -> VoxCPM control text for build_manifest.py; notes not listed
    # (staging, gameplay outcomes such as "Success"/"Failure", ...) are ignored
    "direction_to_instruction": {},
    # render.py: instruction for a delivery label that neither the delivery map nor the variants file covers
    "fallback_instructions": {
        "natural": "Natural and conversational",
        "calm": "Calm and even, relaxed pacing",
        "warm": "Warm and sincere",
        "wry": "Dry wit, a hint of amusement",
        "troubled": "Troubled and uneasy",
        "weary": "Weary and tired, low energy",
        "guarded": "Guarded and wary",
        "urgent": "Urgent, forward momentum",
        "angry": "Angry, controlled intensity",
        "questioning": "Curious, questioning cadence",
        "sad": "Sad and subdued",
        "resolute": "Firm and resolute",
        "reflective": "Reflective, unhurried",
    },
    # render.py: delivery label of the fixed identity clip (--identity-label default) and of script lines whose
    # delivery is missing or not in the delivery map; null = "natural" when the delivery map has that label
    # (Bao-Dur), else the map's first label (Jolee's draft map starts with "conversational")
    "neutral_label": None,
    # render.py --baseline: run label and provenance text of the installed Vox adapter (needs vox_profile_id)
    "baseline_label": "baseline-vox-lora",
    "baseline_provenance": "Vox train.jsonl (read in place)",
    # extra/overriding respellings, same entry format as VOX_PRONUNCIATIONS ({term, spoken, aliases?})
    "pronunciations": [],
}
RECIPE_DOC_KEYS = ("about", "character")  # documentation; "character" must name this character if present


def load_recipe(path: Path = RECIPE_FILE) -> tuple[dict, str]:
    """(<lab>\\voxcpm\\recipe.json, its sha256). Required for every character, so content can never silently
    fall back to generic values because a file went missing ("{}" takes every default)."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"voxcpm recipe {path} not found for character '{SLUG}' ({DISPLAY_NAME}): create it "
                         "(keys: see common.RECIPE_DEFAULTS; '{}' takes every generic default)")
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    unknown = sorted(set(doc) - set(RECIPE_DEFAULTS) - set(RECIPE_DOC_KEYS))
    if unknown:
        raise SystemExit(f"{path}: unknown keys {unknown} (known: {sorted(RECIPE_DEFAULTS)})")
    if doc.get("character") not in (None, SLUG):
        raise SystemExit(f"{path} is the recipe of character '{doc['character']}', not '{SLUG}'")
    for pair in doc.get("text_rewrites") or []:
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
            raise SystemExit(f"{path}: text_rewrites entries must be [regex, replacement], got {pair!r}")
        re.compile(pair[0])
    for key in ("direction_to_instruction", "fallback_instructions"):
        if not isinstance(doc.get(key, {}), dict):
            raise SystemExit(f"{path}: {key} must be an object")
    for entry in doc.get("pronunciations") or []:
        if not (isinstance(entry, dict) and isinstance(entry.get("term"), str) and isinstance(entry.get("spoken"), str)):
            raise SystemExit(f"{path}: pronunciations entries need string 'term' and 'spoken', got {entry!r}")
    for key in ("neutral_label", "baseline_label", "baseline_provenance"):
        if doc.get(key) is not None and not (isinstance(doc[key], str) and doc[key].strip()):
            raise SystemExit(f"{path}: {key} must be a non-empty string or null")
    return doc, hashlib.sha256(raw).hexdigest()


RECIPE, RECIPE_SHA256 = load_recipe()


def recipe_value(key: str):
    if key == "fallback_instructions":
        return {**RECIPE_DEFAULTS[key], **(RECIPE.get(key) or {})}
    return RECIPE.get(key, RECIPE_DEFAULTS[key])


TEXT_REWRITES: tuple[tuple[str, str], ...] = tuple((p, r) for p, r in recipe_value("text_rewrites"))
BASELINE_LABEL = recipe_value("baseline_label")
BASELINE_PROVENANCE = recipe_value("baseline_provenance")


def offline_env() -> None:
    """Keep every Hugging Face component offline and its caches inside the lab data root."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))


# --------------------------------------------------------------------------------------------- corpus
def load_corpus(path: Path = CORPUS) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def corpus_index(rows: list[dict] | None = None) -> dict[str, dict]:
    return {row["id"]: row for row in (rows if rows is not None else load_corpus())}


def test_ids(rows: list[dict] | None = None) -> set[str]:
    return {row["id"] for row in (rows if rows is not None else load_corpus()) if row["split"] == "test"}


def test_id_pattern(ids: set[str]) -> re.Pattern:
    return re.compile(r"(?<![0-9a-z])(" + "|".join(sorted(map(re.escape, ids))) + r")(?![0-9])", re.I)


def load_speaker_screen(path: Path = SPEAKER_SCREEN) -> dict:
    """The delivery task's speaker screen as sets. Missing file -> SystemExit (never train blind on it).

    confirmed: {id: character} - other characters' lines; never trained on, cached or used as reference.
    uncertain: embedding outliers without text evidence (for Bao-Dur mostly his own sub-second barks/efforts).
    suspect:   weak outliers (z < -3 on one encoder), mostly short or emotional lines of the character.
    near_duplicate_of_test: {id: reason} - train clips that (nearly) speak a held-out test line.
    """
    path = Path(path)
    if not path.exists():
        how = "" if SLUG == _character.DEFAULT else f" with VOICE_LAB_CHARACTER={SLUG}"
        raise SystemExit(f"speaker screen {path} not found (run baodur-lab\\delivery\\exclusions.py{how} first)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    confirmed = {entry["id"]: name for name, entries in (doc.get("confirmed") or {}).items() for entry in entries}
    listed = set(doc.get("confirmed_ids") or [])
    if listed and listed != set(confirmed):
        raise SystemExit(f"{path}: confirmed_ids disagree with the per-character lists")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "confirmed": confirmed,
        "uncertain": {entry["id"] if isinstance(entry, dict) else str(entry) for entry in doc.get("uncertain") or []},
        "suspect": set(doc.get("suspect_short_lines") or []),
        "near_duplicate_of_test": dict(doc.get("near_duplicate_of_test") or {}),
    }


def source_audio(row: dict, variant: str) -> Path:
    if variant == "original":
        return Path(row["audio"])
    if variant == "restored":
        return RESTORED_DIR / f"{row['id']}.wav"
    raise ValueError(f"unknown audio variant {variant!r}; use one of {AUDIO_VARIANTS}")


def cache_dir(variant: str) -> Path:
    """Trimmed 16 kHz clips of one AUDIO variant (shared by every dataset built from that variant)."""
    return DATA_ROOT / "cache" / variant


def dataset_name(variant: str, with_val: bool = False) -> str:
    """Manifest/run name: the audio variant, plus '_trainval' for the production retrain on train + val."""
    return f"{variant}_trainval" if with_val else variant


def manifest_dir(dataset: str) -> Path:
    return DATA_ROOT / "manifests" / dataset


def run_dir(dataset: str) -> Path:
    return DATA_ROOT / "runs" / f"lora_v2_{dataset}"


def manifest_report_status(train_manifest: Path) -> tuple[dict | None, list[str]]:
    """(manifest_report.json, problems) for the folder holding `train_manifest`.

    Problems: no report; written by another MANIFEST_SCHEMA or by another revision of build_manifest.py
    (sha256 of the file recorded at build time); no explicit ref pairing; train/val files that no longer
    hash to what that build wrote. An empty list means the manifests are exactly what the current
    build_manifest.py produced."""
    folder = Path(train_manifest).parent
    path = folder / "manifest_report.json"
    if not path.exists():
        return None, [f"{path} is missing: these manifests were not written by build_manifest.py"]
    report = json.loads(path.read_text(encoding="utf-8"))
    builder = report.get("builder") or {}
    problems = []
    if builder.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"manifest schema {builder.get('schema')!r} != current {MANIFEST_SCHEMA}")
    if builder.get("build_manifest_sha256") != sha256_file(BUILD_MANIFEST):
        problems.append("built by another revision of build_manifest.py (sha256 differs)")
    if builder.get("character") not in (None, SLUG):
        problems.append(f"built for character '{builder.get('character')}', but VOICE_LAB_CHARACTER selects '{SLUG}'")
    if "recipe_sha256" in builder and builder.get("recipe_sha256") != RECIPE_SHA256:
        problems.append(f"built with another version of the character recipe {RECIPE_FILE} (sha256 differs)")
    if "ref_pairing" not in (report.get("args") or {}):
        problems.append("the report records no ref_pairing (pre-2026-09-30-17:14 build)")
    hashes = report.get("manifest_sha256") or {}
    if "train.jsonl" not in hashes:
        problems.append("the report records no train.jsonl sha256")
    for name, digest in hashes.items():
        target = folder / name
        if not target.exists():
            problems.append(f"{target} is missing")
        elif sha256_file(target) != digest:
            problems.append(f"{target} changed after build_manifest.py wrote it")
    return report, problems


# --------------------------------------------------------------------------------------------- text
def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.casefold().replace("-", " ")).split()


def clean_text(text: str) -> str:
    """Whitespace/quote cleanup plus TEXT_REWRITES. Punctuation (incl. '...' and ' - ') is kept:
    it is how the TLK encodes the actor's pauses and VoxCPM reads it as phrasing."""
    text = (text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
            .replace("\u201d", '"').replace("\u2014", " - ").replace("\u2013", " - ").replace("\u2026", "..."))
    for pattern, replacement in TEXT_REWRITES:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str, min_words: int = 4) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\s+-\s+|\.\.\.", text)
    out = []
    for part in parts:
        normalized = " ".join(words(part))
        if len(normalized.split()) >= min_words:
            out.append(normalized)
    return out


# Words that carry no line identity: function words, contractions, fillers and the character's form of address
# (character config address_term: Bao-Dur's ubiquitous "General", Jolee's "kid").
STOPWORDS = frozenset("""
a an the and or but if then so to of in on at by for with from as into onto about over under up down out off
is are was were be been being am do does did done have has had having can could will would should shall may
might must i you he she it we they me him her us them my your his its our their mine yours this that these those
there here what which who whom whose when where why how not no yes just all any some more most very too also only
now than again still even well oh uh um yeah okay ok right sure much many
i'm you're it's don't i'll i've i'd that's can't won't let's he's she's we're they're isn't aren't wasn't weren't
didn't doesn't haven't hasn't hadn't couldn't wouldn't shouldn't there's what's you'll you've you'd we'll we've
able get got go going gone know think see like want make
""".split()) | frozenset(words(ADDRESS_TERM))


def content_phrases(text: str, n: int = 4, min_content: int = 3) -> set[str]:
    """Runs of `n` words that hold at least `min_content` non-stopwords: a line's distinctive wording
    ("a power cell emitter"), as opposed to stock phrases ("should be able to do")."""
    tokens = words(text)
    out = set()
    for i in range(len(tokens) - n + 1):
        gram = tokens[i:i + n]
        if sum(1 for t in gram if t not in STOPWORDS) >= min_content:
            out.add(" ".join(gram))
    return out


def overlap_reason(candidate: str, protected: str, jaccard: float = 0.9) -> str | None:
    """Why `candidate` (partly) speaks the line `protected`, or None. Rules, in order:
    a full sentence (>= 4 words) of `protected` inside `candidate` (word-bounded); the same line (word
    Jaccard >= `jaccard`); a shared distinctive phrase (content_phrases), which catches partial repeats such
    as train gblbaodur128 "We need a power cell, emitter matrix, lens..." vs held-out 276 "...still missing a
    power cell, emitter matrix and lens."."""
    padded = f" {' '.join(words(candidate))} "
    for sentence in sentences(protected):
        if sentence and f" {sentence} " in padded:
            return f"sentence: {sentence}"
    a, b = set(words(candidate)), set(words(protected))
    if a and b and len(a & b) / len(a | b) >= jaccard:
        return f"same line (word Jaccard {len(a & b) / len(a | b):.2f})"
    shared = content_phrases(candidate) & content_phrases(protected)
    if shared:
        return f"phrase: {sorted(shared)[0]}"
    return None


def overlaps_text(candidate: str, protected: str, jaccard: float = 0.9) -> bool:
    """True when `candidate` speaks a sentence of `protected`, is nearly the same line, or repeats one of its
    distinctive phrases (see overlap_reason)."""
    return overlap_reason(candidate, protected, jaccard) is not None


# --------------------------------------------------------------------------------------------- audio
def read_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32), int(rate)


def resample(audio: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    if rate_in == rate_out:
        return audio.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(audio.astype(np.float32), rate_in, rate_out, quality="VHQ").astype(np.float32)
    except ImportError:  # pragma: no cover - soxr ships with the Vox runtime
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(rate_in, rate_out)
        return resample_poly(audio, rate_out // g, rate_in // g).astype(np.float32)


def frame_db(audio: np.ndarray, rate: int, frame_ms: float = 10.0) -> tuple[np.ndarray, int]:
    hop = max(1, int(rate * frame_ms / 1000.0))
    count = len(audio) // hop
    if count == 0:
        return np.zeros(0), hop
    frames = audio[: count * hop].astype(np.float64).reshape(count, hop)
    return 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-10), hop


def trim_silence(
    audio: np.ndarray,
    rate: int,
    lead_pad: float = 0.10,
    tail_pad: float = 0.20,
    rel_db: float = 45.0,
    fade_ms: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Cut leading/trailing silence, keeping at most `lead_pad`/`tail_pad` seconds of it.

    Only cuts, never pads. The activity threshold sits `rel_db` below the loudest 10 ms frame, raised
    to 6 dB over the clip's noise floor when the floor is higher (noisy clip) but never above
    peak-30 dB, so breaths and soft fricative onsets (typically 30-45 dB under the peak in these game
    clips) stay inside the kept region. A short fade is applied only where a cut happened.
    """
    db, hop = frame_db(audio, rate)
    info = {"lead_in": None, "tail_in": None, "lead_out": None, "tail_out": None, "trimmed": False}
    if db.size < 3:
        return audio.copy(), info
    peak = float(db.max())
    floor = float(np.percentile(db, 5))
    threshold = min(max(peak - rel_db, floor + 6.0), peak - 30.0)
    active = np.flatnonzero(db >= threshold)
    if active.size == 0 or (active[-1] - active[0] + 1) * hop < 0.2 * rate:
        return audio.copy(), info
    speech_start = int(active[0]) * hop
    speech_end = min(len(audio), (int(active[-1]) + 1) * hop)
    start = max(0, speech_start - int(round(lead_pad * rate)))
    end = min(len(audio), speech_end + int(round(tail_pad * rate)))
    out = audio[start:end].astype(np.float32, copy=True)
    fade = min(int(rate * fade_ms / 1000.0), len(out) // 4)
    if start > 0 and fade > 0:
        out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    if end < len(audio) and fade > 0:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    info.update(
        lead_in=round(speech_start / rate, 3),
        tail_in=round((len(audio) - speech_end) / rate, 3),
        lead_out=round((speech_start - start) / rate, 3),
        tail_out=round((end - speech_end) / rate, 3),
        trimmed=bool(start > 0 or end < len(audio)),
        threshold_db=round(threshold, 1),
        peak_db=round(peak, 1),
        floor_db=round(floor, 1),
    )
    return out, info


def write_wav_pcm16(path: Path, audio: np.ndarray, rate: int) -> dict:
    """Atomically write mono PCM16; returns level stats. Never clips silently."""
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scaled = False
    if peak > 0.999:
        audio = audio * (0.999 / peak)
        scaled = True
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp.wav", dir=str(path.parent))
    os.close(fd)
    try:
        sf.write(tmp, audio, rate, subtype="PCM_16")
        replace_file(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    return {
        "peak_dbfs": round(20 * np.log10(max(min(peak, 0.999), 1e-9)), 2),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 2),
        "peak_rescaled": scaled,
        "duration_s": round(len(audio) / rate, 3),
    }


# --------------------------------------------------------------------------------------------- misc
def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        remaining = limit
        while True:
            chunk = stream.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return digest.hexdigest()


def replace_file(src, dst, attempts: int = 10, first_wait: float = 0.05) -> None:
    """os.replace with retries: on Windows a virus scanner, the indexer or a reader holding `dst` open
    makes the rename fail with a sharing violation (PermissionError) for a moment. ~5 s in total."""
    wait = first_wait
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)
            wait = min(wait * 2, 1.0)


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        replace_file(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        replace_file(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_seed(*parts) -> int:
    digest = hashlib.blake2s("|".join(map(str, parts)).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


# --------------------------------------------------------------------------------------------- memory
# Parameter counts read from the 32279eff model.safetensors header (2,290,004,544 params, stored bf16).
PARAMS_TOTAL = 2_290_004_544
PARAMS_GRAD_PATH = 2_080_000_000  # base_lm + residual_lm + LocDiT + projections: on the backward path
LORA_PARAMS_R32 = 18_087_936  # q/k/v/o in base_lm(28) + residual_lm(8) + LocDiT(12) at r=32
VAE_GB = 0.377
KV_CACHE_GB = 0.302  # static bf16 KV caches the model allocates at init (max_length 8192)
CONTEXT_GB = 0.6  # CUDA context + cuBLAS workspace on Windows (not governed by the allocator cap)
# Saved activations per packed sequence position under bf16 autocast (LM 28x~93 KB + residual 8x~93 KB
# + LocDiT 11 tokens x 12 layers x ~53 KB); with checkpointing only layer inputs plus one layer's
# recompute stay resident.
MB_PER_POSITION = {"none": 10.3, "dit": 4.2, "all": 1.0}


def load_tokenizer():
    """The exact tokenizer the official trainer builds (LlamaTokenizerFast + char-split wrapper)."""
    offline_env()
    from transformers import LlamaTokenizerFast
    from voxcpm.model.utils import mask_multichar_chinese_tokens

    return mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(str(SNAPSHOT)))


def sequence_lengths(rows: list[dict], tokenizer, vae_fps: float = 25.0, patch: int = 4) -> list[int]:
    """Packed length per row, same arithmetic as voxcpm.training.data.compute_sample_lengths."""
    import math

    out = []
    for row in rows:
        text = len(tokenizer(row["text"]))
        audio = math.ceil(math.ceil(float(row["duration"]) * vae_fps) / patch)
        ref = row.get("ref_duration")
        ref_len = math.ceil(math.ceil(float(ref) * vae_fps) / patch) if ref else 0
        out.append(text + audio + ref_len + (4 if ref_len else 2))
    return out


def estimate_train_memory(cfg: dict, lengths: list[int], frozen: str = "bf16", ckpt: str = "all") -> dict:
    """Rough (+/-20%) device-memory estimate for one LoRA training process."""
    batch = int(cfg["batch_size"])
    rank = int((cfg.get("lora") or {}).get("r", 32))
    worst = batch * max(lengths)
    typical = batch * int(np.median(lengths))
    weights = PARAMS_TOTAL * (2 if frozen == "bf16" else 4) / 1e9
    # fp32 frozen weights: autocast re-casts them to bf16 per op (no cache for requires_grad=False)
    # and autograd keeps those copies for the backward pass.
    casts = 0.0 if frozen == "bf16" else {"none": PARAMS_GRAD_PATH * 2 / 1e9, "dit": 3.72, "all": 0.25}[ckpt]
    lora = LORA_PARAMS_R32 * rank / 32 * 16 / 1e9  # fp32 param + grad + 2 AdamW moments
    act_worst = MB_PER_POSITION[ckpt] * worst / 1e3
    act_typical = MB_PER_POSITION[ckpt] * typical / 1e3
    static = weights + casts + VAE_GB + lora + KV_CACHE_GB
    worst_total = static + max(act_worst, 0.5) * 1.1 + CONTEXT_GB
    return {
        "frozen_dtype": frozen,
        "grad_ckpt": ckpt,
        "batch_size": batch,
        "max_positions_per_sample": max(lengths),
        "median_positions_per_sample": int(np.median(lengths)),
        "weights_gb": round(weights, 2),
        "autocast_weight_copies_gb": round(casts, 2),
        "lora_and_adam_gb": round(lora, 2),
        "activations_worst_gb": round(act_worst, 2),
        "total_typical_gb": round(static + max(act_typical, 0.5) * 1.1 + CONTEXT_GB, 1),
        "total_worst_gb": round(worst_total, 1),
    }


RENDER_NEED_GB = 6.8  # bf16 weights 4.58 + VAE 0.38 + KV 0.30 + decode/DiT transients ~0.9 + context 0.6


def gpu_headroom_gb(need_gb: float) -> float:
    """Margin the allocator cap must keep above the (+/-20%) point estimate before a GPU job may start."""
    return round(max(1.0, 0.15 * need_gb), 2)


def gpu_budget(max_vram_gb: float, reserve_gb: float, need_gb: float, context_gb: float = CONTEXT_GB):
    """Decide the allocator cap for a shared GPU. Returns (cap_gb, info) or raises SystemExit.

    `need_gb` is this job's estimated device total (allocator + CUDA context). The cap leaves
    `reserve_gb` free for Substance Painter at all times and never exceeds `max_vram_gb` in total.
    The job starts only when the cap is at least need + gpu_headroom_gb(need) (the estimate is +/-20%;
    starting at cap == estimate would make a mid-run OOM likely). Must run before the job allocates
    anything on the GPU.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this step is a GPU step (use --dry-run for CPU checks).")
    torch.cuda.init()
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    # `free` already excludes this process's freshly created context.
    cap_total = min(max_vram_gb, free_gb + context_gb - reserve_gb - 0.25)
    headroom = gpu_headroom_gb(need_gb)
    info = {
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(total_gb, 2),
        "free_at_start_gb": round(free_gb, 2),
        "reserve_for_others_gb": reserve_gb,
        "max_vram_gb": max_vram_gb,
        "estimated_need_gb": need_gb,
        "required_headroom_gb": headroom,
        "cap_total_gb": round(cap_total, 2),
    }
    if cap_total < need_gb + headroom:
        raise SystemExit(
            "Not enough free VRAM to run without squeezing the shared GPU: "
            f"free {free_gb:.1f} GB, must leave {reserve_gb:.1f} GB free, estimated need {need_gb:.1f} GB "
            f"+ {headroom:.1f} GB headroom (cap would be {cap_total:.1f} GB). Wait for other GPU jobs to finish "
            f"and retry; never raise --max-vram-gb. {json.dumps(info)}"
        )
    alloc_cap = max(1.0, cap_total - context_gb)
    torch.cuda.set_per_process_memory_fraction(min(1.0, alloc_cap / total_gb), 0)
    info["allocator_cap_gb"] = round(alloc_cap, 2)
    return cap_total, info


# --------------------------------------------------------------------------------------------- host memory
# Peak host COMMIT (Windows commit charge, what runs out on this machine) of one process while it loads
# VoxCPM2 on the CPU before moving it to the GPU. Parts: model.safetensors 4.58 GB (bf16, read fully into
# RAM by safetensors.load_file), audiovae.pth 0.38 GB (fp32, torch.load), Python + torch + voxcpm imported
# 1.9 GB (measured 2026-09-30 with process_commit(), CPU only) + CUDA context/driver ~0.5-1 GB.
#  - official from_local(training=True): fp32 model 9.16 + LoRA 0.07 + VAE 0.38 + state dicts 4.96
#    + overhead -> ~17.5 GB
#  - train_lora_v2.py --frozen-dtype bf16 (builds the model under default dtype bf16): bf16 model 4.58 +
#    VAE 0.38 + state dicts 4.96 + overhead -> ~13 GB
#  - inference from_local (render.py, Vox): fp32 build 9.16 (+ VAE, its state dict) or, after the bf16 cast,
#    bf16 model 4.58 + state dicts 4.96 -> ~13 GB either way
# After the model moves to the GPU the process drops to ~3 GB of commit. The real peaks are recorded
# (train_summary.json host_commit, render manifest gpu.host) so these numbers can be corrected.
HOST_LOAD_GB = {"train_bf16": 13.0, "train_fp32": 17.5, "render": 13.0}
HOST_MARGIN_GB = 2.0  # commit that must still be free for other apps while we sit at our peak


def host_memory() -> dict | None:
    """System-wide memory state (Windows GlobalMemoryStatusEx). commit_free_gb = ullAvailPageFile, the
    commit the system can still grant (WMI calls it FreeVirtualMemory). None when not on Windows."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {
        "commit_free_gb": round(status.ullAvailPageFile / 1e9, 2),
        "commit_limit_gb": round(status.ullTotalPageFile / 1e9, 2),
        "phys_free_gb": round(status.ullAvailPhys / 1e9, 2),
        "phys_total_gb": round(status.ullTotalPhys / 1e9, 2),
    }


def process_commit() -> dict | None:
    """This process's commit now and its peak so far (GetProcessMemoryInfo PrivateUsage/PeakPagefileUsage)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return {"commit_gb": round(counters.PrivateUsage / 1e9, 2),
            "peak_commit_gb": round(counters.PeakPagefileUsage / 1e9, 2),
            "peak_working_set_gb": round(counters.PeakWorkingSetSize / 1e9, 2)}


def host_ram_budget(need_gb: float, margin_gb: float, what: str) -> dict:
    """Refuse to start a large CPU-side model load unless `need_gb` of commit is free on top of
    `margin_gb` for everything else (the machine has run out of commit before and crashed other apps'
    native loads). Like gpu_budget it is checked once, right before the load."""
    info = host_memory()
    if info is None:
        return {"checked": False, "reason": "not Windows; commit not checked"}
    own = process_commit() or {}
    # what this process already holds counts toward its own peak, so it is not demanded twice
    info.update(checked=True, need_gb=need_gb, margin_gb=margin_gb, process_commit_before_gb=own.get("commit_gb"))
    required = need_gb - float(own.get("commit_gb") or 0.0) + margin_gb
    info["required_free_gb"] = round(required, 2)
    if info["commit_free_gb"] < required:
        raise SystemExit(
            f"Not enough free host memory (Windows commit) for the {what}: {info['commit_free_gb']:.1f} GB free of "
            f"{info['commit_limit_gb']:.1f} GB, need ~{required:.1f} GB (peak ~{need_gb:.1f} GB while loading on the "
            f"CPU, minus {float(own.get('commit_gb') or 0):.1f} GB already held, plus {margin_gb:.1f} GB kept free "
            f"for other apps). Wait for other large loads to finish and retry. {json.dumps(info)}"
        )
    return info
