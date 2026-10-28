"""Shared helpers for the take evaluation harness of a voice-lab character.

The character is VOICE_LAB_CHARACTER=<slug> (default "baodur", so Bao-Dur's commands behave exactly as before);
evalchar.py resolves it through voice-lab/characters/character.py. <lab> below is that character's lab folder
(BaoDurLab, JoleeLab, ...); its character-specific evaluation content (impostor voices, other-speaker screen
thresholds, name-probe stems, pitch range, self-test clips) is <lab>/eval/eval_character.json (see evalchar.py).

Every metric script imports this module, and they run in different venvs (GPT-SoVITS py3.11,
seed-vc py3.10, VoxCPM2 py3.12; numpy 1.26 and 2.x). So this file only needs the standard library,
numpy and soundfile at import time; scipy is imported lazily for resampling.

Conventions
- Real clips come from <lab>/corpus.jsonl. "original" = the decoded game clip (row["audio"]),
  "restored" = <lab>/restored/<id>.wav (optional). Held-out TEST clips are only ever used as ground truth.
- Real reference sets: train = split "train" and tts_usable (speech, 1-20 s); val/test = speech clips
  of those splits. The speaker centroid and every "real distribution" come from train only.
- A character's clip set can hold other characters' lines (Bao-Dur's gblbaodur set holds HK-47/G0-T0/Atton).
  Clips the delivery task confirmed as other voices (<lab>/delivery/exclude_other_speakers.json,
  "confirmed_ids", optional) and clips the speaker models themselves score as another voice are left out of
  the centroid and of every real distribution. The other-speaker screen is anchored on VAL clips only, so
  test clips never influence which train clips are recommended for dropping.
- Some train clips repeat a held-out test line (same line recorded twice, e.g. Bao-Dur's gblbaodur218 ~ test
  gblbaodur200). near_duplicates_of_test() finds them so the scorecard can flag held-out lines whose
  paired scores may be optimistic, and a take prompted with (or a model trained on) such a clip on that
  line counts as a leak.
- Paired scoring: a take on a held-out line (take["heldout_id"], a test clip) or on a train/val diagnostic
  line (take["paired_id"], for the memorisation check) is also compared with that real recording.
- Results for real clips are cached by the SHA-256 of the audio file plus the metric settings, under
  <lab>/eval/_cache/<metric>/<settings-hash>.json, so re-runs only process new takes.
- Per-take failures are classified (guarded()): resource/infrastructure errors (memory, CUDA, a file
  still being written) are retried once and, if they persist, recorded as "unscored" so the take is
  re-run instead of being failed; every metric JSON counts them in "take_errors" and its script exits 3.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:  # evalchar sits next to this file
    sys.path.append(str(CODE_DIR))
import evalchar  # noqa: E402  (VOICE_LAB_CHARACTER selects the character; default baodur)

CHAR = evalchar.CHAR
SLUG, NAME, SPEAKER = evalchar.SLUG, evalchar.NAME, evalchar.SPEAKER
HE, HIM, HIS = evalchar.HE, evalchar.HIM, evalchar.HIS   # scorecard wording ("his own clips")
EVAL_CHARACTER = evalchar.SETTINGS            # <lab>/eval/eval_character.json merged with evalchar.DEFAULTS
EVAL_CHARACTER_FILE = evalchar.FILE           # None when the character has no such file (generic defaults)
VOICES = evalchar.VOICES                      # folder holding the labs and RvcDatasets
LAB = evalchar.LAB
VOICE_LAB = Path(os.environ.get("BAODUR_VOICE_LAB", r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab"))
LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local"))
CORPUS = LAB / "corpus.jsonl"
RESTORED_DIR = LAB / "restored"
RESTORE_REPORT = LAB / "restore_report.json"
EVAL_ROOT = LAB / "eval"
CACHE_ROOT = EVAL_ROOT / "_cache"
DEFAULT_SCRIPT = LAB / "delivery" / "takes_script.json"
DEFAULT_DELIVERY_MAP = LAB / "delivery" / "delivery_map.json"
# Written by the delivery task (optional): confirmed other-speaker clips and near-duplicates of test clips.
EXCLUSIONS = Path(os.environ.get("BAODUR_EXCLUSIONS", str(LAB / "delivery" / "exclude_other_speakers.json")))

_SCREEN = EVAL_CHARACTER["speaker_screen"]
# Other game voices (other characters' decoded clips): an "impostor" floor for the similarity scores, from the
# character's eval file (Bao-Dur: Atton and Carth, other male voices from the same decode pipeline).
IMPOSTOR_DIRS = [evalchar.voices_path(p) for p in _SCREEN["impostor_dirs"]]
IMPOSTORS_PER_DIR = int(_SCREEN["impostors_per_dir"])
# In-set impostors: delivery-task groups of another voice inside the character's own clip set (Bao-Dur: the
# confirmed Atton lines in his gbl files, same files and mastering, restored copies exist), so the impostor floor
# is channel-matched for both audio variants. Droid voices (HK-47/G0-T0) are processed, too unlike any clone to
# anchor "another voice", so the eval file lists the groups to use.
INSET_IMPOSTOR_SPEAKERS = tuple(_SCREEN["inset_impostor_speakers"])
INSET_IMPOSTOR_MIN_S = _SCREEN["inset_impostor_min_s"]
# Train clips whose leave-one-out identity (0 = impostor mean, 1 = real VAL mean) falls below this are
# treated as another speaker (Bao-Dur's gblbaodur set contains HK-47/G0-T0 lines) and left out of the centroid.
SUSPECT_IDENTITY = _SCREEN["suspect_identity"]
# Clips shorter than this give noisy embeddings (Bao-Dur's kept clips: leave-one-out identity median 0.61 under
# 2 s vs 0.96-1.06 at 3-8 s), so they are scored but not averaged into the centroid.
MIN_CENTROID_S = _SCREEN["min_centroid_s"]


def _speaker_of_dir(folder: Path) -> str:
    """'AttonDecoded' -> 'atton', 'CarthExact' -> 'carth' (impostor speaker label)."""
    name = folder.name.replace("Decoded", "")
    return (name[:-len("Exact")] if name.endswith("Exact") and len(name) > len("Exact") else name).lower()


# How the scorecard names the impostor floor (identity 0.00).
IMPOSTOR_LABEL = _SCREEN["impostor_label"] or (
    "other game voices (" + ", ".join(_speaker_of_dir(d).capitalize() for d in IMPOSTOR_DIRS) + ")"
    if IMPOSTOR_DIRS else "no impostor voices configured")

VARIANTS = ("original", "restored")

# Same proper-noun prompt the corpus ASR used (the character config's asr_prompt), so real-clip and take WERs
# are comparable.
ASR_NAMES_PROMPT = CHAR["asr_prompt"]
# PROBE_NAMES (defined after norm_words below): proper nouns whose pronunciation is checked with a second,
# prompt-free ASR pass (the prompt steers Whisper to the intended spelling and can hide a mispronunciation).


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    text = f"[{stamp}] {message}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


# ----------------------------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------------------------

def clean(obj):
    """Make an object JSON-safe: numpy scalars/arrays to Python, NaN/inf to None, Paths to str."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path | str, obj) -> None:
    """Atomic write (unique temp file + replace, retried: Windows refuses to replace a file being read)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.3)


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Cache:
    """Per-metric result cache keyed by audio SHA-256; one JSON file per settings signature."""

    def __init__(self, metric: str, settings: dict):
        signature = hashlib.sha1(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        self.path = CACHE_ROOT / metric / f"{signature}.json"
        self.settings = settings
        self.items: dict = {}
        self._dirty = 0
        if self.path.exists():
            try:
                self.items = read_json(self.path).get("items", {})
            except (OSError, ValueError):
                log(f"cache {self.path} unreadable; starting empty")
                self.items = {}

    def get(self, sha: str):
        return self.items.get(sha)

    def put(self, sha: str, value) -> None:
        self.items[sha] = clean(value)
        self._dirty += 1
        if self._dirty >= 25:
            self.save()

    def save(self) -> None:
        if self._dirty:
            if self.path.exists():  # merge entries another run may have added meanwhile
                try:
                    on_disk = read_json(self.path).get("items", {})
                    on_disk.update(self.items)
                    self.items = on_disk
                except (OSError, ValueError):
                    pass
            write_json(self.path, {"settings": self.settings, "items": self.items})
            self._dirty = 0


# ----------------------------------------------------------------------------------------------
# Failure handling shared by the metric scripts
# ----------------------------------------------------------------------------------------------

_RESOURCE_RE = re.compile(
    r"out of memory|mkl_malloc|bad_alloc|unable to allocate|cannot allocate|not enough memory|paging file|"
    r"winerror 1455|winerror 8\b|winerror 32\b|errno 12\b|resource exhausted|insufficient system resources|"
    r"\bcuda\b|cublas|cudnn|cufft|curand|nvrtc|device-side|ctranslate2.*(device|cuda)", re.I)


def is_resource_error(exc: BaseException) -> bool:
    """Memory, CUDA/driver and file-in-use failures: the machine's state, not the take's content."""
    if isinstance(exc, (MemoryError, PermissionError)):
        return True
    return bool(_RESOURCE_RE.search(f"{type(exc).__name__}: {exc}"))


def guarded(fn, *args, retries: int = 1, pause_s: float = 2.0):
    """Run fn(*args). A resource error is retried (after gc and a pause); one that persists is returned as an
    'unscored' entry (re-run the step), any other error as a 'take' entry (the take itself is broken).
    Returns (value, None) or (None, {"error", "kind", "unscored"})."""
    for attempt in range(retries + 1):
        try:
            return fn(*args), None
        except Exception as exc:  # noqa: BLE001 - classified below
            resource = is_resource_error(exc)
            if resource and attempt < retries:
                gc.collect()
                time.sleep(pause_s)
                continue
            return None, {"error": f"{type(exc).__name__}: {exc}"[:500], "kind": "resource" if resource else "take",
                          "unscored": resource}
    return None, {"error": "unreachable", "kind": "take", "unscored": False}


def takes_signature(takes: list[dict]) -> str:
    """Identifies a take set (keys + audio hashes): metric outputs computed for another set are detected."""
    body = "\n".join(sorted(f"{t['key']}|{t.get('wav_sha256') or ''}" for t in takes))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def finish_metric(run_dir: Path | str, metric: str, out: dict, run: dict) -> int:
    """Write <run>/metrics/<metric>.json with take_errors + takes_signature; exit code 3 if takes are unscored."""
    takes = out.get("takes") or {}
    unscored = sorted(k for k, v in takes.items() if isinstance(v, dict) and v.get("unscored"))
    out["take_errors"] = len(unscored)
    out["unscored_takes"] = unscored
    out["takes_signature"] = takes_signature(run["takes"])
    write_json(metric_out_path(run_dir, metric), out)
    if unscored:
        log(f"!! {metric}: {len(unscored)} take(s) unscored after a retry (resource errors); re-run this step "
            f"(run_eval.py --metrics {metric}); first: {takes[unscored[0]].get('error')}")
        return 3
    return 0


# ----------------------------------------------------------------------------------------------
# Corpus, real clip sets, run files
# ----------------------------------------------------------------------------------------------

def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_index() -> dict[str, dict]:
    return {row["id"]: row for row in load_corpus()}


def test_ids() -> set[str]:
    return {row["id"] for row in load_corpus() if row["split"] == "test"}


def real_audio_path(row: dict, variant: str) -> Path | None:
    if variant == "original":
        path = Path(row["audio"])
    elif variant == "restored":
        path = RESTORED_DIR / f"{row['id']}.wav"
    else:
        raise ValueError(f"unknown audio variant {variant!r}")
    return path if path.exists() else None


def real_sets(variant: str) -> dict[str, list[dict]]:
    """Real reference clips per role; each item: id, split, path, text, duration, directions, delivery_tags."""
    sets: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "missing": []}
    for row in load_corpus():
        if row["split"] == "train":
            wanted = bool(row.get("tts_usable"))
        else:
            wanted = row["kind"] == "speech"
        if not wanted:
            continue
        path = real_audio_path(row, variant)
        if path is None:
            sets["missing"].append({"id": row["id"], "split": row["split"]})
            continue
        sets[row["split"]].append({
            "id": row["id"], "split": row["split"], "path": str(path), "text": row["text"],
            "text_source": row["text_source"], "duration": row["duration"],
            "directions": row.get("directions", []), "delivery_tags": row.get("delivery_tags", []),
        })
    return sets


def available_variants() -> list[str]:
    """Audio variants whose real test and val clips are all present (restored copies may still be generating)."""
    found = []
    for variant in VARIANTS:
        sets = real_sets(variant)
        if sets["test"] and sets["train"] and not [m for m in sets["missing"] if m["split"] in ("test", "val")]:
            found.append(variant)
    return found


def impostor_clips(per_dir: int = IMPOSTORS_PER_DIR, dirs: list[Path] | None = None) -> list[dict]:
    """Deterministic sample of other game voices (2-10 s clips, evenly spaced by file name)."""
    chosen: list[dict] = []
    for folder in dirs if dirs is not None else IMPOSTOR_DIRS:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        speaker = _speaker_of_dir(folder)
        eligible = []
        for path in sorted(folder.glob("*.wav")):
            try:
                info = sf.info(str(path))
            except RuntimeError:
                continue
            if 2.0 <= info.frames / info.samplerate <= 10.0:
                eligible.append(path)
        if not eligible:
            continue
        step = max(1, len(eligible) // per_dir)
        for path in eligible[::step][:per_dir]:
            chosen.append({"id": f"{speaker}:{path.stem}", "speaker": speaker, "path": str(path)})
    return chosen


def other_speaker_groups() -> dict[str, list[str]]:
    """Delivery task's confirmed other-speaker clips grouped by character ({} if the file is absent)."""
    if not EXCLUSIONS.exists():
        return {}
    try:
        data = read_json(EXCLUSIONS)
    except (OSError, ValueError):
        return {}
    groups: dict[str, list[str]] = {}
    for speaker, entries in (data.get("confirmed") or {}).items():
        groups[speaker] = [e["id"] for e in entries if isinstance(e, dict) and e.get("id")]
    return groups


def other_speakers_label() -> str:
    """The delivery task's other-speaker groups as text, e.g. 'HK-47/G0-T0/Atton' ('' when there are none)."""
    return "/".join(other_speaker_groups())


def inset_impostor_clips(variant: str, corpus: dict[str, dict] | None = None) -> list[dict]:
    """Confirmed lines of another voice inside the character's own clip set (the eval file's
    inset_impostor_speakers, e.g. Atton in Bao-Dur's gbl set), in the given audio variant (>= 2 s)."""
    corpus = corpus or corpus_index()
    chosen = []
    for speaker, ids in other_speaker_groups().items():
        if speaker not in INSET_IMPOSTOR_SPEAKERS:
            continue
        for clip_id in sorted(ids):
            row = corpus.get(clip_id)
            if not row or row["split"] == "test" or (row.get("duration") or 0) < INSET_IMPOSTOR_MIN_S:
                continue
            path = real_audio_path(row, variant)
            if path:
                chosen.append({"id": f"inset-{speaker.lower()}:{clip_id}", "speaker": f"{speaker.lower()} (in-set)",
                               "path": str(path)})
    return chosen


def known_other_speakers() -> tuple[set[str], str | None]:
    """Clip ids the delivery task confirmed as other characters (empty if its file is absent/unreadable)."""
    if not EXCLUSIONS.exists():
        return set(), None
    try:
        data = read_json(EXCLUSIONS)
    except (OSError, ValueError):
        log(f"warning: {EXCLUSIONS} unreadable; other-speaker exclusion uses the speaker models only")
        return set(), None
    ids = set(data.get("confirmed_ids") or [])
    for entries in (data.get("confirmed") or {}).values():
        ids.update(e["id"] for e in entries if isinstance(e, dict) and e.get("id"))
    return ids, str(EXCLUSIONS)


def _lcs_words(a: list[str], b: list[str]) -> int:
    """Length of the longest common word subsequence."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


NEAR_DUP_MIN_WORDS = 4
NEAR_DUP_RATIO = 0.7


def near_duplicates_of_test(corpus: dict[str, dict] | None = None) -> dict[str, list[dict]]:
    """For each test clip: non-test clips that repeat its line (text-based), plus the delivery task's
    near_duplicate_of_test list. Rule: word LCS >= 4 and >= 70 % of the shorter text."""
    corpus = corpus or corpus_index()
    tests = {i: norm_words(r["text"]) for i, r in corpus.items() if r["split"] == "test"}
    found: dict[str, dict[str, dict]] = {t: {} for t in tests}
    for clip_id, row in corpus.items():
        if row["split"] == "test":
            continue
        words = norm_words(row["text"])
        for test_id, test_words in tests.items():
            common = _lcs_words(test_words, words)
            if common >= NEAR_DUP_MIN_WORDS and common >= NEAR_DUP_RATIO * min(len(test_words), len(words)):
                found[test_id][clip_id] = {"id": clip_id, "split": row["split"], "common_words": common,
                                           "why": "text", "text": row["text"]}
    if EXCLUSIONS.exists() and tests:
        try:
            listed = read_json(EXCLUSIONS).get("near_duplicate_of_test") or {}
        except (OSError, ValueError):
            listed = {}
        # the held-out clip ids each note names ("... with held-out gblbaodur052")
        test_re = re.compile("(" + "|".join(map(re.escape, sorted(tests, key=len, reverse=True))) + r")(?![0-9])")
        for clip_id, note in listed.items():
            for test_id in test_re.findall(str(note)):
                if test_id in found and clip_id in corpus and clip_id not in found[test_id]:
                    found[test_id][clip_id] = {"id": clip_id, "split": corpus[clip_id]["split"], "common_words": None,
                                               "why": f"delivery exclusions: {note}", "text": corpus[clip_id]["text"]}
    return {t: sorted(v.values(), key=lambda e: e["id"]) for t, v in found.items() if v}


def load_run(run_dir: Path | str) -> dict:
    return read_json(Path(run_dir) / "takes.json")


def metric_out_path(run_dir: Path | str, metric: str) -> Path:
    return Path(run_dir) / "metrics" / f"{metric}.json"


def delivery_reference_ids(run: dict, tests: set[str]) -> dict[str, list[str]]:
    """label -> reference clip ids from the run's delivery map (test clips dropped)."""
    path = run.get("delivery_map")
    if not path or not Path(path).exists():
        return {}
    out = {}
    for label, spec in (read_json(path).get("deliveries") or {}).items():
        ids = [r.get("id") for r in spec.get("references", []) if r.get("id") and r.get("id") not in tests]
        if ids:
            out[label.casefold()] = ids
    return out


# ----------------------------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------------------------

def resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = math.gcd(int(rate), int(target))
    return resample_poly(x, int(target) // g, int(rate) // g).astype(np.float32)


def read_mono(path: Path | str, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1).astype(np.float32)
    if target_rate and rate != target_rate:
        x = resample(x, rate, target_rate)
        rate = target_rate
    return x, int(rate)


FRAME_S = 0.025
HOP_S = 0.010
ACTIVE_REL_DB = 35.0      # frame is "active" if within 35 dB of the clip's loud (95th pct) frames ...
ACTIVE_FLOOR_DB = -55.0   # ... and above -55 dBFS
MIN_ACTIVE_S = 0.03       # ignore active blips shorter than this (clicks)
MIN_PAUSE_S = 0.15        # internal silences at least this long count as pauses


def frame_db(x: np.ndarray, rate: int, frame_s: float = FRAME_S, hop_s: float = HOP_S) -> tuple[np.ndarray, int, int]:
    """Frame RMS in dBFS. Returns (db per frame, frame length, hop) in samples."""
    n = max(1, int(round(frame_s * rate)))
    hop = max(1, int(round(hop_s * rate)))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    windows = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    rms = np.empty(len(windows), dtype=np.float64)
    for start in range(0, len(windows), 512):  # chunked: a whole-clip float64 copy fails under memory pressure
        block = windows[start:start + 512].astype(np.float64)
        rms[start:start + 512] = np.sqrt(np.mean(block * block, axis=1) + 1e-20)
    return 20.0 * np.log10(rms + 1e-12), n, hop


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def activity(x: np.ndarray, rate: int) -> dict:
    """Energy-based speech activity: speech span, internal pauses, and per-frame dB.

    Returns start_s/end_s of the first/last active frame (None if nothing active), pauses as
    [start_s, end_s] pairs inside the span, and the frame arrays for further statistics.
    """
    db, n, hop = frame_db(x, rate)
    loud = float(np.percentile(db, 95)) if db.size else -120.0
    threshold = max(ACTIVE_FLOOR_DB, loud - ACTIVE_REL_DB)
    mask = db > threshold
    min_frames = max(1, int(round(MIN_ACTIVE_S / HOP_S)))
    for a, b in _runs(mask):
        if b - a < min_frames:
            mask[a:b] = False
    active = np.flatnonzero(mask)
    duration = len(x) / rate
    result = {"threshold_db": threshold, "loud_db": loud, "frame_db": db, "mask": mask, "hop": hop, "n": n,
              "duration_s": duration}
    if active.size == 0:
        result.update(start_s=None, end_s=None, pauses=[], span_s=0.0)
        return result
    first, last = int(active[0]), int(active[-1])
    start_s = first * hop / rate
    end_s = min(duration, (last * hop + n) / rate)
    pauses = []
    min_pause_frames = int(round(MIN_PAUSE_S / HOP_S))
    inside = ~mask[first:last + 1]
    for a, b in _runs(inside):
        if b - a >= min_pause_frames:
            pauses.append([(first + a) * hop / rate + n / rate / 2, (first + b) * hop / rate + n / rate / 2])
    result.update(start_s=start_s, end_s=end_s, pauses=pauses, span_s=max(0.0, end_s - start_s))
    return result


def active_rms_db(x: np.ndarray, rate: int) -> float | None:
    """Power-mean level of the active (speech) frames in dBFS; None when nothing is active."""
    act = activity(x, rate)
    db = act["frame_db"][act["mask"]]
    if not db.size:
        return None
    return float(10 * np.log10(np.mean(10 ** (db / 10))))


def speech_segment(x: np.ndarray, rate: int, pad_s: float = 0.1, min_s: float = 0.5) -> np.ndarray:
    """The speech span with a little padding (whole clip if nothing, or too little, is active)."""
    act = activity(x, rate)
    if act["start_s"] is None or act["span_s"] < min_s:
        return x
    a = max(0, int((act["start_s"] - pad_s) * rate))
    b = min(len(x), int((act["end_s"] + pad_s) * rate))
    return x[a:b]


# ----------------------------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------------------------

FILLERS = {"uh", "uhh", "um", "umm", "hmm", "hm", "mm", "mmm", "mhm", "er", "erm", "ah", "ahh", "eh", "oh", "huh"}
EQUIVALENTS = {"alright": ["all", "right"], "ok": ["okay"], "mr": ["mister"], "mrs": ["missus"], "dr": ["doctor"],
               "til": ["until"], "till": ["until"], "gonna": ["going", "to"], "wanna": ["want", "to"],
               # unambiguous modal + have contractions (2026-10-07: Whisper writes SB-11 "should've" as "should
               # have", which failed every take on WER)
               "should've": ["should", "have"], "could've": ["could", "have"], "would've": ["would", "have"],
               "might've": ["might", "have"], "must've": ["must", "have"]}


def norm_words(text: str) -> list[str]:
    """ASR-scoring normalisation: casefold, unify apostrophes, split hyphens, drop punctuation and fillers."""
    t = (text or "").casefold().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[-\u2010-\u2015]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    out: list[str] = []
    for word in t.split():
        word = word.strip("'")
        if not word or word in FILLERS:
            continue
        out.extend(EQUIVALENTS.get(word, [word]))
    return out


def derive_probe_names(prompt: str, skip=()) -> tuple[str, ...]:
    """Name-probe stems from an ASR prompt: each comma-separated item normalised (norm_words), minus the
    words in skip and the character's address term (common words Whisper spells without help)."""
    drop = {" ".join(norm_words(s)) for s in skip} | {" ".join(norm_words(CHAR.get("address_term") or ""))}
    stems = []
    for item in (prompt or "").split(","):
        stem = " ".join(norm_words(item))
        if stem and stem not in drop and stem not in stems:
            stems.append(stem)
    return tuple(stems)


# Proper nouns whose pronunciation is checked with a second, prompt-free ASR pass. Normalised (norm_words)
# stems; the last token matches as a prefix (Bao-Dur's "mandalor" covers Mandalore/Mandalorians). From the eval
# character file (asr.probe_names), else derived from the ASR prompt minus asr.probe_skip (General, Jedi, Sith):
# common words Whisper gets without help.
PROBE_NAMES = (tuple(EVAL_CHARACTER["asr"]["probe_names"]) if EVAL_CHARACTER["asr"].get("probe_names")
               else derive_probe_names(ASR_NAMES_PROMPT, EVAL_CHARACTER["asr"].get("probe_skip") or ()))


def edit_distance(ref: list, hyp: list) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment as (ref index | None, hyp index | None) pairs, in order."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def find_stems(words: list[str], stems=PROBE_NAMES) -> list[tuple[str, int, int]]:
    """(stem, start, end) word spans where a probe name occurs; the stem's last token matches as a prefix."""
    found = []
    for stem in stems:
        tokens = stem.split()
        k = len(tokens)
        for i in range(len(words) - k + 1):
            if all(words[i + j] == tokens[j] for j in range(k - 1)) and words[i + k - 1].startswith(tokens[-1]):
                found.append((stem, i, i + k))
    return found


def aligned_span(ref: list[str], hyp: list[str], start: int, end: int) -> str:
    """The hypothesis words aligned to ref[start:end] (substitutions and insertions inside the span)."""
    pairs = align_words(ref, hyp)
    inside = False
    out = []
    for i, j in pairs:
        if i is not None:
            inside = start <= i < end
        if inside and j is not None:
            out.append(hyp[j])
    return " ".join(out)


def wer_details(reference: str, hypothesis: str) -> dict:
    ref, hyp = norm_words(reference), norm_words(hypothesis)
    errors = edit_distance(ref, hyp)
    ref_chars, hyp_chars = list(" ".join(ref)), list(" ".join(hyp))
    char_errors = edit_distance(ref_chars, hyp_chars)
    return {
        "wer": (errors / len(ref)) if ref else (0.0 if not hyp else 1.0),
        "word_errors": errors, "ref_words": len(ref), "hyp_words": len(hyp),
        "cer": (char_errors / len(ref_chars)) if ref_chars else (0.0 if not hyp_chars else 1.0),
    }


def spoken_words(text: str) -> list[str]:
    """Words as spoken (fillers kept) for speaking-rate counts."""
    t = (text or "").casefold().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[-\u2010-\u2015]", " ", t)
    return [w.strip("'") for w in re.sub(r"[^a-z0-9' ]", " ", t).split() if w.strip("'")]


_DIGIT_SYLLABLES = {"0": 2, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 2, "8": 1, "9": 1}


def syllables_in_word(word: str) -> int:
    """Heuristic English syllable count (vowel groups with silent-e rules). Biased the same way for
    real clips and takes, so rate comparisons stay fair."""
    if word.isdigit():
        return sum(_DIGIT_SYLLABLES.get(c, 1) for c in word)
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$", lambda m: m.group(0)[0], w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def count_syllables(text: str) -> int:
    return sum(syllables_in_word(w) for w in spoken_words(text))


# ----------------------------------------------------------------------------------------------
# Statistics and devices
# ----------------------------------------------------------------------------------------------

def stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "p05": float(q05), "p25": float(q25), "median": float(q50),
            "p75": float(q75), "p95": float(q95), "max": float(v.max())}


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def torch_device(requested: str, cap_gb: float = 4.0, threads: int | None = None) -> str:
    """Resolve cpu/cuda for torch scripts; on cuda cap this process's share of the shared GPU.
    threads bounds torch's intra-op CPU threads (the machine is shared; default 8)."""
    import torch
    torch.set_num_threads(max(1, int(threads or 8)))
    if requested != "cuda":
        return "cpu"
    if not torch.cuda.is_available():
        log("cuda requested but not available; using cpu")
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    fraction = max(0.02, min(0.9, cap_gb / total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    log(f"cuda: capped at {fraction:.3f} of {total:.1f} GB (~{cap_gb:.1f} GB)")
    return "cuda"


def common_args(parser) -> None:
    parser.add_argument("--run-dir", required=True, help=f"{LAB.name}/eval/<run> folder holding takes.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (torch / CTranslate2)")


# ----------------------------------------------------------------------------------------------
# Speaker-similarity driver shared by the speaker-model scripts
# ----------------------------------------------------------------------------------------------

def _speaker_baselines(metric: str, real: dict[str, np.ndarray], imp: dict[str, np.ndarray], imp_meta: list[dict],
                       sets: dict, known: set[str], known_source: str | None) -> tuple[dict, np.ndarray]:
    """Centroid (iterative other-speaker screen) and real/impostor baselines for one audio variant."""
    train_ids = [c["id"] for c in sets["train"] if c["id"] in real]
    if not train_ids:
        raise SystemExit(f"{metric}: no train clips embedded")
    test_list = [c["id"] for c in sets["test"] if c["id"] in real]
    val_list = [c["id"] for c in sets["val"] if c["id"] in real]
    info = {c["id"]: c for c in sets["train"]}
    long_ids = {i for i in train_ids if (info[i].get("duration") or 0) >= MIN_CENTROID_S}
    known = known & set(train_ids)

    def build(kept: list[str]):
        members = [i for i in kept if i in long_ids] or list(kept)
        member_set = set(members)
        total = np.sum([real[i] for i in members], axis=0)
        centre = unit(total)
        loo = {i: float(real[i] @ unit(total - real[i])) if i in member_set else float(real[i] @ centre) for i in kept}
        # Purity anchor: VAL clips only, so held-out test clips never steer a training-data decision.
        if val_list:
            anchor = float(np.mean([real[i] @ centre for i in val_list]))
        else:
            anchor = float(np.median([loo[i] for i in members]))
        imp_mean = float(np.mean([v @ centre for v in imp.values()])) if imp else None
        return centre, members, loo, anchor, imp_mean

    def state(excluded: set[str]):
        kept = [i for i in train_ids if i not in excluded]
        centre, members, loo, anchor, imp_mean = build(kept)
        cos = {i: loo[i] if i in loo else float(real[i] @ centre) for i in train_ids}
        if imp_mean is None or anchor - imp_mean <= 0.02:
            ident = {i: None for i in train_ids}
        else:
            ident = {i: (c - imp_mean) / (anchor - imp_mean) for i, c in cos.items()}
        return kept, centre, members, loo, anchor, imp_mean, cos, ident

    suspect_ids: set[str] = set(known)
    kept, centroid, members, loo, anchor, imp_mean, cos_now, identity = state(suspect_ids)
    for _round in range(6):
        flagged = {i for i, v in identity.items() if v is not None and v < SUSPECT_IDENTITY} | known
        if flagged == suspect_ids or len(train_ids) - len(flagged) < 20:
            break
        suspect_ids = flagged
        kept, centroid, members, loo, anchor, imp_mean, cos_now, identity = state(suspect_ids)

    def why(i: str) -> str:
        reasons = (["delivery exclusions"] if i in known else []) + \
                  (["speaker model"] if identity[i] is not None and identity[i] < SUSPECT_IDENTITY else [])
        return " + ".join(reasons)

    suspects = sorted(({"id": i, "cos": cos_now[i], "identity": identity[i], "duration": info[i]["duration"],
                        "text_source": info[i]["text_source"], "text": info[i]["text"], "source": why(i)}
                       for i in suspect_ids), key=lambda s: (s["identity"] is None, s["identity"] or 0.0))
    if suspects:
        log(f"{metric}: {len(suspects)} train clips left out as other speakers ({len(known)} confirmed by the delivery "
            f"task, {sum(1 for s in suspects if 'speaker model' in s['source'])} with identity < {SUSPECT_IDENTITY}); "
            f"centroid built from {len(members)} of the other {len(kept)} (clips >= {MIN_CENTROID_S} s)")

    vs_centroid = {i: float(real[i] @ centroid) for i in real}
    pairs_real = [float(real[a] @ real[b]) for k, a in enumerate(test_list) for b in test_list[k + 1:]]
    pairs_imp = [float(imp[a] @ real[b]) for a in imp for b in test_list]
    by_speaker: dict[str, list[float]] = {}
    for clip in imp_meta:
        if clip["id"] in imp:
            by_speaker.setdefault(clip["speaker"], []).append(float(imp[clip["id"]] @ centroid))
    lowest = sorted(loo, key=loo.get)[:10]
    baselines = {
        "centroid_train_clips": len(members),
        "centroid_train_ids": members,
        "centroid_min_seconds": MIN_CENTROID_S,
        "kept_train_clips": len(kept),
        "train_suspects": suspects,
        "known_other_speakers": {"source": known_source, "ids_in_train": sorted(known)},
        "suspect_rule": f"confirmed other-speaker clips (delivery exclusions) plus identity < {SUSPECT_IDENTITY} "
                        "(leave-one-out cosine anchored 0 = impostor mean, 1 = real VAL mean; test clips are not "
                        "used here), iterated until stable",
        "purity_anchor": {"val_mean_cos": anchor, "n_val": len(val_list), "impostor_mean_cos": imp_mean},
        "train_identity": {i: round(v, 4) for i, v in identity.items() if v is not None},
        "train_loo_cos": {i: round(loo[i], 5) for i in kept},
        "test_vs_centroid": {**stats([vs_centroid[i] for i in test_list]),
                             "per_clip": {i: vs_centroid[i] for i in test_list}},
        "val_vs_centroid": {**stats([vs_centroid[i] for i in val_list]),
                            "per_clip": {i: vs_centroid[i] for i in val_list}},
        "heldout_real_vs_centroid": stats([vs_centroid[i] for i in test_list + val_list]),
        "train_leave_one_out": {**stats([loo[i] for i in kept]),
                                "lowest": [{"id": i, "cos": loo[i], "duration": info[i]["duration"]} for i in lowest]},
        "impostor_vs_centroid": {**stats([v for vals in by_speaker.values() for v in vals]),
                                 "by_speaker": {s: stats(v) for s, v in by_speaker.items()}},
        "test_pairwise": stats(pairs_real),
        "impostor_vs_test_pairwise": stats(pairs_imp),
        "impostor_ids": [c["id"] for c in imp_meta if c["id"] in imp],
    }
    return baselines, centroid


def run_speaker_metric(metric: str, settings: dict, embed_file, run_dir: Path | str, device: str = "cpu",
                       embed_take=None, take_settings: dict | None = None, side: dict | None = None) -> int:
    """Embed takes and real clips, build a train centroid per reference audio variant, write raw cosines + baselines.

    embed_file(path) -> 1-D vector, or (vector, side_vector) when `side` is given; vectors are normalised here.
    embed_take(path): optional different preprocessing for takes (e.g. level normalisation), cached separately
    under take_settings. side = {"name", "settings"}: a second per-file vector (e.g. WavLM layer features for
    delivery), cached under its own settings and written for takes and the delivery map's reference clips.
    The run's "sv_variants" (default: its audio variant) lists the reference variants scored; each take is
    embedded once and compared with every variant's centroid and real clips. Returns the exit code.
    """
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    primary = run["audio_variant"]
    variants = [v for v in (run.get("sv_variants") or [primary]) if v in VARIANTS]
    if primary not in variants:
        variants.insert(0, primary)
    corpus = corpus_index()
    held_out = {i for i, r in corpus.items() if r["split"] == "test"}
    cache = Cache(metric, settings)
    take_cache = Cache(f"{metric}_takes", {**settings, **(take_settings or {})}) if embed_take else cache
    side_cache = Cache(f"{metric}_{side['name']}", {**settings, **side["settings"]}) if side else None
    started = time.time()

    def embedding(path: str, role: str = "real", need_side: bool = False):
        sha = file_sha256(path)
        store = take_cache if role == "take" else cache
        hit = store.get(sha)
        side_hit = side_cache.get(sha) if (side_cache is not None and need_side) else None
        if hit is not None and (not need_side or side_hit is not None):
            return np.asarray(hit, dtype=np.float64), sha, (np.asarray(side_hit) if side_hit is not None else None)
        fn = embed_take if (role == "take" and embed_take) else embed_file
        value = fn(path)
        vector, extra = (value if isinstance(value, tuple) else (value, None))
        vector = unit(vector)
        store.put(sha, [round(float(x), 7) for x in vector])
        if side_cache is not None and extra is not None:
            side_cache.put(sha, [round(float(x), 5) for x in np.asarray(extra).ravel()])
            side_hit = side_cache.get(sha)
        return vector, sha, (np.asarray(side_hit) if side_hit is not None else None)

    def embed_real(path: str, need_side: bool = False):
        value, err = guarded(embedding, path, "real", need_side)
        if err and err["unscored"]:
            for c in (cache, take_cache, side_cache):
                if c is not None:
                    c.save()
            log(f"!! {metric}: real clip {path} failed twice ({err['error']}); the baseline would be incomplete")
            raise SystemExit(3)
        return value, err

    known, known_source = known_other_speakers()
    delivery_refs = {i for ids in delivery_reference_ids(run, held_out).values() for i in ids} if side else set()
    external = impostor_clips()
    per_variant: dict[str, dict] = {}
    missing: dict[str, list] = {}
    side_real: dict[str, list] = {}
    for variant in variants:
        sets = real_sets(variant)
        missing[variant] = sets["missing"]
        if not sets["test"] or not sets["train"]:
            log(f"{metric}: no {variant} real clips; variant skipped")
            continue
        real: dict[str, np.ndarray] = {}
        for role in ("train", "val", "test"):
            log(f"{metric}: embedding {len(sets[role])} {role} clips ({variant})")
            for i, clip in enumerate(sets[role], 1):
                want_side = bool(side) and variant == primary and clip["id"] in delivery_refs
                value, err = embed_real(clip["path"], want_side)
                if err:
                    log(f"   {clip['id']}: unreadable, left out ({err['error']})")
                    missing[variant].append({"id": clip["id"], "split": role, "error": err["error"]})
                    continue
                real[clip["id"]] = value[0]
                if want_side and value[2] is not None:
                    side_real[clip["id"]] = [round(float(x), 5) for x in value[2]]
                if i % 50 == 0:
                    log(f"  {i}/{len(sets[role])}")
        imp_meta = external + inset_impostor_clips(variant, corpus)
        if not imp_meta:
            log(f"!! {metric}: no impostor voices for {NAME} (speaker_screen.impostor_dirs in "
                f"{EVAL_CHARACTER_FILE or LAB / 'eval' / evalchar.FILE_NAME}); identity has no 0 anchor and the "
                "other-speaker screen uses the delivery exclusions only")
        imp: dict[str, np.ndarray] = {}
        log(f"{metric}: embedding {len(imp_meta)} impostor clips ({variant})")
        for clip in imp_meta:
            value, err = embed_real(clip["path"])
            if not err:
                imp[clip["id"]] = value[0]
        cache.save()
        baselines, centroid = _speaker_baselines(metric, real, imp, imp_meta, sets, known, known_source)
        per_variant[variant] = {"baselines": baselines, "centroid": centroid, "real": real}
    if primary not in per_variant:
        raise SystemExit(f"{metric}: no real clips for the run's audio variant {primary}")

    takes_out: dict[str, dict] = {}
    side_takes: dict[str, list] = {}
    log(f"{metric}: embedding {len(run['takes'])} takes against {', '.join(per_variant)} references")
    for take in run["takes"]:
        value, err = guarded(embedding, take["wav"], "take", bool(side))
        if err:
            takes_out[take["key"]] = err
            continue
        vector, sha, extra = value
        entry: dict = {"sha256": sha, "by_variant": {}}
        heldout = take.get("paired_id") or take.get("heldout_id")  # the line's real recording, if any
        if heldout:
            entry["paired_id"] = heldout
        ref = take.get("reference_id")
        for variant, pv in per_variant.items():
            e = {"centroid_cos": float(vector @ pv["centroid"])}
            if heldout and heldout in pv["real"]:
                e["heldout_cos"] = float(vector @ pv["real"][heldout])
            if ref and ref in pv["real"] and ref not in held_out:
                e["reference_cos"] = float(vector @ pv["real"][ref])
            entry["by_variant"][variant] = e
        takes_out[take["key"]] = entry
        if extra is not None:
            side_takes[take["key"]] = [round(float(x), 5) for x in extra]
    for c in (cache, take_cache, side_cache):
        if c is not None:
            c.save()

    out = {"metric": metric, "settings": settings, "take_settings": take_settings, "audio_variant": primary,
           "variants": list(per_variant), "device": device, "seconds": round(time.time() - started, 1),
           "missing_real": missing, "baselines": {v: pv["baselines"] for v, pv in per_variant.items()},
           "takes": takes_out}
    if side:
        out["side"] = {"name": side["name"], "settings": side["settings"], "variant": primary,
                       "real": side_real, "takes": side_takes}
    for variant, pv in per_variant.items():
        t = pv["baselines"]["test_vs_centroid"]
        i = pv["baselines"]["impostor_vs_centroid"]
        log(f"{metric} [{variant}]: real test vs centroid mean {t.get('mean', float('nan')):.3f} "
            f"(min {t.get('min', float('nan')):.3f}); impostors mean {i.get('mean', float('nan')):.3f}")
    code = finish_metric(run_dir, metric, out, run)
    log(f"{metric}: wrote {metric_out_path(run_dir, metric)}")
    return code


def exit_with(code: int) -> None:
    sys.stdout.flush()
    sys.exit(code)
