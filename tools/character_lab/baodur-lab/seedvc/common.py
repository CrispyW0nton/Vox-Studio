"""Shared paths and helpers for the Seed-VC V2 voice-clone recipe (baodur-lab/seedvc), for any lab character.

Character: VOICE_LAB_CHARACTER=<slug> picks voice-lab/characters/<slug>.json (default "baodur", so commands without
the variable behave exactly as before). Its "lab" folder is the data root: corpus.jsonl, delivery/, eval/, takes/ and
this recipe's seedvc/. Character-specific Seed-VC content (impostor voices, pYIN range, self-test prompt labels, the
check clip, ...) lives in <lab>/seedvc/settings.json; keys it leaves out take the recipe defaults below
(_RECIPE_DEFAULTS), and nothing character-identifying is hard-coded here.

Every entry point calls setup_seedvc() BEFORE importing torch/transformers/huggingface_hub, so that
- Hugging Face resolves only the prefetched local caches (HF_HOME=voice-lab/hf-cache, HF_HUB_OFFLINE=1);
  upstream train_v2.py instead forces HF_HUB_CACHE=./checkpoints/hf_cache, which is empty here;
- Seed-VC's relative paths work (hf_utils.load_custom_model_from_hf uses cache_dir="./checkpoints");
- the Seed-VC source tree is importable (modules.*, hf_utils).
CPU-only entry points (prepare.py, check_offline.py, selftest.py) call hide_cuda() first, so they cannot touch the
GPU even if a library would pick it by default.
This module itself only imports the standard library at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
import zlib
from pathlib import Path

VOICE_LAB = Path(r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab")
SEEDVC_ROOT = VOICE_LAB / "seed-vc"
HF_HOME = VOICE_LAB / "hf-cache"
CONFIG = SEEDVC_ROOT / "configs" / "v2" / "vc_wrapper.yaml"
LAB_CODE = VOICE_LAB / "baodur-lab"     # the lab's code folder (every character; the name is historical)
DELIVERY_CODE = LAB_CODE / "delivery"   # build_delivery_map.test_overlap: the delivery task's held-out leak rule
EVAL_CODE = LAB_CODE / "eval"           # evalcommon: the eval harness's near-duplicate rule, impostor sample
CHARACTERS_CODE = VOICE_LAB / "characters"


def _load_character() -> dict:
    """voice-lab/characters/character.load(): the VOICE_LAB_CHARACTER config (default baodur), paths as Path."""
    folder = str(CHARACTERS_CODE)
    sys.path.insert(0, folder)
    try:
        import character
    finally:
        sys.path.remove(folder)
    return character.load()


CHARACTER = _load_character()
NAME = CHARACTER["display_name"]    # as spoken in messages and listening notes ("Bao-Dur", "Jolee Bindo")

LAB = CHARACTER["lab"]
CORPUS = LAB / "corpus.jsonl"
RESTORED = LAB / "restored"
RESTORE_REPORT = LAB / "restore_report.json"
DELIVERY_MAP = LAB / "delivery" / "delivery_map.json"
SPEAKER_EXCLUSIONS = LAB / "delivery" / "exclude_other_speakers.json"   # written by the delivery task
DECISIONS = LAB / "DECISIONS.md"
EVAL_ROOT = LAB / "eval"
TAKES = LAB / "takes"
DATA = LAB / "seedvc"               # all Seed-VC data outputs live under here
PREPARED = DATA / "data"            # prepare.py -> PREPARED/<audio>/{train,val}/*.wav + prepare_report.json
RUNS = DATA / "runs"                # train.py  -> RUNS/<run>/{CFM,EMA-CFM}_epoch_*_step_*.pth, logs, run_meta.json
SELFTEST = DATA / "selftest"        # selftest.py sources/scores, convert.py outputs for checkpoint selection
SETTINGS_PATH = DATA / "settings.json"   # per-character Seed-VC content (see _RECIPE_DEFAULTS)
# The shared folder of decoded game voices (RvcDatasets): the character's own source_audio sits in it, and relative
# impostor_dirs resolve against it.
DATASETS = CHARACTER["source_audio"].parent if CHARACTER.get("source_audio") else LAB.parent / "RvcDatasets"

# Recipe defaults for <lab>/seedvc/settings.json keys. Only lab-wide rules belong here; a character's own content
# goes into its settings file (Bao-Dur's file records his values explicitly).
_RECIPE_DEFAULTS = {
    # Other voices for the identity floor (0 = impostor mean) and the self-test's out-of-domain sources: the eval
    # harness's impostor rule (evalcommon.IMPOSTOR_DIRS). Folder names resolve against DATASETS; a folder holding the
    # character's own voice is always dropped.
    "impostor_dirs": ["AttonDecoded", "CarthDecoded"],
    "impostor_voice": "other male game voice",     # how the ood self-test manifest describes them
    # pYIN range for every F0 statistic; must cover the voice and its -3/+4 st self-test copies.
    "f0": {"fmin": 60.0, "fmax": 400.0},
    "neutral_label": "natural",                    # the delivery map's neutral label (convert.py fallback)
    "promptsens_labels": None,      # three contrasting delivery labels for selftest promptsens; None: set skipped
    "check_source_id": None,        # check_offline.py source clip; None: first val speech clip of 3-6 s
    "loudness_fallback": None,      # {"p05", "p95"} when no prepare_report.json exists; None: no level clamp
    "audio_decision": None,         # the training-audio decision as recorded in reports; None: generic sentence
}


def _load_settings() -> dict:
    settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _RECIPE_DEFAULTS.items()}
    if SETTINGS_PATH.exists():
        doc = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        unknown = sorted(k for k in doc if k not in settings and not k.startswith("_"))
        if unknown:
            raise ValueError(f"{SETTINGS_PATH}: unknown keys {unknown} (known: {sorted(settings)}; '_' keys are notes)")
        settings.update({k: v for k, v in doc.items() if not k.startswith("_")})
    return settings


SETTINGS = _load_settings()
NEUTRAL_LABEL = str(SETTINGS["neutral_label"])


def impostor_speaker(folder: Path) -> str:
    """'AttonDecoded' -> 'atton', 'KreiaDecoded101' -> 'kreia', 'CarthExact' -> 'carth'."""
    return re.sub(r"(Decoded|Exact|Conversational)\d*$", "", folder.name).lower() or folder.name.lower()


def _impostor_dirs() -> list[Path]:
    own = os.path.normcase(os.path.abspath(str(CHARACTER["source_audio"]))) if CHARACTER.get("source_audio") else None
    voice = str(CHARACTER.get("speaker") or CHARACTER["slug"]).lower()
    dirs = []
    for entry in SETTINGS["impostor_dirs"]:
        folder = Path(entry) if Path(entry).is_absolute() else DATASETS / entry
        if os.path.normcase(os.path.abspath(str(folder))) == own or voice in impostor_speaker(folder):
            continue   # never the character's own voice
        dirs.append(folder)
    return dirs


IMPOSTOR_DIRS = _impostor_dirs()
IMPOSTOR_SPEAKERS = [impostor_speaker(folder) for folder in IMPOSTOR_DIRS]
IMPOSTOR_LABEL = "/".join(s.capitalize() for s in IMPOSTOR_SPEAKERS) or "no impostors"   # "Atton/Carth"
IMPOSTOR_VOICE = str(SETTINGS["impostor_voice"])

MODEL_SR = 22050      # configs/v2/vc_wrapper.yaml: sr, mel fmax=None -> 11025 Hz, BigVGAN 22k/80-band
FEATURE_SR = 16000    # HuBERT/ASTRAL content tokens and CAMPPlus style vectors
OUT_SR = 48000
PAINTER_RESERVE_GB = 8.0
# Outside the allocator cap even after the context exists: kernel images loaded lazily on first use, cuBLAS/cuDNN
# handles. The context itself is already excluded from torch.cuda.mem_get_info() (see cap_vram).
CONTEXT_GROWTH_GB = 0.5
# pYIN settings for every F0 statistic in this recipe; fmin/fmax per character (settings "f0"). The default 60-400 Hz
# covers Bao-Dur (~110 Hz median), the +-4 st self-test copies and other male voices; 0.2 st bins make pYIN ~7x faster
# than librosa's 0.1 default.
_F0 = SETTINGS["f0"]
F0_KW = {"fmin": float(_F0["fmin"]), "fmax": float(_F0["fmax"]), "frame_length": int(_F0.get("frame_length", 1024)),
         "hop_length": int(_F0.get("hop_length", 256)), "resolution": float(_F0.get("resolution", 0.2))}

USER_AUDIO_DECISION = SETTINGS["audio_decision"] or (
    "the user chose the original game audio for training data AND reference/prompt clips for every character "
    f"({LAB.name}/DECISIONS.md, 2026-09-30); restored/ is only an optional post-processing experiment")

_HF_OVERRIDES = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                 "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE")


def hide_cuda() -> None:
    """CPU-only entry points: make CUDA invisible to this process (must run before torch is imported).

    "-1", not "": on Windows an empty environment value does not exist (Python's os.environ and Git Bash's
    CUDA_VISIBLE_DEVICES="" both leave the variable unset for Win32, measured 2026-09-30: torch still sees the GPU),
    while an invalid device id hides every GPU."""
    if "torch" in sys.modules:
        raise RuntimeError("hide_cuda() must run before torch is imported")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def setup_seedvc() -> None:
    """Pin offline HF caches, make Seed-VC importable and chdir into it. Call before importing torch/HF."""
    for name in ("huggingface_hub", "transformers"):
        if name in sys.modules:
            raise RuntimeError(f"setup_seedvc() must run before {name} is imported")
    os.environ["HF_HOME"] = str(HF_HOME)
    for key in _HF_OVERRIDES:
        os.environ.pop(key, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = str(SEEDVC_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def shim_pkg_resources() -> None:
    """webrtcvad (resemblyzer's VAD) imports pkg_resources, which setuptools>=81 no longer ships."""
    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        import importlib.metadata as metadata
        stub = types.ModuleType("pkg_resources")
        stub.get_distribution = lambda name: types.SimpleNamespace(version=metadata.version(name))
        sys.modules["pkg_resources"] = stub


def require_original(audio: str, allowed: bool, what: str) -> None:
    """Refuse restored audio for training data or prompts unless the user has decided otherwise since 2026-09-30."""
    if audio != "original" and not allowed:
        raise SystemExit(f"ERROR: {what} would use {audio!r} audio, but {USER_AUDIO_DECISION}. "
                         "Pass --allow-restored only after a new user decision (record it in DECISIONS.md).")


# ----------------------------------------------------------------------------------------------- corpus

def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_path(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def restored_path(clip_id: str) -> Path:
    return RESTORED / f"{clip_id}.wav"


def clip_path(row: dict, audio: str) -> Path:
    if audio == "original":
        return Path(row["audio"])
    if audio == "restored":
        return restored_path(row["id"])
    raise ValueError(f"audio must be original|restored, got {audio!r}")


class HeldOutGuard:
    """The split=='test' clips (10 per character) are evaluation ground truth only: never training, validation or
    prompts."""

    def __init__(self, rows: list[dict]):
        test = [r for r in rows if r["split"] == "test"]
        self.ids = {r["id"] for r in test}
        self.sha256 = {r["sha256"] for r in test}
        self.paths = {norm_path(r["audio"]) for r in test} | {norm_path(restored_path(r["id"])) for r in test}

    def file_reason(self, path) -> str | None:
        path = Path(path)
        if norm_path(path) in self.paths:
            return "path is a held-out test clip"
        if path.exists() and sha256_file(path) in self.sha256:
            return "content hash matches a held-out test clip"
        return None


def lab_module(folder: Path, name: str):
    """Import another lab task's module (shared rule, read-only) without writing __pycache__ into its folder."""
    import importlib
    if name in sys.modules:
        return sys.modules[name]
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(folder))
    try:
        return importlib.import_module(name)
    finally:
        sys.dont_write_bytecode = previous
        try:
            sys.path.remove(str(folder))
        except ValueError:
            pass


def _lcs_words(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1] if a and b else 0


def text_overlap(a: str, b: str) -> str | None:
    """Why two transcripts count as the same line under the lab's rules, or None. Both rules are imported live:
    the delivery rule (build_delivery_map.test_overlap: character ratio >= 0.6 or >= 5 words in a row; it decides the
    delivery map and exclude_other_speakers.json) and the eval rule (evalcommon: word LCS >= 4 and >= 70 % of the
    shorter line; it decides which training clips count as a held-out leak in the scorecard)."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return None
    bdm = lab_module(DELIVERY_CODE, "build_delivery_map")
    ec = lab_module(EVAL_CODE, "evalcommon")
    reasons = []
    hit = bdm.test_overlap(a, [("other", b)])
    if hit:
        reasons.append(f"delivery rule: '{hit['phrase']}' ({hit['run']} words in a row, ratio {hit['ratio']:.2f})")
    wa, wb = ec.norm_words(a), ec.norm_words(b)
    common = _lcs_words(wa, wb)
    if wa and wb and common >= getattr(ec, "NEAR_DUP_MIN_WORDS", 4) and \
            common >= getattr(ec, "NEAR_DUP_RATIO", 0.7) * min(len(wa), len(wb)):
        reasons.append(f"eval rule: {common} words in common")
    return "; ".join(reasons) or None


def near_duplicate_screen(rows: list[dict]) -> dict:
    """Recompute, from the corpus, which train/val clips repeat a held-out TEST line and which train clips repeat a
    VAL line, under both lab rules (text_overlap). Live, so a rule or corpus change is picked up even before the
    delivery task rewrites exclude_other_speakers.json."""
    tests = [r for r in rows if r["split"] == "test"]
    vals = [r for r in rows if r["split"] == "val"]
    near_test: dict[str, list[str]] = {}
    near_val: dict[str, list[str]] = {}
    for row in rows:
        text = (row.get("text") or "").strip()
        if row["split"] == "test" or not text:
            continue
        for test in tests:
            why = text_overlap(text, test["text"])
            if why:
                near_test.setdefault(row["id"], []).append(f"{why} with held-out {test['id']}")
        if row["split"] == "train":
            for val in vals:
                why = text_overlap(text, val["text"])
                if why:
                    near_val.setdefault(row["id"], []).append(f"{why} with val {val['id']}")
    return {"near_test": {k: "; ".join(v) for k, v in near_test.items()},
            "near_val": {k: "; ".join(v) for k, v in near_val.items()}}


def speaker_exclusions(required: bool = True, rows: list[dict] | None = None) -> dict:
    """Clips that must not be trained on or prompted with (the character's <lab>/delivery screen).

    confirmed: from delivery/exclude_other_speakers.json: embedding outliers on two encoders AND another
               character's words (Bao-Dur: HK-47, G0-T0, Atton, 241.5 s, all in the train split).
    uncertain: embedding outliers without text evidence (Bao-Dur: mostly sub-second combat barks and the 3 effort
               sounds), left out unless someone listened and confirmed they are the character.
    near_test: train/val clips repeating a held-out test line: the delivery file's list UNION both lab rules
               recomputed live (near_duplicate_screen). Training on one makes eval flag the line as a leak.
    near_val:  train clips repeating a val line (both rules): they would let fine-tuned checkpoints "know" val lines
               that pretrained never heard, biasing checkpoint selection.
    """
    rows = rows if rows is not None else load_corpus()
    live = near_duplicate_screen(rows)
    if SPEAKER_EXCLUSIONS.exists():
        doc = read_json(SPEAKER_EXCLUSIONS)
        path, digest = str(SPEAKER_EXCLUSIONS), sha256_file(SPEAKER_EXCLUSIONS)[:16]
    elif required:
        raise FileNotFoundError(f"{SPEAKER_EXCLUSIONS} is missing: run baodur-lab/delivery/exclusions.py first "
                                "(the corpus contains other characters' lines)")
    else:
        doc, path, digest = {}, None, None
    confirmed = {clip["id"]: name for name, clips in doc.get("confirmed", {}).items() for clip in clips}
    confirmed.update({cid: confirmed.get(cid, "other speaker") for cid in doc.get("confirmed_ids", [])})
    file_near = {str(k): str(v) for k, v in (doc.get("near_duplicate_of_test") or {}).items()}
    near_test = dict(file_near)
    for cid, why in live["near_test"].items():
        near_test[cid] = f"{near_test[cid]}; {why}" if cid in near_test else why
    return {"path": path, "sha256": digest, "confirmed": confirmed,
            "uncertain": {u["id"]: "speaker screen outlier, not attributed (listen first)" for u in doc.get("uncertain", [])},
            "near_test": near_test, "near_test_file": file_near, "near_val": live["near_val"]}


def other_speaker_names() -> list[str]:
    """Characters the delivery screen found inside this character's clips, in file order ([] without a screen)."""
    try:
        return [str(name) for name in (read_json(SPEAKER_EXCLUSIONS).get("confirmed") or {})]
    except (OSError, ValueError, AttributeError):
        return []


def check_source_default(rows: list[dict]) -> str:
    """check_offline.py's source clip: settings check_source_id, else the first val speech clip of 3-6 s (a short
    clip keeps the CPU check fast), else the first val speech clip."""
    if SETTINGS["check_source_id"]:
        return str(SETTINGS["check_source_id"])
    val = [r for r in rows if r["split"] == "val" and r["kind"] == "speech"]
    pick = next((r for r in val if 3.0 <= float(r["duration"]) <= 6.0), val[0] if val else None)
    if pick is None:
        raise SystemExit(f"ERROR: {CORPUS} has no val speech clip for the check")
    return pick["id"]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def impostor_clips(per_dir: int = 15, min_s: float = 2.0, max_s: float = 10.0, offset: float = 0.0) -> list[dict]:
    """Deterministic sample of the impostor voices' game lines (IMPOSTOR_DIRS, Bao-Dur: Atton/Carth; evalcommon's
    impostor rule; offset shifts the sampling grid by a fraction of a step, so a second sample avoids the first)."""
    import soundfile as sf
    chosen = []
    for folder in IMPOSTOR_DIRS:
        if not folder.is_dir():
            continue
        eligible = []
        for path in sorted(folder.glob("*.wav")):
            try:
                info = sf.info(str(path))
            except RuntimeError:
                continue
            if min_s <= info.frames / info.samplerate <= max_s:
                eligible.append((path, info.frames / info.samplerate))
        step = max(1, len(eligible) // per_dir)
        start = min(len(eligible) - 1, int(step * offset)) if eligible else 0
        for path, seconds in eligible[start::step][:per_dir]:
            chosen.append({"speaker": impostor_speaker(folder), "path": str(path),
                           "stem": path.stem, "seconds": round(seconds, 3)})
    return chosen


# ------------------------------------------------------------------------------------------------ audio

def resample(x, sr_in: int, sr_out: int):
    import librosa
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out:
        return x
    return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq").astype(np.float32)


def load_mono(path, sr: int | None = None):
    """Read any soundfile-readable file as float32 mono; resample with soxr_hq when sr is given."""
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr is not None and rate != sr:
        audio, rate = resample(audio, rate, sr), sr
    return audio, rate


def write_pcm16(path, x, sr: int) -> None:
    import numpy as np
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 32767 / 32768)
    sf.write(str(path), x, sr, subtype="PCM_16")


def active_rms_db(x, sr: int, frame_s: float = 0.02) -> float:
    """Mean power of 20 ms frames within 40 dB of the loudest frame (and above -60 dBFS), in dBFS."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(sr * frame_s))
    n = len(x) // hop
    if n == 0:
        return float(10 * np.log10(np.mean(x ** 2) + 1e-12))
    power = (x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    level = 10 * np.log10(power + 1e-12)
    active = level > max(-60.0, level.max() - 40.0)
    if not active.any():
        active = np.ones_like(active)
    return float(10 * np.log10(power[active].mean() + 1e-12))


def f0_track(wave22):
    """pYIN F0 (Hz, NaN where unvoiced) and voiced flags of a 22.05 kHz wave, hop 256 (11.6 ms)."""
    import librosa
    import numpy as np
    f0, voiced, _ = librosa.pyin(np.asarray(wave22, dtype=np.float32), sr=MODEL_SR, **F0_KW)
    return f0, voiced


def f0_summary(f0, voiced) -> dict | None:
    """Median F0 (Hz) and p10-p90 range (semitones) of voiced frames; None when (almost) unvoiced."""
    import numpy as np
    f0 = np.asarray(f0)[np.asarray(voiced, dtype=bool) & np.isfinite(f0)]
    if len(f0) < 10:
        return None
    semis = 12 * np.log2(f0)
    return {"median_hz": round(float(np.median(f0)), 1),
            "range_st": round(float(np.percentile(semis, 90) - np.percentile(semis, 10)), 2)}


def f0_stats(wave22) -> dict | None:
    return f0_summary(*f0_track(wave22))


def f0_pair(src_track, out_track, aligned: bool = True) -> dict:
    """How much of a source's pitch survives a conversion: median shift (st), range ratio, contour correlation
    (frame-aligned, only meaningful when the conversion kept the timing)."""
    import numpy as np
    s, o = f0_summary(*src_track), f0_summary(*out_track)
    out = {"source_f0": s, "output_f0": o, "median_shift_st": None, "range_ratio": None, "f0_corr": None}
    if s and o:
        out["median_shift_st"] = round(float(12 * np.log2(o["median_hz"] / s["median_hz"])), 2)
        out["range_ratio"] = round(float(o["range_st"] / max(s["range_st"], 0.1)), 3)
    if aligned:
        (fs, vs), (fo, vo) = src_track, out_track
        m = min(len(fs), len(fo))
        both = np.asarray(vs[:m], bool) & np.asarray(vo[:m], bool) & np.isfinite(fs[:m]) & np.isfinite(fo[:m])
        if both.sum() >= 10:
            out["f0_corr"] = round(float(np.corrcoef(np.log2(fs[:m][both]), np.log2(fo[:m][both]))[0, 1]), 3)
    return out


def hf_tilt_db(wave22, split_hz: float = 4000.0, top_hz: float = 10500.0) -> float | None:
    """Energy of 4-10.5 kHz relative to 50 Hz-4 kHz over active frames (within 40 dB of the loudest), in dB.
    A rise against the ground truth / the pretrained model is residual diffusion noise or hiss."""
    import numpy as np
    x = np.asarray(wave22, dtype=np.float64)
    if len(x) < 2048:
        return None
    n_fft, hop = 1024, 256
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft)[None, :], axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / MODEL_SR)
    level = 10 * np.log10(spec.sum(axis=1) + 1e-12)
    active = level > level.max() - 40
    low = spec[active][:, (freqs >= 50) & (freqs < split_hz)].sum()
    high = spec[active][:, (freqs >= split_hz) & (freqs < top_hz)].sum()
    return float(10 * np.log10((high + 1e-12) / (low + 1e-12)))


def fade(x, sr: int, ms: float = 5.0):
    import numpy as np
    x = np.array(x, dtype=np.float32, copy=True)
    n = min(len(x) // 2, int(sr * ms / 1000))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def finalize(y22, target_db: float | None = None, ceiling_db: float = -1.0):
    """22.05 kHz model output -> 48 kHz take: soxr resample, optional level match, 5 ms fades, peak ceiling.

    Returns (wave48, gain_db, ceiling_gain_db). The ceiling is a plain gain (no limiter), so the waveform shape
    is untouched; ceiling_gain_db says how much a peak forced the take below its level target.
    """
    import numpy as np
    y48 = resample(y22, MODEL_SR, OUT_SR)
    gain_db = 0.0 if target_db is None else float(target_db - active_rms_db(y48, OUT_SR))
    y48 = fade(y48 * np.float32(10 ** (gain_db / 20)), OUT_SR)
    peak = float(np.abs(y48).max()) if len(y48) else 0.0
    ceiling = 10 ** (ceiling_db / 20)
    ceiling_gain_db = 0.0
    if peak > ceiling:
        ceiling_gain_db = float(20 * np.log10(ceiling / peak))
        y48 = y48 * np.float32(ceiling / peak)
    return y48.astype(np.float32), gain_db, ceiling_gain_db


# ------------------------------------------------------------------------------------------------- misc

def take_seed(base: int, key: str) -> int:
    return (int(base) + zlib.crc32(key.encode("utf-8"))) % (2 ** 31)


def safe_name(text: str, limit: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text)).strip("_")[:limit] or "x"


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seedvc_commit() -> str:
    head = SEEDVC_ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (SEEDVC_ROOT / ".git" / ref.split(" ", 1)[1]).read_text().strip()[:12]
        return ref[:12]
    except OSError:
        return "unknown"


def pid_alive(pid) -> bool:
    """True if a process with this id is running (on Windows via OpenProcess/GetExitCodeProcess, never a signal)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5             # access denied: it exists, it just is not ours
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# -------------------------------------------------------------------------------------------------- GPU

def pick_device(requested: str = "cuda"):
    """'cuda' fails loudly when CUDA is missing (a 2000-step run must never fall back to the CPU); 'cpu' for smoke
    tests; 'auto' picks CUDA when present."""
    import torch
    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available (pass --device cpu for a smoke test)")
    return torch.device("cuda", 0)


def cap_vram(device, max_gb: float, reserve_gb: float = PAINTER_RESERVE_GB, check_headroom: bool = True) -> dict:
    """Cap this process's CUDA caching allocator and refuse to start if Painter's reserve would be squeezed.

    torch.cuda.mem_get_info() is this process's first CUDA call, so it creates the context and the free memory it
    reports already excludes that context. What can still grow outside the cap is CONTEXT_GROWTH_GB (lazily loaded
    kernels, library handles). Rule: free_after_context - cap - CONTEXT_GROWTH_GB >= Painter reserve.
    """
    import torch
    if device.type != "cuda":
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info(device)
    free_gb, total_gb = free / 2 ** 30, total / 2 ** 30
    fraction = min(0.95, max_gb / total_gb)
    info = {"device": torch.cuda.get_device_name(device), "total_gb": round(total_gb, 2),
            "free_gb_after_context": round(free_gb, 2), "cap_gb": max_gb, "fraction": round(fraction, 4),
            "context_growth_gb": CONTEXT_GROWTH_GB, "reserve_gb": reserve_gb,
            "headroom_after_job_gb": round(free_gb - max_gb - CONTEXT_GROWTH_GB, 2)}
    if check_headroom and free_gb - max_gb - CONTEXT_GROWTH_GB < reserve_gb:
        raise RuntimeError(
            f"GPU headroom check failed: {free_gb:.1f} GB free after this process's CUDA context; a {max_gb:.1f} GB "
            f"allocator cap + ~{CONTEXT_GROWTH_GB} GB of lazily loaded kernels would leave "
            f"{free_gb - max_gb - CONTEXT_GROWTH_GB:.1f} GB < {reserve_gb:.0f} GB for Substance Painter. "
            "Lower --max-vram-gb or wait.")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return info


# ------------------------------------------------------------------------------------------------ model

def build_wrapper():
    """Instantiate configs/v2/vc_wrapper.yaml (HuBERT-large, whisper-small tokenizer, BigVGAN from HF_HOME)."""
    import yaml
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg = DictConfig(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    return instantiate(cfg)


def _strip(state: dict) -> dict:
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


def load_finetuned(module, regulator, path, key: str) -> dict:
    """Strictly load a {'net': {key, 'length_regulator'}} checkpoint (upstream load_checkpoints is strict=False)."""
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    module.load_state_dict(_strip(ckpt["net"][key]), strict=True)
    regulator.load_state_dict(_strip(ckpt["net"]["length_regulator"]), strict=True)
    return {"iters": ckpt.get("iters"), "epoch": ckpt.get("epoch"), "meta": ckpt.get("meta", {})}


def load_weights(wrapper, cfm_ckpt=None, ar_ckpt=None) -> dict:
    """Pretrained Seed-VC V2 (cfm_small/ar_base, ASTRAL, CAMPPlus from seed-vc/checkpoints), then fine-tunes."""
    wrapper.load_checkpoints()
    info = {"cfm": "pretrained:Plachta/Seed-VC v2/cfm_small.pth", "ar": "pretrained:Plachta/Seed-VC v2/ar_base.pth"}
    if cfm_ckpt:
        load_finetuned(wrapper.cfm, wrapper.cfm_length_regulator, cfm_ckpt, "cfm")
        info["cfm"] = str(cfm_ckpt)
    if ar_ckpt:
        load_finetuned(wrapper.ar, wrapper.ar_length_regulator, ar_ckpt, "ar")
        info["ar"] = str(ar_ckpt)
    return info


def checkpoint_step(path) -> int:
    match = re.search(r"_step_(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else -1


def run_checkpoints(run_dir, kind: str = "CFM") -> list[Path]:
    """kind: CFM | AR (raw weights) or EMA-CFM | EMA-AR (moving-average weights)."""
    return sorted(Path(run_dir).glob(f"{kind}_epoch_*_step_*.pth"), key=checkpoint_step)


def checkpoint_tag(path) -> str:
    """<run>_s<step> for raw weights, <run>_ema_s<step> for EMA weights."""
    path = Path(path)
    ema = "_ema" if path.name.startswith("EMA-") else ""
    return f"{path.parent.name}{ema}_s{checkpoint_step(path):05d}"


def selected_checkpoint(runs_root, run: str) -> tuple[str, str | None]:
    """(tag, checkpoint path or None for pretrained) a run's selection.json resolves to: the pick confirmed by
    listening (selftest.py confirm), else the gated proxy pick (pretrained when nothing beat it)."""
    path = Path(runs_root) / run / "selection.json"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found: run selftest.py score --train-run {run} first")
    selection = read_json(path)
    pick = selection.get("confirmed") or {"tag": selection["tag"], "cfm_checkpoint": selection.get("cfm_checkpoint")}
    return pick["tag"], pick.get("cfm_checkpoint")


def frozen_training_report(ckpt_path) -> Path | None:
    """The prepare_report.json copy train.py froze into the run folder (what that checkpoint was trained on)."""
    path = Path(ckpt_path).parent / "training_data" / "prepare_report.json"
    return path if path.exists() else None
