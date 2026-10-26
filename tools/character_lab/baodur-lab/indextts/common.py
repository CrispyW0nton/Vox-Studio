"""Shared paths and helpers for the voice lab's IndexTTS-2.5 recipe (render.py, smoke_cpu.py).

Character: voice-lab\\characters (VOICE_LAB_CHARACTER=<slug>; unset = baodur, Bao-Dur). Its data folder <lab>
(characters\\<slug>.json "lab") holds corpus.jsonl and the delivery step's outputs (<lab>\\delivery); this recipe
writes trimmed references and smoke reports under <lab>\\indextts and takes under <lab>\\takes\\indextts25.
Character-specific content lives in <lab>\\indextts, never in this code:
  emotion_controls.json  per-delivery emotion vectors + QwenEmotion descriptions (render.py; required to render),
                         optional "base_label" = the neutral delivery for maps without "natural" (base_delivery)
  text_rewrites.json     spelling -> spoken-form rewrites applied by clean_text (optional; none when absent)
  smoke_defaults.json    smoke_cpu.py default clips/lines (optional; derived from the delivery outputs when absent)

Run with the IndexTTS venv (its own uv venv; never pip install into other lab venvs):
  C:\\Users\\NewAdmin\\Documents\\GDeveloper\\External\\voice-lab\\index-tts\\.venv\\Scripts\\python.exe

Install: voice-lab\\index-tts = github.com/index-tts/index-tts @ d9e41aac (unmodified), weights =
huggingface.co/IndexTeam/IndexTTS-2.5 @ c39ce5ba in index-tts\\checkpoints, and the external models the 2.5 code
loads (w2v-bert-2.0, CAMPPlus, BigVGAN v2 22 kHz) in the flat layout infer_v2_5.py checks FIRST
(checkpoints\\hf_cache\\{w2v-bert-2.0\\, campplus_cn_common.bin, bigvgan\\}), so its downloader
(indextts.utils.model_download.ensure_models_available) is never reached. offline_env() + net_guard() make any
attempt to fetch something at runtime fail loudly instead of downloading. Provenance, sizes, hashes and license
terms: index-tts\\PROVENANCE.json and index-tts\\LICENSE_NOTES.md.

Lab rules carried over from the other recipes (voxcpm/common.py, gsv, seedvc); they hold for every character:
  * User decision 2026-09-30 (BaoDurLab\\DECISIONS.md; each lab's DECISIONS.md adopts it): reference/prompt audio is
    the ORIGINAL game audio; the ClearerVoice-restored copies are never conditioning input (restored needs
    --allow-restored-audio).
  * Held-out TEST clips are never read as audio and never condition a take.
  * The speaker screen (<lab>\\delivery\\exclude_other_speakers.json) is enforced: 'confirmed' other-character
    clips stop the run, 'uncertain' ones are skipped, 'suspect' ones are tagged.
  * Shared GPU: keep >= 8 GB VRAM free for Substance Painter (gpu_budget) and do not start a large CPU-side load
    unless enough Windows commit is free (host_ram_budget). Never kill other processes.
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

# --------------------------------------------------------------------------------------------- character + paths
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR.parents[1] / "characters"))
import character  # noqa: E402  (voice-lab\characters; VOICE_LAB_CHARACTER selects the character, default baodur)

CHAR = character.load()
# The selected character's data folder. BAODUR_LAB is the override this recipe had before it served more than one
# character; it still applies to Bao-Dur only.
LAB = Path(os.environ.get("BAODUR_LAB", str(CHAR["lab"]))) if CHAR["slug"] == "baodur" else Path(CHAR["lab"])
CORPUS = LAB / "corpus.jsonl"
RESTORED_DIR = LAB / "restored"
DELIVERY_MAP = LAB / "delivery" / "delivery_map.json"
TAKES_SCRIPT = LAB / "delivery" / "takes_script.json"
SPEAKER_SCREEN = LAB / "delivery" / "exclude_other_speakers.json"
DECISIONS = LAB / "DECISIONS.md"
DATA_ROOT = LAB / "indextts"  # trimmed reference cache, smoke reports, the character's recipe data (below)
REF_CACHE = DATA_ROOT / "refs"
TAKES_ROOT = LAB / "takes" / "indextts25"
EMOTION_CONTROLS_FILE = DATA_ROOT / "emotion_controls.json"  # per-delivery emotion controls (render.py)
TEXT_REWRITES_FILE = DATA_ROOT / "text_rewrites.json"  # spelling -> spoken form (clean_text); optional
SMOKE_DEFAULTS_FILE = DATA_ROOT / "smoke_defaults.json"  # smoke_cpu.py default clips/lines; optional

VOICE_LAB = Path(os.environ.get("BAODUR_VOICE_LAB", r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab"))
INDEX_ROOT = VOICE_LAB / "index-tts"
INDEX_PYTHON = INDEX_ROOT / ".venv" / "Scripts" / "python.exe"
MODEL_DIR = INDEX_ROOT / "checkpoints"
CFG_PATH = MODEL_DIR / "config.yaml"
AUX_DIR = MODEL_DIR / "hf_cache"
PROVENANCE = INDEX_ROOT / "PROVENANCE.json"
FETCH_LOG = INDEX_ROOT / "_provenance" / "fetch_log.json"

ENGINE = "indextts25"
MODEL_VERSION = "IndexTTS-2.5"
CODE_REPO = "https://github.com/index-tts/index-tts"
CODE_COMMIT = "d9e41aac89fd00b3d71497fddb287b7f24613712"
HF_REPO = "IndexTeam/IndexTTS-2.5"
HF_REVISION = "c39ce5ba981572cb187443877ff559dfb246ce63"
# Conditioning encoders the generator uses (eval/aggregate.py reports a speaker metric as coupled only when an
# engine conditions on that metric's own model; none of these is ERes2Net, Resemblyzer/GE2E or WavLM).
CONDITIONING_ENCODERS = ["campplus (funasr/campplus, speaker style)", "w2v-bert-2.0 (facebook, semantic/emotion)"]

NATIVE_SR = 22050  # IndexTTS-2.5 s2mel + BigVGAN output rate
OUTPUT_SR = 48000  # lab take format: 48 kHz mono PCM16
AUDIO_VARIANTS = ("original", "restored")
# the lab-wide decision as recorded in the character's own lab (Bao-Dur: BaoDurLab\DECISIONS.md)
RESTORED_DECISION = (f"user decision 2026-09-30 ({CHAR['lab'].name}\\DECISIONS.md): training and reference audio stay "
                     "ORIGINAL game audio; restored copies are only an optional post-processing experiment")
EMO_ORDER = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")


def load_text_rewrites(path: Path = TEXT_REWRITES_FILE) -> tuple[tuple[str, str], ...]:
    """The character's spelling -> spoken-form rewrites, applied in order by clean_text (<lab>\\indextts\\
    text_rewrites.json: {"rewrites": [{"pattern": regex, "replacement": str, "why": str}]}; no file -> none).
    Bao-Dur's: the game writes the droid's name "G0-T0" with zeros, the actor says "Goto" (as in the other recipes)."""
    path = Path(path)
    if not path.exists():
        return ()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rewrites = tuple((str(entry["pattern"]), str(entry["replacement"])) for entry in doc.get("rewrites") or [])
        for pattern, _ in rewrites:
            re.compile(pattern)
    except (ValueError, KeyError, TypeError, AttributeError, re.error) as exc:
        raise SystemExit(f"{path} is not a text_rewrites file ({type(exc).__name__}: {exc})") from exc
    return rewrites


TEXT_REWRITES: tuple[tuple[str, str], ...] = load_text_rewrites()


def offline_env() -> None:
    """Every Hugging Face / ModelScope component offline, caches inside the IndexTTS install.

    Call before importing torch/transformers/indextts (huggingface_hub reads these at import). infer_v2_5.py sets
    HF_HUB_CACHE='./checkpoints/hf_cache' (relative to the CWD) at import, so callers also chdir to INDEX_ROOT
    (enter_index_root). Values are assigned, not defaulted: this machine has a global HF_HOME
    (E:\\LocalAgentDrive\\Caches\\HuggingFace) that must not be read or written by this install. Process-local only."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str(INDEX_ROOT / ".hf_home")
    os.environ["HF_HUB_CACHE"] = str(AUX_DIR)  # the folder infer_v2_5 itself points HF_HUB_CACHE at
    os.environ["MODELSCOPE_CACHE"] = str(INDEX_ROOT / ".modelscope_cache")


def enter_index_root() -> None:
    """chdir to the IndexTTS checkout so its import-time relative HF_HUB_CACHE resolves inside the install."""
    os.chdir(INDEX_ROOT)


_BLOCKED: list[str] = []


def _loopback(host) -> bool:
    host = str(host or "").strip("[]").lower()
    return host in ("", "localhost", "::1", "0.0.0.0") or host.startswith("127.")


def net_guard() -> list[str]:
    """Make any non-loopback socket connect / DNS lookup from Python code raise (defense in depth on top of the
    *_OFFLINE variables: a missing file must fail loudly, never download). Returns the live list of blocked
    attempts so callers can report them."""
    import socket

    if getattr(socket.socket.connect, "_voice_lab_guard", False):
        return _BLOCKED
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if not _loopback(host):
            _BLOCKED.append(f"connect {address}")
            raise OSError(f"offline guard: network connection to {address} blocked (IndexTTS must run offline)")
        return real_connect(self, address)

    def connect_ex(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if not _loopback(host):
            _BLOCKED.append(f"connect_ex {address}")
            raise OSError(f"offline guard: network connection to {address} blocked (IndexTTS must run offline)")
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _loopback(host):
            _BLOCKED.append(f"dns {host}")
            raise socket.gaierror(f"offline guard: DNS lookup of {host} blocked (IndexTTS must run offline)")
        return real_getaddrinfo(host, *args, **kwargs)

    connect._voice_lab_guard = True  # type: ignore[attr-defined]
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    return _BLOCKED


# --------------------------------------------------------------------------------------------- checkpoints
CFG_KEYS = ("gpt_checkpoint", "s2mel_checkpoint", "w2v_stat", "emo_matrix", "spk_matrix", "qwen_emo_path")


def top_level_cfg(path: Path = CFG_PATH) -> dict[str, str]:
    """The top-level scalar keys of checkpoints\\config.yaml that name files (no OmegaConf needed, so a dry run
    stays light). Missing config -> SystemExit (it is part of the pinned HF snapshot, never fetched at runtime)."""
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing (IndexTTS-2.5 snapshot not installed; see PROVENANCE.json)")
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\S.*?)\s*$", line)
        if m and m.group(1) in CFG_KEYS:
            out[m.group(1)] = m.group(2).strip("'\"")
    missing = [k for k in CFG_KEYS if k not in out]
    if missing:
        raise SystemExit(f"{path} lacks {missing}; not an IndexTTS-2.5 config")
    return out


def checkpoint_paths(cfg: dict | None = None) -> dict[str, Path]:
    """Every file/dir IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR) of infer_v2_5.py opens, resolved exactly
    as its __init__ does (config keys joined onto model_dir; aux models in the flat hf_cache layout)."""
    cfg = cfg or top_level_cfg()
    return {
        "config": CFG_PATH,
        "gpt": MODEL_DIR / cfg["gpt_checkpoint"],
        "s2mel": MODEL_DIR / cfg["s2mel_checkpoint"],
        "semantic_codec": MODEL_DIR / "codec.pth",
        "w2v_stat": MODEL_DIR / cfg["w2v_stat"],
        "emo_matrix": MODEL_DIR / cfg["emo_matrix"],
        "spk_matrix": MODEL_DIR / cfg["spk_matrix"],
        "text_tokenizer": MODEL_DIR / "multilingual_zh_ja_yue_char_del.tiktoken",
        "qwen_emo": MODEL_DIR / str(cfg["qwen_emo_path"]).rstrip("/\\"),
        "w2v_bert_dir": AUX_DIR / "w2v-bert-2.0",
        "w2v_bert_weights": AUX_DIR / "w2v-bert-2.0" / "model.safetensors",
        "w2v_bert_config": AUX_DIR / "w2v-bert-2.0" / "config.json",
        "w2v_bert_preprocessor": AUX_DIR / "w2v-bert-2.0" / "preprocessor_config.json",
        "campplus": AUX_DIR / "campplus_cn_common.bin",
        "bigvgan_dir": AUX_DIR / "bigvgan",
        "bigvgan_weights": AUX_DIR / "bigvgan" / "bigvgan_generator.pt",
        "bigvgan_config": AUX_DIR / "bigvgan" / "config.json",
    }


def downloader_would_run(paths: dict[str, Path]) -> list[str]:
    """The conditions under which infer_v2_5.IndexTTS2.__init__ calls ensure_models_available() (a download)."""
    out = []
    if not paths["w2v_bert_dir"].is_dir():
        out.append(f"{paths['w2v_bert_dir']} is not a directory")
    if not paths["campplus"].is_file():
        out.append(f"{paths['campplus']} is not a file")
    if not paths["bigvgan_dir"].is_dir():
        out.append(f"{paths['bigvgan_dir']} is not a directory")
    return out


def verified_files() -> dict[str, dict]:
    """local path (casefolded) -> fetch-log entry (repo, revision, size, sha256) for every downloaded/copied file."""
    if not FETCH_LOG.exists():
        return {}
    doc = json.loads(FETCH_LOG.read_text(encoding="utf-8"))
    return {str(Path(f["dest"])).casefold(): f for f in doc.get("files", {}).values()}


def verify_sizes() -> list[str]:
    """Cheap integrity check for every model file under MODEL_DIR recorded in the fetch log (size only; the CPU
    smoke test re-hashes). Returns problems (empty = all present with the recorded sizes)."""
    entries = verified_files()
    if not entries:
        return [f"{FETCH_LOG} missing: no record of the fetched model files"]
    problems = []
    for entry in entries.values():
        path = Path(entry["dest"])
        if MODEL_DIR not in path.parents:
            continue  # model cards / license records under _provenance
        if not path.exists():
            problems.append(f"missing {path}")
        elif path.stat().st_size != int(entry["size"]):
            problems.append(f"size of {path} is {path.stat().st_size}, recorded {entry['size']}")
    return problems


# Parameters (millions) the 2.5 modules actually load from the pinned checkpoints, measured by smoke_cpu.py
# (state_dict entries matched by name and shape; revision c39ce5ba + aux models above). codec.pth holds more than
# the 50.6M loaded tensors (its file is 0.61 GB), so file sizes would overstate the codec.
PARAMS_M = {"gpt": 814.85, "w2v_bert": 580.49, "semantic_codec": 50.58, "s2mel": 103.19, "bigvgan": 112.23,
            "qwen_emo": 596.05}


def vram_estimate(bf16: bool = True, qwen_emo: bool = False, max_text_tokens: int = 120,
                  max_mel_tokens: int = 1500, num_beams: int = 3) -> dict:
    """Rough (+/-25%) device-memory estimate for one IndexTTS-2.5 inference process.

    infer_v2_5 keeps the GPT in bf16 when use_bf16=True and everything else in fp32: w2v-bert-2.0 (all 24 layers
    resident although only hidden_states[17] is used), semantic codec, s2mel (DiT + WaveNet + length regulator),
    BigVGAN, CAMPPlus. QwenEmotion (Qwen3-0.6B) is loaded fp16 only for the emotion-text mode. Transients: GPT KV
    cache for num_beams sequences (24 layers x 2 x 1280 x 2 B per token), the CFM decoder (25 steps with CFG) and
    BigVGAN on a ~15 s segment. CUDA context + cuBLAS workspace (~0.6 GB on Windows) is outside the allocator cap.
    Measured peaks are recorded in each manifest's gpu block to recalibrate this (upstream model card: ~6 GB)."""
    parts = {
        "gpt": round(PARAMS_M["gpt"] * (2 if bf16 else 4) / 1e3, 2),
        "w2v_bert_fp32": round(PARAMS_M["w2v_bert"] * 4 / 1e3, 2),
        "semantic_codec_fp32": round(PARAMS_M["semantic_codec"] * 4 / 1e3, 2),
        "s2mel_fp32": round(PARAMS_M["s2mel"] * 4 / 1e3, 2),
        "bigvgan_fp32": round(PARAMS_M["bigvgan"] * 4 / 1e3, 2),
        "campplus_and_matrices": 0.04,
    }
    if qwen_emo:
        parts["qwen_emo_fp16"] = round(PARAMS_M["qwen_emo"] * 2 / 1e3, 2)
    kv_gb = num_beams * (max_text_tokens + max_mel_tokens + 64) * 24 * 2 * 1280 * (2 if bf16 else 4) / 1e9
    transients = {"gpt_kv_cache_worst": round(kv_gb, 2), "cfm_decoder": 0.35, "bigvgan_activations": 0.5,
                  "dit_mask_and_caches": 0.15, "allocator_fragmentation": 0.3}
    if qwen_emo:
        transients["qwen_generate"] = 0.15
    allocator = sum(parts.values()) + sum(transients.values())
    context = 0.6
    return {"weights_gb": parts, "transients_gb": transients, "allocator_gb": round(allocator, 2),
            "cuda_context_gb": context, "total_gb": round(allocator + context, 1),
            "basis": "loaded parameter counts (smoke_cpu.py) x dtype bytes + transient terms; +/-25%; "
                     "upstream model card: ~6 GB"}


# --------------------------------------------------------------------------------------------- corpus
def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_corpus(path: Path = CORPUS) -> list[dict]:
    return read_jsonl(path)


def test_id_pattern(ids: set[str]) -> re.Pattern:
    return re.compile(r"(?<![0-9a-z])(" + "|".join(sorted(map(re.escape, ids))) + r")(?![0-9])", re.I)


def step_hint(script: str) -> str:
    """How to run a delivery-step script for this character (error messages): the default character keeps the
    original wording, any other one also gets the VOICE_LAB_CHARACTER setting it needs."""
    hint = CODE_DIR.parent.name + "\\delivery\\" + script
    return hint if CHAR["slug"] == character.DEFAULT else f"{hint} with VOICE_LAB_CHARACTER={CHAR['slug']}"


def base_delivery(delivery_map: Path | None = None) -> str:
    """The character's neutral delivery label (default identity label; fallback for lines whose delivery is missing or
    unknown): characters\\<slug>.json "base_delivery" if that key exists, else "natural" when the delivery map has it
    (the delivery recipe's label sets always do; Bao-Dur), else "base_label" of the character's emotion_controls.json
    when the map has that label (or there is no map yet), else the map's first label."""
    if CHAR.get("base_delivery"):
        return str(CHAR["base_delivery"]).casefold()
    path = Path(delivery_map or DELIVERY_MAP)
    labels = [k.casefold() for k in json.loads(path.read_text(encoding="utf-8"))["deliveries"]] if path.exists() else []
    if "natural" in labels:
        return "natural"
    doc = json.loads(EMOTION_CONTROLS_FILE.read_text(encoding="utf-8")) if EMOTION_CONTROLS_FILE.exists() else {}
    preferred = str(doc.get("base_label") or "").casefold()
    if preferred and (preferred in labels or not labels):
        return preferred
    return labels[0] if labels else "natural"


def load_emotion_controls(path: Path | None = None, base: Path | None = None) -> dict[str, dict]:
    """Per-delivery IndexTTS emotion controls {label: {"vector": [8 floats ordered EMO_ORDER], "text": str,
    "duration_factor": float}}: the character's table (`base`, default <lab>\\indextts\\emotion_controls.json) with
    the optional --emo-config file `path` merged over it. Both files are {"labels": {label: {"vector", "text",
    "duration_factor"}}}; a label first seen in a file starts neutral (all-zero vector, no text, factor 1.0). Without
    the character's file only `path`'s labels exist (render.py then names the file to write)."""
    base = EMOTION_CONTROLS_FILE if base is None else Path(base)
    table: dict[str, dict] = {}
    for source in (base if base.exists() else None, path):
        if source is None:
            continue
        doc = json.loads(Path(source).read_text(encoding="utf-8"))
        for label, entry in (doc.get("labels") or {}).items():
            slot = table.setdefault(label.casefold(), {"vector": [0.0] * 8, "text": "", "duration_factor": 1.0})
            if "vector" in entry:
                slot["vector"] = [float(x) for x in entry["vector"]]
            if "text" in entry:
                slot["text"] = str(entry["text"])
            if "duration_factor" in entry:
                slot["duration_factor"] = float(entry["duration_factor"])
    for label, entry in table.items():
        vec = entry["vector"]
        if len(vec) != 8 or any(not 0.0 <= x <= 1.2 for x in vec):
            raise SystemExit(f"emotion control '{label}': vector must be 8 values in [0, 1.2] ordered {EMO_ORDER}")
        if not 0.5 <= entry["duration_factor"] <= 2.0:
            raise SystemExit(f"emotion control '{label}': duration_factor must be within 0.5-2.0")
    return table


def load_speaker_screen(path: Path = SPEAKER_SCREEN) -> dict:
    """The delivery task's speaker screen as sets (same reading as voxcpm/common.py). Missing file -> SystemExit."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"speaker screen {path} not found (run {step_hint('exclusions.py')} first)")
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


# --------------------------------------------------------------------------------------------- text
def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.casefold().replace("-", " ")).split()


def clean_text(text: str) -> str:
    """Whitespace/quote cleanup plus TEXT_REWRITES; punctuation (incl. '...' and ' - ') is kept as phrasing."""
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


FUNCTION_WORDS = frozenset("""
a an the and or but if then so to of in on at by for with from as into onto about over under up down out off
is are was were be been being am do does did done have has had having can could will would should shall may
might must i you he she it we they me him her us them my your his its our their mine yours this that these those
there here what which who whom whose when where why how not no yes just all any some more most very too also only
now than again still even well oh uh um yeah okay ok right sure much many
i'm you're it's don't i'll i've i'd that's can't won't let's he's she's we're they're isn't aren't wasn't weren't
didn't doesn't haven't hasn't hadn't couldn't wouldn't shouldn't there's what's you'll you've you'd we'll we've
able get got go going gone know think see like want make
""".split())
# + the character's address term for the player (Bao-Dur: "General"): as common as a function word in his lines, so it
# never makes a phrase count as content in the held-out overlap rule.
STOPWORDS = FUNCTION_WORDS | frozenset(words(str(CHAR.get("address_term") or "")))


def content_phrases(text: str, n: int = 4, min_content: int = 3) -> set[str]:
    tokens = words(text)
    out = set()
    for i in range(len(tokens) - n + 1):
        gram = tokens[i:i + n]
        if sum(1 for t in gram if t not in STOPWORDS) >= min_content:
            out.add(" ".join(gram))
    return out


def overlap_reason(candidate: str, protected: str, jaccard: float = 0.9) -> str | None:
    """Why `candidate` (partly) speaks the line `protected`, or None (same rules as voxcpm/common.py)."""
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
    except ImportError:  # pragma: no cover - soxr is in the IndexTTS venv (librosa dependency)
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


def trim_silence(audio: np.ndarray, rate: int, lead_pad: float = 0.10, tail_pad: float = 0.20,
                 rel_db: float = 45.0, fade_ms: float = 5.0) -> tuple[np.ndarray, dict]:
    """Cut leading/trailing silence, keeping at most lead_pad/tail_pad seconds of it (voxcpm/common.py rules:
    threshold rel_db under the loudest 10 ms frame, >= floor + 6 dB, <= peak - 30 dB; only cuts, never pads)."""
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
    info.update(lead_in=round(speech_start / rate, 3), tail_in=round((len(audio) - speech_end) / rate, 3),
                lead_out=round((speech_start - start) / rate, 3), tail_out=round((end - speech_end) / rate, 3),
                trimmed=bool(start > 0 or end < len(audio)), threshold_db=round(threshold, 1),
                peak_db=round(peak, 1), floor_db=round(floor, 1))
    return out, info


def prepare_reference(row: dict, variant: str, trim: bool, cache_root: Path = REF_CACHE) -> tuple[Path, dict]:
    """The conditioning WAV for corpus row `row`: the source clip itself (trim=False) or a silence-trimmed copy at
    the clip's NATIVE rate (22.05 kHz game audio; IndexTTS resamples internally to 22.05k/16k, so no 16 kHz cache).
    The copy keeps the clip id as its file name (the eval leak check reads ids from paths) and is rebuilt when the
    source hash changes. Never called for test clips (callers refuse them first)."""
    if row.get("split") == "test":
        raise SystemExit(f"refusing to read held-out test clip {row['id']} as reference audio")
    src = source_audio(row, variant)
    if not src.exists():
        raise FileNotFoundError(f"reference audio {src} not found")
    if not trim:
        return src, {"trimmed": False, "source": str(src)}
    dest = Path(cache_root) / variant / f"{row['id']}.wav"
    meta_path = dest.with_suffix(".json")
    src_sha = sha256_file(src)
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_sha256") == src_sha and meta.get("trimmer") == "trim_silence v1":
            return dest, meta
    audio, rate = read_mono(src)
    out, info = trim_silence(audio, rate)
    stats = write_wav_pcm16(dest, out, rate)
    meta = {"id": row["id"], "source": str(src), "source_sha256": src_sha, "variant": variant, "rate": rate,
            "trimmer": "trim_silence v1", "trim": info, **stats}
    atomic_write_json(meta_path, meta)
    return dest, meta


def write_wav_pcm16(path: Path, audio: np.ndarray, rate: int) -> dict:
    """Atomically write mono PCM16; returns level stats. Never clips silently (peaks above 0.999 are rescaled)."""
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scaled = False
    if peak > 0.999:
        audio = audio * (0.999 / peak)
        scaled = True
    path = Path(path)
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
    return {"peak_dbfs": round(20 * np.log10(max(min(peak, 0.999), 1e-9)), 2),
            "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 2), "peak_rescaled": scaled,
            "duration_s": round(len(audio) / rate, 3)}


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
    """os.replace with retries (Windows sharing violations from scanners/indexers/readers)."""
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
    path = Path(path)
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


def stable_seed(*parts) -> int:
    digest = hashlib.blake2s("|".join(map(str, parts)).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


# --------------------------------------------------------------------------------------------- memory
# Peak host COMMIT while IndexTTS2.__init__ builds everything on the CPU before moving each part to the GPU:
# gpt.pth is torch.load-ed fully (3.26 GB fp32) while the fp32 UnifiedVoice exists (3.26 GB) -> ~6.5 GB for the
# GPT step, + Python/torch/transformers/modelscope imports (~2 GB) + CUDA context/driver (~0.5-1 GB). w2v-bert
# (2.32 GB), codec (0.61 GB x2 while loading), s2mel (0.41 GB x2) and BigVGAN (0.45 GB x2) come after the GPT has
# moved to the GPU. Estimate ~10 GB; the measured peak is written to each manifest (gpu.host) to correct it.
HOST_LOAD_GB = {"render": 10.0, "render_qwen": 11.0}
HOST_MARGIN_GB = 2.0
CONTEXT_GB = 0.6


def host_memory() -> dict | None:
    """System-wide memory state (Windows GlobalMemoryStatusEx); commit_free_gb = ullAvailPageFile."""
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
    return {"commit_free_gb": round(status.ullAvailPageFile / 1e9, 2),
            "commit_limit_gb": round(status.ullTotalPageFile / 1e9, 2),
            "phys_free_gb": round(status.ullAvailPhys / 1e9, 2),
            "phys_total_gb": round(status.ullTotalPhys / 1e9, 2)}


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
    """Refuse a large CPU-side model load unless need_gb of commit is free on top of margin_gb for other apps."""
    info = host_memory()
    if info is None:
        return {"checked": False, "reason": "not Windows; commit not checked"}
    own = process_commit() or {}
    info.update(checked=True, need_gb=need_gb, margin_gb=margin_gb, process_commit_before_gb=own.get("commit_gb"))
    required = need_gb - float(own.get("commit_gb") or 0.0) + margin_gb
    info["required_free_gb"] = round(required, 2)
    if info["commit_free_gb"] < required:
        raise SystemExit(
            f"Not enough free host memory (Windows commit) for the {what}: {info['commit_free_gb']:.1f} GB free of "
            f"{info['commit_limit_gb']:.1f} GB, need ~{required:.1f} GB (peak ~{need_gb:.1f} GB while loading on the "
            f"CPU, minus what this process already holds, plus {margin_gb:.1f} GB kept free for other apps). Wait "
            f"for other large loads to finish and retry. {json.dumps(info)}")
    return info


def gpu_headroom_gb(need_gb: float) -> float:
    return round(max(1.0, 0.15 * need_gb), 2)


def gpu_budget(max_vram_gb: float, reserve_gb: float, need_gb: float, context_gb: float = CONTEXT_GB):
    """Allocator cap for the shared GPU (same policy as voxcpm/common.py). Returns (cap_total_gb, info) or raises
    SystemExit. The cap leaves reserve_gb free for Substance Painter / other sessions, never exceeds max_vram_gb in
    total (allocator + context), and the job starts only when cap >= need + headroom. Must run before anything is
    allocated on the GPU."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; rendering is a GPU step (use --dry-run for CPU checks).")
    torch.cuda.init()
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    cap_total = min(max_vram_gb, free_gb + context_gb - reserve_gb - 0.25)
    headroom = gpu_headroom_gb(need_gb)
    info = {"device": torch.cuda.get_device_name(0), "total_gb": round(total_gb, 2),
            "free_at_start_gb": round(free_gb, 2), "reserve_for_others_gb": reserve_gb, "max_vram_gb": max_vram_gb,
            "estimated_need_gb": need_gb, "required_headroom_gb": headroom, "cap_total_gb": round(cap_total, 2)}
    if cap_total < need_gb + headroom:
        raise SystemExit(
            "Not enough free VRAM to run without squeezing the shared GPU: "
            f"free {free_gb:.1f} GB, must leave {reserve_gb:.1f} GB free, estimated need {need_gb:.1f} GB "
            f"+ {headroom:.1f} GB headroom (cap would be {cap_total:.1f} GB). Wait for other GPU jobs to finish "
            f"and retry; do not lower --reserve-gb below what Painter needs. {json.dumps(info)}")
    alloc_cap = max(1.0, cap_total - context_gb)
    torch.cuda.set_per_process_memory_fraction(min(1.0, alloc_cap / total_gb), 0)
    info["allocator_cap_gb"] = round(alloc_cap, 2)
    return cap_total, info
