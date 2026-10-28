"""Shared paths and helpers for the headless per-character GPT-SoVITS (v2ProPlus / v2Pro) recipe.

Character: chosen by VOICE_LAB_CHARACTER through voice-lab/characters/character.py (default "baodur", so a command
without the variable behaves exactly as it did when the recipe served Bao-Dur only). Everything per character
comes from that config or from the character's own lab folder <lab> (config key "lab"):
  * data: <lab>/corpus.jsonl, <lab>/delivery/ (speaker screen, delivery map, takes script), <lab>/gsv/ (lists,
    experiments, weights, selection), <lab>/takes/gsv/;
  * names: list speaker label, experiment names (<speaker>_<version>_<audio>[_trval]) and list names from the
    config's "speaker", log/notes text from "display_name";
  * content: pronunciation lexicon <lab>/gsv/lexicon.json, transcript corrections <lab>/gsv/text_overrides.json,
    text_check.py's default clips <lab>/gsv/text_check_ids.json (each character keeps its own files).
The BAODUR_LAB_DATA / BAODUR_EXCLUSIONS overrides only apply while the character is baodur.

Module import is stdlib-only, so the orchestrators (export_dataset.py, prepare.py, train.py) can run under any
Python 3.10+. Functions that touch torch or the GPT-SoVITS text frontend import lazily and must run inside the
GPT-SoVITS venv (GSV_PY) with bootstrap_gsv_runtime() called first. Child processes inherit the environment, so
every step of one run serves the same character.

Replicates what GPT-SoVITS webui.py does without the webui:
  * webui writes a users.pth into site-packages to put the repo on sys.path; we never touch the venv and use
    PYTHONPATH (gsv_pythonpath) + cwd = GSV_ROOT instead (the upstream scripts resolve relative model paths
    and write their torch.save temp files in the cwd). Every path this recipe hands to upstream is absolute.
  * webui passes each dataset step its settings through environment variables (gsv_env + prepare.py).

Safety rails that every entry point shares:
  * HeldOut: the 10 split=="test" clips are never exported for training and never used as prompts, matched by
    id, file stem (original and restored copies share it) and sha256 of both copies.
  * Exclusions: clips that sit in the character's clip set but are spoken by another character (the delivery
    task's speaker screen, <lab>/delivery/exclude_other_speakers.json; for Bao-Dur: HK-47, G0-T0 and Atton lines
    in his gbl files), unconfirmed speaker outliers, and train/val lines that give away a held-out test line never
    train the model and never prompt it.
  * Original audio only (<lab>/DECISIONS.md, user decision 2026-09-30, every character): training data and prompt
    clips are the decoded game audio. Restored (ClearerVoice) copies are refused unless --allow-restored is passed.
  * CPU mode hides the GPU with CUDA_VISIBLE_DEVICES=-1 (never an empty string: on Windows os.environ[k] = ""
    deletes the variable, which would re-expose the GPU) and asserts torch sees no CUDA device.
  * GPU mode never re-exposes a GPU the caller hid: an inherited CUDA_VISIBLE_DEVICES of "" or "-1" (how a
    CPU-only workflow hides the GPU) is a hard error; the variable is only filled in when it is absent.
  * GPU mode caps the caching allocator (set_per_process_memory_fraction) and refuses to start unless at least
    `reserve_gb` (default 8 GB, Substance Painter's headroom) stays free after our cap plus CUDA context.
  * Host commit guard: on Windows every process (and every spawned DataLoader worker, which re-imports torch:
    measured 1.48 GB private bytes for `import torch`, 2.36 GB with the SoVITS data loader's imports) is
    charged against the system commit limit. Each heavy step first checks that its estimated commit plus a
    margin is available (GlobalMemoryStatusEx.ullAvailPageFile) and waits / refuses (exit 5) otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Redirected Windows consoles default to cp1252; tqdm bars and upstream logs contain characters it cannot encode,
# which would otherwise crash an orchestrator mid-run while it echoes a child's output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# ----------------------------------------------------------------------------------------------- character
LAB_CODE = Path(__file__).resolve().parent
VOICE_LAB = LAB_CODE.parent.parent  # .../voice-lab: characters/ sits next to baodur-lab/ and the engine repos
_CHARACTERS_DIR = str(VOICE_LAB / "characters")
if _CHARACTERS_DIR not in sys.path:
    sys.path.insert(0, _CHARACTERS_DIR)
import character as _character  # noqa: E402  (voice-lab/characters/character.py, stdlib only)

CHARACTER = _character.load()  # VOICE_LAB_CHARACTER, default "baodur"
CHARACTER_SLUG = CHARACTER["slug"]
DISPLAY_NAME = CHARACTER["display_name"]
SPEAKER = CHARACTER["speaker"]  # list speaker label and the prefix of list / experiment names
if not re.fullmatch(r"[a-z0-9_]+", SPEAKER or ""):
    raise SystemExit(f"character {CHARACTER_SLUG}: speaker {SPEAKER!r} must be lower-case letters, digits or '_' "
                     f"(it names lists, experiments and weights, and is a field of the '|'-separated .list)")
# Possessive pronoun for notes written into takes manifests (config key "pronoun_possessive", optional).
POSSESSIVE = CHARACTER.get("pronoun_possessive") or "his"
IS_DEFAULT_CHARACTER = CHARACTER_SLUG == _character.DEFAULT


def _baodur_override(name: str) -> str | None:
    """The BAODUR_* path overrides predate the per-character config: honoured for Bao-Dur only, never applied to
    (and so never mixing Bao-Dur data into) another character's run."""
    value = os.environ.get(name)
    if value and CHARACTER_SLUG != "baodur":
        print(f"[gsv] ignoring {name}: BAODUR_* overrides only apply while VOICE_LAB_CHARACTER is baodur "
              f"(selected: {CHARACTER_SLUG})", file=sys.stderr, flush=True)
        return None
    return value or None


# ----------------------------------------------------------------------------------------------- paths
GSV_ROOT = Path(os.environ.get("VOICE_LAB_GSV_ROOT") or os.environ.get("BAODUR_GSV_ROOT")
                or r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab\GPT-SoVITS")  # shared engine repo
GSV_PY = GSV_ROOT / ".venv" / "Scripts" / "python.exe"
GSV_PKG = GSV_ROOT / "GPT_SoVITS"
DATA = Path(_baodur_override("BAODUR_LAB_DATA") or CHARACTER["lab"])
CORPUS = DATA / "corpus.jsonl"
DECISIONS = DATA / "DECISIONS.md"
RESTORED_DIR = DATA / "restored"
GSV_DATA = DATA / "gsv"
LISTS_DIR = GSV_DATA / "lists"
EXP_ROOT = GSV_DATA / "exp"
WEIGHTS_ROOT = GSV_DATA / "weights"
SELECTION = GSV_DATA / "selection.json"
TAKES_ROOT = DATA / "takes" / "gsv"
DELIVERY_MAP = DATA / "delivery" / "delivery_map.json"
TAKES_SCRIPT = DATA / "delivery" / "takes_script.json"
EXCLUSIONS = Path(_baodur_override("BAODUR_EXCLUSIONS") or DATA / "delivery" / "exclude_other_speakers.json")
# Per-character content (each character's own files; Bao-Dur's are the files this recipe used before it served
# several characters, moved unchanged from the code folder on 2026-09-30).
LEXICON = GSV_DATA / "lexicon.json"
TEXT_OVERRIDES = GSV_DATA / "text_overrides.json"
TEXT_CHECK_IDS = GSV_DATA / "text_check_ids.json"
REEXEC_ENV = "VOICE_LAB_GSV_REEXEC"  # loop guard for scripts that re-run themselves with the GPT-SoVITS venv python

PRETRAINED = GSV_PKG / "pretrained_models"
BERT_DIR = PRETRAINED / "chinese-roberta-wwm-ext-large"
HUBERT_DIR = PRETRAINED / "chinese-hubert-base"
SV_CKPT = PRETRAINED / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
S1_PRETRAINED = PRETRAINED / "s1v3.ckpt"
S1_BASE_CONFIG = GSV_PKG / "configs" / "s1longer-v2.yaml"
VERSIONS = ("v2ProPlus", "v2Pro")
S2_PRETRAINED = {
    "v2ProPlus": (PRETRAINED / "v2Pro" / "s2Gv2ProPlus.pth", PRETRAINED / "v2Pro" / "s2Dv2ProPlus.pth"),
    "v2Pro": (PRETRAINED / "v2Pro" / "s2Gv2Pro.pth", PRETRAINED / "v2Pro" / "s2Dv2Pro.pth"),
}
S2_BASE_CONFIG = {v: GSV_PKG / "configs" / f"s2{v}.json" for v in VERSIONS}
# v2Pro / v2ProPlus add forward3() of this speaker-verification network (ERes2NetV2, SV_CKPT) to the SoVITS timbre
# embedding of every reference. The evaluation's sv_eres2net metric uses the same network, so takes declare it
# (params.conditioning_encoders) and the aggregator can score their identity with uncoupled models only.
CONDITIONING_ENCODERS = {"v2ProPlus": ["eres2netv2"], "v2Pro": ["eres2netv2"]}

LANG = "en"
EXACT_SOURCES = ("tlk-exact", "dlg")
AUDIO_MODES = ("original", "restored")
DEFAULT_SPLITS = ("train",)  # val is held out (like Seed-VC / VoxCPM): ground truth for checkpoint selection
# TTS._set_prompt_semantic rejects prompts outside 48000..160000 samples at 16 kHz (3-10 s); keep a hair inside.
REF_MIN_S, REF_MAX_S = 3.02, 9.98
NATIVE_SR = 32000  # v2Pro/v2ProPlus SoVITS output rate
OUT_SR = 48000     # shared takes format

# Host commit estimates (GB) per process. Measured on this machine: `import torch` 1.48 GB private, + the SoVITS
# loader imports 2.36 GB (= one spawned DataLoader worker); CPU smoke peaks: SoVITS trainer 6.0 GB working set /
# 9.0 GB commit (fp32 G+D and AdamW states on the CPU; on the GPU those live in VRAM), GPT trainer 4.6 GB working
# set / 5.8 GB commit (6 batches incl. one ScaledAdam step), inference 4.4 GB working set / 7.7 GB commit (every
# model in fp32 on the CPU), prep step 1 2.7 GB. GPU processes add the CUDA runtime's host side.
COMMIT_GB = {"s2_main": 7.0, "s2_worker": 2.4, "s1_main": 6.0, "s1_worker": 2.4, "infer": 6.0,
             "prep_text": 4.0, "prep_hubert": 3.0, "prep_sv": 3.0, "prep_semantic": 3.0}
COMMIT_MARGIN_GB = 2.0  # left free for the other apps sharing this machine


# ----------------------------------------------------------------------------------------------- small utils
_LOG_TAG = "gsv" if IS_DEFAULT_CHARACTER else f"gsv {CHARACTER_SLUG}"  # other characters' runs say whose they are


def log(msg: str) -> None:
    print(f"[{_LOG_TAG} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reexec_in_gsv_venv(script: str | Path) -> None:
    """Re-run `script` (same argv, same environment, hence the same character) with the GPT-SoVITS venv python
    unless this process already is that python; exits with the child's exit code."""
    if Path(sys.executable).resolve() == GSV_PY.resolve() or os.environ.get(REEXEC_ENV) \
            or os.environ.get("BAODUR_GSV_REEXEC"):  # the guard's name before the recipe served several characters
        return
    import subprocess

    sys.exit(subprocess.call([str(GSV_PY), str(Path(script).resolve()), *sys.argv[1:]],
                             env=dict(os.environ, **{REEXEC_ENV: "1"})))


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def abspath(value: str | Path | None) -> Path | None:
    """Resolve a CLI path against the caller's cwd before anything chdirs to the GPT-SoVITS root."""
    return None if value in (None, "") else Path(value).expanduser().resolve()


def exp_name(version: str, audio: str, splits=DEFAULT_SPLITS) -> str:
    if version not in VERSIONS:
        raise SystemExit(f"unsupported version {version!r}; choose one of {VERSIONS}")
    if audio not in AUDIO_MODES:
        raise SystemExit(f"unsupported audio mode {audio!r}; choose one of {AUDIO_MODES}")
    return f"{SPEAKER}_{version}_{audio}{splits_suffix(splits)}"


def splits_suffix(splits) -> str:
    """'' for the default train-only data, '_trval' for train+val, '_<a>-<b>' otherwise."""
    s = sorted({x.strip() for x in splits if x and x.strip()})
    if s == sorted(DEFAULT_SPLITS):
        return ""
    if s == ["train", "val"]:
        return "_trval"
    return "_" + "-".join(s)


def default_list_path(audio: str, splits=DEFAULT_SPLITS) -> Path:
    return LISTS_DIR / f"{SPEAKER}_{audio}{splits_suffix(splits)}.list"


def parse_splits(text: str) -> tuple[str, ...]:
    splits = tuple(sorted({s.strip() for s in (text or "").split(",") if s.strip()}))
    if "test" in splits:
        raise SystemExit("refusing: split 'test' is held out for evaluation and never used for training")
    bad = [s for s in splits if s not in ("train", "val")]
    if bad or "train" not in splits:
        raise SystemExit(f"bad --splits {text!r}; use train or train,val")
    return splits


def require_original_audio(kind: str, allow: bool, what: str) -> None:
    """The user's decision (DECISIONS.md, 2026-09-30): original game audio for training data AND prompt clips."""
    if kind == "original":
        return
    if not allow:
        raise SystemExit(f"refusing {what} with {kind} audio: per {DECISIONS} the user chose original game audio "
                         f"for every engine's training data and reference/prompt clips ('restored/' is never "
                         f"training or reference input). Pass --allow-restored only if that decision changes.")
    log(f"WARNING: --allow-restored: {what} uses {kind} audio against the recorded decision in {DECISIONS}")


def weights_dirs(name: str, weights_root: Path = WEIGHTS_ROOT, gpt_tag: str = "") -> tuple[Path, Path]:
    """(SoVITS dir, GPT dir) for an experiment; gpt_tag separates GPT arms (e.g. '_dpo')."""
    return weights_root / name / "SoVITS", weights_root / name / f"GPT{gpt_tag}"


def check_tag(tag: str) -> str:
    tag = tag or ""
    if tag and not re.fullmatch(r"_[A-Za-z0-9]+", tag):
        raise SystemExit(f"bad tag {tag!r}: use '' or an underscore plus letters/digits, e.g. _dpo")
    return tag


_UPSTREAM_TMP = re.compile(r"^\d{9,}\.\d+\.pth$")  # upstream my_save(): torch.save("<time()><part>.pth") then move


def upstream_tmp_names() -> set[str]:
    try:
        return {p.name for p in GSV_ROOT.iterdir() if p.is_file() and _UPSTREAM_TMP.match(p.name)}
    except OSError:
        return set()


def sweep_upstream_tmp(before: set[str], since: float, dest: Path, settle_s: float = 5.0) -> list[str]:
    """Upstream saves write a temp file into the cwd (the repo root) and then move it; if the move fails the file
    stays in the repo. Move such files out of the repo into `dest` - only names that did not exist before our
    child started, were written after it started, and have not changed for `settle_s` seconds (another
    GPT-SoVITS process using the same repo moves its own temp file within a moment of writing it; still, no
    other GPT-SoVITS job should run from this repo while a recipe step runs)."""
    moved = []
    candidates = [n for n in upstream_tmp_names() - set(before)]
    deadline = time.time() + 2 * settle_s
    for name in candidates:
        p = GSV_ROOT / name
        while p.exists() and time.time() - p.stat().st_mtime < settle_s and time.time() < deadline:
            time.sleep(0.5)
        try:
            st = p.stat()
        except OSError:
            continue  # its owner moved it
        if st.st_mtime < since - 2 or time.time() - st.st_mtime < settle_s:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(p, dest / name)
            moved.append(str(dest / name))
        except OSError:
            pass
    return moved


def run_logged(cmd: list, env: dict, log_path: Path, cwd: Path | None = None) -> int:
    """Run a child (default cwd: the GPT-SoVITS root), echoing its output and appending it to log_path.
    Afterwards, stray upstream temp checkpoints from this run are moved out of the repo next to the log."""
    import subprocess

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = upstream_tmp_names()
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(str(c) for c in cmd)}\n")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd or GSV_ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        code = proc.wait()
        moved = sweep_upstream_tmp(before, started, log_path.parent / "stray_upstream_tmp")
        if moved:
            note = f"[gsv] moved {len(moved)} stray upstream temp file(s) out of the repo: {moved}\n"
            sys.stdout.write(note)
            log_file.write(note)
        return code


def archive(path: Path, tag: str = "stale") -> Path | None:
    """Move a file we generated earlier out of the way instead of deleting it."""
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.{tag}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(path, target)
    return target


# ----------------------------------------------------------------------------------------------- corpus
def load_corpus(path: Path = CORPUS) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def restored_path(clip_id: str) -> Path:
    return RESTORED_DIR / f"{clip_id}.wav"


class HeldOut:
    """The split=="test" clips: ground truth for evaluation only (never training data, never a prompt)."""

    def __init__(self, rows: list[dict]):
        test = [r for r in rows if r.get("split") == "test"]
        if not test:
            raise SystemExit("corpus has no split=='test' rows; refusing to run without the held-out guard")
        self.ids = {r["id"].casefold() for r in test}
        self.stems = set(self.ids) | {Path(r["audio"]).stem.casefold() for r in test}
        self.sha = {str(r.get("sha256", "")).lower() for r in test if r.get("sha256")}
        for r in test:  # restored copies have their own hashes
            p = restored_path(r["id"])
            if p.exists():
                self.sha.add(sha256_file(p))
        self._sha_cache: dict[str, str] = {}

    def reason(self, clip_id: str | None = None, path: str | Path | None = None) -> str | None:
        """Why this clip/path is held out (None when it is safe to use)."""
        if clip_id and str(clip_id).casefold() in self.ids:
            return f"clip id {clip_id} is a held-out test clip"
        if path:
            p = Path(path)
            if p.stem.casefold() in self.stems:
                return f"file {p.name} is a held-out test clip"
            if p.exists():
                key = str(p.resolve()).casefold()
                if key not in self._sha_cache:
                    self._sha_cache[key] = sha256_file(p)
                if self._sha_cache[key] in self.sha:
                    return f"file {p} has the sha256 of a held-out test clip"
        return None


def norm_words(text: str) -> list[str]:
    text = (text or "").casefold().replace("’", "'").replace("-", " ")
    return re.sub(r"[^a-z0-9' ]", " ", text).split()


def _longest_common_run(a: list[str], b: list[str]) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def sentence_keys(text: str, min_sentence_words: int = 4) -> list[str]:
    sentences = [" ".join(norm_words(s)) for s in re.split(r"[.!?]+", text or "")]
    return [s for s in sentences if len(s.split()) >= min_sentence_words]


def overlap_reason(words: list[str], other_words: list[str], other_sentences: list[str],
                   jaccard: float = 0.5, run_words: int = 6) -> str | None:
    """Why a line (words) gives away another line: it contains one of its sentences (>= 4 words), shares most of its
    vocabulary (word Jaccard) or a run of consecutive words. None when the two lines are unrelated."""
    if not words or not other_words:
        return None
    joined, vocab, other_vocab = " ".join(words), set(words), set(other_words)
    jac = len(vocab & other_vocab) / max(1, len(vocab | other_vocab))
    if any(f" {s} " in f" {joined} " for s in other_sentences):
        return "contains a whole sentence of"
    if jac >= jaccard:
        return f"shares {jac:.0%} of its words with"
    if _longest_common_run(words, other_words) >= run_words:
        return f"shares a run of >= {run_words} words with"
    return None


def test_line_leaks(rows: list[dict], jaccard: float = 0.5, run_words: int = 6, min_sentence_words: int = 4) -> dict:
    """train/val clips whose words give away a held-out test line: they contain a whole test sentence
    (>= 4 words), share most of its vocabulary (word Jaccard >= 0.5) or a run of >= 6 consecutive words.
    Training on such a clip, or prompting with it, would leak the held-out performance into the evaluation.
    E.g. in Bao-Dur's corpus: gblbaodur218 (same line as test 200) and 273/274/277/278 (variants of test 276)."""
    tests = [(r["id"], norm_words(r.get("text")), sentence_keys(r.get("text"), min_sentence_words))
             for r in rows if r.get("split") == "test"]
    leaks = {}
    for r in rows:
        if r.get("split") not in ("train", "val"):
            continue
        words = norm_words(r.get("text"))
        for test_id, twords, tsentences in tests:
            why = overlap_reason(words, twords, tsentences, jaccard, run_words)
            if why:
                leaks[r["id"]] = f"text {why} held-out test line {test_id}"
                break
    return leaks


class Exclusions:
    """Clips that must neither train nor prompt the character's model (in addition to the held-out test split).

    From EXCLUSIONS (<lab>/delivery/exclude_other_speakers.json, written by the delivery task's speaker screen):
    'confirmed' lines by other characters that live in this character's clip set (for Bao-Dur: HK-47, G0-T0 and
    Atton in his gbl files), 'uncertain' embedding outliers nobody has listened to yet, 'near_duplicate_of_test'
    lines; plus our own test_line_leaks(). 'suspect_short_lines' (for Bao-Dur mostly his own barks and strained
    deliveries) are kept unless exclude_suspect. Missing file = hard error unless required=False (the caller then
    says so loudly)."""

    def __init__(self, rows: list[dict], path: Path | str | None = EXCLUSIONS, *, include_uncertain: bool = False,
                 include_near_duplicates: bool = False, exclude_suspect: bool = False, required: bool = True):
        self.reasons: dict[str, str] = {}
        self.suspect: set[str] = set()
        self.path = str(path) if path else None
        self.sha256 = None
        self.file_used = False
        self.options = {"include_uncertain": include_uncertain, "include_near_duplicates": include_near_duplicates,
                        "exclude_suspect": exclude_suspect}
        if path is not None:
            path = Path(path)
            if path.exists():
                data = read_json(path)
                self.sha256, self.file_used = sha256_file(path), True
                speaker = {}
                for name, items in (data.get("confirmed") or {}).items():
                    for item in items or []:
                        speaker[item["id"] if isinstance(item, dict) else item] = name
                confirmed = set(data.get("confirmed_ids") or []) | set(speaker)
                # Format check: the screen must say who else speaks; an empty answer is valid (e.g. Jolee's clip set
                # holds no other voices), a file without the keys is not.
                if not confirmed and "confirmed_ids" not in data and "confirmed" not in data:
                    raise SystemExit(f"{path}: no 'confirmed_ids'/'confirmed' keys; unexpected format, refusing")
                for clip in confirmed:
                    self.reasons[clip] = f"spoken by {speaker.get(clip, 'another character')} (speaker screen)"
                if not include_uncertain:
                    for item in data.get("uncertain") or []:
                        clip = item["id"] if isinstance(item, dict) else item
                        self.reasons.setdefault(clip, "uncertain speaker (screen outlier nobody has listened to)")
                if not include_near_duplicates:
                    for clip, why in (data.get("near_duplicate_of_test") or {}).items():
                        self.reasons.setdefault(clip, f"near-duplicate of a held-out test line: {why}")
                self.suspect = set(data.get("suspect_short_lines") or [])
                if exclude_suspect:
                    for clip in self.suspect:
                        self.reasons.setdefault(clip, "speaker screen 'suspect' (--exclude-suspect)")
            elif required:
                raise SystemExit(f"speaker-screen exclusions not found: {path}. Without it, lines by other characters "
                                 f"that sit in {DISPLAY_NAME}'s clip set would train the model. Run the delivery "
                                 f"task's speaker screen (exclusions.py) for {CHARACTER_SLUG}, pass --exclusions "
                                 f"<file>, or --no-exclusions to accept the risk.")
        if not include_near_duplicates:
            for clip, why in test_line_leaks(rows).items():
                self.reasons.setdefault(clip, why)

    def reason(self, clip_id: str | None) -> str | None:
        return self.reasons.get(clip_id) if clip_id else None

    def describe(self) -> dict:
        return {"file": self.path, "file_used": self.file_used, "sha256": self.sha256, **self.options,
                "excluded_ids": len(self.reasons), "suspect_ids_kept": sorted(self.suspect - set(self.reasons))}


def screen_training_ids(ids, rows: list[dict], exclusions_path: Path | str | None = EXCLUSIONS) -> dict:
    """Last line of defence before features are extracted or a model is trained on a set of clip ids.
    'heldout' and 'other_speaker' entries must stop the caller; 'screened' (uncertain speakers, lines that give
    away a test line) are only reported, because export_dataset.py can include them on purpose.
    exclusions_path=None skips the speaker checks (the caller warns)."""
    guard = HeldOut(rows)
    hard = Exclusions(rows, exclusions_path, include_uncertain=True, include_near_duplicates=True) \
        if exclusions_path else None
    soft = Exclusions(rows, exclusions_path) if exclusions_path else None
    out: dict = {"heldout": [], "other_speaker": [], "screened": {}}
    for clip in ids:
        if guard.reason(clip):
            out["heldout"].append(clip)
        elif hard is not None and hard.reason(clip):
            out["other_speaker"].append(f"{clip} ({hard.reason(clip)})")
        elif soft is not None and soft.reason(clip):
            out["screened"][clip] = soft.reason(clip)
    return out


def trained_clip_ids(exp_dir: Path) -> set[str] | None:
    """Clip ids an experiment's models were trained on (2-name2text.txt, i.e. what both trainers read);
    None when the experiment folder has no text table."""
    table = Path(exp_dir) / "2-name2text.txt"
    if not table.exists():
        return None
    return {Path(ln.split("\t")[0]).stem for ln in table.read_text(encoding="utf-8").splitlines()
            if ln.count("\t") == 3}


# ----------------------------------------------------------------------------------------------- transcripts
def load_text_overrides(path: Path | str | None = TEXT_OVERRIDES, rows: list[dict] | None = None) -> dict:
    """Transcript corrections where the actor's words differ from the corpus script line (<lab>/gsv/
    text_overrides.json, evidence in <lab>/gsv/text_check.json). {"overrides": {id: text}, "sha256", "path"};
    an entry whose recorded 'script' no longer matches the corpus is stale and refused. Never applies to held-out
    test clips. A missing file means no corrections (a new character starts without any)."""
    empty = {"overrides": {}, "sha256": None, "path": None}
    if path is None or not Path(path).exists():
        return empty
    raw = read_json(path)
    by_id = {r["id"]: r for r in (rows or [])}
    out = {}
    for clip, entry in (raw.get("overrides") or {}).items():
        text = entry["text"] if isinstance(entry, dict) else str(entry)
        if rows is not None:
            row = by_id.get(clip)
            if row is None:
                raise SystemExit(f"{path}: {clip} is not a corpus clip")
            if row.get("split") == "test":
                raise SystemExit(f"{path}: {clip} is a held-out test clip; its transcript is the evaluation's truth")
            script = entry.get("script") if isinstance(entry, dict) else None
            if script is not None and script != row.get("text"):
                raise SystemExit(f"{path}: override for {clip} was written for script {script!r} but the corpus now "
                                 f"says {row.get('text')!r}; re-check it")
        out[clip] = clean_line_text(text)
    return {"overrides": out, "sha256": sha256_file(path), "path": str(Path(path))}


def apply_text_overrides(rows: list[dict], overrides: dict) -> list[dict]:
    """Rows with 'text' replaced by the override (the corpus script kept in 'script_text')."""
    table = overrides.get("overrides") or {}
    out = []
    for r in rows:
        if r["id"] in table and table[r["id"]] != r.get("text"):
            r = {**r, "script_text": r.get("text"), "text": table[r["id"]], "text_override": True}
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------- lexicon
def require_lexicon(path: Path | str | None) -> Path | None:
    """The lexicon path, or a clear stop when the character has none yet (None = --no-lexicon)."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"pronunciation lexicon not found: {path}. Each character keeps its own <lab>/gsv/lexicon.json "
                         f"(respellings + ARPAbet for names the CMU dictionary lacks; g2p_audit.py --no-lexicon "
                         f"lists them); create it, pass --lexicon <file>, or --no-lexicon for raw text.")
    return path


def load_lexicon(path: Path | str | None = LEXICON) -> dict:
    """Pronunciation fixes: regex respellings applied to raw text, plus ARPAbet entries injected into the
    in-memory CMU dictionary of GPT-SoVITS' English g2p (repo files such as engdict-hot.rep stay untouched)."""
    if path is None:
        return {"respell": [], "arpabet": {}, "sha256": None, "path": None}
    path = require_lexicon(path)
    raw = read_json(path)
    respell = []
    for entry in raw.get("respell", []):
        respell.append((re.compile(entry["pattern"]), entry["replace"]))
    arpabet = {}
    for word, phones in raw.get("arpabet", {}).items():
        if word.startswith("_"):
            continue
        arpabet[word.casefold()] = phones.split() if isinstance(phones, str) else list(phones)
    return {"respell": respell, "arpabet": arpabet, "sha256": sha256_file(path), "path": str(path)}


def respell(text: str, lexicon: dict) -> str:
    for pattern, replacement in lexicon.get("respell", []):
        text = pattern.sub(replacement, text)
    return text


def clean_line_text(text: str) -> str:
    """One-line, list-safe text ('|' separates .list fields)."""
    text = text.replace("|", ",").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s{2,}", " ", text).strip()


def install_g2p_lexicon(lexicon: dict) -> int:
    """Inject ARPAbet overrides into text.english._g2p.cmu (GPT-SoVITS runtime must be importable)."""
    if not lexicon.get("arpabet"):
        return 0
    from text import symbols2  # noqa: E402  (GPT-SoVITS package)
    import text.english as english  # noqa: E402

    valid = set(symbols2.symbols)
    for word, phones in lexicon["arpabet"].items():
        bad = [p for p in phones if p not in valid]
        if bad:
            raise ValueError(f"lexicon entry {word!r} has phones outside GPT-SoVITS symbols: {bad}")
        english._g2p.cmu[word] = [list(phones)]
    return len(lexicon["arpabet"])


# ----------------------------------------------------------------------------------------------- runtime / env
def gsv_pythonpath() -> list[str]:
    """What webui.py's users.pth adds (minus tools/asr and tools/uvr5, which this recipe never uses)."""
    return [str(GSV_ROOT), str(GSV_PKG / "BigVGAN"), str(GSV_ROOT / "tools"), str(GSV_PKG)]


def normalize_device(device: str) -> str:
    device = (device or "cpu").strip().lower()
    if device == "gpu":
        device = "cuda"
    if device != "cpu" and not re.fullmatch(r"cuda(:\d+)?", device):
        raise SystemExit(f"bad --device {device!r}; use cpu, cuda or cuda:N")
    return device


def cuda_index(device: str) -> str:
    return device.split(":", 1)[1] if ":" in device else "0"


def gpu_visible_devices(device: str, environ: dict | None = None) -> str:
    """CUDA_VISIBLE_DEVICES for a GPU-mode process. An inherited "" or "-1" means the caller hid the GPU on
    purpose (that is how CPU-only workflows run): refuse instead of re-exposing it. When the variable is absent,
    the physical index of --device is used; an inherited id list is kept as is (then --device must be plain
    'cuda' or 'cuda:0', the first visible GPU)."""
    device = normalize_device(device)
    if device == "cpu":
        raise ValueError("gpu_visible_devices() is for GPU mode")
    env = os.environ if environ is None else environ
    inherited = env.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return cuda_index(device)
    value = inherited.strip()
    first = value.split(",")[0].strip()
    if not first or first.startswith("-"):
        raise SystemExit(f"--device {device} requested, but CUDA_VISIBLE_DEVICES={inherited!r} was inherited: the "
                         f"caller hid the GPU (a CPU-only run). Refusing to re-expose it. Unset the variable or set "
                         f"it to the GPU id to train/render on the GPU, or pass --device cpu.")
    if cuda_index(device) != "0":
        raise SystemExit(f"--device {device} with inherited CUDA_VISIBLE_DEVICES={inherited!r} is ambiguous (device "
                         f"indices are relative to the visible set); use --device cuda or unset the variable")
    return value


def gsv_env(device: str, threads: int = 8, extra: dict | None = None) -> dict:
    """Environment for a GPT-SoVITS child process (dataset step or training wrapper)."""
    device = normalize_device(device)
    env = dict(os.environ)
    old = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(gsv_pythonpath() + [str(LAB_CODE)] + old)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE", "TORCH_DISTRIBUTED_DEBUG": "INFO", "USE_LIBUV": "0",
        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
    })
    if device == "cpu":
        # Upstream s1_train copies _CUDA_VISIBLE_DEVICES into CUDA_VISIBLE_DEVICES; leave it unset so nothing
        # can overwrite our "-1" (and never use "", which Windows' putenv treats as delete).
        env.pop("_CUDA_VISIBLE_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = env["_CUDA_VISIBLE_DEVICES"] = gpu_visible_devices(device)
    for key, value in (extra or {}).items():
        env[key] = str(value)
    return env


def bootstrap_gsv_runtime(device: str) -> None:
    """For in-process use of GPT-SoVITS (inference, audits, training wrappers). Call before importing torch."""
    device = normalize_device(device)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.pop("_CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible_devices(device)
    for key, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("KMP_DUPLICATE_LIB_OK", "TRUE"),
                       ("PYTHONIOENCODING", "utf-8")):
        os.environ.setdefault(key, value)
    os.chdir(GSV_ROOT)
    for p in reversed(gsv_pythonpath()):
        if p not in sys.path:
            sys.path.insert(0, p)


def assert_device(device: str) -> None:
    """Hard stop if the process can see a GPU in CPU mode, or cannot in GPU mode."""
    import torch

    device = normalize_device(device)
    if device == "cpu":
        if torch.cuda.is_available():
            raise SystemExit("CPU mode requested but torch can see a CUDA device; refusing to continue")
    elif not torch.cuda.is_available():
        raise SystemExit(f"--device {device} requested but torch sees no CUDA device "
                         f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")


def is_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory in any of its forms: torch.OutOfMemoryError, or the RuntimeErrors cuBLAS / cuDNN / the
    CUDA runtime raise when a handle or workspace cannot be allocated under the per-process cap."""
    try:
        import torch

        oom_type = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        if isinstance(exc, oom_type):
            return True
    except Exception:  # pragma: no cover
        pass
    if isinstance(exc, (RuntimeError, MemoryError)):
        text = str(exc)
        return any(token in text for token in ("out of memory", "CUBLAS_STATUS_ALLOC_FAILED",
                                               "CUDNN_STATUS_NOT_ENOUGH_WORKSPACE", "CUDNN_STATUS_ALLOC_FAILED",
                                               "cudaErrorMemoryAllocation"))
    return False


def cap_gpu_memory(mem_gb: float, reserve_gb: float = 8.0, wait_min: float = 0.0, context_gb: float = 0.8) -> dict:
    """Refuse to start unless `reserve_gb` stays free for other GPU users after our cap + `context_gb`, then cap
    this process's caching allocator at `mem_gb` (an allocation beyond it raises torch.OutOfMemoryError instead of
    growing). mem_get_info() already runs after our CUDA context exists; `context_gb` is headroom for what grows
    outside the allocator later (lazily loaded kernels, cuDNN/cuBLAS handles), so the check is conservative.
    Returns a small report dict. Device 0 of the visible set."""
    import torch

    deadline = time.time() + wait_min * 60
    need = (mem_gb + context_gb + reserve_gb) * 2**30
    while True:
        free, total = torch.cuda.mem_get_info(0)
        if free >= need:
            break
        if time.time() >= deadline:
            print(f"[gsv] VRAM guard: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB; this step needs "
                  f"{mem_gb:.1f} GB cap + {context_gb:.1f} GB context while leaving {reserve_gb:.1f} GB free "
                  f"(= {need / 2**30:.1f} GB). Not starting.", flush=True)
            raise SystemExit(4)
        time.sleep(20)
    fraction = min(0.95, mem_gb * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    report = {"total_gb": round(total / 2**30, 2), "free_at_start_gb": round(free / 2**30, 2),
              "cap_gb": mem_gb, "fraction": round(fraction, 4), "reserve_gb": reserve_gb,
              "device_name": torch.cuda.get_device_name(0)}
    print(f"[gsv] VRAM cap {mem_gb:.1f} GB (fraction {fraction:.3f} of {total / 2**30:.1f} GB); "
          f"{free / 2**30:.1f} GB free at start", flush=True)
    return report


def commit_free_gb() -> float | None:
    """Commit this process could still allocate (GlobalMemoryStatusEx.ullAvailPageFile), in GB; None off Windows."""
    if os.name != "nt":
        return None
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPageFile / 2**30


def require_commit(need_gb: float, what: str, wait_min: float = 0.0, margin_gb: float = COMMIT_MARGIN_GB) -> dict:
    """Host commit guard: wait (up to wait_min) until need_gb + margin_gb of commit is available, else exit 5.
    Run it in the parent before a heavy child starts (or before importing torch in-process)."""
    deadline = time.time() + wait_min * 60
    want = need_gb + margin_gb
    while True:
        free = commit_free_gb()
        if free is None or free >= want:
            break
        if time.time() >= deadline:
            print(f"[gsv] host commit guard: {free:.1f} GB of commit available; {what} needs about {need_gb:.1f} GB "
                  f"plus {margin_gb:.1f} GB left for other apps (= {want:.1f} GB). Not starting (exit 5); free "
                  f"memory, lower the DataLoader workers, or wait (--wait-min).", flush=True)
            raise SystemExit(5)
        time.sleep(20)
    report = {"commit_free_gb": None if free is None else round(free, 1), "commit_need_gb": round(need_gb, 1),
              "commit_margin_gb": margin_gb}
    if free is not None:
        log(f"host commit: {free:.1f} GB available, {what} needs ~{need_gb:.1f} GB + {margin_gb:.1f} GB margin")
    return report


def peak_memory_report(device: str) -> dict:
    """Peak allocator usage (GPU) or peak working set / commit (CPU) of this process."""
    out: dict = {}
    try:
        import torch

        if normalize_device(device) != "cpu" and torch.cuda.is_available():
            out["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 3)
            out["cuda_max_reserved_gb"] = round(torch.cuda.max_memory_reserved(0) / 2**30, 3)
    except Exception as exc:  # pragma: no cover - diagnostics only
        out["cuda_error"] = repr(exc)
    try:
        import psutil

        info = psutil.Process().memory_info()
        out["rss_gb"] = round(info.rss / 2**30, 3)
        if hasattr(info, "peak_wset"):
            out["peak_working_set_gb"] = round(info.peak_wset / 2**30, 3)
        if hasattr(info, "peak_pagefile"):
            out["peak_commit_gb"] = round(info.peak_pagefile / 2**30, 3)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------------------------- weights
_SOVITS_EPOCH = re.compile(r"_e(\d+)_s(\d+)\.pth$", re.I)
_GPT_EPOCH = re.compile(r"-e(\d+)\.ckpt$", re.I)


def list_weights(name: str, weights_root: Path = WEIGHTS_ROOT, gpt_tag: str = "") -> dict:
    """{'sovits': [(epoch, path)], 'gpt': [(epoch, path)]} sorted by epoch, then age (newest last), so a stage
    that was re-trained from scratch into the same folder resolves to its newest file for each epoch."""
    sovits_dir, gpt_dir = weights_dirs(name, weights_root, gpt_tag)

    def scan(folder: Path, pattern: str, regex) -> list:
        if not folder.exists():
            return []
        found = [(int(m.group(1)), p.stat().st_mtime, p) for p in folder.glob(pattern) if (m := regex.search(p.name))]
        return [(e, str(p)) for e, _, p in sorted(found, key=lambda t: (t[0], t[1]))]

    return {"sovits": scan(sovits_dir, "*.pth", _SOVITS_EPOCH), "gpt": scan(gpt_dir, "*.ckpt", _GPT_EPOCH)}


def pick_weight(entries: list, epoch: int | None, kind: str) -> str:
    if not entries:
        raise SystemExit(f"no {kind} weights found; train first or pass an explicit path")
    if epoch is None:
        return entries[-1][1]
    matches = [p for e, p in entries if e == epoch]
    if matches:
        return matches[-1]
    raise SystemExit(f"no {kind} weights for epoch {epoch}; available: {sorted({e for e, _ in entries})}")
