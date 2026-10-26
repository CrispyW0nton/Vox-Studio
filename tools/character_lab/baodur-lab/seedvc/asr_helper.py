"""faster-whisper large-v3 WER for selftest.py (run with the VoxCPM2 runtime python; never pip install into it).

  [VOICE_LAB_CHARACTER=<slug>] %LOCALAPPDATA%/VoxStudio/engines/voxcpm2/.venv/Scripts/python.exe asr_helper.py \
      --items items.json --cache <lab>/seedvc/selftest/cache/asr_large-v3.json [--device cpu|cuda]

items.json: [{"wav": path, "text": intended text}, ...]. Results are cached by (WAV sha256, model, compute type),
so reruns only transcribe new files and a GPU float16 pass never reuses CPU int8 transcripts. Decoding and normalisation mirror baodur-lab/eval (language en, beam 5, no VAD, no
conditioning on previous text, the same proper-noun prompt; casefold, hyphens split, fillers dropped), so the
numbers are comparable with the eval harness. CPU uses int8, CUDA float16 (~4-5 GB). CTranslate2 has no hard cap:
selftest.py bounds its caching allocator (CT2_CUDA_CACHING_ALLOCATOR_CONFIG, as eval/run_eval.py) and runs a
watchdog that kills this process if the GPU's free memory drops below Painter's 8 GB reserve.
The proper-noun prompt is the character's asr_prompt (voice-lab/characters/<slug>.json via VOICE_LAB_CHARACTER,
default baodur; selftest.py passes its own environment through). The cache key does not hold the prompt: each
character's cache lives in its own lab, so delete that cache file after changing the character's asr_prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

WHISPER_CACHE = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local")) / "VoxStudio" / "engines" / \
    "voxcpm2" / "cache" / "faster-whisper"
CHARACTERS_CODE = Path(r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab") / "characters"


def _character() -> dict:
    """The VOICE_LAB_CHARACTER config (standard library only, so the hash-locked Vox python can import it)."""
    sys.path.insert(0, str(CHARACTERS_CODE))
    try:
        import character
    finally:
        sys.path.remove(str(CHARACTERS_CODE))
    return character.load()


NAMES_PROMPT = _character().get("asr_prompt") or None
FILLERS = {"uh", "uhh", "um", "umm", "hmm", "hm", "mm", "mmm", "mhm", "er", "erm", "ah", "ahh", "eh", "oh", "huh"}
EQUIVALENTS = {"alright": ["all", "right"], "ok": ["okay"], "mr": ["mister"], "mrs": ["missus"], "dr": ["doctor"],
               "til": ["until"], "till": ["until"], "gonna": ["going", "to"], "wanna": ["want", "to"]}


def norm_words(text: str) -> list[str]:
    t = (text or "").casefold().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[-\u2010-\u2015]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    out: list[str] = []
    for word in t.split():
        word = word.strip("'")
        if word and word not in FILLERS:
            out.extend(EQUIVALENTS.get(word, [word]))
    return out


def edit_distance(ref: list, hyp: list) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--compute-type", default=None)
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    todo = []
    for item in items:
        item["sha256"] = sha256_file(item["wav"])
        item["key"] = f"{item['sha256']}|{args.model}|{compute_type}"
        if item["key"] not in cache:
            todo.append(item)
    print(f"{len(items)} items, {len(todo)} to transcribe ({args.model}, {compute_type}, {args.device})", flush=True)
    if todo:
        if args.device == "cuda":
            import torch  # noqa: F401  (puts torch's cuBLAS DLLs on the search path for CTranslate2, as eval does)
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=args.device, compute_type=compute_type,
                             download_root=str(WHISPER_CACHE), local_files_only=True, cpu_threads=args.threads)
        started = time.time()
        for n, item in enumerate(todo, 1):
            segments, _ = model.transcribe(item["wav"], language="en", beam_size=5, vad_filter=False,
                                           condition_on_previous_text=False, initial_prompt=NAMES_PROMPT)
            cache[item["key"]] = {"hyp": " ".join(s.text.strip() for s in segments).strip(),
                                     "model": args.model, "compute_type": compute_type}
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ({time.time() - started:.0f} s)", flush=True)
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                args.cache.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    for item in items:
        hyp = cache[item["key"]]["hyp"]
        ref = norm_words(item["text"])
        item.update({"hyp": hyp, "wer": edit_distance(ref, norm_words(hyp)) / max(1, len(ref))})
    args.items.write_text(json.dumps(items, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
