"""Intelligibility: faster-whisper transcripts and WER against the intended text, plus a name-pronunciation probe.

Run with the VoxCPM2 runtime python (read-only venv; running it is fine, never pip install into it):
  set CUDA_VISIBLE_DEVICES=-1  (CPU default)
  %LOCALAPPDATA%/VoxStudio/engines/voxcpm2/.venv/Scripts/python.exe metric_asr.py --run-dir <lab>/eval/<run>

Decoding matches how the character's corpus.jsonl was transcribed (language en, beam 5, no VAD filter, no
conditioning on previous text, the same proper-noun initial prompt: the character config's asr_prompt), from the
cached model with local_files_only.
CPU uses int8, CUDA float16 (override with --compute-type). WER uses evalcommon.norm_words (casefold,
hyphens split, punctuation and fillers such as "uh" dropped). The real baseline is the same ASR run on
every held-out test clip against its exact transcript.

Name probe: the initial prompt spells the proper nouns for Whisper, which can hide a mispronounced
"Malachor". Takes whose text contains a name from evalcommon.PROBE_NAMES (per character: <lab>/eval/
eval_character.json asr.probe_names, else derived from the asr_prompt) get a second, prompt-free pass; the
words aligned to the name are compared with the name itself and with how the same ASR hears the character's REAL
train/val clips containing that name without the prompt (up to 4 per name: the accepted spellings).

CUDA: torch is imported first so its cuBLAS DLLs are on the search path (as Vox's sidecar does). If loading
the model OR the first CUDA transcription fails (e.g. a missing cuBLAS/cuDNN DLL only surfaces at the first
inference), the step falls back to CPU int8 once and says so in asr.json ("fell_back"). Later per-take
failures are retried once and then left unscored (exit 3). Writes <run>/metrics/asr.json.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

WHISPER_CACHE = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local")) / "VoxStudio" / "engines" / "voxcpm2" / "cache" / "faster-whisper"
VERSION = 1
CALIBRATION_CLIPS = 4
EXACT_SOURCES = ("tlk-exact", "dlg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name in the Vox cache")
    parser.add_argument("--compute-type", default=None, help="default int8 on cpu, float16 on cuda")
    parser.add_argument("--no-name-probe", action="store_true", help="skip the prompt-free proper-noun pass")
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        # Like Vox's sidecar: import torch first so its bundled cuBLAS DLLs (torch/lib) are on the DLL search
        # path for CTranslate2 (which ships its own cuDNN 9).
        try:
            import torch
            import ctranslate2
            if not torch.cuda.is_available() or ctranslate2.get_cuda_device_count() < 1:
                ec.log("cuda requested but no GPU is visible to torch/CTranslate2; using cpu")
                device = "cpu"
        except Exception as exc:
            ec.log(f"cuda requested but CUDA init failed ({type(exc).__name__}: {exc}); using cpu")
            device = "cpu"
    from faster_whisper import WhisperModel

    state = {"device": device, "compute_type": args.compute_type or ("float16" if device == "cuda" else "int8"),
             "model": None, "cuda_ok": False, "fell_back": None}

    def load(dev: str, compute_type: str) -> None:
        state["model"] = None
        gc.collect()
        state["model"] = WhisperModel(args.model, device=dev, compute_type=compute_type,
                                      download_root=str(WHISPER_CACHE), local_files_only=True, cpu_threads=args.threads)
        state["device"], state["compute_type"] = dev, compute_type

    def to_cpu(reason: str) -> None:
        compute_type = args.compute_type if args.compute_type and "float16" not in args.compute_type else "int8"
        ec.log(f"WARNING: whisper on cuda failed ({reason}); falling back to cpu {compute_type} "
               "(slow: 1-3 min per clip on a loaded machine)")
        state["fell_back"] = reason
        load("cpu", compute_type)

    started = time.time()
    try:
        load(state["device"], state["compute_type"])
    except Exception as exc:
        if state["device"] != "cuda":
            raise
        to_cpu(f"load: {type(exc).__name__}: {exc}")
    ec.log(f"faster-whisper {args.model} ({state['compute_type']}) on {state['device']}, "
           f"loaded in {time.time() - started:.1f}s")

    caches: dict[tuple, ec.Cache] = {}

    def settings(prompted: bool) -> dict:
        return {"metric": "asr", "version": VERSION, "model": args.model, "compute_type": state["compute_type"],
                "beam_size": 5, "prompt": ec.ASR_NAMES_PROMPT if prompted else None}

    def cache_for(prompted: bool) -> ec.Cache:
        key = (state["compute_type"], prompted)
        if key not in caches:
            caches[key] = ec.Cache("asr", settings(prompted))
        return caches[key]

    def save_caches() -> None:
        for c in caches.values():
            c.save()

    def run_model(path: str, prompted: bool) -> dict:
        segments, _info = state["model"].transcribe(
            str(path), language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False,
            initial_prompt=ec.ASR_NAMES_PROMPT if prompted else None)
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        spans = [max(1e-3, s.end - s.start) for s in segments]
        return {
            "asr": text,
            "avg_logprob": (sum(s.avg_logprob * d for s, d in zip(segments, spans)) / sum(spans)) if segments else None,
            "max_no_speech_prob": max((s.no_speech_prob for s in segments), default=None),
            "max_compression_ratio": max((s.compression_ratio for s in segments), default=None),
        }

    def transcribe(path: str, prompted: bool = True) -> dict:
        sha = ec.file_sha256(path)
        hit = cache_for(prompted).get(sha)
        if hit is not None:
            return {**hit, "sha256": sha}
        try:
            result = run_model(path, prompted)
        except Exception as exc:
            if state["device"] == "cuda" and not state["cuda_ok"] and ec.is_resource_error(exc):
                # The first CUDA inference is where missing cuBLAS/cuDNN libraries surface.
                to_cpu(f"first transcription: {type(exc).__name__}: {exc}")
                return transcribe(path, prompted)
            raise
        if state["device"] == "cuda":
            state["cuda_ok"] = True
        cache_for(prompted).put(sha, result)
        return {**result, "sha256": sha}

    run = ec.load_run(args.run_dir)
    variant = run["audio_variant"]
    sets = ec.real_sets(variant)
    real, missing = {}, []
    ec.log(f"asr: {len(sets['test'])} real test clips (baseline)")
    for clip in sets["test"]:
        result, err = ec.guarded(transcribe, clip["path"])
        if err:
            if err["unscored"]:
                save_caches()
                ec.log(f"!! asr: real test clip {clip['id']} failed twice ({err['error']}); baseline incomplete")
                raise SystemExit(3)
            missing.append({"id": clip["id"], "error": err["error"]})
            continue
        real[clip["id"]] = {**result, **ec.wer_details(clip["text"], result["asr"]), "text": clip["text"]}
    save_caches()

    # Name-probe calibration on real train/val clips (never test clips) that contain each name in the takes.
    probe = not args.no_name_probe
    stems = sorted({s for t in run["takes"] for s, _a, _b in ec.find_stems(ec.norm_words(t["text"]))}) if probe else []
    calibration: dict[str, dict] = {}
    if stems:
        rows = sorted((r for r in ec.load_corpus() if r["split"] in ("train", "val") and r["kind"] == "speech"
                       and r["text_source"] in EXACT_SOURCES), key=lambda r: r["id"])
        known, _src = ec.known_other_speakers()
        for stem in stems:
            clips = [r for r in rows if r["id"] not in known and ec.find_stems(ec.norm_words(r["text"]), [stem])]
            heard = []
            for row in clips[:CALIBRATION_CLIPS]:
                path = ec.real_audio_path(row, variant)
                if not path:
                    continue
                result, err = ec.guarded(transcribe, str(path), False)
                if err:
                    continue
                ref, hyp = ec.norm_words(row["text"]), ec.norm_words(result["asr"])
                for _s, a, b in ec.find_stems(ref, [stem]):
                    span = ec.aligned_span(ref, hyp, a, b)
                    heard.append({"id": row["id"], "heard": span, "ok": bool(ec.find_stems(span.split(), [stem]))})
            calibration[stem] = {"clips": [h["id"] for h in heard], "heard": heard,
                                 "accepted": sorted({h["heard"] for h in heard if h["heard"]}),
                                 "recognised": sum(h["ok"] for h in heard)}
            ec.log(f"asr probe '{stem}': real clips heard without the prompt as "
                   f"{[h['heard'] for h in heard] or 'n/a (no real clip with this name)'}")
        save_caches()

    def score_take(take: dict) -> dict:
        result = transcribe(take["wav"])
        entry = {**result, **ec.wer_details(take["text"], result["asr"])}
        ref = ec.norm_words(take["text"])
        found = ec.find_stems(ref) if probe else []
        if found:
            free = transcribe(take["wav"], False)
            hyp = ec.norm_words(free["asr"])
            results = []
            for stem, a, b in found:
                span = ec.aligned_span(ref, hyp, a, b)
                cal = calibration.get(stem) or {}
                ok = bool(ec.find_stems(span.split(), [stem])) or span in (cal.get("accepted") or [])
                results.append({"stem": stem, "heard": span, "ok": ok, "calibrated": bool(cal.get("clips"))})
            entry["probe"] = {"free_asr": free["asr"], "results": results}
        return entry

    takes = {}
    ec.log(f"asr: {len(run['takes'])} takes")
    for i, take in enumerate(run["takes"], 1):
        entry, err = ec.guarded(score_take, take)
        takes[take["key"]] = err if err else entry
        if i % 10 == 0:
            ec.log(f"  {i}/{len(run['takes'])}")
            save_caches()
    save_caches()

    out = {"metric": "asr", "settings": settings(True), "device": state["device"],
           "compute_type": state["compute_type"], "fell_back": state["fell_back"], "audio_variant": variant,
           "seconds": round(time.time() - started, 1),
           "baselines": {"test_real": {**ec.stats([r["wer"] for r in real.values()]), "per_clip": real,
                                       "missing": missing},
                         "probe_calibration": calibration},
           "takes": takes}
    code = ec.finish_metric(args.run_dir, "asr", out, run)
    ec.log(f"asr: real test WER mean {out['baselines']['test_real'].get('mean', float('nan')):.3f}; "
           f"wrote {ec.metric_out_path(args.run_dir, 'asr')}")
    ec.exit_with(code)


if __name__ == "__main__":
    main()
