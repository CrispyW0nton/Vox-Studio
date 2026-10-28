"""Emotion with emotion2vec+ base (FunASR): 9-way class scores and the utterance embedding per take, compared with
the delivery map's reference clips for the take's delivery label.

Run with the eval-extra venv (Python 3.11, torch 2.8.0+cu128, funasr 1.4.16):
  set CUDA_VISIBLE_DEVICES=-1
  voice-lab/eval-extra/.venv/Scripts/python.exe metric_emotion.py --run-dir <lab>/eval/<run> [--refs all|run|none]
  voice-lab/eval-extra/.venv/Scripts/python.exe metric_emotion.py --wav take.wav [--delivery calm]
      (standalone: one file's classes + embedding comparison with that label's references, JSON on stdout)
(run_eval.py runs it as the optional step "emotion": --metrics ...,emotion). <lab> is the lab of the character
VOICE_LAB_CHARACTER selects (default BaoDurLab).

Model: emotion2vec_plus_base (~90M parameters), local copy voice-lab/eval-extra/models/emotion2vec_plus_base
(huggingface.co/emotion2vec/emotion2vec_plus_base @ b318240; model.pt sha256 in eval-extra/PROVENANCE.json), loaded
through funasr.AutoModel from that folder (no hub access, update check off). Licence: FunASR Model Open Source
License 1.1 (free use/modification/sharing; attribute the source and keep the model name: "emotion2vec+ base",
emotion2vec authors / Alibaba FunASR).
Input: 16 kHz mono (resampled here), trimmed to the speech span (+0.1 s) like dnsmos, so lead/tail silence does not
dilute the utterance mean; the model layer-normalises the waveform, so level does not matter.
Per file: probabilities over angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown (softmax of
the utterance head) and the 768-d utterance embedding (mean of the last-layer frame features).
Delivery references: the run's delivery map (<lab>/delivery/delivery_map.json; test clips dropped), primary
audio variant. --refs all embeds every label's references (needed for the label rank and for the aggregator's
leave-one-out check of the signal), run only the labels the takes use, none skips them; --max-refs-per-label
keeps the first N of each label. compare_with_references() (numpy-only helper, importable without torch) gives per take:
cosine to each reference of its label (a reference byte-identical to the take is skipped), mean / max / nearest,
cosine to the label centroid and, with >= 3 labels, the rank of its own label among all label centroids
(match = 1 - (rank - 1) / (labels - 1)). Embeddings are centred on the mean of all embedded references when at least
CENTER_MIN_REFS are available (as the aggregator's WavLM delivery test does), otherwise cosines are raw.
A paired take (held-out or train/val diagnostic line) is also compared with that line's real recording (ground truth
only, never a reference). emotion2vec+ was trained on everyday speech: the classes are descriptive; the scorecard
reports whether the embedding separates the character's own delivery labels, and emotion never enters the composite.
Cached by SHA-256 + settings in <lab>/eval/_cache/emotion. Writes <run>/metrics/emotion.json; exit 3 when takes
stayed unscored after a retry (resource errors).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 1
EXTRA_DIR = Path(os.environ.get("BAODUR_EVAL_EXTRA", str(ec.VOICE_LAB / "eval-extra")))
MODEL_DIR = EXTRA_DIR / "models" / "emotion2vec_plus_base"
MODEL_NAME = "emotion2vec_plus_base"
MODEL_SHA256 = "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"
RATE = 16000
LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")
CENTER_MIN_REFS = 12
MIN_RANK_LABELS = 3


# ----------------------------------------------------------------------------------------------
# Helpers (numpy only, importable without torch; aggregate.py reads their output from metrics/emotion.json)
# ----------------------------------------------------------------------------------------------

def canonical_label(token: str) -> str:
    """'中立/neutral' -> 'neutral'; '<unk>' -> 'unknown'."""
    token = str(token).strip()
    if token in ("<unk>", "unk"):
        return "unknown"
    return token.split("/")[-1].strip().casefold()


def top_class(probs) -> tuple[str | None, float | None]:
    if not probs:
        return None, None
    i = int(np.argmax(probs))
    return LABELS[i], float(probs[i])


def reference_center(refs: dict[str, np.ndarray]) -> np.ndarray | None:
    """Mean of the embedded reference clips (raw embeddings), or None with fewer than CENTER_MIN_REFS."""
    if len(refs) < CENTER_MIN_REFS:
        return None
    return np.mean([np.asarray(v, dtype=np.float64) for v in refs.values()], axis=0)


def _prep(vector, center):
    v = np.asarray(vector, dtype=np.float64)
    return ec.unit(v - center) if center is not None else ec.unit(v)


def compare_with_references(vector, label: str, refs: dict[str, np.ndarray], labels_to_ids: dict[str, list[str]],
                            skip=(), center: np.ndarray | None = None) -> dict:
    """One embedding vs a delivery label's reference clips (cosines, centroid) and its label's rank among all label
    centroids. refs: clip id -> raw embedding; labels_to_ids: casefolded label -> reference ids; skip: ids left out
    (the take itself when it is a reference clip); center: subtract before the cosine (reference_center())."""
    label = (label or "").casefold()
    out: dict = {"label": label, "centered": center is not None}
    t = _prep(vector, center)
    prepared = {i: _prep(v, center) for i, v in refs.items() if i not in set(skip)}
    ids = [i for i in labels_to_ids.get(label, []) if i in prepared]
    if not ids:
        out["error"] = "no reference embeddings for this delivery label"
        return out
    cos = {i: float(t @ prepared[i]) for i in ids}
    nearest = max(cos, key=cos.get)
    out.update(n_refs=len(ids), per_ref={i: round(c, 4) for i, c in cos.items()}, mean_cos=float(np.mean(list(cos.values()))),
               max_cos=cos[nearest], nearest_ref=nearest)
    if skip:
        out["skipped"] = sorted(set(skip) & set(labels_to_ids.get(label, [])))
    centroids = {}
    for lab, lab_ids in labels_to_ids.items():
        vs = [prepared[i] for i in lab_ids if i in prepared]
        if vs:
            centroids[lab] = ec.unit(np.mean(vs, axis=0))
    out["centroid_cos"] = float(t @ centroids[label])
    if len(centroids) >= MIN_RANK_LABELS:
        sims = {lab: float(t @ c) for lab, c in centroids.items()}
        order = sorted(sims, key=sims.get, reverse=True)
        rank = order.index(label) + 1
        out.update(rank=rank, of=len(order), match=1.0 - (rank - 1) / (len(order) - 1), nearest_label=order[0],
                   top_labels=[{"label": l, "cos": round(sims[l], 4)} for l in order[:3]])
    return out


def delivery_references(map_path: str | Path | None, tests: set[str], labels: set[str] | None = None,
                        max_per_label: int = 0) -> dict[str, list[str]]:
    """casefolded label -> reference clip ids from a delivery map (test clips dropped; first N per label if set)."""
    if not map_path or not Path(map_path).exists():
        return {}
    out = {}
    for label, spec in (ec.read_json(map_path).get("deliveries") or {}).items():
        key = label.casefold()
        if labels is not None and key not in labels:
            continue
        ids = [r.get("id") for r in spec.get("references", []) if r.get("id") and r.get("id") not in tests]
        if max_per_label > 0:
            ids = ids[:max_per_label]
        if ids:
            out[key] = ids
    return out


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------

def offline_env() -> None:
    os.environ["HF_HOME"] = str(EXTRA_DIR / "hf-home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_CACHE"] = str(EXTRA_DIR / "hf-home" / "modelscope")


def load_model(device: str, threads: int):
    for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt"):
        if not (MODEL_DIR / name).exists():
            raise SystemExit(f"emotion2vec+ base file missing: {MODEL_DIR / name} (eval-extra install)")
    import logging
    from funasr import AutoModel
    # importing funasr runs funasr/bin/realtime_ws.py's logging.basicConfig(INFO): ~200 "init param" lines per load
    logging.getLogger().setLevel(logging.ERROR)
    model = AutoModel(model=str(MODEL_DIR), device=device, ncpu=max(1, threads), disable_update=True,
                      disable_pbar=True, log_level="ERROR")
    logging.getLogger().setLevel(logging.ERROR)
    return model


def analyse(model, path: str) -> dict:
    x, _ = ec.read_mono(path, RATE)
    x = np.ascontiguousarray(ec.speech_segment(x, RATE), dtype=np.float32)
    if not len(x) or not np.any(x) or not np.all(np.isfinite(x)):
        raise ValueError("silent or non-finite audio")
    result = model.generate(x, granularity="utterance", extract_embedding=True, fs=RATE, disable_pbar=True)[0]
    probs = {canonical_label(l): float(s) for l, s in zip(result["labels"], result["scores"])}
    missing = [l for l in LABELS if l not in probs]
    if missing:
        raise ValueError(f"unexpected emotion2vec labels {list(probs)} (missing {missing})")
    vector = np.asarray(result["feats"], dtype=np.float64).ravel()
    if vector.size != 768 or not np.all(np.isfinite(vector)):
        raise ValueError(f"bad embedding (size {vector.size})")
    p = [round(probs[l], 5) for l in LABELS]
    top, top_p = top_class(p)
    return {"probs": p, "top": top, "top_p": top_p, "embedding": [round(float(v), 6) for v in vector],
            "speech_s": round(len(x) / RATE, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", help=f"{ec.LAB.name}/eval/<run> folder holding takes.json")
    target.add_argument("--wav", help="standalone: score one file and print JSON")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (torch)")
    parser.add_argument("--refs", choices=("all", "run", "none"), default="all",
                        help="delivery reference clips embedded: every label (default), the labels the takes use, none")
    parser.add_argument("--max-refs-per-label", type=int, default=0, help="first N references per label (0 = all)")
    parser.add_argument("--no-paired", action="store_true", help="skip the comparison with a paired line's real clip")
    parser.add_argument("--delivery", default=None, help="standalone: delivery label to compare with")
    parser.add_argument("--delivery-map", default=None,
                        help=f"standalone: delivery map (default {ec.LAB.name}/delivery/delivery_map.json)")
    parser.add_argument("--audio", choices=ec.VARIANTS, default="original", help="standalone: reference audio variant")
    parser.add_argument("--gpu-cap-gb", type=float, default=1.5, help="per-process GPU memory cap on cuda")
    args = parser.parse_args()

    offline_env()
    device = ec.torch_device(args.device, cap_gb=args.gpu_cap_gb, threads=args.threads)
    sha = ec.file_sha256(MODEL_DIR / "model.pt") if (MODEL_DIR / "model.pt").exists() else None
    if sha and sha != MODEL_SHA256:
        raise SystemExit(f"emotion2vec+ base model.pt sha256 {sha} differs from the installed one {MODEL_SHA256}")
    started = time.time()
    loaded: list = []

    def get_model():  # loaded on the first cache miss (a fully cached re-run never loads the model)
        if not loaded:
            t0 = time.time()
            loaded.append(load_model(device, args.threads))
            ec.log(f"emotion2vec+ base on {device} ({args.threads} threads), loaded in {time.time() - t0:.1f} s")
        return loaded[0]

    settings = {"metric": "emotion", "version": VERSION, "model": MODEL_NAME, "model_sha256": sha,
                "granularity": "utterance", "rate": RATE, "trim": "speech_span+0.1s"}
    cache = ec.Cache("emotion", settings)

    def cached(path: str) -> dict:
        file_sha = ec.file_sha256(path)
        hit = cache.get(file_sha)
        if hit is None:
            hit = analyse(get_model(), path)
            cache.put(file_sha, hit)
        return {"sha256": file_sha, **hit}

    corpus = ec.corpus_index()
    tests = {i for i, r in corpus.items() if r["split"] == "test"}

    def embed_references(variant: str, labels_to_ids: dict[str, list[str]], real: dict, real_errors: list):
        ids = sorted({i for v in labels_to_ids.values() for i in v})
        ec.log(f"emotion: {len(ids)} delivery reference clips ({variant}) for {len(labels_to_ids)} label(s)")
        for clip_id in ids:
            row = corpus.get(clip_id)
            path = ec.real_audio_path(row, variant) if row else None
            if path is None:
                real_errors.append({"id": clip_id, "error": "audio not found", "kind": "take", "unscored": False})
                continue
            value, err = ec.guarded(cached, str(path))
            if err:
                real_errors.append({"id": clip_id, **err})
                continue
            real[clip_id] = {"split": row["split"], "role": "reference",
                             "labels": sorted(l for l, v in labels_to_ids.items() if clip_id in v), **value}
        cache.save()

    if args.wav:  # standalone
        map_path = args.delivery_map or str(ec.DEFAULT_DELIVERY_MAP)
        label = (args.delivery or "").casefold() or None
        wanted = None if args.refs == "all" else ({label} if label else set())
        labels_to_ids = {} if args.refs == "none" else delivery_references(map_path, tests, wanted,
                                                                             args.max_refs_per_label)
        real, real_errors = {}, []
        if labels_to_ids:
            embed_references(args.audio, labels_to_ids, real, real_errors)
        value = cached(args.wav)
        refs = {i: np.asarray(r["embedding"]) for i, r in real.items()}
        out = {"wav": args.wav, **{k: v for k, v in value.items() if k != "embedding"},
               "probs_by_label": dict(zip(LABELS, value["probs"]))}
        if label:
            skip = [i for i, r in real.items() if r.get("sha256") == value["sha256"]]
            out["reference"] = compare_with_references(value["embedding"], label, refs, labels_to_ids, skip,
                                                       reference_center(refs))
        out["real_errors"] = real_errors
        print(json.dumps(ec.clean(out), indent=2, ensure_ascii=False))
        cache.save()
        ec.exit_with(0)

    run = ec.load_run(args.run_dir)
    variant = run["audio_variant"]
    take_labels = {(t.get("delivery") or "").casefold() for t in run["takes"]} - {""}
    if args.refs == "none":
        labels_to_ids = {}
    else:
        labels_to_ids = delivery_references(run.get("delivery_map"), tests,
                                            None if args.refs == "all" else take_labels, args.max_refs_per_label)
        if not labels_to_ids:
            ec.log("emotion: no delivery map / references; takes get classes and embeddings only")
    real: dict[str, dict] = {}
    real_errors: list[dict] = []
    if labels_to_ids:
        embed_references(variant, labels_to_ids, real, real_errors)
    if not args.no_paired:
        paired = sorted({t["paired_id"] for t in run["takes"] if t.get("paired_id")} - set(real))
        if paired:
            ec.log(f"emotion: {len(paired)} paired real recordings ({variant}, ground truth only)")
        for clip_id in paired:
            row = corpus.get(clip_id)
            path = ec.real_audio_path(row, variant) if row else None
            if path is None:
                real_errors.append({"id": clip_id, "error": "audio not found", "kind": "take", "unscored": False})
                continue
            value, err = ec.guarded(cached, str(path))
            if err:
                real_errors.append({"id": clip_id, **err})
                continue
            real[clip_id] = {"split": row["split"], "role": "paired", "labels": [], **value}
        cache.save()
    ref_vectors = {i: np.asarray(r["embedding"], dtype=np.float64) for i, r in real.items() if r["role"] == "reference"}
    center = reference_center(ref_vectors)

    takes: dict[str, dict] = {}
    ec.log(f"emotion: {len(run['takes'])} takes")
    for take in run["takes"]:
        value, err = ec.guarded(cached, take["wav"])
        if err:
            takes[take["key"]] = err
            continue
        entry = dict(value)
        label = (take.get("delivery") or "").casefold()
        entry["delivery"] = label or None
        if label and labels_to_ids:
            skip = [i for i, r in real.items() if r["role"] == "reference" and r.get("sha256") == value["sha256"]]
            entry["reference"] = compare_with_references(value["embedding"], label, ref_vectors, labels_to_ids,
                                                         skip, center)
        pid = take.get("paired_id")
        if pid and pid in real:
            r = real[pid]
            entry["paired"] = {"id": pid, "split": r["split"], "cos": float(_prep(value["embedding"], center) @
                                                                         _prep(r["embedding"], center)),
                               "centered": center is not None, "real_top": r["top"], "real_top_p": r["top_p"],
                               "same_top": r["top"] == value["top"]}
        takes[take["key"]] = entry
    cache.save()
    out = {"metric": "emotion", "settings": settings, "audio_variant": variant, "device": device,
           "seconds": round(time.time() - started, 1), "labels": list(LABELS), "references_scope": args.refs,
           "max_refs_per_label": args.max_refs_per_label, "reference_labels": labels_to_ids,
           "center": {"n_refs": len(ref_vectors), "centered": center is not None, "min_refs": CENTER_MIN_REFS},
           "real": real, "real_errors": real_errors, "takes": takes}
    code = ec.finish_metric(args.run_dir, "emotion", out, run)
    if any(e.get("unscored") for e in real_errors):
        ec.log(f"!! emotion: {len(real_errors)} real clip(s) failed (resource errors); re-run this step")
        code = 3
    tops: dict[str, int] = {}
    for v in takes.values():
        if isinstance(v, dict) and v.get("top"):
            tops[v["top"]] = tops.get(v["top"], 0) + 1
    ec.log(f"emotion: take top classes {tops}; {len(ref_vectors)} reference embeddings "
           f"({'centred' if center is not None else 'raw cosines'}); wrote {ec.metric_out_path(args.run_dir, 'emotion')}")
    ec.exit_with(code)


if __name__ == "__main__":
    main()
