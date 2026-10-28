"""Prosody and delivery features for takes and for every real clip of the character (VOICE_LAB_CHARACTER).

Run with the seed-vc venv (librosa 0.10.2, pyloudnorm 0.2.0):
  set CUDA_VISIBLE_DEVICES=-1
  seed-vc/.venv/Scripts/python.exe metric_prosody.py --run-dir <lab>/eval/<run> [--workers 4]

Per clip, on the energy-VAD speech span (evalcommon.activity):
- F0 with librosa.pyin at 16 kHz (fmin 55, fmax 420 Hz, 64 ms frames, 10 ms hop, 0.2-semitone
  resolution: ~10x faster than 0.1 and within 0.1 st on median/IQR in spot checks). The search range, the creak
  threshold and the fold passes are per character (<lab>/eval/eval_character.json "prosody"; the values here are
  Bao-Dur's, also the defaults), and they are part of the cache settings.
  Bao-Dur's voice is soft and gravelly: pyin tracks isolated octave jumps and long creaky (vocal fry) stretches
  at 55-85 Hz under a 100-180 Hz modal voice, which made per-clip F0 IQR bimodal (2.5 vs 12 st). So:
  (1) octave errors are folded back toward the local median contour: first voiced frames within 2 st of
  a whole number of octaves away from the median of a +-750 ms neighbourhood (catches sustained 300-500 ms
  sub-harmonic stretches, e.g. gblbaodur056: IQR 8.6 -> 5.2 st, while 7 other clips including angry and
  questioning ones were unchanged), then within 3 st of the +-250 ms median (isolated jumps); two
  iterations each; f0_fold_ratio = folded share of voiced frames; (2) frames below min(85 Hz, 0.65 x the
  clip's 80th-percentile F0) count as creak: creak_ratio = creak / voiced frames, a voice-quality
  feature of its own, and the F0 statistics use the modal (non-creak) frames. Semitones are relative
  to 100 Hz. f0_median_st, f0_iqr_st, f0_range_st (p95 - p5), f0_end_st (median of the last quarter of
  modal frames minus their median: rising or falling line endings), voiced_ratio.
- Rate: heuristic syllables (and words) of the intended text per second of speech span, plus
  articulation rate (pauses removed). pause_ratio = internal silences >= 150 ms / span.
- Level: integrated loudness (BS.1770, pyloudnorm) of the whole clip, active-frame RMS dB and the IQR of
  active-frame dB (energy dynamics).
Real clips (train, val, test of the run's audio variant) are cached by SHA-256; the aggregator builds
the train distribution, per-delivery targets and held-out targets from them. pyin is single-threaded, so
--workers N runs N processes (~0.25 GB each). Writes <run>/metrics/prosody.json; exit 3 when takes stayed
unscored after a retry (resource errors).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 4
_PROSODY = ec.EVAL_CHARACTER["prosody"]   # per-character voice settings (Bao-Dur: 55-420 Hz, creak 85 Hz / 0.65)
FMIN, FMAX = _PROSODY["fmin_hz"], _PROSODY["fmax_hz"]
RATE = 16000
# Octave-error folding passes: (frames of 10 ms each side of a voiced frame for the median, semitones from an
# exact octave multiple that still count as an octave error). Wide and strict first, then local.
FOLD_PASSES = tuple(tuple(p) for p in _PROSODY["fold_passes"])
CREAK_MAX_HZ = _PROSODY["creak_max_hz"]   # creak threshold: min(CREAK_MAX_HZ, CREAK_REL x clip p80 F0)
CREAK_REL = _PROSODY["creak_rel"]
SETTINGS = {"metric": "prosody", "version": VERSION, "fmin": FMIN, "fmax": FMAX, "rate": RATE,
            "frame": 1024, "hop": 160, "resolution": 0.2, "fold": [list(p) for p in FOLD_PASSES],
            "creak": [CREAK_MAX_HZ, CREAK_REL], "vad": [ec.ACTIVE_REL_DB, ec.ACTIVE_FLOOR_DB, ec.MIN_PAUSE_S]}
_METERS: dict = {}


def fold_octave_errors(frames, st):
    """Fold octave jumps toward the median contour. frames: voiced frame indices (sorted), st: their
    semitone values. Returns (folded values, number of frames changed)."""
    import numpy as np
    st = st.astype(np.float64).copy()
    changed = np.zeros(st.size, dtype=bool)
    for window, tolerance in FOLD_PASSES:
        for _ in range(2):
            lo = np.searchsorted(frames, frames - window, side="left")
            hi = np.searchsorted(frames, frames + window, side="right")
            local = np.array([np.median(st[a:b]) for a, b in zip(lo, hi)])
            diff = st - local
            octaves = np.round(diff / 12.0)
            fold = (octaves != 0) & (np.abs(diff - 12.0 * octaves) < tolerance)
            if not fold.any():
                break
            st[fold] -= 12.0 * octaves[fold]
            changed |= fold
    return st, int(changed.sum())


def lufs(x, rate: int):
    import numpy as np
    try:
        import pyloudnorm
    except ImportError:
        return None
    if len(x) < int(0.4 * rate) + 1:
        return None
    meter = _METERS.setdefault(rate, pyloudnorm.Meter(rate))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(meter.integrated_loudness(x.astype(np.float64)))
    return value if np.isfinite(value) else None


def features(path: str, text: str) -> dict:
    import numpy as np
    import librosa
    x, rate = ec.read_mono(path)
    loudness = lufs(x, rate)
    x16 = ec.resample(x, rate, RATE)
    act = ec.activity(x16, RATE)
    if act["start_s"] is None or act["span_s"] < 0.1:
        return {"error": "no speech detected", "duration_s": len(x) / rate, "lufs": loudness}
    seg = x16[int(act["start_s"] * RATE):int(act["end_s"] * RATE)]
    if len(seg) < 1024:
        seg = np.pad(seg, (0, 1024 - len(seg)))
    f0, voiced, _prob = librosa.pyin(seg, fmin=FMIN, fmax=FMAX, sr=RATE, frame_length=1024, hop_length=160,
                                     resolution=0.2)
    good = voiced & np.isfinite(f0)
    folded = 0
    if good.sum() >= 5:
        st_folded, folded = fold_octave_errors(np.flatnonzero(good), 12 * np.log2(f0[good] / 100.0))
        f0v = 100.0 * 2 ** (st_folded / 12)
    else:
        f0v = f0[good]
    span = act["span_s"]
    pause = float(sum(b - a for a, b in act["pauses"]))
    syllables = ec.count_syllables(text)
    words = len(ec.spoken_words(text))
    db = act["frame_db"][act["mask"]]
    out = {
        "duration_s": len(x) / rate, "span_s": span, "pause_s": pause, "n_pauses": len(act["pauses"]),
        "pause_ratio": pause / span if span > 0 else None,
        "syllables": syllables, "words": words,
        "rate_syl_s": syllables / span if span > 0 else None,
        "artic_rate_syl_s": syllables / (span - pause) if span - pause > 0.05 else None,
        "words_s": words / span if span > 0 else None,
        "voiced_ratio": float(good.mean()) if good.size else 0.0,
        "lufs": loudness,
        "active_rms_db": float(10 * np.log10(np.mean(10 ** (db / 10)))) if db.size else None,
        "energy_iqr_db": float(np.subtract(*np.percentile(db, [75, 25]))) if db.size else None,
    }
    if f0v.size >= 5:
        creak_hz = min(CREAK_MAX_HZ, CREAK_REL * float(np.percentile(f0v, 80)))
        creak = f0v < creak_hz
        modal = f0v[~creak] if (~creak).sum() >= 5 else f0v
        st = 12 * np.log2(modal / 100.0)
        p05, p25, p50, p75, p95 = np.percentile(st, [5, 25, 50, 75, 95])
        tail = st[int(len(st) * 0.75):]
        out.update({
            "f0_median_hz": float(100 * 2 ** (p50 / 12)), "f0_p05_hz": float(100 * 2 ** (p05 / 12)),
            "f0_p95_hz": float(100 * 2 ** (p95 / 12)), "f0_median_st": float(p50),
            "f0_iqr_st": float(p75 - p25), "f0_range_st": float(p95 - p05),
            "f0_end_st": float(np.median(tail) - p50) if tail.size else None,
            "creak_ratio": float(creak.mean()), "creak_hz": creak_hz,
            "f0_fold_ratio": folded / f0v.size,
        })
    else:
        out["f0_note"] = "fewer than 5 voiced frames"
    return out


def job(item: tuple) -> tuple:
    """Worker entry: (tag, cache key, path, text) -> (tag, cache key, value | None, error | None)."""
    tag, key, path, text = item
    value, err = ec.guarded(features, path, text)
    return tag, key, value, err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    parser.add_argument("--workers", type=int, default=1, help="parallel pyin processes")
    args = parser.parse_args()

    try:
        import pyloudnorm  # noqa: F401
    except ImportError:
        ec.log("pyloudnorm missing: integrated loudness will be null (RMS dB still reported)")
    cache = ec.Cache("prosody", SETTINGS)

    def cache_key(path: str, text: str) -> tuple[str, str]:
        sha = ec.file_sha256(path)
        return sha, f"{sha}:{ec.count_syllables(text)}:{len(ec.spoken_words(text))}"

    started = time.time()
    run = ec.load_run(args.run_dir)
    sets = ec.real_sets(run["audio_variant"])
    real, takes, todo, shas = {}, {}, [], {}
    real_errors = []
    for role in ("train", "val", "test"):
        for clip in sets[role]:
            sha, key = cache_key(clip["path"], clip["text"])
            hit = cache.get(key)
            if hit is None:
                todo.append((("real", clip["id"], role), key, clip["path"], clip["text"]))
            else:
                real[clip["id"]] = {"split": role, **hit}
    for take in run["takes"]:
        try:
            sha, key = cache_key(take["wav"], take["text"])
        except OSError as exc:
            takes[take["key"]] = {"error": f"{type(exc).__name__}: {exc}", "kind": "resource"
                                  if ec.is_resource_error(exc) else "take", "unscored": ec.is_resource_error(exc)}
            continue
        shas[take["key"]] = sha
        hit = cache.get(key)
        if hit is None:
            todo.append((("take", take["key"], None), key, take["wav"], take["text"]))
        else:
            takes[take["key"]] = {"sha256": sha, **hit}
    ec.log(f"prosody: {len(real)} real clips and {len(takes)} takes cached; {len(todo)} to analyse "
           f"with {max(1, args.workers)} worker(s)")

    def collect(tag, key, value, err) -> None:
        kind, ident, role = tag
        if err is None:
            cache.put(key, value)
        if kind == "real":
            if err is None:
                real[ident] = {"split": role, **value}
            else:
                real_errors.append({"id": ident, **err})
        else:
            takes[ident] = {"sha256": shas.get(ident), **value} if err is None else err

    done = 0
    if args.workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(job, todo, chunksize=1):
                collect(*result)
                done += 1
                if done % 50 == 0:
                    ec.log(f"  {done}/{len(todo)}")
                    cache.save()
    else:
        for item in todo:
            collect(*job(item))
            done += 1
            if done % 50 == 0:
                ec.log(f"  {done}/{len(todo)}")
                cache.save()
    cache.save()
    out = {"metric": "prosody", "settings": SETTINGS, "audio_variant": run["audio_variant"],
           "seconds": round(time.time() - started, 1), "workers": args.workers, "real": real,
           "real_errors": real_errors, "takes": takes}
    code = ec.finish_metric(args.run_dir, "prosody", out, run)
    if any(e.get("unscored") for e in real_errors):
        ec.log(f"!! prosody: {len(real_errors)} real clip(s) failed (resource errors); re-run this step")
        code = 3
    ec.log(f"prosody: wrote {ec.metric_out_path(args.run_dir, 'prosody')} in {out['seconds']}s")
    ec.exit_with(code)


if __name__ == "__main__":
    main()
