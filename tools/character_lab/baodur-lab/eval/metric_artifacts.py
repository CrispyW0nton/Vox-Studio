"""Signal artifacts for takes, with the character's real clips (VOICE_LAB_CHARACTER) as the reference distribution.

Run with the seed-vc venv (numpy, scipy, soundfile):
  set CUDA_VISIBLE_DEVICES=-1
  seed-vc/.venv/Scripts/python.exe metric_artifacts.py --run-dir <lab>/eval/<run>

Per clip at its native rate (the calibration notes below are Bao-Dur's 280 real train clips):
- format (rate/channels/subtype vs the 48 kHz mono PCM16 contract), non-finite samples, peak, DC offset;
- clipping: runs of >= 3 consecutive samples at the clip's own ceiling, i.e. |x| >= min(0.999, peak - 1.5 LSB)
  when the peak is >= -20 dBFS. Engines rescale peaks to 0.999 (GSV, VoxCPM) or -1 dBFS (Seed-VC) after
  synthesis, which hides clipped flat tops from a fixed 0.999 threshold; flat tops at the clip's own peak do
  not occur in his real clips (0 of 280);
- clicks: max |x[n] - x[n-1]| and its ratio to the clip's 99.9th percentile step, plus the isolated-spike
  measure the aggregator flags on: second difference d2[n] = |x[n] - (x[n-1] + x[n+1]) / 2|, spike_ratio =
  max d2 / its 99.9th percentile, spike_rel_peak = max d2 / peak, spike_at_s = where. GPT-SoVITS casts audio
  normalised to exactly +1.0 to int16 (TTS_infer_pack/TTS.py), which wraps that sample to -32768: a
  full-scale one-sample click. Simulated through the lab's 32 kHz -> 48 kHz resample it gives spike_ratio
  10-16 and spike_rel_peak 0.68-0.78; Bao-Dur's 280 real train clips stay below spike_ratio 7.6 (original) and
  3.5 (restored, full band; there spike_rel_peak reaches 1.4, so only the ratio is compared with real clips);
- lead/tail silence (energy VAD, evalcommon.activity) and the old in-silence noise floor (only with >= 100 ms
  of silent frames);
- VAD-free noise measures (trimmed takes have no silence for the VAD to measure): floor_p05_db / floor_p20_db
  = 5th/20th percentile of 25 ms frame levels over the whole clip (frames above -110 dBFS), floor_spread_db =
  their difference (a steady hiss makes the quiet frames all alike), quiet_flatness = spectral flatness
  (1-10 kHz) of the quietest 20 % of frames (white hiss ~0.9-1.0), snr_floor_db = loud level - floor_p05;
- cut onset/ending: level of the first/last 10 ms relative to the loud (95th pct) frames and to the floor;
- high-band energy: 10*log10(power above 11.5 kHz / total power), the same measure restore.py logs.
Real reference: train clips of the run's audio variant; the other variant's high-band numbers come from
<lab>/restore_report.json when present. The flag rules live in aggregate.artifact_flags (calibrated on all 280 of
Bao-Dur's real train clips: none fire on a real clip; the scorecard reports how many fire on the scored character's
own train clips). Writes <run>/metrics/artifacts.json; exit 3 on unscored takes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 4
HIGH_BAND_HZ = 11500.0
CLIP_LEVEL = 0.999
LSB = 1.0 / 32768
MIN_CLIP_PEAK = 0.1      # relative flat-top detection only for clips peaking at >= -20 dBFS
MIN_SILENT_FRAMES = 10   # 10 ms hop: 100 ms of silence needed for the in-silence noise floor
LIVE_DB = -110.0         # frames below this are digital silence (padding), not a noise floor
EDGE_S = 0.010


def measure(path: str) -> dict:
    import numpy as np
    import soundfile as sf
    info = sf.info(path)
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    finite = np.isfinite(x)
    out = {"rate": int(rate), "channels": int(info.channels), "subtype": info.subtype,
           "duration_s": len(x) / rate, "nonfinite_samples": int((~finite).sum())}
    x = np.where(finite, x, 0.0).astype(np.float32)
    data = np.nan_to_num(data)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    out["peak_dbfs"] = 20 * np.log10(peak + 1e-12)
    ceiling = min(CLIP_LEVEL, peak - 1.5 * LSB) if peak >= MIN_CLIP_PEAK else CLIP_LEVEL
    hot = np.any(np.abs(data) >= ceiling, axis=1)
    runs = [(a, b) for a, b in ec._runs(hot) if b - a >= 3]
    out["clip_ceiling"] = ceiling
    out["clip_runs"] = len(runs)
    out["clipped_fraction"] = float(sum(b - a for a, b in runs) / max(1, len(x)))
    out["full_scale_samples"] = int(np.sum(np.any(np.abs(data) >= CLIP_LEVEL, axis=1)))
    if len(x) > 1:
        step = np.abs(np.diff(x))
        out["max_step"] = float(step.max())
        out["step_p999"] = float(np.percentile(step, 99.9))
        out["step_ratio"] = out["max_step"] / max(out["step_p999"], 1e-6)
    if len(x) > 2 and peak >= 1e-4:
        d2 = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:]))
        k = int(np.argmax(d2))
        out["spike_ratio"] = float(d2[k] / max(float(np.percentile(d2, 99.9)), 1e-7))
        out["spike_rel_peak"] = float(d2[k] / peak)
        out["spike_at_s"] = (k + 1) / rate
    out["dc_offset"] = float(x.mean()) if x.size else 0.0
    act = ec.activity(x, rate)
    if act["start_s"] is None:
        out.update(lead_silence_s=None, tail_silence_s=None, speech_detected=False)
    else:
        out.update(lead_silence_s=act["start_s"], tail_silence_s=max(0.0, act["duration_s"] - act["end_s"]),
                   speech_detected=True)
    db, n, hop = act["frame_db"], act["n"], act["hop"]
    quiet = db[~act["mask"]]
    out["silent_frames"] = int(quiet.size)
    out["noise_floor_db"] = float(np.percentile(quiet, 10)) if quiet.size >= MIN_SILENT_FRAMES else None
    loud = act["loud_db"]
    out["loud_db"] = loud
    live = np.flatnonzero(db > LIVE_DB)
    if live.size >= 10:
        floor05, floor20 = np.percentile(db[live], [5, 20])
        out.update(floor_p05_db=float(floor05), floor_p20_db=float(floor20), floor_spread_db=float(floor20 - floor05),
                   snr_floor_db=float(loud - floor05))
        quietest = live[db[live] <= floor20]
        top = min(10000.0, 0.47 * rate)
        if top > 1500 and quietest.size:
            window = np.hanning(n).astype(np.float32)
            freqs = np.fft.rfftfreq(n, 1 / rate)
            band = (freqs >= 1000) & (freqs <= top)
            power = np.zeros(freqs.size, dtype=np.float64)
            count = 0
            for start in range(0, quietest.size, 256):
                idx = quietest[start:start + 256]
                frames = np.stack([x[i * hop:i * hop + n] for i in idx if i * hop + n <= len(x)]) if idx.size else None
                if frames is None or not len(frames):
                    continue
                power += np.sum(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0)
                count += len(frames)
            if count:
                spec = power[band] / count + 1e-30
                out["quiet_flatness"] = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    else:
        out.update(floor_p05_db=None, floor_p20_db=None, floor_spread_db=None, snr_floor_db=None)
    edge = max(1, int(EDGE_S * rate))
    if len(x) >= 2 * edge:
        def level(seg):
            return float(10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-20))
        onset, offset = level(x[:edge]), level(x[-edge:])
        out.update(onset_rel_db=onset - loud, offset_rel_db=offset - loud)
        if out.get("floor_p05_db") is not None:
            out.update(onset_above_floor_db=onset - out["floor_p05_db"], offset_above_floor_db=offset - out["floor_p05_db"])
    if rate / 2 > HIGH_BAND_HZ and x.size:
        power = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / rate)
        out["high_band_db"] = float(10 * np.log10(power[freqs > HIGH_BAND_HZ].sum() / (power.sum() + 1e-30) + 1e-12))
    else:
        out["high_band_db"] = None
        out["high_band_note"] = f"sample rate {rate} Hz has no band above {HIGH_BAND_HZ:.0f} Hz"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    args = parser.parse_args()

    settings = {"metric": "artifacts", "version": VERSION, "high_band_hz": HIGH_BAND_HZ, "clip_level": CLIP_LEVEL,
                "vad": [ec.ACTIVE_REL_DB, ec.ACTIVE_FLOOR_DB, ec.MIN_ACTIVE_S], "min_silent_frames": MIN_SILENT_FRAMES,
                "live_db": LIVE_DB, "edge_s": EDGE_S}
    cache = ec.Cache("artifacts", settings)

    def cached(path: str) -> dict:
        sha = ec.file_sha256(path)
        hit = cache.get(sha)
        if hit is None:
            hit = measure(path)
            cache.put(sha, hit)
        return {"sha256": sha, **hit}

    started = time.time()
    run = ec.load_run(args.run_dir)
    variant = run["audio_variant"]
    sets = ec.real_sets(variant)
    ec.log(f"artifacts: {len(sets['train'])} real train clips ({variant})")
    real, real_errors = {}, []
    for clip in sets["train"]:
        value, err = ec.guarded(cached, clip["path"])
        if err:
            real_errors.append({"id": clip["id"], **err})
        else:
            real[clip["id"]] = {k: v for k, v in value.items() if k != "sha256"}
    cache.save()
    fields = ("lead_silence_s", "tail_silence_s", "high_band_db", "noise_floor_db", "peak_dbfs", "clipped_fraction",
              "dc_offset", "max_step", "step_ratio", "spike_ratio", "spike_rel_peak", "floor_p05_db", "floor_spread_db",
              "quiet_flatness",
              "snr_floor_db", "onset_rel_db", "offset_rel_db", "onset_above_floor_db", "offset_above_floor_db")
    baseline = {f: ec.stats([r.get(f) for r in real.values()]) for f in fields}
    for f in ("floor_p05_db", "noise_floor_db"):
        values = [r.get(f) for r in real.values() if r.get(f) is not None]
        if values:
            import numpy as np
            baseline[f]["p99"] = float(np.percentile(values, 99))
    baseline["noise_floor_db"]["note"] = "clips with >= 100 ms of silence only"
    baseline["clips_with_clip_runs"] = sum(1 for r in real.values() if r.get("clip_runs"))
    baseline["n"] = len(real)

    other = {}
    if ec.RESTORE_REPORT.exists():
        report = ec.read_json(ec.RESTORE_REPORT)
        clips = report.get("clips", [])
        train_ids = {c["id"] for c in sets["train"]}
        other = {
            "source": str(ec.RESTORE_REPORT),
            "original_high_band_db": ec.stats([c.get("high_band_db_before") for c in clips if c["id"] in train_ids]),
            "restored_high_band_db": ec.stats([c.get("high_band_db_after") for c in clips if c["id"] in train_ids]),
        }

    takes = {}
    ec.log(f"artifacts: {len(run['takes'])} takes")
    for take in run["takes"]:
        value, err = ec.guarded(cached, take["wav"])
        takes[take["key"]] = err if err else value
    cache.save()

    out = {"metric": "artifacts", "settings": settings, "audio_variant": variant,
           "seconds": round(time.time() - started, 1), "baselines": {"train_real": baseline, "restore_report": other},
           "real": real, "real_errors": real_errors, "takes": takes}
    code = ec.finish_metric(args.run_dir, "artifacts", out, run)
    if any(e.get("unscored") for e in real_errors):
        ec.log(f"!! artifacts: {len(real_errors)} real clip(s) failed (resource errors); re-run this step")
        code = 3
    ec.log(f"artifacts: wrote {ec.metric_out_path(args.run_dir, 'artifacts')}")
    ec.exit_with(code)


if __name__ == "__main__":
    main()
