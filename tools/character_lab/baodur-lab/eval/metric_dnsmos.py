"""Naturalness / signal quality with DNSMOS (Microsoft's DNS-challenge MOS predictors), CPU only.

Run with the GPT-SoVITS venv (onnxruntime-gpu 1.30, used with its CPU provider; librosa 0.10.2):
  set CUDA_VISIBLE_DEVICES=-1
  GPT-SoVITS/.venv/Scripts/python.exe metric_dnsmos.py --run-dir <lab>/eval/<run>

Models: Seed-VC's local copies in seed-vc/baselines/dnsmos (nothing is downloaded): sig_bak_ovr.onnx (ITU-T
P.835: SIG speech signal, BAK background, OVRL overall; raw outputs mapped with Microsoft's non-personalised
polynomials) and model_v8.onnx (P.808 overall MOS). Features follow Microsoft's dnsmos_local.py: 16 kHz,
9.01 s windows hopped by 1 s (shorter audio is looped to 9.01 s), P.808 on a 120-band librosa mel spectrogram
(n_fft 321, hop 160, power_to_db ref=max, (+40)/40). Seed-VC's dnsmos_computor.py builds that mel with
torchaudio on CUDA only; librosa gives the reference implementation on CPU. Audio is trimmed to the speech span
(+0.1 s) first, so a take's lead/tail silence (flagged by the artifacts step) does not move the scores.
Speech quality, not prosody: a pitch-flattened copy of a real clip keeps its scores, white hiss at -30 dBFS
drops BAK from ~4.1 to ~2.1.
Real reference: the character's clips (VOICE_LAB_CHARACTER) of the run's primary variant (train = the distribution
the aggregator anchors on; val/test = ground truth for held-out lines only). Cached by SHA-256 in
<lab>/eval/_cache/dnsmos.
Writes <run>/metrics/dnsmos.json; exit 3 when takes stayed unscored after a retry (resource errors).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 1
DNSMOS_DIR = ec.VOICE_LAB / "seed-vc" / "baselines" / "dnsmos"
RATE = 16000
INPUT_LENGTH = 9.01
# Microsoft dnsmos_local.py, is_personalized_MOS=False
POLY = {"SIG": (-0.08397278, 1.22083953, 0.0052439), "BAK": (-0.13166888, 1.60915514, -0.39604546),
        "OVRL": (-0.06766283, 1.11546468, 0.04602535)}
SCORES = ("SIG", "BAK", "OVRL", "P808")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    args = parser.parse_args()

    import numpy as np
    import librosa
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, args.threads)
    options.inter_op_num_threads = 1
    paths = {"p835": DNSMOS_DIR / "sig_bak_ovr.onnx", "p808": DNSMOS_DIR / "model_v8.onnx"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"DNSMOS models not found: {missing}; this step is optional (leave it out with --metrics)")
    p835 = ort.InferenceSession(str(paths["p835"]), sess_options=options, providers=["CPUExecutionProvider"])
    p808 = ort.InferenceSession(str(paths["p808"]), sess_options=options, providers=["CPUExecutionProvider"])
    poly = {k: np.poly1d(v) for k, v in POLY.items()}
    ec.log(f"DNSMOS P.835 + P.808 on CPU (onnxruntime {ort.__version__}, {args.threads} threads)")

    def score(path: str) -> dict:
        x, _ = ec.read_mono(path, RATE)
        x = ec.speech_segment(x, RATE).astype(np.float32)
        actual = len(x) / RATE
        if not len(x) or not np.any(x):
            raise ValueError("silent audio")
        need = int(INPUT_LENGTH * RATE)
        while len(x) < need:
            x = np.append(x, x)
        hops = int(np.floor(len(x) / RATE) - INPUT_LENGTH) + 1
        values = {k: [] for k in SCORES}
        for i in range(hops):
            seg = x[int(i * RATE):int((i + INPUT_LENGTH) * RATE)]
            if len(seg) < need:
                continue
            mel = librosa.feature.melspectrogram(y=seg[:-160], sr=RATE, n_fft=321, hop_length=160, n_mels=120)
            mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
            values["P808"].append(float(p808.run(None, {"input_1": mel.T.astype(np.float32)[np.newaxis]})[0][0][0]))
            sig, bak, ovr = p835.run(None, {"input_1": seg.astype(np.float32)[np.newaxis]})[0][0]
            for key, raw in (("SIG", sig), ("BAK", bak), ("OVRL", ovr)):
                values[key].append(float(poly[key](raw)))
        if not values["OVRL"]:
            raise ValueError("no 9 s window")
        return {**{k: float(np.mean(v)) for k, v in values.items()}, "windows": len(values["OVRL"]),
                "speech_s": actual}

    settings = {"metric": "dnsmos", "version": VERSION, "models": ["sig_bak_ovr.onnx", "model_v8.onnx"],
                "mel": "librosa-120-321-160", "trim": "speech_span+0.1s", "personalized": False}
    cache = ec.Cache("dnsmos", settings)

    def cached(path: str) -> dict:
        sha = ec.file_sha256(path)
        hit = cache.get(sha)
        if hit is None:
            hit = score(path)
            cache.put(sha, hit)
        return {"sha256": sha, **hit}

    started = time.time()
    run = ec.load_run(args.run_dir)
    variant = run["audio_variant"]
    sets = ec.real_sets(variant)
    real, real_errors = {}, []
    for role in ("train", "val", "test"):
        ec.log(f"dnsmos: {len(sets[role])} real {role} clips ({variant})")
        for clip in sets[role]:
            value, err = ec.guarded(cached, clip["path"])
            if err:
                real_errors.append({"id": clip["id"], **err})
                continue
            real[clip["id"]] = {"split": role, **{k: v for k, v in value.items() if k != "sha256"}}
    cache.save()
    train = [r for r in real.values() if r["split"] == "train"]
    baseline = {k: ec.stats([r.get(k) for r in train]) for k in SCORES}

    takes = {}
    ec.log(f"dnsmos: {len(run['takes'])} takes")
    for take in run["takes"]:
        value, err = ec.guarded(cached, take["wav"])
        takes[take["key"]] = err if err else value
    cache.save()
    out = {"metric": "dnsmos", "settings": settings, "audio_variant": variant,
           "seconds": round(time.time() - started, 1), "baselines": {"train_real": baseline},
           "real": real, "real_errors": real_errors, "takes": takes}
    code = ec.finish_metric(args.run_dir, "dnsmos", out, run)
    if any(e.get("unscored") for e in real_errors):
        ec.log(f"!! dnsmos: {len(real_errors)} real clip(s) failed (resource errors); re-run this step")
        code = 3
    ec.log(f"dnsmos: real train OVRL median {baseline['OVRL'].get('median', float('nan')):.2f}; "
           f"wrote {ec.metric_out_path(args.run_dir, 'dnsmos')}")
    ec.exit_with(code)


if __name__ == "__main__":
    main()
