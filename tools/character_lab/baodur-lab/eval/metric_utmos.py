"""Naturalness MOS with UTMOSv2 (UTokyo-SaruLab, VoiceMOS Challenge 2024 system T05), fold 0 only.

Run with the eval-extra venv (Python 3.11, torch 2.8.0+cu128, utmosv2 @ cc2700d, transformers 4.57.6, timm 1.0.25):
  set CUDA_VISIBLE_DEVICES=-1
  voice-lab/eval-extra/.venv/Scripts/python.exe metric_utmos.py --run-dir <lab>/eval/<run> [--real all|train|none]
(<lab>: the lab of the character VOICE_LAB_CHARACTER selects, default BaoDurLab)
(run_eval.py runs it as the optional step "utmos": --metrics ...,utmos).

Model: config fusion_stage3 (wav2vec2-base SSL branch + four EfficientNetV2-S mel-spectrogram branches + a
data-domain id), fold-0 checkpoint voice-lab/eval-extra/models/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth
(huggingface.co/sarulab-speech/UTMOSv2, MIT; sha256 in eval-extra/PROVENANCE.json). Every weight comes from that
checkpoint, loaded strict: the wav2vec2-base and timm backbones are only built from their configs
(eval-extra/models/facebook--wav2vec2-base holds config.json + preprocessor_config.json; timm pretrained=False),
so nothing is downloaded at run time.
Prediction (docs/inference.md): data-domain id --predict-dataset (default "sarulab"), --repetitions 5 (each draws
a random 3 s SSL crop and two 1.4 s spectrogram frames with inner mixup, as in the paper; the score is their mean,
identical to num_repetitions=5, and their sd is kept as the estimate's noise). The numpy RNG is seeded per file from
its SHA-256 (+ --seed), so a file always gets the same score in any order.
Audio: 16 kHz mono, trimmed to the speech span (+0.1 s) like dnsmos (--silence trim, matching UTMOSv2's training
trim); the library's remove_silent_section (drops every >= 0.3 s run of samples below 0.1: level dependent, it also
cuts his pauses) is used only with --silence utmos. On cuda the library predicts under fp16 autocast, so cuda and
cpu results are cached apart.
Scale: MOS 1-5 in the chosen domain; naturalness of (synthetic) speech, not identity or emotion. Read it against the
character's own clips (game audio; Bao-Dur's are 22 kHz MP3-sourced): the scorecard reports a take's percentile in
the character's train distribution.
Real reference (--real, default all): the character's clips of the run's primary variant (train = the distribution
the aggregator anchors on; val/test = ground truth for held-out lines only; none = takes only, for smoke tests).
Cost: the CPU smoke test took ~11 s per repetition per clip on 4 threads of the shared machine (~1 min per clip at 5
repetitions), so the first --real all baseline (Bao-Dur ~306 clips, Jolee ~459) is hours on CPU: run it with
--device cuda when the GPU is free (or --real train / none). Model load ~30 s, ~1.7 GB RAM on CPU.
Cached by SHA-256 + settings in <lab>/eval/_cache/utmos. Writes <run>/metrics/utmos.json; exit 3 when takes
stayed unscored after a retry (resource errors).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 1
EXTRA_DIR = Path(os.environ.get("BAODUR_EVAL_EXTRA", str(ec.VOICE_LAB / "eval-extra")))
MODELS = EXTRA_DIR / "models"
CONFIG = "fusion_stage3"
CHECKPOINT_SEED = 42          # the published checkpoints are fold{n}_s42
W2V_NAME = "facebook/wav2vec2-base"
W2V_DIR = MODELS / "facebook--wav2vec2-base"
CHECKPOINT_SHA256 = {0: "c8149d988e4bbf3f347e6966b5d769de347a5f8c59ffca1dc4bd4bf5b8585e57"}
RATE = 16000
DOMAINS = ("sarulab", "bvcc", "blizzard2008", "blizzard2009", "blizzard2010-EH1", "blizzard2010-EH2",
           "blizzard2010-ES1", "blizzard2010-ES3", "blizzard2011", "somos")


def checkpoint_path(fold: int) -> Path:
    return MODELS / "utmosv2" / "models" / CONFIG / f"fold{fold}_s{CHECKPOINT_SEED}_best_model.pth"


def offline_env() -> None:
    """Library caches inside the install folder; hub access off (every file is local)."""
    os.environ["HF_HOME"] = str(EXTRA_DIR / "hf-home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["UTMOSV2_CHACHE"] = str(MODELS / "utmosv2")  # sic: the library's variable name
    for pattern in (r".*pin_memory.*", r".*torch\.cuda\.amp\.autocast.*", r".*CUDA is not available.*",
                    r".*gradient_checkpointing.*"):
        warnings.filterwarnings("ignore", message=pattern)


def load_model(fold: int, device: str):
    """UTMOSv2 fusion_stage3 with every weight from the local fold checkpoint (strict load), no downloads."""
    ckpt = checkpoint_path(fold)
    if not ckpt.exists():
        raise SystemExit(f"UTMOSv2 fold {fold} checkpoint not found: {ckpt} (the eval-extra install holds fold 0 only)")
    if not (W2V_DIR / "config.json").exists():
        raise SystemExit(f"wav2vec2-base config not found in {W2V_DIR}")
    import timm
    import transformers
    import utmosv2
    import utmosv2.model.ssl as ssl_module

    create = timm.create_model

    def create_without_download(*args, **kwargs):  # backbone weights come from the checkpoint
        kwargs["pretrained"] = False
        return create(*args, **kwargs)

    class LocalFeatureExtractor:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            if name != W2V_NAME:
                raise ValueError(f"unexpected SSL model {name}")
            return transformers.AutoFeatureExtractor.from_pretrained(str(W2V_DIR))

    class ConfigOnlyModel:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            if name != W2V_NAME:
                raise ValueError(f"unexpected SSL model {name}")
            return transformers.AutoModel.from_config(transformers.AutoConfig.from_pretrained(str(W2V_DIR)))

    saved = (ssl_module.AutoFeatureExtractor, ssl_module.AutoModel)
    timm.create_model = create_without_download
    ssl_module.AutoFeatureExtractor, ssl_module.AutoModel = LocalFeatureExtractor, ConfigOnlyModel
    try:
        model = utmosv2.create_model(pretrained=True, config=CONFIG, fold=fold, checkpoint_path=ckpt,
                                     seed=CHECKPOINT_SEED, device=device)
    finally:
        timm.create_model = create
        ssl_module.AutoFeatureExtractor, ssl_module.AutoModel = saved
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    parser.add_argument("--real", choices=("all", "train", "none"), default="all",
                        help="real clips scored as the baseline: all (train distribution + val/test ground truth), "
                             "train only, or none (takes only)")
    parser.add_argument("--fold", type=int, default=0, help="checkpoint fold (only fold 0 is installed)")
    parser.add_argument("--predict-dataset", choices=DOMAINS, default="sarulab", help="UTMOSv2 data-domain id")
    parser.add_argument("--repetitions", type=int, default=5, help="random-crop repetitions averaged per file")
    parser.add_argument("--silence", choices=("trim", "utmos", "none"), default="trim",
                        help="trim = speech span +0.1 s (default); utmos = the library's remove_silent_section")
    parser.add_argument("--seed", type=int, default=1234, help="mixed into each file's RNG seed")
    parser.add_argument("--gpu-cap-gb", type=float, default=3.0, help="per-process GPU memory cap on cuda")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")

    offline_env()
    import numpy as np
    import torch

    device = ec.torch_device(args.device, cap_gb=args.gpu_cap_gb, threads=args.threads)
    expected = CHECKPOINT_SHA256.get(args.fold)
    ckpt_sha = ec.file_sha256(checkpoint_path(args.fold)) if checkpoint_path(args.fold).exists() else None
    if expected and ckpt_sha and ckpt_sha != expected:
        raise SystemExit(f"UTMOSv2 fold {args.fold} checkpoint sha256 {ckpt_sha} differs from the installed one {expected}")
    started = time.time()
    loaded: list = []

    def get_model():  # loaded on the first cache miss (a fully cached re-run never loads the 0.8 GB model)
        if not loaded:
            t0 = time.time()
            loaded.append(load_model(args.fold, device))
            ec.log(f"UTMOSv2 {CONFIG} fold {args.fold} on {device} ({args.threads} threads), domain "
                   f"{args.predict_dataset}, {args.repetitions} repetitions, silence {args.silence} "
                   f"(loaded in {time.time() - t0:.1f} s)")
        return loaded[0]

    settings = {"metric": "utmos", "version": VERSION, "config": CONFIG, "fold": args.fold,
                "checkpoint_sha256": ckpt_sha, "predict_dataset": args.predict_dataset,
                "repetitions": args.repetitions, "silence": args.silence, "rng_seed": args.seed,
                "amp_fp16": device == "cuda"}
    cache = ec.Cache("utmos", settings)

    def score(path: str, sha: str) -> dict:
        x, _ = ec.read_mono(path, RATE)
        if args.silence == "trim":
            x = ec.speech_segment(x, RATE)
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not len(x) or not np.any(x) or not np.all(np.isfinite(x)):
            raise ValueError("silent or non-finite audio")
        model = get_model()
        seed = (int(sha[:8], 16) ^ args.seed) & 0x7FFFFFFF
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        reps = []
        for _ in range(args.repetitions):  # same RNG stream as one call with num_repetitions=n, values kept
            value = model.predict(data=x, sr=RATE, predict_dataset=args.predict_dataset, device=device,
                                  num_workers=0, batch_size=1, num_repetitions=1,
                                  remove_silent_section=args.silence == "utmos", verbose=False)
            reps.append(float(np.asarray(value).reshape(-1)[0]))
        if not all(np.isfinite(reps)):
            raise ValueError(f"non-finite prediction {reps}")
        return {"mos": float(np.mean(reps)), "rep_sd": float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0,
                "reps": [round(r, 4) for r in reps], "speech_s": round(len(x) / RATE, 3)}

    def cached(path: str) -> dict:
        sha = ec.file_sha256(path)
        hit = cache.get(sha)
        if hit is None:
            hit = score(path, sha)
            cache.put(sha, hit)
        return {"sha256": sha, **hit}

    run = ec.load_run(args.run_dir)
    variant = run["audio_variant"]
    roles = {"all": ("train", "val", "test"), "train": ("train",), "none": ()}[args.real]
    sets = ec.real_sets(variant) if roles else {}
    real, real_errors = {}, []
    for role in roles:
        ec.log(f"utmos: {len(sets[role])} real {role} clips ({variant})")
        for i, clip in enumerate(sets[role], 1):
            value, err = ec.guarded(cached, clip["path"])
            if err:
                real_errors.append({"id": clip["id"], **err})
                continue
            real[clip["id"]] = {"split": role, **{k: v for k, v in value.items() if k != "sha256"}}
            if i % 25 == 0:
                ec.log(f"  {i}/{len(sets[role])}")
                cache.save()
    cache.save()
    train = [r["mos"] for r in real.values() if r["split"] == "train"]
    baseline = ec.stats(train)

    takes = {}
    ec.log(f"utmos: {len(run['takes'])} takes")
    for take in run["takes"]:
        value, err = ec.guarded(cached, take["wav"])
        takes[take["key"]] = err if err else value
    cache.save()
    out = {"metric": "utmos", "settings": settings, "audio_variant": variant, "device": device,
           "real_scope": args.real, "seconds": round(time.time() - started, 1),
           "baselines": {"train_real": baseline}, "real": real, "real_errors": real_errors, "takes": takes}
    code = ec.finish_metric(args.run_dir, "utmos", out, run)
    if any(e.get("unscored") for e in real_errors):
        ec.log(f"!! utmos: {len(real_errors)} real clip(s) failed (resource errors); re-run this step")
        code = 3
    scored = [v["mos"] for v in takes.values() if isinstance(v, dict) and v.get("mos") is not None]
    ec.log(f"utmos: takes median {np.median(scored) if scored else float('nan'):.2f} (n {len(scored)}); real train "
           f"median {baseline.get('median', float('nan')):.2f} (n {baseline.get('n', 0)}); "
           f"wrote {ec.metric_out_path(args.run_dir, 'utmos')}")
    ec.exit_with(code)


if __name__ == "__main__":
    main()
