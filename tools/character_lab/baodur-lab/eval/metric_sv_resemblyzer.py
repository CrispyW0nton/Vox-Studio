"""Speaker similarity with Resemblyzer (GE2E d-vector, 256-d), a second, independent opinion.

Run with the seed-vc venv (Resemblyzer 0.1.4 and its bundled pretrained.pt are installed there):
  set CUDA_VISIBLE_DEVICES=-1  (CPU default)
  seed-vc/.venv/Scripts/python.exe metric_sv_resemblyzer.py --run-dir <lab>/eval/<run> [--device cuda]
(<lab>: the lab of the character VOICE_LAB_CHARACTER selects; impostor voices from <lab>/eval/eval_character.json)

That venv has setuptools 84, which no longer ships pkg_resources, and webrtcvad 2.0.10 (imported by
Resemblyzer) still does `import pkg_resources` just to read its own version. A tiny in-process shim
provides that one call so nothing in the venv has to change.
Preprocessing is Resemblyzer's own preprocess_wav (16 kHz, -30 dBFS increase-only, webrtcvad long-silence trim).
Its encoder sees linear mel power, so the embedding moves with playback level (a -12 dB copy of a real clip
lost ~0.4 identity). Takes are therefore first scaled so their active-speech level (evalcommon.active_rms_db)
equals the median of the character's real train clips of the run's primary variant; real clips keep their native
level (their spread is part of the real baseline) so the cached real embeddings stay valid.
Writes <run>/metrics/sv_resemblyzer.json. Exit 3 when takes stayed unscored after a retry.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 1
LEVEL_VERSION = 1


def ensure_pkg_resources() -> None:
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass
    from importlib import metadata

    class _Distribution:
        def __init__(self, name: str):
            self.project_name = name
            try:
                self.version = metadata.version(name)
            except metadata.PackageNotFoundError:
                self.version = "0"

    shim = types.ModuleType("pkg_resources")
    shim.get_distribution = _Distribution  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


def real_level_target(variant: str) -> float:
    """Median active-speech level (dBFS, 16 kHz) of the real train clips of a variant; cached per file."""
    import numpy as np
    cache = ec.Cache("levels", {"metric": "levels", "version": LEVEL_VERSION, "rate": 16000,
                                "vad": [ec.ACTIVE_REL_DB, ec.ACTIVE_FLOOR_DB, ec.MIN_ACTIVE_S]})
    levels = []
    for clip in ec.real_sets(variant)["train"]:
        sha = ec.file_sha256(clip["path"])
        hit = cache.get(sha)
        if hit is None:
            value, err = ec.guarded(lambda p: ec.active_rms_db(ec.read_mono(p, 16000)[0], 16000), clip["path"])
            if err:
                continue
            hit = {"active_rms_db": value}
            cache.put(sha, hit)
        if hit.get("active_rms_db") is not None:
            levels.append(hit["active_rms_db"])
    cache.save()
    if not levels:
        raise SystemExit(f"no real {variant} train clip levels")
    return float(np.median(levels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    args = parser.parse_args()

    import numpy as np
    import torch

    ensure_pkg_resources()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from resemblyzer import VoiceEncoder, preprocess_wav

    device = ec.torch_device(args.device, cap_gb=1.0, threads=args.threads)
    torch.set_grad_enabled(False)
    encoder = VoiceEncoder(device=device, verbose=False)
    ec.log(f"Resemblyzer VoiceEncoder on {device}")
    run = ec.load_run(args.run_dir)
    target_db = round(real_level_target(run["audio_variant"]), 1)
    ec.log(f"takes are level-matched to {target_db:.1f} dBFS active speech (median of real {run['audio_variant']} train clips)")

    def embed_wave(x: np.ndarray) -> np.ndarray:
        wav = preprocess_wav(x, source_sr=None)
        if len(wav) < 16000 // 2:  # VAD removed nearly everything: fall back to the raw clip
            wav = x
        return encoder.embed_utterance(wav)

    def embed_file(path: str) -> np.ndarray:
        x, _ = ec.read_mono(path, 16000)
        return embed_wave(x)

    def embed_take(path: str) -> np.ndarray:
        x, _ = ec.read_mono(path, 16000)
        level = ec.active_rms_db(x, 16000)
        if level is not None:
            x = (x * np.float32(10 ** ((target_db - level) / 20))).astype(np.float32)
        return embed_wave(x)

    settings = {"metric": "sv_resemblyzer", "version": VERSION, "model": "resemblyzer-0.1.4-pretrained",
                "preprocess": "preprocess_wav-16k"}
    take_settings = {"take_level": "active-rms-to-real-train-median", "target_db": target_db,
                     "level_version": LEVEL_VERSION}
    ec.exit_with(ec.run_speaker_metric("sv_resemblyzer", settings, embed_file, args.run_dir, device=device,
                                       embed_take=embed_take, take_settings=take_settings))


if __name__ == "__main__":
    main()
