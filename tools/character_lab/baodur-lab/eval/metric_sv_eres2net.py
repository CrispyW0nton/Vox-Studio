"""Speaker similarity with GPT-SoVITS's ERes2NetV2 speaker-verification model.

Run with the GPT-SoVITS venv (it has the repo's eres2net code and the sv checkpoint):
  set CUDA_VISIBLE_DEVICES=-1  (CPU default)
  GPT-SoVITS/.venv/Scripts/python.exe metric_sv_eres2net.py --run-dir <lab>/eval/<run> [--device cuda]
(<lab>: the lab of the character VOICE_LAB_CHARACTER selects; impostor voices from <lab>/eval/eval_character.json)

The model is loaded through GPT_SoVITS/sv.py (class SV, checkpoint
pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt, ERes2NetV2 baseWidth=24 scale=4 expansion=4).
Scoring uses the model's speaker-verification head, forward(): 80-bin Kaldi fbank at 16 kHz, dither 0,
per-utterance mean normalisation (the 3D-Speaker inference recipe) -> 192-d embedding -> cosine. The mean
normalisation removes a constant gain exactly, so playback level does not move this score.
COUPLING: GPT-SoVITS v2Pro/v2ProPlus condition their generator on forward3() of this same network, so the
aggregator leaves this model out of the identity of GSV v2Pro* takes ("uncoupled identity").
Audio is trimmed to the speech span first. Writes <run>/metrics/sv_eres2net.json (raw cosines per reference
variant + real/impostor baselines). Exit 3 when takes stayed unscored after a retry (resource errors).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

GSV_ROOT = ec.VOICE_LAB / "GPT-SoVITS"
VERSION = 1


def load_sv(device: str):
    """Instantiate GPT-SoVITS's SV loader. sv.py resolves its checkpoint and eres2net folder
    relative to the working directory, so import and construct it from the repo root."""
    previous = os.getcwd()
    os.chdir(GSV_ROOT)
    try:
        package_dir = str(GSV_ROOT / "GPT_SoVITS")
        if package_dir not in sys.path:
            sys.path.append(package_dir)
        import sv as sv_module  # GPT_SoVITS/sv.py
        if Path(sv_module.__file__).resolve().parent != (GSV_ROOT / "GPT_SoVITS").resolve():
            raise SystemExit(f"imported the wrong sv module: {sv_module.__file__}")
        import kaldi as kaldi_module  # GPT_SoVITS/eres2net/kaldi.py (sv.py put it on sys.path)
        model = sv_module.SV(device, False)
    finally:
        os.chdir(previous)
    return model, kaldi_module, (GSV_ROOT / sv_module.sv_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    args = parser.parse_args()

    import numpy as np
    import torch

    device = ec.torch_device(args.device, cap_gb=2.0, threads=args.threads)
    torch.set_grad_enabled(False)
    sv, kaldi, checkpoint = load_sv(device)
    net = sv.embedding_model.eval()
    ec.log(f"ERes2NetV2 loaded from {checkpoint} on {device}")

    def embed_file(path: str) -> np.ndarray:
        x, _ = ec.read_mono(path, 16000)
        x = ec.speech_segment(x, 16000)
        if len(x) < 16000 // 4:  # fbank needs a few frames; pad very short takes
            x = np.pad(x, (0, 16000 // 4 - len(x)))
        wav = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(device)
        feat = kaldi.fbank(wav, num_mel_bins=80, sample_frequency=16000, dither=0)
        feat = feat - feat.mean(0, keepdim=True)
        emb = net(feat.unsqueeze(0))  # forward(): (1, 192)
        return emb[0].float().cpu().numpy()

    settings = {"metric": "sv_eres2net", "version": VERSION, "checkpoint": checkpoint.name,
                "features": "fbank80-16k-cmn", "trim": "speech_span+0.1s", "embedding": "forward-192d"}
    ec.exit_with(ec.run_speaker_metric("sv_eres2net", settings, embed_file, args.run_dir, device=device))


if __name__ == "__main__":
    main()
