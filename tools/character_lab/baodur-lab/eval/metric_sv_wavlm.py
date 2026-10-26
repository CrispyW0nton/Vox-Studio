"""Speaker similarity with WavLM-base-plus-SV (x-vector, 512-d): the identity model no engine conditions on.

Run with the seed-vc venv (transformers 4.46.3, torch 2.8):
  set CUDA_VISIBLE_DEVICES=-1  (CPU default)
  seed-vc/.venv/Scripts/python.exe metric_sv_wavlm.py --run-dir <lab>/eval/<run> [--device cuda]
(<lab>: the lab of the character VOICE_LAB_CHARACTER selects; impostor voices from <lab>/eval/eval_character.json)

Weights: Vox Studio's local copy of microsoft/wavlm-base-plus-sv (the model its audition uses) under
%LOCALAPPDATA%/VoxStudio/engines/voxcpm2/cache, read in place with local_files_only (nothing is downloaded or
written there). GPT-SoVITS v2Pro conditions on ERes2NetV2 and Seed-VC V2 on CAM++; neither touches WavLM,
so it is the uncoupled third opinion for engine comparisons. The feature encoder group-normalises its first
conv (no bias), so playback level does not move the score.
Side output: the mean over time of hidden layer --layer (default 6 of 12), written for takes and the delivery
map's reference clips. The aggregator uses it as a delivery (expression) signal ONLY if it first separates
the character's own reference clips by delivery label in a leave-one-out test that beats chance (see aggregate.py).
Audio is trimmed to the speech span first. Writes <run>/metrics/sv_wavlm.json. Exit 3 on unscored takes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

VERSION = 1
VOX_CACHE = ec.LOCALAPPDATA / "VoxStudio" / "engines" / "voxcpm2" / "cache"
REPO = "models--microsoft--wavlm-base-plus-sv"


def find_snapshot() -> Path:
    for root in (VOX_CACHE / "hub", VOX_CACHE):
        base = root / REPO / "snapshots"
        if not base.is_dir():
            continue
        for snap in sorted(base.iterdir()):
            if (snap / "config.json").exists() and any((snap / w).exists() for w in ("pytorch_model.bin", "model.safetensors")):
                return snap
    raise SystemExit(f"WavLM-SV weights not found under {VOX_CACHE} (needs {REPO} with config.json + weights); "
                     "this step is optional: leave it out with --metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ec.common_args(parser)
    parser.add_argument("--layer", type=int, default=6, help="hidden layer pooled for the delivery side vector")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import WavLMForXVector

    device = ec.torch_device(args.device, cap_gb=2.0, threads=args.threads)
    torch.set_grad_enabled(False)
    snapshot = find_snapshot()
    model = WavLMForXVector.from_pretrained(str(snapshot), local_files_only=True).eval().to(device)
    ec.log(f"WavLM-SV loaded from {snapshot} on {device}")

    def embed_file(path: str):
        x, _ = ec.read_mono(path, 16000)
        x = ec.speech_segment(x, 16000)
        if len(x) < 16000 // 2:
            x = np.pad(x, (0, 16000 // 2 - len(x)))
        wav = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(device)
        out = model(input_values=wav, output_hidden_states=True)
        emb = out.embeddings[0].float().cpu().numpy()
        hidden = out.hidden_states[args.layer][0].float().mean(0).cpu().numpy()
        return emb, hidden

    settings = {"metric": "sv_wavlm", "version": VERSION, "snapshot": snapshot.name, "trim": "speech_span+0.1s",
                "embedding": "xvector-512"}
    side = {"name": "delivery", "settings": {"layer": args.layer, "pool": "mean"}}
    ec.exit_with(ec.run_speaker_metric("sv_wavlm", settings, embed_file, args.run_dir, device=device, side=side))


if __name__ == "__main__":
    main()
