"""Run one upstream GPT-SoVITS dataset script in-process with this recipe's guards (used by prepare.py).

  python gsv_run_step.py --device cpu|cuda[:N] [--mem-gb 3] [--reserve-gb 8] [--wait-min 0]
                         [--threads 8] [--lexicon lexicon.json] --script GPT_SoVITS/prepare_datasets/<step>.py

The step reads its settings from environment variables exactly as webui.py passes them (inp_text, opt_dir,
exp_name, i_part, all_parts, is_half, bert_pretrained_dir, cnhubert_base_dir, sv_path, pretrained_s2G,
s2config_path, version ...); prepare.py sets them. Before handing over we: chdir to the GPT-SoVITS root, put the
repo on sys.path (what webui's users.pth does), verify the device (CPU mode must not see a GPU), cap VRAM in GPU
mode, and optionally inject lexicon.json into the English g2p dictionary (step 1 only).
Exit codes: 0 ok, 3 CUDA out of memory, 4 VRAM guard refused to start, anything else = step error.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-gb", type=float, default=3.0)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--wait-min", "--wait-vram-min", dest="wait_min", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lexicon", default=None)
    args = ap.parse_args()
    args.lexicon = str(C.abspath(args.lexicon)) if args.lexicon else None  # before bootstrap chdirs to the repo
    device = C.normalize_device(args.device)

    C.bootstrap_gsv_runtime(device)
    import torch

    C.assert_device(device)
    if device != "cpu":
        C.cap_gpu_memory(args.mem_gb, args.reserve_gb, args.wait_min)
    torch.set_num_threads(max(1, args.threads))
    if args.lexicon:
        n = C.install_g2p_lexicon(C.load_lexicon(args.lexicon))
        C.log(f"lexicon: {n} ARPAbet overrides injected")

    script = Path(args.script)
    if not script.is_absolute():
        script = C.GSV_ROOT / script
    if not script.is_file():
        raise SystemExit(f"step script not found: {script}")
    sys.argv = [str(script)]
    t0 = time.time()
    code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except Exception as exc:
        code = 3 if C.is_oom(exc) else 1
        if code == 1:
            raise
        print(f"[gsv] CUDA out of memory in {script.name}: {exc}", flush=True)
    finally:
        report = {"step": script.name, "device": device, "seconds": round(time.time() - t0, 1)}
        report.update(C.peak_memory_report(device))
        print("[gsv-step] " + json.dumps(report), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
