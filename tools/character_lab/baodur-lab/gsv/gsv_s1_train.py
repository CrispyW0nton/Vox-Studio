"""In-process launcher for GPT_SoVITS/s1_train.py (GPT / text-to-semantic fine-tune) with this recipe's guards.

Runs upstream s1_train.main() in this process (Lightning trains a single GPU in-process, no child scripts), after
capping this process's CUDA allocator and verifying the device. Used by train.py.

Single-GPU without torch.distributed: upstream asks Lightning for DDPStrategy(process_group_backend="gloo") whenever
CUDA is available, and the torch 2.8.0+cu128 Windows build here cannot create a gloo device (see gsv_s2_train.py).
With one visible GPU we let Lightning pick its single-device strategy instead ("auto"), and give upstream's
DistributedBucketSampler num_replicas=1, rank=0 explicitly (it otherwise calls dist.get_world_size() whenever CUDA
is available). For one rank this is numerically the same training.

  python gsv_s1_train.py --config <exp>/configs/s1.yaml --device cuda --mem-gb 6 [--reserve-gb 8] [--wait-min 0]
                         [--smoke-max-steps N] [--report out.json]

Exit codes: 0 ok, 3 CUDA out of memory (train.py retries with a smaller batch), 4 VRAM guard refused to start.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-gb", type=float, default=6.0)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--wait-min", "--wait-vram-min", dest="wait_min", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke-max-steps", type=int, default=0, help="limit_train_batches for smoke tests")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    args.config, args.report = str(C.abspath(args.config)), (str(C.abspath(args.report)) if args.report else None)
    device = C.normalize_device(args.device)

    C.bootstrap_gsv_runtime(device)
    sys.path.insert(0, str(C.GSV_PKG))  # upstream runs with GPT_SoVITS/ as the script directory
    config = str(Path(args.config).resolve())
    sys.argv = [str(C.GSV_PKG / "s1_train.py"), "--config_file", config]
    import s1_train  # noqa: E402  copies _CUDA_VISIBLE_DEVICES (set by train.py in GPU mode only)

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    C.assert_device(device)
    report: dict = {"config": config, "device": device, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if device != "cpu":
        report["vram"] = C.cap_gpu_memory(args.mem_gb, args.reserve_gb, args.wait_min)
    torch.set_num_threads(max(1, args.threads))
    if args.smoke_max_steps > 0:
        s1_train.Trainer = functools.partial(s1_train.Trainer, limit_train_batches=args.smoke_max_steps)
    import AR.data.data_module as data_module  # noqa: E402

    s1_train.DDPStrategy = lambda *_a, **_k: "auto"
    data_module.DistributedBucketSampler = functools.partial(data_module.DistributedBucketSampler,
                                                             num_replicas=1, rank=0)

    code, t0 = 0, time.time()
    try:
        s1_train.main(argparse.Namespace(config_file=config))
    except Exception as exc:  # torch.OutOfMemoryError or a cuBLAS/cuDNN allocation RuntimeError -> exit 3
        code = 3 if C.is_oom(exc) else 1
        if code == 1:
            raise
        print(f"[gsv] CUDA out of memory in GPT training: {exc}", flush=True)
    finally:
        report.update({"seconds": round(time.time() - t0, 1), "exit_code": code})
        report.update(C.peak_memory_report(device))
        print("[gsv-s1] " + json.dumps(report), flush=True)
        if args.report:
            C.write_json(args.report, report)
    sys.exit(code)


if __name__ == "__main__":
    main()
