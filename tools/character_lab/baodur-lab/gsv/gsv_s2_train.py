"""In-process launcher for GPT_SoVITS/s2_train.py (SoVITS fine-tune) with this recipe's guards (used by train.py).

Why not `python GPT_SoVITS/s2_train.py --config cfg` like webui.py: upstream main() mp.spawn()s one child per GPU,
so a VRAM cap set in the launching process would not apply to the process that actually trains. We import the
module instead (it parses --config at import time, exactly as upstream) and call its run(rank=0, n_gpus=1, hps)
- the same code path mp.spawn runs for a single GPU - after capping this process's CUDA allocator.

Single-GPU without torch.distributed: the torch 2.8.0+cu128 Windows build in the GPT-SoVITS venv cannot create a
gloo device ("makeDeviceForHostname(): unsupported gloo device", reproduced with localhost, 127.0.0.1, USE_LIBUV
and GLOO_DEVICE_TRANSPORT variants), so upstream's init_process_group(gloo) fails before training starts. With one
rank the process group and DDP do nothing numerically (all-reduce over one rank is the identity), so we make
init_process_group a no-op and give run() a pass-through DDP with the same .module layout. The samplers already
receive num_replicas=1, rank=0 explicitly; nothing else in s2_train touches torch.distributed.

  python gsv_s2_train.py --config <exp>/configs/s2.json --device cuda --mem-gb 8 [--reserve-gb 8] [--wait-min 0]
                         [--loader-workers N] [--smoke-max-steps N] [--report out.json]

Exit codes: 0 ok, 3 CUDA out of memory (train.py retries with a smaller batch), 4 VRAM guard refused to start.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--wait-min", "--wait-vram-min", dest="wait_min", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--loader-workers", type=int, default=None, help="override upstream's 5 DataLoader workers")
    ap.add_argument("--smoke-max-steps", type=int, default=0, help="stop each epoch after N batches (smoke tests)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    args.config, args.report = str(C.abspath(args.config)), (str(C.abspath(args.report)) if args.report else None)
    device = C.normalize_device(args.device)

    C.bootstrap_gsv_runtime(device)
    sys.path.insert(0, str(C.GSV_PKG))  # upstream runs with GPT_SoVITS/ as the script directory
    sys.argv = [str(C.GSV_PKG / "s2_train.py"), "--config", str(Path(args.config).resolve())]
    import s2_train  # noqa: E402  parses --config; sets CUDA_VISIBLE_DEVICES from train.gpu_numbers

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # before anything initialises CUDA
    import torch

    C.assert_device(device)
    report: dict = {"config": args.config, "device": device, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "batch_size": s2_train.hps.train.batch_size, "epochs": s2_train.hps.train.epochs}
    if device != "cpu":
        report["vram"] = C.cap_gpu_memory(args.mem_gb, args.reserve_gb, args.wait_min)
    torch.set_num_threads(max(1, args.threads))

    if args.loader_workers is not None:
        base_loader = s2_train.DataLoader
        workers = max(0, args.loader_workers)

        def loader(*a, **k):
            k["num_workers"] = workers
            if workers == 0:
                k.pop("persistent_workers", None)
                k.pop("prefetch_factor", None)
            return base_loader(*a, **k)

        s2_train.DataLoader = loader
    if args.smoke_max_steps > 0:
        s2_train.tqdm = lambda iterable, *a, **k: itertools.islice(iterable, args.smoke_max_steps)

    def no_spectrogram_images(*_a, **_k):
        # upstream's plot_spectrogram_to_numpy calls fig.canvas.tostring_rgb(), which matplotlib >= 3.10 (3.11 in
        # this venv) removed: every log interval it opened a figure, raised before plt.close() and leaked it.
        # train_and_evaluate catches the error and logs scalars only, which is what happens now, minus the leak.
        raise RuntimeError("spectrogram images disabled")

    s2_train.utils.plot_spectrogram_to_numpy = no_spectrogram_images

    class SingleProcessDDP(torch.nn.Module):
        """DistributedDataParallel for world_size 1: same .module layout, forward passes straight through."""

        def __init__(self, module, *_args, **_kwargs):
            super().__init__()
            self.module = module

        def forward(self, *a, **k):
            return self.module(*a, **k)

    class NoProcessGroup:
        """torch.distributed with init_process_group as a no-op (single process, see module docstring)."""

        def __getattr__(self, name):
            return getattr(torch.distributed, name)

        @staticmethod
        def init_process_group(*_a, **_k):
            return None

    s2_train.DDP = SingleProcessDDP
    s2_train.dist = NoProcessGroup()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(random.randint(20000, 55555))
    code, t0 = 0, time.time()
    try:
        s2_train.run(0, 1, s2_train.hps)
    except Exception as exc:  # torch.OutOfMemoryError or a cuBLAS/cuDNN allocation RuntimeError -> exit 3
        code = 3 if C.is_oom(exc) else 1
        if code == 1:
            raise
        print(f"[gsv] CUDA out of memory in SoVITS training (batch {s2_train.hps.train.batch_size}): {exc}", flush=True)
    finally:
        report.update({"seconds": round(time.time() - t0, 1), "exit_code": code,
                       "global_step": getattr(s2_train, "global_step", None)})
        report.update(C.peak_memory_report(device))
        print("[gsv-s2] " + json.dumps(report), flush=True)
        if args.report:
            C.write_json(args.report, report)
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
    sys.exit(code)


if __name__ == "__main__":
    main()
