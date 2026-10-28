"""Shared-machine guards for the delivery task's GPU steps: host commit and VRAM headroom (stdlib only).

The machine runs near its Windows commit limit (other apps and parallel agents have crashed native loads), and
the RTX 5090 is shared with Substance Painter, which needs >= 8 GB free at all times. Every GPU-facing script
here checks both before loading a model, refuses with a clear message instead of squeezing other apps, and
records what it saw so the estimates can be corrected.
"""
from __future__ import annotations

import json
import os

# Peak host commit of a VoxCPM2 inference load: from_local builds the model in fp32 on the CPU (~9.2 GB for
# 2.29 B params) and reads the 4.58 GB bf16 state dict before .to(cuda), plus Python/torch (~2 GB). The voxcpm
# render step budgets the same ~13 GB (voxcpm/common.py HOST_LOAD_GB["render"]).
VOXCPM_LOAD_COMMIT_GB = 13.0
HOST_MARGIN_GB = 2.0          # commit that must stay free for other apps while we sit at our peak
CUDA_OVERHEAD_GB = 1.0        # CUDA context + cuBLAS/cuDNN workspaces: outside torch's allocator cap


def host_memory() -> dict | None:
    """System commit state (GlobalMemoryStatusEx; commit_free_gb = ullAvailPageFile). None off Windows."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {"commit_free_gb": round(status.ullAvailPageFile / 1e9, 2),
            "commit_limit_gb": round(status.ullTotalPageFile / 1e9, 2),
            "phys_free_gb": round(status.ullAvailPhys / 1e9, 2)}


def process_commit() -> dict | None:
    """This process's commit now and at its peak (GetProcessMemoryInfo). None off Windows."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return {"commit_gb": round(counters.PrivateUsage / 1e9, 2), "peak_commit_gb": round(counters.PeakPagefileUsage / 1e9, 2)}


def require_host_commit(need_gb: float, margin_gb: float, what: str) -> dict:
    """Refuse a large CPU-side load unless need_gb (minus what this process already holds) + margin_gb of commit
    is free. Checked right before the load."""
    info = host_memory()
    if info is None:
        return {"checked": False, "reason": "not Windows"}
    own = float((process_commit() or {}).get("commit_gb") or 0.0)
    required = need_gb - own + margin_gb
    info.update(checked=True, need_gb=need_gb, margin_gb=margin_gb, process_commit_before_gb=own,
                required_free_gb=round(required, 2))
    if info["commit_free_gb"] < required:
        raise SystemExit(f"Not enough free host memory (Windows commit) for the {what}: {info['commit_free_gb']:.1f} GB "
                         f"free, need ~{required:.1f} GB (~{need_gb:.0f} GB peak while loading on the CPU + "
                         f"{margin_gb:.0f} GB kept free for other apps). Wait for other large loads to finish and "
                         f"retry. {json.dumps(info)}")
    return info


def vram_headroom(torch, need_gb: float, keep_free_gb: float, margin_gb: float = 0.0) -> tuple[bool, dict]:
    """(ok, info): ok when free VRAM >= need_gb + keep_free_gb + margin_gb, i.e. this job at its peak still
    leaves keep_free_gb for other apps. Creates this process's CUDA context, so call it right before loading."""
    if not torch.cuda.is_available():
        return False, {"reason": "CUDA not available"}
    free_b, total_b = torch.cuda.mem_get_info(0)
    info = {"device": torch.cuda.get_device_name(0), "free_gb": round(free_b / 2**30, 2),
            "total_gb": round(total_b / 2**30, 2), "need_gb": need_gb, "keep_free_gb": keep_free_gb,
            "margin_gb": margin_gb}
    info["required_free_gb"] = round(need_gb + keep_free_gb + margin_gb, 2)
    return free_b / 2**30 >= info["required_free_gb"], info
