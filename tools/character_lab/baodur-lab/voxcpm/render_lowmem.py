"""render.py with a low host-RAM model load and a stop-at-checkpoint memory guard (same arguments as render.py).

Why: render.py's VoxCPM.from_pretrained builds VoxCPM2 in fp32 on the CPU (9.2 GB) and then reads the whole bf16
checkpoint into RAM (4.6 GB): ~13 GB host commit at peak. On 2026-10-07 the PC had ~12 GB free (overnight builds),
so this wrapper loads the same weights with a much lower peak:
  1. the AudioVAE is built and loaded in fp32 exactly as upstream;
  2. VoxCPM2Model itself is constructed under torch default dtype bfloat16 (its weights are overwritten by the
     checkpoint; rotary tables are computed with explicit .float() and cast to bf16 by model.to(lm_dtype) exactly
     as upstream; KV caches use explicit dtypes; LoRA scaling buffers are re-set by set_lora_enabled; lora_B = 0);
  3. model.safetensors is streamed tensor by tensor (safetensors.safe_open) into the parameters instead of
     load_file() of the whole dict. Same values, same dtype conversion (Tensor.copy_ as load_state_dict does).
  Peak ~1.9 (python/torch) + 4.6 (bf16 model) + 0.8 (VAE + its state dict) ~ 7.5 GB.
Guard: after every take (render.py rewrites the manifest atomically after each take = a resumable checkpoint) the
run stops with exit code 4 if system free physical RAM < --stop-ram-gb (default 3) or device free VRAM <
--stop-vram-gb (default 1.0); re-run with --resume to continue. Writes nothing beyond what render.py writes.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ctypes  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _pop_float(flag: str, default: float) -> float:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        value = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return value
    return default


STOP_RAM_GB = _pop_float("--stop-ram-gb", 3.0)
STOP_VRAM_GB = _pop_float("--stop-vram-gb", 1.0)
C.HOST_LOAD_GB["render"] = 13.0  # Windows commit, not RAM: WDDM charges the ~6.6 GB of GPU allocations to commit too
# (measured 2026-10-07: 12.4 GB process commit after the load, 4 GB working set)


def free_physical_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1e9


def install_lowmem_loader() -> None:
    import json
    import struct

    import torch
    from transformers import LlamaTokenizerFast
    from voxcpm.model import voxcpm2 as V
    from voxcpm.model.utils import get_dtype

    def from_local(cls, path, optimize=True, training=False, device=None, lora_config=None):
        if training:
            raise SystemExit("render_lowmem: inference only")
        free0 = free_physical_gb()
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as fh:
            config = V.VoxCPMConfig.model_validate_json(fh.read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        vae_cfg = getattr(config, "audio_vae_config", None)
        audio_vae = V.AudioVAEV2(config=vae_cfg) if vae_cfg else V.AudioVAEV2()
        vae_st = os.path.join(path, "audiovae.safetensors")
        if os.path.exists(vae_st):
            vae_sd = V.load_file(vae_st, device="cpu")
        else:
            ckpt = torch.load(os.path.join(path, "audiovae.pth"), map_location="cpu", weights_only=True)
            vae_sd = ckpt.get("state_dict", ckpt)
        previous = torch.get_default_dtype()
        lm_dtype = get_dtype(config.dtype)
        torch.set_default_dtype(torch.bfloat16 if lm_dtype == torch.bfloat16 else previous)
        try:
            model = cls(config, tokenizer, audio_vae, lora_config, device=device)
        finally:
            torch.set_default_dtype(previous)
        model = model.to(get_dtype(model.config.dtype))
        model.audio_vae = model.audio_vae.to(torch.float32)
        missing, unexpected = model.audio_vae.load_state_dict(vae_sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"render_lowmem: AudioVAE keys missing {missing[:5]} unexpected {unexpected[:5]}")
        del vae_sd
        target = model.state_dict()  # shares storage with the parameters/buffers
        loaded, unexpected = set(), []
        # Plain reads, no memory map: safetensors' copy-on-write mapping of the 4.6 GB file is charged to the
        # Windows commit limit (2026-10-07: a second load failed with os error 1455 "paging file too small").
        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
                  "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
                  "BOOL": torch.bool}
        with torch.no_grad(), open(os.path.join(path, "model.safetensors"), "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(size))
            base = 8 + size
            entries = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                             key=lambda kv: kv[1]["data_offsets"][0])
            for key, meta in entries:
                if key not in target:
                    unexpected.append(key)
                    continue
                start, end = meta["data_offsets"]
                fh.seek(base + start)
                buf = bytearray(fh.read(end - start))
                tensor = (torch.frombuffer(buf, dtype=dtypes[meta["dtype"]]).reshape(meta["shape"]) if buf
                          else torch.empty(meta["shape"], dtype=dtypes[meta["dtype"]]))
                target[key].copy_(tensor)
                loaded.add(key)
                del buf, tensor
        missing = [k for k in target if k not in loaded and not k.startswith("audio_vae.") and "lora_" not in k]
        print(json.dumps({"render_lowmem_load": {"tensors_loaded": len(loaded), "missing_non_lora": missing[:10],
                                                 "unexpected": unexpected[:10],
                                                 "free_ram_gb_before": round(free0, 2),
                                                 "free_ram_gb_at_cpu_peak": round(free_physical_gb(), 2)}}),
              flush=True)
        if missing:
            raise SystemExit(f"render_lowmem: base weights missing from the checkpoint: {missing[:10]}")
        return model.to(model.device).eval().optimize(disable=not optimize)

    V.VoxCPM2Model.from_local = classmethod(from_local)


def install_guard() -> None:
    original = C.atomic_write_json

    def atomic_write_json(path, data, *a, **k):
        result = original(path, data, *a, **k)
        if Path(path).name == "manifest.json":
            ram = free_physical_gb()
            vram = None
            try:  # nvidia-smi, not torch.cuda.mem_get_info: under WDDM the latter reported 32.5 GB free of 34
                import subprocess  # while nvidia-smi showed 11 GB (2026-10-07)
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=20).stdout
                vram = float(out.strip().splitlines()[0]) * 1048576 / 1e9
            except Exception:
                vram = None
            if ram < STOP_RAM_GB or (vram is not None and vram < STOP_VRAM_GB):
                print(f"GUARD STOP at checkpoint: free RAM {ram:.2f} GB (stop < {STOP_RAM_GB}), free VRAM "
                      f"{vram if vram is None else round(vram, 2)} GB (stop < {STOP_VRAM_GB}); "
                      f"manifest saved, re-run with --resume", flush=True)
                os._exit(4)
        return result

    C.atomic_write_json = atomic_write_json


if __name__ == "__main__":
    import render as R

    install_lowmem_loader()
    install_guard()
    R.C.atomic_write_json = C.atomic_write_json
    R.main()
