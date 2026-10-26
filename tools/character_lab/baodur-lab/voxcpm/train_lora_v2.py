"""Run the OFFICIAL VoxCPM2 LoRA trainer for a voice-lab character inside the shared-GPU and shared-RAM budget.

  cd C:\\Users\\NewAdmin\\Documents\\GDeveloper\\External\\voice-lab\\baodur-lab\\voxcpm
  %LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe train_lora_v2.py ^
      --config C:\\Users\\NewAdmin\\Documents\\KotorMods\\Voices\\BaoDurLab\\voxcpm\\manifests\\original\\train_lora_v2.yaml
  (Bao-Dur. Another character: set VOICE_LAB_CHARACTER, e.g. jolee, and pass its
  <lab>\\voxcpm\\manifests\\<dataset>\\train_lora_v2.yaml; save_path must lie under that <lab>\\voxcpm.)
  Preconditions: the GPU all-clear recorded in <lab>\\DECISIONS.md (Bao-Dur: the BetterArmor session's), and a
  passing FULL check_config.py report for exactly this config + manifests + code (the run refuses otherwise).

The official script (source-2.0.3\\scripts\\train_voxcpm_finetune.py) is imported unmodified and its
train() is called with the YAML, exactly like its own __main__. This wrapper only adds:
- a VRAM preflight + hard cap: refuses to start unless it can stay <= --max-vram-gb (default 9) with
  headroom over the estimate while leaving --reserve-gb (default 8) free for Substance Painter, then sets
  torch.cuda.set_per_process_memory_fraction before anything is allocated (an overrun becomes an OOM
  in this process instead of starving other apps);
- a host-RAM preflight: refuses to start unless the Windows commit headroom covers the CPU-side load
  peak (~13 GB with --frozen-dtype bf16, ~17.5 GB with fp32) plus --host-margin-gb (default 2) for the
  other apps; the process's real peak is written to run_config.json/train_summary.json;
- --frozen-dtype bf16 (default): frozen base weights are stored in bf16 instead of fp32. Under the
  trainer's bf16 autocast they are cast to bf16 per op anyway, and because they do not require grad
  autocast does not cache those casts and autograd keeps each bf16 copy for the backward pass; storing
  them in bf16 removes both the 9.2 GB fp32 master and ~4 GB of copies. The model is also BUILT under
  default dtype bf16 (from_local would otherwise build 9.2 GB of fp32 weights and then read the 4.6 GB
  checkpoint on top), which halves the host peak. LoRA weights + scaling buffers, grads and AdamW state
  stay fp32 and the AudioVAE stays fp32 (from_local casts it before its fp32 weights load). Every frozen
  weight is overwritten by the bf16 checkpoint, so the only difference to an fp32 build is lora_A's random
  initialisation: drawing it under default dtype bf16 consumes the RNG differently, so it is a different
  random draw (same kaiming-uniform distribution, bf16-representable values), not a rounding of the fp32
  draw. lora_B starts at zero, so the adapter's initial output is zero either way and training is not
  biased (check_config.py 6b: identical loss and gradients once both models share one LoRA init).
  Numerically this matches inference, which runs all-bf16;
- --grad-ckpt all|dit|none (default all): activation checkpointing of the MiniCPM decoder layers
  (base LM, residual LM and/or LocDiT) during training forward passes only (~30% more compute,
  activations from ~3.7 GB to ~0.4 GB at the longest batch);
- deterministic validation (--val-seed, default 20260930; -1 = off): every official validate() call runs
  with the same RNG draws (CFM timestep/noise, CFG dropout, val shuffle order, previews), so val losses
  of different steps are comparable instead of differing by sampling noise (Bao-Dur's July Vox run showed
  +/-0.006 total / +/-0.02 stop between two validations of nearly the same weights). The training RNG
  stream is saved and restored around each call. Even so, val loss is a sanity check (finite, no
  blow-up) and a hint for the sweep, not the checkpoint selector: pick checkpoints from the render sweep
  (select_checkpoint.py) and by ear. train_summary.json lists the val curve and its minimum;
- frozen training data: train.jsonl/val.jsonl (+ manifest_report.json) are copied into
  runs\\<run>\\data\\ and the official trainer reads those copies; run_config.json records their sha256,
  the audio variant and the dataset, so render.py and the evaluation's provenance check read exactly the
  rows the adapter was trained on even if the manifests are rebuilt later;
- guards: the manifests must be what the CURRENT build_manifest.py writes for the CURRENT character and its
  recipe.json (manifest_report.json builder sha256 + schema + character + recipe sha256 + file hashes),
  never restored audio (user decision 2026-09-30) unless
  --allow-restored-audio, no held-out test or other-character clip, a fresh save_path (the official
  trainer silently resumes from <save_path>\\latest), and a passing full check_config report;
- --steps N: a short, fully annealed retrain (num_iters = max_steps = N, warmup min(100, max(10, N/10)))
  into runs\\<run>_s<N>\\, for when the sweep's winner is an early, un-annealed checkpoint;
- peak-memory columns in every train/val log line, run_config.json / train_summary.json next to the
  checkpoints;
- --preview-interval N (default 200): the official validate() renders two zero-shot TensorBoard
  previews at every validation (~1.5 min each time); they now run only every N steps and on the last
  step, while the cheap val loss still runs every valid_interval. 0 disables previews.
Estimated device memory (see check_config.py): official settings ~16-19 GB; this wrapper ~6.7 GB.
Ctrl+C / SIGTERM: the official handler saves <save_path>\\step_XXXXXXX + latest and exits at once
(train_summary.json is then not written); continue with --resume (same --steps, if any).
No bytecode is written (sys.dont_write_bytecode): importing the official trainer used to leave a
__pycache__ inside Vox's source-2.0.3\\scripts.

--dry-run does every check on CPU (no CUDA, no training, nothing written) and prints the plan.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never leave __pycache__ in Vox's engine tree (source-2.0.3) or the venv

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DEFAULT_VAL_SEED = 20260930
CHECK_REPORT = "check_config_report.json"
# files whose revision a check_config report must match before a real run (they define what it verified)
CHECKED_CODE = ("train_lora_v2.py", "check_config.py", "common.py")
DEFAULT_SWEEP_STEPS = (100, 200, 300, 400, 500, 700, 1000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-vram-gb", type=float, default=9.0, help="hard ceiling for this process (device total)")
    p.add_argument("--reserve-gb", type=float, default=8.0, help="VRAM that must stay free for other apps")
    p.add_argument("--host-margin-gb", type=float, default=C.HOST_MARGIN_GB,
                   help="Windows commit that must stay free for other apps at our loading peak")
    p.add_argument("--frozen-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--grad-ckpt", choices=("all", "dit", "none"), default="all")
    p.add_argument("--preview-interval", type=int, default=200,
                   help="TensorBoard audio previews every N steps (+ last step); 0 = none")
    p.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED,
                   help="fixed RNG seed for every validation (comparable val losses); -1 = official random")
    p.add_argument("--steps", type=int, default=0,
                   help="short fully annealed retrain: num_iters = max_steps = N, into runs\\<run>_s<N>")
    p.add_argument("--allow-restored-audio", action="store_true",
                   help="override the user's 2026-09-30 decision to train on original audio only")
    p.add_argument("--skip-check-report", action="store_true",
                   help="start without a matching passing check_config report (not recommended)")
    p.add_argument("--resume", action="store_true", help="continue from <save_path>\\latest")
    p.add_argument("--dry-run", action="store_true", help="CPU-only validation; do not touch the GPU")
    args = p.parse_args()
    if args.steps and args.steps < 50:
        p.error("--steps must be >= 50")
    return args


# --------------------------------------------------------------------------------------------- official code
def use_official_source() -> None:
    """Import voxcpm from source-2.0.3\\src, the tree the official script puts first on sys.path."""
    sys.dont_write_bytecode = True
    src = str(C.OFFICIAL_SRC / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    C.offline_env()


def load_official_trainer():
    sys.dont_write_bytecode = True  # exec_module would otherwise write scripts\__pycache__ into Vox's tree
    spec = importlib.util.spec_from_file_location("voxcpm_official_finetune", str(C.OFFICIAL_TRAINER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_signature(trainer) -> inspect.Signature:
    fn = trainer.train
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return inspect.signature(fn)


def train_globals_used(trainer) -> set[str]:
    """Global names the official train() body looks up at call time (validate, generate_sample_audio)."""
    fn = trainer.train
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return set(fn.__code__.co_names)


# --------------------------------------------------------------------------------------------- checks
def apply_steps_override(cfg: dict, steps: int) -> dict:
    """--steps N: a short schedule that anneals fully at N, written to runs\\<run>_s<N>\\."""
    if not steps:
        return cfg
    cfg = dict(cfg)
    run_root = Path(cfg["save_path"]).parent
    new_root = run_root.with_name(f"{run_root.name}_s{steps}")
    cfg.update(num_iters=steps, max_steps=steps,
               warmup_steps=min(int(cfg["warmup_steps"]), max(10, round(0.1 * steps))),
               save_interval=min(int(cfg["save_interval"]), steps),
               valid_interval=min(int(cfg["valid_interval"]), steps),
               log_interval=min(int(cfg["log_interval"]), steps),
               save_path=str(new_root / "checkpoints"), tensorboard=str(new_root / "logs"))
    return cfg


def manifest_files(cfg: dict) -> dict[str, Path]:
    out = {"train.jsonl": Path(cfg["train_manifest"])}
    if cfg.get("val_manifest"):
        out["val.jsonl"] = Path(cfg["val_manifest"])
    return out


def preflight(cfg: dict, resume: bool, allow_restored: bool = False, check_builder: bool = True) -> list[str]:
    """Hard checks shared with check_config.py. Returns human-readable notes; raises on problems."""
    from voxcpm.model.voxcpm2 import LoRAConfig

    notes: list[str] = []
    errors: list[str] = []
    pretrained = Path(cfg["pretrained_path"])
    if pretrained.resolve() != C.SNAPSHOT.resolve():
        errors.append(f"pretrained_path is {pretrained}, expected the current snapshot {C.SNAPSHOT}")
    model_cfg = json.loads((pretrained / "config.json").read_text(encoding="utf-8"))
    if model_cfg.get("architecture", "").lower() != "voxcpm2":
        errors.append("snapshot is not a voxcpm2 model")
    vae = model_cfg.get("audio_vae_config") or {}
    if int(cfg.get("sample_rate", 0)) != int(vae.get("sample_rate", -1)):
        errors.append(f"sample_rate {cfg.get('sample_rate')} != AudioVAE encoder rate {vae.get('sample_rate')}")
    if int(cfg.get("out_sample_rate", 0)) not in (0, int(vae.get("out_sample_rate", -1))):
        errors.append(f"out_sample_rate {cfg.get('out_sample_rate')} != decoder rate {vae.get('out_sample_rate')}")
    lora = cfg.get("lora") or {}
    unknown = set(lora) - set(LoRAConfig.model_fields)
    if unknown:
        errors.append(f"unknown lora keys {sorted(unknown)}")
    LoRAConfig(**lora)
    if int(cfg["num_iters"]) != int(cfg["max_steps"]):
        errors.append("num_iters must equal max_steps (the loop runs num_iters; the cosine uses max_steps)")
    if not 0 <= int(cfg["warmup_steps"]) < int(cfg["max_steps"]):
        errors.append("warmup_steps must be < max_steps")
    for key in ("valid_interval", "save_interval", "log_interval"):
        if not 0 < int(cfg[key]) <= int(cfg["num_iters"]):
            errors.append(f"{key} must be within (0, num_iters]")
    for key in ("save_path", "tensorboard"):
        target = Path(cfg[key]).resolve()
        if C.DATA_ROOT.resolve() not in target.parents:
            errors.append(f"{key} must live under {C.DATA_ROOT}")
    latest = Path(cfg["save_path"]) / "latest"
    if latest.exists() and not resume:
        errors.append(f"{latest} exists: the official trainer would silently resume. Pass --resume to "
                      "continue that run, or point save_path at a new folder.")
    files = manifest_files(cfg)
    folders = {p.resolve().parent for p in files.values()}
    if len(folders) != 1:
        errors.append(f"train/val manifests come from different folders {sorted(map(str, folders))}")
    report, problems = C.manifest_report_status(files["train.jsonl"])
    if not check_builder:  # resuming from the run's frozen copies: only their own hashes matter
        problems = [p for p in problems if "changed after" in p or "missing" in p]
    for problem in problems:
        errors.append(f"manifests: {problem}; rebuild with build_manifest.py and re-run check_config.py")
    screen = C.load_speaker_screen()
    if report is not None:
        recorded = set((report.get("manifest_sha256") or {}))
        if set(files) - recorded:
            errors.append(f"manifest_report.json does not cover {sorted(set(files) - recorded)}")
        audio = report.get("audio") or report.get("variant")
        if audio != "original" and not allow_restored:
            errors.append(f"these manifests use '{audio}' audio: {C.RESTORED_DECISION} "
                          "(--allow-restored-audio overrides)")
        built_with = (report.get("speaker_screen") or {}).get("sha256")
        if check_builder and built_with != screen["sha256"]:  # the delivery task edits the screen
            errors.append(f"the speaker screen changed after the manifests were built ({str(built_with)[:12]} -> "
                          f"{screen['sha256'][:12]}); rebuild with build_manifest.py and re-run check_config.py")
    tests = C.test_ids()
    pattern = C.test_id_pattern(tests)
    others = set(screen["confirmed"])
    other_pattern = C.test_id_pattern(others) if others else None
    for key, path in files.items():
        if not path.exists():
            errors.append(f"{key} not found: {path} (run build_manifest.py)")
            continue
        text = path.read_text(encoding="utf-8")
        hits = sorted(set(m.group(0) for m in pattern.finditer(text)))
        if hits:
            errors.append(f"{key} names held-out test clips {hits}")
        other_hits = sorted(set(m.group(0) for m in other_pattern.finditer(text))) if other_pattern else []
        if other_hits:
            errors.append(f"{key} contains other characters' clips {other_hits} (rebuild with build_manifest.py)")
    if errors:
        raise SystemExit("config problems:\n  - " + "\n  - ".join(errors))
    notes.append(f"snapshot {C.SNAPSHOT_ID} (weights identical to {C.OLD_SNAPSHOT_ID})")
    notes.append(f"speaker screen {screen['sha256'][:12]}: no other-character clip in the manifests")
    if report is not None:
        notes.append(f"manifests: dataset {report.get('dataset')}, audio {report.get('audio')}, "
                     f"built {report.get('created')} by build_manifest.py {report['builder']['build_manifest_sha256'][:12]}"
                     f" (schema {report['builder']['schema']})")
    return notes


def code_hashes() -> dict[str, str]:
    return {name: C.sha256_file(C.CODE_DIR / name) for name in CHECKED_CODE}


def check_report_status(config: Path, files: dict[str, Path]) -> tuple[dict | None, list[str]]:
    """Is there a passing FULL check_config report for exactly this config, these manifests and this code?"""
    path = config.parent / CHECK_REPORT
    if not path.exists():
        return None, [f"{path} missing: run check_config.py --config {config} first"]
    report = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if not report.get("ok"):
        problems.append(f"the last check_config run failed: {str(report.get('failure'))[:300]}")
    if not report.get("model_tests"):
        problems.append("the last check_config run skipped the model tests (--skip-model-tests)")
    if report.get("config_sha256") != C.sha256_file(config):
        problems.append("the config changed after check_config ran")
    want = {name: C.sha256_file(p) for name, p in files.items()}
    if report.get("manifest_sha256") != want:
        problems.append("the manifests changed after check_config ran")
    if report.get("code_sha256") != code_hashes():
        problems.append(f"{'/'.join(CHECKED_CODE)} changed after check_config ran")
    if problems:
        problems[-1] += f" -> re-run check_config.py --config {config}"
    return report, problems


def expected_checkpoints(cfg: dict) -> list[str]:
    """Folders the official loop writes: step % save_interval == 0, step == num_iters-1, then num_iters."""
    n, every = int(cfg["num_iters"]), int(cfg["save_interval"])
    steps = sorted({s for s in range(n) if s % every == 0} | {n - 1, n})
    return [f"step_{s:07d}" for s in steps]


VAL_LINE = re.compile(r"^\[val\] step (\d+):(.*)$")
VAL_VALUE = re.compile(r"loss/(total|diff|stop): ([0-9.eE+-]+)")


def val_curve(train_log: Path, saved_steps: list[int], num_iters: int) -> dict:
    """The validation curve from the official train.log, its minimum and a suggested render sweep."""
    points: dict[int, dict] = {}
    if train_log.exists():
        for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = VAL_LINE.match(line.strip())
            if not m:
                continue
            values = {k: float(v) for k, v in VAL_VALUE.findall(m.group(2))}
            if {"total", "diff", "stop"} <= set(values):  # the last validation of a step wins (a resumed
                points[int(m.group(1))] = {"step": int(m.group(1)), **values}  # run re-validates its first step)
    curve = [points[s] for s in sorted(points)]
    out: dict = {"points": curve}
    trained = [p for p in curve if p["step"] > 0]
    if not trained:
        return out
    best = min(trained, key=lambda p: p["total"])
    candidates = [s for s in saved_steps if s not in (0, num_iters - 1)]  # step_0000999 == step_0001000
    nearest = min(candidates, key=lambda s: (abs(s - best["step"]), s)) if candidates else None
    extra = sorted({s for s in (nearest, (nearest or 0) + 200) if s and s in candidates}
                   - set(DEFAULT_SWEEP_STEPS))
    sweep = sorted({s for s in DEFAULT_SWEEP_STEPS if s in saved_steps or s == num_iters} | set(extra))
    out.update(best_total=best, best_diff=min(trained, key=lambda p: p["diff"]),
               best_stop=min(trained, key=lambda p: p["stop"]),
               suggested_sweep_steps=sweep, val_minimum_steps_not_in_default_sweep=extra,
               note="val loss is a hint, not the selector: render the sweep and compare by select_checkpoint.py + ear")
    return out


def throttle_previews(trainer, interval: int, num_iters: int) -> None:
    """Run the official zero-shot TensorBoard previews only every `interval` steps and on the last step.

    validate() looks generate_sample_audio up as a module global at call time, so rebinding the name in
    the loaded official module is enough; the val-loss part of validate() is untouched.
    """
    original = trainer.generate_sample_audio
    if getattr(original, "_lab_throttled", False):
        original = original._lab_original

    def generate_sample_audio(model, val_ds, audio_vae, writer, step, *args, **kwargs):
        if interval <= 0 or (step % interval != 0 and step != num_iters - 1):
            return None
        return original(model, val_ds, audio_vae, writer, step, *args, **kwargs)

    generate_sample_audio._lab_throttled = True
    generate_sample_audio._lab_original = original
    trainer.generate_sample_audio = generate_sample_audio


def deterministic_validation(trainer, seed: int) -> None:
    """Give every official validate() call the same RNG draws (common random numbers).

    train() looks validate up as a module global at call time, so rebinding it in the loaded official
    module wraps every validation. Inside, torch (CPU + current CUDA device), Python and NumPy RNGs are
    seeded with `seed`; their previous states are restored afterwards, so the training stream is
    exactly what it would have been. seed < 0 restores the official behaviour.
    """
    import random

    import numpy as np
    import torch

    original = trainer.validate
    if getattr(original, "_lab_deterministic", False):
        original = original._lab_original
    if seed < 0:
        trainer.validate = original
        return

    def validate(*args, **kwargs):
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                return original(*args, **kwargs)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    validate._lab_deterministic = True
    validate._lab_original = original
    validate._lab_seed = seed
    trainer.validate = validate


# --------------------------------------------------------------------------------------------- patches
def prepare_for_training(model, frozen_dtype: str = "bf16", grad_ckpt: str = "all") -> dict:
    """Give the parameters the dtypes this recipe trains with and flag decoder layers for checkpointing.

    Frozen non-VAE parameters -> bf16 (or left fp32 with frozen_dtype fp32); trainable (LoRA) parameters
    and the LoRA scaling buffers -> fp32, which is what the official fp32 build has, whatever default
    dtype the model was built under. The AudioVAE is left alone (from_local keeps it fp32).
    """
    import torch
    from voxcpm.modules.layers.lora import LoRALinear

    cast = trainable = frozen_bf16 = lora_recast = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                lora_recast += 1
            continue
        if name.startswith("audio_vae."):
            continue
        if frozen_dtype == "bf16" and param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
            cast += param.numel()
        if param.dtype == torch.bfloat16:
            frozen_bf16 += param.numel()
    scaling_recast = 0
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.scaling.dtype != torch.float32:
            module.scaling = module.scaling.to(torch.float32)
            scaling_recast += 1
    stacks = []
    if grad_ckpt == "all":
        stacks += [model.base_lm, model.residual_lm]
    if grad_ckpt in ("all", "dit"):
        stacks.append(model.feat_decoder.estimator.decoder)
    flagged = 0
    for stack in stacks:
        for layer in stack.layers:
            layer._lab_checkpoint = True
            flagged += 1
    return {"frozen_params_bf16": frozen_bf16, "frozen_params_cast_to_bf16": cast, "trainable_params": trainable,
            "lora_tensors_recast_to_fp32": lora_recast, "lora_scaling_buffers_recast_to_fp32": scaling_recast,
            "checkpointed_layers": flagged}


_PATCHED: dict = {}


def install_patches(frozen_dtype: str, grad_ckpt: str) -> dict:
    """Idempotently patch VoxCPM2Model.from_local, MiniCPMDecoderLayer.forward and the tracker."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    from voxcpm.modules.minicpm4 import model as minicpm
    from voxcpm.training import tracker as tracker_module

    stats: dict = _PATCHED.setdefault("stats", {})
    _PATCHED["options"] = (frozen_dtype, grad_ckpt)
    if _PATCHED.get("installed"):
        return stats
    original_from_local = VoxCPM2Model.from_local.__func__
    _PATCHED["original_from_local"] = original_from_local

    def from_local(cls, path, optimize=True, training=False, device=None, lora_config=None):
        frozen, ckpt = _PATCHED["options"]
        build_bf16 = bool(training) and frozen == "bf16"
        previous = torch.get_default_dtype()
        if build_bf16:
            # Build in bf16 instead of fp32 (9.2 GB) + cast later: ~17.5 GB -> ~13 GB host peak. Tensors
            # created at construction with the default dtype are weights (overwritten by the bf16
            # checkpoint), the VAE (cast to fp32 by from_local before its fp32 weights load), the LoRA
            # parameters and LoRA scaling buffers (restored to fp32 by prepare_for_training; lora_A's random
            # init is a different draw than under fp32, lora_B is zero); rotary tables and KV caches use
            # explicit dtypes. check_config.py verifies this on a tiny model.
            torch.set_default_dtype(torch.bfloat16)
        try:
            model = original_from_local(cls, path, optimize=optimize, training=training, device=device,
                                        lora_config=lora_config)
        finally:
            torch.set_default_dtype(previous)
        if training:
            stats.update(prepare_for_training(model, frozen, ckpt))
            stats["built_with_default_dtype"] = "bfloat16" if build_bf16 else str(previous).replace("torch.", "")
            stats["host_after_load"] = C.process_commit()
        return model

    VoxCPM2Model.from_local = classmethod(from_local)

    original_forward = minicpm.MiniCPMDecoderLayer.forward

    def forward(self, *args, **kwargs):
        if getattr(self, "_lab_checkpoint", False) and self.training and torch.is_grad_enabled():
            return checkpoint(original_forward, self, *args, use_reentrant=False, **kwargs)
        return original_forward(self, *args, **kwargs)

    minicpm.MiniCPMDecoderLayer.forward = forward

    original_log = tracker_module.TrainingTracker.log_metrics

    def log_metrics(self, metrics, split):
        if torch.cuda.is_available():
            metrics = dict(metrics)
            metrics["mem/peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
            metrics["mem/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        return original_log(self, metrics, split)

    tracker_module.TrainingTracker.log_metrics = log_metrics
    _PATCHED["installed"] = True
    return stats


# --------------------------------------------------------------------------------------------- data copies
def frozen_copies(cfg: dict, run_root: Path, resume: bool, dry_run: bool) -> tuple[dict, dict]:
    """Point the run at its own copies of the manifests (runs\\<run>\\data\\). Returns (cfg, info).

    Fresh run: copy train/val/manifest_report into data\\ (overwriting a copy left by an attempt that never
    saved a checkpoint). --resume: reuse the copies the run started with and verify their hashes against
    run_config.json, so a resumed run keeps training on exactly the same rows."""
    data = run_root / "data"
    sources = manifest_files(cfg)
    copies = {name: data / name for name in sources}
    info = {"dir": str(data), "sources": {k: str(v) for k, v in sources.items()}}
    record_path = run_root / "run_config.json"
    if resume and all(p.exists() for p in copies.values()):
        recorded = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        want = {"train.jsonl": recorded.get("train_manifest_sha256"), "val.jsonl": recorded.get("val_manifest_sha256")}
        for name, path in copies.items():
            if want.get(name) and C.sha256_file(path) != want[name]:
                raise SystemExit(f"--resume: {path} no longer matches the sha256 in {record_path}")
        info["reused"] = True
    elif not dry_run:
        data.mkdir(parents=True, exist_ok=True)
        for name, src in sources.items():
            shutil.copy2(src, copies[name])
        report = sources["train.jsonl"].parent / "manifest_report.json"
        if report.exists():
            shutil.copy2(report, data / "manifest_report.json")
        info["reused"] = False
    cfg = dict(cfg, train_manifest=str(copies["train.jsonl"]),
               val_manifest=str(copies["val.jsonl"]) if "val.jsonl" in copies else "")
    return cfg, info


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    if args.dry_run:  # "-1", not "": on Windows an empty value deletes the variable
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    use_official_source()
    from voxcpm.training.config import load_yaml_config

    cfg = apply_steps_override(load_yaml_config(args.config), args.steps)
    trainer = load_official_trainer()
    signature = train_signature(trainer)
    unknown = set(cfg) - set(signature.parameters)
    if unknown:
        raise SystemExit(f"{args.config}: keys the official train() does not accept: {sorted(unknown)}")
    signature.bind(**cfg)
    missing_globals = {"validate", "generate_sample_audio"} - (train_globals_used(trainer)
                                                               | set(trainer.validate.__code__.co_names))
    if missing_globals:
        raise SystemExit(f"official trainer no longer resolves {sorted(missing_globals)} as module globals; "
                         "the preview throttle / deterministic validation would not apply")
    run_root = Path(cfg["save_path"]).parent
    resuming_copies = args.resume and (run_root / "data" / "train.jsonl").exists()
    source_files = manifest_files(cfg)
    data_files = source_files
    if resuming_copies:  # the run's own frozen data decides; the source manifests may have been rebuilt
        frozen_cfg, _ = frozen_copies(cfg, run_root, resume=True, dry_run=True)
        notes = preflight(frozen_cfg, True, args.allow_restored_audio, check_builder=False)
        data_files = manifest_files(frozen_cfg)
    else:
        notes = preflight(cfg, args.resume, args.allow_restored_audio)
    manifest_report, _ = C.manifest_report_status(data_files["train.jsonl"])
    check_report, check_problems = check_report_status(args.config, source_files)
    valid_every = int(cfg["valid_interval"])
    if args.preview_interval < 0 or (args.preview_interval and args.preview_interval % valid_every):
        raise SystemExit(f"--preview-interval must be 0 or a multiple of valid_interval ({valid_every})")
    train_rows = C.read_jsonl(data_files["train.jsonl"])
    lengths = C.sequence_lengths(train_rows, C.load_tokenizer())
    estimate = C.estimate_train_memory(cfg, lengths, args.frozen_dtype, args.grad_ckpt)
    eff = int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])
    num_iters = int(cfg["num_iters"])
    host_need = C.HOST_LOAD_GB[f"train_{args.frozen_dtype}"]
    has_val = bool(cfg.get("val_manifest"))
    plan = {
        "config": str(args.config),
        "run_dir": str(run_root),
        "dataset": (manifest_report or {}).get("dataset"),
        "audio": (manifest_report or {}).get("audio"),
        "steps_override": args.steps or None,
        "train_rows": len(train_rows),
        "train_minutes": round(sum(float(r["duration"]) for r in train_rows) / 60, 2),
        "ref_rows": sum(1 for r in train_rows if r.get("ref_audio")),
        "instruction_rows": sum(1 for r in train_rows if r.get("instruction")),
        "effective_batch": eff,
        "steps_per_epoch": round(len(train_rows) / eff, 2),
        "epochs": round(num_iters * eff / max(1, len(train_rows)), 1),
        "schedule": {k: cfg[k] for k in ("num_iters", "max_steps", "warmup_steps", "learning_rate",
                                         "valid_interval", "save_interval")},
        "validations_at": (sorted({s for s in range(num_iters) if s % valid_every == 0} | {num_iters - 1})
                           if has_val else "none (no val manifest)"),
        "validation_seed": args.val_seed if args.val_seed >= 0 else "official (random)",
        "previews_at": (sorted({s for s in range(num_iters) if args.preview_interval and s % args.preview_interval == 0}
                               | ({num_iters - 1} if args.preview_interval else set())) if has_val else []),
        "checkpoints": expected_checkpoints(cfg),
        "memory_estimate": estimate,
        "gpu_rule": {"max_vram_gb": args.max_vram_gb, "reserve_gb": args.reserve_gb,
                     "headroom_gb": C.gpu_headroom_gb(estimate["total_worst_gb"]),
                     "min_free_vram_to_start_gb": round(estimate["total_worst_gb"]
                                                        + C.gpu_headroom_gb(estimate["total_worst_gb"])
                                                        - C.CONTEXT_GB + args.reserve_gb + 0.25, 1)},
        "host_memory": {"load_peak_estimate_gb": host_need, "margin_gb": args.host_margin_gb,
                        "required_free_commit_gb": round(host_need + args.host_margin_gb, 1),
                        "now": C.host_memory()},
        "frozen_training_data": str(run_root / "data") + (" (reused, --resume)" if resuming_copies else ""),
        "check_config_report": "ok" if not check_problems else check_problems,
        "notes": notes,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        now = plan["host_memory"]["now"] or {}
        if now and now.get("commit_free_gb", 0) < host_need + args.host_margin_gb:
            print(f"note: only {now['commit_free_gb']:.1f} GB of commit is free right now; the real run would "
                  "refuse to start until enough is free.")
        if check_problems and not args.skip_check_report and not args.resume:
            print("note: the real run would refuse to start: " + "; ".join(check_problems))
        print("dry run: official train() signature bound, config and manifests validated; GPU untouched, "
              "nothing written.")
        return
    # a resumed run was already gated by the report when it first started (recorded in run_config.json)
    if check_problems and not args.skip_check_report and not args.resume:
        raise SystemExit("check_config report does not cover this run:\n  - " + "\n  - ".join(check_problems))

    import torch

    cap, gpu = C.gpu_budget(args.max_vram_gb, args.reserve_gb, estimate["total_worst_gb"])
    host = C.host_ram_budget(host_need, args.host_margin_gb, "training model load")
    stats = install_patches(args.frozen_dtype, args.grad_ckpt)
    throttle_previews(trainer, args.preview_interval, num_iters)
    deterministic_validation(trainer, args.val_seed)
    run_root.mkdir(parents=True, exist_ok=True)
    run_cfg, copies = frozen_copies(cfg, run_root, args.resume, dry_run=False)
    record = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "wrapper": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "config": run_cfg,
        "config_file": str(args.config),
        "config_sha256": C.sha256_file(args.config),
        "audio_variant": (manifest_report or {}).get("audio"),
        "dataset": (manifest_report or {}).get("dataset"),
        "train_manifest_sha256": C.sha256_file(Path(run_cfg["train_manifest"])),
        "val_manifest_sha256": C.sha256_file(Path(run_cfg["val_manifest"])) if run_cfg.get("val_manifest") else None,
        "training_data": copies,
        "manifest_builder": (manifest_report or {}).get("builder"),
        "check_config_report": {"ok": not check_problems, "problems": check_problems,
                                "finished": (check_report or {}).get("finished")},
        "code_sha256": code_hashes(),
        "official_trainer": str(C.OFFICIAL_TRAINER),
        "torch": torch.__version__,
        "gpu": gpu,
        "host_memory_at_start": host,
        "plan": plan,
    }
    if args.resume and (run_root / "run_config.json").exists():  # keep the first start's record
        first = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
        record["resumed_from"] = {k: first.get(k) for k in ("started", "argv", "train_manifest_sha256")}
        record["started_first"] = first.get("started_first") or first.get("started")
    C.atomic_write_json(run_root / "run_config.json", record)
    started = time.time()
    status = "failed"
    try:
        trainer.train(**run_cfg)
        status = "completed"
    finally:
        saved = sorted(p.name for p in Path(cfg["save_path"]).glob("step_*") if p.is_dir())
        steps_saved = [int(name.split("_")[1]) for name in saved]
        summary = {
            "status": status,
            "minutes": round((time.time() - started) / 60, 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "allocator_cap_gb": gpu.get("allocator_cap_gb"),
            "host_commit": C.process_commit(),
            "patch_stats": stats,
            "validation_seed": args.val_seed,
            "checkpoints": saved,
            "val_curve": val_curve(Path(cfg["save_path"]) / "train.log", steps_saved, num_iters),
            "train_log": str(Path(cfg["save_path"]) / "train.log"),
        }
        C.atomic_write_json(run_root / "train_summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
