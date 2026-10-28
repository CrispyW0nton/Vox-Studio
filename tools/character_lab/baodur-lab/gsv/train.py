"""Headless GPT-SoVITS fine-tuning for one character: SoVITS (s2) then GPT (s1), as webui open1Ba / open1Bb do.

Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters; <lab> is its lab folder and
<speaker> its config "speaker" (experiments are <speaker>_<version>_<audio>[_trval]).
Configs are built from the upstream templates exactly like the webui (s2<version>.json with fp16, batch, epochs,
text_low_lr_rate, pretrained G/D, save flags; s1longer-v2.yaml with precision, batch, epochs, pretrained_s1,
save flags, data paths) and written to <exp>/configs/. Weights land outside the repo:
  <lab>/gsv/weights/<exp>/SoVITS/<exp>_e<E>_s<S>.pth      (half precision, version byte 06 = v2ProPlus)
  <lab>/gsv/weights/<exp>/GPT<tag>/<exp><tag>-e<E>.ckpt    (tag '' or e.g. '_dpo' for the DPO arm)
Training state (resumable): <exp>/logs_s2_<version>/G_*.pth, D_*.pth and <exp>/logs_s1_<version><tag>/ckpt/.

Defaults were sized on the first character's screened single-speaker train split (Bao-Dur: 231 clips, 15.2 min,
median 3.3 s, max 12.2 s; the val clips stay held out as ground truth for checkpoint selection, see infer.py
--val-lines). Updates per epoch scale with the clip count, so a larger set makes more updates per export:
  SoVITS 12 epochs, batch 4, exports every 2 epochs (e2 ... e12). Nobody knows the best epoch for a set in
         advance, so the exports bracket the plausible range and the selection sweep picks one: e8 is the webui
         default (8 epochs; batch 4 is the webui default on an 8 GB card), e2 is about the number of updates the
         webui would make on this 32 GB GPU for the 231-clip set (batch 16 x 8 epochs, ~117 updates), e12 tests
         longer training. Upstream LR: 1e-4 (text embedding / encoder / MRTE at 0.4x), decayed by 0.999875 per epoch.
  GPT    15 epochs, batch 8, exports e5/e10/e15 (webui defaults). The optimizer steps once per 4 batches
         (t2s_lightning_module.py: batch_idx > 0 and batch_idx % 4 == 0), ~7 updates per epoch on 231 clips.
         The LR is constant: upstream WarmupCosineLRSchedule pins it at 0.002 on every scheduler step
         (AR/modules/lr_schedulers.py:58), and the very first update runs at ScaledAdam's construction LR 0.01
         because scheduler.step() follows opt.step(). The yaml's warmup_steps / decay_steps / lr_end are inert;
         epochs (and batch, effective batch ~32) are the only levers - choose the GPT export by the sweep.
  --s1-dpo trains upstream's experimental DPO GPT (negative samples against skips/repeats; the data module
         halves the batch, ~2x time) as a separate arm: tag '_dpo' -> logs_s1_<version>_dpo, weights GPT_dpo.
Every GPU stage caps its CUDA allocator (--s2-mem-gb 8 / --s1-mem-gb 6; the CUDA context comes on top) and refuses
to start unless --reserve-gb (8 GB, Substance Painter's headroom) stays free. A CUDA OOM (torch.OutOfMemoryError or
a cuBLAS/cuDNN allocation failure) ends the attempt; the stage restarts with half the batch (not below --min-batch)
and resumes from its last checkpoint.
Host commit: on Windows each DataLoader worker is a spawned process that re-imports torch (+ the SoVITS loader's
imports: 2.36 GB commit measured per worker). Defaults are 0 SoVITS workers (loading runs in the trainer, a few
minutes slower) and 1 GPT worker (upstream needs >= 1). Before each attempt the host commit guard checks for
the estimated commit (SoVITS ~7 GB + 2.4 GB per worker, GPT ~6 GB + 2.4 GB per worker) plus a 2 GB margin.
--device is required (no default): CPU training of the full set would take hours.

Resume rules: a stage whose training state was produced from the same prepared data (sha256 of 2-name2text.txt and
6-name2semantic.tsv, kept in <exp>/train_state.json) and the same mode (not a smoke run) resumes; finished stages
resume at their final epoch and exit at once (raise --s2-epochs / --s1-epochs to continue training). If the
prepared data changed, or the state came from a smoke run, upstream would silently resume it, so this script
refuses until --restart archives that state, its weights and (s2) its TensorBoard files.
train_report.json gets per-epoch loss curves from the TensorBoard logs (s2: loss/g/mel, loss/g/kl_ssl, loss/g/total,
loss/d/total; s1: total_loss_epoch, top_3_acc_epoch) so over- or under-fitting is visible next to the exports.

Usage:
  python train.py --device cuda                                  # <speaker>_v2ProPlus_original (train split)
  python train.py --device cuda --stage s1 --s1-dpo              # optional DPO GPT arm on the same data
  python train.py --device cuda --splits train,val --s2-epochs 8 --s1-epochs 10   # production retrain (frozen)
  python train.py --device cuda --restart                        # after re-exporting / re-preparing the data
v2Pro fallback (if v2ProPlus cannot train even at batch 2) needs its own prepared experiment, because its
semantic tokens come from s2Gv2Pro:
  prepare.py --version v2Pro ; train.py --version v2Pro --device cuda ; infer.py --exp <speaker>_v2Pro_original
Smoke test on CPU (tiny, separate roots are mandatory with --smoke-max-steps):
  train.py --device cpu --smoke-max-steps 6 --exp-root <scratch>/exp --weights-root <scratch>/weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402

# PyYAML (s1 config) and tensorboard (loss curves for train_report.json) live in the GPT-SoVITS venv: re-run there.
C.reexec_in_gsv_venv(__file__)
import yaml  # noqa: E402


def preflight(exp_dir: Path, version: str, allow_incomplete: bool, exclusions: str | None,
              allow_restored: bool) -> list[str]:
    need = ["2-name2text.txt", "4-cnhubert", "5-wav32k", "6-name2semantic.tsv"] + (["7-sv_cn"] if "Pro" in version else [])
    missing = [n for n in need if not (exp_dir / n).exists()]
    if missing:
        raise SystemExit(f"{exp_dir} is not prepared (missing {missing}); run prepare.py first")
    report_path = exp_dir / "prep_report.json"
    if not report_path.exists() and not allow_incomplete:
        raise SystemExit(f"{report_path} missing; run prepare.py (or pass --allow-incomplete-prep)")
    prep = C.read_json(report_path) if report_path.exists() else {}
    if prep and not prep.get("complete") and not allow_incomplete:
        raise SystemExit(f"prep_report says prep is incomplete ({prep.get('problems')}, missing per output "
                         f"{prep.get('coverage_missing')}); run prepare.py or pass --allow-incomplete-prep")
    stamp_path = exp_dir / "prep_stamp.json"
    stamp = C.read_json(stamp_path) if stamp_path.exists() else {}
    kind = stamp.get("audio_kind") or prep.get("audio_kind")
    if kind is None and not allow_incomplete:
        raise SystemExit(f"{exp_dir}: prep_stamp.json does not say which audio was prepared; re-run prepare.py")
    C.require_original_audio(kind or "original", allow_restored, f"training on {exp_dir.name}")
    if stamp.get("version") not in (None, version):
        raise SystemExit(f"{exp_dir} was prepared for {stamp['version']} (semantic tokens from its s2G), not "
                         f"{version}; prepare a separate experiment with prepare.py --version {version}")
    corpus = C.load_corpus()
    guard = C.HeldOut(corpus)
    names = [ln.split("\t")[0] for ln in (exp_dir / "2-name2text.txt").read_text(encoding="utf-8").splitlines()
             if ln.count("\t") == 3]
    semantic = [ln.split("\t")[0] for ln in (exp_dir / "6-name2semantic.tsv").read_text(encoding="utf-8").splitlines()[1:]
                if "\t" in ln]
    on_disk = [p.name for p in (exp_dir / "5-wav32k").glob("*")]
    for name in names + semantic + on_disk:
        reason = guard.reason(clip_id=Path(name).stem)
        if reason:
            raise SystemExit(f"refusing to train: {reason} is present in {exp_dir}")
    if not names:
        raise SystemExit(f"{exp_dir / '2-name2text.txt'} lists no clips")
    # Both trainers read exactly the clips in 2-name2text.txt (SoVITS) / its intersection with the semantic
    # table (GPT), so that is the set that must be free of other characters' lines.
    screen = C.screen_training_ids([Path(n).stem for n in set(names) | set(semantic)], corpus, exclusions)
    if screen["other_speaker"]:
        raise SystemExit(f"refusing to train: {exp_dir} was prepared from a list with other characters' lines "
                         f"{screen['other_speaker']}; re-run export_dataset.py and prepare.py")
    for clip, why in screen["screened"].items():
        C.log(f"note: training includes {clip}, which the default screen leaves out: {why}")
    if exclusions is None:
        C.log("WARNING: --no-exclusions: the prepared clips were not checked against the speaker screen")
    splits = sorted({r["split"] for r in corpus if r["id"] in {Path(n).stem for n in names}})
    C.log(f"prepared clips by split: {splits}")
    return names


def data_fingerprint(exp_dir: Path) -> str:
    h = hashlib.sha256()
    for part in ("2-name2text.txt", "6-name2semantic.tsv"):
        h.update(part.encode())
        h.update((exp_dir / part).read_bytes())
    return h.hexdigest()


def stage_key(stage: str, tag: str) -> str:
    return stage if stage == "s2" else f"s1{tag}"


def stage_state_paths(stage: str, exp_dir: Path, version: str, sovits_dir: Path, gpt_dir: Path,
                      tag: str = "") -> tuple[Path, Path]:
    """(resumable training-state folder, exported weights folder) of a stage."""
    if stage == "s2":
        return exp_dir / f"logs_s2_{version}", sovits_dir
    return exp_dir / f"logs_s1_{version}{tag}", gpt_dir


def has_training_state(stage: str, state_dir: Path) -> bool:
    if stage == "s2":
        return any(state_dir.glob("G_*.pth")) or any(state_dir.glob("D_*.pth"))
    return any((state_dir / "ckpt").glob("*.ckpt"))


def s2_event_files(exp_dir: Path) -> list[Path]:
    """upstream s2_train writes its TensorBoard files into s2_ckpt_dir (= the experiment folder) and <exp>/eval."""
    return sorted(list(exp_dir.glob("events.out.tfevents.*")) + list((exp_dir / "eval").glob("events.out.tfevents.*")))


def guard_resume(args, stage: str, exp_dir: Path, sovits_dir: Path, gpt_dir: Path, fingerprint: str,
                 state: dict) -> None:
    """Upstream resumes whatever checkpoint sits in the state folder. Refuse to resume state that was trained on
    different prepared data or by a smoke run; with --restart archive it (and the weights exported from it)."""
    tag = args.s1_tag
    state_dir, weights_dir = stage_state_paths(stage, exp_dir, args.version, sovits_dir, gpt_dir, tag)
    previous = state.get(stage_key(stage, tag)) or {}
    exists = state_dir.exists() and has_training_state(stage, state_dir)
    mode_now = {"smoke_max_steps": args.smoke_max_steps}
    mode_then = {"smoke_max_steps": previous.get("smoke_max_steps", 0)}
    if exists and previous.get("data_sha256") == fingerprint and mode_now == mode_then and not args.restart:
        C.log(f"{stage}{tag if stage == 's1' else ''}: resuming existing training state in {state_dir}")
        return
    if exists and not args.restart:
        why = (f"different prepared data ({'unknown' if not previous.get('data_sha256') else previous['data_sha256'][:12]}"
               f" != {fingerprint[:12]})" if previous.get("data_sha256") != fingerprint
               else f"a different run mode (then {mode_then}, now {mode_now})")
        raise SystemExit(f"{stage}: {state_dir} holds training state from {why}; upstream would resume it silently. "
                         f"Pass --restart to archive it and train from the pretrained model.")
    if args.restart:
        stamp = f"stale-{time.strftime('%Y%m%d-%H%M%S')}"
        for folder in (state_dir, weights_dir):
            if folder.exists() and any(folder.iterdir()):
                target = folder.with_name(f"{folder.name}.{stamp}")
                os.replace(folder, target)
                C.log(f"{stage}: archived {folder} -> {target}")
        if stage == "s2":
            events = s2_event_files(exp_dir)
            if events:
                target = exp_dir / f"tb_s2.{stamp}"
                target.mkdir(parents=True, exist_ok=True)
                for p in events:
                    os.replace(p, target / (p.name if p.parent == exp_dir else f"eval.{p.name}"))
                C.log(f"s2: archived {len(events)} TensorBoard file(s) -> {target}")
        weights_dir.mkdir(parents=True, exist_ok=True)


def s2_config(args, exp_dir: Path, name: str, batch: int, sovits_dir: Path) -> dict:
    version, on_gpu = args.version, args.device != "cpu"
    data = json.loads(C.S2_BASE_CONFIG[version].read_text(encoding="utf-8"))
    g, d = C.S2_PRETRAINED[version]
    data["train"].update({
        "fp16_run": on_gpu, "batch_size": batch, "epochs": args.s2_epochs, "text_low_lr_rate": args.text_low_lr_rate,
        "pretrained_s2G": str(g), "pretrained_s2D": str(d), "if_save_latest": True, "if_save_every_weights": True,
        "save_every_epoch": args.s2_save_every,
        # upstream s2_train sets CUDA_VISIBLE_DEVICES from this at import: keep the caller's GPU in GPU mode
        "gpu_numbers": C.gpu_visible_devices(args.device) if on_gpu else "0",
        "grad_ckpt": False, "lora_rank": "32", "log_interval": args.log_interval,
    })
    if args.s2_lr:
        data["train"]["learning_rate"] = args.s2_lr
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(exp_dir)
    data["save_weight_dir"] = str(sovits_dir)
    data["name"] = name
    data["version"] = version
    return data


def s1_config(args, exp_dir: Path, name: str, batch: int, gpt_dir: Path) -> dict:
    data = yaml.safe_load(C.S1_BASE_CONFIG.read_text(encoding="utf-8"))
    if args.device == "cpu":
        data["train"]["precision"] = "32"
    data["train"].update({
        "batch_size": batch, "epochs": args.s1_epochs, "save_every_n_epoch": args.s1_save_every,
        "if_save_every_weights": True, "if_save_latest": True, "if_dpo": bool(args.s1_dpo),
        "half_weights_save_dir": str(gpt_dir), "exp_name": f"{name}{args.s1_tag}",
    })
    data["data"]["num_workers"] = max(1, args.s1_workers)  # upstream uses persistent workers (needs >= 1)
    data["pretrained_s1"] = str(C.S1_PRETRAINED)
    data["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
    data["output_dir"] = str(exp_dir / f"logs_s1_{args.version}{args.s1_tag}")
    return data


def commit_need(args, stage: str) -> float:
    if stage == "s2":
        return C.COMMIT_GB["s2_main"] + C.COMMIT_GB["s2_worker"] * max(0, args.loader_workers)
    return C.COMMIT_GB["s1_main"] + C.COMMIT_GB["s1_worker"] * max(1, args.s1_workers)


def run_stage(args, stage: str, exp_dir: Path, build, batch: int, mem_gb: float, extra_env: dict) -> dict:
    wrapper = {"s2": "gsv_s2_train.py", "s1": "gsv_s1_train.py"}[stage]
    suffix = {"s2": ".json", "s1": ".yaml"}[stage]
    label = stage if stage == "s2" else f"s1{args.s1_tag}"
    attempts = []
    while True:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        data = build(batch)
        text = json.dumps(data, indent=2) if stage == "s2" else yaml.dump(data, default_flow_style=False)
        cfg = exp_dir / "configs" / f"{label}{suffix}"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(text, encoding="utf-8")
        (exp_dir / "configs" / f"{label}-{stamp}{suffix}").write_text(text, encoding="utf-8")  # provenance
        report_path = exp_dir / "logs" / f"{label}_run_{stamp}.json"
        cmd = [C.GSV_PY, C.LAB_CODE / wrapper, "--config", cfg, "--device", args.device, "--mem-gb", mem_gb,
               "--reserve-gb", args.reserve_gb, "--wait-min", args.wait_min, "--threads", args.threads,
               "--report", report_path]
        if args.smoke_max_steps:
            cmd += ["--smoke-max-steps", args.smoke_max_steps]
        if stage == "s2":
            cmd += ["--loader-workers", args.loader_workers]
        commit = C.require_commit(commit_need(args, stage), f"{label} training", args.wait_min)
        C.log(f"{label}: batch {batch}, cap {mem_gb} GB on {args.device} -> {cfg}")
        t0 = time.time()
        rc = C.run_logged(cmd, C.gsv_env(args.device, args.threads, extra_env), exp_dir / "logs" / f"train_{label}.log")
        attempt = {"batch": batch, "exit_code": rc, "seconds": round(time.time() - t0, 1), "config": str(cfg),
                   "host_commit": commit}
        if report_path.exists():
            attempt["report"] = C.read_json(report_path)
        attempts.append(attempt)
        if rc == 0:
            return {"attempts": attempts, "batch": batch}
        if rc == 3 and batch // 2 >= args.min_batch:
            C.log(f"{label}: out of memory at batch {batch}; retrying with batch {batch // 2} (resumes from checkpoint)")
            batch //= 2
            continue
        hint = {4: " (VRAM guard: not enough free VRAM; see --wait-min)",
                5: " (host commit guard: not enough commit; see --wait-min / --loader-workers)"}.get(rc, "")
        raise SystemExit(f"{label} failed (exit {rc}){hint}; log: {exp_dir / 'logs' / f'train_{label}.log'}")


# ----------------------------------------------------------------------------------------------- loss curves
def _scalars(files: list[Path]) -> dict[str, list[tuple[float, int, float]]]:
    """{tag: [(wall_time, step, value)]} from TensorBoard event files (tensorboard is in the GPT-SoVITS venv)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

    out: dict[str, list] = {}
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((e.wall_time, e.step, e.value) for e in acc.Scalars(tag))
    for tag in out:
        out[tag].sort()
    return out


def loss_curves(stage: str, exp_dir: Path, version: str, tag: str, epochs: int, since: float | None) -> dict:
    """Per-epoch means of the main training scalars. s2 logs every --log-interval steps, so its points are binned
    into `epochs` equal step ranges (approximate when an OOM retry changed the batch); s1 logs once per epoch.
    Only events written since this stage's first start in the current state (`since`) are used."""
    try:
        if stage == "s2":
            files = s2_event_files(exp_dir)
            files = [f for f in files if f.parent == exp_dir]  # the eval writer stays empty upstream
            wanted = ("loss/g/mel", "loss/g/kl_ssl", "loss/g/total", "loss/d/total", "loss/g/fm", "grad_norm_g")
        else:
            files = sorted((exp_dir / f"logs_s1_{version}{tag}").glob("**/events.out.tfevents.*"))
            wanted = ("total_loss_epoch", "top_3_acc_epoch")
        if not files:
            return {"note": "no TensorBoard event files found"}
        scalars = _scalars(files)
    except Exception as exc:  # diagnostics only; never fail a finished training over it
        return {"error": f"{type(exc).__name__}: {exc}"}
    curves: dict = {}

    def recent(name: str) -> list:
        return [(w, s, v) for w, s, v in scalars.get(name, []) if since is None or w >= since - 5]

    # Lightning logs an 'epoch' scalar (0-based) at the same step as the epoch-end metrics; a resumed run repeats
    # epochs after its last checkpoint, and the later value wins.
    epoch_at_step = {s: int(round(v)) + 1 for _, s, v in recent("epoch")}
    for name in wanted:
        points = recent(name)
        if not points:
            continue
        if stage == "s2":
            last = max(s for _, s, _ in points) + 1
            bins: dict[int, list[float]] = {}
            for _, step, value in points:
                bins.setdefault(min(epochs, int(step * epochs / last) + 1), []).append(value)
            per_epoch = {e: round(sum(v) / len(v), 4) for e, v in sorted(bins.items())}
        elif epoch_at_step:
            per_epoch = {}
            for _, step, value in points:
                if step in epoch_at_step:
                    per_epoch[epoch_at_step[step]] = round(value, 4)
            per_epoch = dict(sorted(per_epoch.items()))
        else:
            per_epoch = {i + 1: round(v, 4) for i, (_, _, v) in enumerate(points[-epochs:])}
        curves[name] = {"points": len(points), "per_epoch": per_epoch}
    if stage == "s2" and curves:
        curves["note"] = "per_epoch = mean of the logged steps in each 1/epochs share of the step range"
    return curves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", choices=C.AUDIO_MODES, default="original")
    ap.add_argument("--allow-restored", action="store_true",
                    help="permit restored audio (DECISIONS.md: the user chose original audio for training)")
    ap.add_argument("--splits", default=",".join(C.DEFAULT_SPLITS),
                    help="which prepared experiment: train (default) or train,val (production retrain)")
    ap.add_argument("--version", choices=C.VERSIONS, default="v2ProPlus")
    ap.add_argument("--name", default=None, help=f"experiment name (default {C.SPEAKER}_<version>_<audio>[_trval])")
    ap.add_argument("--exp-root", default=str(C.EXP_ROOT))
    ap.add_argument("--weights-root", default=str(C.WEIGHTS_ROOT))
    ap.add_argument("--stage", choices=("all", "s2", "s1"), default="all")
    ap.add_argument("--device", required=True, help="cuda (GPU training) or cpu (smoke tests only)")
    ap.add_argument("--restart", action="store_true",
                    help="archive existing training state + weights of the selected stage(s) and start fresh")
    ap.add_argument("--s2-epochs", type=int, default=12)
    ap.add_argument("--s2-batch", type=int, default=4,
                    help="estimate (unmeasured): ~2.6 GB fixed (G+D weights, grads, AdamW) + ~0.7 GB per item in fp16")
    ap.add_argument("--s2-save-every", type=int, default=2)
    ap.add_argument("--s2-lr", type=float, default=None, help="SoVITS learning rate (template: 1e-4)")
    ap.add_argument("--text-low-lr-rate", type=float, default=0.4, help="webui default 0.4")
    ap.add_argument("--s2-mem-gb", type=float, default=8.0)
    ap.add_argument("--s1-epochs", type=int, default=15)
    ap.add_argument("--s1-batch", type=int, default=8)
    ap.add_argument("--s1-save-every", type=int, default=5)
    ap.add_argument("--s1-dpo", action="store_true", help="upstream's experimental DPO GPT training (2x cost)")
    ap.add_argument("--s1-tag", default=None,
                    help="separate GPT arm (state logs_s1_<version><tag>, weights GPT<tag>); --s1-dpo implies _dpo")
    ap.add_argument("--s1-mem-gb", type=float, default=6.0)
    ap.add_argument("--s1-workers", type=int, default=1, help="GPT DataLoader workers (>= 1; upstream template: 4)")
    ap.add_argument("--loader-workers", type=int, default=0,
                    help="SoVITS DataLoader workers (upstream: 5); each costs ~2.4 GB of host commit on Windows")
    ap.add_argument("--min-batch", type=int, default=2)
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    ap.add_argument("--wait-min", "--wait-vram-min", dest="wait_min", type=float, default=0.0,
                    help="wait up to N minutes for enough free VRAM / host commit")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--log-interval", type=int, default=20, help="SoVITS loss logging interval (steps)")
    ap.add_argument("--smoke-max-steps", type=int, default=0,
                    help="smoke tests only (needs scratch --exp-root/--weights-root): cap batches per epoch")
    ap.add_argument("--allow-incomplete-prep", action="store_true")
    ap.add_argument("--exclusions", default=str(C.EXCLUSIONS), help="speaker-screen exclusions JSON")
    ap.add_argument("--no-exclusions", action="store_true",
                    help="DANGER: do not refuse prepared data that contains other characters' lines")
    args = ap.parse_args()
    args.device = C.normalize_device(args.device)
    for key in ("exp_root", "weights_root", "exclusions"):
        setattr(args, key, C.abspath(getattr(args, key)))
    args.s1_tag = C.check_tag(args.s1_tag if args.s1_tag is not None else ("_dpo" if args.s1_dpo else ""))
    if args.device != "cpu":
        C.gpu_visible_devices(args.device)  # fail fast: never re-expose a GPU the caller hid
    if args.smoke_max_steps and (args.exp_root == C.EXP_ROOT.resolve() or args.weights_root == C.WEIGHTS_ROOT.resolve()):
        raise SystemExit("--smoke-max-steps writes partially trained state and weights: use scratch --exp-root and "
                         "--weights-root, never the real experiment folders")
    C.require_original_audio(args.audio, args.allow_restored, "training")

    for stage, epochs, every in (("s2", args.s2_epochs, args.s2_save_every), ("s1", args.s1_epochs, args.s1_save_every)):
        if every < 1 or epochs % every:
            raise SystemExit(f"--{stage}-epochs {epochs} is not a multiple of --{stage}-save-every {every}: upstream "
                             f"only exports weights on multiples, so the final epoch would be lost")

    splits = C.parse_splits(args.splits)
    name = args.name or C.exp_name(args.version, args.audio, splits)
    exp_dir = args.exp_root / name
    weights_root = args.weights_root
    sovits_dir, gpt_dir = C.weights_dirs(name, weights_root, args.s1_tag)
    clips = preflight(exp_dir, args.version, args.allow_incomplete_prep,
                      None if args.no_exclusions else args.exclusions, args.allow_restored)
    fingerprint = data_fingerprint(exp_dir)
    state_path = exp_dir / "train_state.json"
    state = C.read_json(state_path) if state_path.exists() else {}
    stages = [s for s in ("s2", "s1") if args.stage in ("all", s)]
    for stage in stages:  # check both before any GPU work starts
        guard_resume(args, stage, exp_dir, sovits_dir, gpt_dir, fingerprint, state)
    sovits_dir.mkdir(parents=True, exist_ok=True)
    gpt_dir.mkdir(parents=True, exist_ok=True)
    # webui open1Ba creates this before launching s2_train, whose checkpoint saves assume it exists
    (exp_dir / f"logs_s2_{args.version}").mkdir(parents=True, exist_ok=True)
    C.log(f"{name}: {len(clips)} prepared clips in {exp_dir} (data {fingerprint[:12]})")

    report_path = exp_dir / "train_report.json"
    report = C.read_json(report_path) if report_path.exists() else {}
    report.update({"exp": name, "exp_dir": str(exp_dir), "version": args.version, "device": args.device,
                   "clips": len(clips), "data_sha256": fingerprint, "weights": {}})
    report.setdefault("stages", {})
    env = {"version": args.version, "hz": "25hz"}

    for stage in stages:
        key = stage_key(stage, args.s1_tag)
        previous = state.get(key) or {}
        resumed = previous.get("data_sha256") == fingerprint and not args.restart
        state[key] = {"data_sha256": fingerprint, "device": args.device, "smoke_max_steps": args.smoke_max_steps,
                      "first_started": previous.get("first_started") if resumed and previous.get("first_started")
                      else time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "first_started_ts": previous.get("first_started_ts") if resumed and previous.get("first_started_ts")
                      else time.time(),
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if stage == "s1":
            state[key].update({"dpo": bool(args.s1_dpo), "tag": args.s1_tag})
        C.write_json(state_path, state)
        if stage == "s2":
            result = run_stage(args, "s2", exp_dir, lambda b: s2_config(args, exp_dir, name, b, sovits_dir),
                               args.s2_batch, args.s2_mem_gb, env)
            epochs = [e for e, _ in C.list_weights(name, weights_root)["sovits"]]
            final, kind, folder, total = args.s2_epochs, "SoVITS", sovits_dir, args.s2_epochs
        else:
            result = run_stage(args, "s1", exp_dir, lambda b: s1_config(args, exp_dir, name, b, gpt_dir),
                               args.s1_batch, args.s1_mem_gb, env)
            epochs = [e for e, _ in C.list_weights(name, weights_root, args.s1_tag)["gpt"]]
            final, kind, folder, total = args.s1_epochs, f"GPT{args.s1_tag}", gpt_dir, args.s1_epochs
        curves = loss_curves(stage, exp_dir, args.version, args.s1_tag, total, state[key]["first_started_ts"])
        report["stages"][key] = {**result, "epochs": final, "curves": curves,
                                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if stage == "s1":
            report["stages"][key].update({"dpo": bool(args.s1_dpo), "tag": args.s1_tag})
        state[key]["finished"] = report["stages"][key]["finished"]
        C.write_json(state_path, state)
        C.write_json(report_path, report)
        for curve, info in curves.items() if isinstance(curves, dict) else []:
            if isinstance(info, dict) and "per_epoch" in info:
                C.log(f"{key} {curve}: " + ", ".join(f"e{e} {v}" for e, v in info["per_epoch"].items()))
        if final not in epochs:
            raise SystemExit(f"{kind} training finished but no e{final} weights in {folder} (have {epochs})")

    report["weights"] = C.list_weights(name, weights_root)
    if args.s1_tag:
        report["weights"][f"gpt{args.s1_tag}"] = C.list_weights(name, weights_root, args.s1_tag)["gpt"]
    C.write_json(report_path, report)
    for kind, entries in report["weights"].items():
        for epoch, path in entries:
            C.log(f"{kind} e{epoch}: {path}")
    C.log(f"report -> {report_path}")


if __name__ == "__main__":
    main()
