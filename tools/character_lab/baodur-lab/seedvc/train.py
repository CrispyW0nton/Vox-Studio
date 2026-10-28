"""Fine-tune Seed-VC V2 on a character's train split, VRAM-capped; checkpoints go to <lab>/seedvc/runs/<run>/.

GPU (main session, serial; nothing else of ours on the GPU at the same time):
  cd voice-lab/baodur-lab/seedvc
  [VOICE_LAB_CHARACTER=<slug>] ../../seed-vc/.venv/Scripts/python.exe train.py --audio original --run-name cfm_original --device cuda

The character comes from VOICE_LAB_CHARACTER (voice-lab/characters/<slug>.json, default baodur); every path below is
under its lab folder (BaoDurLab, JoleeLab, ...), so run names such as cfm_original never collide across characters.
Recipe (defaults; data sizes measured on Bao-Dur):
- Fine-tune the CFM (DiT + content length regulator), i.e. upstream `train_v2.py --train-cfm`, from
  Plachta/Seed-VC v2/cfm_small.pth with upstream's optimiser exactly (optimizers.build_single_optimizer: AdamW
  lr 2e-5, betas .9/.98, eps 1e-6, wd .01, exponential decay .999996 floored at 1e-5; grad clip 1000).
- Batch 4, 2000 steps, checkpoint + validation at step 100 and every 250 steps, every checkpoint kept, plus an EMA
  copy of the weights (decay 0.999, upstream's released fine-tune is an EMA) saved beside each one as EMA-CFM_*.
  Data after the speaker and leak screens: ~231 items / ~15 min -> ~57 steps per epoch, so 2000 steps is ~35
  epochs at a small, nearly constant LR (2e-5 -> 1.98e-5). Upstream's "minimum 100 steps" is for one-utterance
  adaptation (hence the step-100 checkpoint); with 15 min of one speaker, similarity keeps improving for longer,
  while over-training shows up as articulation/WER loss on unseen content (README: fine-tuning "may slightly
  increase WER"). The checkpoint sweep + selftest.py's gated selection + listening pick the knee instead of guessing;
  if the pick is the last step, continue with --init-cfm (see the plan).
- Memory: allocator cap 7 GB (--max-vram-gb). A CPU measurement of the worst batch (4 x 1051 frames, fp32) peaked at
  4.1 GB of activations + gradients, ~5.3 GB with weights, AdamW state and the EMA copy. CUDA may pick a different
  attention kernel for the boolean mask, so a preflight measures the worst batch on the GPU first and, with
  --auto-batch (default), steps the batch down (4 -> 3 -> 2) until it fits; the batch actually used is recorded.
- The AR model (only used by convert.py --convert-style) stays pretrained unless --train-ar is given.
- --perturb-content P (A/B, default off): with probability P per item, the target region's content tokens come from
  a pitch/formant-shifted copy of the clip (+-2..5 st) while the prompt region keeps the original tokens, as upstream
  V1 train.py does with its OpenVoice-perturbed S_alt. Fine-tuning otherwise only ever sees the character's own
  tokens (Bao-Dur's change on 70-80 % of frames under a pitch shift), while production sources are other voices. Judge this A/B
  on the out-of-domain self-test only: the pitch-shifted val self-test would favour it unfairly.

Deliberate differences from upstream train_v2.py:
1. Frozen feature extractors stay in eval mode. Upstream calls model.train() on the whole wrapper, so HuBERT runs
   with SpecAugment time masking (mask_time_prob .075), LayerDrop .1 and dropout .1, and CAMPPlus normalises with
   batch statistics (and drifts its BatchNorm running stats) while inference uses eval mode; the CFM would be
   trained on features it never sees at inference. Because eval-mode features are deterministic, content tokens
   and style vectors are computed once per clip, without cross-item padding (HuBERT, ConvNeXt, CAMPPlus stats
   pooling and the regulator's GroupNorm all see padding in upstream's batched path), using the same librosa soxr
   16 kHz resampling as inference. `--upstream-train-mode` reproduces upstream exactly for an A/B.
2. Checkpoints are written to <lab>/seedvc/runs/<run>/ and never deleted (upstream keeps only the latest).
   Format is upstream's ({"net": {"cfm", "length_regulator"}, "iters", "epoch"} + "meta"), so inference_v2.py
   --cfm-checkpoint-path also accepts them (raw and EMA files alike).
3. Deterministic validation loss on the val split (fixed seeds, 4 draws of t/noise/prompt split) at step 0 and at
   every save, for the raw and the EMA weights, logged to val_log.jsonl.
4. Batch size >= 2 and drop_last: upstream crashes on a batch of 1 (CFM.forward squeezes t to a 0-d tensor).
5. No accelerate/DataLoader: single GPU, clips preloaded in RAM, no worker processes (Windows spawn-safe; upstream
   also says num_workers 0 on Windows).
6. Offline HF caches (common.setup_seedvc) instead of upstream's hard-coded HF_HUB_CACHE=./checkpoints/hf_cache.
7. VRAM cap (torch.cuda.set_per_process_memory_fraction), refusal to start if Substance Painter's 8 GB reserve
   is at risk, and the worst-batch preflight. --device defaults to cuda and never falls back to the CPU.
8. Refuses prepared data that holds a held-out test clip, a clip the CURRENT screen excludes (other speakers,
   near-duplicates of test or val lines, uncertain clips when excluded), data prepared under an older
   exclude_other_speakers.json (--allow-stale-screen after checking the change), or restored audio (user decision).
9. Provenance: prepare_report.json is frozen into runs/<run>/training_data/ and run_meta.json "data" points there,
   so rebuilding the prepared folder later cannot change what eval/provenance.py reads for this run.
10. A run folder is never reused: a complete run is refused; a failed or crashed one (status failed_*, or "running"
   with its process gone) is moved aside to <run>_failed_<time>, so a retry under the same name starts clean.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=f"Fine-tune Seed-VC V2 on {C.NAME}")
    p.add_argument("--audio", choices=["original", "restored"], default="original",
                   help=f"which prepared folder to train on ({C.LAB.name}/seedvc/data/<audio>)")
    p.add_argument("--allow-restored", action="store_true",
                   help="only after a new user decision: the user chose original audio on 2026-09-30")
    p.add_argument("--allow-stale-screen", action="store_true",
                   help="train although exclude_other_speakers.json changed after prepare.py ran")
    p.add_argument("--data", type=Path, default=None, help="explicit prepared folder (overrides --audio)")
    p.add_argument("--run-name", default=None, help="default cfm_<audio> (ar/cfm_ar when --train-ar)")
    p.add_argument("--out-root", type=Path, default=C.RUNS)
    p.add_argument("--train-cfm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-ar", action="store_true", help="also fine-tune the AR style model (convert-style only)")
    p.add_argument("--init-cfm", default=None,
                   help="start from this CFM checkpoint (path, or best:<run> = that run's selection) instead of cfm_small")
    p.add_argument("--init-ar", type=Path, default=None, help="start from this AR checkpoint instead of ar_base")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--auto-batch", action=argparse.BooleanOptionalAction, default=True,
                   help="step the batch down to 3, then 2 if the preflight does not fit under the cap")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--save-at", default="100", help="extra comma-separated checkpoint steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.999996)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1000.0)
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA of the trained weights; 0 disables")
    p.add_argument("--perturb-content", type=float, default=0.0,
                   help="probability per item of target-region tokens from a pitch-shifted copy (A/B; 0 = off)")
    p.add_argument("--perturb-semitones", default="2,5", help="min,max absolute shift of the perturbed copies")
    p.add_argument("--perturb-copies", type=int, default=2, help="shifted copies per item (alternating up/down)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-repeats", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu only for smoke tests")
    p.add_argument("--max-vram-gb", type=float, default=7.0, help="allocator cap for this process")
    p.add_argument("--painter-reserve-gb", type=float, default=C.PAINTER_RESERVE_GB)
    p.add_argument("--skip-headroom-check", action="store_true")
    p.add_argument("--skip-preflight", action="store_true", help="skip the worst-case batch memory probe")
    p.add_argument("--mixed-precision", choices=["no", "bf16"], default="no", help="upstream default is fp32")
    p.add_argument("--upstream-train-mode", action="store_true",
                   help="reproduce train_v2.py: whole wrapper in train mode, features recomputed every step")
    p.add_argument("--limit-items", type=int, default=0, help="smoke tests: use only the first N train items")
    p.add_argument("--val-max-items", type=int, default=0, help="smoke tests: use only the first N val items")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def check_prepared(prep: dict, rows: list[dict], allow_stale: bool) -> str | None:
    """Return a reason to refuse the prepared folder, or None."""
    config = prep.get("config", {})
    if not config.get("speaker_exclusions_sha256"):
        others = "/".join(C.other_speaker_names()) or "other characters'"
        return f"prepared before the other-speaker screen ({others} clips); rerun prepare.py --force"
    test_ids = {r["id"] for r in rows if r["split"] == "test"}
    used = {s for item in prep.get("items", []) + prep.get("val_items", []) for s in item["sources"]}
    train_used = {s for item in prep.get("items", []) for s in item["sources"]}
    if used & test_ids:
        return f"held-out test clips in the prepared data: {sorted(used & test_ids)}"
    current = C.speaker_exclusions(required=True, rows=rows)
    problems = []
    for ids, what, pool in ((current["confirmed"], "confirmed other-speaker clips", used),
                            (current["near_test"], "near-duplicates of held-out test lines (eval counts them as leaks)",
                             used),
                            (current["near_val"] if config.get("near_val", "exclude") == "exclude" else {},
                             "near-duplicates of val lines", train_used),
                            (current["uncertain"] if config.get("uncertain") == "exclude" else {},
                             "uncertain speaker-screen outliers", used)):
        hit = sorted(pool & set(ids))
        if hit:
            problems.append(f"{what}: {hit}")
    if problems:
        return "the current screen excludes clips this folder holds (" + "; ".join(problems) + \
               "); rerun prepare.py --force"
    if current["sha256"] != config["speaker_exclusions_sha256"]:
        message = ("delivery/exclude_other_speakers.json changed after prepare.py ran "
                   f"({config['speaker_exclusions_sha256']} -> {current['sha256']}); rerun prepare.py --force")
        if not allow_stale:
            return message + " (or pass --allow-stale-screen after checking the change)"
        print(f"WARNING: {message}; continuing because of --allow-stale-screen", file=sys.stderr)
    return None


def claim_run_dir(run_dir: Path) -> str | None:
    """Refuse a complete or live run; move a failed or crashed one aside. Returns a refusal or None."""
    if not run_dir.exists():
        return None
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = C.read_json(meta_path)
        status = meta.get("status")
        if status == "complete":
            return f"{run_dir} holds a complete run; choose another --run-name"
        if status == "running" and C.pid_alive(meta.get("pid")):
            return f"{run_dir} belongs to a train.py that is still running (pid {meta.get('pid')})"
        target = run_dir.with_name(f"{run_dir.name}_failed_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        run_dir.rename(target)
        print(f"moved the earlier run (status {status!r}) aside to {target}; convert.py refuses incomplete runs")
        return None
    if any(run_dir.iterdir()):
        return f"{run_dir} is not empty and has no run_meta.json; choose another --run-name"
    return None


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        print("ERROR: --batch-size must be >= 2 (Seed-VC CFM.forward fails on a batch of 1)", file=sys.stderr)
        return 2
    if not (args.train_cfm or args.train_ar):
        print("ERROR: nothing to train", file=sys.stderr)
        return 2
    if args.perturb_content and (args.upstream_train_mode or not args.train_cfm):
        print("ERROR: --perturb-content needs CFM training in the default (precomputed) mode", file=sys.stderr)
        return 2
    data_dir = (args.data or C.PREPARED / args.audio).resolve()
    prep = C.read_json(data_dir / "prepare_report.json")
    audio = prep["config"]["audio"]
    C.require_original(audio, args.allow_restored, f"train.py on {data_dir}")
    rows = C.load_corpus()
    refusal = check_prepared(prep, rows, args.allow_stale_screen)
    if refusal:
        print(f"ERROR: {data_dir}: {refusal}", file=sys.stderr)
        return 2
    run_name = args.run_name or "_".join((["cfm"] if args.train_cfm else []) + (["ar"] if args.train_ar else []) + [audio])
    run_dir = (args.out_root / run_name).resolve()
    refusal = claim_run_dir(run_dir)
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 2
    if args.init_cfm and str(args.init_cfm).startswith("best:"):
        tag, path = C.selected_checkpoint(args.out_root, str(args.init_cfm)[5:])
        if path is None:
            print(f"ERROR: {args.init_cfm} resolves to pretrained cfm_small; nothing to continue from", file=sys.stderr)
            return 2
        print(f"--init-cfm {args.init_cfm} -> {tag} ({path})")
        args.init_cfm = path
    args.init_cfm = Path(args.init_cfm).resolve() if args.init_cfm else None   # before setup_seedvc() changes the cwd
    args.init_ar = args.init_ar.resolve() if args.init_ar else None
    save_at = {int(s) for s in args.save_at.split(",") if s.strip()}

    C.setup_seedvc()
    import numpy as np
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence

    def seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(args.seed)
    device = C.pick_device(args.device)
    vram = C.cap_vram(device, args.max_vram_gb, args.painter_reserve_gb, not args.skip_headroom_check)
    run_dir.mkdir(parents=True, exist_ok=True)   # only once the device and the headroom check passed
    frozen = run_dir / "training_data" / "prepare_report.json"   # what eval/provenance.py reads for this run
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / "prepare_report.json", frozen)
    mode = "upstream" if args.upstream_train_mode else "frozen-eval-precomputed"
    started = time.time()
    meta = {"status": "running", "pid": os.getpid(), "run": run_name, "run_dir": str(run_dir),
            "data": str(frozen.parent), "prepared_from": str(data_dir), "training_data": str(frozen), "audio": audio,
            "train_mode": mode, "seedvc_commit": C.seedvc_commit(), "torch": torch.__version__, "vram": vram,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "data_summary": {k: prep.get(k) for k in ("config", "train", "val", "excluded_summary")},
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    C.write_json(run_dir / "run_meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("run", "data", "train_mode", "vram")}, indent=1))

    def fail(status: str, message: str) -> int:
        meta.update({"status": status, "error": message, "seconds": round(time.time() - started, 1),
                     "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        C.write_json(run_dir / "run_meta.json", meta)
        print(f"ERROR ({status}): {message}", file=sys.stderr)
        return 3

    # ------------------------------------------------------------------ model
    wrapper = C.build_wrapper()
    weights = C.load_weights(wrapper, args.init_cfm, args.init_ar)
    meta["base_weights"] = weights
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    trainable = ([("CFM", wrapper.cfm, wrapper.cfm_length_regulator)] if args.train_cfm else []) + \
                ([("AR", wrapper.ar, wrapper.ar_length_regulator)] if args.train_ar else [])
    modules = [m for _, net, reg in trainable for m in (net, reg)]
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    params = [p for module in modules for p in module.parameters()]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f} M")

    # ------------------------------------------------------------------- data
    def load_items(folder: Path, limit: int) -> list[dict]:
        files = sorted(folder.glob("*.wav"))[: limit or None]
        items = []
        for f in files:
            wave, _ = C.load_mono(f, C.MODEL_SR)
            if not (C.MODEL_SR * 1.0 <= len(wave) <= C.MODEL_SR * 30.0):
                raise RuntimeError(f"{f} outside Seed-VC's 1-30 s range; rerun prepare.py")
            mel = wrapper.mel_fn(torch.from_numpy(wave)[None])[0]
            items.append({"name": f.stem, "wave": wave, "mel": mel})
        return items

    train_items = load_items(data_dir / "train", args.limit_items)
    val_items = load_items(data_dir / "val", args.val_max_items)
    if len(train_items) < args.batch_size:
        print("ERROR: fewer train items than the batch size", file=sys.stderr)
        return fail("failed_data", "fewer train items than the batch size")
    print(f"train items {len(train_items)} ({sum(len(i['wave']) for i in train_items) / C.MODEL_SR:.0f} s), "
          f"val items {len(val_items)}")

    if args.upstream_train_mode:
        for name, module in wrapper.named_children():
            if name != "vocoder":
                module.to(device)
    else:
        feature_modules = [wrapper.content_extractor_wide, wrapper.style_encoder] + \
                          ([wrapper.content_extractor_narrow] if args.train_ar else [])
        for module in feature_modules:
            module.to(device)
        t0 = time.time()
        lo_st, hi_st = (float(v) for v in args.perturb_semitones.split(","))
        shift_rng = np.random.default_rng(args.seed + 7)

        def wide_tokens(wave22):
            w16 = torch.from_numpy(C.resample(wave22, C.MODEL_SR, C.FEATURE_SR))[None].to(device)
            return wrapper._process_content_features(w16, is_narrow=False)[0].cpu(), w16

        with torch.no_grad():
            for item in train_items + val_items:
                item["wide"], w16 = wide_tokens(item["wave"])
                item["style"] = wrapper.compute_style(w16).float().cpu()
                if args.train_ar:
                    item["narrow"] = wrapper._process_content_features(w16, is_narrow=True)[0].cpu()
            if args.perturb_content > 0:
                import librosa
                for item in train_items:   # never the val items: validation stays unperturbed and comparable
                    item["wide_alt"], item["alt_shifts"] = [], []
                    for k in range(args.perturb_copies):
                        shift = (1 if k % 2 == 0 else -1) * float(shift_rng.uniform(lo_st, hi_st))
                        shifted = librosa.effects.pitch_shift(item["wave"], sr=C.MODEL_SR, n_steps=shift,
                                                              res_type="soxr_hq").astype(np.float32)
                        alt, _ = wide_tokens(shifted)
                        if alt.shape != item["wide"].shape:
                            raise RuntimeError(f"{item['name']}: shifted copy gave {alt.shape} tokens vs {item['wide'].shape}")
                        item["wide_alt"].append(alt)
                        item["alt_shifts"].append(round(shift, 2))
                agree = float(np.mean([(it["wide_alt"][0] == it["wide"]).float().mean() for it in train_items]))
                meta["perturbation"] = {"p": args.perturb_content, "semitones": [lo_st, hi_st],
                                        "copies": args.perturb_copies,
                                        "token_agreement_first_copy": round(agree, 3)}
                print(f"perturbed copies: content tokens agree with the original on {agree:.0%} of frames")
        for module in feature_modules:
            module.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"precomputed content tokens + style vectors in {time.time() - t0:.1f} s")
        for module in modules:
            module.to(device)

    # ----------------------------------------------------------------- losses
    def pad_mels(batch):
        mel_lens = torch.tensor([it["mel"].size(1) for it in batch], dtype=torch.long)
        mels = torch.full((len(batch), batch[0]["mel"].size(0), int(mel_lens.max())), -10.0)
        for i, it in enumerate(batch):
            mels[i, :, : it["mel"].size(1)] = it["mel"]
        return mels.to(device), mel_lens.to(device)

    def cfm_loss(batch, perturb: bool):
        mels, mel_lens = pad_mels(batch)
        prompt_len = (torch.rand([len(batch)], device=device) * (mel_lens - 1)).floor().to(dtype=torch.long)
        prompt_len[torch.rand([len(batch)], device=device) < 0.1] = 0          # as VoiceConversionWrapper.forward_cfm
        conds = []
        for i, it in enumerate(batch):
            ylens = mel_lens[i: i + 1]
            cond = wrapper.cfm_length_regulator(it["wide"][None].to(device), ylens=ylens)[0][0]
            if perturb and it.get("wide_alt") and float(torch.rand(())) < args.perturb_content:
                alt_tokens = it["wide_alt"][int(torch.randint(len(it["wide_alt"]), ()))]
                alt = wrapper.cfm_length_regulator(alt_tokens[None].to(device), ylens=ylens)[0][0]
                cut = int(prompt_len[i])   # prompt region keeps the original tokens (upstream V1 train.py)
                cond = torch.cat([cond[:cut], alt[cut:]], dim=0)
            conds.append(cond)
        cond = pad_sequence(conds, batch_first=True)
        style = torch.cat([it["style"] for it in batch]).to(device)
        return wrapper.cfm(mels, mel_lens, prompt_len, cond, style)

    def ar_loss(batch):
        reduced = [wrapper.duration_reduction_func(it["narrow"].to(device), 1)[0] for it in batch]
        red_lens = torch.tensor([len(r) for r in reduced], dtype=torch.long, device=device)
        cond, _ = wrapper.ar_length_regulator(pad_sequence(reduced, batch_first=True, padding_value=0))
        wide = pad_sequence([it["wide"].to(device) for it in batch], batch_first=True, padding_value=0)
        wide_lens = torch.tensor([len(it["wide"]) for it in batch], dtype=torch.long, device=device)
        return wrapper.ar(cond, red_lens, wide, wide_lens)

    def upstream_losses(batch):
        mels, mel_lens = pad_mels(batch)
        waves = pad_sequence([torch.from_numpy(it["wave"]) for it in batch], batch_first=True)
        wave_lens = torch.tensor([len(it["wave"]) for it in batch], dtype=torch.long)
        waves_16k = torchaudio.functional.resample(waves, C.MODEL_SR, C.FEATURE_SR)
        lens_16k = (wave_lens.float() * C.FEATURE_SR / C.MODEL_SR).long()
        return wrapper(waves_16k.to(device), mels, lens_16k.to(device), mel_lens,
                       forward_ar=args.train_ar, forward_cfm=args.train_cfm)

    zero = torch.zeros((), device=device)

    def losses(batch, perturb: bool = True):
        if args.upstream_train_mode:
            loss_ar, loss_cfm = upstream_losses(batch)
            return loss_cfm, loss_ar
        return (cfm_loss(batch, perturb) if args.train_cfm else zero), (ar_loss(batch) if args.train_ar else zero)

    def set_mode(training: bool):
        wrapper.eval()
        if training:
            if args.upstream_train_mode:
                wrapper.train()
                wrapper.vocoder.eval()
            else:
                for module in modules:
                    module.train()

    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.mixed_precision == "bf16")

    # -------------------------------------------------------------------- EMA
    ema_pairs = []   # (live tensor, EMA tensor), parameters and buffers of the trained modules
    if args.ema_decay > 0:
        for module in modules:
            for tensor in module.state_dict(keep_vars=True).values():
                ema_pairs.append((tensor, tensor.detach().clone()))
    ema_float = [(live, avg) for live, avg in ema_pairs if avg.dtype.is_floating_point]
    ema_other = [(live, avg) for live, avg in ema_pairs if not avg.dtype.is_floating_point]

    @torch.no_grad()
    def ema_update():
        if not ema_pairs:
            return
        averages = [avg for _, avg in ema_float]
        torch._foreach_mul_(averages, args.ema_decay)
        torch._foreach_add_(averages, [live.detach() for live, _ in ema_float], alpha=1.0 - args.ema_decay)
        for live, avg in ema_other:
            avg.copy_(live)

    @torch.no_grad()
    def swap_ema():
        """Exchange live and EMA values in place (call twice to restore)."""
        for live, avg in ema_pairs:
            held = live.detach().clone()
            live.data.copy_(avg)
            avg.copy_(held)

    # -------------------------------------------------------------- preflight
    optimizer_state_bytes = 2 * sum(p.numel() * p.element_size() for p in params)   # AdamW exp_avg + exp_avg_sq
    batch_size = args.batch_size
    if device.type == "cuda" and not args.skip_preflight:
        tried = []
        while True:
            longest = sorted(train_items, key=lambda it: it["mel"].size(1))[-batch_size:]
            set_mode(True)
            torch.cuda.reset_peak_memory_stats(device)
            estimate_gb = peak = loss = loss_cfm = loss_ar = None
            try:
                with amp:
                    loss_cfm, loss_ar = losses(longest)
                    loss = loss_cfm + loss_ar
                loss.backward()
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                estimate_gb = (peak + optimizer_state_bytes) / 2 ** 30
            except torch.cuda.OutOfMemoryError:
                pass
            loss = loss_cfm = loss_ar = None   # drop any graph an out-of-memory backward left behind
            for p in params:
                p.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            fits = estimate_gb is not None and estimate_gb <= args.max_vram_gb * 0.97
            tried.append({"batch_size": batch_size, "frames": longest[-1]["mel"].size(1),
                          "peak_allocated_gb": None if peak is None else round(peak / 2 ** 30, 2),
                          "estimated_train_peak_gb": None if estimate_gb is None else round(estimate_gb, 2),
                          "fits": fits})
            print(f"preflight batch {batch_size}: " + (f"worst batch peak {peak / 2 ** 30:.2f} GB, with AdamW state "
                  f"~{estimate_gb:.2f} GB" if estimate_gb is not None else "out of memory") +
                  f" (cap {args.max_vram_gb} GB)", flush=True)
            if fits:
                break
            if not args.auto_batch or batch_size <= 2:
                meta["preflight"] = {"tried": tried, "cap_gb": args.max_vram_gb}
                return fail("failed_preflight", f"the {batch_size} longest items do not fit under --max-vram-gb "
                            f"{args.max_vram_gb} ({tried[-1]}); lower --batch-size or raise the cap if Painter allows")
            batch_size -= 1
        meta["preflight"] = {"tried": tried, "cap_gb": args.max_vram_gb, "batch_size": batch_size,
                             "estimated_train_peak_gb": tried[-1]["estimated_train_peak_gb"]}
        C.write_json(run_dir / "run_meta.json", meta)
        seed_all(args.seed)  # the probe consumed random numbers; keep runs reproducible
    meta["batch_size"] = batch_size
    if len(train_items) < batch_size:
        return fail("failed_data", "fewer train items than the batch size")

    val_batches = [val_items[i: i + batch_size] for i in range(0, len(val_items), batch_size)]
    if val_batches and len(val_batches[-1]) == 1:
        if len(val_batches) > 1:
            val_batches[-2].extend(val_batches.pop())
        else:
            val_batches[-1] = val_batches[-1] * 2

    @torch.no_grad()
    def validate() -> dict:
        if not val_batches:
            return {}
        set_mode(False)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        total_cfm = total_ar = count = 0.0
        for repeat in range(args.val_repeats):
            torch.manual_seed(args.seed + 1000 + repeat)
            for batch in val_batches:
                with amp:
                    loss_cfm, loss_ar = losses(batch, perturb=False)
                total_cfm += float(loss_cfm) * len(batch)
                total_ar += float(loss_ar) * len(batch)
                count += len(batch)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        set_mode(True)
        out = {}
        if args.train_cfm:
            out["val_loss_cfm"] = round(total_cfm / count, 5)
        if args.train_ar:
            out["val_loss_ar"] = round(total_ar / count, 5)
        return out

    def validate_ema() -> dict:
        if not ema_pairs or not val_batches:
            return {}
        swap_ema()
        try:
            return {f"{k}_ema": v for k, v in validate().items()}
        finally:
            swap_ema()

    # --------------------------------------------------------------- checkpoints
    def cpu_state(module):
        return {k: v.detach().to("cpu").clone() for k, v in module.state_dict().items()}

    def save(step: int, epoch: int) -> list[str]:
        ckpt_meta = {"run": run_name, "step": step, "epoch": epoch, "audio": audio, "train_mode": mode,
                     "batch_size": batch_size, "lr": args.lr, "seedvc_commit": meta["seedvc_commit"],
                     "base_cfm": weights["cfm"], "base_ar": weights["ar"], "training_data": str(frozen),
                     "perturb_content": args.perturb_content,
                     "speaker_exclusions_sha256": prep["config"]["speaker_exclusions_sha256"]}
        paths = []
        for ema in ([False, True] if ema_pairs else [False]):
            if ema:
                swap_ema()
            try:
                for kind, net, reg in trainable:
                    key = "cfm" if kind == "CFM" else "ar"
                    prefix = f"EMA-{kind}" if ema else kind
                    path = run_dir / f"{prefix}_epoch_{epoch:05d}_step_{step:05d}.pth"
                    if path.exists():
                        raise RuntimeError(f"refusing to overwrite {path}")
                    extra = {"ema_decay": args.ema_decay, "pretrained_weight_left": round(args.ema_decay ** step, 4)} \
                        if ema else {}
                    torch.save({"net": {key: cpu_state(net), "length_regulator": cpu_state(reg)}, "iters": step,
                                "epoch": epoch, "meta": {**ckpt_meta, "ema": ema, **extra}}, str(path))
                    paths.append(str(path))
            finally:
                if ema:
                    swap_ema()
        return paths

    def log(path: Path, record: dict):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ train
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    floor = min(args.min_lr, args.lr)

    def lr_at(step: int) -> float:
        lr = max(floor, args.lr * args.gamma ** step)
        return lr * min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps else lr

    val0 = validate()
    print(f"step 0 (pretrained) validation: {val0}")
    val_history = [{"step": 0, "epoch": 0, **val0, "checkpoints": []}]
    log(run_dir / "val_log.jsonl", val_history[-1])

    set_mode(True)
    generator = torch.Generator().manual_seed(args.seed)
    steps_per_epoch = len(train_items) // batch_size
    step, epoch, t_last, step_last = 0, 0, time.time(), 0
    try:
        while step < args.max_steps:
            order = torch.randperm(len(train_items), generator=generator).tolist()
            for start in range(0, steps_per_epoch * batch_size, batch_size):
                batch = [train_items[i] for i in order[start: start + batch_size]]
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                with amp:
                    loss_cfm, loss_ar = losses(batch)
                    loss = loss_cfm + loss_ar
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
                optimizer.step()
                ema_update()
                step += 1
                if step % args.log_every == 0 or step == 1:
                    now = time.time()
                    record = {"step": step, "epoch": epoch, "loss": round(float(loss.detach()), 5),
                              "loss_cfm": round(float(loss_cfm.detach()), 5), "loss_ar": round(float(loss_ar.detach()), 5),
                              "grad_norm": round(grad_norm, 3), "lr": lr_at(step - 1),
                              "sec_per_step": round((now - t_last) / (step - step_last), 3)}
                    if device.type == "cuda":
                        record["max_alloc_gb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
                    t_last, step_last = now, step
                    log(run_dir / "train_log.jsonl", record)
                    print(json.dumps(record), flush=True)
                if step % args.save_every == 0 or step in save_at or step == args.max_steps:
                    paths = save(step, epoch)
                    val = {**validate(), **validate_ema()}
                    val_history.append({"step": step, "epoch": epoch, **val, "checkpoints": paths})
                    log(run_dir / "val_log.jsonl", val_history[-1])
                    print(f"saved step {step}: {val}", flush=True)
                if step >= args.max_steps:
                    break
            epoch += 1
    except torch.cuda.OutOfMemoryError as error:
        meta["validation"] = val_history
        return fail("failed_oom", f"out of memory at step {step} under the {args.max_vram_gb} GB cap: {error}")
    except Exception as error:  # record the failure so convert.py refuses the partial run
        meta["validation"] = val_history
        fail("failed_error", f"{type(error).__name__} at step {step}: {error}")
        raise

    key = "val_loss_cfm" if args.train_cfm else "val_loss_ar"
    best_val = {}
    for suffix, prefix in (("", args.train_cfm and "CFM_" or "AR_"), ("_ema", args.train_cfm and "EMA-CFM_" or "EMA-AR_")):
        scored = [v for v in val_history if v["step"] > 0 and key + suffix in v]
        if scored:
            best = min(scored, key=lambda v: v[key + suffix])
            best_val["ema" if suffix else "raw"] = {
                "step": best["step"], key + suffix: best[key + suffix],
                "checkpoints": [p for p in best["checkpoints"] if Path(p).name.startswith(prefix)]}
    meta.update({
        "status": "complete", "seconds": round(time.time() - started, 1), "steps": step, "epochs": epoch,
        "steps_per_epoch": steps_per_epoch, "train_items": len(train_items), "val_items": len(val_items),
        "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None,
        "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 2) if device.type == "cuda" else None,
        "validation": val_history,
        "best_val": best_val.get("raw"), "best_val_ema": best_val.get("ema"),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })
    C.write_json(run_dir / "run_meta.json", meta)
    print("validation:", [(v["step"], v.get(key), v.get(key + "_ema")) for v in val_history])
    print(f"done in {meta['seconds']} s; best val step raw {best_val.get('raw', {}).get('step')}, "
          f"EMA {best_val.get('ema', {}).get('step')}; run_meta.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
