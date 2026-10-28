"""CPU-only validation of a voice-lab character's VoxCPM2 LoRA v2 config, manifests and training patches.

  %LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe check_config.py ^
      --config C:\\Users\\NewAdmin\\Documents\\KotorMods\\Voices\\BaoDurLab\\voxcpm\\manifests\\original\\train_lora_v2.yaml
  (Bao-Dur; for another character set VOICE_LAB_CHARACTER, e.g. jolee, and pass its
  <lab>\\voxcpm\\manifests\\<dataset>\\train_lora_v2.yaml; see common.py)

CUDA is hidden inside this process. Nothing is trained and the 2.3B model is never loaded. Checks:
 1. the YAML binds to the official train() signature (no unknown keys, types match the defaults);
 2. train_lora_v2.preflight: current snapshot, encoder/decoder rates, LoRAConfig, schedule sanity,
    output paths, fresh save_path, manifests written by the CURRENT build_manifest.py (builder sha256,
    schema, file hashes), original audio only, no held-out test id anywhere in the manifests;
 3. the official voxcpm.training.validate.validate_manifest on train and val (16 kHz, files, text;
    a config without val_manifest, i.e. the --with-val production dataset, has only train);
 4. the official load_audio_text_datasets + the trainer's tokenizer + compute_sample_lengths (nothing
    dropped by max_batch_tokens; our length arithmetic equals the official one);
 5. the official HFVoxCPMDataset/collate_fn/BatchProcessor packer with the REAL AudioVAE on CPU for rows
    with and without ref_audio ([103 ref 104][text 101][audio 102], loss only on target audio);
 6. a tiny random VoxCPM2Model fed that real packed batch under CPU bf16 autocast: official fp32-frozen
    path vs the wrapper's bf16-frozen path (same loss/grad direction) vs bf16 + layer checkpointing
    (identical), gradients only on LoRA params, one AdamW step moves them, and the official validation
    preview path (eval + VAE attached + generate() under autocast) runs on the bf16-frozen/fp32-LoRA mix;
 6b. the wrapper's patched from_local (model BUILT under default dtype bf16, to halve the host-RAM peak)
    against the official from_local + the post-load cast, on a tiny model written to disk in the
    snapshot's format (bf16 model.safetensors, fp32 audiovae.pth, real tokenizer): every parameter and
    buffer has the same dtype and value except lora_A (a different random draw under default dtype bf16,
    harmless because lora_B starts at zero), the default dtype is restored, and once both share one LoRA
    init the training loss and LoRA gradients on the real packed batch are identical;
 7. memory estimates for the official settings and for the wrapper's settings;
 8. the wrapper's preview throttle and deterministic validation against the real official module
    (train() resolves validate, validate() resolves generate_sample_audio as module globals; repeated
    validations draw identical noise/shuffles and leave the training RNG untouched) and the
    speaker-screen policy (no suspect ref_audio).
The model tests are skipped with a failure unless enough Windows commit is free for this process to
reach its peak (MODEL_TEST_COMMIT_GB 5.5; measured 5.1 GB) with 1.5 GB to spare for other apps, i.e. about
4.2 GB free beyond the ~2.8 GB the process holds after its imports (shared machine). Writes check_config_report.json next to the config with the sha256
of the config, the manifests and train_lora_v2.py/check_config.py/common.py; train_lora_v2.py refuses a
real run unless that report passed WITH the model tests for exactly those files. Exits non-zero on any
failure.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing the official trainer must not write into Vox's engine tree

import argparse  # noqa: E402
import copy  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# "-1", not "": on Windows an empty value deletes the variable and every GPU stays visible.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import train_lora_v2 as W  # noqa: E402

MODEL_TEST_COMMIT_GB = 5.5  # measured peak 5.12 GB (2026-09-30 review run); the report records each run's peak
TINY_VAE = {"encoder_dim": 8, "encoder_rates": [2, 5, 8, 8], "latent_dim": 64, "decoder_dim": 64,
            "decoder_rates": [8, 6, 5, 2, 2, 2], "sr_bin_boundaries": [20000, 30000, 40000],
            "sample_rate": 16000, "out_sample_rate": 48000}


class Failure(Exception):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-model-tests", action="store_true", help="skip steps 5-6 (AudioVAE + tiny model)")
    parser.add_argument("--allow-restored-audio", action="store_true",
                        help="check a restored-audio config (against the user's 2026-09-30 decision)")
    args = parser.parse_args()

    W.use_official_source()
    import torch

    torch.set_num_threads(args.threads)
    expect(not torch.cuda.is_available(), "CUDA must be hidden for this check")
    report: dict = {"config": str(args.config), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_tests": not args.skip_model_tests, "checks": {}}
    checks = report["checks"]
    try:
        # 1. YAML vs official signature ----------------------------------------------------------------
        from voxcpm.training.config import load_yaml_config

        report["config_sha256"] = C.sha256_file(args.config)
        report["code_sha256"] = W.code_hashes()
        cfg = load_yaml_config(args.config)
        files = W.manifest_files(cfg)
        report["manifest_sha256"] = {name: C.sha256_file(p) for name, p in files.items() if p.exists()}
        trainer = W.load_official_trainer()
        signature = W.train_signature(trainer)
        unknown = sorted(set(cfg) - set(signature.parameters))
        expect(not unknown, f"keys not accepted by official train(): {unknown}")
        bound = signature.bind(**cfg)
        bound.apply_defaults()
        type_problems = []
        for name, param in signature.parameters.items():
            if name not in cfg or param.default in (inspect.Parameter.empty, None):
                continue
            want, got = type(param.default), cfg[name]
            ok = isinstance(got, want) or (want is float and isinstance(got, int) and not isinstance(got, bool))
            if want is int and isinstance(got, bool):
                ok = False
            if not ok:
                type_problems.append(f"{name}: {type(got).__name__} (default is {want.__name__})")
        expect(not type_problems, f"type mismatches: {type_problems}")
        expect(set(cfg["lambdas"]) == {"loss/diff", "loss/stop"}, "lambdas must weight loss/diff and loss/stop")
        checks["signature"] = {"ok": True, "params": list(signature.parameters), "yaml_keys": sorted(cfg)}

        # 8a. preview throttle: validate() must look generate_sample_audio up as a global ------------------
        import types

        expect("generate_sample_audio" in trainer.validate.__code__.co_names,
               "official validate() no longer calls generate_sample_audio as a module global")
        calls: list[int] = []
        fake = types.SimpleNamespace(
            generate_sample_audio=lambda model, val_ds, audio_vae, writer, step, *a, **k: calls.append(step))
        n_iters, v_every = int(cfg["num_iters"]), int(cfg["valid_interval"])
        W.throttle_previews(fake, 200, n_iters)
        for s in sorted({s for s in range(n_iters) if s % v_every == 0} | {n_iters - 1}):
            fake.generate_sample_audio(None, None, None, None, s, None, 16000, out_sample_rate=48000)
        want = sorted({s for s in range(n_iters) if s % 200 == 0} | {n_iters - 1})
        expect(calls == want, f"throttled previews at {calls}, expected {want}")
        calls.clear()
        W.throttle_previews(fake, 0, n_iters)  # re-throttling wraps the original, 0 disables
        fake.generate_sample_audio(None, None, None, None, 0)
        expect(not calls, "--preview-interval 0 must disable previews")
        checks["preview_throttle"] = {"ok": True, "previews_at": want,
                                      "checkpoints": W.expected_checkpoints(cfg)}
        checks["deterministic_validation"] = deterministic_validation_test(torch, trainer)

        # 2. preflight ---------------------------------------------------------------------------------------
        notes = W.preflight(cfg, resume=False, allow_restored=args.allow_restored_audio)
        checks["preflight"] = {"ok": True, "notes": notes}

        # 3. official manifest validator ------------------------------------------------------------------
        from voxcpm.training.validate import validate_manifest

        screen = C.load_speaker_screen()
        rows = {}
        for key in ("train_manifest", "val_manifest"):
            if not cfg.get(key):
                checks[key] = {"ok": True, "skipped": "no val manifest (train + val production dataset)"}
                continue
            result = validate_manifest(cfg[key], sample_rate=int(cfg["sample_rate"]))
            expect(result.is_valid, f"{key}: {result.errors[:5]}")
            rows[key] = C.read_jsonl(Path(cfg[key]))
            ids = {r["id"] for r in rows[key]} | {r["ref_id"] for r in rows[key] if r.get("ref_id")}
            expect(not ids & (set(screen["confirmed"]) | screen["uncertain"]),
                   f"{key}: screened clips present {sorted(ids & (set(screen['confirmed']) | screen['uncertain']))}")
            suspect_refs = sorted({r["ref_id"] for r in rows[key] if r.get("ref_id") in screen["suspect"]})
            expect(not suspect_refs, f"{key}: 'suspect' clips used as ref_audio {suspect_refs}")
            drift = [r["id"] for r in rows[key]
                     if abs(float(r["duration"]) - C.read_mono(Path(r["audio"]))[0].size / C.ENCODER_SR) > 0.011]
            expect(not drift, f"{key}: duration field disagrees with the audio for {drift[:5]}")
            checks[key] = {"ok": True, "valid": result.valid_samples, "with_ref_audio": result.has_ref_audio,
                           "minutes": round(sum(result.audio_durations) / 60, 2), "warnings": result.warnings[:10]}

        # 4. official HF loading + tokenizer + length filter ------------------------------------------------
        from voxcpm.training.data import compute_sample_lengths, load_audio_text_datasets

        train_ds, val_ds = load_audio_text_datasets(train_manifest=cfg["train_manifest"],
                                                    val_manifest=cfg.get("val_manifest") or "",
                                                    sample_rate=int(cfg["sample_rate"]))
        expect((val_ds is None) == (not cfg.get("val_manifest")), "val dataset presence differs from the config")
        for name in ("audio", "text", "ref_audio", "dataset_id", "duration", "ref_duration"):
            expect(name in train_ds.column_names, f"train dataset lacks column {name}")
        with_ref = [i for i, r in enumerate(rows["train_manifest"]) if r.get("ref_audio")]
        without = [i for i, r in enumerate(rows["train_manifest"]) if not r.get("ref_audio")]
        sample_ref, sample_plain = train_ds[with_ref[0]], train_ds[without[0]]
        expect(sample_ref["ref_audio"] is not None and sample_ref["ref_audio"]["sampling_rate"] == 16000,
               "ref_audio did not decode at 16 kHz")
        expect(sample_plain["ref_audio"] is None, "rows without a ref must decode ref_audio as None")
        tokenizer = C.load_tokenizer()

        def tokenize(batch):
            return {"text_ids": [tokenizer(t) for t in batch["text"]]}

        train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        official = compute_sample_lengths(train_tok, audio_vae_fps=25, patch_size=4)
        mine = C.sequence_lengths(rows["train_manifest"], tokenizer)
        expect(official == mine, "sequence length arithmetic differs from the official compute_sample_lengths")
        cap = int(cfg["max_batch_tokens"]) // int(cfg["batch_size"])
        dropped = sum(1 for n in official if n > cap)
        expect(dropped == 0, f"max_batch_tokens would drop {dropped} samples")
        checks["hf_loading"] = {"ok": True, "train_rows": len(train_ds), "val_rows": len(val_ds) if val_ds else 0,
                                "ref_rows": len(with_ref), "max_positions": max(official),
                                "median_positions": sorted(official)[len(official) // 2],
                                "per_sample_cap": cap, "dropped": dropped}

        # 7. memory estimates (cheap, before the heavier model tests) ---------------------------------------
        checks["memory_estimates"] = {
            "official_as_is_fp32_frozen": C.estimate_train_memory(cfg, official, "fp32", "none"),
            "official_batch1": C.estimate_train_memory({**cfg, "batch_size": 1}, official, "fp32", "none"),
            "wrapper_bf16_no_ckpt": C.estimate_train_memory(cfg, official, "bf16", "none"),
            "wrapper_bf16_ckpt_dit": C.estimate_train_memory(cfg, official, "bf16", "dit"),
            "wrapper_default_bf16_ckpt_all": C.estimate_train_memory(cfg, official, "bf16", "all"),
            "render_inference_gb": C.RENDER_NEED_GB,
        }

        if not args.skip_model_tests:
            try:  # this machine runs near its commit limit: do not push it over with a CPU check
                checks["host_memory_gate"] = C.host_ram_budget(MODEL_TEST_COMMIT_GB, 1.5, "check_config model tests")
            except SystemExit as exc:
                raise Failure(f"deferred, too little free commit for the model tests: {exc}") from None
            # 5. packer smoke with the real AudioVAE -------------------------------------------------------
            from voxcpm.model.voxcpm2 import VoxCPMConfig
            from voxcpm.modules.audiovae import AudioVAEV2
            from voxcpm.training.data import BatchProcessor, HFVoxCPMDataset

            model_cfg = VoxCPMConfig.model_validate_json((C.SNAPSHOT / "config.json").read_text(encoding="utf-8"))
            vae = AudioVAEV2(config=model_cfg.audio_vae_config)
            state = torch.load(str(C.SNAPSHOT / "audiovae.pth"), map_location="cpu", weights_only=True)
            missing, unexpected = vae.load_state_dict(state.get("state_dict", state), strict=False)
            expect(not missing, f"AudioVAE missing keys {missing[:5]}")
            vae = vae.float().eval()
            processor = BatchProcessor(config=model_cfg, audio_vae=vae, dataset_cnt=1, device=torch.device("cpu"))
            dataset = HFVoxCPMDataset(train_tok)
            by_len = sorted(range(len(official)), key=lambda i: official[i])
            pick = [i for i in by_len if i in set(with_ref)][:2] + [i for i in by_len if i in set(without)][:2]
            batch = HFVoxCPMDataset.collate_fn([dataset[i] for i in pick])
            packed = processor(batch)
            B, T = packed["text_tokens"].shape
            expect(tuple(packed["audio_feats"].shape) == (B, T, model_cfg.patch_size, model_cfg.feat_dim),
                   f"audio_feats shape {tuple(packed['audio_feats'].shape)}")
            per_sample = []
            for b, i in enumerate(pick):
                row = rows["train_manifest"][i]
                tokens = packed["text_tokens"][b].tolist()
                has_ref_tokens = 103 in tokens and 104 in tokens
                expect(has_ref_tokens == bool(row.get("ref_audio")), f"row {row['id']}: ref tokens {has_ref_tokens}")
                target = int(packed["loss_mask"][b].sum())
                expected = math.ceil(row["duration"] * C.ENCODER_SR / 2560)
                expect(abs(target - expected) <= 1, f"row {row['id']}: {target} loss patches, expected ~{expected}")
                length = int((packed["position_ids"][b] > 0).sum()) + 1
                expect(abs(length - official[i]) <= 2, f"row {row['id']}: packed {length} vs estimated {official[i]}")
                per_sample.append({"id": row["id"], "ref": row.get("ref_id"), "packed_len": length,
                                   "loss_patches": target, "stop_labels": int(packed["labels"][b].sum())})
            checks["packer_real_audiovae"] = {"ok": True, "batch_shape": [B, T], "samples": per_sample}

            # 6. tiny model: official fp32-frozen vs bf16-frozen vs bf16 + checkpointing ------------------
            checks["tiny_model_patch_test"] = tiny_model_test(torch, vae, packed)
            # 6b. patched from_local (bf16 build) vs official from_local + cast, from disk -----------------
            checks["from_local_bf16_build_test"] = from_local_test(torch, packed)
        report["ok"] = True
    except Failure as exc:
        report["ok"] = False
        report["failure"] = str(exc)
    except SystemExit as exc:  # preflight problems
        report["ok"] = False
        report["failure"] = str(exc)
    out = args.config.parent / "check_config_report.json"
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["process_commit"] = C.process_commit()
    C.atomic_write_json(out, report)
    print(json.dumps(report, indent=2)[:6000])
    print(f"report -> {out}")
    sys.exit(0 if report.get("ok") else 1)


def tiny_config(vae_config: dict | None = None) -> dict:
    """The real VoxCPM2 config shrunk to a few thousand parameters (same architecture and feature layout)."""
    real = json.loads((C.SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    lm = dict(real["lm_config"])
    lm.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
              num_hidden_layers=2, kv_channels=16)
    rope = dict(lm["rope_scaling"])
    rope["long_factor"], rope["short_factor"] = rope["long_factor"][:8], rope["short_factor"][:8]
    lm["rope_scaling"] = rope
    tiny = dict(real)
    tiny.update(
        lm_config=lm, residual_lm_num_layers=1, scalar_quantization_latent_dim=16, max_length=512, device="cpu",
        encoder_config={"hidden_dim": 32, "ffn_dim": 64, "num_heads": 2, "num_layers": 1, "kv_channels": 16},
        dit_config={"hidden_dim": 32, "ffn_dim": 64, "num_heads": 2, "num_layers": 2, "kv_channels": 16,
                    "cfm_config": real["dit_config"]["cfm_config"]},
    )
    if vae_config is not None:
        tiny["audio_vae_config"] = vae_config
    return tiny


def train_step(torch, model, packed) -> tuple[float, dict]:
    """One official-style training forward/backward (CPU bf16 autocast) -> loss, LoRA grads."""
    torch.manual_seed(1234)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(packed["text_tokens"], packed["text_mask"], packed["audio_feats"], packed["audio_mask"],
                    packed["loss_mask"], packed["position_ids"], packed["labels"], progress=0.1)
    loss = out["loss/diff"] + out["loss/stop"]
    loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def deterministic_validation_test(torch, trainer) -> dict:
    """train() must call validate as a module global, and the wrapper must make every validation draw the
    same numbers (torch/Python/NumPy RNGs, a shuffled DataLoader) while leaving the training RNG as it was."""
    import random
    import types

    import numpy as np

    expect("validate" in W.train_globals_used(trainer), "official train() no longer calls validate as a module global")
    draws: list = []

    def fake_validate(*_args, **_kwargs):
        loader = torch.utils.data.DataLoader(list(range(16)), batch_size=2, shuffle=True)
        draws.append((torch.rand(4).tolist(), torch.randn(2).tolist(), random.random(), float(np.random.rand()),
                      [int(x) for batch in loader for x in batch]))

    fake = types.SimpleNamespace(validate=fake_validate)
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    torch_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    W.deterministic_validation(fake, W.DEFAULT_VAL_SEED)
    fake.validate()
    fake.validate()
    expect(draws[0] == draws[1], "two deterministic validations drew different numbers")
    expect(torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == py_state
           and np.array_equal(np.random.get_state()[1], np_state[1]), "deterministic validation moved the training RNG")
    W.deterministic_validation(fake, W.DEFAULT_VAL_SEED)  # re-wrapping wraps the original, not the wrapper
    fake.validate()
    expect(draws[2] == draws[0], "re-wrapped validation differs")
    W.deterministic_validation(fake, -1)  # official behaviour back
    expect(not getattr(fake.validate, "_lab_deterministic", False), "--val-seed -1 must restore the original")
    fake.validate()
    fake.validate()
    expect(draws[3] != draws[4], "control: unseeded validations should differ")
    return {"ok": True, "seed": W.DEFAULT_VAL_SEED, "shuffled_order": draws[0][4][:8]}


def from_local_test(torch, packed) -> dict:
    """Patched from_local (bf16 build) vs official from_local + post-load cast, loaded from a tiny on-disk
    snapshot: same dtypes and values everywhere, default dtype restored, identical loss and LoRA grads."""
    from safetensors.torch import save_file
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    tiny = tiny_config(TINY_VAE)
    config = VoxCPMConfig.model_validate(tiny)
    tmp = Path(tempfile.mkdtemp(prefix="tiny_snapshot_", dir=str(C.DATA_ROOT)))
    try:
        (tmp / "config.json").write_text(json.dumps(tiny), encoding="utf-8")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenization_voxcpm2.py"):
            if (C.SNAPSHOT / name).exists():
                shutil.copy2(C.SNAPSHOT / name, tmp / name)
        torch.manual_seed(0)
        source = VoxCPM2Model(config, LlamaTokenizerFast.from_pretrained(str(C.SNAPSHOT)),
                              AudioVAEV2(config=config.audio_vae_config), None, device="cpu")
        for param in source.parameters():  # non-trivial values everywhere (norm weights too)
            param.data.add_(torch.randn_like(param) * 0.02)
        weights = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in source.state_dict().items()
                   if not k.startswith("audio_vae.") and v.is_floating_point()}
        save_file(weights, str(tmp / "model.safetensors"))  # the real checkpoint is bf16 too
        torch.save({"state_dict": source.audio_vae.state_dict()}, str(tmp / "audiovae.pth"))  # fp32 like the real one
        del source
        lora = LoRAConfig(enable_lm=True, enable_dit=True, r=4, alpha=4)
        W.install_patches("bf16", "none")
        original = W._PATCHED["original_from_local"]
        torch.manual_seed(1)
        official = original(VoxCPM2Model, str(tmp), optimize=False, training=True, lora_config=lora)
        expect(torch.get_default_dtype() == torch.float32, "default dtype changed by the official load")
        W.prepare_for_training(official, "bf16", "none")  # the previous wrapper path (checked against fp32 in 6)
        torch.manual_seed(1)
        patched = VoxCPM2Model.from_local(str(tmp), optimize=False, training=True, lora_config=lora)
        expect(torch.get_default_dtype() == torch.float32, "patched from_local did not restore the default dtype")
        stats = dict(W._PATCHED["stats"])
        expect(stats.get("built_with_default_dtype") == "bfloat16", f"model was not built in bf16: {stats}")
        mismatched, lora_a_draw_diff = [], []
        p_off, p_pat = dict(official.named_parameters()), dict(patched.named_parameters())
        expect(set(p_off) == set(p_pat), "parameter names differ")
        for name, a in p_off.items():
            b = p_pat[name]
            if a.dtype != b.dtype or a.requires_grad != b.requires_grad:
                mismatched.append(f"{name}: {a.dtype}/{a.requires_grad} vs {b.dtype}/{b.requires_grad}")
            elif name.endswith("lora_A"):  # random init drawn under default dtype fp32 vs bf16: a different draw
                lora_a_draw_diff.append(float((a.detach() - b.detach()).abs().max()))
            elif not torch.equal(a, b):
                mismatched.append(f"{name}: values differ")
        b_off, b_pat = dict(official.named_buffers()), dict(patched.named_buffers())
        expect(set(b_off) == set(b_pat), "buffer names differ")
        for name, a in b_off.items():
            b = b_pat[name]
            if a.dtype != b.dtype or not torch.equal(a, b):
                mismatched.append(f"buffer {name}: {a.dtype} vs {b.dtype}")
        expect(not mismatched, f"bf16 build differs from official+cast: {mismatched[:6]}")
        dtypes = {
            "frozen_non_vae": sorted({str(p.dtype) for n, p in p_pat.items()
                                      if not p.requires_grad and not n.startswith("audio_vae.")}),
            "lora": sorted({str(p.dtype) for p in p_pat.values() if p.requires_grad}),
            "audio_vae": sorted({str(p.dtype) for n, p in p_pat.items() if n.startswith("audio_vae.")}),
            "buffers": sorted({f"{n.rsplit('.', 1)[-1]}:{str(b.dtype)}" for n, b in b_pat.items()}),
        }
        expect(dtypes["frozen_non_vae"] == ["torch.bfloat16"] and dtypes["lora"] == ["torch.float32"]
               and dtypes["audio_vae"] == ["torch.float32"], f"unexpected dtypes {dtypes}")
        with torch.no_grad():  # same LoRA init in both, then one training step must match exactly
            for name, param in p_pat.items():
                if name.endswith("lora_A") or name.endswith("lora_B"):
                    param.copy_(p_off[name])
                    if name.endswith("lora_B"):
                        torch.nn.init.normal_(param, std=0.02)
                        p_off[name].copy_(param)
        for model in (official, patched):
            model.audio_vae = None  # the trainer keeps the VAE only in BatchProcessor
        loss_off, grads_off = train_step(torch, official, packed)
        loss_pat, grads_pat = train_step(torch, patched, packed)
        expect(set(grads_off) == set(grads_pat) and all(n in {k for k, p in p_pat.items() if p.requires_grad}
                                                        for n in grads_pat), "gradient sets differ")
        max_diff = max(float((grads_off[n] - grads_pat[n]).abs().max()) for n in grads_off)
        expect(loss_off == loss_pat and max_diff == 0.0,
               f"training step differs: loss {loss_off} vs {loss_pat}, max grad diff {max_diff}")
        return {"ok": True, "loss": round(loss_pat, 6), "max_grad_diff": max_diff, "dtypes": dtypes,
                "lora_A_init_max_abs_diff": max(lora_a_draw_diff) if lora_a_draw_diff else None,
                "patch_stats": {k: v for k, v in stats.items() if k != "host_after_load"},
                "params": sum(p.numel() for p in p_pat.values())}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tiny_model_test(torch, vae, packed) -> dict:
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model, VoxCPMConfig

    config = VoxCPMConfig.model_validate(tiny_config())
    torch.manual_seed(0)
    base = VoxCPM2Model(config, LlamaTokenizerFast.from_pretrained(str(C.SNAPSHOT)), vae,
                        LoRAConfig(enable_lm=True, enable_dit=True, r=4, alpha=4), device="cpu")
    for name, param in base.named_parameters():  # what from_local(training=True) does
        param.requires_grad = "lora" in name and not name.startswith("audio_vae.")
        if name.endswith("lora_B"):
            torch.nn.init.normal_(param, std=0.02)  # so lora_A also receives gradient in this test
        elif not param.requires_grad and not name.startswith("audio_vae."):
            param.data = param.data.to(torch.bfloat16).to(torch.float32)  # identical values in both storages
    del base.audio_vae  # the trainer keeps the VAE only in BatchProcessor
    lora_names = {n for n, p in base.named_parameters() if p.requires_grad}
    W.install_patches("bf16", "all")

    def run(model):
        torch.manual_seed(1234)
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(packed["text_tokens"], packed["text_mask"], packed["audio_feats"], packed["audio_mask"],
                        packed["loss_mask"], packed["position_ids"], packed["labels"], progress=0.1)
        loss = out["loss/diff"] + out["loss/stop"]
        loss.backward()
        grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
        return float(loss.detach()), grads

    official_model = copy.deepcopy(base)
    W.prepare_for_training(official_model, "fp32", "none")
    bf16_model = copy.deepcopy(base)
    stats_bf16 = W.prepare_for_training(bf16_model, "bf16", "none")
    ckpt_model = copy.deepcopy(base)
    stats_ckpt = W.prepare_for_training(ckpt_model, "bf16", "all")

    loss_official, grads_official = run(official_model)
    loss_bf16, grads_bf16 = run(bf16_model)
    loss_ckpt, grads_ckpt = run(ckpt_model)

    def flat(grads):
        return torch.cat([grads[n].reshape(-1) for n in sorted(grads)])

    cos = float(torch.nn.functional.cosine_similarity(flat(grads_official), flat(grads_bf16), dim=0))
    ckpt_diff = float((flat(grads_ckpt) - flat(grads_bf16)).abs().max())
    frozen_dtypes = sorted({str(p.dtype) for n, p in bf16_model.named_parameters() if not p.requires_grad})
    lora_dtypes = sorted({str(p.dtype) for n, p in bf16_model.named_parameters() if p.requires_grad})
    expect(set(grads_bf16) == lora_names, "gradients must reach exactly the LoRA parameters")
    expect(all(torch.isfinite(g).all() for g in grads_bf16.values()), "non-finite LoRA gradients")
    expect(frozen_dtypes == ["torch.bfloat16"], f"frozen params not all bf16: {frozen_dtypes}")
    expect(lora_dtypes == ["torch.float32"], f"LoRA params not fp32: {lora_dtypes}")
    expect(abs(loss_bf16 - loss_official) <= 0.05 * abs(loss_official) + 1e-3,
           f"bf16-frozen loss {loss_bf16} vs official {loss_official}")
    expect(cos > 0.98, f"bf16-frozen gradient direction differs from official (cos {cos:.4f})")
    expect(abs(loss_ckpt - loss_bf16) <= 1e-6 * max(1.0, abs(loss_bf16)) and ckpt_diff <= 1e-6,
           f"checkpointing changed the result (loss {loss_ckpt} vs {loss_bf16}, max grad diff {ckpt_diff})")
    before = {n: p.detach().clone() for n, p in ckpt_model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in ckpt_model.parameters() if p.requires_grad), lr=1e-4)
    torch.nn.utils.clip_grad_norm_(ckpt_model.parameters(), max_norm=1.0)
    optimizer.step()
    moved = sum(1 for n, p in ckpt_model.named_parameters() if p.requires_grad and not torch.equal(p, before[n]))
    expect(moved == len(before), f"AdamW moved {moved}/{len(before)} LoRA tensors")

    # the official validation preview (generate_sample_audio): eval mode, fp32 VAE re-attached, generate()
    # under bf16 autocast (CUDA in the trainer, CPU here), then back to train mode with the VAE detached
    preview = {}
    greeting = f"Hey, {C.ADDRESS_TERM}." if C.ADDRESS_TERM else "Hey there."  # Bao-Dur: "Hey, General."
    for label, model in (("official_fp32_frozen", official_model), ("wrapper_bf16_frozen_ckpt", ckpt_model)):
        model.eval()
        model.audio_vae = vae.to(torch.float32)
        torch.manual_seed(7)
        try:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                wav = model.generate(target_text=f"(Troubled){greeting} Got a second?", inference_timesteps=2,
                                     cfg_value=2.0, max_len=6)
        finally:
            model.audio_vae = None
            model.train()
        wav = wav.detach().float().reshape(-1)
        expect(wav.numel() > 0 and bool(torch.isfinite(wav).all()), f"{label}: preview generate() output is bad")
        preview[label] = {"samples": int(wav.numel()), "seconds_at_48k": round(wav.numel() / 48000, 3)}
    return {"ok": True, "loss_official_fp32_frozen": round(loss_official, 6), "loss_bf16_frozen": round(loss_bf16, 6),
            "loss_bf16_ckpt": round(loss_ckpt, 6), "grad_cosine_official_vs_bf16": round(cos, 5),
            "max_grad_diff_ckpt_vs_plain": ckpt_diff, "lora_tensors": len(lora_names),
            "stats_bf16": stats_bf16, "stats_ckpt": stats_ckpt, "frozen_dtypes": frozen_dtypes,
            "lora_dtypes": lora_dtypes, "adamw_moved_tensors": moved, "preview_generate": preview}


if __name__ == "__main__":
    main()
