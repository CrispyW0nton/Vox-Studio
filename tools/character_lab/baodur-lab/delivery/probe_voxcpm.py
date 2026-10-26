"""VoxCPM2 delivery probe: which way of asking for an emotion actually works for the character?

VOICE_LAB_CHARACTER picks the character (default baodur); his map, script and probe lines come from his lab
(<lab>/delivery/, content/lines.json "probe", content/lexicon.json "text_rewrites").
GPU step for the main session (the CPU workflow only runs --dry-run). For every delivery label it renders the
same lines in these variants, conditioned exactly as the voxcpm render step does (same text cleaning, same
"Keep the trained voice stable." clause for adapters, same per-line length guard, same trimmed reference copies):
  A     controlled cloning (render.py idref_instr): text = "(<variant A instruction>)line", reference_wav_path =
        the fixed identity clip (natural rank 1)
  B     the same with the compact variant B instruction
  P     ultimate cloning (render.py ref_prompt): prompt_wav + prompt_text = the label's rank-1 reference, and
        reference_wav_path = that same clip (render.py --identity-ref same)
  P2    the same prompt, reference_wav_path = natural rank 1 (render.py --identity-ref natural)
  none  controlled cloning with no instruction, for EVERY line and seed: the paired neutral baseline
Lines per label (4): two emotionally neutral lines (two neutral QUESTIONS for questioning, which is scored on
question lines only), the label's line from takes_script.json, and one more label-appropriate probe line
(lines.json probe.extra; Bao-Dur's reflective has no script line, so two). Held-out lines are never rendered here,
so an adapter trained on held-out clips (Bao-Dur's July Vox adapter was) cannot leak into this probe.
Bao-Dur's default: 4 seeds x 4 lines x (A, B, P, P2) for 13 labels (P2 is skipped for natural: identical to P) +
none for every line: 936 takes at cfg 1.5 (the render step's default), roughly 50-100 min on the shared RTX 5090.
--cfg 1.5,2.0 sweeps both (variants x cfg); --labels splits the run into shorter GPU sessions.

Output: <lab>/delivery/probe/<run>/<take_id>.wav (48 kHz mono PCM16) and manifest.json in the shared takes
format plus a "settings" block (map/variants/script sha256, the prompt clip of each label, the identity clip);
the manifest is rewritten atomically after every take. --resume continues a run and refuses when the stored
settings differ. Score on CPU (or with GPU ASR) with: score_delivery.py --manifest <run>/manifest.json
--pick-instructions; apply after listening with score_delivery.py --apply-from <run>/voxcpm_instruction_probe.json.

Shared-machine guards (resources.py): refuses to load unless ~13 GB of Windows commit + 2 GB margin are free
(VoxCPM2 builds in fp32 on the CPU first), and unless free VRAM >= --max-vram-gb + --keep-free-gb; the job's whole
footprint stays inside --max-vram-gb because torch's allocator is capped at --max-vram-gb - 1 GB (the CUDA context
and cuBLAS/cuDNN workspaces live outside the allocator; VoxCPM2 needs ~6.2 GB of allocator memory, per the voxcpm
task's estimate). Logs peak reserved VRAM and the process's peak commit. Nothing is downloaded.

Run with the VoxCPM2 runtime python (read-only venv; this never writes bytecode, and numba's cache goes to
<lab>/delivery/.cache/numba):
  %LOCALAPPDATA%/VoxStudio/engines/voxcpm2/.venv/Scripts/python.exe probe_voxcpm.py --run instr_probe_v1
  [--lora <adapter dir>] [--labels angry,warm] [--seeds 11,23,37,53] [--cfg 1.5] [--variants A,B,P,P2]
  [--ref-source auto|corpus] [--resume] [--dry-run]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labcfg  # noqa: E402  (stdlib only)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(labcfg.NUMBA_CACHE))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

import resources  # noqa: E402

PROBE_VERSION = 2
LAB = labcfg.LAB
DELIVERY = labcfg.DELIVERY
VOXCPM_REF_CACHE = LAB / "voxcpm" / "cache"   # the voxcpm render step's trimmed reference copies (read only)
ENGINE = Path(os.environ.get("LOCALAPPDATA", "")) / "VoxStudio" / "engines" / "voxcpm2"
SNAPSHOT = ENGINE / "cache" / "models--openbmb--VoxCPM2" / "snapshots" / "32279effe8c19989596f05d353d1447f51d9e915"
OUT_RATE = 48000
BADCASE_RATIO = 6.0      # VoxCPM.generate default retry_badcase_ratio_threshold
SECONDS_PER_TAKE = (3.0, 6.5)  # rough range on the shared RTX 5090 (10 steps, ~4-6 s lines) for the estimate
VARIANTS = ("A", "B", "P", "P2")
LORA_IDENTITY = "Keep the trained voice stable."  # Vox sidecar / render.py identity clause for adapters


def probe_lines() -> tuple[dict[str, str], dict[str, str], dict[str, list[tuple[str, str]]]]:
    """The character's probe lines (content/lines.json "probe", written for the lab, not game lines, each checked
    against every corpus text and the held-out lines): neutral lines, neutral questions, per-label extra lines."""
    probe = labcfg.content("lines")["probe"]
    extra = {label: [(str(line_id), str(text)) for line_id, text in items]
             for label, items in probe.get("extra", {}).items() if not label.startswith("_")}
    return dict(probe["neutral_lines"]), dict(probe["neutral_questions"]), extra


def text_rewrites() -> list[tuple[str, str]]:
    """The voxcpm render step's clean_text rewrites for this character (content/lexicon.json; Bao-Dur: G0-T0 ->
    Goto, spelled as spoken)."""
    return [(str(pattern), str(replacement)) for pattern, replacement in labcfg.content("lexicon").get("text_rewrites", [])]


def clean_text(text: str) -> str:
    """The voxcpm render step's clean_text: quotes/dashes/ellipsis normalised, code-names spelled as spoken."""
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("—", " - ").replace("–", " - ").replace("…", "..."))
    for pattern, replacement in text_rewrites():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def compose_instruction(instruction: str, clause: bool) -> str:
    """render.py compose_instruction: identity clause first, instruction's first letter capitalised."""
    instruction = instruction.strip().strip("()").strip()
    if not clause:
        return instruction
    return f"{LORA_IDENTITY} {instruction[:1].upper()}{instruction[1:]}" if instruction else LORA_IDENTITY


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    for attempt in range(5):  # a reader (indexer, virus scanner) can hold the target for a moment on Windows
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    os.replace(temporary, path)


def reference_audio(ref: dict, audio_key: str, ref_source: str) -> tuple[str, bool]:
    """The audio file the render step would condition on: its trimmed cached copy when present (auto)."""
    if ref_source == "auto" and audio_key == "audio":
        cached = VOXCPM_REF_CACHE / "original" / f"{ref['id']}.wav"
        if cached.is_file():
            return str(cached), True
    return ref[audio_key], False


def plan(args, delivery_map: dict, variants: dict, script: dict) -> tuple[list[dict], dict]:
    deliveries = delivery_map["deliveries"]
    labels = [l for l in (args.labels.split(",") if args.labels else deliveries) if l in deliveries]
    script_lines = {}
    for line in script["lines"]:
        if line["source"] == "new" and line["delivery"] not in script_lines:
            script_lines[line["delivery"]] = (line["id"], line["text"])
    identity = deliveries["natural"]["references"][0]
    audio_key = "audio" if args.audio == "original" else "restored_audio"
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = [float(c) for c in args.cfg.split(",")]
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(wanted) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {VARIANTS}")
    clause = bool(args.lora)
    NEUTRAL_LINES, NEUTRAL_QUESTIONS, PROBE_EXTRA = probe_lines()
    jobs: list[dict] = []
    baseline_lines: dict[str, str] = {}
    prompts = {}
    for label in labels:
        lines = dict(NEUTRAL_QUESTIONS if label == "questioning" else NEUTRAL_LINES)
        if label in script_lines:
            lines[script_lines[label][0]] = script_lines[label][1]
        for line_id, text in PROBE_EXTRA.get(label, []):
            if len(lines) < 4:
                lines[line_id] = text
        baseline_lines.update(lines)
        rank1 = deliveries[label]["references"][0]
        prompts[label] = rank1["id"]
        for cfg in cfgs:
            tag = f"_c{cfg:g}" if len(cfgs) > 1 else ""
            for line_id, text in lines.items():
                for seed in seeds:
                    for variant in wanted:
                        if variant == "P2" and label == "natural":
                            continue  # natural's rank 1 is the identity clip: P2 would repeat P
                        if variant in ("A", "B"):
                            core = variants[variant][label]
                            job = {"mode": "controlled", "instruction_core": core,
                                   "instruction": compose_instruction(core, clause), "prompt": None,
                                   "reference": identity}
                        else:
                            job = {"mode": "continuation", "instruction_core": "", "instruction": "",
                                   "prompt": rank1, "reference": rank1 if variant == "P" else identity}
                        job.update(take_id=f"{label}_{variant}_{line_id}_s{seed}{tag}", delivery=label, variant=variant,
                                   line_id=line_id, text=text, seed=seed, cfg=cfg, audio_key=audio_key)
                        jobs.append(job)
    for cfg in cfgs:  # the paired neutral baseline: every line, every seed, no instruction
        tag = f"_c{cfg:g}" if len(cfgs) > 1 else ""
        for line_id, text in baseline_lines.items():
            for seed in seeds:
                jobs.append({"take_id": f"none_{line_id}_s{seed}{tag}", "delivery": "natural", "variant": "none",
                             "line_id": line_id, "text": text, "seed": seed, "cfg": cfg, "mode": "controlled",
                             "instruction_core": "", "instruction": compose_instruction("", clause) if clause else "",
                             "prompt": None, "reference": identity, "audio_key": audio_key})
    meta = {"labels": labels, "prompt_ids": prompts, "identity_id": identity["id"], "seeds": seeds, "cfgs": cfgs,
            "variants": wanted, "lines": baseline_lines}
    return jobs, meta


def load_model(args):
    import torch
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm2 import LoRAConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this probe is a GPU step (use --dry-run on the CPU).")
    # The job's whole footprint must fit --max-vram-gb: torch's allocator gets that minus the CUDA context and
    # workspaces (CUDA_OVERHEAD_GB), which live outside the allocator cap. mem_get_info runs after this process's
    # context exists, so "free" already excludes it.
    ok, gpu = resources.vram_headroom(torch, args.max_vram_gb, args.keep_free_gb, 0.0)
    if not ok:
        raise SystemExit(f"Not enough free VRAM: {gpu['free_gb']:.1f} GB free, need {gpu['required_free_gb']:.1f} GB "
                         f"(this job {args.max_vram_gb} GB + {args.keep_free_gb} GB kept free for other apps). Wait "
                         f"and retry. {json.dumps(gpu)}")
    allocator_cap = max(1.0, args.max_vram_gb - resources.CUDA_OVERHEAD_GB)
    torch.cuda.set_per_process_memory_fraction(min(1.0, allocator_cap / gpu["total_gb"]), 0)
    gpu["allocator_cap_gb"] = allocator_cap
    host = resources.require_host_commit(resources.VOXCPM_LOAD_COMMIT_GB, resources.HOST_MARGIN_GB, "VoxCPM2 load")
    lora_config = None
    if args.lora:
        config_path = Path(args.lora) / "lora_config.json"
        info = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        settings = info.get("lora_config") or {
            "enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 32, "alpha": 32, "dropout": 0.0}
        base = str(info.get("base_model", ""))
        if base and SNAPSHOT.name not in base:
            print(f"note: adapter was trained on {base}; loading it on snapshot {SNAPSHOT.name}", flush=True)
        lora_config = LoRAConfig(**settings)
    started = time.time()
    model = VoxCPM.from_pretrained(str(SNAPSHOT), load_denoiser=False, local_files_only=True, optimize=False,
                                   device="cuda", lora_config=lora_config)
    host["process_after_load"] = resources.process_commit()
    print(f"model loaded in {time.time() - started:.0f} s; host commit {json.dumps(host['process_after_load'])}; "
          f"allocator cap {allocator_cap:.1f} GB", flush=True)
    if args.lora:
        loaded, skipped = model.load_lora(str(args.lora))
        if not loaded:
            raise SystemExit(f"No compatible LoRA weights in {args.lora}")
        model.set_lora_enabled(True)
        print(f"LoRA loaded: {len(loaded)} tensors ({len(skipped)} skipped)", flush=True)
    return model, {"gpu": gpu, "host": host}


def load_tokenizer():
    """The tokenizer the render step's length guard counts with (official trainer's char-split wrapper)."""
    from transformers import LlamaTokenizerFast
    from voxcpm.model.utils import mask_multichar_chinese_tokens

    return mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(str(SNAPSHOT), local_files_only=True))


def to_pcm16_48k(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate != OUT_RATE:
        import soxr

        audio = soxr.resample(audio, rate, OUT_RATE, quality="VHQ").astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:  # never clip; the mastering chain sets final level
        audio = audio * (0.999 / peak)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM2 delivery-control probe (GPU).")
    parser.add_argument("--run", default="instr_probe_v1")
    parser.add_argument("--audio", choices=("original", "restored"), default="original",
                        help="reference copy (user decision 2026-09-30: original game audio only)")
    parser.add_argument("--ref-source", choices=("auto", "corpus"), default="auto",
                        help="auto = the voxcpm render step's trimmed copy when it exists (what render.py uses)")
    parser.add_argument("--lora", type=Path, default=None, help="optional VoxCPM2 LoRA adapter directory")
    parser.add_argument("--labels", default="", help="comma list; default all labels in delivery_map.json")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="11,23,37,53")
    parser.add_argument("--cfg", default="1.5", help="cfg_value, or a comma list to sweep (render.py default 1.5)")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--max-vram-gb", type=float, default=9.0)
    parser.add_argument("--keep-free-gb", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run with the same settings")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; no GPU")
    args = parser.parse_args()
    if args.audio != "original":
        print("warning: the user decided (DECISIONS.md, 2026-09-30) to condition on ORIGINAL audio only; "
              "--audio restored is for an explicit experiment", flush=True)

    map_path, variants_path = DELIVERY / "delivery_map.json", DELIVERY / "voxcpm_instruction_variants.json"
    script_path = DELIVERY / "takes_script.json"
    delivery_map = json.loads(map_path.read_text(encoding="utf-8"))
    variants = json.loads(variants_path.read_text(encoding="utf-8"))["variants"]
    script = json.loads(script_path.read_text(encoding="utf-8"))
    jobs, meta = plan(args, delivery_map, variants, script)
    test_ids = set(json.loads((LAB / "corpus_summary.json").read_text(encoding="utf-8"))["test_ids"])
    for job in jobs:
        for ref in (job["reference"], job["prompt"]):
            if ref is None:
                continue
            if ref["id"] in test_ids:
                raise SystemExit(f"{job['take_id']}: held-out test clip {ref['id']} as a reference")
            if not Path(ref[job["audio_key"]]).is_file():
                raise SystemExit(f"reference audio missing: {ref[job['audio_key']]}")
    settings = {
        "probe_version": PROBE_VERSION, "run": args.run, "audio": args.audio, "ref_source": args.ref_source,
        "lora": str(args.lora) if args.lora else None, "snapshot": SNAPSHOT.name, "timesteps": args.timesteps,
        "seeds": meta["seeds"], "cfgs": meta["cfgs"], "variants": meta["variants"], "labels": meta["labels"],
        "identity_id": meta["identity_id"], "prompt_ids": meta["prompt_ids"],
        "delivery_map_sha256": sha256_file(map_path), "variants_sha256": sha256_file(variants_path),
        "script_sha256": sha256_file(script_path), "takes_planned": len(jobs),
        "instructions": {label: {v: variants[v][label] for v in ("A", "B")} for label in meta["labels"]},
    }
    out_dir = DELIVERY / "probe" / args.run
    if args.dry_run:
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job["variant"]] = counts.get(job["variant"], 0) + 1
        low, high = (len(jobs) * s / 60 for s in SECONDS_PER_TAKE)
        cached = sum(reference_audio(j["reference"], j["audio_key"], args.ref_source)[1] for j in jobs)
        print(json.dumps({"takes": len(jobs), "by_variant": counts, "estimated_minutes": [round(low), round(high)],
                          "out_dir": str(out_dir), "snapshot_exists": SNAPSHOT.is_dir(),
                          "reference_from_render_cache": f"{cached}/{len(jobs)}",
                          "prompt_ids": meta["prompt_ids"], "identity_id": meta["identity_id"],
                          "lines": len(meta["lines"]), "first": {k: v for k, v in jobs[0].items()
                                                                 if k not in ("reference", "prompt")}}, indent=1))
        return

    manifest_path = out_dir / "manifest.json"
    manifest = {"takes": [], "failed": [], "settings": settings}
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.resume:
            raise SystemExit(f"{out_dir} already has files; pass --resume to continue it or choose a new --run "
                             "(existing takes are never overwritten).")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = {k: v for k, v in manifest.get("settings", {}).items() if k != "takes_planned"}
        current = {k: v for k, v in settings.items() if k != "takes_planned"}
        if stored != current:
            changed = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
            raise SystemExit(f"--resume refused: settings changed since the run started ({changed}); use a new --run")
        manifest["takes"] = [t for t in manifest["takes"] if (out_dir / t["wav"]).is_file()]
        manifest["failed"] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {t["take_id"] for t in manifest["takes"]}

    import soundfile as sf
    import torch

    model, machine = load_model(args)
    manifest["settings"]["machine"] = machine
    tokenizer = load_tokenizer()
    rate = int(model.tts_model.sample_rate)
    started, rendered = time.time(), 0
    for index, job in enumerate(jobs, 1):
        if job["take_id"] in done:
            continue
        target = clean_text(job["text"])
        ref_path, ref_cached = reference_audio(job["reference"], job["audio_key"], args.ref_source)
        options = {"reference_wav_path": ref_path, "cfg_value": job["cfg"], "inference_timesteps": args.timesteps,
                   "normalize": False, "denoise": False}
        prompt_text = None
        if job["mode"] == "continuation":
            # VoxCPM2 joins prompt_text + text with no separator: end the prompt with a space (render.py does).
            prompt_path, _ = reference_audio(job["prompt"], job["audio_key"], args.ref_source)
            prompt_text = clean_text(job["prompt"]["text"]) + " "
            options.update(text=target, prompt_wav_path=prompt_path, prompt_text=prompt_text)
        else:
            options["text"] = f"({job['instruction']}){target}" if job["instruction"] else target
            full, line = len(tokenizer(options["text"])), len(tokenizer(target))
            if full > line:  # render.py --length-guard line: the runaway cap follows the line, not the instruction
                options["retry_badcase_ratio_threshold"] = BADCASE_RATIO * line / full
        random.seed(job["seed"])
        np.random.seed(job["seed"] % (2**32))
        torch.manual_seed(job["seed"])
        torch.cuda.manual_seed_all(job["seed"])
        try:
            audio = model.generate(**options)
        except torch.cuda.OutOfMemoryError:
            manifest["failed"].append({"take_id": job["take_id"], "error": "CUDA out of memory (allocator cap)"})
            write_json(manifest_path, manifest)
            raise SystemExit(f"{job['take_id']}: CUDA OOM under the allocator cap; stopped. Rerun with --resume.")
        except Exception as exc:  # keep going; the failure is recorded and --resume retries it
            manifest["failed"].append({"take_id": job["take_id"], "error": f"{type(exc).__name__}: {exc}",
                                       "trace": traceback.format_exc(limit=3)})
            write_json(manifest_path, manifest)
            print(f"FAILED {job['take_id']}: {exc}", flush=True)
            continue
        wav_name = f"{job['take_id']}.wav"
        sf.write(str(out_dir / wav_name), to_pcm16_48k(audio, rate), OUT_RATE, subtype="PCM_16")
        manifest["takes"].append({
            "take_id": job["take_id"], "engine": "voxcpm2", "line_id": job["line_id"], "text": job["text"],
            "delivery": job["delivery"], "wav": wav_name, "reference_id": job["reference"]["id"], "seed": job["seed"],
            "params": {"variant": job["variant"], "mode": job["mode"], "instruction": job["instruction"],
                       "instruction_core": job["instruction_core"], "synth_text": options["text"],
                       "prompt_id": job["prompt"]["id"] if job["prompt"] else None, "prompt_text": prompt_text,
                       "identity_ref_id": job["reference"]["id"], "reference_wav": ref_path,
                       "reference_trimmed_copy": ref_cached, "cfg_value": job["cfg"],
                       "inference_timesteps": args.timesteps, "audio": args.audio,
                       "badcase_ratio": options.get("retry_badcase_ratio_threshold", BADCASE_RATIO),
                       "lora": str(args.lora) if args.lora else None, "snapshot": SNAPSHOT.name,
                       "probe_version": PROBE_VERSION},
        })
        write_json(manifest_path, manifest)
        rendered += 1
        if rendered % 10 == 0 or index == len(jobs):
            reserved = torch.cuda.max_memory_reserved(0) / 2**30
            print(f"{index}/{len(jobs)} takes ({rendered} this session), {time.time() - started:.0f} s, peak VRAM "
                  f"reserved {reserved:.1f} GB (allocator cap {machine['gpu']['allocator_cap_gb']:.1f} GB)", flush=True)
    manifest["settings"]["machine"]["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(0) / 2**30, 2)
    manifest["settings"]["machine"]["process_commit_end"] = resources.process_commit()
    write_json(manifest_path, manifest)
    print(f"done: {len(manifest['takes'])} takes in {out_dir} ({len(manifest['failed'])} failed)", flush=True)


if __name__ == "__main__":
    main()
