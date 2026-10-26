"""Render a voice-lab character's takes with VoxCPM2 + LoRA through the voxcpm Python API (GPU step; not the Vox
sidecar). The character is VOICE_LAB_CHARACTER (unset = baodur; see common.py); <lab> is its lab folder
(KotorMods\\Voices\\BaoDurLab, JoleeLab, ...).

  cd C:\\Users\\NewAdmin\\Documents\\GDeveloper\\External\\voice-lab\\baodur-lab\\voxcpm
  set R=C:\\Users\\NewAdmin\\Documents\\KotorMods\\Voices\\BaoDurLab\\voxcpm\\runs\\lora_v2_original\\checkpoints
  %LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe render.py ^
      --lora %R%\\step_0000100 %R%\\step_0000200 %R%\\step_0000300 %R%\\step_0000400 %R%\\step_0000500 ^
             %R%\\step_0000700 %R%\\step_0001000 ^
      --baseline --no-lora --modes ref_prompt,idref_instr --seeds 1 --cfg 1.5 --instruction-variant B --output both
  (Bao-Dur's coarse checkpoint sweep; then add seeds to the two best with the same command, --lora <best two>
  --seeds 4 --resume. --dry-run validates everything and prints the plan on CPU without loading the model.
  Another character: set VOICE_LAB_CHARACTER=jolee and point --lora at <lab>\\voxcpm\\runs\\...; --baseline
  needs the character config's vox_profile_id.)

Inputs: <lab>\\delivery\\takes_script.json {"lines":[{id,text,delivery,source,heldout_id?}]} and
<lab>\\delivery\\delivery_map.json {"deliveries":{label:{references:[{id,audio,restored_audio,text,
duration,why}],voxcpm_instruction,notes}}}. Output: one run folder per adapter and mode (--split-modes,
default) <lab>\\takes\\voxcpm\\<adapter>--<mode>\\manifest.json {"takes":[{take_id,engine,line_id,text,
delivery,wav,reference_id,seed,params,heldout_id?}]} with 48 kHz mono PCM16 WAVs beside it (manifest
rewritten atomically after every take; --resume continues a run, adds seeds, and refuses when the stored
settings differ from the current ones). One folder per mode lets the evaluation (--group-by run) and
select_checkpoint.py compare adapters within a mode instead of pooling ref_prompt and idref_instr takes;
--no-split-modes writes one folder per adapter (<adapter>) as before.

Conditioning modes (--modes, default ref_prompt,idref_instr), mirroring how Vox's sidecar drives VoxCPM2:
  ref_prompt        ultimate cloning: reference_wav_path + prompt_wav_path/prompt_text = the delivery's
                    reference clip and its exact transcript, no instruction; the clip's delivery carries
                    into the line (--identity-ref natural puts the fixed identity clip in reference_wav).
  idref_instr       controlled cloning as Vox production runs it (use_stable_character_identity +
                    use_controlled_cloning): reference_wav_path = ONE fixed identity clip for every line
                    (the top usable reference of --identity-label, default the character's neutral label:
                    recipe.json neutral_label, else natural if mapped, else the map's first label; Bao-Dur:
                    natural = gblbaodur248),
                    text = "(<identity clause> <instruction>)line", no prompt. Only the instruction differs
                    between deliveries, so this mode measures instruction control and whether an adapter
                    keeps it. (Vox's own Bao-Dur profile reference.wav is held-out test clip 003 per the review;
                    it can never be used here: every conditioning clip is resolved from a train/val corpus id.)
  ref_instr         controlled cloning with the delivery's OWN clip as reference_wav: style comes from both
                    the clip and the instruction (secondary; confounds the two).
  ref_prompt_instr  experimental: ref_prompt with the instruction prefixed to the prompt transcript, so it
                    still opens the text sequence (Vox never mixes the two).
  ref               reference_wav_path only, plain text (timbre only; the flattest reading).
Each line x mode x reference (--refs-per-line, ranked best first; idref_instr has one) x seed (--seeds;
seeds depend only on line id and index, so every adapter/checkpoint gets the same seeds).
Instructions: --instruction-variant map (delivery_map.json voxcpm_instruction: variant A until the delivery
task's probe writes its per-label winner back), A or B (<lab>\\delivery\\voxcpm_instruction_variants.json;
B is compact and positive-only, A carries timbre words and one negated guard clause that can pull the voice
away from an adapter); a label none of them covers falls back to the character's recipe.json
fallback_instructions (generic defaults in common.RECIPE_DEFAULTS). --identity-clause auto (default) prefixes
Vox's adapter identity clause "Keep the trained voice stable." for LoRA adapters (never for the base-model
control). --cfg defaults to 1.5, the value Vox's profiles render with (Bao-Dur: cfg_value/controlled_cfg_value);
the official default is 2.0.
Length guard (--length-guard line, default): VoxCPM caps a take at 6 patches per token of the text it is
given (+10) and retries when it hits that; with "(instruction)line" the instruction tokens would loosen the
cap by ~30 s, so instruction modes get a ratio rescaled to the line's own token count (same cap as plain
modes). "full" keeps VoxCPM's behaviour (what Vox does).

Adapters: --lora <checkpoint dir> (repeatable, hot-swapped into one loaded model), --baseline (the character's
installed Vox adapter, voxcpm_profiles\\<vox_profile_id>\\lora, read in place, never written; Bao-Dur's is the
July adapter, run label from recipe.json baseline_label), --no-lora (base-model control), --lora-scale s [s ...]
(renders each adapter at each LoRA strength; label suffix -x<s> when s != 1). The base weights always come from
snapshot 32279eff (byte-identical to the bffb3df5 snapshot the July adapter names). Provenance: a --lora
checkpoint's training rows are read from <run>\\run_config.json -> config.train_manifest (train_lora_v2.py
points it at the run's frozen copy in <run>\\data\\) and must still hash to the recorded
train_manifest_sha256; a missing or changed manifest stops the run, and a checkpoint without run_config.json
needs --allow-unknown-provenance (its held-out lines are then treated as trained on). Held-out lines an
adapter was trained on (its training manifest holds the held-out clip, a screen near-duplicate of it, or a
text repeating it or one of its distinctive phrases - Bao-Dur's July adapter trained on 181/200/223/251/276) are
NOT rendered for that adapter, so no evaluation can score memorised lines as unseen: the July adapter
renders 13 of the 18 lines. --keep-trained-heldout renders them under the line id "<line>--adapter-trained"
without heldout_id (listening only; params.adapter_trained_on_heldout = true). Reference audio is ORIGINAL
(user decision 2026-09-30); --ref-audio restored needs --allow-restored-audio, and a LoRA trained on another
audio variant than --ref-audio is refused unless --allow-ref-variant-mismatch. Respellings (--pronunciations):
Vox's shared KOTOR lexicon plus the character's recipe.json "pronunciations" entries (same term: the recipe wins).

Output (--output): raw (default) writes the decoder output untouched, so lead/tail silence and runaway
endings stay visible to the evaluation (params.output_trim_s says what trimming would remove); trimmed
writes the Vox-style end-trimmed take; both writes raw as `wav` plus <take>.trimmed.wav for listening.

Safety: held-out test clips are refused as references/prompts; for held-out lines, references that speak
the same sentence or are screen near-duplicates of it are skipped, for the delivery reference and the
identity clip alike. Speaker screen (<lab>\\delivery\\exclude_other_speakers.json): a 'confirmed'
other-character clip as a reference stops the run, 'uncertain' ones are skipped, 'suspect' ones are used but
tagged params.reference_screen = "suspect". On CUDA the process refuses to start unless it can stay under
--max-vram-gb while leaving --reserve-gb free, and caps its allocator accordingly (~6.8 GB estimated: bf16
weights 4.6 + VAE 0.4 + KV cache 0.3 + decode 0.9 + context 0.6). Host RAM: VoxCPM2Model.from_local builds
the model in fp32 on the CPU first (~13 GB commit peak while loading); the process refuses to load unless
that much plus --host-margin-gb stays free. No bytecode is written.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the Vox venv's site-packages

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODES = ("ref_prompt", "idref_instr", "ref_instr", "ref_prompt_instr", "ref")
INSTRUCTION_MODES = {"idref_instr", "ref_instr", "ref_prompt_instr"}
SINGLE_REF_MODES = {"idref_instr"}  # the delivery reference is not used, so one slot only
IDENTITY_CLAUSE = "Keep the trained voice stable."  # vox_profiles.control_identity_instruction for adapters
BADCASE_RATIO = 6.0  # VoxCPM.generate default retry_badcase_ratio_threshold
MAX_LEN = 4096  # VoxCPM.generate default max_len
TRAINED_SUFFIX = "--adapter-trained"
RESUME_KEYS = ("cfg_value", "inference_timesteps", "ref_audio", "ref_trimmed", "identity_ref", "identity_label",
               "identity_clause", "instruction_variant", "pronunciations", "optimize", "output", "length_guard")
# label -> instruction when neither the delivery map nor the variants file has one: the character's recipe.json
# fallback_instructions over the generic common.RECIPE_DEFAULTS (Bao-Dur's file keeps his tuned wording)
FALLBACK_INSTRUCTIONS = {str(k).casefold(): str(v) for k, v in C.recipe_value("fallback_instructions").items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora", nargs="+", type=Path, default=[], help="LoRA checkpoint dir(s) (step_XXXXXXX)")
    p.add_argument("--baseline", action="store_true",
                   help="render the character's installed Vox adapter (vox_profile_id; read-only)")
    p.add_argument("--no-lora", action="store_true", help="render the base model as a control")
    p.add_argument("--lora-scale", nargs="+", type=float, default=[1.0],
                   help="LoRA strength(s) for --lora/--baseline adapters (1.0 = as trained)")
    p.add_argument("--run-prefix", default="", help="prefix for run folder names")
    p.add_argument("--script", type=Path, default=C.TAKES_SCRIPT)
    p.add_argument("--delivery-map", type=Path, default=C.DELIVERY_MAP)
    p.add_argument("--instruction-variants", type=Path, default=C.INSTRUCTION_VARIANTS)
    p.add_argument("--modes", default="ref_prompt,idref_instr")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--refs-per-line", type=int, default=1)
    p.add_argument("--cfg", type=float, default=1.5,
                   help="cfg_value (Vox profiles render with 1.5, e.g. Bao-Dur's; official default 2.0)")
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--lines", default="", help="comma-separated line ids to render (default all)")
    p.add_argument("--limit-takes", type=int, default=0, help="stop after N takes per adapter (smoke tests)")
    p.add_argument("--ref-audio", choices=C.AUDIO_VARIANTS, default="original")
    p.add_argument("--allow-restored-audio", action="store_true",
                   help="allow --ref-audio restored (against the user's 2026-09-30 decision)")
    p.add_argument("--allow-ref-variant-mismatch", action="store_true",
                   help="allow --ref-audio to differ from the audio variant a LoRA was trained on")
    p.add_argument("--allow-unknown-provenance", action="store_true",
                   help="render a --lora dir without run_config.json (its held-out lines count as trained on)")
    p.add_argument("--split-modes", action=argparse.BooleanOptionalAction, default=True,
                   help="one run folder per adapter and mode, <adapter>--<mode> (default); --no-split-modes = one per adapter")
    p.add_argument("--no-ref-trim", action="store_true", help="use untrimmed source clips as reference/prompt")
    p.add_argument("--identity-ref", choices=("same", "natural"), default="same",
                   help="reference_wav in ref_prompt/ref modes: the delivery clip itself (same) or the fixed identity "
                        "clip of --identity-label (natural; the keyword keeps its name for every character)")
    p.add_argument("--identity-label", default=None,
                   help="delivery label whose top usable reference is the fixed identity clip (default: the "
                        "neutral label = recipe.json neutral_label, else 'natural' if the delivery map has it, "
                        "else the map's first label)")
    p.add_argument("--instruction-variant", choices=("map", "A", "B"), default="map")
    p.add_argument("--identity-clause", choices=("auto", "on", "off"), default="auto",
                   help="prefix 'Keep the trained voice stable.' to instructions (auto = adapters only)")
    p.add_argument("--length-guard", choices=("line", "full"), default="line")
    p.add_argument("--keep-trained-heldout", action="store_true",
                   help="also render held-out lines an adapter trained on, renamed and unpaired (listening only)")
    p.add_argument("--pronunciations", choices=("auto", "vox", "none"), default="auto",
                   help="auto = Vox respellings only for names never heard in the training text")
    p.add_argument("--out-root", type=Path, default=C.TAKES_ROOT)
    p.add_argument("--output", choices=("raw", "trimmed", "both"), default="raw")
    p.add_argument("--max-vram-gb", type=float, default=9.0)
    p.add_argument("--reserve-gb", type=float, default=8.0)
    p.add_argument("--host-margin-gb", type=float, default=C.HOST_MARGIN_GB)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--optimize", action="store_true", help="torch.compile (faster, more VRAM, slow warmup)")
    p.add_argument("--resume", action="store_true", help="continue existing run folders")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in args.mode_list if m not in MODES]
    if bad:
        p.error(f"unknown modes {bad}; choose from {MODES}")
    if not (args.lora or args.baseline or args.no_lora):
        p.error("choose at least one of --lora, --baseline, --no-lora")
    if args.baseline and C.BASELINE_LORA is None:
        p.error(f"--baseline: {C.DISPLAY_NAME} has no installed Vox adapter (vox_profile_id is null in "
                f"voice-lab\\characters\\{C.SLUG}.json)")
    if args.seeds < 1 or args.refs_per_line < 1:
        p.error("--seeds and --refs-per-line must be >= 1")
    if any(not 0.0 < s <= 2.0 for s in args.lora_scale) or len(set(args.lora_scale)) != len(args.lora_scale):
        p.error("--lora-scale values must be distinct and within (0, 2]")
    if args.ref_audio == "restored" and not args.allow_restored_audio:
        p.error(f"--ref-audio restored is off: {C.RESTORED_DECISION}. Pass --allow-restored-audio only if the "
                "user reverses that decision.")
    args.identity_label = args.identity_label.casefold() if args.identity_label else None  # None: neutral_label()
    return args


# --------------------------------------------------------------------------------------------- adapters
def lora_dir_info(path: Path) -> dict:
    weights = path / "lora_weights.safetensors"
    config = path / "lora_config.json"
    if not weights.exists() or not config.exists():
        raise SystemExit(f"{path} is not a LoRA checkpoint (needs lora_weights.safetensors + lora_config.json)")
    info = json.loads(config.read_text(encoding="utf-8"))
    return {"weights": weights, "lora_config": info.get("lora_config") or {}, "base_model": info.get("base_model")}


def training_exposure(rows: list[dict]) -> tuple[set[str], list[str]]:
    """Clip ids (targets and ref_audio) and texts an adapter was trained on, from its train manifest."""
    ids: set[str] = set()
    texts: list[str] = []
    for row in rows:
        ids.add(str(row.get("id") or Path(row["audio"]).stem).casefold())
        if row.get("ref_id") or row.get("ref_audio"):
            ids.add(str(row.get("ref_id") or Path(row["ref_audio"]).stem).casefold())
        texts.append(re.sub(r"^\([^)]*\)", "", str(row.get("text") or "")).strip())
    return ids, texts


def lora_provenance(path: Path, label: str, allow_unknown: bool) -> dict:
    """What a --lora checkpoint was trained on, from <run>\\run_config.json (written by train_lora_v2.py).

    The manifest it names must still hash to the recorded train_manifest_sha256; otherwise the held-out
    exposure checks below (and the evaluation's leak audit, which follows the same file) would read rows the
    adapter never saw."""
    run_config = path.parent.parent / "run_config.json"
    if not run_config.exists():
        if not allow_unknown:
            raise SystemExit(f"{label}: no {run_config}; cannot tell which clips it trained on (held-out leak "
                             "check). Pass --allow-unknown-provenance for a listening-only render.")
        return {"trained_ids": set(), "trained_texts": [], "train_variant": None, "provenance": "unknown",
                "train_manifest": None}
    record = json.loads(run_config.read_text(encoding="utf-8"))
    manifest = Path(record["config"]["train_manifest"])
    if not manifest.exists():
        raise SystemExit(f"{label}: its training manifest {manifest} (from {run_config}) is missing")
    recorded = record.get("train_manifest_sha256")
    actual = C.sha256_file(manifest)
    if recorded and recorded != actual:
        raise SystemExit(f"{label}: {manifest} changed after training (sha256 {actual[:12]} != recorded "
                         f"{recorded[:12]} in {run_config}); restore the run's own copy in <run>\\data\\ or retrain")
    variant = record.get("audio_variant") or (manifest.parent.name if manifest.parent.name in C.AUDIO_VARIANTS
                                              else None)
    ids, texts = training_exposure(C.read_jsonl(manifest))
    return {"trained_ids": ids, "trained_texts": texts, "train_variant": variant, "train_manifest": str(manifest),
            "provenance": "verified (sha256)" if recorded else "unverified (no train_manifest_sha256 recorded)"}


def resolve_adapters(args) -> list[dict]:
    adapters = []
    for path in args.lora:
        path = path.resolve()
        info = lora_dir_info(path)
        run_root = path.parent.parent
        label = f"{args.run_prefix}{run_root.name}-{path.name}"
        adapters.append({"label": label, "kind": "lora", "path": path,
                         **lora_provenance(path, label, args.allow_unknown_provenance), **info})
    if args.baseline:
        info = lora_dir_info(C.BASELINE_LORA)
        known = C.BASELINE_TRAIN_MANIFEST.exists()
        ids, texts = training_exposure(C.read_jsonl(C.BASELINE_TRAIN_MANIFEST)) if known else (set(), [])
        adapters.append({"label": f"{args.run_prefix}{C.BASELINE_LABEL}", "kind": "baseline",
                         "path": C.BASELINE_LORA, "trained_ids": ids, "trained_texts": texts,
                         "train_variant": "original", "train_manifest": str(C.BASELINE_TRAIN_MANIFEST) if known else None,
                         "provenance": C.BASELINE_PROVENANCE if known else "unknown", **info})
    expanded = []
    for adapter in adapters:
        for scale in args.lora_scale:
            item = dict(adapter)
            item["lora_scale"] = scale
            if scale != 1.0:
                item["label"] = f"{adapter['label']}-x{scale:g}"
            expanded.append(item)
    if args.no_lora:
        expanded.append({"label": f"{args.run_prefix}base-no-lora", "kind": "none", "path": None, "weights": None,
                         "lora_config": {"enable_lm": True, "enable_dit": True, "enable_proj": False, "r": 32,
                                         "alpha": 32, "dropout": 0.0},
                         "base_model": str(C.SNAPSHOT), "trained_ids": set(), "trained_texts": [],
                         "train_variant": None, "train_manifest": None, "provenance": "base model, no adapter",
                         "lora_scale": None})
    for adapter in expanded:
        base = adapter.get("base_model") or ""
        if base and C.SNAPSHOT_ID not in base and C.OLD_SNAPSHOT_ID not in base:
            raise SystemExit(f"{adapter['label']} was trained on an unknown base model: {base}")
        variant = adapter.get("train_variant")
        if adapter["kind"] == "lora" and variant and variant != args.ref_audio and not args.allow_ref_variant_mismatch:
            raise SystemExit(f"{adapter['label']} was trained on '{variant}' audio but --ref-audio is "
                             f"'{args.ref_audio}': render it with --ref-audio {variant} (and its own --run-prefix), "
                             "or pass --allow-ref-variant-mismatch to separate the adapter and reference effects")
    labels = [a["label"] for a in expanded]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate run labels {labels}; use --run-prefix")
    return expanded


def lora_signature(cfg: dict) -> tuple:
    keys = ("enable_lm", "enable_dit", "enable_proj", "r", "alpha", "target_modules_lm", "target_modules_dit",
            "target_proj_modules")
    from voxcpm.model.voxcpm2 import LoRAConfig

    full = LoRAConfig(**cfg).model_dump()
    return tuple(json.dumps(full[k]) for k in keys)


def trained_heldout_reason(adapter: dict, heldout_id: str, heldout_text: str, near_dup_of: dict) -> str | None:
    """Why `adapter` has effectively heard held-out line `heldout_id` in training (None if it has not)."""
    hid = heldout_id.casefold()
    if adapter.get("provenance") == "unknown":
        return "its training data is unknown (no run_config.json / train manifest)"
    if hid in adapter["trained_ids"]:
        return "the held-out clip is in its training data"
    dups = {cid.casefold() for cid, targets in near_dup_of.items() if hid in {t.casefold() for t in targets}}
    if dups & adapter["trained_ids"]:
        return f"near-duplicate(s) {sorted(dups & adapter['trained_ids'])} of the held-out line in its training data"
    if any(t and (C.overlaps_text(t, heldout_text) or C.overlaps_text(heldout_text, t))
           for t in adapter["trained_texts"]):
        return "a training text repeats the held-out line"
    return None


def set_lora_scale(model, scale: float) -> int:
    """Scale every LoRALinear's contribution (after set_lora_enabled(True) restored alpha/r)."""
    count = 0
    for module in model.tts_model._iter_lora_modules():
        module.scaling.fill_(module._base_scaling * scale)
        count += 1
    return count


# --------------------------------------------------------------------------------------------- plan
def pronunciation_entries() -> list[dict]:
    """Vox's shared lexicon in file order, minus the terms the character's recipe.json "pronunciations"
    redefines, then the recipe's own entries (an empty recipe list leaves Vox's lexicon exactly as it is)."""
    vox = (json.loads(C.VOX_PRONUNCIATIONS.read_text(encoding="utf-8")).get("entries", [])
           if C.VOX_PRONUNCIATIONS.exists() else [])
    own = list(C.recipe_value("pronunciations") or [])
    redefined = {str(entry["term"]).casefold() for entry in own}
    return [entry for entry in vox if str(entry["term"]).casefold() not in redefined] + own


def pronunciation_rules(mode: str, training_texts: list[str]) -> list[tuple[str, str, re.Pattern]]:
    if mode == "none":
        return []
    rules = []
    for entry in pronunciation_entries():
        alternatives = sorted({entry["term"], *entry.get("aliases", [])}, key=len, reverse=True)
        pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, alternatives)) + r")(?![A-Za-z0-9])",
                             re.IGNORECASE)
        if mode == "auto" and any(pattern.search(t) for t in training_texts):
            continue  # the adapter heard the actor say it; keep the real spelling
        rules.append((entry["term"], entry["spoken"], pattern, max(map(len, alternatives))))
    rules.sort(key=lambda r: -r[3])
    return [(t, s, p) for t, s, p, _ in rules]


def apply_rules(text: str, rules) -> tuple[str, list[str]]:
    matched = []
    for term, spoken, pattern in rules:
        text, count = pattern.subn(spoken, text)
        if count:
            matched.append(term)
    return text, matched


def reference_path(ref_id: str, corpus: dict, variant: str, trimmed: bool) -> Path:
    if trimmed:
        cached = C.cache_dir(variant) / f"{ref_id}.wav"
        if cached.exists():
            return cached
    return C.source_audio(corpus[ref_id], variant)


def load_instructions(args, deliveries: dict) -> dict[str, str]:
    """label -> delivery instruction (without parentheses or identity clause) for --instruction-variant."""
    variants = {}
    if args.instruction_variant != "map":
        if not args.instruction_variants.exists():
            raise SystemExit(f"--instruction-variant {args.instruction_variant} needs {args.instruction_variants}")
        doc = json.loads(args.instruction_variants.read_text(encoding="utf-8"))
        variants = {k.casefold(): v for k, v in (doc.get("variants", {}).get(args.instruction_variant) or {}).items()}
        if not variants:
            raise SystemExit(f"{args.instruction_variants} has no variant {args.instruction_variant}")
    out = {}
    for label, entry in deliveries.items():
        text = variants.get(label) or (entry.get("voxcpm_instruction") if args.instruction_variant == "map" else "")
        out[label] = (text or FALLBACK_INSTRUCTIONS.get(label, "")).strip().strip("()").strip()
    return out


def neutral_label(deliveries: dict) -> str:
    """Delivery label of the fixed identity clip and of lines without a known delivery: the character's
    recipe.json neutral_label, else 'natural' when the delivery map has it (Bao-Dur), else the map's first label."""
    configured = C.recipe_value("neutral_label")
    if configured:
        return str(configured).strip().casefold()
    if "natural" in deliveries or not deliveries:
        return "natural"
    return next(iter(deliveries))


def compose_instruction(instruction: str, clause: bool) -> str:
    if not instruction:
        return IDENTITY_CLAUSE if clause else ""
    if not clause:
        return instruction
    return f"{IDENTITY_CLAUSE} {instruction[:1].upper()}{instruction[1:]}"


def build_plan(args, corpus: dict[str, dict], screen: dict) -> tuple[list[dict], list[str], dict]:
    warnings: list[str] = []
    script = json.loads(args.script.read_text(encoding="utf-8"))
    dmap = json.loads(args.delivery_map.read_text(encoding="utf-8"))["deliveries"]
    deliveries = {k.casefold(): v for k, v in dmap.items()}
    neutral = neutral_label(deliveries)
    if args.identity_label is None:  # resolved here so the run manifests record the label actually used
        args.identity_label = neutral
    instructions = load_instructions(args, deliveries)
    wanted = {x.strip() for x in args.lines.split(",") if x.strip()}
    tests = {i for i, r in corpus.items() if r["split"] == "test"}
    trimmed = not args.no_ref_trim
    noted: set[str] = set()
    # screen's near-duplicate list: {train clip: held-out clips it (nearly) repeats}; the reason text names them
    # by corpus clip id (gblbaodur052, nm35aajole09069_, ...), matched word-bounded against the corpus ids
    clip_ids = C.test_id_pattern(set(corpus)) if corpus else None
    near_dup_of = {cid: ({m.group(0) for m in clip_ids.finditer(str(reason))} if clip_ids else set()) - {cid}
                   for cid, reason in screen["near_duplicate_of_test"].items()}

    def warn_once(key: str, message: str) -> None:
        if key not in noted:
            warnings.append(message)
            noted.add(key)

    def valid_refs(label: str, heldout_text: str | None, heldout_id: str | None = None) -> list[dict]:
        out = []
        for ref in deliveries.get(label, {}).get("references", []):
            rid = ref.get("id", "")
            row = corpus.get(rid)
            if row is None:
                warn_once(f"x:{label}:{rid}", f"{label}: reference {rid} is not in the corpus; skipped")
                continue
            if rid in tests or row["split"] == "test":
                raise SystemExit(f"delivery '{label}' uses held-out test clip {rid} as a reference")
            if rid in screen["confirmed"]:
                raise SystemExit(f"delivery '{label}' uses {rid} as a reference, but that clip is "
                                 f"{screen['confirmed'][rid]} speaking (speaker screen), not {C.DISPLAY_NAME}")
            if rid in screen["uncertain"]:
                warn_once(f"u:{label}:{rid}",
                          f"{label}: reference {rid} is speaker-'uncertain' (not verified {C.DISPLAY_NAME}); skipped")
                continue
            if row["kind"] != "speech":
                warn_once(f"n:{label}:{rid}", f"{label}: reference {rid} is nonverbal; skipped")
                continue
            text = C.clean_text(ref.get("text") or row["text"])
            if heldout_text and (C.overlaps_text(text, heldout_text) or C.overlaps_text(heldout_text, text)
                                 or (heldout_id and heldout_id.casefold()
                                     in {x.casefold() for x in near_dup_of.get(rid, ())})):
                warn_once(f"h:{label}:{rid}:{heldout_id}",
                          f"{label}: reference {rid} speaks held-out line {heldout_id}; skipped for that line")
                continue
            path = reference_path(rid, corpus, args.ref_audio, trimmed)
            if not path.exists():
                warn_once(f"p:{label}:{rid}", f"{label}: audio for {rid} not found at {path}; skipped")
                continue
            out.append({"id": rid, "text": text, "path": path, "duration": ref.get("duration", row["duration"]),
                        "screen": "suspect" if rid in screen["suspect"] else None})
        return out

    needs_identity = "idref_instr" in args.mode_list or args.identity_ref == "natural"
    if needs_identity and args.identity_label not in deliveries:
        raise SystemExit(f"--identity-label {args.identity_label} is not a delivery in {args.delivery_map}")
    plan: list[dict] = []
    for line in script.get("lines", []):
        line_id = str(line["id"])
        if wanted and line_id not in wanted:
            continue
        label = str(line.get("delivery") or neutral).casefold()
        if label not in deliveries:
            warnings.append(f"line {line_id}: delivery '{label}' not in the delivery map; using {neutral}")
            label = neutral
        heldout = line.get("heldout_id") if line.get("source") == "heldout" else None
        if line.get("source") == "heldout" and not heldout:
            warnings.append(f"line {line_id}: source heldout without heldout_id")
        if heldout and heldout not in tests:
            warnings.append(f"line {line_id}: heldout_id {heldout} is not a test clip")
        text = str(line["text"])
        refs = valid_refs(label, text if heldout else None, heldout)
        # the fixed identity clip, filtered for THIS line exactly like the delivery references
        identity_refs = valid_refs(args.identity_label, text if heldout else None, heldout) if needs_identity else []
        identity_line = identity_refs[0] if identity_refs else None
        instruction = instructions.get(label, "")
        for mode in args.mode_list:
            if mode == "idref_instr" or (args.identity_ref == "natural" and mode in ("ref_prompt", "ref_prompt_instr",
                                                                                        "ref")):
                if identity_line is None:
                    warnings.append(f"line {line_id}: no usable '{args.identity_label}' identity clip; {mode} skipped")
                    continue
            slots = [identity_line] if mode in SINGLE_REF_MODES else refs[: args.refs_per_line]
            if not slots:
                warnings.append(f"line {line_id}: no usable reference for '{label}'; {mode} skipped")
                continue
            for ri, ref in enumerate(slots):
                if mode in SINGLE_REF_MODES or mode == "ref_instr":
                    identity = ref
                else:
                    identity = identity_line if args.identity_ref == "natural" else ref
                for k in range(args.seeds):
                    plan.append({
                        "take_id": f"{line_id}-{mode}-r{ri}-s{k}", "line_id": line_id, "text": text,
                        "delivery": label, "heldout_id": heldout, "mode": mode, "ref": ref, "identity": identity,
                        "instruction": instruction if mode in INSTRUCTION_MODES else "",
                        "instruction_variant": args.instruction_variant,
                        "seed": C.stable_seed("voxcpm", line_id, k),
                    })
    return plan, warnings, near_dup_of


def conditioning(take: dict, rules, clause: bool) -> tuple[dict, dict]:
    """Keyword arguments for VoxCPM.generate plus the bookkeeping that goes into params."""
    target, respelled = apply_rules(C.clean_text(take["text"]), rules)
    ref, identity, mode = take["ref"], take["identity"], take["mode"]
    instruction = compose_instruction(take["instruction"], clause) if mode in INSTRUCTION_MODES else ""
    prompt_text = None
    kwargs: dict = {"reference_wav_path": str(identity["path"])}
    if mode in ("ref_prompt", "ref_prompt_instr"):
        prompt_text = ref["text"] if ref["text"].endswith((" ", "\n")) else ref["text"] + " "
        if mode == "ref_prompt_instr" and instruction:
            prompt_text = f"({instruction}){prompt_text}"
        kwargs.update(prompt_wav_path=str(ref["path"]), prompt_text=prompt_text)
        synth = target
        used = [identity, ref]
    elif mode in ("idref_instr", "ref_instr") and instruction:
        synth = f"({instruction}){target}"
        used = [identity]
    else:
        synth = target
        used = [identity]
    kwargs["text"] = synth
    meta = {
        "mode": mode,
        "reference_screen": "suspect" if any(c.get("screen") == "suspect" for c in used) else None,
        "reference_wav": kwargs["reference_wav_path"],
        "identity_ref_id": identity["id"],
        "prompt_id": ref["id"] if prompt_text is not None else None,
        "prompt_wav": kwargs.get("prompt_wav_path"),
        "prompt_text": prompt_text,
        "instruction": instruction,
        "instruction_variant": take["instruction_variant"] if instruction else None,
        "identity_clause": bool(instruction) and clause,
        "synth_text": synth,
        "target_text_spoken": target,
        "pronunciations": respelled,
    }
    return kwargs, meta


def length_guard(tokenizer, synth: str, target: str, guard: str) -> tuple[dict, dict]:
    """generate() kwargs that put the runaway cap/retry threshold on the line's own token count."""
    if tokenizer is None:
        return {}, {}
    full, line = len(tokenizer(synth)), len(tokenizer(target))
    ratio = BADCASE_RATIO
    if guard == "line" and full > line:
        ratio = BADCASE_RATIO * line / full
    info = {"text_tokens": full, "line_tokens": line, "badcase_ratio": round(ratio, 4),
            "max_patches": min(int(full * ratio + 10), MAX_LEN)}
    info["max_seconds"] = round(info["max_patches"] * 4 / 25.0, 2)  # patch = 4 latent frames at 25 fps
    return ({"retry_badcase_ratio_threshold": ratio} if ratio != BADCASE_RATIO else {}), info


def clause_for(args, adapter: dict) -> bool:
    if args.identity_clause == "auto":
        return adapter["kind"] in ("lora", "baseline")
    return args.identity_clause == "on"


# --------------------------------------------------------------------------------------------- audio
def trim_output(audio: np.ndarray, rate: int, lead_pad: float = 0.04, tail_pad: float = 0.12,
                fade_ms: float = 8.0) -> tuple[np.ndarray, list[float]]:
    """Vox-style end trim (threshold 4% of the 90th-percentile frame RMS) with gentler padding."""
    hop = int(rate * 0.01)
    n = len(audio) // hop
    if n < 3:
        return audio, [0.0, 0.0]
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    threshold = max(0.0008, min(0.006, float(np.percentile(rms, 90)) * 0.04))
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return audio, [0.0, 0.0]
    start = max(0, int(active[0]) * hop - int(lead_pad * rate))
    end = min(len(audio), (int(active[-1]) + 1) * hop + int(tail_pad * rate))
    out = audio[start:end].astype(np.float32, copy=True)
    fade = min(int(rate * fade_ms / 1000), len(out) // 4)
    if fade > 0:
        out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out, [round(start / rate, 3), round((len(audio) - end) / rate, 3)]


# --------------------------------------------------------------------------------------------- main
def adapter_plan(args, adapter: dict, plan: list[dict], near_dup_of: dict) -> tuple[list[dict], list[dict]]:
    """The takes this adapter renders, and the held-out lines skipped because it trained on them."""
    takes, skipped, reasons = [], [], {}
    for take in plan:
        heldout = take["heldout_id"]
        if heldout and heldout not in reasons:
            reasons[heldout] = trained_heldout_reason(adapter, heldout, take["text"], near_dup_of)
        reason = reasons.get(heldout) if heldout else None
        if not reason:
            takes.append(take)
            continue
        if not args.keep_trained_heldout:
            if not any(s["line_id"] == take["line_id"] for s in skipped):
                skipped.append({"line_id": take["line_id"], "heldout_id": heldout, "reason": reason})
            continue
        renamed = dict(take)
        renamed.update(line_id=f"{take['line_id']}{TRAINED_SUFFIX}", heldout_id=None, trained_line=True,
                       take_id=take["take_id"].replace(take["line_id"], f"{take['line_id']}{TRAINED_SUFFIX}", 1))
        takes.append(renamed)
    return takes, skipped


def run_groups(args, adapter: dict, takes: list[dict]) -> list[tuple[str, str | None, list[dict]]]:
    """(run folder label, mode or None, takes) for one adapter: one folder per mode unless --no-split-modes."""
    if not args.split_modes:
        return [(adapter["label"], None, takes)]
    return [(f"{adapter['label']}--{mode}", mode, [t for t in takes if t["mode"] == mode]) for mode in args.mode_list]


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or args.dry_run:  # "-1", not "": on Windows "" deletes the variable
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    C.offline_env()
    rows = C.load_corpus()
    corpus = {r["id"]: r for r in rows}
    for path in (args.script, args.delivery_map):
        if not path.exists():
            raise SystemExit(f"missing input {path} (the delivery step writes it)")
    screen = C.load_speaker_screen()
    plan, warnings, near_dup_of = build_plan(args, corpus, screen)
    adapters = resolve_adapters(args)
    screened = set(screen["confirmed"]) | screen["uncertain"]  # other characters' words were never this character's
    training_texts = [C.clean_text(r["text"]) for r in rows if r["split"] in ("train", "val") and r["tts_usable"]
                      and r["id"] not in screened]
    rules = pronunciation_rules(args.pronunciations, training_texts)
    tests = {i for i, r in corpus.items() if r["split"] == "test"}
    forbidden = C.test_id_pattern(tests | screened)
    allowed_dirs = {C.cache_dir(v).resolve() for v in C.AUDIO_VARIANTS} | {C.RESTORED_DIR.resolve()} | {
        Path(r["audio"]).resolve().parent for r in rows if r["split"] != "test"}
    for take in plan:  # belt and braces: nothing conditioning a take may name a test or screened clip
        _, meta = conditioning(take, rules, True)
        line_forbidden = forbidden
        if take["heldout_id"]:
            dups = {cid for cid, targets in near_dup_of.items()
                    if take["heldout_id"].casefold() in {t.casefold() for t in targets}}
            line_forbidden = C.test_id_pattern(tests | screened | dups)
        hits = [m.group(0) for key in ("reference_wav", "prompt_wav", "identity_ref_id", "prompt_id")
                for m in line_forbidden.finditer(str(meta.get(key) or ""))]
        if hits:
            raise SystemExit(f"take {take['take_id']} would be conditioned on held-out/screened clip(s) {hits}")
        for key in ("reference_wav", "prompt_wav"):
            if meta.get(key) and Path(meta[key]).resolve().parent not in allowed_dirs:
                raise SystemExit(f"take {take['take_id']}: {key} {meta[key]} is not a lab corpus clip")
    per_adapter = {a["label"]: adapter_plan(args, a, plan, near_dup_of) for a in adapters}
    tokenizer = None
    if args.dry_run and args.length_guard == "line" and any(m in INSTRUCTION_MODES for m in args.mode_list):
        try:
            tokenizer = C.load_tokenizer()
        except Exception as exc:  # noqa: BLE001 - informative only
            warnings.append(f"dry run: tokenizer unavailable ({exc}); length caps not shown")
    caps = []
    if tokenizer is not None:
        for take in plan:
            if take["mode"] in INSTRUCTION_MODES:
                kwargs, meta = conditioning(take, rules, True)
                _, info = length_guard(tokenizer, kwargs["text"], meta["target_text_spoken"], args.length_guard)
                _, official = length_guard(tokenizer, kwargs["text"], meta["target_text_spoken"], "full")
                caps.append((info["max_seconds"], official["max_seconds"]))

    def est_minutes(takes: list[dict]) -> float:
        return round(sum(max(1.0, len(t["text"]) / 14.0) * 1.2 + 1.5 for t in takes) / 60, 1)

    groups = {a["label"]: run_groups(args, a, per_adapter[a["label"]][0]) for a in adapters}
    limit = args.limit_takes or 10**9
    summary = {
        "lines": len({t["line_id"] for t in plan}),
        "takes_in_plan": len(plan),
        "adapters": [{"label": a["label"], "kind": a["kind"], "path": str(a["path"]) if a["path"] else None,
                      "lora_scale": a.get("lora_scale"), "identity_clause": clause_for(args, a),
                      "provenance": a.get("provenance"), "train_manifest": a.get("train_manifest"),
                      "takes": sum(min(len(t), limit) for _, _, t in groups[a["label"]]),
                      "runs": {label: min(len(t), limit) for label, _, t in groups[a["label"]]},
                      "skipped_heldout_lines_it_trained_on": per_adapter[a["label"]][1]} for a in adapters],
        "run_folders": len([1 for a in adapters for _ in groups[a["label"]]]),
        "modes": args.mode_list,
        "split_modes": args.split_modes,
        "cfg_value": args.cfg,
        "instruction_variant": args.instruction_variant,
        "identity_clips": sorted({t["identity"]["id"] for t in plan if t["mode"] in SINGLE_REF_MODES
                                  or args.identity_ref == "natural"}),
        "length_guard": ({"mode": args.length_guard,
                          "instruction_take_cap_seconds": [min(c[0] for c in caps), max(c[0] for c in caps)],
                          "official_cap_seconds_for_the_same_takes": [min(c[1] for c in caps),
                                                                      max(c[1] for c in caps)]}
                         if caps else {"mode": args.length_guard}),
        "output": args.output,
        "respelled_terms": [r[0] for r in rules],
        "references": sorted({t["ref"]["id"] for t in plan}),
        "suspect_references": sorted({c["id"] for t in plan for c in (t["ref"], t["identity"])
                                      if c.get("screen") == "suspect"}),
        "estimated_gpu_minutes": round(sum(est_minutes(t[: args.limit_takes or None])
                                           for a in adapters for _, _, t in groups[a["label"]]), 1),
        "host_memory_now": C.host_memory(),
        "host_load_need_gb": round(C.HOST_LOAD_GB["render"] + args.host_margin_gb, 1),
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("dry run: inputs validated; no model loaded, nothing written.")
        return
    for adapter in adapters:
        for label, _, _ in groups[adapter["label"]]:
            manifest_path = args.out_root / label / "manifest.json"
            if manifest_path.exists() and not args.resume:
                raise SystemExit(f"{manifest_path} exists; pass --resume to continue it or change --run-prefix")

    import torch

    if args.device == "cuda":
        _, gpu = C.gpu_budget(args.max_vram_gb, args.reserve_gb, C.RENDER_NEED_GB)
    else:
        torch.set_num_threads(args.cpu_threads)
        gpu = {"device": "cpu", "threads": args.cpu_threads}
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm2 import LoRAConfig

    model = None
    model_signature = None
    lora_tensor_count = 0
    host = None
    for adapter in adapters:
        signature = lora_signature(adapter["lora_config"])
        if model is None or signature != model_signature:
            if model is not None:
                del model
                model = None
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            host = C.host_ram_budget(C.HOST_LOAD_GB["render"], args.host_margin_gb, "VoxCPM2 load for rendering")
            t0 = time.perf_counter()
            model = VoxCPM.from_pretrained(hf_model_id=str(C.SNAPSHOT), load_denoiser=False, optimize=args.optimize,
                                           device=args.device, lora_config=LoRAConfig(**adapter["lora_config"]),
                                           local_files_only=True)
            model_signature = signature
            lora_tensor_count = sum(1 for n, _ in model.tts_model.named_parameters() if "lora_" in n)
            host = {**host, "process_after_load": C.process_commit()}
            print(f"model loaded in {time.perf_counter() - t0:.1f}s ({lora_tensor_count} LoRA tensors); "
                  f"host commit {json.dumps(host['process_after_load'])}", flush=True)
        if adapter["kind"] == "none":
            model.unload_lora()
            model.set_lora_enabled(False)
            weights_sha = None
        else:
            loaded, skipped = model.load_lora(str(adapter["path"]))
            if skipped or len(loaded) != lora_tensor_count:
                raise SystemExit(f"{adapter['label']}: loaded {len(loaded)}/{lora_tensor_count} LoRA tensors, "
                                 f"skipped {len(skipped)}")
            model.set_lora_enabled(True)
            if adapter["lora_scale"] != 1.0:
                set_lora_scale(model, adapter["lora_scale"])
            weights_sha = C.sha256_file(adapter["weights"])
        skipped_lines = per_adapter[adapter["label"]][1]
        for label, mode, takes in groups[adapter["label"]]:
            render_adapter(args, model, adapter, label, mode, takes, skipped_lines, rules, {**gpu, "host": host},
                           weights_sha, torch)
    if args.device == "cuda":
        print(f"peak reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB", flush=True)


def render_adapter(args, model, adapter: dict, run_label: str, mode: str | None, plan: list[dict],
                   skipped_lines: list[dict], rules, gpu: dict, weights_sha, torch) -> None:
    out_dir = args.out_root / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    clause = clause_for(args, adapter)
    settings = {"modes": [mode] if mode else args.mode_list, "seeds": args.seeds, "refs_per_line": args.refs_per_line,
                "cfg_value": args.cfg, "inference_timesteps": args.timesteps, "ref_audio": args.ref_audio,
                "ref_trimmed": not args.no_ref_trim, "identity_ref": args.identity_ref,
                "identity_label": args.identity_label, "identity_clause": clause,
                "instruction_variant": args.instruction_variant, "pronunciations": args.pronunciations,
                "optimize": args.optimize, "output": args.output, "length_guard": args.length_guard}
    manifest = {"takes": [], "failed": []}
    if manifest_path.exists() and args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = manifest.get("settings") or {}
        diffs = {k: [old.get(k), settings[k]] for k in RESUME_KEYS if old.get(k) != settings[k]}
        old_adapter = manifest.get("adapter") or {}
        if old_adapter.get("weights_sha256") != weights_sha:
            diffs["weights_sha256"] = [old_adapter.get("weights_sha256"), weights_sha]
        if old_adapter.get("lora_scale") != adapter.get("lora_scale"):
            diffs["lora_scale"] = [old_adapter.get("lora_scale"), adapter.get("lora_scale")]
        if diffs:
            raise SystemExit(f"--resume: {manifest_path} was rendered with other settings {json.dumps(diffs)}; "
                             "use a new --run-prefix instead of mixing takes")
        manifest["takes"] = [t for t in manifest.get("takes", []) if (out_dir / t["wav"]).exists()]
        manifest["failed"] = []
    done = {t["take_id"] for t in manifest["takes"]}
    rate_out = C.OUTPUT_SR
    step = re.search(r"step_(\d+)", str(adapter["path"] or ""))
    tokenizer = model.tts_model.text_tokenizer if args.length_guard == "line" else None
    manifest.update({
        "engine": "voxcpm",
        "run": run_label,
        "adapter_label": adapter["label"],
        "mode": mode,
        "adapter": {"kind": adapter["kind"], "path": str(adapter["path"]) if adapter["path"] else None,
                    "lora_config": adapter["lora_config"], "weights_sha256": weights_sha,
                    "lora_scale": adapter.get("lora_scale"), "train_audio_variant": adapter.get("train_variant"),
                    "train_manifest": adapter.get("train_manifest"), "provenance": adapter.get("provenance"),
                    "base_model_recorded": adapter.get("base_model")},
        "base_snapshot": C.SNAPSHOT_ID,
        "settings": settings,
        "skipped_heldout_lines_it_trained_on": skipped_lines,
        "gpu": gpu,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    rendered = 0
    for take in plan:
        if args.limit_takes and rendered >= args.limit_takes:
            break
        if take["take_id"] in done:
            continue
        kwargs, meta = conditioning(take, rules, clause)
        guard_kwargs, guard_info = length_guard(tokenizer, kwargs["text"], meta["target_text_spoken"],
                                                args.length_guard)
        seed = int(take["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        t0 = time.perf_counter()
        try:
            audio = model.generate(cfg_value=args.cfg, inference_timesteps=args.timesteps, normalize=False,
                                   denoise=False, **guard_kwargs, **kwargs)
        except torch.cuda.OutOfMemoryError:
            manifest["failed"].append({"take_id": take["take_id"], "error": "CUDA out of memory (allocator cap)"})
            C.atomic_write_json(manifest_path, manifest)
            raise SystemExit(f"{take['take_id']}: CUDA OOM under the allocator cap; stopped (manifest saved)")
        except Exception as exc:  # keep the batch going; the failure is recorded
            manifest["failed"].append({"take_id": take["take_id"], "error": f"{type(exc).__name__}: {exc}",
                                       "trace": traceback.format_exc(limit=3)})
            C.atomic_write_json(manifest_path, manifest)
            print(f"FAILED {take['take_id']}: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - t0
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        rate = int(model.tts_model.sample_rate)
        if rate != rate_out:
            audio = C.resample(audio, rate, rate_out)
        raw_seconds = len(audio) / rate_out
        trimmed, trim = trim_output(audio, rate_out)
        wav_name = f"{take['take_id']}.wav"
        stats = C.write_wav_pcm16(out_dir / wav_name, trimmed if args.output == "trimmed" else audio, rate_out)
        extra = {}
        if args.output == "both":
            extra["trimmed_output_wav"] = f"{take['take_id']}.trimmed.wav"
            C.write_wav_pcm16(out_dir / extra["trimmed_output_wav"], trimmed, rate_out)
        heldout = take["heldout_id"]
        entry = {
            "take_id": take["take_id"],
            "engine": "voxcpm",
            "line_id": take["line_id"],
            "text": take["text"],
            "delivery": take["delivery"],
            "wav": wav_name,
            "reference_id": meta["prompt_id"] or meta["identity_ref_id"],
            "seed": seed,
            "params": {
                **meta,
                "adapter": adapter["label"],
                "adapter_kind": adapter["kind"],
                "lora_path": str(adapter["path"]) if adapter["path"] else None,
                "lora_step": int(step.group(1)) if step else None,
                "lora_sha256": weights_sha,
                "lora_scale": adapter.get("lora_scale"),
                "base_snapshot": C.SNAPSHOT_ID,
                "cfg_value": args.cfg,
                "inference_timesteps": args.timesteps,
                "ref_audio_variant": args.ref_audio,
                "ref_trimmed": not args.no_ref_trim,
                "length_guard": {"mode": args.length_guard, **guard_info},
                "gen_seconds": round(elapsed, 2),
                "rtf": round(elapsed / max(raw_seconds, 1e-3), 3),
                "raw_seconds": round(raw_seconds, 3),
                "output_trimmed": args.output == "trimmed",
                "output_trim_s": trim,
                **extra,
                **stats,
                # held-out lines an adapter trained on are skipped; with --keep-trained-heldout they are rendered
                # renamed (<line>--adapter-trained, no heldout_id) and flagged here
                "adapter_trained_on_heldout": bool(take.get("trained_line")),
                "adapter_trained_on_this_line": bool(take.get("trained_line")),
                "device": args.device,
            },
        }
        if heldout:
            entry["heldout_id"] = heldout
        manifest["takes"].append(entry)
        manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        C.atomic_write_json(manifest_path, manifest)
        rendered += 1
        print(f"[{run_label}] {take['take_id']}: {stats['duration_s']:.2f}s in {elapsed:.1f}s", flush=True)
    print(f"[{run_label}] {rendered} new takes, {len(manifest['failed'])} failed, "
          f"{len(skipped_lines)} held-out lines skipped (trained on) -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
