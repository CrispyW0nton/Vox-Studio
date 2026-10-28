"""Render a voice-lab character's takes with IndexTTS-2.5 (GPU step): an emotion-controllable performance donor.

The character comes from voice-lab\\characters (VOICE_LAB_CHARACTER=<slug>; unset = baodur, Bao-Dur); <lab> below is
its data folder (characters\\<slug>.json "lab", e.g. BaoDurLab or JoleeLab).
  cd C:\\Users\\NewAdmin\\Documents\\GDeveloper\\External\\voice-lab\\baodur-lab\\indextts
  set PY=C:\\Users\\NewAdmin\\Documents\\GDeveloper\\External\\voice-lab\\index-tts\\.venv\\Scripts\\python.exe
  %PY% render.py --dry-run                                    (CPU only: validates inputs/references, prints the
                                                               plan, VRAM + host-memory estimates; loads no model)
  %PY% render.py --modes spk_ref,id_emo_ref,id_emo_vec --seeds 3          (first comparison)
  %PY% render.py --modes id_emo_ref --ref-alpha 0.3 0.5 0.7 --seeds 2     (V03 alpha audition, one folder per alpha)
  %PY% render.py --modes id_emo_text --seeds 2                           (loads QwenEmotion: +1.2 GB VRAM)
  add --lines <id>,<id> --limit-takes 2 for a first GPU smoke; --resume continues/extends runs.
  (another character: set VOICE_LAB_CHARACTER=jolee first; everything below then resolves under JoleeLab)

Inputs: <lab>\\delivery\\takes_script.json {"lines":[{id,text,delivery,source,heldout_id?}]},
<lab>\\delivery\\delivery_map.json {"deliveries":{label:{references:[{id,text,duration,why}],...}}}, corpus.jsonl,
the speaker screen and the character's emotion controls (<lab>\\indextts\\emotion_controls.json). Output: one folder
per mode (and per emotion strength) under <lab>\\takes\\indextts25\\
<run>\\manifest.json in the lab's shared takes format {"takes":[{take_id,engine:"indextts25",line_id,text,delivery,
wav,reference_id,seed,params,heldout_id?}]} with 48 kHz mono PCM16 WAVs beside it (IndexTTS renders 22.05 kHz;
resampled with soxr VHQ, never upsampled content). The manifest is rewritten atomically after every take.

Conditioning modes (--modes; IndexTTS-2.5 infer_v2_5.IndexTTS2.infer, lang="EN"):
  spk_ref      speaker prompt = the delivery's own reference clip; no separate emotion input, so IndexTTS also
               takes the emotion from that clip (timbre + delivery from one clip; the baseline).
  id           speaker prompt = ONE fixed identity clip (top usable --identity-label clip; default = the character's
               base delivery, common.base_delivery: Bao-Dur natural -> gblbaodur248), no emotion input: the control
               for the id_* modes.
  id_emo_ref   identity clip as speaker + the delivery's reference clip as emo_audio_prompt, mixed by emo_alpha
               (--ref-alpha, list; V03: audition 0.3/0.5/0.7). Timbre and emotion come from different clips.
  id_emo_vec   identity clip + the delivery's 8-dim emotion vector [happy, angry, sad, afraid, disgusted,
               melancholic, surprised, calm] (the character's emotion_controls.json, --emo-config merged over it),
               passed through the WebUI's normalize_emo_vec bias (--no-emo-bias to skip) and scaled by emo_alpha
               (--vec-alpha, list). Bao-Dur's vectors are understated audition values (V03), not measured mappings;
               his base delivery natural = all zero (identical conditioning to `id`).
  id_emo_text  identity clip + QwenEmotion (bundled Qwen3-0.6B) turning the delivery's emotion description
               (--emo-text-source description, default) or the line itself (line) into the vector, scaled by
               emo_alpha (--text-alpha, list; V03: start 0.4, upstream <= 0.6). The detected vector is recorded.
use_random stays False (upstream: random emotion exemplars reduce cloning fidelity). Each line x mode x alpha x
reference slot (--refs-per-line; spk_ref and id_emo_ref only) x seed (--seeds; seeds depend only on line id and
index, so every mode/alpha gets the same seeds). Sampling defaults are upstream's (do_sample, top_p 0.8, top_k 30,
temperature 0.8, num_beams 3, repetition_penalty 10, max_mel_tokens 1500); duration_factor = --duration-factor x the
label's factor (all 1.0 by default).

References: ORIGINAL game audio only (user decision 2026-09-30; --ref-audio restored is refused unless
--allow-restored-audio). TRAIN-split clips only by default (validation clips need --allow-val-refs; V03 keeps val
clips out of reference pools). Held-out TEST clips are refused outright, never read; for held-out lines, clips that
speak the same line or are screen near-duplicates of it are skipped (speaker and emotion reference alike). Speaker
screen: a 'confirmed' other-character clip stops the run, 'uncertain' ones are skipped, 'suspect' ones are used and
tagged params.reference_screen="suspect". Clips are silence-trimmed (common.trim_silence) at their native rate into
<lab>\\indextts\\refs\\original\\<id>.wav (--no-ref-trim uses the source files).

GPU (shared with Substance Painter and other sessions): the process refuses to start unless it can stay under
--max-vram-gb in total while leaving --reserve-gb free, then caps its allocator (torch.cuda.
set_per_process_memory_fraction). Estimate (common.vram_estimate, +/-25%, from the measured parameter counts):
~7.6 GB total with the bf16 GPT (5.05 GB weights), ~8.9 GB with QwenEmotion, ~9.8 GB with --no-bf16; upstream's
model card says ~6 GB. Measured peaks go into each manifest. Host RAM: IndexTTS2 builds every part on the CPU first
(~10 GB commit peak estimated); the load is refused unless that much plus --host-margin-gb is free.
Offline: HF/ModelScope offline variables + a socket guard; a missing file fails loudly instead of downloading.

License: IndexTTS-2.5 is under the bilibili Model Use License Agreement; generated audio and anything made from it
(Seed-VC conversions included) are "Derivative Works". See voice-lab\\index-tts\\LICENSE_NOTES.md before
distributing anything rendered here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODES = ("spk_ref", "id", "id_emo_ref", "id_emo_vec", "id_emo_text")
DELIVERY_REF_MODES = {"spk_ref", "id_emo_ref"}  # use the delivery's reference clips (one slot per --refs-per-line)
ALPHA_MODES = {"id_emo_ref": "ref_alpha", "id_emo_vec": "vec_alpha", "id_emo_text": "text_alpha"}
RESUME_KEYS = ("mode", "emo_alpha", "ref_audio", "ref_trimmed", "identity_label", "identity_ref", "allow_val_refs",
               "lang", "text_normalization", "generation", "duration_factor", "emo_bias", "emo_text_source",
               "emotion_controls_sha256", "bf16", "hf_revision", "code_commit", "output", "pronunciations_sha256")

# Per-delivery emotion controls are character content: <lab>\indextts\emotion_controls.json (C.EMOTION_CONTROLS_FILE;
# vector order = C.EMO_ORDER; text = the English emotion description QwenEmotion classifies in id_emo_text mode, no
# timbre words). Bao-Dur's file holds the table this script used to carry (Astra V03 audition values).


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", default="spk_ref,id_emo_ref,id_emo_vec", help=f"comma list from {MODES}")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--refs-per-line", type=int, default=1, help="delivery reference slots for spk_ref/id_emo_ref")
    p.add_argument("--ref-alpha", type=float, nargs="+", default=[0.7], help="emo_alpha values for id_emo_ref")
    p.add_argument("--vec-alpha", type=float, nargs="+", default=[1.0], help="emo_alpha values for id_emo_vec")
    p.add_argument("--text-alpha", type=float, nargs="+", default=[0.4], help="emo_alpha values for id_emo_text")
    p.add_argument("--emo-config", type=Path, default=None,
                   help='JSON {"labels": {label: {"vector": [8 floats], "text": str, "duration_factor": f}}} '
                        f"merged over the character's {C.EMOTION_CONTROLS_FILE.name} ({C.EMOTION_CONTROLS_FILE})")
    p.add_argument("--no-emo-bias", action="store_true",
                   help="pass id_emo_vec vectors raw instead of through normalize_emo_vec (WebUI behaviour)")
    p.add_argument("--emo-text-source", choices=("description", "line"), default="description")
    p.add_argument("--identity-label", default="",
                   help="delivery whose top usable clip is the identity clip (default: the character's base delivery, "
                        "common.base_delivery: Bao-Dur 'natural')")
    p.add_argument("--identity-ref", default="", help="explicit identity clip id (train split) instead")
    p.add_argument("--allow-val-refs", action="store_true", help="also use validation-split clips as references")
    p.add_argument("--script", type=Path, default=C.TAKES_SCRIPT)
    p.add_argument("--delivery-map", type=Path, default=C.DELIVERY_MAP)
    p.add_argument("--lines", default="", help="comma-separated line ids (default all)")
    p.add_argument("--limit-takes", type=int, default=0, help="stop after N new takes per run folder (smoke tests)")
    p.add_argument("--ref-audio", choices=C.AUDIO_VARIANTS, default="original")
    p.add_argument("--allow-restored-audio", action="store_true",
                   help="allow --ref-audio restored (against the user's 2026-09-30 decision)")
    p.add_argument("--no-ref-trim", action="store_true", help="condition on the untrimmed source clips")
    p.add_argument("--pronunciations", type=Path, default=None,
                   help='JSON {"term": "CMU PHONES"}: wraps each term as <term|PHONES> (IndexTTS-2.5 annotation)')
    p.add_argument("--no-text-normalization", action="store_true", help="skip IndexTTS's wetext normalizer")
    p.add_argument("--duration-factor", type=float, default=1.0, help="global speed factor (>1 slower; 0.5-2.0)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--repetition-penalty", type=float, default=10.0)
    p.add_argument("--max-mel-tokens", type=int, default=1500)
    p.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 GPT (upstream recommendation for 2.5; --no-bf16 = fp32, +1.6 GB VRAM)")
    p.add_argument("--run-prefix", default="", help="prefix for run folder names")
    p.add_argument("--out-root", type=Path, default=C.TAKES_ROOT)
    p.add_argument("--output", choices=("raw", "trimmed", "both"), default="raw")
    p.add_argument("--max-vram-gb", type=float, default=11.0, help="hard cap on this process's total VRAM")
    p.add_argument("--reserve-gb", type=float, default=8.0, help="VRAM that must stay free for other apps")
    p.add_argument("--host-margin-gb", type=float, default=C.HOST_MARGIN_GB)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--allow-cpu", action="store_true", help="permit --device cpu (very slow, ~10 GB RAM)")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--no-net-guard", action="store_true", help="do not block network sockets (debug only)")
    p.add_argument("--resume", action="store_true", help="continue existing run folders")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in args.mode_list if m not in MODES]
    if bad or not args.mode_list:
        p.error(f"unknown/empty modes {bad}; choose from {MODES}")
    if args.seeds < 1 or args.refs_per_line < 1:
        p.error("--seeds and --refs-per-line must be >= 1")
    for name in ("ref_alpha", "vec_alpha", "text_alpha"):
        values = getattr(args, name)
        if any(not 0.0 <= v <= 1.0 for v in values) or len(set(values)) != len(values):
            p.error(f"--{name.replace('_', '-')} values must be distinct and within [0, 1]")
    if not 0.5 <= args.duration_factor <= 2.0:
        p.error("--duration-factor must be within 0.5-2.0 (upstream range)")
    if args.ref_audio == "restored" and not args.allow_restored_audio:
        p.error(f"--ref-audio restored is off: {C.RESTORED_DECISION}. Pass --allow-restored-audio only if the user "
                "reverses that decision.")
    if args.device == "cpu" and not (args.allow_cpu or args.dry_run):
        p.error("--device cpu loads ~8 GB of weights into RAM and runs for minutes per take on this shared machine; "
                "pass --allow-cpu if that is really intended")
    # absolute paths: the GPU path chdirs into the IndexTTS checkout before importing it
    for name in ("script", "delivery_map", "emo_config", "pronunciations", "out_root"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve())
    args.identity_label = (args.identity_label or C.base_delivery(args.delivery_map)).casefold()
    return args


# --------------------------------------------------------------------------------------------- emotion controls
def load_emotion_controls(path: Path | None) -> dict[str, dict]:
    """The character's emotion controls (C.EMOTION_CONTROLS_FILE) with the --emo-config file merged over them."""
    return C.load_emotion_controls(path)


def bias_vector(vec: list[float], apply_bias: bool) -> list[float]:
    """infer_v2_5.IndexTTS2.normalize_emo_vec (what the WebUI applies to slider vectors), reproduced so the dry run
    can show it without the model; render() cross-checks it against the loaded model's own method."""
    if apply_bias:
        bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
        vec = [v * b for v, b in zip(vec, bias)]
    total = sum(vec)
    if total > 0.8:
        vec = [v * 0.8 / total for v in vec]
    return [round(v, 6) for v in vec]


# --------------------------------------------------------------------------------------------- plan
def run_specs(args) -> list[dict]:
    """(run label, mode, emo_alpha) for every run folder, in render order."""
    specs = []
    for mode in args.mode_list:
        alphas = getattr(args, ALPHA_MODES[mode]) if mode in ALPHA_MODES else [None]
        for alpha in alphas:
            tag = f"-a{alpha:g}" if alpha is not None else ""
            specs.append({"label": f"{args.run_prefix}{mode}{tag}", "mode": mode, "alpha": alpha, "tag": tag})
    labels = [s["label"] for s in specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate run labels {labels}")
    return specs


def build_plan(args, corpus: dict[str, dict], screen: dict, controls: dict) -> tuple[list[dict], list[str], dict]:
    warnings: list[str] = []
    noted: set[str] = set()
    script = json.loads(args.script.read_text(encoding="utf-8"))
    deliveries = {k.casefold(): v for k, v in json.loads(args.delivery_map.read_text(encoding="utf-8"))
                  ["deliveries"].items()}
    tests = {i for i, r in corpus.items() if r["split"] == "test"}
    wanted = {x.strip() for x in args.lines.split(",") if x.strip()}
    # the corpus clip ids each near-duplicate reason names (e.g. "... with held-out gblbaodur276")
    id_pattern = C.test_id_pattern(set(corpus)) if corpus else None
    near_dup_of = {cid: ({m.group(0) for m in id_pattern.finditer(reason)} if id_pattern else set()) - {cid}
                   for cid, reason in screen["near_duplicate_of_test"].items()}

    def warn_once(key: str, message: str) -> None:
        if key not in noted:
            warnings.append(message)
            noted.add(key)

    def check_clip(rid: str, context: str, heldout_text: str | None, heldout_id: str | None,
                   ref_text: str | None = None) -> dict | None:
        row = corpus.get(rid)
        if row is None:
            warn_once(f"x:{context}:{rid}", f"{context}: reference {rid} is not in the corpus; skipped")
            return None
        if rid in tests or row["split"] == "test":
            raise SystemExit(f"{context} uses held-out test clip {rid} as a reference")
        if rid in screen["confirmed"]:
            raise SystemExit(f"{context} uses {rid} as a reference, but that clip is {screen['confirmed'][rid]} "
                             f"speaking (speaker screen), not {C.CHAR['display_name']}")
        if rid in screen["uncertain"]:
            warn_once(f"u:{context}:{rid}", f"{context}: reference {rid} is speaker-'uncertain'; skipped")
            return None
        if row["split"] != "train" and not args.allow_val_refs:
            warn_once(f"v:{context}:{rid}", f"{context}: reference {rid} is a {row['split']} clip; skipped "
                      "(train-only references; --allow-val-refs to include)")
            return None
        if row["kind"] != "speech":
            warn_once(f"n:{context}:{rid}", f"{context}: reference {rid} is nonverbal; skipped")
            return None
        text = C.clean_text(ref_text or row["text"])
        if heldout_text and (C.overlaps_text(text, heldout_text) or C.overlaps_text(heldout_text, text)
                             or (heldout_id and heldout_id.casefold()
                                 in {x.casefold() for x in near_dup_of.get(rid, ())})):
            warn_once(f"h:{context}:{rid}:{heldout_id}",
                      f"{context}: reference {rid} speaks held-out line {heldout_id}; skipped for that line")
            return None
        src = C.source_audio(row, args.ref_audio)
        if not src.exists():
            warn_once(f"p:{context}:{rid}", f"{context}: audio for {rid} not found at {src}; skipped")
            return None
        return {"id": rid, "text": text, "row": row, "source": src, "duration": row["duration"],
                "screen": "suspect" if rid in screen["suspect"] else None}

    def delivery_refs(label: str, heldout_text: str | None, heldout_id: str | None) -> list[dict]:
        out = []
        for ref in deliveries.get(label, {}).get("references", []):
            clip = check_clip(str(ref.get("id", "")), f"delivery '{label}'", heldout_text, heldout_id, ref.get("text"))
            if clip is not None:
                out.append(clip)
        return out

    if not args.identity_ref and args.identity_label not in deliveries:
        raise SystemExit(f"--identity-label {args.identity_label} is not a delivery in {args.delivery_map}")
    plan: list[dict] = []
    specs = run_specs(args)
    base = C.base_delivery(args.delivery_map)  # the character's neutral delivery (Bao-Dur: natural)
    for line in script.get("lines", []):
        line_id = str(line["id"])
        if wanted and line_id not in wanted:
            continue
        label = str(line.get("delivery") or base).casefold()
        if label not in deliveries:
            warnings.append(f"line {line_id}: delivery '{label}' not in the delivery map; using {base}")
            label = base
        if label not in controls:
            raise SystemExit(f"line {line_id}: no emotion control for delivery '{label}' (add it to "
                             f"{C.EMOTION_CONTROLS_FILE} or via --emo-config)")
        heldout = line.get("heldout_id") if line.get("source") == "heldout" else None
        if line.get("source") == "heldout" and not heldout:
            warnings.append(f"line {line_id}: source heldout without heldout_id")
        if heldout and heldout not in tests:
            warnings.append(f"line {line_id}: heldout_id {heldout} is not a test clip")
        text = str(line["text"])
        htext = text if heldout else None
        refs = delivery_refs(label, htext, heldout)
        if args.identity_ref:
            ident = check_clip(args.identity_ref, "--identity-ref", htext, heldout)
            identity_pool = [ident] if ident else []
        else:
            identity_pool = delivery_refs(args.identity_label, htext, heldout)
        identity = identity_pool[0] if identity_pool else None
        for spec in specs:
            mode = spec["mode"]
            if mode != "spk_ref" and identity is None:
                warnings.append(f"line {line_id}: no usable identity clip; {spec['label']} skipped")
                continue
            if mode in DELIVERY_REF_MODES:
                slots = refs[: args.refs_per_line]
                if mode == "id_emo_ref" and identity is not None:
                    # an emotion reference identical to the identity clip adds nothing: take the next clip
                    distinct = [r for r in refs if r["id"] != identity["id"]]
                    slots = (distinct or refs)[: args.refs_per_line]
                if not slots:
                    warnings.append(f"line {line_id}: no usable reference for '{label}'; {spec['label']} skipped")
                    continue
            else:
                slots = [None]
            for ri, ref in enumerate(slots):
                spk = ref if mode == "spk_ref" else identity
                emo = ref if mode == "id_emo_ref" else None
                for k in range(args.seeds):
                    plan.append({
                        "take_id": f"{line_id}-{mode}{spec['tag']}-r{ri}-s{k}", "line_id": line_id, "text": text,
                        "delivery": label, "heldout_id": heldout, "run": spec["label"], "mode": mode,
                        "alpha": spec["alpha"], "spk": spk, "emo": emo, "seed": C.stable_seed("indextts25", line_id, k),
                    })
    return plan, warnings, near_dup_of


def audit_plan(plan: list[dict], corpus: dict, screen: dict, near_dup_of: dict) -> None:
    """Belt and braces: no take may be conditioned on a test clip, a screened clip, a near-duplicate of its own
    held-out line, or audio outside the corpus folders."""
    tests = {i for i, r in corpus.items() if r["split"] == "test"}
    screened = set(screen["confirmed"]) | screen["uncertain"]
    base = C.test_id_pattern(tests | screened)
    allowed = {Path(r["audio"]).resolve().parent for r in corpus.values() if r["split"] != "test"}
    allowed.add(C.RESTORED_DIR.resolve())
    for take in plan:
        pattern = base
        if take["heldout_id"]:
            dups = {cid for cid, targets in near_dup_of.items()
                    if take["heldout_id"].casefold() in {t.casefold() for t in targets}}
            pattern = C.test_id_pattern(tests | screened | dups)
        for role in ("spk", "emo"):
            clip = take[role]
            if clip is None:
                continue
            hits = [m.group(0) for m in pattern.finditer(f"{clip['id']} {clip['source']}")]
            if hits:
                raise SystemExit(f"take {take['take_id']} would be conditioned on held-out/screened clip(s) {hits}")
            if Path(clip["source"]).resolve().parent not in allowed:
                raise SystemExit(f"take {take['take_id']}: {role} audio {clip['source']} is not a lab corpus clip")


def load_pronunciations(path: Path | None) -> list[tuple[str, str, re.Pattern]]:
    if path is None:
        return []
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = []
    for term, phones in sorted(doc.items(), key=lambda kv: -len(kv[0])):
        if not re.fullmatch(r"[A-Z]{1,3}[0-2]?( [A-Z]{1,3}[0-2]?| \.)*", str(phones).strip()):
            raise SystemExit(f"--pronunciations: '{term}' -> '{phones}' is not a CMU phone string (e.g. 'M AE1 L')")
        rules.append((term, str(phones).strip(),
                      re.compile(r"(?<![A-Za-z0-9<|])" + re.escape(term) + r"(?![A-Za-z0-9|>])", re.IGNORECASE)))
    return rules


def synth_text(text: str, rules) -> tuple[str, list[str]]:
    out = C.clean_text(text)
    used = []
    for term, phones, pattern in rules:
        out, count = pattern.subn(lambda m: f"<{m.group(0)}|{phones}>", out)
        if count:
            used.append(term)
    return out, used


# --------------------------------------------------------------------------------------------- audio
def trim_output(audio: np.ndarray, rate: int, lead_pad: float = 0.04, tail_pad: float = 0.12,
                fade_ms: float = 8.0) -> tuple[np.ndarray, list[float]]:
    """Vox-style end trim (threshold 4% of the 90th-percentile frame RMS), same as voxcpm/render.py."""
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
def settings_for(args, spec: dict, controls_sha: str, pron_sha: str | None) -> dict:
    return {
        "mode": spec["mode"], "emo_alpha": spec["alpha"], "seeds": args.seeds, "refs_per_line": args.refs_per_line,
        "ref_audio": args.ref_audio, "ref_trimmed": not args.no_ref_trim, "identity_label": args.identity_label,
        "identity_ref": args.identity_ref or None, "allow_val_refs": args.allow_val_refs, "lang": "EN",
        "text_normalization": not args.no_text_normalization,
        "generation": {"do_sample": True, "top_p": args.top_p, "top_k": args.top_k, "temperature": args.temperature,
                       "length_penalty": 0.0, "num_beams": args.num_beams,
                       "repetition_penalty": args.repetition_penalty, "max_mel_tokens": args.max_mel_tokens,
                       "max_text_tokens_per_segment": args.max_text_tokens_per_segment, "interval_silence": 200},
        "duration_factor": args.duration_factor, "emo_bias": not args.no_emo_bias,
        "emo_text_source": args.emo_text_source, "emotion_controls_sha256": controls_sha, "bf16": args.bf16,
        "use_random": False, "hf_revision": C.HF_REVISION, "code_commit": C.CODE_COMMIT, "output": args.output,
        "pronunciations_sha256": pron_sha,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or args.dry_run:  # "-1", not "": on Windows os.environ[...] = "" deletes the variable
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    C.offline_env()
    blocked = [] if args.no_net_guard else C.net_guard()
    rows = C.load_corpus()
    corpus = {r["id"]: r for r in rows}
    for path in (args.script, args.delivery_map):
        if not path.exists():
            raise SystemExit(f"missing input {path} (the delivery step writes it)")
    screen = C.load_speaker_screen()
    controls = load_emotion_controls(args.emo_config)
    controls_sha = hashlib.sha256(json.dumps(controls, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    rules = load_pronunciations(args.pronunciations)
    pron_sha = C.sha256_file(args.pronunciations)[:16] if args.pronunciations else None
    plan, warnings, near_dup_of = build_plan(args, corpus, screen, controls)
    audit_plan(plan, corpus, screen, near_dup_of)
    specs = run_specs(args)
    needs_qwen = any(s["mode"] == "id_emo_text" for s in specs)
    estimate = C.vram_estimate(bf16=args.bf16, qwen_emo=needs_qwen, max_text_tokens=args.max_text_tokens_per_segment,
                               max_mel_tokens=args.max_mel_tokens, num_beams=args.num_beams)
    limit = args.limit_takes or 10**9
    per_run = {s["label"]: [t for t in plan if t["run"] == s["label"]] for s in specs}
    missing_files = [str(p) for p in _required_files() if not p.exists()] + C.verify_sizes()
    summary = {
        "engine": C.ENGINE,
        "model": f"{C.HF_REPO}@{C.HF_REVISION[:8]} (code {C.CODE_COMMIT[:8]})",
        "lines": len({t["line_id"] for t in plan}),
        "takes_in_plan": len(plan),
        "runs": {label: min(len(t), limit) for label, t in per_run.items()},
        "identity_clips": sorted({t["spk"]["id"] for t in plan if t["mode"] != "spk_ref"}),
        "delivery_references": sorted({(t["emo"] or t["spk"])["id"] for t in plan if t["mode"] in DELIVERY_REF_MODES}),
        "suspect_references": sorted({c["id"] for t in plan for c in (t["spk"], t["emo"]) if c and c["screen"]}),
        "reference_audio": f"{args.ref_audio}, {'trimmed copies in ' + str(C.REF_CACHE) if not args.no_ref_trim else 'source files'}",
        "emotion_controls": {k: v for k, v in controls.items() if k in {t["delivery"] for t in plan}},
        "vram_estimate_gb": estimate,
        "gpu_policy": {"max_vram_gb": args.max_vram_gb, "reserve_gb": args.reserve_gb,
                       "needs_cap_at_least_gb": round(estimate["total_gb"] + C.gpu_headroom_gb(estimate["total_gb"]), 1)},
        "host_load_need_gb": C.HOST_LOAD_GB["render_qwen" if needs_qwen else "render"] + args.host_margin_gb,
        "host_memory_now": C.host_memory(),
        "estimated_gpu_minutes": round(sum(min(len(t), limit) for t in per_run.values()) * 2.5 / 60, 1),
        "model_file_problems": missing_files,
        "license": "bilibili Model Use License: rendered takes are Derivative Works (see index-tts\\LICENSE_NOTES.md)",
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing_files:
        raise SystemExit(f"model files missing or changed (see PROVENANCE.json; nothing is downloaded at runtime): "
                         f"{missing_files}")
    if args.dry_run:
        print("dry run: inputs, references and model files validated; no model loaded, nothing written.")
        return
    for spec in specs:
        manifest_path = args.out_root / spec["label"] / "manifest.json"
        if manifest_path.exists() and not args.resume:
            raise SystemExit(f"{manifest_path} exists; pass --resume to continue it or change --run-prefix")

    # reference copies (CPU, cheap) before anything touches the GPU
    ref_meta: dict[str, dict] = {}
    for take in plan:
        for clip in (take["spk"], take["emo"]):
            if clip is not None and clip["id"] not in ref_meta:
                path, meta = C.prepare_reference(clip["row"], args.ref_audio, not args.no_ref_trim)
                ref_meta[clip["id"]] = {"path": str(path), **{k: meta.get(k) for k in ("trimmed", "trim", "duration_s",
                                                                                      "source_sha256")}}

    import torch

    # host commit first (no side effects), then the GPU cap (creates the CUDA context) - both before any load
    host = C.host_ram_budget(C.HOST_LOAD_GB["render_qwen" if needs_qwen else "render"], args.host_margin_gb,
                             "IndexTTS-2.5 load")
    if args.device == "cuda":
        _, gpu = C.gpu_budget(args.max_vram_gb, args.reserve_gb, estimate["total_gb"])
        device = "cuda:0"
    else:
        torch.set_num_threads(args.cpu_threads)
        gpu, device = {"device": "cpu", "threads": args.cpu_threads}, "cpu"
    C.enter_index_root()  # infer_v2_5 sets HF_HUB_CACHE relative to the CWD at import
    from indextts.infer_v2_5 import IndexTTS2

    t0 = time.perf_counter()
    tts = IndexTTS2(cfg_path=str(C.CFG_PATH), model_dir=str(C.MODEL_DIR), use_bf16=args.bf16, device=device,
                    use_cuda_kernel=False, use_deepspeed=False, use_accel=False, use_torch_compile=False,
                    use_qwen_emo=needs_qwen)
    host = {**host, "process_after_load": C.process_commit()}
    load_s = round(time.perf_counter() - t0, 1)
    print(f"IndexTTS-2.5 loaded in {load_s}s; host commit {json.dumps(host['process_after_load'])}", flush=True)
    probe = [0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    if [round(x, 6) for x in tts.normalize_emo_vec(list(probe), apply_bias=True)] != bias_vector(probe, True):
        raise SystemExit("upstream normalize_emo_vec changed; update bias_vector() before rendering")
    if device.startswith("cuda"):
        gpu["after_load_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    text_vectors: dict[str, dict] = {}
    for spec in specs:
        render_run(args, tts, spec, per_run[spec["label"]], controls, controls_sha, pron_sha, rules, ref_meta,
                   text_vectors, {**gpu, "host": host, "model_load_s": load_s}, device, torch, blocked)
    if device.startswith("cuda"):
        print(f"peak reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB", flush=True)


def _required_files() -> list[Path]:
    paths = C.checkpoint_paths()
    return [p for key, p in paths.items() if key not in ("w2v_bert_dir", "bigvgan_dir", "qwen_emo")] + [
        paths["qwen_emo"] / "model.safetensors"]


def effective_vector(vec: list[float], alpha: float) -> list[float]:
    """What infer_generator does to an emotion vector before use: scale by emo_alpha clamped to [0, 1] and
    truncate to 4 decimals, but only when the scale is not exactly 1.0."""
    scale = max(0.0, min(1.0, alpha))
    return list(vec) if scale == 1.0 else [int(x * scale * 10000) / 10000 for x in vec]


def emotion_for(tts, take: dict, controls: dict, args, text_vectors: dict) -> dict:
    """infer() keyword arguments for the take's emotion control, plus what to record."""
    mode, alpha, label = take["mode"], take["alpha"], take["delivery"]
    out = {"kwargs": {}, "record": {"emo_alpha": alpha, "use_random": False}}
    if mode == "id_emo_ref":
        out["kwargs"] = {"emo_audio_prompt": take["emo_path"], "emo_alpha": float(alpha)}
    elif mode == "id_emo_vec":
        raw = list(controls[label]["vector"])
        applied = bias_vector(raw, not args.no_emo_bias)
        out["kwargs"] = {"emo_vector": applied, "emo_alpha": float(alpha)}
        out["record"].update(emo_vector_input=raw, emo_vector=applied, emo_bias=not args.no_emo_bias,
                             emo_vector_effective=effective_vector(applied, alpha))
    elif mode == "id_emo_text":
        emo_text = controls[label]["text"] if args.emo_text_source == "description" else C.clean_text(take["text"])
        if emo_text not in text_vectors:
            detected = tts.qwen_emo.inference(emo_text)  # greedy decoding: deterministic per text
            text_vectors[emo_text] = {k: float(v) for k, v in detected.items()}
        vec = [text_vectors[emo_text][k] for k in C.EMO_ORDER]
        # exactly what infer(use_emo_text=True, emo_text=...) does internally, but recorded
        out["kwargs"] = {"emo_vector": vec, "emo_alpha": float(alpha)}
        out["record"].update(emo_text=emo_text, emo_text_source=args.emo_text_source, emo_text_vector=vec,
                             emo_vector_effective=effective_vector(vec, alpha))
    return out


def render_run(args, tts, spec: dict, plan: list[dict], controls: dict, controls_sha: str, pron_sha, rules,
               ref_meta: dict, text_vectors: dict, gpu: dict, device: str, torch, blocked: list) -> None:
    out_dir = args.out_root / spec["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    settings = settings_for(args, spec, controls_sha, pron_sha)
    manifest = {"takes": [], "failed": []}
    if manifest_path.exists() and args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = manifest.get("settings") or {}
        diffs = {k: [old.get(k), settings[k]] for k in RESUME_KEYS if old.get(k) != settings[k]}
        if diffs:
            raise SystemExit(f"--resume: {manifest_path} was rendered with other settings {json.dumps(diffs)}; "
                             "use a new --run-prefix instead of mixing takes")
        manifest["takes"] = [t for t in manifest.get("takes", []) if (out_dir / t["wav"]).exists()]
        manifest["failed"] = []
    done = {t["take_id"] for t in manifest["takes"]}
    verified = C.verified_files()
    checkpoints = {}
    for key, path in C.checkpoint_paths().items():
        entry = verified.get(str(path).casefold())
        checkpoints[key] = {"path": str(path), "sha256": entry["sha256"] if entry else None,
                            "source": f"{entry['repo']}@{entry['revision'][:8]}" if entry else None}
    manifest.update({
        "engine": C.ENGINE,
        "run": spec["label"],
        "mode": spec["mode"],
        "emo_alpha": spec["alpha"],
        "model_version": C.MODEL_VERSION,
        "model": {"code_repo": C.CODE_REPO, "code_commit": C.CODE_COMMIT, "hf_repo": C.HF_REPO,
                  "hf_revision": C.HF_REVISION, "checkpoints": checkpoints, "provenance": str(C.PROVENANCE)},
        # provenance contract (eval/provenance.py): pretrained weights, never trained on any of the character's clips
        "trained_on_ids": [],
        "training_data_note": "pretrained IndexTTS-2.5 weights, used zero-shot; not fine-tuned in this lab "
                              "(upstream training corpus undisclosed)",
        "conditioning_encoders": C.CONDITIONING_ENCODERS,
        "license": {"name": "bilibili Model Use License Agreement", "notes": str(C.INDEX_ROOT / "LICENSE_NOTES.md"),
                    "outputs_are_derivative_works": True},
        "emotion_controls": {k: controls[k] for k in sorted({t["delivery"] for t in plan})},
        "settings": settings,
        "gpu": gpu,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    gen = settings["generation"]
    rendered = 0
    for take in plan:
        if args.limit_takes and rendered >= args.limit_takes:
            break
        if take["take_id"] in done:
            continue
        spk_path = ref_meta[take["spk"]["id"]]["path"]
        take["emo_path"] = ref_meta[take["emo"]["id"]]["path"] if take["emo"] else None
        text, respelled = synth_text(take["text"], rules)
        factor = round(args.duration_factor * controls[take["delivery"]].get("duration_factor", 1.0), 4)
        factor = min(2.0, max(0.5, factor))
        seed = int(take["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        t0 = time.perf_counter()
        try:
            emotion = emotion_for(tts, take, controls, args, text_vectors)
            result = tts.infer(spk_audio_prompt=spk_path, text=text, output_path=None, lang="EN",
                               use_random=False, verbose=False, interval_silence=gen["interval_silence"],
                               max_text_tokens_per_segment=gen["max_text_tokens_per_segment"],
                               duration_factor=factor, text_normalization=not args.no_text_normalization,
                               do_sample=True, top_p=gen["top_p"], top_k=gen["top_k"], temperature=gen["temperature"],
                               length_penalty=gen["length_penalty"], num_beams=gen["num_beams"],
                               repetition_penalty=gen["repetition_penalty"], max_mel_tokens=gen["max_mel_tokens"],
                               **emotion["kwargs"])
            if not result:
                raise RuntimeError("IndexTTS returned no audio")
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
        rate, pcm = result
        audio = np.asarray(pcm).reshape(-1).astype(np.float32) / 32767.0
        native_seconds = len(audio) / int(rate)
        audio = C.resample(audio, int(rate), C.OUTPUT_SR)
        trimmed, trim = trim_output(audio, C.OUTPUT_SR)
        wav_name = f"{take['take_id']}.wav"
        stats = C.write_wav_pcm16(out_dir / wav_name, trimmed if args.output == "trimmed" else audio, C.OUTPUT_SR)
        extra = {}
        if args.output == "both":
            extra["trimmed_output_wav"] = f"{take['take_id']}.trimmed.wav"
            C.write_wav_pcm16(out_dir / extra["trimmed_output_wav"], trimmed, C.OUTPUT_SR)
        spk, emo = take["spk"], take["emo"]
        entry = {
            "take_id": take["take_id"],
            "engine": C.ENGINE,
            "line_id": take["line_id"],
            "text": take["text"],
            "delivery": take["delivery"],
            "wav": wav_name,
            "reference_id": spk["id"],
            "seed": seed,
            "params": {
                "mode": take["mode"],
                "run": spec["label"],
                "spk_ref_id": spk["id"],
                "spk_audio_prompt": spk_path,
                "emo_ref_id": emo["id"] if emo else None,
                "emo_audio_prompt": take["emo_path"],
                **emotion["record"],
                "reference_screen": "suspect" if any(c and c["screen"] for c in (spk, emo)) else None,
                "ref_audio_variant": args.ref_audio,
                "ref_trimmed": not args.no_ref_trim,
                "lang": "EN",
                "synth_text": text,
                "pronunciations": respelled,
                "text_normalization": not args.no_text_normalization,
                "duration_factor": factor,
                "generation": gen,
                "bf16": args.bf16,
                "model_version": C.MODEL_VERSION,
                "hf_revision": C.HF_REVISION,
                "conditioning_encoders": C.CONDITIONING_ENCODERS,
                "trained_on_ids": [],
                "native_sr": int(rate),
                "output_sr": C.OUTPUT_SR,
                "native_seconds": round(native_seconds, 3),
                "gen_seconds": round(elapsed, 2),
                "rtf": round(elapsed / max(native_seconds, 1e-3), 3),
                "output_trimmed": args.output == "trimmed",
                "output_trim_s": trim,
                **extra,
                **stats,
                "device": device,
            },
        }
        if take["heldout_id"]:
            entry["heldout_id"] = take["heldout_id"]
        manifest["takes"].append(entry)
        manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if device.startswith("cuda"):
            manifest["gpu"]["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            manifest["gpu"]["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        manifest["gpu"]["host_now"] = C.process_commit()
        manifest["network_attempts_blocked"] = list(blocked)
        C.atomic_write_json(manifest_path, manifest)
        rendered += 1
        print(f"[{spec['label']}] {take['take_id']}: {stats['duration_s']:.2f}s in {elapsed:.1f}s", flush=True)
    print(f"[{spec['label']}] {rendered} new takes, {len(manifest['failed'])} failed -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
