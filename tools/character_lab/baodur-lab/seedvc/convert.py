"""Convert takes (another engine's TTS output or a human read) into a character's timbre with Seed-VC V2.

  cd voice-lab/baodur-lab/seedvc
  [VOICE_LAB_CHARACTER=<slug>] ../../seed-vc/.venv/Scripts/python.exe convert.py \
      --in-manifest <<lab>/takes/<engine>/<run>/manifest.json> [--cfm-checkpoint shortlist:cfm_original] --device cuda

The character comes from VOICE_LAB_CHARACTER (voice-lab/characters/<slug>.json, default baodur); <lab> is its lab
folder, and runs, references, delivery map and outputs all resolve under it. Measurements quoted below are Bao-Dur's.
Input: one or more takes manifests {"takes":[{take_id, engine, line_id, text, delivery, wav, ...}]}; relative wav
paths resolve against the manifest folder. `--in-manifest auto` picks every <lab>/takes/*/*/manifest.json not
made by Seed-VC. Output per (input manifest, checkpoint):
  <out-root>/seedvc_from_<engine>/<input run>__<checkpoint tag>[_style][_ref<select>][suffix]/manifest.json
  (engine "seedvc_from_<engine>") + one 48 kHz mono PCM16 WAV per take + run_info.json (settings, references,
  failures, timing). Existing outputs are skipped; --overwrite replaces an output convert.py wrote itself.

Checkpoints (--cfm-checkpoint, repeatable, default best:cfm_original; weights are swapped in place so sweeps load
the models once): pretrained | <path.pth> | run:<run>[@step,step] (raw weights) | ema:<run>[@step,step] (EMA
weights) | best:<run> | shortlist:<run>. best: and shortlist: read runs/<run>/selection.json from selftest.py: a
pick confirmed by listening (selftest.py confirm) wins; otherwise the gated proxy pick, which resolves to pretrained
when no fine-tuned checkpoint beat it (beats_pretrained false). shortlist: = the best two fine-tuned checkpoints +
pretrained, so the eval and the listener decide on real takes. A named file that is missing is an error (never a
silent fallback), and runs whose run_meta.json status is not "complete" are refused (--allow-incomplete-run).

Character reference (the in-context prompt), chosen per take, accumulated until >= --ref-min-sec (6 s) or --ref-count
clips within --ref-max-sec, joined with 0.2 s gaps:
  --ref-select source (default): the source take's own reference_id first (the clip the TTS engine was prompted
      with, so the conversion keeps the register that take was rendered toward), then the delivery label's
      references ordered by closeness to the SOURCE's pitch: |median difference| (st) + |range difference| (st);
  --ref-select prosody: the label's references by that pitch closeness only;
  --ref-select rank: delivery-map order (the self-test uses this: its sources are pitch-shifted on purpose).
Labels come from the take's "delivery" in <lab>/delivery/delivery_map.json (unknown labels fall back to the neutral
label, settings neutral_label "natural"; with no map, the prepared data's neutral fallback list). Refused as
references: held-out test clips, clips the delivery speaker screen attributes to another character (Bao-Dur:
HK-47/G0-T0/Atton) or leaves uncertain, near-duplicates of test lines
(delivery file + both lab rules, live), the take's own line, and any clip whose words repeat the take's text (both
lab rules), so the prompt never contains the answer. Held-out test clips are also refused as sources.

What the prompt controls (measured 2026-09-30 on the CPU, pretrained cfm_small, 30 steps, prompt gblbaodur173 =
111.6 Hz median / 2.6 st p10-p90 range; source val gblbaodur134 shifted -4/0/+4 st = 92/115/146 Hz, range 4.8 st):
all three outputs came out at 111.6 Hz with a 2.2-2.6 st range, contour correlation with the source 0.45,
voicing agreement 0.90 and identical duration. So Seed-VC V2 (no F0 conditioning) takes timing, words and part of
the contour shape from the source, but register AND pitch range from the prompt: the reference choice decides the
register, which is why the default matches it to the source take. With --f0-report on (default) every take records
params.prosody_pair: source and output median F0 / range, median shift (st), range ratio and frame-aligned contour
correlation, plus within_pair_gate (range ratio 0.75-1.33 and |shift| <= 1 st); pairs.py applies the full pair gate
(identity +0.05, WER +0.02) after the eval, so a conversion that flattens an expressive source cannot silently
outrank it.

Style conversion (--convert-style, default OFF). Off: the CFM re-renders the source's 2048-code ASTRAL content
tokens frame for frame, so timing, phrasing, stress, voice quality and part of the intonation come from the source
performance and identity from the character - what this recipe is for. On: the AR model first regenerates those
tokens from the prosody-stripped 32-code tokens, conditioned on the character prompt: accent, rhythm and intonation move
toward the reference (stronger identity, useful for a flat human read), but the source's emotion and timing are
replaced, durations change (7.92 s -> 8.15 s in the CPU check), sampling adds randomness (--top-p/--temperature)
and words can be dropped or slurred. Use it only as an A/B, ideally with an AR fine-tuned by train.py --train-ar.

Other knobs: --diffusion-steps (30; 25 upstream README, 30-50 recommended), --length-adjust (<1 faster, >1 slower;
honoured in both modes, upstream ignores it without style conversion), --intelligibility-cfg-rate /
--similarity-cfg-rate (0.7/0.7 as inference_v2.py; app_vc_v2.py uses 0.5/0.5; both > 0 costs 3 DiT passes per
step), --seed (per take: seed + crc32(take_id)).
Output level (--loudness): "source" (default) keeps the source take's active level so shouted and whispered takes
stay apart, clamped to the character's measured range (prepare_report.json loudness p05 - 2 .. p95 + 2 dB; settings
loudness_fallback when no report exists, else unclamped); "reference" copies the prompt's level; "none" leaves the
model's (it renders 2-6 dB hotter than the source). BigVGAN's final [-1, 1] clamp is skipped and peaks are limited to
-1 dBFS by gain only, so no clipping. Seed-VC V2 synthesises at 22.05 kHz, so WAVs are resampled (soxr) to 48 kHz
but carry nothing above 11.025 kHz - the same bandwidth as the game's own Bao-Dur clips.
Provenance for eval/provenance.py: fine-tuned takes carry params.training_data (the prepare_report.json frozen in
the run folder), pretrained ones params.trained_on_ids = [].
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PAIR_RANGE = (0.75, 1.33)
PAIR_SHIFT_ST = 1.0


def parse_args():
    p = argparse.ArgumentParser(description=f"Seed-VC V2 conversion of takes into {C.NAME}'s voice")
    p.add_argument("--in-manifest", action="append", required=True, help="takes manifest path, repeatable, or 'auto'")
    p.add_argument("--cfm-checkpoint", action="append", default=None,
                   help="pretrained | path | run:<run>[@s,s] | ema:<run>[@s,s] | best:<run> | shortlist:<run>; "
                        "repeatable (default best:cfm_original)")
    p.add_argument("--ar-checkpoint", default=None,
                   help="AR checkpoint for --convert-style: path | run:<run> (latest) | run:<run>@<step>; default ar_base")
    p.add_argument("--runs-root", type=Path, default=C.RUNS, help="where run:/ema:/best:/shortlist: specs are looked up")
    p.add_argument("--allow-incomplete-run", action="store_true", help="use checkpoints of a run that did not complete")
    p.add_argument("--out-root", type=Path, default=C.TAKES)
    p.add_argument("--run", default=None, help="output run name (single manifest x single checkpoint only)")
    p.add_argument("--run-suffix", default="", help="appended to the automatic output run name")
    p.add_argument("--delivery-map", type=Path, default=C.DELIVERY_MAP)
    p.add_argument("--ref-audio", choices=["auto", "original", "restored"], default="auto",
                   help="reference files: auto = what the checkpoint was trained on (original by user decision)")
    p.add_argument("--allow-restored", action="store_true",
                   help="only after a new user decision: the user chose original audio for prompts on 2026-09-30")
    p.add_argument("--ref-min-sec", type=float, default=6.0)
    p.add_argument("--ref-max-sec", type=float, default=15.0)
    p.add_argument("--ref-count", type=int, default=3)
    p.add_argument("--ref-gap-sec", type=float, default=0.2)
    p.add_argument("--ref-select", choices=["source", "prosody", "rank"], default="source",
                   help="source: the take's own reference_id, then the label's references closest to the source's "
                        "pitch median+range; prosody: pitch closeness only; rank: delivery-map order")
    p.add_argument("--f0-report", choices=["on", "off"], default="on",
                   help="record source/output pitch statistics per take (pYIN, ~1 s per take on the CPU)")
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--length-adjust", type=float, default=1.0)
    p.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
    p.add_argument("--similarity-cfg-rate", type=float, default=0.7)
    p.add_argument("--convert-style", action="store_true", help="AR accent/emotion conversion (default off)")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loudness", choices=["source", "reference", "none"], default="source")
    p.add_argument("--fp16", action="store_true", help="fp16 autocast for HuBERT/ASTRAL and AR (upstream default)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu only for smoke tests")
    p.add_argument("--max-vram-gb", type=float, default=6.0)
    p.add_argument("--painter-reserve-gb", type=float, default=C.PAINTER_RESERVE_GB)
    p.add_argument("--skip-headroom-check", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="convert only the first N takes of each manifest")
    p.add_argument("--overwrite", action="store_true", help="replace an output run previously written by convert.py")
    return p.parse_args()


# ----------------------------------------------------------------------------------------- checkpoints

def run_meta(run_dir: Path) -> dict:
    meta = run_dir / "run_meta.json"
    return C.read_json(meta) if meta.exists() else {}


def check_run(run_dir: Path, allow_incomplete: bool) -> dict:
    meta = run_meta(run_dir)
    status = meta.get("status")
    if status != "complete" and not allow_incomplete:
        raise SystemExit(f"ERROR: {run_dir} has status {status!r} (not 'complete'); its checkpoints may be partial. "
                         "Use a complete run or pass --allow-incomplete-run.")
    return meta


def ckpt_entry(path, spec: str, note: str | None = None) -> dict:
    if path is None:
        return {"tag": "pretrained", "path": None, "audio": "original", "spec": spec, "training_data": None,
                "note": note}
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"ERROR: checkpoint {path} (from {spec}) does not exist")
    meta = run_meta(path.parent)
    report = C.frozen_training_report(path)
    if report is None and meta.get("data"):
        legacy = Path(meta["data"]) / "prepare_report.json"   # runs from before train.py froze its report
        report = legacy if legacy.exists() else None
    return {"tag": C.checkpoint_tag(path), "path": path, "audio": meta.get("audio", "original"), "spec": spec,
            "training_data": str(report) if report else None, "note": note}


def steps_of(text: str) -> set[int]:
    return {int(s) for s in text.split(",") if s.strip()}


def resolve_checkpoints(specs: list[str], runs_root: Path, allow_incomplete: bool) -> list[dict]:
    out = []
    for spec in specs:
        if spec == "pretrained":
            out.append(ckpt_entry(None, spec))
            continue
        kind, sep, rest = spec.partition(":")
        if sep and kind in ("run", "ema", "best", "shortlist"):
            name, _, step = rest.partition("@")
            run_dir = runs_root / name
            check_run(run_dir, allow_incomplete)
            if kind in ("run", "ema"):
                ckpts = C.run_checkpoints(run_dir, "CFM" if kind == "run" else "EMA-CFM")
                if not ckpts:
                    raise SystemExit(f"ERROR: no {'EMA ' if kind == 'ema' else ''}CFM checkpoints in {run_dir}")
                wanted = steps_of(step)
                chosen = [c for c in ckpts if not wanted or C.checkpoint_step(c) in wanted]
                missing = wanted - {C.checkpoint_step(c) for c in chosen}
                if missing:
                    raise SystemExit(f"ERROR: no step(s) {sorted(missing)} among the {kind} checkpoints of {run_dir}")
                out += [ckpt_entry(c, spec) for c in chosen]
                continue
            selection_path = run_dir / "selection.json"
            if not selection_path.exists():
                if kind == "shortlist":
                    raise SystemExit(f"ERROR: {selection_path} missing: run selftest.py score --train-run {name}")
                best = run_meta(run_dir).get("best_val") or {}
                files = [p for p in best.get("checkpoints", []) if Path(p).name.startswith("CFM_")]
                if not files:
                    raise SystemExit(f"ERROR: {run_dir} has neither selection.json nor a best_val checkpoint")
                print(f"WARNING: {spec}: no selection.json; using the lowest validation loss ({files[0]}), "
                      "which is not a listening-quality criterion", flush=True)
                out.append(ckpt_entry(files[0], spec, "lowest val loss (no selftest selection)"))
                continue
            selection = C.read_json(selection_path)
            confirmed = selection.get("confirmed")
            if kind == "best":
                if confirmed:
                    out.append(ckpt_entry(confirmed.get("cfm_checkpoint"), spec,
                                          f"confirmed by listening: {confirmed.get('tag')}"))
                elif selection.get("beats_pretrained") is False:
                    print(f"NOTE: {spec}: no fine-tuned checkpoint passed the gates and beat pretrained; using "
                          "pretrained cfm_small", flush=True)
                    out.append(ckpt_entry(None, spec, "selection: pretrained beats every fine-tuned checkpoint"))
                else:
                    print(f"WARNING: {spec}: {selection.get('tag')} is the proxy pick, not yet confirmed by "
                          f"listening (selftest.py confirm); consider shortlist:{name}", flush=True)
                    out.append(ckpt_entry(selection.get("cfm_checkpoint"), spec, "selftest proxy pick (unconfirmed)"))
            else:
                entries = list(selection.get("shortlist") or [])
                if confirmed:
                    entries = [confirmed] + [e for e in entries if e.get("tag") != confirmed.get("tag")]
                if not entries:
                    raise SystemExit(f"ERROR: {selection_path} has an empty shortlist")
                out += [ckpt_entry(e.get("cfm_checkpoint"), spec, f"shortlist: {e.get('tag')}") for e in entries]
            continue
        out.append(ckpt_entry(Path(spec).resolve(), spec))
    unique = {}
    for ckpt in out:  # run:X, best:X and shortlist:X can name the same file; convert it once
        unique.setdefault(C.norm_path(ckpt["path"]) if ckpt["path"] else "pretrained", ckpt)
    return list(unique.values())


def resolve_ar(spec: str | None, runs_root: Path, allow_incomplete: bool) -> str | None:
    if not spec:
        return None
    if spec.startswith("run:"):
        name, _, step = spec[4:].partition("@")
        check_run(runs_root / name, allow_incomplete)
        ckpts = [c for c in C.run_checkpoints(runs_root / name, "AR") if not step or C.checkpoint_step(c) == int(step)]
        if not ckpts:
            raise SystemExit(f"ERROR: no AR checkpoint {spec} in {runs_root / name}")
        return str(ckpts[-1])
    path = Path(spec).resolve()
    if not path.exists():
        raise SystemExit(f"ERROR: {path} does not exist")
    return str(path)


def resolve_manifests(specs: list[str]) -> list[Path]:
    out = []
    for spec in specs:
        if spec == "auto":
            for path in sorted(C.TAKES.glob("*/*/manifest.json")):
                if not path.parent.parent.name.startswith("seedvc_from_"):
                    out.append(path.resolve())
        else:
            out.append(Path(spec).resolve())
    return out


# ------------------------------------------------------------------------------------------ references

class ReferencePicker:
    def __init__(self, rows, audio: str, delivery_map: Path, args, speakers: dict):
        self.rows = {r["id"]: r for r in rows}
        self.audio = audio
        self.min_sec, self.max_sec, self.count, self.gap = args.ref_min_sec, args.ref_max_sec, args.ref_count, args.ref_gap_sec
        self.map = C.read_json(delivery_map)["deliveries"] if delivery_map and delivery_map.exists() else None
        self.map_source = str(delivery_map) if self.map else None
        report = C.PREPARED / audio / "prepare_report.json"
        if not report.exists():
            report = C.PREPARED / "original" / "prepare_report.json"
        prep = C.read_json(report) if report.exists() else {}
        self.fallback = prep.get("fallback_references", [])
        self.loudness = prep.get("loudness", C.SETTINGS["loudness_fallback"] or {})   # {} = no clamp (no report)
        self.refused = {**{k: "near duplicate of a held-out test line" for k in speakers["near_test"]},
                        **{k: "uncertain speaker (speaker screen)" for k in speakers["uncertain"]},
                        **{k: f"not {C.NAME} ({v})" for k, v in speakers["confirmed"].items()}}
        self.select = args.ref_select
        self._overlap: dict = {}
        self.warnings: list[str] = []

    _stats: dict = {}   # (audio, clip id) -> pitch stats, shared by every picker of this process

    def ref_stats(self, rid) -> dict | None:
        """Cached pYIN median F0 / p10-p90 range of a reference clip (as the checkpoint's training audio)."""
        key = (self.audio, rid)
        if key not in self._stats:
            row = self.rows.get(rid)
            path = C.clip_path(row, self.audio) if row else None
            self._stats[key] = C.f0_stats(C.load_mono(path, C.MODEL_SR)[0]) if path and path.exists() else None
        return self._stats[key]

    def _order(self, refs, source_stats) -> list:
        """The label's references by pitch closeness to the source: |median diff| (st) + |range diff| (st)."""
        if self.select == "rank" or not source_stats:
            return list(refs)
        import math

        def key(pair):
            rank, ref = pair
            stats = self.ref_stats(ref.get("id"))
            if not stats:
                return (1, 0.0, rank)
            shift = abs(12 * math.log2(stats["median_hz"] / source_stats["median_hz"]))
            return (0, shift + abs(stats["range_st"] - source_stats["range_st"]), rank)
        return [ref for _, ref in sorted(enumerate(refs), key=key)]

    def _warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)
            print("WARNING:", text, flush=True)

    def _repeats_line(self, rid: str, take_text: str | None) -> str | None:
        if not take_text:
            return None
        key = (rid, take_text)
        if key not in self._overlap:
            self._overlap[key] = C.text_overlap(take_text, self.rows[rid].get("text") or "")
        return self._overlap[key]

    def _accumulate(self, refs, exclude, take_text) -> list[dict]:
        chosen = []
        for ref in refs:
            rid = ref.get("id")
            row = self.rows.get(rid)
            if row is None:
                self._warn(f"reference {rid!r} is not in corpus.jsonl; skipped")
                continue
            if row["split"] == "test":
                self._warn(f"reference {rid} is a held-out TEST clip; refused")
                continue
            if rid in self.refused:
                self._warn(f"reference {rid} refused: {self.refused[rid]}")
                continue
            if row["split"] not in ("train", "val") or row["kind"] != "speech" or rid in exclude:
                continue
            if any(c["id"] == rid for c in chosen):
                continue
            why = self._repeats_line(rid, take_text)
            if why:
                self._warn(f"reference {rid} skipped for a take whose line it repeats ({why})")
                continue
            path = C.clip_path(row, self.audio)
            if not path.exists():
                self._warn(f"reference audio missing: {path}")
                continue
            total = sum(c["duration"] for c in chosen) + self.gap * len(chosen) + float(row["duration"])
            if chosen and total > self.max_sec:
                continue
            chosen.append({"id": rid, "path": str(path), "duration": float(row["duration"]), "text": row["text"]})
            if total >= self.min_sec or len(chosen) >= self.count:
                break
        return chosen

    def pick(self, delivery, exclude, source_stats=None, source_ref=None, take_text=None):
        """Return (label used, where the references came from, chosen refs, whether the source's own ref led)."""
        neutral = C.NEUTRAL_LABEL
        label = str(delivery or neutral).strip().lower()
        lead = [{"id": source_ref}] if self.select == "source" and source_ref else []
        if self.map:
            for candidate in (label, neutral):
                entry = self.map.get(candidate) or {}
                chosen = self._accumulate(lead + self._order(entry.get("references") or [], source_stats), exclude,
                                          take_text)
                if chosen:
                    if candidate != label:
                        self._warn(f"delivery {label!r} has no usable references in the map; used {neutral!r}")
                    return candidate, "delivery_map", chosen, bool(lead) and chosen[0]["id"] == source_ref
        chosen = self._accumulate(lead + self._order(self.fallback, source_stats), exclude, take_text)
        if not chosen:
            raise RuntimeError(f"no usable {C.NAME} reference (delivery map and fallback list both empty)")
        return neutral, "prepare_report.fallback_references", chosen, bool(lead) and chosen[0]["id"] == source_ref

    def wave(self, chosen):
        import numpy as np
        gap = np.zeros(int(round(self.gap * C.MODEL_SR)), dtype=np.float32)
        parts = []
        for i, ref in enumerate(chosen):
            parts += ([gap] if i else []) + [C.load_mono(ref["path"], C.MODEL_SR)[0]]
        return np.concatenate(parts)[: int(self.max_sec * C.MODEL_SR)]


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    args.out_root = args.out_root.resolve()           # resolve user paths before setup_seedvc() changes the cwd
    args.delivery_map = args.delivery_map.resolve()
    runs_root = args.runs_root.resolve()
    manifests = resolve_manifests(args.in_manifest)
    checkpoints = resolve_checkpoints(args.cfm_checkpoint or ["best:cfm_original"], runs_root, args.allow_incomplete_run)
    ar_ckpt = resolve_ar(args.ar_checkpoint, runs_root, args.allow_incomplete_run)
    ar_report = C.frozen_training_report(ar_ckpt) if ar_ckpt else None
    for ckpt in checkpoints:
        C.require_original(ckpt["audio"] if args.ref_audio == "auto" else args.ref_audio, args.allow_restored,
                           f"prompts for checkpoint {ckpt['tag']}")
    if args.run and (len(manifests) != 1 or len(checkpoints) != 1):
        print("ERROR: --run needs exactly one input manifest and one checkpoint", file=sys.stderr)
        return 2
    if not manifests:
        print("nothing to convert (no input manifests found)")
        return 0

    rows = C.load_corpus()
    speakers = C.speaker_exclusions(required=False, rows=rows)   # imports the lab's leak rules before setup_seedvc
    C.setup_seedvc()
    import numpy as np
    import torch
    from svc_core import Converter

    guard = C.HeldOutGuard(rows)
    inputs = []
    for manifest in manifests:
        takes = C.read_json(manifest)["takes"]
        engines = {t.get("engine") for t in takes if t.get("engine")}
        inputs.append((manifest, takes, C.safe_name(engines.pop() if len(engines) == 1 else manifest.parent.parent.name, 60)))
    ref_suffix = "" if args.ref_select == "source" else f"_ref{args.ref_select}"
    jobs = []
    for ckpt in checkpoints:  # checkpoint-major: each checkpoint's weights are loaded once
        for manifest, takes, engine in inputs:
            run = args.run or C.safe_name(f"{manifest.parent.name}__{ckpt['tag']}"
                                          f"{'_style' if args.convert_style else ''}{ref_suffix}{args.run_suffix}")
            out_dir = args.out_root / f"seedvc_from_{engine}" / run
            if (out_dir / "manifest.json").exists() or (out_dir / "run_info.json").exists():
                if not args.overwrite:
                    print(f"skip (exists): {out_dir}  (--overwrite to redo)")
                    continue
                if not (out_dir / "run_info.json").exists():
                    print(f"ERROR: {out_dir} was not written by convert.py; refusing to overwrite", file=sys.stderr)
                    return 2
            jobs.append((manifest, takes, engine, ckpt, out_dir))
    if not jobs:
        print("nothing to do")
        return 0

    device = C.pick_device(args.device)
    vram = C.cap_vram(device, args.max_vram_gb, args.painter_reserve_gb, not args.skip_headroom_check)
    wrapper = C.build_wrapper()
    C.load_weights(wrapper, None, ar_ckpt)
    pretrained_cfm = {k: v.clone() for k, v in wrapper.cfm.state_dict().items()}
    pretrained_reg = {k: v.clone() for k, v in wrapper.cfm_length_regulator.state_dict().items()}
    converter = Converter(wrapper, device, fp16=args.fp16, convert_style=args.convert_style)
    loaded_tag = "pretrained"
    source_tracks: dict = {}   # source sha256 -> pYIN track (sources recur across checkpoints)
    want_f0 = args.f0_report == "on"

    for manifest, takes, engine, ckpt, out_dir in jobs:
        if ckpt["tag"] != loaded_tag:
            if ckpt["path"] is None:  # load_state_dict copies in place, so weights stay on their device
                wrapper.cfm.load_state_dict(pretrained_cfm)
                wrapper.cfm_length_regulator.load_state_dict(pretrained_reg)
            else:
                C.load_finetuned(wrapper.cfm, wrapper.cfm_length_regulator, ckpt["path"], "cfm")
            wrapper.eval()
            converter.clear_cache()
            loaded_tag = ckpt["tag"]
        ref_audio = ckpt["audio"] if args.ref_audio == "auto" else args.ref_audio
        picker = ReferencePicker(rows, ref_audio, args.delivery_map, args, speakers)
        lo = picker.loudness.get("p05", float("-inf")) - 2.0   # the character's measured level range, +-2 dB
        hi = picker.loudness.get("p95", float("inf")) + 2.0
        reports = [r for r in ([ckpt["training_data"]] + ([str(ar_report)] if ar_report and args.convert_style else []))
                   if r]
        trained_on_known = ckpt["path"] is None or bool(ckpt["training_data"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "manifest.json").exists():  # --overwrite of our own earlier output: drop its listed WAVs
            for old in C.read_json(out_dir / "manifest.json").get("takes", []):
                old_wav = Path(str(old.get("wav", "")))
                if old_wav.parent.resolve() == out_dir.resolve() and old_wav.suffix == ".wav" and old_wav.exists():
                    old_wav.unlink()
        new_takes, failures, busy, pair_counts = [], [], 0.0, {"within": 0, "outside": 0, "unknown": 0}
        selected = takes[: args.limit or None]
        print(f"\n== {manifest} -> {out_dir} ({len(selected)} takes, checkpoint {ckpt['tag']}, refs {ref_audio}, "
              f"select {args.ref_select})")
        for take in selected:
            take_id = str(take.get("take_id") or Path(str(take.get("wav"))).stem)
            try:
                src = Path(str(take["wav"]))
                src = src if src.is_absolute() else (manifest.parent / src)
                reason = guard.file_reason(src)
                if reason:
                    raise RuntimeError(f"source refused: {reason}")
                source22, _ = C.load_mono(src, C.MODEL_SR)
                peak = float(np.abs(source22).max()) if len(source22) else 0.0
                if peak > 0.99:
                    source22 = source22 * (0.99 / peak)
                params_in = take.get("params") or {}
                exclude = {take.get("line_id"), take.get("heldout_id"), params_in.get("source_clip_id")}
                src_track = None
                if want_f0 or args.ref_select != "rank":
                    sha = C.sha256_file(src)
                    if sha not in source_tracks:
                        source_tracks[sha] = C.f0_track(source22)
                    src_track = source_tracks[sha]
                source_f0 = C.f0_summary(*src_track) if src_track else None
                label, ref_source, refs, led = picker.pick(take.get("delivery"), exclude, source_f0,
                                                           take.get("reference_id"), take.get("text"))
                key = (tuple(r["id"] for r in refs), ref_audio)
                ref_wave = picker.wave(refs)
                ref = converter.reference(key, ref_wave)
                seed = C.take_seed(args.seed, take_id)
                started = time.time()
                y22 = converter.convert(source22, ref, diffusion_steps=args.diffusion_steps,
                                        length_adjust=args.length_adjust,
                                        intelligibility_cfg_rate=args.intelligibility_cfg_rate,
                                        similarity_cfg_rate=args.similarity_cfg_rate, top_p=args.top_p,
                                        temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                                        seed=seed)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                busy += elapsed
                prosody_pair = None
                if want_f0:
                    aligned = not args.convert_style and abs(len(y22) / max(1, len(source22)) - 1) < 0.02
                    prosody_pair = C.f0_pair(src_track, C.f0_track(y22), aligned=aligned)
                    ratio, shift = prosody_pair["range_ratio"], prosody_pair["median_shift_st"]
                    prosody_pair["within_pair_gate"] = None if ratio is None else \
                        bool(PAIR_RANGE[0] <= ratio <= PAIR_RANGE[1] and abs(shift) <= PAIR_SHIFT_ST)
                    pair_counts[{True: "within", False: "outside", None: "unknown"}[prosody_pair["within_pair_gate"]]] += 1
                target_db = None
                if args.loudness == "source":
                    target_db = min(max(C.active_rms_db(source22, C.MODEL_SR), lo), hi)
                elif args.loudness == "reference":
                    target_db = C.active_rms_db(ref_wave, C.MODEL_SR)
                raw_peak_db = float(20 * np.log10(np.abs(y22).max() + 1e-9))
                y48, gain_db, ceiling_db = C.finalize(y22, target_db)
                new_id = f"{take_id}__svc-{ckpt['tag']}"
                wav_path = out_dir / f"{C.safe_name(new_id, 150)}.wav"
                C.write_pcm16(wav_path, y48, C.OUT_SR)
                params = {
                    "source_take_id": take_id, "source_engine": take.get("engine"), "source_wav": str(src),
                    "source_manifest": str(manifest), "source_seconds": round(len(source22) / C.MODEL_SR, 3),
                    "output_seconds": round(len(y48) / C.OUT_SR, 3),
                    "cfm_checkpoint": str(ckpt["path"]) if ckpt["path"] else "pretrained:cfm_small",
                    "cfm_tag": ckpt["tag"], "cfm_spec": ckpt["spec"], "cfm_note": ckpt["note"],
                    "ar_checkpoint": (ar_ckpt or "pretrained:ar_base") if args.convert_style else None,
                    "diffusion_steps": args.diffusion_steps, "length_adjust": args.length_adjust,
                    "intelligibility_cfg_rate": args.intelligibility_cfg_rate,
                    "similarity_cfg_rate": args.similarity_cfg_rate, "convert_style": args.convert_style,
                    "top_p": args.top_p if args.convert_style else None,
                    "temperature": args.temperature if args.convert_style else None,
                    "repetition_penalty": args.repetition_penalty if args.convert_style else None,
                    "fp16": converter.fp16, "reference_ids": [r["id"] for r in refs],
                    "reference_seconds": round(ref.seconds, 2), "reference_audio": ref_audio,
                    "reference_label": label, "reference_source": ref_source,
                    "reference_select": args.ref_select, "source_reference_led": led,
                    "source_reference_id": take.get("reference_id"), "source_f0": source_f0,
                    "reference_f0": [picker.ref_stats(r["id"]) for r in refs] if source_f0 and args.ref_select != "rank"
                    else None,
                    "prosody_pair": prosody_pair, "run_suffix": args.run_suffix,
                    "loudness": args.loudness, "target_db": None if target_db is None else round(target_db, 2),
                    "gain_db": round(gain_db, 2), "ceiling_gain_db": round(ceiling_db, 2),
                    "raw_peak_dbfs": round(raw_peak_db, 2), "rtf": round(elapsed / max(1e-6, len(source22) / C.MODEL_SR), 3),
                    "source_clip_id": params_in.get("source_clip_id"), "heldout_id": take.get("heldout_id"),
                }
                if reports:
                    params["training_data"] = reports            # eval/provenance.py contract field
                elif trained_on_known and not (args.convert_style and ar_ckpt):
                    params["trained_on_ids"] = []                # pretrained Seed-VC: none of the character's clips
                new_takes.append({
                    "take_id": new_id, "engine": f"seedvc_from_{engine}", "line_id": take.get("line_id"),
                    "text": take.get("text"), "delivery": take.get("delivery"), "wav": str(wav_path),
                    "reference_id": refs[0]["id"], "seed": seed, "params": params,
                })
                print(f"  {new_id}: {len(source22) / C.MODEL_SR:.2f} s -> {len(y48) / C.OUT_SR:.2f} s, "
                      f"refs {'+'.join(key[0])}, {elapsed:.1f} s", flush=True)
            except Exception as error:  # keep converting the rest; failures are listed in run_info.json
                failures.append({"take_id": take_id, "error": f"{type(error).__name__}: {error}"})
                print(f"  FAILED {take_id}: {error}", flush=True)
        C.write_json(out_dir / "manifest.json", {"takes": new_takes})
        C.write_json(out_dir / "run_info.json", {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "engine": f"seedvc_from_{engine}", "source_manifest": str(manifest), "checkpoint": {
                "tag": ckpt["tag"], "spec": ckpt["spec"], "note": ckpt["note"],
                "path": str(ckpt["path"]) if ckpt["path"] else None, "trained_on": ckpt["audio"],
                "training_data": ckpt["training_data"]},
            "ar_checkpoint": ar_ckpt, "delivery_map": picker.map_source, "reference_audio": ref_audio,
            "reference_select": args.ref_select, "prosody_pair_gate": {"range_ratio": PAIR_RANGE,
                                                                       "abs_median_shift_st": PAIR_SHIFT_ST,
                                                                       "takes": pair_counts if want_f0 else None},
            "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "device": vram, "converted": len(new_takes), "failed": len(failures), "failures": failures,
            "warnings": picker.warnings, "busy_seconds": round(busy, 1),
            "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None,
        })
        print(f"wrote {out_dir / 'manifest.json'}: {len(new_takes)} converted, {len(failures)} failed"
              + (f"; pitch pair gate {pair_counts}" if want_f0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
