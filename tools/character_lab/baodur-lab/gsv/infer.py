"""Headless GPT-SoVITS (v2ProPlus / v2Pro) inference for one character's takes, through GPT_SoVITS/TTS_infer_pack.

Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters; <lab> is its lab folder and
<speaker> its config "speaker" (experiments are <speaker>_<version>_<audio>[_trval]).
For every line and every seed, the model is prompted with a reference clip picked for the line's delivery from
<lab>/delivery/delivery_map.json (prompt audio + its exact corpus transcript; optional aux refs that are averaged
into the timbre/style embedding), rendered at 32 kHz, resampled to 48 kHz mono PCM16 (soxr VHQ), and recorded in
the shared takes manifest:
  <lab>/takes/gsv/<run>/manifest.json  {"takes": [{take_id, engine, line_id, text, delivery, wav,
                                                   reference_id, seed, params}], ...}

Lines: <lab>/delivery/takes_script.json ("new" lines and the 10 "heldout" test lines), plus --val-lines, which adds
the val split's clips as ground-truth lines (source "val", heldout_id = the val clip; GSV trains on the train split
only, so these are unseen performances the evaluation can compare against). --source picks which kinds to render
(a character without a takes script yet: --script "" with --source val, or --text).

Selection protocol (the held-out TEST lines never choose a model):
  1. sweep checkpoints on --source new,val --val-lines all with fixed seeds and references, one factor at a
     time (SoVITS epochs at a fixed GPT epoch, GPT epochs at a fixed SoVITS epoch), plus the zero-shot base;
  2. evaluate those runs and listen; freeze the choice with freeze_selection.py (<lab>/gsv/selection.json);
  3. render the full script once with --selection. Rendering held-out lines from fine-tuned weights without
     --selection is refused (--allow-unfrozen-heldout overrides; zero-shot is always allowed).

Reference rules (GPT-SoVITS continues the prompt's semantic tokens, so the prompt drives delivery):
  * held-out test clips are refused by id, file name and sha256 of original + restored copies; naming one with
    --ref-id/--ref-for is a hard error, one found in the delivery map is skipped, reported and listed in the
    manifest (top level, outside the takes) - pass --strict-refs to make that fatal too;
  * so are clips the speaker screen attributes to other characters or has not confirmed, and train/val lines that
    give away a held-out test line (they would leak its performance into the held-out takes), see gsv_common;
  * a val line is never prompted (or aux-referenced) by its own clip or by a clip whose text overlaps it;
  * only train/val speech clips with an exact transcript (tlk-exact/dlg) whose script-vs-ASR WER is at most
    --max-ref-wer, and 3-10 s long (GPT-SoVITS rejects prompts outside that window); the audio file is always
    derived from the corpus, original game audio (DECISIONS.md; restored prompts need --allow-restored);
  * the first valid reference in the delivery's ranked list is used; --refs-per-line R rotates seeds through the
    top R; a delivery without a usable reference falls back to "natural", then to a corpus pick;
    --ref-for LABEL=ID[,ID] replaces a delivery's ranked list (same checks), e.g. when the map's top 'urgent'
    clips are angry ones; deliveries that share a planned reference are reported as a warning.
Text (target and prompt) goes through <lab>/gsv/lexicon.json exactly as training did, and prompt transcripts
through <lab>/gsv/text_overrides.json, so names are pronounced consistently and the prompt text matches what the
clip says.
Each take records params.conditioning_encoders (["eres2netv2"] for v2Pro/v2ProPlus, whose SoVITS adds this speaker
network's embedding of every reference) so the evaluation can score identity with uncoupled models, and a
duration check (take length vs the corpus speaking rate for its text) that flags likely skips or repeats.

Usage (GPT-SoVITS venv python; re-execs itself there if needed):
  python infer.py --exp <speaker>_v2ProPlus_original --gpt-epoch 15 --sovits-epoch 8 --source new,val \\
      --val-lines all --seed-list 1,2 --run sel_g15_s8 --device cuda          # selection sweep point
  python infer.py --zero-shot --source new,val --val-lines all --seed-list 1,2 --run sel_zeroshot --device cuda
  python infer.py --selection <lab>/gsv/selection.json --refs-per-line 2 --seeds 6 --run final --device cuda
  python infer.py --exp <speaker>_v2ProPlus_original --source new,val --val-lines all --dry-run   # plan only
  python infer.py --exp ... --text "We need to talk." --delivery troubled --seeds 2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402

ENGINE = "gsv"
PRESETS = {  # AR sampling for the text-to-semantic model; upstream webui defaults are "default"
    "default": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35},
    "stable": {"top_k": 5, "top_p": 0.9, "temperature": 0.8, "repetition_penalty": 1.35},
    "expressive": {"top_k": 20, "top_p": 0.95, "temperature": 1.05, "repetition_penalty": 1.35},
}
SPLIT_METHODS = ("auto", "cut0", "cut1", "cut2", "cut3", "cut4", "cut5")
SOURCES = ("new", "heldout", "val")


def norm_words(text: str) -> str:
    return " ".join(C.norm_words(text))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "take"


def audio_duration(path: Path, fallback: float | None) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except ImportError:
        return fallback


def n_chars(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text or ""))


def duration_model(rows: list[dict]) -> dict:
    """The character's speaking rate from the train split (seconds per letter/digit of the transcript) and the spread
    of real clips around it; a take outside the widened real range likely skipped words or repeated / stalled."""
    pairs = [(float(r["duration"]), n_chars(r["text"])) for r in rows
             if r.get("split") == "train" and r.get("kind") == "speech" and r.get("tts_usable")
             and r.get("text_source") in C.EXACT_SOURCES and n_chars(r.get("text")) >= 20]
    rate = statistics.median(d / n for d, n in pairs)
    ratios = sorted(d / (n * rate) for d, n in pairs)
    lo, hi = ratios[max(0, int(0.02 * len(ratios)))], ratios[min(len(ratios) - 1, int(0.98 * len(ratios)))]
    return {"s_per_char": round(rate, 5), "clips": len(pairs), "real_ratio_p2_p98": [round(lo, 3), round(hi, 3)],
            "flag_below": round(lo * 0.85, 3), "flag_above": round(hi * 1.15, 3)}


# ----------------------------------------------------------------------------------------------- references
class References:
    def __init__(self, dmap: dict | None, rows: list[dict], guard: C.HeldOut, excl: C.Exclusions | None,
                 audio_mode: str, max_wer: float, corpus_fallback: bool, ref_for: dict[str, list[str]]):
        self.dmap, self.rows, self.guard, self.excl = dmap or {}, rows, guard, excl
        self.by_id = {r["id"]: r for r in rows}
        self.audio_mode, self.max_wer, self.corpus_fallback_ok = audio_mode, max_wer, corpus_fallback
        self.ref_for = ref_for
        self.skipped: list[dict] = []
        self._cache: dict[str, list[dict]] = {}

    def _skip(self, label: str, rid, reason: str, heldout: bool = False) -> None:
        entry = {"delivery": label, "id": rid, "reason": reason, "heldout": heldout}
        if entry not in self.skipped:
            self.skipped.append(entry)
            if heldout:
                C.log(f"REFUSED reference {rid} for '{label}': {reason}")

    def check(self, rid: str | None, label: str, rank: int, ref: dict | None = None) -> dict | None:
        ref = ref or {}
        row = self.by_id.get(rid) if rid else None
        if self.audio_mode == "restored":
            map_path, corpus_path = ref.get("restored_audio"), (C.restored_path(rid) if rid else None)
        else:
            map_path, corpus_path = ref.get("audio"), (row["audio"] if row else None)
        for candidate in (map_path, corpus_path):
            reason = self.guard.reason(rid, candidate)
            if reason:
                self._skip(label, rid, reason, heldout=True)
                return None
        if row is None:
            self._skip(label, rid, "not a corpus clip")
            return None
        why = self.excl.reason(rid) if self.excl else None
        if why:
            self._skip(label, rid, why)
            return None
        path = corpus_path  # the corpus is the source of truth; the map's copy of the path is only cross-checked
        path_note = ""
        if map_path and corpus_path and Path(map_path).resolve() != Path(corpus_path).resolve():
            path_note = f"delivery map audio {map_path} differs from the corpus path; corpus path used"
        if row.get("split") not in ("train", "val"):
            self._skip(label, rid, f"split {row.get('split')} is not train/val")
            return None
        if row.get("kind") != "speech" or row.get("text_source") not in C.EXACT_SOURCES:
            self._skip(label, rid, "no exact (tlk-exact/dlg) transcript")
            return None
        if not row.get("tts_usable"):
            self._skip(label, rid, "corpus marks the clip tts_usable=false")
            return None
        if row.get("wer") is not None and float(row["wer"]) > self.max_wer and not row.get("text_override"):
            self._skip(label, rid, f"script vs ASR WER {row['wer']} > {self.max_wer} (prompt text may not match audio)")
            return None
        if not path or not Path(path).exists():
            self._skip(label, rid, f"audio missing: {path}")
            return None
        duration = audio_duration(Path(path), row.get("duration"))
        if duration is None or not C.REF_MIN_S <= duration <= C.REF_MAX_S:
            self._skip(label, rid, f"duration {duration} s outside GPT-SoVITS' 3-10 s prompt window")
            return None
        notes = [path_note] if path_note else []
        if ref.get("text") and norm_words(ref["text"]) != norm_words(row.get("script_text") or row["text"]):
            notes.append("delivery-map text differs from the corpus transcript; corpus transcript used")
        if row.get("text_override"):
            notes.append(f"prompt text from text_overrides.json (script: {row.get('script_text')!r})")
        if self.excl and rid in self.excl.suspect:
            notes.append(f"speaker screen 'suspect' ({C.DISPLAY_NAME}, but far from {C.POSSESSIVE} usual timbre)")
        return {"id": rid, "audio": str(Path(path)), "text": row["text"], "duration": round(duration, 3),
                "rank": rank, "delivery": label, "why": ref.get("why", ""), "note": "; ".join(notes),
                "words": C.norm_words(row["text"]), "split": row.get("split")}

    def candidates(self, label: str) -> list[dict]:
        if label not in self._cache:
            if label in self.ref_for:
                entries = [{"id": rid, "why": "--ref-for override"} for rid in self.ref_for[label]]
            else:
                entries = ((self.dmap.get("deliveries") or {}).get(label) or {}).get("references") or []
            found = []
            for rank, ref in enumerate(entries):
                checked = self.check(ref.get("id"), label, rank, ref)
                if checked:
                    found.append(checked)
            self._cache[label] = found
        return self._cache[label]

    def corpus_pick(self, n: int = 8) -> list[dict]:
        if "__corpus__" not in self._cache:
            pool = [r for r in self.rows if r.get("split") in ("train", "val") and r.get("kind") == "speech"
                    and r.get("text_source") in C.EXACT_SOURCES and not r.get("wer")
                    and C.REF_MIN_S + 0.5 <= float(r.get("duration") or 0) <= C.REF_MAX_S - 0.5
                    and not r.get("directions")]
            pool.sort(key=lambda r: (abs(float(r["duration"]) - 6.0), r["id"]))
            picked = []
            for r in pool:
                checked = self.check(r["id"], "corpus-fallback", len(picked))
                if checked:
                    picked.append(checked)
                if len(picked) >= n:
                    break
            self._cache["__corpus__"] = picked
        return self._cache["__corpus__"]

    @staticmethod
    def allowed(cands: list[dict], line: dict) -> list[dict]:
        """Drop references that would give away a ground-truth line: its own clip, or a clip whose text overlaps
        it (contains a sentence of it, shares most words or a run of 6 words)."""
        gt = line.get("heldout_id")
        if not gt:
            return cands
        words, sentences = C.norm_words(line["text"]), C.sentence_keys(line["text"])
        return [c for c in cands if c["id"] != gt and not C.overlap_reason(c["words"], words, sentences)]

    def resolve(self, label: str, line: dict) -> tuple[list[dict], str]:
        found = self.allowed(self.candidates(label), line)
        if found:
            return found, ("cli-ref-for" if label in self.ref_for else "delivery_map")
        if label != "natural":
            natural = self.allowed(self.candidates("natural"), line)
            if natural:
                return natural, "fallback-natural"
        if self.corpus_fallback_ok:
            picked = self.allowed(self.corpus_pick(), line)
            if picked:
                return picked, "corpus-fallback"
        return [], "none"


# ----------------------------------------------------------------------------------------------- plan
def parse_ref_for(items: list[str] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"bad --ref-for {item!r}; use LABEL=ID[,ID...]")
        label, ids = item.split("=", 1)
        ids_list = [i.strip() for i in ids.split(",") if i.strip()]
        if not label.strip() or not ids_list:
            raise SystemExit(f"bad --ref-for {item!r}; use LABEL=ID[,ID...]")
        out[label.strip()] = ids_list
    return out


def seen_in_train(row: dict, rows: list[dict]) -> str | None:
    """A train-split clip that repeats this val line (either contains a sentence of the other, shares most words
    or a 6-word run): a model trained on it has effectively heard the line, so the line is not an unseen test."""
    words, sentences = C.norm_words(row["text"]), C.sentence_keys(row["text"])
    for t in rows:
        if t.get("split") != "train":
            continue
        twords = C.norm_words(t.get("text"))
        why = C.overlap_reason(twords, words, sentences) or C.overlap_reason(words, twords, C.sentence_keys(t.get("text")))
        if why:
            return f"train clip {t['id']} {why.replace(' with', '')} it"
    return None


def val_lines(spec: str | None, rows: list[dict], guard: C.HeldOut, excl: C.Exclusions | None,
              dmap: dict | None, include_seen: bool = False, dropped: list | None = None) -> list[dict]:
    """Val-split clips as ground-truth lines (source 'val'), labelled with the delivery whose reference list ranks
    the clip best (else 'natural'). spec: none | all | N | comma list of clip ids. Val clips whose text a train
    clip repeats (e.g. Bao-Dur's 237 ~ train 239/243, 261 ~ train 258) are left out unless include_seen; `dropped`
    collects them."""
    spec = (spec or "none").strip()
    if spec == "none":
        return []
    label_of: dict[str, tuple[int, str]] = {}
    for label, entry in ((dmap or {}).get("deliveries") or {}).items():
        for rank, ref in enumerate(entry.get("references") or []):
            current = label_of.get(ref.get("id"))
            if current is None or rank < current[0]:
                label_of[ref.get("id")] = (rank, label)
    pool = sorted((r for r in rows if r.get("split") == "val" and r.get("kind") == "speech" and r.get("tts_usable")
                   and r.get("text_source") in C.EXACT_SOURCES and not guard.reason(r["id"])
                   and not (excl and excl.reason(r["id"]))), key=lambda r: r["id"])
    if not include_seen:
        kept = []
        for r in pool:
            why = seen_in_train(r, rows)
            if why:
                if dropped is not None:
                    dropped.append({"id": r["id"], "reason": why})
            else:
                kept.append(r)
        pool = kept
    if spec == "all":
        chosen = pool
    elif spec.isdigit():
        chosen = pool[: int(spec)]
    else:
        wanted = [i.strip() for i in spec.split(",") if i.strip()]
        by_id = {r["id"]: r for r in pool}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            seen = {d["id"]: d["reason"] for d in (dropped or [])}
            raise SystemExit(f"--val-lines: {missing} are not usable val clips (split val, speech, exact transcript, "
                             f"tts_usable, not screened out, not repeated by a train clip: "
                             f"{ {i: seen[i] for i in missing if i in seen} }; --val-include-seen keeps the latter)")
        chosen = [by_id[i] for i in wanted]
    return [{"id": f"val_{r['id']}", "text": r["text"], "delivery": label_of.get(r["id"], (0, "natural"))[1],
             "source": "val", "heldout_id": r["id"]} for r in chosen]


def load_lines(args, rows, guard, excl, dmap) -> list[dict]:
    if args.text:
        lines = [{"id": args.line_id, "text": args.text, "delivery": args.delivery or "natural", "source": "new"}]
    else:
        lines, seen = [], set()
        for script in args.script:
            if not script.exists():
                raise SystemExit(f"takes script not found: {script} (or pass --text)")
            for i, raw in enumerate(C.read_json(script).get("lines", [])):
                text = (raw.get("text") or "").strip()
                if not text:
                    continue
                line_id = str(raw.get("id") or f"line{i + 1:03d}")
                if line_id in seen:
                    line_id = f"{line_id}-{i + 1}"
                seen.add(line_id)
                source = raw.get("source", "new")
                lines.append({"id": line_id, "text": text, "delivery": (raw.get("delivery") or "natural").strip(),
                              "source": source, "heldout_id": raw.get("heldout_id") if source == "heldout" else None})
        lines += val_lines(args.val_lines, rows, guard, excl, dmap, args.val_include_seen, args._val_dropped)
    sources = {s.strip() for s in (args.source or ",".join(SOURCES)).split(",") if s.strip()}
    bad = sources - set(SOURCES)
    if bad:
        raise SystemExit(f"bad --source {sorted(bad)}; use {SOURCES}")
    lines = [ln for ln in lines if ln["source"] in sources or (args.text and ln["source"] == "new")]
    if args.lines:
        wanted = {x.strip() for x in args.lines.split(",") if x.strip()}
        lines = [ln for ln in lines if ln["id"] in wanted]
    if args.deliveries:
        wanted = {x.strip() for x in args.deliveries.split(",") if x.strip()}
        lines = [ln for ln in lines if ln["delivery"] in wanted]
    if args.max_lines:
        lines = lines[: args.max_lines]
    if not lines:
        raise SystemExit("no lines to render")
    return lines


def build_plan(args, lines: list[dict], refs: References, guard: C.HeldOut, trained: set[str] | None):
    raw_seeds = [int(s) for s in args.seed_list.split(",") if s.strip()] if args.seed_list else \
        [args.base_seed + i for i in range(args.seeds)]
    seeds = list(dict.fromkeys(raw_seeds))  # unique, order kept (take ids must be unique)
    if not seeds:
        raise SystemExit("no seeds")
    forced = None
    if args.ref_id:
        reason = guard.reason(args.ref_id, None)
        if reason:
            raise SystemExit(f"refusing --ref-id {args.ref_id}: {reason}")
        forced = refs.check(args.ref_id, "cli", 0)
        if not forced:
            why = [s["reason"] for s in refs.skipped if s["id"] == args.ref_id and s["delivery"] == "cli"]
            raise SystemExit(f"refusing --ref-id {args.ref_id}: {why[-1] if why else 'not usable'}")
    plan, failures = [], []
    for line in lines:
        gt = line.get("heldout_id")
        if gt and trained is not None and gt in trained:
            failures.append({"line_id": line["id"], "error": f"the model was trained on {gt}, this line's ground "
                                                             f"truth; not rendered (it would be scored as unseen)"})
            continue
        if forced:
            cands, source = References.allowed([forced], line), "cli"
            if not cands:
                failures.append({"line_id": line["id"], "error": f"--ref-id {args.ref_id} would give away this "
                                                                 f"ground-truth line"})
                continue
        else:
            cands, source = refs.resolve(line["delivery"], line)
        if not cands:
            failures.append({"line_id": line["id"], "error": f"no usable reference for delivery {line['delivery']!r}"})
            continue
        rotate = max(1, min(args.refs_per_line, len(cands)))
        base = line["id"] if line["id"].endswith(line["delivery"]) else f"{line['id']}_{line['delivery']}"
        for i, seed in enumerate(seeds):
            ref = cands[i % rotate]
            aux_pool = refs.allowed(refs.candidates(args.aux_delivery), line) if args.aux_delivery else cands
            aux = [c for c in aux_pool if c["id"] != ref["id"]][: args.aux_refs]
            take_id = f"{base}_s{seed}" + (f"_r{ref['rank']}" if args.refs_per_line > 1 else "")
            plan.append({"take_id": safe_name(take_id), "line": line, "seed": seed, "ref": ref, "aux": aux,
                         "ref_source": source})
    return plan, failures


def shared_references(plan: list[dict]) -> list[str]:
    """Deliveries prompted by the same clip render alike (the prompt is GSV's delivery control): report every pair
    of deliveries whose planned references overlap."""
    used: dict[str, set[str]] = {}
    for item in plan:
        if item["ref_source"] in ("delivery_map", "cli-ref-for"):
            used.setdefault(item["line"]["delivery"], set()).add(item["ref"]["id"])
    labels = sorted(used)
    out = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            common = sorted(used[a] & used[b])
            if common:
                out.append(f"deliveries '{a}' and '{b}' share reference(s) {common}, so they will sound alike "
                           f"(--ref-for LABEL=ID[,ID] gives one of them its own prompts)")
    return out


def split_method(args, text: str) -> str:
    if args.text_split_method != "auto":
        return args.text_split_method
    return "cut0" if len(text) <= 220 else "cut4"  # whole line in one pass keeps prosody; long text by sentence


# ----------------------------------------------------------------------------------------------- synthesis
def ensure_noise_scale(tts, noise_scale: float) -> None:
    """TTS.run never passes noise_scale to SynthesizerTrn.decode (upstream default 0.5); wrap decode to set it."""
    decode = tts.vits_model.decode
    if getattr(decode, "_lab_noise_scale", None) == noise_scale:
        return
    base = getattr(decode, "_lab_base", decode)

    def wrapped(*a, **k):
        k.setdefault("noise_scale", noise_scale)
        return base(*a, **k)

    wrapped._lab_noise_scale = noise_scale
    wrapped._lab_base = base
    tts.vits_model.decode = wrapped


def trim_padding(x, sr: int, keep_s: float = 0.05):
    """Drop the exact-zero padding TTS appends after the last fragment (fragment_interval), keep keep_s."""
    import numpy as np

    nz = np.flatnonzero(x)
    if nz.size == 0:
        return x
    return x[: min(x.size, int(nz[-1]) + 1 + int(keep_s * sr))]


def load_selection(path: Path) -> dict:
    """selection.json from freeze_selection.py; the weights must still be the files that were chosen."""
    if not path.exists():
        raise SystemExit(f"{path} not found: freeze the checkpoint choice with freeze_selection.py first")
    sel = C.read_json(path)
    for key in ("gpt", "sovits"):
        p = Path(sel[key])
        if not p.exists():
            raise SystemExit(f"selection {key} weights missing: {p}")
        if sel.get(f"{key}_sha256") and C.sha256_file(p) != sel[f"{key}_sha256"]:
            raise SystemExit(f"selection {key} weights {p} changed since the choice was frozen (sha256 mismatch)")
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--exp", default=None, help=f"experiment name, e.g. {C.exp_name('v2ProPlus', 'original')}")
    g.add_argument("--audio", choices=C.AUDIO_MODES, default=None, help="with --version, derives --exp")
    g.add_argument("--version", choices=C.VERSIONS, default="v2ProPlus")
    g.add_argument("--exp-root", default=str(C.EXP_ROOT), help="where experiment folders live (training data)")
    g.add_argument("--weights-root", default=str(C.WEIGHTS_ROOT))
    g.add_argument("--gpt", default=None, help="explicit GPT .ckpt (overrides --exp lookup)")
    g.add_argument("--sovits", default=None, help="explicit SoVITS .pth (overrides --exp lookup)")
    g.add_argument("--gpt-epoch", type=int, default=None)
    g.add_argument("--sovits-epoch", type=int, default=None)
    g.add_argument("--gpt-tag", default="", help="GPT arm, e.g. _dpo (weights from GPT<tag>)")
    g.add_argument("--zero-shot", action="store_true", help="pretrained s1v3 + s2G<version>, no fine-tune")
    g.add_argument("--selection", default=None,
                   help=f"frozen checkpoint choice ({C.SELECTION} from freeze_selection.py)")
    g = ap.add_argument_group("lines and references")
    g.add_argument("--script", default=str(C.TAKES_SCRIPT), help="takes script(s), comma list")
    g.add_argument("--source", default=None, help="comma list of line sources: new, heldout, val (default all)")
    g.add_argument("--val-lines", default="none",
                   help="add val clips as ground-truth lines: none (default) | all | N | comma list of ids")
    g.add_argument("--val-include-seen", action="store_true",
                   help="keep val lines that a train clip repeats (not unseen performances)")
    g.add_argument("--allow-unfrozen-heldout", action="store_true",
                   help="render held-out test lines from fine-tuned weights without --selection")
    g.add_argument("--delivery-map", default=str(C.DELIVERY_MAP))
    g.add_argument("--no-delivery-map", action="store_true", help="use corpus-picked neutral references only")
    g.add_argument("--text", default=None, help="render one ad-hoc line instead of the script")
    g.add_argument("--delivery", default=None, help="delivery label for --text (default natural)")
    g.add_argument("--line-id", default="adhoc")
    g.add_argument("--lines", default=None, help="comma list of line ids")
    g.add_argument("--deliveries", default=None, help="comma list of delivery labels")
    g.add_argument("--max-lines", type=int, default=0)
    g.add_argument("--ref-audio", choices=C.AUDIO_MODES, default="original", help="prompt audio variant")
    g.add_argument("--allow-restored", action="store_true",
                   help="permit restored prompt audio (DECISIONS.md: the user chose original audio)")
    g.add_argument("--ref-id", default=None, help="force one train/val reference clip for every line")
    g.add_argument("--ref-for", action="append", default=None, metavar="LABEL=ID[,ID]",
                   help="replace a delivery's ranked reference list (repeatable)")
    g.add_argument("--refs-per-line", type=int, default=1, help="rotate seeds through the top R references")
    g.add_argument("--aux-refs", type=int, default=None,
                   help="extra references averaged into the timbre embedding (default 0, or the selection's)")
    g.add_argument("--aux-delivery", default=None, help="take aux refs from this delivery (e.g. natural)")
    g.add_argument("--max-ref-wer", type=float, default=0.2)
    g.add_argument("--strict-refs", action="store_true", help="fail if the delivery map names a held-out clip")
    g.add_argument("--no-corpus-fallback", action="store_true")
    g.add_argument("--exclusions", default=str(C.EXCLUSIONS), help="speaker-screen exclusions JSON")
    g.add_argument("--no-exclusions", action="store_true",
                   help="DANGER: allow prompts the speaker screen attributes to other characters")
    g.add_argument("--text-overrides", default=str(C.TEXT_OVERRIDES), help="prompt transcript corrections")
    g.add_argument("--no-text-overrides", action="store_true")
    g = ap.add_argument_group("sampling")
    g.add_argument("--seeds", type=int, default=4)
    g.add_argument("--base-seed", type=int, default=1)
    g.add_argument("--seed-list", default=None, help="explicit comma list of seeds")
    g.add_argument("--preset", choices=sorted(PRESETS), default=None, help="default 'default' (or the selection's)")
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--repetition-penalty", type=float, default=None)
    g.add_argument("--speed", type=float, default=1.0)
    g.add_argument("--sample-steps", type=int, default=32, help="v3/v4 only; recorded for completeness")
    g.add_argument("--fragment-interval", type=float, default=0.3)
    g.add_argument("--text-split-method", choices=SPLIT_METHODS, default="auto")
    g.add_argument("--noise-scale", type=float, default=0.5, help="SoVITS decode noise (upstream fixed 0.5)")
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--no-parallel-infer", action="store_true")
    g.add_argument("--keep-tail", action="store_true", help="keep the zero padding after the last fragment")
    g = ap.add_argument_group("run")
    g.add_argument("--run", default=None)
    g.add_argument("--out-root", default=str(C.TAKES_ROOT))
    g.add_argument("--resume", action="store_true", help="continue an existing run, skipping finished takes")
    g.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    g.add_argument("--no-half", action="store_true", help="fp32 on GPU")
    g.add_argument("--mem-gb", type=float, default=4.0)
    g.add_argument("--reserve-gb", type=float, default=8.0)
    g.add_argument("--wait-min", "--wait-vram-min", dest="wait_min", type=float, default=0.0,
                   help="wait up to N minutes for enough free VRAM / host commit")
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--lexicon", default=str(C.LEXICON))
    g.add_argument("--no-lexicon", action="store_true")
    g.add_argument("--dry-run", action="store_true", help="resolve lines/references and print the plan only")
    args = ap.parse_args()
    device = C.normalize_device(args.device)
    for key in ("exp_root", "weights_root", "gpt", "sovits", "delivery_map", "out_root", "exclusions", "lexicon",
                "text_overrides", "selection"):
        setattr(args, key, C.abspath(getattr(args, key)))
    args.script = [C.abspath(s) for s in str(args.script).split(",") if s.strip()]
    args.gpt_tag = C.check_tag(args.gpt_tag)
    args._val_dropped = []
    if device != "cpu":
        C.gpu_visible_devices(device)  # fail fast: never re-expose a GPU the caller hid
    C.require_original_audio(args.ref_audio, args.allow_restored, "prompting")

    if not args.dry_run:
        C.reexec_in_gsv_venv(__file__)

    # ---- models
    selection = None
    if args.selection:
        clash = [f for f, v in (("--exp", args.exp), ("--audio", args.audio), ("--gpt", args.gpt),
                                ("--sovits", args.sovits), ("--gpt-epoch", args.gpt_epoch),
                                ("--sovits-epoch", args.sovits_epoch), ("--gpt-tag", args.gpt_tag),
                                ("--zero-shot", args.zero_shot)) if v]
        if clash:
            raise SystemExit(f"--selection fixes the model; drop {clash}")
        selection = load_selection(args.selection)
        args.exp, args.gpt, args.sovits = selection["exp"], Path(selection["gpt"]), Path(selection["sovits"])
        args.gpt_tag = selection.get("gpt_tag") or ""
        for key in ("aux_refs", "aux_delivery", "preset"):
            if getattr(args, key) is None and selection.get(key) is not None:
                setattr(args, key, selection[key])
    args.aux_refs = args.aux_refs or 0
    args.preset = args.preset or "default"
    sampling = dict(PRESETS[args.preset])
    for key in ("top_k", "top_p", "temperature", "repetition_penalty"):
        if getattr(args, key) is not None:
            sampling[key] = getattr(args, key)

    exp = args.exp or (C.exp_name(args.version, args.audio) if args.audio else None)
    version_match = re.search(r"_(v2ProPlus|v2Pro)_", f"_{exp}_") if exp else None
    version = version_match.group(1) if version_match else args.version
    ref_audio = args.ref_audio
    weights_note = None
    if args.zero_shot:
        gpt, sovits = str(C.S1_PRETRAINED), str(C.S2_PRETRAINED[version][0])
        gpt_epoch = sovits_epoch = None
    else:
        if not exp and not (args.gpt and args.sovits):
            raise SystemExit("pass --exp (or --audio), explicit --gpt and --sovits, --selection or --zero-shot")
        try:
            sov_w = C.list_weights(exp, args.weights_root)["sovits"] if exp and not args.sovits else []
            gpt_w = C.list_weights(exp, args.weights_root, args.gpt_tag)["gpt"] if exp and not args.gpt else []
            gpt = str(args.gpt) if args.gpt else C.pick_weight(gpt_w, args.gpt_epoch, f"GPT{args.gpt_tag}")
            sovits = str(args.sovits) if args.sovits else C.pick_weight(sov_w, args.sovits_epoch, "SoVITS")
        except SystemExit as exc:
            if not args.dry_run:
                raise
            gpt = sovits = None  # a dry run only resolves lines and references
            weights_note = str(exc)
        m = re.search(r"-e(\d+)\.ckpt$", gpt or "")
        gpt_epoch = int(m.group(1)) if m else None
        m = re.search(r"_e(\d+)_s\d+\.pth$", sovits or "")
        sovits_epoch = int(m.group(1)) if m else None
    for path in (gpt, sovits):
        if path and not Path(path).exists():
            raise SystemExit(f"weights not found: {path}")

    # ---- training data of the model (ground-truth lines it trained on are not rendered)
    trained: set[str] | None
    exp_dir = Path(selection["exp_dir"]) if selection and selection.get("exp_dir") else (args.exp_root / exp if exp else None)
    if args.zero_shot:
        trained = set()
    elif exp_dir is not None and (exp_dir / "2-name2text.txt").exists():
        trained = C.trained_clip_ids(exp_dir)
    else:
        trained = None

    # ---- lines, references, plan
    corpus_rows = C.load_corpus()
    guard = C.HeldOut(corpus_rows)
    overrides = C.load_text_overrides(None if args.no_text_overrides else args.text_overrides, corpus_rows)
    rows = C.apply_text_overrides(corpus_rows, overrides)
    if args.no_exclusions:
        C.log("WARNING: --no-exclusions: references are not checked against the speaker screen")
    excl = None if args.no_exclusions else C.Exclusions(corpus_rows, args.exclusions)
    dmap_path = None if args.no_delivery_map else args.delivery_map
    dmap = None
    if dmap_path:
        if not dmap_path.exists():
            raise SystemExit(f"delivery map not found: {dmap_path} (or pass --no-delivery-map)")
        dmap = C.read_json(dmap_path)
    ref_for = parse_ref_for(args.ref_for)
    for label, ids in ref_for.items():
        for rid in ids:
            reason = guard.reason(rid, None)
            if reason:
                raise SystemExit(f"refusing --ref-for {label}={rid}: {reason}")
    refs = References(dmap, rows, guard, excl, ref_audio, args.max_ref_wer, not args.no_corpus_fallback, ref_for)
    lines = load_lines(args, rows, guard, excl, dmap)
    heldout_lines = [ln["id"] for ln in lines if ln["source"] == "heldout"]
    if heldout_lines and not args.zero_shot and selection is None and not args.allow_unfrozen_heldout:
        raise SystemExit(f"refusing to render held-out test lines {heldout_lines} from fine-tuned weights before the "
                         f"checkpoint choice is frozen: select on --source new,val (with --val-lines), freeze it "
                         f"with freeze_selection.py, then render with --selection (or --allow-unfrozen-heldout)")
    plan, failures = build_plan(args, lines, refs, guard, trained)
    for label, ids in ref_for.items():
        if not refs.candidates(label):
            reasons = [s for s in refs.skipped if s["delivery"] == label]
            raise SystemExit(f"--ref-for {label}: none of {ids} is a usable reference: {reasons}")
    heldout_hits = [s for s in refs.skipped if s["heldout"]]
    if heldout_hits and args.strict_refs:
        raise SystemExit(f"delivery map names held-out clips: {heldout_hits}")
    warnings = shared_references(plan)
    if trained is None and any(ln.get("heldout_id") for ln in lines):
        warnings.append("training data of these weights is unknown (no experiment folder): ground-truth lines are "
                        "rendered without the trained-on check")
    for w in warnings:
        C.log(f"WARNING: {w}")
    lexicon = C.load_lexicon(None if args.no_lexicon else args.lexicon)
    dur = duration_model(corpus_rows)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.run:
        run = safe_name(args.run)
    elif args.zero_shot:
        run = f"zeroshot_{version}_{args.preset}_{stamp}"
    else:
        run = f"{exp or 'custom'}_gpt{args.gpt_tag}-e{gpt_epoch}_sovits-e{sovits_epoch}_{args.preset}_{stamp}"
    run_dir = args.out_root / run

    summary = {
        "run": run, "run_dir": str(run_dir), "gpt": gpt, "sovits": sovits, "version": version,
        "weights_note": weights_note, "ref_audio": ref_audio, "lines": len(lines),
        "lines_by_source": {s: sum(ln["source"] == s for ln in lines) for s in SOURCES}, "takes_planned": len(plan),
        "failures": failures, "warnings": warnings, "val_lines_dropped": args._val_dropped,
        "sampling": sampling, "aux_refs": args.aux_refs,
        "aux_delivery": args.aux_delivery, "trained_on_clips": None if trained is None else len(trained),
        "skipped_references": refs.skipped, "duration_check": dur,
        "delivery_map_sha256": C.sha256_file(dmap_path) if dmap_path else None,
        "script_sha256": None if args.text else [C.sha256_file(s) for s in args.script],
        "references_by_delivery": {},
        "references_by_line": {},
    }
    for item in plan:
        d = item["line"]["delivery"]
        summary["references_by_delivery"].setdefault(d, {"source": item["ref_source"], "used": []})
        if item["ref"]["id"] not in summary["references_by_delivery"][d]["used"]:
            summary["references_by_delivery"][d]["used"].append(item["ref"]["id"])
        per_line = summary["references_by_line"].setdefault(item["line"]["id"], [])
        entry = item["ref"]["id"] + (f" +aux {[a['id'] for a in item['aux']]}" if item["aux"] else "")
        if entry not in per_line:
            per_line.append(entry)
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if not plan:
        raise SystemExit(f"nothing to render: {failures}")

    manifest_path = run_dir / "manifest.json"
    done: dict[str, dict] = {}
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(f"{manifest_path} exists; pass --resume or choose another --run")
        previous = C.read_json(manifest_path)
        done = {t["take_id"]: t for t in previous.get("takes", []) if Path(t["wav"]).exists()}
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- runtime
    commit = C.require_commit(C.COMMIT_GB["infer"] + (1.5 if device != "cpu" else 0.0), "inference", args.wait_min)
    C.bootstrap_gsv_runtime(device)
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import yaml

    C.assert_device(device)
    vram = C.cap_gpu_memory(args.mem_gb, args.reserve_gb, args.wait_min) if device != "cpu" else None
    torch.set_num_threads(max(1, args.threads))
    half = device != "cpu" and not args.no_half
    from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

    lex_n = C.install_g2p_lexicon(lexicon)
    cfg_path = run_dir / "tts_infer.yaml"  # TTS writes its config back here, never into the repo
    cfg_path.write_text(yaml.safe_dump({"custom": {
        "device": "cpu" if device == "cpu" else "cuda", "is_half": half, "version": version,
        "t2s_weights_path": gpt, "vits_weights_path": sovits,
        "bert_base_path": str(C.BERT_DIR), "cnhuhbert_base_path": str(C.HUBERT_DIR)}}), encoding="utf-8")
    t_load = time.time()
    tts = TTS(TTS_Config(str(cfg_path)))
    for kind, wanted, got in (("GPT", gpt, tts.configs.t2s_weights_path), ("SoVITS", sovits, tts.configs.vits_weights_path)):
        if Path(str(got)).resolve() != Path(wanted).resolve():  # TTS_Config silently falls back to its defaults
            raise SystemExit(f"TTS loaded {kind} weights {got}, not the requested {wanted}; refusing to label takes")
    loaded_version = tts.configs.version
    if loaded_version != version:
        C.log(f"warning: weights report model version {loaded_version}, expected {version}")
    C.log(f"models loaded in {time.time() - t_load:.1f}s ({loaded_version}, half={half}, lexicon {lex_n} entries)")
    encoders = C.CONDITIONING_ENCODERS.get(loaded_version, [])

    meta = {
        "engine": ENGINE, "run": run, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": device,
        "model_version": loaded_version, "conditioning_encoders": encoders,
        "gpt_weights": gpt, "sovits_weights": sovits, "gpt_tag": args.gpt_tag, "zero_shot": args.zero_shot,
        "exp": exp, "exp_dir": str(exp_dir) if exp_dir else None,
        "trained_on_clips": None if trained is None else len(trained),
        "selection": ({"path": str(args.selection), "sha256": C.sha256_file(args.selection), **selection}
                      if selection else None),
        "script": None if args.text else [str(s) for s in args.script], "val_lines": args.val_lines,
        "val_lines_dropped": args._val_dropped,
        "source_filter": args.source, "delivery_map": str(dmap_path) if dmap_path else None,
        "delivery_map_sha256": C.sha256_file(dmap_path) if dmap_path else None,
        "script_sha256": None if args.text else [C.sha256_file(s) for s in args.script],
        "ref_audio_mode": ref_audio, "ref_for": ref_for, "lexicon": lexicon["path"],
        "lexicon_sha256": lexicon["sha256"], "text_overrides": overrides["path"],
        "text_overrides_sha256": overrides["sha256"], "vram": vram, "host_commit": commit,
        "exclusions": excl.describe() if excl else None, "duration_check": dur, "warnings": warnings,
        "refused_heldout_references": [s for s in refs.skipped if s["heldout"]],
        "args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                 for k, v in vars(args).items() if not k.startswith("_")},
    }
    takes = list(done.values())
    failures = list(failures)

    def write_manifest() -> None:
        flagged = sum(1 for t in takes if t["params"].get("duration_flag") != "ok")
        C.write_json(manifest_path, {**meta, "skipped_references": refs.skipped, "failures": failures,
                                     "duration_flagged_takes": flagged, "takes": takes})

    write_manifest()
    plan.sort(key=lambda it: (it["ref"]["audio"], tuple(a["audio"] for a in it["aux"]), it["line"]["id"], it["seed"]))
    t_run = time.time()
    for n, item in enumerate(plan, 1):
        if item["take_id"] in done:
            continue
        line, ref = item["line"], item["ref"]
        target = C.clean_line_text(C.respell(line["text"], lexicon))
        prompt = C.clean_line_text(C.respell(ref["text"], lexicon))
        method = split_method(args, target)
        inputs = {
            "text": target, "text_lang": C.LANG, "ref_audio_path": ref["audio"],
            "aux_ref_audio_paths": [a["audio"] for a in item["aux"]], "prompt_text": prompt, "prompt_lang": C.LANG,
            "top_k": sampling["top_k"], "top_p": sampling["top_p"], "temperature": sampling["temperature"],
            "text_split_method": method, "batch_size": args.batch_size, "batch_threshold": 0.75,
            "split_bucket": True, "speed_factor": args.speed, "fragment_interval": args.fragment_interval,
            "seed": item["seed"], "parallel_infer": not args.no_parallel_infer,
            "repetition_penalty": sampling["repetition_penalty"], "sample_steps": args.sample_steps,
            "super_sampling": False, "return_fragment": False, "streaming_mode": False,
        }
        t0 = time.time()
        try:
            ensure_noise_scale(tts, args.noise_scale)
            chunks, sr = [], None
            for sr, audio in tts.run(inputs):
                chunks.append(np.asarray(audio))
            pcm_native = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
            if sr is None or not np.any(pcm_native):
                raise RuntimeError("TTS returned silence")
        except Exception as exc:  # TTS.run re-initialises its models after an error; keep going
            failures.append({"take_id": item["take_id"], "line_id": line["id"], "seed": item["seed"],
                             "error": f"{type(exc).__name__}: {exc}"})
            C.log(f"[{n}/{len(plan)}] {item['take_id']} FAILED: {exc}")
            write_manifest()
            continue
        synth_s = time.time() - t0
        x = pcm_native.astype(np.float32) / 32768.0
        if not args.keep_tail:
            x = trim_padding(x, sr)
        if sr != C.OUT_SR:
            x = soxr.resample(x, sr, C.OUT_SR, quality="VHQ")
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.999:
            x = x * (0.999 / peak)
            peak = 0.999
        wav_path = run_dir / f"{item['take_id']}.wav"
        sf.write(str(wav_path), np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16), C.OUT_SR,
                 subtype="PCM_16")
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        duration_s = round(x.size / C.OUT_SR, 3)
        expected = max(1, n_chars(line["text"])) * dur["s_per_char"] / max(1e-6, args.speed)
        ratio = round(duration_s / expected, 3)
        flag = ("short: possible skipped words" if ratio < dur["flag_below"] else
                "long: possible repeat or stall" if ratio > dur["flag_above"] else "ok")
        params = {
            "model_version": loaded_version, "conditioning_encoders": encoders, "gpt_weights": gpt,
            "sovits_weights": sovits, "gpt_epoch": gpt_epoch, "sovits_epoch": sovits_epoch, "gpt_tag": args.gpt_tag,
            "zero_shot": args.zero_shot, "selection_frozen": selection is not None, "line_source": line["source"],
            "prompt_audio": ref["audio"], "prompt_text": prompt, "prompt_duration_s": ref["duration"],
            "reference_rank": ref["rank"], "reference_delivery": ref["delivery"], "reference_source": item["ref_source"],
            "reference_why": ref["why"], "reference_note": ref["note"], "ref_audio_mode": ref_audio,
            "aux_reference_ids": [a["id"] for a in item["aux"]], "aux_audio": [a["audio"] for a in item["aux"]],
            "aux_delivery": args.aux_delivery, "preset": args.preset, **sampling, "speed_factor": args.speed,
            "sample_steps": args.sample_steps, "fragment_interval": args.fragment_interval,
            "text_split_method": method, "noise_scale": args.noise_scale,
            "parallel_infer": not args.no_parallel_infer, "batch_size": args.batch_size, "g2p_text": target,
            "lexicon_sha256": lexicon["sha256"], "native_sr": sr, "sample_rate": C.OUT_SR, "duration_s": duration_s,
            "expected_duration_s": round(expected, 3), "duration_ratio": ratio, "duration_flag": flag,
            "peak_dbfs": round(20 * math.log10(max(peak, 1e-9)), 2),
            "rms_dbfs": round(20 * math.log10(max(rms, 1e-9)), 2),
            "synth_seconds": round(synth_s, 2), "device": device, "half": half,
        }
        take = {"take_id": item["take_id"], "engine": ENGINE, "line_id": line["id"], "text": line["text"],
                "delivery": line["delivery"], "wav": str(wav_path), "reference_id": ref["id"],
                "seed": item["seed"], "params": params}
        if line.get("heldout_id"):
            take["heldout_id"] = line["heldout_id"]
        takes.append(take)
        write_manifest()
        C.log(f"[{n}/{len(plan)}] {item['take_id']}: {duration_s} s audio in {synth_s:.1f}s "
              f"(ref {ref['id']}, {method}, duration ratio {ratio}{'' if flag == 'ok' else ' ' + flag.upper()})")
    meta["render_seconds"] = round(time.time() - t_run, 1)
    meta.update(C.peak_memory_report(device))
    write_manifest()
    C.log(f"{len(takes)} takes, {len(failures)} failures, "
          f"{sum(1 for t in takes if t['params']['duration_flag'] != 'ok')} duration-flagged -> {manifest_path}")
    if failures and not takes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
