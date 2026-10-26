"""Choose VoxCPM2 checkpoints from a render sweep: per mode, paired, with an expressive-range check (CPU only).

  set PY=%LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe
  %PY% select_checkpoint.py --scorecard ...\\<lab>\\eval\\<eval run>\\scorecard.json --name sweep1
  %PY% select_checkpoint.py --name sweep1 --ab <run folder X> <run folder Y>      (blind listening pack)
  %PY% select_checkpoint.py --tally ...\\<lab>\\voxcpm\\listening\\<pack>        (after the user filled sheet.csv)
  (run the evaluation with --group-by run so its own tables are per adapter and mode too)
  The character is VOICE_LAB_CHARACTER (unset = baodur -> <lab> = KotorMods\\Voices\\BaoDurLab; see common.py).

Why: the evaluation's composite (baodur-lab\\eval\\aggregate.py) ranks takes by speaker similarity first, then
WER and prosody distance. That catches identity loss, text-ignoring overfit and cut endings, but not a clone
that sounds like the character and has lost the emotional range: LoRA training tends to raise similarity while
narrowing prosody, so similarity-first ranking favours late, flatter checkpoints. This tool adds, per
conditioning mode (render.py writes one run folder per adapter and mode, <adapter>--<mode>):
1. Paired comparison: only takes with the same take_id (same line, mode, reference slot and seed; render.py
   gives every adapter the same seeds) are compared. Per line, the paired differences are averaged over
   seeds; held-out lines (real recording to compare against) weigh --heldout-weight (default 2); the 95% CI
   comes from a bootstrap over lines (seeds of one line are not independent). Metrics from the scorecard:
   composite, similarity, WER, prosody distance, gate pass. The leader is the best weighted composite among
   adapters that rendered the full line set (Bao-Dur's July baseline skips the 5 held-out lines it trained on
   and is compared on the common lines only).
2. Expressive range (default mode idref_instr, where only the instruction changes between deliveries): per
   delivery label, the median of F0 median, F0 IQR, speaking rate and loudness over the adapter's takes;
   the spread (sd) of those medians across labels divided by the same spread among the real delivery_map
   reference clips (train/val only, screened, top --range-refs per label, trimmed 16 kHz cache). A ratio
   below --range-min (default 0.5) flags a flattened adapter. Features: build_manifest.delivery_features at
   16 kHz from the WAVs (works without a scorecard). The no-LoRA control shows what the base model keeps.
3. Shortlist: the leader is the best weighted composite among full-coverage adapters that keep range ratio
   >= --range-min (a flattened adapter that tops the composite is reported as such, never used as the
   yardstick); shortlisted are range-keeping adapters whose paired composite CI upper bound vs the leader is
   >= -(--tolerance, default 0.02) and whose gate pass rate is within 10 points of the leader's. The user's
   ears decide among the shortlist: --ab writes a blind pack (pair_NN_x/y.wav in random order, the real
   recording for held-out lines as pair_NN_original.wav, sheet.csv to fill in, key.json hidden mapping),
   --tally counts the answers with a two-sided sign test.
Outputs: <lab>\\voxcpm\\selection\\<name>\\selection.json + selection.md, <lab>\\voxcpm\\listening\\<name>\\.
Reads takes, the delivery map and (for held-out pairs in a listening pack) the real held-out recordings, which
is evaluation use; nothing here trains, conditions or writes outside <lab>\\voxcpm\\.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RANGE_FEATURES = ("f0_med", "f0_iqr", "rate", "loud_db")
METRICS = (("composite", +1), ("similarity", +1), ("wer", -1), ("prosody_distance", -1))
SELECTION_ROOT = C.DATA_ROOT / "selection"
LISTENING_ROOT = C.DATA_ROOT / "listening"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scorecard", type=Path, help="evaluation scorecard.json (similarity/WER/prosody/composite)")
    p.add_argument("--takes-root", type=Path, default=C.TAKES_ROOT, help="folder holding <run>\\manifest.json")
    p.add_argument("--runs", default="", help="regex on run folder names to include (default all)")
    p.add_argument("--name", default="", help="output name (default: a timestamp)")
    p.add_argument("--out-root", type=Path, default=SELECTION_ROOT)
    p.add_argument("--heldout-weight", type=float, default=2.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--range-modes", default="idref_instr,ref_prompt",
                   help="modes to run the expressive-range check on (idref_instr is the decisive one)")
    p.add_argument("--range-refs", type=int, default=3, help="real reference clips per delivery label")
    p.add_argument("--range-min", type=float, default=0.5)
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="composite margin: a run whose CI upper bound is >= -tolerance vs the leader stays shortlisted")
    p.add_argument("--delivery-map", type=Path, default=C.DELIVERY_MAP)
    p.add_argument("--ab", nargs=2, metavar=("RUN_X", "RUN_Y"), help="write a blind A/B pack for two run folders")
    p.add_argument("--ab-root", type=Path, default=LISTENING_ROOT)
    p.add_argument("--ab-max", type=int, default=0, help="at most N pairs (0 = all paired takes)")
    p.add_argument("--no-original", action="store_true", help="do not add the real recording of held-out lines")
    p.add_argument("--tally", type=Path, help="count a filled listening pack (its folder)")
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()
    if not (args.scorecard or args.ab or args.tally or args.range_modes):
        p.error("nothing to do")
    args.name = args.name or time.strftime("sel_%Y%m%d_%H%M%S")
    return args


# --------------------------------------------------------------------------------------------- loading
def load_runs(takes_root: Path, pattern: str) -> dict[str, dict]:
    """run folder -> {"manifest": path, "run_label": engine/folder, "adapter", "takes": [...]}."""
    rx = re.compile(pattern) if pattern else None
    runs = {}
    for manifest in sorted(takes_root.glob("*/manifest.json")):
        folder = manifest.parent.name
        if rx and not rx.search(folder):
            continue
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        takes = []
        for take in doc.get("takes", []):
            params = take.get("params") or {}
            m = re.search(r"-s(\d+)$", str(take["take_id"]))
            takes.append({**take, "mode": params.get("mode") or doc.get("mode") or "?",
                          "seed_index": int(m.group(1)) if m else None,
                          "path": manifest.parent / (params.get("trimmed_output_wav") or take["wav"])})
        runs[folder] = {"manifest": str(manifest), "run_label": f"{manifest.parent.parent.name}/{folder}",
                        "adapter": doc.get("adapter_label") or doc.get("run") or folder,
                        "adapter_kind": (doc.get("adapter") or {}).get("kind"),
                        "step": next((t["params"].get("lora_step") for t in doc.get("takes", [])
                                      if (t.get("params") or {}).get("lora_step") is not None), None),
                        "takes": takes}
    return runs


def attach_scores(runs: dict, scorecard: Path) -> list[str]:
    doc = json.loads(scorecard.read_text(encoding="utf-8"))
    by_key = {(t.get("run_label"), str(t.get("take_id"))): t for t in doc.get("takes", [])}
    notes = []
    for folder, run in runs.items():
        missing = 0
        for take in run["takes"]:
            row = by_key.get((run["run_label"], str(take["take_id"])))
            if row is None:
                missing += 1
                continue
            take["scores"] = {"composite": row.get("composite"), "similarity": row.get("similarity"),
                              "wer": row.get("wer"), "prosody_distance": row.get("prosody_distance"),
                              "gate_pass": bool(row.get("gate_pass")), "violations": row.get("violations") or []}
        if missing:
            notes.append(f"{folder}: {missing}/{len(run['takes'])} takes are not in the scorecard")
    return notes


# --------------------------------------------------------------------------------------------- statistics
def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def paired(a: list[dict], b: list[dict], metric: str, heldout_weight: float, n_boot: int, rng) -> dict | None:
    """Mean (b - a) over lines of the per-line mean paired difference, with a bootstrap-over-lines 95% CI."""
    sa = {t["take_id"]: t for t in a if (t.get("scores") or {}).get(metric) is not None}
    lines: dict[str, list[float]] = {}
    heldout: dict[str, bool] = {}
    for t in b:
        other = sa.get(t["take_id"])
        value = (t.get("scores") or {}).get(metric)
        if other is None or value is None:
            continue
        lines.setdefault(t["line_id"], []).append(float(value) - float(other["scores"][metric]))
        heldout[t["line_id"]] = bool(t.get("heldout_id"))
    if not lines:
        return None
    ids = sorted(lines)
    diffs = np.array([np.mean(lines[i]) for i in ids])
    weights = np.array([heldout_weight if heldout[i] else 1.0 for i in ids])
    point = weighted_mean(diffs, weights)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        boots.append(weighted_mean(diffs[pick], weights[pick]))
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(ids) > 1 else (point, point)
    return {"mean_diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)], "lines": len(ids),
            "pairs": int(sum(len(v) for v in lines.values()))}


def run_summary(takes: list[dict], heldout_weight: float) -> dict:
    out = {"takes": len(takes), "lines": len({t["line_id"] for t in takes})}
    scored = [t for t in takes if t.get("scores")]
    out["scored_takes"] = len(scored)
    if not scored:
        return out
    for metric, _ in METRICS:
        per_line: dict[str, list[float]] = {}
        held: dict[str, bool] = {}
        for t in scored:
            if t["scores"].get(metric) is not None:
                per_line.setdefault(t["line_id"], []).append(float(t["scores"][metric]))
                held[t["line_id"]] = bool(t.get("heldout_id"))
        if per_line:
            ids = sorted(per_line)
            out[f"{metric}_weighted_mean"] = round(weighted_mean(
                np.array([np.mean(per_line[i]) for i in ids]),
                np.array([heldout_weight if held[i] else 1.0 for i in ids])), 4)
    out["gate_pass_rate"] = round(sum(1 for t in scored if t["scores"]["gate_pass"]) / len(scored), 3)
    out["violations"] = sorted({str(v)[:120] for t in scored for v in t["scores"]["violations"]})[:5]
    return out


# --------------------------------------------------------------------------------------------- range
def features_of(path: Path, text: str, cache: dict) -> dict:
    import build_manifest as B

    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    if key not in cache:
        audio, rate = C.read_mono(path)
        audio16 = C.resample(audio, rate, C.ENCODER_SR)
        cache[key] = B.delivery_features(audio16, C.ENCODER_SR, C.clean_text(text))
    return cache[key]


def label_spread(per_label: dict[str, list[dict]]) -> dict:
    medians = {label: {f: float(np.nanmedian([x[f] for x in feats])) for f in RANGE_FEATURES}
               for label, feats in per_label.items() if feats}
    spread = {}
    for f in RANGE_FEATURES:
        values = [m[f] for m in medians.values() if not math.isnan(m[f])]
        spread[f] = float(np.std(values, ddof=1)) if len(values) >= 3 else float("nan")
    return {"labels": sorted(medians), "medians": medians, "spread": spread}


def real_reference_spread(labels: list[str], args, corpus: dict, screen: dict, cache: dict) -> dict:
    deliveries = {k.casefold(): v for k, v in json.loads(args.delivery_map.read_text(encoding="utf-8"))["deliveries"].items()}
    blocked = set(screen["confirmed"]) | screen["uncertain"]
    per_label: dict[str, list[dict]] = {}
    used: dict[str, list[str]] = {}
    for label in labels:
        for ref in (deliveries.get(label) or {}).get("references", []):
            rid = ref.get("id")
            row = corpus.get(rid)
            if not row or row["split"] not in ("train", "val") or rid in blocked or row["kind"] != "speech":
                continue
            path = C.cache_dir("original") / f"{rid}.wav"
            if not path.exists():
                continue
            per_label.setdefault(label, []).append(features_of(path, row["text"], cache))
            used.setdefault(label, []).append(rid)
            if len(used[label]) >= args.range_refs:
                break
    out = label_spread(per_label)
    out["clips"] = used
    return out


def range_check(takes: list[dict], real: dict, cache: dict) -> dict:
    per_label: dict[str, list[dict]] = {}
    for t in takes:
        if t["path"].exists():
            per_label.setdefault(t["delivery"], []).append(features_of(t["path"], t["text"], cache))
    mine = label_spread(per_label)
    ratios = {}
    for f in RANGE_FEATURES:
        r, m = real["spread"].get(f, float("nan")), mine["spread"].get(f, float("nan"))
        ratios[f] = round(m / r, 3) if r and not math.isnan(r) and not math.isnan(m) and r > 1e-9 else None
    valid = [v for v in ratios.values() if v is not None]
    return {"labels": mine["labels"], "takes_measured": sum(len(v) for v in per_label.values()),
            "spread": {f: round(v, 3) for f, v in mine["spread"].items() if not math.isnan(v)},
            "ratio_to_real": ratios, "range_ratio": round(float(np.mean(valid)), 3) if valid else None}


# --------------------------------------------------------------------------------------------- listening
def write_ab(args, runs: dict, corpus: dict) -> Path:
    x, y = args.ab
    for name in (x, y):
        if name not in runs:
            raise SystemExit(f"--ab: run folder {name} not found under {args.takes_root} (or filtered by --runs)")
    tx = {t["take_id"]: t for t in runs[x]["takes"] if t["path"].exists()}
    common = [t for t in runs[y]["takes"] if t["take_id"] in tx and t["path"].exists()]
    common.sort(key=lambda t: (not t.get("heldout_id"), t["line_id"], t["take_id"]))
    if args.ab_max:
        common = common[: args.ab_max]
    if not common:
        raise SystemExit(f"--ab: {x} and {y} share no take_id")
    out = args.ab_root / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; choose another --name")
    out.mkdir(parents=True)
    rng = random.Random(f"{args.seed}|{args.name}")
    key, sheet = {}, []
    for n, t in enumerate(common, 1):
        pair = f"pair_{n:02d}"
        order = [x, y] if rng.random() < 0.5 else [y, x]
        src = {x: tx[t["take_id"]]["path"], y: t["path"]}
        for slot, run in zip(("x", "y"), order):
            shutil.copy2(src[run], out / f"{pair}_{slot}.wav")
        original = None
        held = t.get("heldout_id")
        if held and not args.no_original and held in corpus:
            original = f"{pair}_original.wav"
            shutil.copy2(corpus[held]["audio"], out / original)
        key[pair] = {"x": order[0], "y": order[1], "take_id": t["take_id"]}
        sheet.append({"pair": pair, "line_id": t["line_id"], "delivery": t["delivery"], "text": t["text"],
                      "real_recording": original or "", "preference (x / y / same)": "", "notes": ""})
    C.atomic_write_json(out / "key.json", {"runs": [x, y], "pairs": key,
                                           "note": "do not open before filling sheet.csv"})
    with (out / "sheet.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    (out / "README.txt").write_text(
        "Blind A/B: for each pair, listen to pair_NN_x.wav and pair_NN_y.wav (and pair_NN_original.wav, the game's "
        "real recording, where present). Write x, y or same in the preference column of sheet.csv: which take is "
        f"more natural, more emotionally right for the delivery, and more like {C.DISPLAY_NAME}. Then run "
        "select_checkpoint.py --tally <this folder>.\n", encoding="utf-8")
    return out


def tally(folder: Path) -> dict:
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    votes = {run: 0 for run in key["runs"]}
    same = blank = 0
    with (folder / "sheet.csv").open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            answer = (row.get("preference (x / y / same)") or "").strip().casefold()
            mapping = key["pairs"].get(row["pair"]) or {}
            if answer in ("x", "y"):
                votes[mapping[answer]] += 1
            elif answer in ("same", "=", "tie"):
                same += 1
            else:
                blank += 1
    n = sum(votes.values())
    k = max(votes.values()) if n else 0
    # two-sided exact sign test: P(X >= k) * 2 for X ~ Binomial(n, 0.5)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else None
    result = {"votes": votes, "same": same, "unanswered": blank, "sign_test_p_two_sided": p,
              "preferred": max(votes, key=votes.get) if n and votes[key["runs"][0]] != votes[key["runs"][1]] else None}
    C.atomic_write_json(folder / "tally.json", result)
    return result


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    if args.tally:
        print(json.dumps(tally(args.tally), indent=2))
        return
    corpus = C.corpus_index()
    screen = C.load_speaker_screen()
    runs = load_runs(args.takes_root, args.runs)
    if not runs:
        raise SystemExit(f"no run folders with manifest.json under {args.takes_root}")
    notes = attach_scores(runs, args.scorecard) if args.scorecard else ["no --scorecard: range check only"]
    if args.ab:
        out = write_ab(args, runs, corpus)
        print(f"blind A/B pack -> {out} ({len(list(out.glob('pair_*_x.wav')))} pairs)")
        return
    rng = np.random.default_rng(args.seed)
    range_modes = {m.strip() for m in args.range_modes.split(",") if m.strip()}
    feature_cache: dict = {}
    by_mode: dict[str, dict[str, list[dict]]] = {}
    for folder, run in runs.items():
        for take in run["takes"]:
            by_mode.setdefault(take["mode"], {}).setdefault(folder, []).append(take)
    report = {"name": args.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "scorecard": str(args.scorecard)
              if args.scorecard else None, "takes_root": str(args.takes_root), "heldout_weight": args.heldout_weight,
              "range_min": args.range_min, "tolerance": args.tolerance, "notes": notes, "modes": {}}
    for mode, folders in sorted(by_mode.items()):
        entry: dict = {"runs": {}}
        max_lines = max(len({t["line_id"] for t in takes}) for takes in folders.values())
        for folder, takes in folders.items():
            entry["runs"][folder] = {"adapter": runs[folder]["adapter"], "kind": runs[folder]["adapter_kind"],
                                     "step": runs[folder]["step"], **run_summary(takes, args.heldout_weight)}
        if mode in range_modes:
            labels = sorted({t["delivery"] for takes in folders.values() for t in takes})
            real = real_reference_spread(labels, args, corpus, screen, feature_cache)
            entry["real_reference_range"] = {"clips": real["clips"],
                                             "spread": {f: round(v, 3) for f, v in real["spread"].items()
                                                        if not math.isnan(v)}}
            for folder, takes in folders.items():
                entry["runs"][folder]["range"] = range_check(takes, real, feature_cache)
        scored = {f: s for f, s in entry["runs"].items() if s.get("composite_weighted_mean") is not None}
        for s in scored.values():
            rr = (s.get("range") or {}).get("range_ratio")
            s["range_ok"] = rr is None or rr >= args.range_min
        full = {f: s for f, s in scored.items() if s["lines"] >= max_lines}
        # the leader is the best composite among adapters that keep the expressive range: a flattened adapter
        # can top a similarity-first composite, and must not become the yardstick for the others
        eligible = {f: s for f, s in full.items() if s["range_ok"]}
        if full and not eligible:
            report["notes"].append(f"{mode}: every full-coverage run is below range ratio {args.range_min}; "
                                   "the leader is chosen without the range filter")
        pool = eligible or full
        if pool:
            leader = max(pool, key=lambda f: pool[f]["composite_weighted_mean"])
            entry["leader"] = leader
            top = max(full, key=lambda f: full[f]["composite_weighted_mean"])
            if top != leader:
                entry["flattened_top_composite"] = top
            for folder in folders:
                if folder == leader:
                    continue
                entry["runs"][folder]["vs_leader"] = {
                    metric: paired(folders[leader], folders[folder], metric, args.heldout_weight, args.bootstrap, rng)
                    for metric, _ in METRICS}
            lead = entry["runs"][leader]
            shortlist = []
            for folder, s in scored.items():
                comp = (s.get("vs_leader") or {}).get("composite")
                not_worse = folder == leader or (comp is not None and comp["ci95"][1] >= -args.tolerance)
                gate_ok = s.get("gate_pass_rate", 0) >= lead.get("gate_pass_rate", 0) - 0.10
                s["shortlist"] = bool(not_worse and gate_ok and s["range_ok"])
                s["shortlist_reasons"] = {"not_below_leader_beyond_tolerance": not_worse, "gate_ok": gate_ok,
                                          "range_ok": s["range_ok"]}
                if s["shortlist"]:
                    shortlist.append(folder)
            entry["shortlist"] = sorted(shortlist, key=lambda f: -entry["runs"][f]["composite_weighted_mean"])
        report["modes"][mode] = entry
    out = args.out_root / args.name
    C.atomic_write_json(out / "selection.json", report)
    (out / "selection.md").write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    print(f"-> {out / 'selection.json'}")


def fmt(value, digits: int = 3) -> str:
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def render_md(report: dict) -> str:
    lines = [f"# VoxCPM checkpoint selection: {report['name']}", "",
             f"Scorecard: {report['scorecard'] or 'none (range check only)'}. Held-out lines weigh "
             f"{report['heldout_weight']}x; paired differences are run minus leader, 95% CI by bootstrap over lines. "
             f"Range ratio = spread of per-delivery medians (F0 median, F0 IQR, rate, loudness) / the same spread "
             f"in the real delivery references; < {report['range_min']} flags a flattened adapter.", ""]
    for note in report["notes"]:
        lines.append(f"- note: {note}")
    for mode, entry in report["modes"].items():
        lines += ["", f"## mode {mode}", ""]
        if entry.get("real_reference_range"):
            lines.append(f"Real reference spread: {entry['real_reference_range']['spread']}")
            lines.append("")
        lines += ["| run | step | lines | composite | similarity | WER | prosody d | gate | range ratio | "
                  "composite vs leader (95% CI) | shortlist |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        order = sorted(entry["runs"], key=lambda f: -(entry["runs"][f].get("composite_weighted_mean") or -9))
        for folder in order:
            s = entry["runs"][folder]
            comp = (s.get("vs_leader") or {}).get("composite")
            vs = "leader" if folder == entry.get("leader") else (
                f"{comp['mean_diff']:+.3f} [{comp['ci95'][0]:+.3f}, {comp['ci95'][1]:+.3f}]" if comp else "-")
            lines.append(f"| {folder} | {fmt(s.get('step'))} | {s['lines']} | {fmt(s.get('composite_weighted_mean'))} | "
                         f"{fmt(s.get('similarity_weighted_mean'))} | {fmt(s.get('wer_weighted_mean'))} | "
                         f"{fmt(s.get('prosody_distance_weighted_mean'))} | {fmt(s.get('gate_pass_rate'))} | "
                         f"{fmt((s.get('range') or {}).get('range_ratio'))} | {vs} | "
                         f"{'yes' if s.get('shortlist') else ('no' if 'shortlist' in s else '-')} |")
        if entry.get("flattened_top_composite"):
            lines += ["", f"Warning ({mode}): {entry['flattened_top_composite']} has the highest composite but keeps "
                          f"less than {report['range_min']} of the real expressive range (flattened); it is not the "
                          "yardstick and not shortlisted."]
        if entry.get("shortlist"):
            lines += ["", f"Shortlist ({mode}): {', '.join(entry['shortlist'])}. Listen to the top two with --ab."]
        elif entry.get("leader"):
            lines += ["", f"No shortlist for {mode} (see the reasons in selection.json)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
