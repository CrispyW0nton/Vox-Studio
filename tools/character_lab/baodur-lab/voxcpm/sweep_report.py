"""Rank VoxCPM2 adapter x mode runs from an evaluation scorecard, with an explicit delivery score (CPU, stdlib).

  python sweep_report.py --scorecard <lab>/eval/<run>/scorecard.json --runs "^voxcpm/sb1-" --name sb1_coarse

Per run folder (adapter x conditioning mode) from the eval scorecard (baodur-lab/eval/run_eval.py):
  identity    mean eval identity (1 = as close to his train centroid as his own clips, 3 speaker models)
  wer         mean ASR WER (faster-whisper large-v3, local) and gate pass rate (eval tier 0)
  composite   eval composite (identity - prosody distance to the real line - implausibility - DNSMOS deficit
              - soft flags); held-out lines weigh 2 (they have a real recording to compare against)
  delivery    the director's delivery brief for Bao-Dur (performance bible; SCRIPT_v002 section 9), per take:
                pauses   internal sentence breaks in the text (". ", "? ", "! ", "... ") that got a pause
                         (eval prosody n_pauses), min(n, expected) / expected; lines without a break skip it
                falling  line-end F0 shift f0_end_st: 1 at <= -0.5 st, 0 at >= +0.5 st (statements only)
                unrushed speaking rate rate_syl_s: 1 at <= his p75 (5.43 syl/s), 0 at >= 6.5 syl/s
              delivery = mean of the available parts
  overall     0.8 x composite + 0.2 x delivery (the eval composite has no delivery term by design)
Lines are split into held-out (real recording exists) and saber-story lines (SB-xx, no recording).
A run missing lines (the July adapter skips the 5 held-out lines it trained on) is also scored on the common lines.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

LAB = Path(r"C:\Users\NewAdmin\Documents\KotorMods\Voices\BaoDurLab")
RATE_P75, RATE_ZERO = 5.43, 6.5
BREAK_RX = re.compile(r"[.?!]+[\"']?\s+(?=\S)")


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def delivery_parts(take: dict) -> dict:
    pro = take.get("prosody") or {}
    text = take.get("text") or ""
    parts = {}
    expected = len(BREAK_RX.findall(text.strip()))
    if expected and pro.get("n_pauses") is not None:
        parts["pauses"] = min(int(pro["n_pauses"]), expected) / expected
    end = pro.get("f0_end_st")
    if end is not None and not text.strip().endswith("?"):
        parts["falling"] = clip01((0.5 - float(end)) / 1.0)
    rate = pro.get("rate_syl_s")
    if rate is not None:
        parts["unrushed"] = clip01((RATE_ZERO - float(rate)) / (RATE_ZERO - RATE_P75))
    return parts


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.fmean(xs), 4) if xs else None


def summarise(takes: list[dict], common: set[str] | None = None) -> dict:
    if common is not None:
        takes = [t for t in takes if t["line_id"] in common]
    by_line = defaultdict(list)
    for t in takes:
        by_line[t["line_id"]].append(t)
    wsum = wcomp = wdel = 0.0
    for line, ts in by_line.items():
        w = 2.0 if ts[0].get("heldout_id") else 1.0
        wsum += w
        wcomp += w * st.fmean(t["composite"] for t in ts)
        wdel += w * st.fmean(t["_delivery"] for t in ts)
    comp = wcomp / wsum if wsum else None
    deliv = wdel / wsum if wsum else None
    held = [t for t in takes if t.get("heldout_id")]
    sb = [t for t in takes if not t.get("heldout_id")]
    out = {
        "takes": len(takes), "lines": len(by_line),
        "identity": mean(t.get("similarity") for t in takes),
        "identity_heldout": mean(t.get("similarity") for t in held),
        "identity_sb": mean(t.get("similarity") for t in sb),
        "wer": mean(t.get("wer") for t in takes),
        "pass_rate": round(sum(1 for t in takes if t.get("gate_pass")) / len(takes), 3) if takes else None,
        "prosody_distance_heldout": mean(t.get("prosody_distance") for t in held),
        "composite_w": round(comp, 4) if comp is not None else None,
        "delivery_w": round(deliv, 4) if deliv is not None else None,
        "pauses": mean(t["_parts"].get("pauses") for t in takes),
        "falling": mean(t["_parts"].get("falling") for t in takes),
        "unrushed": mean(t["_parts"].get("unrushed") for t in takes),
        "rate_syl_s": mean((t.get("prosody") or {}).get("rate_syl_s") for t in takes),
        "f0_end_st": mean((t.get("prosody") or {}).get("f0_end_st") for t in takes),
        "duration_s": mean((t.get("prosody") or {}).get("duration_s") for t in takes),
    }
    out["overall"] = round(0.8 * comp + 0.2 * deliv, 4) if comp is not None else None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scorecard", type=Path, required=True)
    p.add_argument("--runs", default="", help="regex on run_label")
    p.add_argument("--name", required=True)
    p.add_argument("--out-root", type=Path, default=LAB / "voxcpm" / "selection")
    args = p.parse_args()
    card = json.loads(args.scorecard.read_text(encoding="utf-8"))
    rx = re.compile(args.runs) if args.runs else None
    runs = defaultdict(list)
    skipped = 0
    for t in card["takes"]:
        if rx and not rx.search(t["run_label"]):
            continue
        if t.get("unscored") or t.get("composite") is None:
            skipped += 1
            continue
        t["_parts"] = delivery_parts(t)
        t["_delivery"] = st.fmean(t["_parts"].values()) if t["_parts"] else 0.5
        runs[t["run_label"]].append(t)
    line_sets = {r: {t["line_id"] for t in ts} for r, ts in runs.items()}
    common = set.intersection(*line_sets.values()) if line_sets else set()
    rows = []
    for run, ts in runs.items():
        mode = run.rsplit("--", 1)[-1]
        adapter = run.split("/", 1)[-1].rsplit("--", 1)[0]
        m = re.search(r"step_(\d+)", run)
        rows.append({"run": run, "adapter": adapter, "mode": mode, "step": int(m.group(1)) if m else None,
                     **summarise(ts), "common_lines": summarise(ts, common)})
    rows.sort(key=lambda r: -(r["overall"] or -9))
    per_mode = defaultdict(list)
    for r in rows:
        per_mode[r["mode"]].append(r)
    # checkpoint ranking pooled over modes (mean of the per-mode overall scores)
    pooled = defaultdict(list)
    for r in rows:
        pooled[r["adapter"]].append(r["overall"])
    pooled_rank = sorted(({"adapter": a, "overall_mean_over_modes": round(st.fmean(v), 4), "modes": len(v)}
                          for a, v in pooled.items()), key=lambda x: -x["overall_mean_over_modes"])
    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"scorecard": str(args.scorecard), "runs_regex": args.runs, "skipped_unscored": skipped,
           "common_lines": sorted(common), "method": __doc__, "rows": rows, "adapter_ranking": pooled_rank}
    (out_dir / "sweep_report.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    cols = ("overall", "composite_w", "delivery_w", "identity", "identity_heldout", "identity_sb", "wer",
            "pass_rate", "prosody_distance_heldout", "pauses", "falling", "unrushed", "rate_syl_s", "lines")
    md = [f"# VoxCPM2 sweep report: {args.name}", "", f"Scorecard `{args.scorecard}`; {skipped} unscored takes "
          "left out.", "", "overall = 0.8 x eval composite (held-out x2) + 0.2 x delivery (pauses at sentence "
          "breaks, falling line ends, speaking rate <= his p75).", ""]
    for mode, rs in per_mode.items():
        md += [f"## mode {mode}", "", "| rank | adapter | " + " | ".join(cols) + " |",
               "|---|---|" + "---|" * len(cols)]
        for i, r in enumerate(rs, 1):
            md.append(f"| {i} | {r['adapter']} | " + " | ".join(str(r[c]) for c in cols) + " |")
        md.append("")
    md += ["## adapters pooled over modes", "", "| rank | adapter | overall (mean over modes) |", "|---|---|---|"]
    md += [f"| {i} | {x['adapter']} | {x['overall_mean_over_modes']} |" for i, x in enumerate(pooled_rank, 1)]
    (out_dir / "sweep_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
