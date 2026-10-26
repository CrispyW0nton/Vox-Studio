"""Build the make_ab_pack.py spec for Bao-Dur's saber-story blind A/B from a fine-sweep scorecard (CPU, stdlib).

  python build_ab_spec.py --scorecard <lab>/eval/sbc_fine/scorecard.json --report <lab>/voxcpm/selection/sbc_fine/sweep_report.json
                          --out <lab>/listening/ab_saberstory_v001_spec.json

Systems: #1 and #2 of the sweep report's adapter ranking (LoRA v2 checkpoints only) and the July adapter
(sbc-baseline-july-lora, the adapter Vox Studio has installed). Per line and system the candidate take is the best
by the same per-take score the sweep report ranks with: 0.8 x eval selection score (composite, -1 for gate
failures) + 0.2 x delivery (pauses at sentence breaks, falling ends, unrushed rate). Pairs (20):
  7 saber-story lines  #1 vs #2         (which checkpoint)
  7 saber-story lines  #1 vs July       (new adapter vs the installed one)
  6 held-out lines     #1 vs #2, with his original game recording as the reference
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_report import delivery_parts  # noqa: E402

LAB = Path(r"C:\Users\NewAdmin\Documents\KotorMods\Voices\BaoDurLab")
JULY = "sbc-baseline-july-lora"
HELDOUT_LINES = ["heldout_gblbaodur003", "heldout_gblbaodur131", "heldout_gblbaodur150", "heldout_gblbaodur181",
                 "heldout_gblbaodur200", "heldout_gblbaodur223"]


def take_score(t: dict) -> float:
    parts = delivery_parts(t)
    deliv = st.fmean(parts.values()) if parts else 0.5
    sel = t.get("selection_score")
    if sel is None:
        sel = (t.get("composite") or 0.0) - (0.0 if t.get("gate_pass") else 1.0)
    return 0.8 * sel + 0.2 * deliv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scorecard", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261007)
    args = p.parse_args()
    card = json.loads(args.scorecard.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    ranked = [r["adapter"] for r in report["adapter_ranking"] if "lora_v2" in r["adapter"]]
    top1, top2 = ranked[0], ranked[1]
    script = json.loads((LAB / "delivery" / "takes_script_saberstory_v001.json").read_text(encoding="utf-8"))
    lines = {ln["id"]: ln for ln in script["lines"]}
    corpus = {}
    for row in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if row.strip():
            r = json.loads(row)
            corpus[r["id"]] = r
    best: dict[tuple[str, str], dict] = {}
    for t in card["takes"]:
        if t.get("unscored") or t.get("composite") is None or t.get("tier") == "leak":
            continue
        adapter = t["run_label"].split("/", 1)[-1].rsplit("--", 1)[0]
        key = (adapter, t["line_id"])
        s = take_score(t)
        if key not in best or s > best[key]["_score"]:
            run_dir = LAB / "takes" / t["run_label"]
            wav = Path(t["wav"]) if Path(t["wav"]).is_absolute() else run_dir / t["wav"]
            best[key] = {"_score": s, "take": t["take_id"], "wav": str(wav), "tier": t.get("tier"),
                         "composite": t.get("composite"), "wer": t.get("wer"), "similarity": t.get("similarity")}

    def side(adapter: str, line_id: str) -> dict:
        b = best[(adapter, line_id)]
        return {"label": f"{adapter} | {b['take']}", "wav": b["wav"], "score": round(b["_score"], 4),
                "tier": b["tier"], "composite": b["composite"], "wer": b["wer"], "identity": b["similarity"]}

    pairs = []
    sb_ids = [i for i in lines if i.startswith("sb_")]
    for comparator, tag in ((top2, "checkpoint #1 vs #2"), (JULY, "LoRA v2 #1 vs July adapter")):
        for lid in sb_ids:
            ln = lines[lid]
            pairs.append({"line_id": lid, "text": ln["text"], "delivery": ln["delivery"],
                          "script_id": ln.get("script_id"), "comparison": tag,
                          "x": side(top1, lid), "y": side(comparator, lid)})
    for lid in HELDOUT_LINES:
        ln = lines[lid]
        pairs.append({"line_id": lid, "text": ln["text"], "delivery": ln["delivery"],
                      "script_id": f"held-out {ln['heldout_id']}", "comparison": "checkpoint #1 vs #2 (held-out)",
                      "x": side(top1, lid), "y": side(top2, lid), "original": corpus[ln["heldout_id"]]["audio"]})
    spec = {"title": "Bao-Dur saber story blind A/B v001", "seed": args.seed, "systems": {"#1": top1, "#2": top2,
            "july": JULY}, "ranking": report["adapter_ranking"], "pairs": pairs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"{len(pairs)} pairs; #1 {top1}, #2 {top2}, comparator July {JULY} -> {args.out}")


if __name__ == "__main__":
    main()
