"""Pair check after the eval: may a Seed-VC take outrank its own source take? (CPU, reads only)

  cd voice-lab/baodur-lab/seedvc
  [VOICE_LAB_CHARACTER=<slug>] ../../seed-vc/.venv/Scripts/python.exe pairs.py --eval-run <name>

Why: Seed-VC V2 copies the prompt's register and pitch range and keeps about half of the source's pitch contour
(convert.py docstring), while eval/aggregate.py ranks mostly on identity (a 1-sd prosody miss costs 0.05 identity).
A conversion can therefore outrank the more expressive TTS take it was made from. For every seedvc_from_X take whose
source take is in the same eval run (<lab>/eval/<name>/takes.json + scorecard.json; <lab> = the VOICE_LAB_CHARACTER
character's lab folder, default BaoDurLab):
  pass = identity(seedvc) - identity(source) >= +0.05        (the scorecard's anchored similarity)
     and WER(seedvc) - WER(source) <= 0.02
     and output/source F0 range ratio in [0.75, 1.33] and |median shift| <= 1 st
         (params.prosody_pair from convert.py --f0-report on; computed here from the two WAVs when missing)
Pass: the conversion may replace its source in the ranking. Fail: present both side by side to the listener (the
.md lists them per line). Writes <lab>/seedvc/pairs/<name>.json and .md; never changes the eval's files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.hide_cuda()  # CPU only

MIN_IDENTITY_GAIN = 0.05
MAX_WER_LOSS = 0.02
RANGE_RATIO = (0.75, 1.33)
MAX_SHIFT_ST = 1.0


def main() -> int:
    p = argparse.ArgumentParser(description="Seed-VC take vs its source take, after eval/run_eval.py")
    p.add_argument("--eval-run", required=True)
    p.add_argument("--out-dir", type=Path, default=C.DATA / "pairs")
    p.add_argument("--eval-root", type=Path, default=C.EVAL_ROOT)
    args = p.parse_args()
    run_dir = args.eval_root / args.eval_run
    run = C.read_json(run_dir / "takes.json")
    card = C.read_json(run_dir / "scorecard.json")
    scores = {row["key"]: row for row in card.get("takes", [])}
    by_source = {(C.norm_path(t["manifest"]), str(t["take_id"])): t for t in run["takes"]}

    pairs, missing = [], 0
    for take in run["takes"]:
        if not str(take.get("engine", "")).startswith("seedvc_from_"):
            continue
        params = take.get("params") or {}
        source = by_source.get((C.norm_path(params.get("source_manifest", "")), str(params.get("source_take_id"))))
        if source is None:
            missing += 1
            continue
        mine, theirs = scores.get(take["key"], {}), scores.get(source["key"], {})
        d_identity = None if mine.get("similarity") is None or theirs.get("similarity") is None \
            else round(mine["similarity"] - theirs["similarity"], 4)
        d_wer = None if mine.get("wer") is None or theirs.get("wer") is None else round(mine["wer"] - theirs["wer"], 4)
        prosody = params.get("prosody_pair")
        if not prosody:
            src = C.load_mono(source["wav"], C.MODEL_SR)[0]
            out = C.load_mono(take["wav"], C.MODEL_SR)[0]
            prosody = C.f0_pair(C.f0_track(src), C.f0_track(out),
                                aligned=not params.get("convert_style") and abs(len(out) / max(1, len(src)) - 1) < 0.02)
        ratio, shift = prosody.get("range_ratio"), prosody.get("median_shift_st")
        checks = {
            "identity_gain": None if d_identity is None else d_identity >= MIN_IDENTITY_GAIN,
            "wer_loss": None if d_wer is None else d_wer <= MAX_WER_LOSS,
            "pitch_range": None if ratio is None else RANGE_RATIO[0] <= ratio <= RANGE_RATIO[1],
            "register_shift": None if shift is None else abs(shift) <= MAX_SHIFT_ST,
        }
        verdict = "may outrank its source" if all(v is True for v in checks.values()) else "listen side by side"
        pairs.append({"line_id": take["line_id"], "seedvc_key": take["key"], "source_key": source["key"],
                      "seedvc_wav": take["wav"], "source_wav": source["wav"], "delivery": take.get("delivery"),
                      "identity": mine.get("similarity"), "source_identity": theirs.get("similarity"),
                      "identity_delta": d_identity, "wer": mine.get("wer"), "source_wer": theirs.get("wer"),
                      "wer_delta": d_wer, "range_ratio": ratio, "median_shift_st": shift,
                      "f0_corr": prosody.get("f0_corr"), "checks": checks, "verdict": verdict,
                      "seedvc_rank_in_line": mine.get("rank_in_line"), "source_rank_in_line": theirs.get("rank_in_line")})
    out = {"eval_run": args.eval_run, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rule": {"min_identity_gain": MIN_IDENTITY_GAIN, "max_wer_loss": MAX_WER_LOSS, "range_ratio": RANGE_RATIO,
                    "max_abs_median_shift_st": MAX_SHIFT_ST},
           "pairs": pairs, "seedvc_takes_without_source_in_run": missing,
           "counts": {v: sum(1 for x in pairs if x["verdict"] == v) for v in ("may outrank its source",
                                                                              "listen side by side")}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(args.out_dir / f"{args.eval_run}.json", out)

    def f(v, signed=False):
        return "-" if v is None else (f"{v:+.3f}" if signed else f"{v:.3f}")
    lines = [f"# Seed-VC vs source takes: {args.eval_run}", "",
             f"Rule: identity gain >= +{MIN_IDENTITY_GAIN}, WER loss <= {MAX_WER_LOSS}, F0 range ratio "
             f"{RANGE_RATIO[0]}-{RANGE_RATIO[1]}, |median shift| <= {MAX_SHIFT_ST} st. {out['counts']}; "
             f"{missing} Seed-VC takes had no source take in this eval run.", "",
             "| line | delivery | verdict | identity d | WER d | range ratio | shift st | f0 corr | seedvc | source |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(pairs, key=lambda x: (x["line_id"], x["verdict"] != "listen side by side", x["seedvc_key"])):
        lines.append(f"| {x['line_id']} | {x['delivery']} | {x['verdict']} | {f(x['identity_delta'], True)} | "
                     f"{f(x['wer_delta'], True)} | {f(x['range_ratio'])} | {f(x['median_shift_st'], True)} | "
                     f"{f(x['f0_corr'])} | {Path(x['seedvc_wav']).name} | {Path(x['source_wav']).name} |")
    (args.out_dir / f"{args.eval_run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(pairs)} pairs {out['counts']}; wrote {args.out_dir / (args.eval_run + '.json')} and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
