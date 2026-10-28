"""Freeze one character's GPT-SoVITS checkpoint choice after the selection sweep (evaluation + listening).

Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters; <lab> is its lab folder and
<speaker> its config "speaker".
The sweep renders only non-test lines (infer.py --source new,val --val-lines all), so the held-out TEST lines
never choose a model. Once the evaluation and a listen have picked the GPT / SoVITS exports (and whether aux
references help), this writes <lab>/gsv/selection.json; infer.py --selection reads it for the one final render
of the full script, including the held-out lines. The weights' sha256 pins the exact files that were chosen, and
--evidence records why (scorecard path, listening notes). Stdlib only, no models.

  python freeze_selection.py --exp <speaker>_v2ProPlus_original --gpt-epoch 15 --sovits-epoch 8 \\
      --evidence "<lab>/eval/gsv_selection/scorecard.md; user listened to the top 3 per line"
  options: --gpt-tag _dpo (DPO arm), --aux-refs 2 --aux-delivery natural, --preset stable, --replace
A production retrain on train+val with the frozen epochs is frozen the same way (--exp ..._trval, --replace).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402

PRESETS = ("default", "stable", "expressive")  # infer.py PRESETS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help=f"experiment name, e.g. {C.exp_name('v2ProPlus', 'original')}")
    ap.add_argument("--gpt-epoch", type=int, required=True)
    ap.add_argument("--sovits-epoch", type=int, required=True)
    ap.add_argument("--gpt-tag", default="", help="GPT arm, e.g. _dpo")
    ap.add_argument("--aux-refs", type=int, default=0)
    ap.add_argument("--aux-delivery", default=None)
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--evidence", required=True, help="why: scorecard path(s), listening notes")
    ap.add_argument("--exp-root", default=str(C.EXP_ROOT))
    ap.add_argument("--weights-root", default=str(C.WEIGHTS_ROOT))
    ap.add_argument("--out", default=str(C.SELECTION))
    ap.add_argument("--replace", action="store_true", help="archive an existing selection and write a new one")
    args = ap.parse_args()
    exp_root, weights_root, out = C.abspath(args.exp_root), C.abspath(args.weights_root), C.abspath(args.out)
    tag = C.check_tag(args.gpt_tag)
    if not args.evidence.strip():
        raise SystemExit("--evidence must say what the choice is based on")
    if args.aux_refs and not args.aux_delivery:
        raise SystemExit("--aux-refs needs --aux-delivery (e.g. natural)")

    gpt = C.pick_weight(C.list_weights(args.exp, weights_root, tag)["gpt"], args.gpt_epoch, f"GPT{tag}")
    sovits = C.pick_weight(C.list_weights(args.exp, weights_root)["sovits"], args.sovits_epoch, "SoVITS")
    exp_dir = exp_root / args.exp
    trained = C.trained_clip_ids(exp_dir)
    if trained is None:
        raise SystemExit(f"{exp_dir} has no 2-name2text.txt; cannot record what the model was trained on")
    rows = C.load_corpus()
    guard = C.HeldOut(rows)
    leaked = sorted(c for c in trained if guard.reason(c))
    if leaked:
        raise SystemExit(f"{args.exp} was trained on held-out test clips {leaked}; refusing to freeze it")
    split_of = {r["id"]: r["split"] for r in rows}
    by_split: dict[str, int] = {}
    for clip in trained:
        by_split[split_of.get(clip, "?")] = by_split.get(split_of.get(clip, "?"), 0) + 1

    selection = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "exp": args.exp, "exp_dir": str(exp_dir),
        "gpt_tag": tag, "gpt_epoch": args.gpt_epoch, "sovits_epoch": args.sovits_epoch,
        "gpt": gpt, "gpt_sha256": C.sha256_file(gpt), "sovits": sovits, "sovits_sha256": C.sha256_file(sovits),
        "aux_refs": args.aux_refs, "aux_delivery": args.aux_delivery, "preset": args.preset,
        "trained_on_clips_by_split": by_split, "evidence": args.evidence.strip(),
    }
    if out.exists():
        if not args.replace:
            raise SystemExit(f"{out} exists (a choice is already frozen); pass --replace to archive it")
        C.log(f"archived previous selection -> {C.archive(out, 'replaced')}")
    C.write_json(out, selection)
    C.log(f"frozen: {args.exp} GPT{tag} e{args.gpt_epoch} + SoVITS e{args.sovits_epoch}"
          f"{f', aux {args.aux_refs} x {args.aux_delivery}' if args.aux_refs else ''} -> {out}")
    if by_split.get("val"):
        C.log("note: this model trained on the val split, so its val-line takes are not unseen performances")


if __name__ == "__main__":
    main()
