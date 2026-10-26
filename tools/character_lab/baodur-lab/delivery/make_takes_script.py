"""Write <lab>/delivery/takes_script.json and takes_script_pairs.json (shared format; VOICE_LAB_CHARACTER, default
baodur).

takes_script.json: the character's new in-character lines (written for the lab, not game lines), one for each
required delivery, plus every held-out test line with its exact corpus text. Both come from
<lab>/delivery/content/lines.json: "new_lines" [id, delivery, text] and "heldout_delivery" {test clip: [delivery,
why]}. Held-out deliveries are chosen the way a director would from the script alone (the line's words and its
TLK performance note), never from the held-out audio. For Bao-Dur: 12 new lines (natural, calm, warm, wry,
troubled, weary, guarded, urgent, angry, questioning, sad, resolute) + his 10 held-out lines.

takes_script_pairs.json (optional render): every non-natural new line again, delivery "natural", ids
"<line id>__natural". Rendering it in the same engine/settings gives score_delivery.py a paired neutral read of
the same words (pass that run with --baseline-manifest), and gives the listener the same A/B: does the troubled
take differ from his natural read of the same words? Without it, direction falls back to a length-matched
baseline from his own clips.

New lines follow his measured habits (lines.json "_about" records them; validate.py checks the word and length
limits, the absence of "!" and the distance from every corpus line).

Run (CPU): python make_takes_script.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labcfg  # noqa: E402

LAB = labcfg.LAB
OUT_DIR = labcfg.DELIVERY
PAIR_SUFFIX = "__natural"


def new_lines() -> list[tuple[str, str, str]]:
    """(id, delivery, text) per new line, from content/lines.json."""
    return [(str(line_id), str(delivery), str(text)) for line_id, delivery, text in labcfg.content("lines")["new_lines"]]


def heldout_delivery() -> dict[str, tuple[str, str]]:
    """Script-only delivery reading of each held-out line (text + TLK note), never its audio: {clip: (delivery, why)}."""
    return {clip: (str(entry[0]), str(entry[1])) for clip, entry in labcfg.content("lines")["heldout_delivery"].items()
            if not clip.startswith("_")}


def write_json(path: Path, document) -> None:
    temporary = path.with_name(path.name + ".tmp")  # atomic: the engine tasks read these files
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Write {labcfg.NAME}'s takes scripts (CPU).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    NEW_LINES, HELDOUT_DELIVERY = new_lines(), heldout_delivery()
    corpus = {json.loads(l)["id"]: json.loads(l) for l in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    test_ids = json.loads((LAB / "corpus_summary.json").read_text(encoding="utf-8"))["test_ids"]
    if sorted(test_ids) != sorted(HELDOUT_DELIVERY):
        raise SystemExit(f"held-out ids changed: {test_ids}")
    lines = [{"id": line_id, "text": text, "delivery": delivery, "source": "new"} for line_id, delivery, text in NEW_LINES]
    for clip_id in test_ids:
        row = corpus[clip_id]
        if row["split"] != "test":
            raise SystemExit(f"{clip_id} is not a test clip")
        lines.append({"id": f"heldout_{clip_id}", "text": row["text"], "delivery": HELDOUT_DELIVERY[clip_id][0],
                      "source": "heldout", "heldout_id": clip_id})
    pairs = [{"id": f"{line_id}{PAIR_SUFFIX}", "text": text, "delivery": "natural", "source": "new"}
             for line_id, delivery, text in NEW_LINES if delivery != "natural"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "takes_script.json", {"lines": lines})
    write_json(args.out_dir / "takes_script_pairs.json", {"lines": pairs})
    print(f"wrote {args.out_dir / 'takes_script.json'} ({len(lines)} lines) and takes_script_pairs.json ({len(pairs)} lines)")


if __name__ == "__main__":
    main()
