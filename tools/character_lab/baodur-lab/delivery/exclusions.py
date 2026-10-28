"""Write <lab>/delivery/exclude_other_speakers.json for the training tasks (CPU; VOICE_LAB_CHARACTER, default baodur).

Attribution needs both kinds of evidence: the clip sits in a speaker_check.py embedding group far from
the character's centroid on BOTH encoders, AND its words belong to another character (for Bao-Dur: HK-47's
"Statement./Analysis./meatbag" register, G0-T0 addressing "Iridonian" about "your remote", Atton asking about the
General "from way back"). The attributions are the character's own (<lab>/delivery/content/screen.json
"attributions"; none until someone attributes a clip). Embedding outliers without such text evidence are listed as
"uncertain" (for Bao-Dur mostly his own sub-second combat barks and effort sounds), which need a listen.

near_duplicate_of_test is generated with the same held-out leak rule the delivery map and validate.py use
(build_delivery_map.test_overlap: character ratio >= 0.6 or >= 5 words in a row shared with a held-out line),
over every train/val clip, so training sets and reference choices exclude exactly the same clips. Each reason
names the held-out clip id (e.g. gblbaodurNNN), which the voxcpm render step parses. Hand-written reasons come from
screen.json "near_duplicate_descriptions" (each must still be caught by the rule).
Run: python exclusions.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_delivery_map import test_overlap  # noqa: E402
import labcfg  # noqa: E402

LAB = labcfg.LAB


def attributions() -> dict[str, list[str]]:
    """{other character: [clip ids]} from content/screen.json (order kept: it is the output order)."""
    return {name: list(ids) for name, ids in labcfg.content("screen").get("attributions", {}).items()
            if not name.startswith("_")}


def described() -> dict[str, str]:
    """Hand-written near-duplicate reasons for the clips found first; every other hit gets a generated reason."""
    return {cid: text for cid, text in labcfg.content("screen").get("near_duplicate_descriptions", {}).items()
            if not cid.startswith("_")}


def near_duplicates(corpus: dict[str, dict]) -> dict[str, str]:
    tests = [(cid, row["text"]) for cid, row in sorted(corpus.items()) if row["split"] == "test"]
    found = {}
    DESCRIBED = described()
    for cid, row in sorted(corpus.items()):
        if row["split"] == "test" or not (row.get("text") or "").strip():
            continue
        hit = test_overlap(row["text"], tests)
        if hit is None:
            continue
        if cid in DESCRIBED and hit["test_id"] in DESCRIBED[cid]:
            found[cid] = DESCRIBED[cid]
        else:
            found[cid] = (f"shares '{hit['phrase']}' ({hit['run']} words in a row, text ratio {hit['ratio']:.2f}) "
                          f"with held-out {hit['test_id']}")
    missing = sorted(set(DESCRIBED) - set(found))
    if missing:
        raise SystemExit(f"described near-duplicates no longer caught by the rule: {missing}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Write {labcfg.NAME}'s exclude_other_speakers.json (CPU).")
    parser.add_argument("--out", type=Path, default=LAB / "delivery" / "exclude_other_speakers.json")
    args = parser.parse_args()
    CHARACTERS = attributions()
    corpus = {json.loads(l)["id"]: json.loads(l) for l in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    check = json.loads((LAB / "delivery" / "speaker_check.json").read_text(encoding="utf-8"))
    flagged = set(check["summary"]["other_speaker"])
    confirmed = {clip for ids in CHARACTERS.values() for clip in ids}
    missing = sorted(confirmed - flagged)
    if missing:
        raise SystemExit(f"attributed clips not flagged by the embedding screen: {missing}")
    uncertain = sorted(flagged - confirmed)
    usable_train = [c for c in sorted(confirmed) if corpus[c]["tts_usable"] and corpus[c]["split"] == "train"]
    near = near_duplicates(corpus)
    document = {
        "rule": "exclude 'confirmed' and 'near_duplicate_of_test' from every TTS/VC training set; listen to "
                "'uncertain' before using them",
        "confirmed": {name: [{"id": c, "duration": corpus[c]["duration"], "text": corpus[c]["text"]} for c in ids]
                      for name, ids in CHARACTERS.items()},
        "confirmed_ids": sorted(confirmed),
        "confirmed_seconds": round(sum(corpus[c]["duration"] for c in confirmed), 1),
        "confirmed_in_tts_usable_train": usable_train,
        "confirmed_in_tts_usable_train_seconds": round(sum(corpus[c]["duration"] for c in usable_train), 1),
        "tts_usable_train_seconds_before": round(sum(r["duration"] for r in corpus.values() if r["tts_usable"] and r["split"] == "train"), 1),
        "uncertain": [{"id": c, "duration": corpus[c]["duration"], "text_source": corpus[c]["text_source"],
                       "tts_usable": corpus[c]["tts_usable"], "text": corpus[c]["text"],
                       "resemblyzer_z": check["clips"][c]["resemblyzer_z"], "campplus_z": check["clips"][c]["campplus_z"]}
                      for c in uncertain],
        "suspect_short_lines": check["summary"]["suspect"],
        "near_duplicate_rule": "build_delivery_map.test_overlap: text ratio >= 0.6 or >= 5 words in a row shared "
                               "with a held-out test line (the same rule the delivery map uses for references)",
        "near_duplicate_of_test": near,
        "near_duplicate_seconds": round(sum(corpus[c]["duration"] for c in near), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")  # atomic: training tasks read this file
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps({k: document[k] for k in ("confirmed_seconds", "confirmed_in_tts_usable_train_seconds",
                                                "tts_usable_train_seconds_before", "near_duplicate_seconds")}, indent=1),
          "\nuncertain:", [u["id"] for u in document["uncertain"]],
          "\nnear_duplicate_of_test:", json.dumps(near, indent=1))


if __name__ == "__main__":
    main()
