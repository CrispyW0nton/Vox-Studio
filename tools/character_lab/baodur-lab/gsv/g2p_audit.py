"""Audit GPT-SoVITS' English g2p on every corpus transcript of one character (CPU only, no models, no GPU).

Lists each word missing from the CMU dictionary with the phones GPT-SoVITS would use before and after the
character's lexicon (<lab>/gsv/lexicon.json), the lines whose phones the lexicon changes, and any line that still
yields UNK phones. Covers all splits (held-out transcripts are synthesized as text during evaluation, never used as
audio here), with <lab>/gsv/text_overrides.json applied exactly as export_dataset.py / infer.py apply it.
Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters. For a new character, run it with
--no-lexicon first: the OOV list is what its lexicon needs.

Usage: python g2p_audit.py [--lexicon <file> | --no-lexicon]   -> <lab>/gsv/g2p_audit.json
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", default=str(C.LEXICON))
    ap.add_argument("--no-lexicon", action="store_true", help="audit raw g2p (a new character without a lexicon)")
    ap.add_argument("--text-overrides", default=str(C.TEXT_OVERRIDES))
    ap.add_argument("--out", default=str(C.GSV_DATA / "g2p_audit.json"))
    args = ap.parse_args()
    C.reexec_in_gsv_venv(__file__)
    # relative paths mean the caller's cwd, not the GPT-SoVITS root that bootstrap chdirs to (absolute ones unchanged)
    for key in ("lexicon", "text_overrides", "out"):
        setattr(args, key, os.path.abspath(getattr(args, key)))
    lexicon_path = None if args.no_lexicon else C.require_lexicon(args.lexicon)

    C.bootstrap_gsv_runtime("cpu")
    import text.english as english  # noqa: E402
    from text.cleaner import clean_text  # noqa: E402

    corpus_rows = C.load_corpus()
    overrides = C.load_text_overrides(args.text_overrides, corpus_rows)
    rows = C.apply_text_overrides(corpus_rows, overrides)
    lexicon = C.load_lexicon(lexicon_path)
    cmu_before = set(english._g2p.cmu)
    homographs = set(english._g2p.homograph2features)

    def tokens(text: str) -> list[str]:
        return english.word_tokenize(english.text_normalize(text))

    oov: dict[str, dict] = defaultdict(lambda: {"count": 0, "ids": []})
    for row in rows:
        for tok in tokens(row["text"]):
            word = tok.lower()
            if len(word) > 1 and re.search("[a-z]", word) and word not in cmu_before and word not in homographs:
                entry = oov[tok]
                entry["count"] += 1
                if row["id"] not in entry["ids"] and len(entry["ids"]) < 6:
                    entry["ids"].append(row["id"])
    for tok, entry in oov.items():
        entry["phones_before"] = " ".join(english._g2p.qryword(tok))
    phones_before = {r["id"]: " ".join(clean_text(r["text"], "en", "v2")[0]) for r in rows}

    installed = C.install_g2p_lexicon(lexicon)
    tokens_after = {tok for row in rows for tok in tokens(C.respell(row["text"], lexicon))}
    for tok, entry in oov.items():
        if tok not in tokens_after:  # e.g. 'Gzero' only exists because the normaliser read 'G0-T0'
            entry["respelled"] = "removed by a respell rule (see changed_lines)"
            entry["phones_after"] = ""
            entry["fixed_by_lexicon"] = True
            continue
        entry["respelled"] = None
        entry["phones_after"] = " ".join(english._g2p.qryword(tok))
        entry["fixed_by_lexicon"] = entry["phones_after"] != entry["phones_before"]
    for tok in sorted(tokens_after - set(oov)):  # words a respell rule introduced (e.g. 'Goto')
        word = tok.lower()
        if len(word) > 1 and re.search("[a-z]", word) and word not in cmu_before and word not in homographs:
            oov[tok] = {"count": 0, "ids": [], "phones_before": "", "respelled": "introduced by a respell rule",
                        "phones_after": " ".join(english._g2p.qryword(tok)), "fixed_by_lexicon": True}
    changed, unk = [], []
    for row in rows:
        respelled = C.respell(row["text"], lexicon)
        after = " ".join(clean_text(respelled, "en", "v2")[0])
        if after != phones_before[row["id"]]:
            changed.append({"id": row["id"], "split": row["split"], "text": row["text"],
                            "respelled": respelled if respelled != row["text"] else None,
                            "phones_before": phones_before[row["id"]], "phones_after": after})
        if "UNK" in after.split():
            unk.append({"id": row["id"], "text": row["text"], "phones": after})

    report = {
        "lexicon": lexicon["path"], "lexicon_sha256": lexicon["sha256"], "arpabet_entries": installed,
        "text_overrides": overrides["path"], "text_overrides_sha256": overrides["sha256"],
        "respell_rules": len(lexicon["respell"]), "rows": len(rows),
        "oov_words": len(oov), "oov_fixed_by_lexicon": sum(e["fixed_by_lexicon"] for e in oov.values()),
        "lines_changed_by_lexicon": len(changed), "lines_with_unk": unk,
        "oov": dict(sorted(oov.items(), key=lambda kv: (-kv[1]["count"], kv[0].lower()))),
        "changed_lines": changed,
    }
    C.write_json(args.out, report)
    C.log(f"{len(rows)} lines, {len(oov)} OOV words ({report['oov_fixed_by_lexicon']} changed by the lexicon), "
          f"{len(changed)} lines re-phonemised, {len(unk)} with UNK -> {args.out}")
    for tok, entry in list(report["oov"].items()):
        flag = "*" if entry["fixed_by_lexicon"] else " "
        print(f" {flag} {tok:<16} x{entry['count']:<3} {entry['phones_before']:<40} -> {entry['phones_after']}")


if __name__ == "__main__":
    main()
