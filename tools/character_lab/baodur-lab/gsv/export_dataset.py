"""Export one character's clips from corpus.jsonl to a GPT-SoVITS training list (wav_path|<speaker>|en|text).

Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters; <lab> is its lab folder and
<speaker> its config "speaker" (e.g. baodur, jolee).
Selection (recipe defaults): split train only, kind speech, tts_usable, 1-20 s, then:
  * never the held-out test split (checked by split, id, file name and sha256 of the original and restored copies);
  * never a clip another character speaks (e.g. Bao-Dur's gbl files also hold HK-47, G0-T0 and Atton lines: 37 of
    his tts_usable train clips, ~4 min). The list comes from the delivery task's speaker screen
    (<lab>/delivery/exclude_other_speakers.json, sha256 recorded in the manifest); its 'uncertain' outliers are
    left out too until someone has listened (--include-uncertain to keep them), 'suspect' clips are kept
    (--exclude-suspect to drop them);
  * never a train/val line that gives away a held-out test line (same line in another take, or a variant sharing
    a sentence/most words), so the held-out evaluation stays honest (--include-test-near-duplicates to keep them);
  * byte-identical duplicate clips once (by the original audio's sha256).
The val split is held out by default, like the Seed-VC and VoxCPM recipes: its clips are the ground truth
infer.py --val-lines renders for checkpoint selection, so the held-out TEST lines are never used to choose a
model. After the choice is frozen, --splits train,val exports the full set (<speaker>_<audio>_trval.list) for an
optional production retrain with the frozen settings.
Audio: original game audio only (<lab>/DECISIONS.md, user decision 2026-09-30); --audio restored is refused
unless --allow-restored.

Text policy:
  text  (default) the corpus 'text' field: the exact TLK/DLG line when one exists, else the Whisper transcript.
        Where script and ASR differ, almost all differences are ASR errors on names or near-homophones (for
        Bao-Dur: "Zerka", "golem", "raised", "talented"), so the script line is the better label. The few clips
        where the actor really said something else are corrected by <lab>/gsv/text_overrides.json (evidence:
        text_check.py -> <lab>/gsv/text_check.json); the override wins over every policy and is recorded in the
        manifest.
  asr   the Whisper large-v3 transcript for every row.
  auto  the script line unless its WER against ASR exceeds --auto-wer (then ASR).
Pronunciation respellings from <lab>/gsv/lexicon.json are applied to the list text (the original text stays in
the manifest); the matching ARPAbet entries are injected at step 1 by prepare.py.

Usage (any Python 3.10+; stdlib only):
  python export_dataset.py                          # train split, original audio -> <speaker>_original.list
  python export_dataset.py --splits train,val       # production retrain set -> <speaker>_original_trval.list
  $env:VOICE_LAB_CHARACTER = "jolee"; python export_dataset.py   # (PowerShell) another character, own lab
Writes <lab>/gsv/lists/<speaker>_<audio>[_trval].list and the matching .manifest.json (owned by this script).
The list is written with LF line endings, so list_sha256 in the manifest is the hash of the file on disk.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402


def choose_text(row: dict, policy: str, auto_wer: float) -> tuple[str, str]:
    text, asr = (row.get("text") or "").strip(), (row.get("asr") or "").strip()
    if row.get("text_override"):
        return text, "override"
    if policy == "asr":
        return (asr or text), ("asr" if asr else row.get("text_source", "text"))
    if policy == "auto" and row.get("text_source") != "asr" and row.get("wer") is not None \
            and float(row["wer"]) > auto_wer and asr:
        return asr, "asr(auto)"
    return text, row.get("text_source", "text")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", choices=C.AUDIO_MODES, default="original",
                    help=f"original = decoded game clip (corpus 'audio'); restored = {C.RESTORED_DIR}\\<id>.wav")
    ap.add_argument("--splits", default=",".join(C.DEFAULT_SPLITS),
                    help="train (default; val stays held out for checkpoint selection) or train,val (never test)")
    ap.add_argument("--allow-restored", action="store_true",
                    help="permit --audio restored (DECISIONS.md: the user chose original audio for training)")
    ap.add_argument("--text-policy", choices=("text", "asr", "auto"), default="text")
    ap.add_argument("--auto-wer", type=float, default=0.2, help="--text-policy auto: WER above which ASR is used")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--include-not-usable", action="store_true",
                    help="ignore the corpus tts_usable flag (still speech-only, never test)")
    ap.add_argument("--exclusions", default=str(C.EXCLUSIONS), help="speaker-screen exclusions JSON")
    ap.add_argument("--no-exclusions", action="store_true",
                    help="DANGER: skip the speaker screen (other characters' lines would train the model)")
    ap.add_argument("--include-uncertain", action="store_true", help="keep the screen's 'uncertain' clips")
    ap.add_argument("--exclude-suspect", action="store_true", help="also drop the screen's 'suspect' clips")
    ap.add_argument("--include-test-near-duplicates", action="store_true",
                    help="keep train/val lines that repeat or overlap a held-out test line")
    ap.add_argument("--keep-duplicate-audio", action="store_true", help="keep byte-identical duplicate clips")
    ap.add_argument("--lexicon", default=str(C.LEXICON))
    ap.add_argument("--no-lexicon", action="store_true", help="export raw text without respellings")
    ap.add_argument("--text-overrides", default=str(C.TEXT_OVERRIDES),
                    help="transcript corrections JSON (missing file = none)")
    ap.add_argument("--no-text-overrides", action="store_true")
    ap.add_argument("--out", default=None,
                    help=f"list path (default {C.LISTS_DIR}\\{C.SPEAKER}_<audio>[_trval].list)")
    args = ap.parse_args()
    for key in ("exclusions", "lexicon", "text_overrides", "out"):
        setattr(args, key, C.abspath(getattr(args, key)))

    splits = set(C.parse_splits(args.splits))
    C.require_original_audio(args.audio, args.allow_restored, "exporting a training list")
    corpus_rows = C.load_corpus()
    guard = C.HeldOut(corpus_rows)
    overrides = C.load_text_overrides(None if args.no_text_overrides else args.text_overrides, corpus_rows)
    rows = C.apply_text_overrides(corpus_rows, overrides)
    if args.no_exclusions:
        C.log("WARNING: --no-exclusions: the speaker screen is OFF; other characters' lines may be exported")
    excl = C.Exclusions(corpus_rows, None if args.no_exclusions else args.exclusions,
                        include_uncertain=args.include_uncertain,
                        include_near_duplicates=args.include_test_near_duplicates,
                        exclude_suspect=args.exclude_suspect)
    lexicon = C.load_lexicon(None if args.no_lexicon else args.lexicon)
    out_list = args.out or C.default_list_path(args.audio, splits)
    out_manifest = out_list.with_suffix(".manifest.json")

    entries, missing, excluded = [], [], Counter()
    excluded_rows: list[dict] = []
    names: dict[str, str] = {}
    seen_sha: dict[str, str] = {}
    for row in rows:
        split = row.get("split")
        if split not in splits:
            excluded[f"split={split}"] += 1
            continue
        if row.get("kind") != "speech":
            excluded["nonverbal"] += 1
            continue
        if not row.get("tts_usable") and not args.include_not_usable:
            excluded["not_tts_usable"] += 1
            continue
        duration = float(row.get("duration") or 0.0)
        if not args.min_dur <= duration <= args.max_dur:
            excluded["duration"] += 1
            excluded_rows.append({"id": row["id"], "reason": f"duration {duration} s outside "
                                                             f"{args.min_dur}-{args.max_dur} s"})
            continue
        why = excl.reason(row["id"])
        if why:
            excluded["speaker_screen_or_test_leak"] += 1
            excluded_rows.append({"id": row["id"], "reason": why, "duration": duration, "text": row.get("text")})
            continue
        sha = str(row.get("sha256") or "").lower()
        if sha and sha in seen_sha and not args.keep_duplicate_audio:
            excluded["duplicate_audio"] += 1
            excluded_rows.append({"id": row["id"], "reason": f"byte-identical audio to {seen_sha[sha]}"})
            continue
        audio = Path(row["audio"]) if args.audio == "original" else C.restored_path(row["id"])
        if not audio.exists():
            missing.append({"id": row["id"], "audio": str(audio)})
            continue
        reason = guard.reason(row["id"], audio)
        if reason:  # cannot happen for a non-test split unless the corpus is inconsistent; stop loudly
            raise SystemExit(f"held-out guard tripped while exporting {row['id']}: {reason}")
        text, text_from = choose_text(row, args.text_policy, args.auto_wer)
        list_text = C.clean_line_text(C.respell(text, lexicon))
        if not re.search(r"[A-Za-z]", list_text):
            excluded["no_text"] += 1
            continue
        key = audio.name.casefold()
        if key in names:  # upstream keys every feature file by basename
            raise SystemExit(f"duplicate basename {audio.name} ({names[key]} and {row['id']})")
        names[key] = row["id"]
        if sha:
            seen_sha[sha] = row["id"]
        entries.append({
            "id": row["id"], "audio": str(audio), "split": split, "duration": duration,
            "text_source": row.get("text_source"), "text_used": text_from, "wer": row.get("wer"),
            "text": text, "list_text": list_text, "respelled": list_text != C.clean_line_text(text),
            "script_text": row.get("script_text"),
            "directions": row.get("directions", []), "speaker_screen_suspect": row["id"] in excl.suspect,
        })

    if missing:
        raise SystemExit(f"{len(missing)} audio files missing, e.g. {missing[:3]} (restore not finished?)")
    if not entries:
        raise SystemExit("nothing to export")

    body = "".join(f"{e['audio']}|{C.SPEAKER}|{C.LANG}|{e['list_text']}\n" for e in entries)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_list.with_name(out_list.name + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")  # LF on Windows too: list_sha256 = hash of the file
    tmp.replace(out_list)
    if C.sha256_file(out_list) != C.sha256_text(body):
        raise SystemExit(f"{out_list} does not hash to the exported text; refusing to write the manifest")

    seconds = sum(e["duration"] for e in entries)
    durations = sorted(e["duration"] for e in entries)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "list": str(out_list), "list_sha256": C.sha256_text(body),
        "audio_mode": args.audio, "splits": sorted(splits), "text_policy": args.text_policy,
        "auto_wer": args.auto_wer if args.text_policy == "auto" else None,
        "corpus": str(C.CORPUS), "corpus_sha256": C.sha256_file(C.CORPUS),
        "lexicon": lexicon["path"], "lexicon_sha256": lexicon["sha256"],
        "text_overrides": overrides["path"], "text_overrides_sha256": overrides["sha256"],
        "text_overrides_applied": sorted(e["id"] for e in entries if e["text_used"] == "override"),
        "exclusions": excl.describe(),
        "totals": {
            "clips": len(entries), "minutes": round(seconds / 60, 2),
            "median_s": durations[len(durations) // 2], "max_s": durations[-1],
            "by_split": dict(Counter(e["split"] for e in entries)),
            "by_text_source": dict(Counter(e["text_source"] for e in entries)),
            "text_used": dict(Counter(e["text_used"] for e in entries)),
            "respelled_rows": sum(e["respelled"] for e in entries),
            "script_vs_asr_mismatch_rows": sum(1 for e in entries if e["wer"]),
            "speaker_screen_suspect_rows_kept": sum(e["speaker_screen_suspect"] for e in entries),
        },
        "excluded": dict(excluded),
        "excluded_rows": excluded_rows,
        "heldout_test_ids": sorted(guard.ids),
        "rows": entries,
    }
    C.write_json(out_manifest, manifest)
    C.log(f"{len(entries)} clips / {seconds / 60:.2f} min -> {out_list}")
    C.log(f"splits {manifest['totals']['by_split']}, text {manifest['totals']['text_used']}, "
          f"respelled {manifest['totals']['respelled_rows']}, excluded {dict(excluded)}")
    screened = [r for r in excluded_rows if "duration" in r and "text" in r]
    C.log(f"speaker screen / test-leak guard removed {len(screened)} clips "
          f"({sum(r['duration'] for r in screened) / 60:.2f} min); suspect clips kept: "
          f"{manifest['totals']['speaker_screen_suspect_rows_kept']}")
    C.log(f"manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
