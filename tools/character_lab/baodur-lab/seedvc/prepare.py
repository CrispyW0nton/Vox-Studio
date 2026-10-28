"""Prepare a character's Seed-VC V2 fine-tuning folders from <lab>/corpus.jsonl (CPU only).

  cd voice-lab/baodur-lab/seedvc
  [VOICE_LAB_CHARACTER=<slug>] ../../seed-vc/.venv/Scripts/python.exe prepare.py --audio original [--force]

The character comes from VOICE_LAB_CHARACTER (voice-lab/characters/<slug>.json, default baodur); <lab> is its lab
folder (BaoDurLab, JoleeLab, ...). Writes <lab>/seedvc/data/<audio>/
  train/*.wav   train split only: the fine-tuning set
  val/*.wav     val split: deterministic validation loss in train.py (never trained on)
  prepare_report.json   config, every exclusion with its reason, loudness stats, neutral fallback prompts
All WAVs are 22.05 kHz mono PCM16: configs/v2/vc_wrapper.yaml trains and infers at sr=22050 (80-band mel up to
11.025 kHz, BigVGAN 22k), so resampling once here (soxr_hq) makes the trainer's load a no-op.
The 10 held-out test clips are never read. Re-running with the same settings and the same screen is a no-op; --force
rebuilds a folder that an earlier prepare.py wrote (only the files its report lists are deleted). train.py refuses a
folder whose screen is stale, so rerun this with --force whenever train.py says so. The character's speaker screen
(<lab>/delivery/exclude_other_speakers.json, written by baodur-lab/delivery/exclusions.py) is required.

Data decisions (lab-wide rules; the numbers were measured on Bao-Dur's corpus, 2026-09-30):
- Audio: original game audio only. The user listened to original vs ClearerVoice-restored copies and chose the
  original for training data and prompts (BaoDurLab/DECISIONS.md, applied to every character); --audio restored
  needs --allow-restored.
- Other speakers: a character's dialog files can also hold conversation partners. Bao-Dur's speaker screen
  (delivery/exclude_other_speakers.json) attributes 38 train clips / 241.5 s to HK-47 (14), G0-T0 (15) and Atton (9).
  Confirmed clips are always excluded: a voice-conversion fine-tune is an identity model, and 20 % foreign data would
  teach the CFM to render other timbres. 'uncertain' outliers (Bao-Dur: sub-second combat barks, the effort sounds,
  3 short lines; 11.5 s) are excluded by default (--uncertain include after a listen).
- Held-out leaks: train clips that repeat a held-out TEST line are always excluded. The set is the delivery file's
  list UNION both lab rules recomputed live (common.near_duplicate_screen: the delivery rule in
  build_delivery_map.test_overlap and the eval rule in evalcommon), so a rule change cannot slip a leak in before the
  delivery task rewrites its file: for Bao-Dur 8 clips (218, 273-275, 277, 278, plus 107 and 128, which share
  'should be able to do' with held-out 052 and 'a power cell emitter matrix' with held-out 276).
- Val fairness (--near-val, default exclude): train clips repeating a VAL line (Bao-Dur: 085/265 ~ 080,
  239/243 ~ 237, 258 ~ 261) are excluded, so fine-tuned checkpoints have not heard the self-test lines either.
- Voice conversion never sees transcripts, so every remaining clean train clip is used, including the ones that are
  not tts_usable (ASR-only text or ASR/TLK mismatch): VC only needs the character's audio.
- Nonverbal clips (Bao-Dur: 3 effort sounds, 1.7 s, --nonverbal): excluded. They are sub-second, carry no content for
  the CFM to align to, sit in the speaker screen's 'uncertain' list (z -8..-13), and a source take's effortful
  phonation survives conversion through the 2048-code content tokens anyway, so they cannot help and could blur
  identity.
- Seed-VC's FT_Dataset rejects clips shorter than 1 s (and silently substitutes a random other clip), so the
  remaining sub-second lines would vanish. --short-policy concat joins them with 0.25 s gaps into ~2 s items;
  CFM training's random in-clip prompt split then often pairs one line as prompt with another as target, which is
  exactly the inference situation (prompt and source are different utterances).
- No loudness normalisation: the spread of active levels (Bao-Dur: ~14 dB, -27.7..-13.9 dBFS RMS) is performance
  dynamics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.hide_cuda()  # CPU only


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--audio", choices=["original", "restored"], default="original")
    p.add_argument("--allow-restored", action="store_true",
                   help="only after a new user decision: the user chose original audio on 2026-09-30")
    p.add_argument("--uncertain", choices=["exclude", "include"], default="exclude",
                   help="speaker-screen outliers without attribution (listen before including)")
    p.add_argument("--near-val", choices=["exclude", "include"], default="exclude",
                   help="train clips repeating a val line's words (keeps checkpoint selection fair)")
    p.add_argument("--nonverbal", choices=["exclude", "include"], default="exclude")
    p.add_argument("--short-policy", choices=["concat", "pad", "drop"], default="concat",
                   help="clips shorter than --min-sec: concatenate into ~2 s items (default), pad with silence, or drop")
    p.add_argument("--gap-sec", type=float, default=0.25, help="silence between concatenated short clips")
    p.add_argument("--concat-target-sec", type=float, default=2.0)
    p.add_argument("--min-sec", type=float, default=1.0, help="Seed-VC FT_Dataset minimum clip length")
    p.add_argument("--max-sec", type=float, default=30.0, help="Seed-VC FT_Dataset maximum clip length")
    p.add_argument("--exclude-ids", default="", help="comma-separated corpus ids to leave out as well")
    p.add_argument("--out", type=Path, default=None, help=f"default {C.LAB.name}/seedvc/data/<audio>")
    p.add_argument("--force", action="store_true", help="rebuild an existing folder written by prepare.py")
    return p.parse_args()


def source_note(audio: str) -> str:
    """Where the clips came from, as recorded in prepare_report.json "source"."""
    if audio == "original":
        decoded = C.CHARACTER["source_audio"]
        return f"{C.LAB.name}/corpus.jsonl {decoded.parent.name}/{decoded.name} (sha256-verified)"
    return f"{C.LAB.name}/corpus.jsonl {C.LAB.name}/restored (MossFormer2_SR_48K)"


def clear_previous(out: Path, report_path: Path) -> None:
    """Delete exactly the WAVs an earlier prepare.py run listed; refuse if anything else is in the way."""
    listed = set()
    if report_path.exists():
        old = C.read_json(report_path)
        listed = {item["file"] for item in old.get("items", []) + old.get("val_items", [])}
    present = {f"{p.parent.name}/{p.name}" for sub in ("train", "val") for p in (out / sub).glob("*")}
    foreign = sorted(present - listed)
    others = sorted(p.name for p in out.iterdir() if p.name not in ("train", "val", "prepare_report.json"))
    if foreign or others or not report_path.exists():
        raise SystemExit(f"ERROR: {out} holds files prepare.py did not list ({(foreign + others)[:5]}...); "
                         "refusing to delete. Pass --out to a new folder.")
    for rel in sorted(present):
        (out / rel).unlink()
    report_path.unlink()
    print(f"cleared {len(present)} files from the previous prepare.py output in {out}")


def main() -> int:
    args = parse_args()
    import numpy as np

    C.require_original(args.audio, args.allow_restored, "prepare.py --audio")
    out = (args.out or C.PREPARED / args.audio).resolve()
    rows = C.load_corpus()
    speakers = C.speaker_exclusions(required=True, rows=rows)
    config = {"audio": args.audio, "uncertain": args.uncertain, "near_test": "exclude", "near_val": args.near_val,
              "nonverbal": args.nonverbal, "short_policy": args.short_policy, "gap_sec": args.gap_sec,
              "concat_target_sec": args.concat_target_sec, "min_sec": args.min_sec, "max_sec": args.max_sec,
              "exclude_ids": sorted(filter(None, args.exclude_ids.split(","))), "sample_rate": C.MODEL_SR,
              "speaker_exclusions_sha256": speakers["sha256"],
              "near_test_ids": sorted(speakers["near_test"]),
              "near_val_ids": sorted(speakers["near_val"]) if args.near_val == "exclude" else []}
    report_path = out / "prepare_report.json"
    if out.exists() and any(out.iterdir()):
        if report_path.exists() and C.read_json(report_path).get("config") == config and not args.force:
            print(f"up to date: {report_path}")
            return 0
        if not args.force:
            print(f"ERROR: {out} already holds a different preparation; pass --force to rebuild it "
                  "(or --out to a new folder)", file=sys.stderr)
            return 2
        clear_previous(out, report_path)

    guard = C.HeldOutGuard(rows)
    restore = {}
    if args.audio == "restored":
        if not C.RESTORE_REPORT.exists():
            print("ERROR: restore_report.json missing (restoration not finished)", file=sys.stderr)
            return 2
        restore = {c["id"]: c for c in C.read_json(C.RESTORE_REPORT)["clips"]}

    def exclusion_reason(row) -> str | None:
        rid = row["id"]
        if rid in speakers["confirmed"]:
            return f"other speaker: {speakers['confirmed'][rid]} (delivery speaker screen, confirmed)"
        if rid in speakers["uncertain"] and args.uncertain == "exclude":
            return f"uncertain speaker: {speakers['uncertain'][rid]}"
        if rid in speakers["near_test"]:
            return f"near duplicate of a held-out test line: {speakers['near_test'][rid]}"
        if rid in speakers["near_val"] and args.near_val == "exclude" and row["split"] == "train":
            return f"near duplicate of a val line: {speakers['near_val'][rid]}"
        if row["kind"] == "nonverbal" and args.nonverbal == "exclude":
            return "nonverbal excluded"
        if rid in config["exclude_ids"]:
            return "--exclude-ids"
        return None

    excluded, train_rows, val_rows = [], [], []
    for row in rows:
        if row["split"] not in ("train", "val"):
            continue
        reason = exclusion_reason(row)
        if reason:
            excluded.append({"id": row["id"], "split": row["split"], "seconds": row["duration"], "reason": reason})
        else:
            (train_rows if row["split"] == "train" else val_rows).append(row)
    assert not any(r["id"] in guard.ids for r in train_rows + val_rows)

    def load(row):
        path = C.clip_path(row, args.audio)
        if not path.exists():
            raise FileNotFoundError(path)
        if C.norm_path(path) in guard.paths:
            raise RuntimeError(f"refusing held-out clip {path}")
        if args.audio == "original" and C.sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {path}")
        if args.audio == "restored":
            info = restore.get(row["id"])
            if not info or info["rate"] != C.OUT_SR or abs(info["duration"] - info["source_duration"]) > 0.05:
                raise RuntimeError(f"restored clip {row['id']} missing from restore_report or drifted")
        wave, _ = C.load_mono(path, C.MODEL_SR)
        return path, wave

    train_dir, val_dir = out / "train", out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    sr = C.MODEL_SR
    items, shorts, levels = [], [], {}

    for row in train_rows:
        path, wave = load(row)
        seconds = len(wave) / sr
        levels[row["id"]] = C.active_rms_db(wave, sr)
        if seconds > args.max_sec:
            excluded.append({"id": row["id"], "split": "train", "seconds": row["duration"],
                             "reason": f"longer than {args.max_sec} s"})
        elif seconds < args.min_sec:
            shorts.append((row, wave))
        else:
            name = f"{row['id']}.wav"
            C.write_pcm16(train_dir / name, wave, sr)
            items.append({"file": f"train/{name}", "sources": [row["id"]], "seconds": round(seconds, 3),
                          "kind": row["kind"], "method": "clip", "active_rms_db": round(levels[row["id"]], 2)})

    gap = np.zeros(int(round(args.gap_sec * sr)), dtype=np.float32)
    if args.short_policy == "drop":
        excluded += [{"id": r["id"], "split": "train", "seconds": r["duration"],
                      "reason": f"shorter than {args.min_sec} s (--short-policy drop)"} for r, _ in shorts]
    elif args.short_policy == "pad":
        for row, wave in shorts:
            need = int(round((args.min_sec + 0.05) * sr)) - len(wave)
            padded = np.concatenate([np.zeros(need // 2, np.float32), wave, np.zeros(need - need // 2, np.float32)])
            name = f"pad_{row['id']}.wav"
            C.write_pcm16(train_dir / name, padded, sr)
            items.append({"file": f"train/{name}", "sources": [row["id"]], "seconds": round(len(padded) / sr, 3),
                          "kind": row["kind"], "method": "padded"})
    else:
        groups, current = [], []
        for pair in sorted(shorts, key=lambda rw: rw[0]["id"]):
            current.append(pair)
            if sum(len(w) for _, w in current) + len(gap) * (len(current) - 1) >= args.concat_target_sec * sr:
                groups.append(current)
                current = []
        if current:
            if sum(len(w) for _, w in current) + len(gap) * (len(current) - 1) >= args.min_sec * sr or not groups:
                groups.append(current)
            else:
                groups[-1].extend(current)
        for index, group in enumerate(groups):
            parts = []
            for j, (_, wave) in enumerate(group):
                parts += ([gap] if j else []) + [wave]
            joined = np.concatenate(parts)
            if len(joined) < args.min_sec * sr:  # only possible when every short clip fits in one tiny group
                need = int(round((args.min_sec + 0.05) * sr)) - len(joined)
                joined = np.concatenate([joined, np.zeros(need, np.float32)])
            ids = [r["id"] for r, _ in group]
            name = f"concat_{index:02d}_{'+'.join(ids)}.wav"
            C.write_pcm16(train_dir / name, joined, sr)
            items.append({"file": f"train/{name}", "sources": ids, "seconds": round(len(joined) / sr, 3),
                          "kind": "+".join(sorted({r["kind"] for r, _ in group})), "method": "concat"})

    val_items = []
    for row in val_rows:
        path, wave = load(row)
        levels[row["id"]] = C.active_rms_db(wave, sr)
        name = f"{row['id']}.wav"
        C.write_pcm16(val_dir / name, wave, sr)
        val_items.append({"file": f"val/{name}", "sources": [row["id"]], "seconds": round(len(wave) / sr, 3),
                          "active_rms_db": round(levels[row["id"]], 2)})

    train_speech = np.array([levels[r["id"]] for r in train_rows if r["kind"] == "speech"])
    loudness = {"basis": "active RMS dBFS of train speech clips (20 ms frames within 40 dB of the loudest)",
                "p05": round(float(np.percentile(train_speech, 5)), 2),
                "median": round(float(np.median(train_speech)), 2),
                "p95": round(float(np.percentile(train_speech, 95)), 2)}

    # Neutral fallback prompts for convert.py when no delivery_map.json exists (this character's train/val clips only).
    median = loudness["median"]
    candidates = [r for r in train_rows + val_rows
                  if r["kind"] == "speech" and r["text_source"] in ("tlk-exact", "dlg")
                  and r["wer"] is not None and r["wer"] <= 0.15 and not r["directions"] and not r["delivery_tags"]
                  and 4.0 <= r["duration"] <= 10.0]
    candidates.sort(key=lambda r: (abs(levels[r["id"]] - median) > 2.5, abs(r["duration"] - 6.5),
                                   abs(levels[r["id"]] - median)))
    fallback = [{"id": r["id"], "audio": r["audio"], "restored_audio": str(C.restored_path(r["id"])),
                 "text": r["text"], "duration": r["duration"],
                 "why": (f"neutral read (no TLK direction), exact transcript, {r['duration']:.1f} s, active level "
                         f"{levels[r['id']]:.1f} dBFS vs median {median:.1f}")} for r in candidates[:5]]

    def seconds_of(reason_prefix: str) -> float:
        return round(sum(e["seconds"] for e in excluded if e["reason"].startswith(reason_prefix)), 1)

    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "source": source_note(args.audio),
        "speaker_exclusions": speakers["path"],
        "audio_decision": C.USER_AUDIO_DECISION if args.audio == "original" else "--allow-restored (new user decision)",
        "seedvc_band_limit_hz": C.MODEL_SR // 2,
        "train": {"items": len(items), "seconds": round(sum(i["seconds"] for i in items), 1),
                  "clips_used": sum(len(i["sources"]) for i in items),
                  "short_clips": len(shorts), "short_seconds": round(sum(len(w) for _, w in shorts) / sr, 1)},
        "val": {"items": len(val_items), "seconds": round(sum(i["seconds"] for i in val_items), 1)},
        "excluded_summary": {"clips": len(excluded), "seconds": round(sum(e["seconds"] for e in excluded), 1),
                             "other_speaker_seconds": seconds_of("other speaker"),
                             "uncertain_seconds": seconds_of("uncertain"),
                             "near_test_seconds": seconds_of("near duplicate of a held-out"),
                             "near_val_seconds": seconds_of("near duplicate of a val")},
        "held_out_test_ids_never_read": sorted(guard.ids),
        "excluded": excluded,
        "loudness": loudness,
        "fallback_references": fallback,
        "items": items,
        "val_items": val_items,
    }
    C.write_json(report_path, report)
    print(f"wrote {report_path}")
    print({k: report[k] for k in ("train", "val", "excluded_summary", "loudness")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
