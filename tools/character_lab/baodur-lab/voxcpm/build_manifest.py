"""Build the VoxCPM2 LoRA v2 train/val manifests of a voice-lab character (CPU only; never imports torch).

  %LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe build_manifest.py
      -> <lab>\\voxcpm\\manifests\\original\\ (train.jsonl, val.jsonl, train_lora_v2.yaml, report)
  ... build_manifest.py --with-val
      -> manifests\\original_trainval\\: train = train + val rows, no val manifest (the optional production
         retrain once the checkpoint sweep has fixed the step count; test clips stay held out)
  The character is VOICE_LAB_CHARACTER (unset = baodur -> <lab> = KotorMods\\Voices\\BaoDurLab; jolee ->
  JoleeLab); see common.py. Its <lab>\\voxcpm\\recipe.json holds the character-specific content used here
  (text_rewrites, direction_to_instruction) and <lab>\\voxcpm\\train_lora_v2.template.yaml, if present,
  replaces the default --template.
  Rebuild after ANY edit of this file or of the recipe: train_lora_v2.py refuses manifests whose
  manifest_report.json was written by another revision of build_manifest.py (sha256), another
  MANIFEST_SCHEMA, another character or another recipe.json (sha256). Rebuilds are deterministic.

What it does, and why (fixes for Bao-Dur's July Vox run: 156 clips, no ref_audio rows, 133 clips without text):
- Rows: corpus.jsonl split train -> train.jsonl, split val -> val.jsonl, only `tts_usable` speech
  (Bao-Dur: 280 train / 16 val before the guards below; 230 train / 15.2 min and 16 val / 1.4 min after them,
  2026-09-30 18:50 build with the 18:45 speaker screen). split test is never read, trimmed, cached, used as a
  reference or written anywhere; the script asserts that at the end.
- Audio: ORIGINAL game audio only (user decision 2026-09-30, <lab>\\DECISIONS.md, every character). --audio
  restored exits unless --allow-restored-audio is given; restored copies are never training or reference input.
- Speaker screen (<lab>\\delivery\\exclude_other_speakers.json, required): a character's voice files can also
  contain other characters (Bao-Dur's gblbaodur* files: 'confirmed' 14 HK-47, 15 G0-T0, 9 Atton; 37 of them
  tts_usable train rows = 4.0 min, all with ASR-only text). 'confirmed' clips are never trained on, cached or
  used as ref_audio. 'uncertain' clips (embedding outliers without text evidence; Bao-Dur:
  gblbaodur013/032/331 among the usable rows, plus barks/efforts) are left out until someone listens
  (--include-uncertain keeps them). 'suspect' clips (weak outliers, mostly short or emotional lines) stay in
  training but are not used as ref_audio, so every identity reference is a clearly verified clip of the
  character.
- Held-out leak guard (on by default): train/val clips that (partly) speak a test line are dropped, never
  cached and never referenced: a full sentence of it, the same line (word Jaccard >= 0.9), or a shared
  distinctive phrase (4 words with >= 3 content words, common.overlap_reason), plus everything in the
  screen's near_duplicate_of_test list (its own rule: >= 5 words in a row or text ratio >= 0.6). Bao-Dur:
  gblbaodur218 (same line as test 200), 273/274/275/277/278 (the "No, you're still missing ..." variants
  of test 276), 128 ("We need a power cell, emitter matrix, lens ..." repeats test 276's "a power cell,
  emitter matrix and lens") and, by the screen's rule, 107 ("should be able to do", test 052).
- Byte-identical duplicate audio (Bao-Dur: gblbaodur074/075) is kept once.
- Audio processing: leading/trailing silence cut to at most --lead-pad/--tail-pad (defaults 0.10/0.20 s,
  both <= 0.3 s; upstream asks for < 0.5 s of trailing silence; only cuts, never pads), then resampled with
  soxr VHQ to 16 kHz, the AudioVAE encoder rate the official trainer asserts.
- ref_audio: ~40% of rows (--ref-ratio; upstream recommends 30-50%) get a *different* train clip of the same
  character as `ref_audio`, 5-12.5 s long (--ref-min/--ref-max; the VoxCPM usage guide gives 5-30 s as the
  practical cloning-reference range, and Vox's Bao-Dur identity clip gblbaodur248 is 5.8 s), each ref used at
  most 3 times, never a clip that shares the line or a distinctive phrase with the target. EVERY instruction
  row gets a ref (the [ref][(instruction)text][audio] layout that idref_instr/Vox controlled cloning uses
  at inference); the rest of the quota is drawn per duration quartile. The ref pool is the kept train
  rows minus 'suspect' clips. Pairing (--ref-pairing):
    random  (default) the ref is drawn uniformly from the eligible pool, as the VoxCPM fine-tuning guide
            says ("randomly choose another clean recording from the same speaker"): the isolated
            reference channel then carries timbre, not the target's prosody. That is what Vox production
            needs: it uses ONE fixed identity reference for every line (use_stable_character_identity) and
            steers delivery with the instruction, so an adapter that learned to copy prosody from the
            reference would flatten every line toward that one clip and weaken instruction control.
    nearest the ref is one of the --ref-topk nearest by prosody (F0 level/spread/range, loudness,
            dynamics, pace, pauses, brightness, terminal pitch, same-direction bonus): trains emotion
            transfer from the reference (kept for an A/B only, not recommended).
    mixed   50/50 per row.
  ref_distance (prosody distance target<->ref) is recorded either way for inspection.
- Nonverbal clips (Bao-Dur: UGH!/Ugh!/Uh-oh!, plus the 'Grrrrr!' bark) are excluded from TTS rows and from
  the ref pool: their "transcripts" are onomatopoeia guesses that map inconsistently to sound (the
  LoRA would learn to blurt effort noises for ordinary interjections), they are < 1 s and skew the
  stop predictor toward clipped endings, and Vox already plays the authentic in-game efforts directly
  as vocal actions (the speaker screen also flags all four as embedding outliers). Short speech barks
  (< 1 s) follow the corpus tts_usable rule unless --include-barks is given; the screen's 'uncertain'
  barks (Bao-Dur: 315/316/323/324/325/332/333) additionally need --include-uncertain.
- Transcripts: tlk-exact/dlg rows are verified text. ASR-only rows (Whisper large-v3; the corpus has no
  WER for them because there is nothing to compare against) are NOT verified (Bao-Dur: 11 remain,
  gblbaodur262-266/268-270, the armor-modification lines, 281, 350, 354; they read as coherent lines), and
  --exact-only drops them all. Rows that are ASR-only AND speaker-'suspect' are dropped by default
  (doubly unverified: every confirmed other-character clip was ASR-only; Bao-Dur: gblbaodur353),
  --keep-asr-suspect keeps them.
- Instructions (--instructions directions, default): rows whose TLK performance note describes the
  delivery ("Troubled", "A little wary", ...) get it as a VoxCPM control prefix "(note)text", so the
  LoRA keeps the instruction pathway the render step's instruction modes rely on and learns what the
  notes mean in the character's real performances. Which notes describe delivery, and their control text,
  is the character's recipe.json direction_to_instruction (Bao-Dur: 23 KOTOR II notes; Jolee: none, his
  KOTOR 1 notes are only the gameplay outcomes "Success"/"Failure"); other notes are ignored. The corpus
  delivery_tags are heuristics an older tool derived from those same notes; a tag is appended only when it
  agrees with the note's delivery family (so Bao-Dur's gblbaodur239/243 "Angry at the end" are no longer also
  tagged "Urgent"). --instructions none gives plain text.

Outputs (<lab>\\voxcpm\\):
  cache\\<audio>\\<id>.wav                     trimmed 16 kHz PCM16 clips (screened train + val speech only;
                                             stale files from earlier builds are removed)
  manifests\\<dataset>\\train.jsonl, val.jsonl  official fields audio/text/duration/ref_audio/ref_duration
                                             (+ id/ref_id/instruction bookkeeping columns)
  manifests\\<dataset>\\features.csv            per-clip prosody features (ref inspection, selection)
  manifests\\<dataset>\\manifest_report.json    counts, exclusions and reasons, ref pairs, trim stats, the
                                             builder sha256/schema/character/recipe sha256 and the sha256 of
                                             every manifest written
  manifests\\<dataset>\\train_lora_v2.yaml      the template config with this dataset's paths
  (<dataset> = original, or original_trainval with --with-val)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FEATURES = ("f0_med", "f0_iqr", "f0_range", "loud_db", "dyn_db", "pause", "rate", "centroid", "f0_end")
WEIGHTS = {"f0_med": 1.0, "f0_iqr": 0.8, "f0_range": 0.7, "loud_db": 1.0, "dyn_db": 0.5, "pause": 0.5,
           "rate": 0.8, "centroid": 0.8, "f0_end": 0.4}

# The character's TLK performance notes that describe delivery -> control text, from <lab>\voxcpm\recipe.json
# (keys stripped + casefolded). Anything not listed (staging, gameplay conditions, "Fade in", "Light Side
# face", upgrade notes, persuade "Success"/"Failure"...) is ignored.
DIRECTION_TO_INSTRUCTION = {str(note).strip().casefold(): str(text)
                            for note, text in C.recipe_value("direction_to_instruction").items()}
TAG_TO_INSTRUCTION = {"urgent": "Urgent", "wry": "Wry", "guarded": "Guarded", "warm": "Warm",
                      "questioning": "Questioning", "calm": "Calm", "angry": "Angry", "sad": "Sad",
                      "weary": "Weary", "troubled": "Troubled", "resolute": "Resolute"}
# coarse delivery families for the ref-matching bonus and the tag/direction agreement test
FAMILY = [
    ("troubled", r"troubl|distress|unhappy|heav|disapprov"),
    ("angry", r"angry|anger|intense|fervent|accus"),
    ("wry", r"amuse|wry|disbelief|insinuat"),
    ("guarded", r"wary|suspicious|guarded|pointed"),
    ("warm", r"reassur|warm|conciliator|prompting|agree"),
    ("questioning", r"question|consider"),
    ("urgent", r"urgent|brisk"),
]
NONVERBAL_WORD = re.compile(r"^(?:g+r+|u+g+h+|a+r+g+h+|h+m+|u+h+|a+h+|o+h+|n+g+h*)$", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", choices=C.AUDIO_VARIANTS, default="original", help="source audio variant")
    p.add_argument("--allow-restored-audio", action="store_true",
                   help="override the user's 2026-09-30 decision to train on original audio only")
    p.add_argument("--with-val", action="store_true",
                   help="production dataset: train on train + val rows (no val manifest), named <audio>_trainval")
    p.add_argument("--corpus", type=Path, default=C.CORPUS)
    p.add_argument("--ref-ratio", type=float, default=0.40, help="share of rows that get ref_audio (0.3-0.5)")
    p.add_argument("--ref-min", type=float, default=5.0, help="min ref duration after trim (s)")
    p.add_argument("--ref-max", type=float, default=12.5, help="max ref duration after trim (s)")
    p.add_argument("--ref-pairing", choices=("random", "nearest", "mixed"), default="random",
                   help="random = upstream-style same-speaker pairing (default); nearest = prosody-matched")
    p.add_argument("--ref-topk", type=int, default=5, help="nearest/mixed: pick at random among the k nearest")
    p.add_argument("--ref-max-uses", type=int, default=3)
    p.add_argument("--lead-pad", type=float, default=0.10, help="max leading silence kept (s, <= 0.3)")
    p.add_argument("--tail-pad", type=float, default=0.20, help="max trailing silence kept (s, <= 0.3)")
    p.add_argument("--instructions", choices=("directions", "none"), default="directions")
    p.add_argument("--include-barks", action="store_true", help="also train on < 1 s speech barks")
    p.add_argument("--exact-only", action="store_true", help="drop rows whose text is ASR-only")
    p.add_argument("--keep-asr-suspect", action="store_true",
                   help="keep rows that are ASR-only AND speaker-'suspect' (dropped by default)")
    p.add_argument("--keep-test-overlap", action="store_true",
                   help="do NOT drop clips that (partly) speak a test line (own detector + the screen's list)")
    p.add_argument("--speaker-screen", type=Path, default=C.SPEAKER_SCREEN,
                   help="the delivery task's exclude_other_speakers.json (required)")
    p.add_argument("--include-uncertain", action="store_true",
                   help="keep the screen's 'uncertain' clips (only after listening to them)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--pairs-from", choices=C.AUDIO_VARIANTS, default=None,
                   help="reuse that variant's ref pairs and val order (original-vs-restored A/B only)")
    p.add_argument("--template", type=Path, default=C.TEMPLATE_YAML)
    args = p.parse_args()
    if args.audio == "restored" and not args.allow_restored_audio:
        p.error(f"--audio restored is off: {C.RESTORED_DECISION}. Pass --allow-restored-audio only if the user "
                "reverses that decision.")
    if args.pairs_from == args.audio:
        p.error("--pairs-from must name the other variant")
    if not (0.0 <= args.ref_ratio <= 1.0):
        p.error("--ref-ratio must be within [0, 1]")
    if not 0 < args.ref_min < args.ref_max:
        p.error("--ref-min must be > 0 and < --ref-max")
    if max(args.lead_pad, args.tail_pad) > 0.3 or min(args.lead_pad, args.tail_pad) < 0:
        p.error("--lead-pad/--tail-pad must be within [0, 0.3] s")
    return args


# --------------------------------------------------------------------------------------------- selection
def unusable_reason(row: dict) -> str:
    if row["kind"] != "speech":
        return "nonverbal"
    if row.get("wer") is not None and row["wer"] > 0.35:
        return "transcript-mismatch"
    if row["duration"] < 1.0:
        return "short-bark"
    if row["duration"] > 20.0:
        return "too-long"
    return "not-tts-usable"


def bark_ok(row: dict) -> bool:
    tokens = C.words(C.clean_text(row["text"]))
    return (row["kind"] == "speech" and 0.45 <= row["duration"] < 1.0 and bool(tokens)
            and (row.get("wer") is None or row["wer"] <= 0.35)
            and not any(NONVERBAL_WORD.match(t) for t in tokens)
            and not any(ch.isdigit() for ch in C.clean_text(row["text"])))


def family_of(text: str) -> set[str]:
    blob = text.casefold()
    return {name for name, pattern in FAMILY if re.search(pattern, blob)}


def instruction_for(row: dict) -> str:
    """Control text from the TLK performance notes; a corpus delivery tag is added only when it agrees with
    the note's delivery family (tags are an older tool's heuristic reading of the same notes)."""
    phrases: list[str] = []
    note_families: set[str] = set()
    for direction in row.get("directions") or []:
        text = DIRECTION_TO_INSTRUCTION.get(direction.strip().casefold())
        if text and text not in phrases:
            phrases.append(text)
            note_families |= family_of(direction)
    for tag in row.get("delivery_tags") or []:
        text = TAG_TO_INSTRUCTION.get(str(tag).strip().casefold())
        if not text or text in phrases:
            continue
        if phrases and not (family_of(str(tag)) & note_families):
            continue  # e.g. 'urgent' on "Angry at the end": the actor's note wins
        phrases.append(text)
    return ", ".join(phrases[:2])


def families(row: dict) -> set[str]:
    return family_of(" ".join([*(row.get("directions") or []), *map(str, row.get("delivery_tags") or [])]))


# --------------------------------------------------------------------------------------------- features
def delivery_features(audio: np.ndarray, rate: int, text: str) -> dict:
    import librosa
    from scipy.signal import medfilt

    hop = rate // 100
    f0 = librosa.yin(audio, fmin=65.0, fmax=420.0, sr=rate, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=audio, frame_length=4 * hop, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, rms_db = f0[:n], 20.0 * np.log10(rms[:n] + 1e-10)
    top = float(rms_db.max()) if n else -100.0
    active = rms_db > top - 35.0
    voiced = active & (rms_db > top - 25.0) & (f0 > 66.0) & (f0 < 410.0)
    out = {k: float("nan") for k in FEATURES}
    semis = 12.0 * np.log2(f0[voiced] / 55.0)
    if semis.size >= 5:
        semis = semis - 12.0 * np.round((semis - medfilt(semis, 5)) / 12.0)
        pos = np.flatnonzero(voiced) / max(1, n - 1)
        out["f0_med"] = float(np.median(semis))
        out["f0_iqr"] = float(np.percentile(semis, 75) - np.percentile(semis, 25))
        out["f0_range"] = float(np.percentile(semis, 90) - np.percentile(semis, 10))
        head, tail = semis[pos <= 0.3], semis[pos >= 0.7]
        out["f0_end"] = float(np.median(tail) - np.median(head)) if head.size and tail.size else 0.0
    if active.any():
        out["loud_db"] = float(np.mean(rms_db[active]))
        out["dyn_db"] = float(np.percentile(rms_db[active], 90) - np.percentile(rms_db[active], 10))
        out["pause"] = float(1.0 - active.mean())
        letters = len(re.sub(r"[^A-Za-z]", "", text))
        out["rate"] = letters / max(0.3, active.sum() * hop / rate)
        cent = librosa.feature.spectral_centroid(y=audio, sr=rate, n_fft=512, hop_length=hop)[0][:n]
        out["centroid"] = float(np.mean(cent[active[: len(cent)]]))
    return out


def zscore_table(feats: dict[str, dict]) -> dict[str, np.ndarray]:
    ids = list(feats)
    matrix = np.array([[feats[i][k] for k in FEATURES] for i in ids], dtype=np.float64)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[stds < 1e-9] = 1.0
    z = (matrix - means) / stds
    z = np.where(np.isnan(z), 0.0, z)  # missing feature -> population mean
    w = np.array([WEIGHTS[k] for k in FEATURES])
    return {i: z[j] * np.sqrt(w) for j, i in enumerate(ids)}


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    t0 = time.time()
    rows = C.load_corpus(args.corpus)
    by_id = {r["id"]: r for r in rows}
    tests = [r for r in rows if r["split"] == "test"]
    test_set = {r["id"] for r in tests}
    variant = args.audio
    dataset = C.dataset_name(variant, args.with_val)
    cache = C.cache_dir(variant)
    out_dir = C.manifest_dir(dataset)
    cache.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- speaker screen ---------------------------------------------------------------------
    screen = C.load_speaker_screen(args.speaker_screen)
    other_speaker = screen["confirmed"]
    uncertain = set() if args.include_uncertain else set(screen["uncertain"])
    screen_near_dup = set() if args.keep_test_overlap else set(screen["near_duplicate_of_test"])
    unknown_ids = sorted((set(other_speaker) | screen["uncertain"] | screen["suspect"]
                          | set(screen["near_duplicate_of_test"])) - set(by_id))
    if unknown_ids:
        raise SystemExit(f"speaker screen names clips that are not in the corpus: {unknown_ids[:10]}")
    screened_tests = sorted(test_set & (set(other_speaker) | set(screen["near_duplicate_of_test"])))
    if screened_tests:
        print(f"note: the screen lists held-out test clips {screened_tests}; they are never read anyway")

    # ---- choose rows ------------------------------------------------------------------------
    excluded: dict[str, list[str]] = {}
    chosen: dict[str, list[dict]] = {"train": [], "val": []}
    seen_sha: dict[str, str] = {}
    leak_ids: set[str] = set()  # (partly) speaks a held-out test line: never cached, trained on or referenced
    leak_detail: dict[str, list[str]] = {}
    for row in rows:
        if row["split"] == "test":
            continue
        if not args.keep_test_overlap:
            for t in tests:
                why = C.overlap_reason(row["text"], t["text"])
                if why:
                    leak_ids.add(row["id"])
                    leak_detail.setdefault(row["id"], []).append(f"{t['id']}: {why}")
        if row["id"] in screen_near_dup:
            leak_ids.add(row["id"])
            leak_detail.setdefault(row["id"], []).append(f"speaker screen: {screen['near_duplicate_of_test'][row['id']]}")
        reason = None
        if row["id"] in leak_ids:
            reason = "speaks-a-test-line"
        elif row["id"] in other_speaker:
            reason = f"other-speaker-{other_speaker[row['id']]}"
        elif row["kind"] != "speech":
            reason = "nonverbal"  # policy reason first (these are also speaker-'uncertain' outliers)
        elif row["id"] in uncertain:
            reason = "speaker-uncertain"
        elif row["sha256"] in seen_sha:
            reason = f"duplicate-audio-of-{seen_sha[row['sha256']]}"
        elif not row["tts_usable"] and not (args.include_barks and bark_ok(row)):
            reason = unusable_reason(row)
        elif args.exact_only and row["text_source"] == "asr":
            reason = "asr-only-text"
        elif (not args.keep_asr_suspect and row["text_source"] == "asr" and row["id"] in screen["suspect"]):
            reason = "asr-text-and-speaker-suspect"
        seen_sha.setdefault(row["sha256"], row["id"])
        if reason:
            excluded.setdefault(reason, []).append(row["id"])
        else:
            chosen[row["split"]].append(row)

    # ---- audio: trim + resample every non-test speech clip that may be trained on or referenced --
    # (the rows chosen above plus the other screened-clean speech clips, e.g. transcript-mismatch ones,
    # which the render step may still be handed as references; never other speakers or leak clips)
    blocked = leak_ids | set(other_speaker) | uncertain
    to_process = [r for r in rows if r["split"] in ("train", "val") and r["kind"] == "speech"
                  and r["id"] not in blocked]
    trims: dict[str, dict] = {}
    durations: dict[str, float] = {}
    feats: dict[str, dict] = {}
    problems: list[str] = []
    for row in to_process:
        src = C.source_audio(row, variant)
        if not src.exists():
            problems.append(f"{row['id']}: missing {src}")
            continue
        if variant == "original" and C.sha256_file(src) != row["sha256"]:
            problems.append(f"{row['id']}: sha256 differs from corpus.jsonl")
            continue
        audio, rate = C.read_mono(src)
        trimmed, info = C.trim_silence(audio, rate, args.lead_pad, args.tail_pad)
        audio16 = C.resample(trimmed, rate, C.ENCODER_SR)
        stats = C.write_wav_pcm16(cache / f"{row['id']}.wav", audio16, C.ENCODER_SR)
        info.update(source_rate=rate, source_seconds=round(len(audio) / rate, 3), **stats)
        trims[row["id"]] = info
        durations[row["id"]] = stats["duration_s"]
        feats[row["id"]] = delivery_features(audio16, C.ENCODER_SR, C.clean_text(row["text"]))
    if problems:
        raise SystemExit("audio problems:\n  " + "\n  ".join(problems))
    for split in chosen:
        chosen[split] = [r for r in chosen[split] if r["id"] in durations]
    stale = sorted(p for p in cache.glob("*.wav") if p.stem not in durations)
    for path in stale:  # clips an earlier build cached that are now screened out (e.g. other speakers)
        path.unlink()
    if args.with_val:  # production dataset: the val rows become training rows; no validation set
        chosen = {"train": sorted(chosen["train"] + chosen["val"], key=lambda r: r["id"]), "val": []}

    # ---- reference pool and prosody distances --------------------------------------------------
    # kept train rows only (verified text, not a leak, not another speaker), minus weak 'suspect' outliers
    pool = [r for r in chosen["train"] if args.ref_min <= durations[r["id"]] <= args.ref_max
            and r["id"] not in screen["suspect"]]
    zs = zscore_table(feats)
    fam = {r["id"]: families(r) for r in rows if r["split"] != "test"}

    def distance(a: str, b: str) -> float:
        d = float(np.linalg.norm(zs[a] - zs[b]))
        fa, fb = fam.get(a, set()), fam.get(b, set())
        if fa and fb:
            d *= 0.6 if fa & fb else 1.2
        return d

    rng = random.Random(args.seed)
    uses: Counter = Counter()
    pairs: list[dict] = []
    pairing_kind: dict[str, str] = {}
    unpaired_instruction_rows: list[str] = []

    def assign_refs(split_rows: list[dict]) -> dict[str, tuple[str, float]]:
        if args.ref_ratio <= 0 or not split_rows:
            return {}
        # every instruction row gets a ref (the layout idref_instr uses at inference); the rest of the
        # ref quota is drawn per duration quartile so short and long targets both see a reference
        forced = [r for r in split_rows if args.instructions == "directions" and instruction_for(r)]
        forced_ids = {r["id"] for r in forced}
        rest = sorted((r for r in split_rows if r["id"] not in forced_ids), key=lambda r: durations[r["id"]])
        quota = min(len(rest), max(0, round(args.ref_ratio * len(split_rows)) - len(forced)))
        chunks = [rest[q * len(rest) // 4:(q + 1) * len(rest) // 4] for q in range(4)]  # duration quartiles
        exact = [quota * len(chunk) / len(rest) if rest else 0.0 for chunk in chunks]
        counts = [int(x) for x in exact]
        for q in sorted(range(4), key=lambda i: (counts[i] - exact[i], i))[: quota - sum(counts)]:
            counts[q] += 1  # largest remainder: the quota is met exactly, also for the 16-row val split
        picked: list[dict] = list(forced)
        for chunk, count in zip(chunks, counts):
            picked += rng.sample(chunk, count)
        result: dict[str, tuple[str, float]] = {}
        for row in sorted(picked, key=lambda r: r["id"]):
            candidates = [p for p in pool if p["id"] != row["id"] and uses[p["id"]] < args.ref_max_uses
                          and not C.overlaps_text(p["text"], row["text"])
                          and not C.overlaps_text(row["text"], p["text"])]
            if not candidates:
                if row["id"] in forced_ids:
                    unpaired_instruction_rows.append(row["id"])
                continue
            nearest = args.ref_pairing == "nearest" or (args.ref_pairing == "mixed" and rng.random() < 0.5)
            if nearest:
                ranked = sorted(candidates, key=lambda p: distance(row["id"], p["id"]))[: args.ref_topk]
                ref = rng.choice(ranked)
            else:  # upstream-style: any eligible same-speaker clip
                ref = rng.choice(sorted(candidates, key=lambda p: p["id"]))
            uses[ref["id"]] += 1
            result[row["id"]] = (ref["id"], distance(row["id"], ref["id"]))
            pairing_kind[row["id"]] = "nearest" if nearest else "random"
        return result

    source_val_order: list[str] = []
    if args.pairs_from:
        source = C.manifest_dir(C.dataset_name(args.pairs_from, args.with_val))
        source_rows = {split: (C.read_jsonl(source / f"{split}.jsonl") if (source / f"{split}.jsonl").exists() else [])
                       for split in ("train", "val")}
        wanted = {r["id"] for split in chosen.values() for r in split}
        have = {e["id"] for rows_ in source_rows.values() for e in rows_}
        if wanted != have:
            raise SystemExit(f"--pairs-from {args.pairs_from}: row sets differ ({len(wanted ^ have)} ids); "
                             "rebuild both variants with the same options")
        refs = {}
        for e in source_rows["train"] + source_rows["val"]:
            if e.get("ref_id"):
                if e["ref_id"] not in durations:
                    raise SystemExit(f"--pairs-from: ref {e['ref_id']} missing from the {variant} cache")
                refs[e["id"]] = (e["ref_id"], distance(e["id"], e["ref_id"]))
                pairing_kind[e["id"]] = e.get("ref_pairing") or "copied"
                uses[e["ref_id"]] += 1
        source_val_order = [e["id"] for e in source_rows["val"]]
    else:
        refs = {**assign_refs(chosen["train"]), **assign_refs(chosen["val"])}

    # ---- manifests -------------------------------------------------------------------------------
    instructions: dict[str, str] = {}

    def entry(row: dict) -> dict:
        text = C.clean_text(row["text"])
        instr = instruction_for(row) if args.instructions == "directions" else ""
        if instr:
            instructions[row["id"]] = instr
        e = {
            "audio": str(cache / f"{row['id']}.wav"),
            "text": f"({instr}){text}" if instr else text,
            "duration": durations[row["id"]],
            "dataset_id": 0,
            "id": row["id"],
            "split": row["split"],
            "text_source": row["text_source"],
            "instruction": instr,
            "ref_audio": None,
            "ref_duration": None,
            "ref_id": None,
            "ref_distance": None,
            "ref_pairing": None,
        }
        if row["id"] in refs:
            ref_id, dist = refs[row["id"]]
            e.update(ref_audio=str(cache / f"{ref_id}.wav"), ref_duration=durations[ref_id], ref_id=ref_id,
                     ref_distance=round(dist, 3), ref_pairing=pairing_kind.get(row["id"]))
            pairs.append({"id": row["id"], "split": row["split"], "ref_id": ref_id, "distance": round(dist, 3),
                          "ref_duration": durations[ref_id], "pairing": pairing_kind.get(row["id"]),
                          "instruction": bool(instr)})
        return e

    train_rows = [entry(r) for r in sorted(chosen["train"], key=lambda r: r["id"])]
    val_rows = [entry(r) for r in chosen["val"]]
    # The trainer renders val rows 0 and 1 to TensorBoard (zero-shot): lead with the most typical
    # and the most animated plain-text val lines so those previews are informative.
    plain = [e for e in val_rows if not e["ref_audio"] and not e["instruction"]]
    if source_val_order:
        position = {cid: n for n, cid in enumerate(source_val_order)}
        val_rows.sort(key=lambda e: position[e["id"]])
    elif len(plain) >= 2:
        centroid = np.mean([zs[e["id"]] for e in val_rows], axis=0)
        typical = min(plain, key=lambda e: float(np.linalg.norm(zs[e["id"]] - centroid)))
        animated = max((e for e in plain if e is not typical),
                       key=lambda e: feats[e["id"]]["loud_db"] + feats[e["id"]]["f0_range"])
        val_rows = [typical, animated] + [e for e in val_rows if e is not typical and e is not animated]
    for manifest in (train_rows, val_rows):
        if manifest and not any(e["ref_audio"] for e in manifest):
            for e in manifest:  # an all-null ref column would be typed null by pyarrow
                for key in ("ref_audio", "ref_duration", "ref_id", "ref_distance", "ref_pairing"):
                    e.pop(key, None)

    # ---- leak guard: no test id anywhere in what we wrote -------------------------------------------
    pattern = C.test_id_pattern(test_set)
    blob = json.dumps(train_rows + val_rows)
    leaked = sorted(set(m.group(0) for m in pattern.finditer(blob)))
    cached_test = sorted(p.stem for p in cache.glob("*.wav") if p.stem in test_set)
    if leaked or cached_test:
        raise SystemExit(f"held-out test clip leaked into the manifests/cache: {leaked or cached_test}")
    written = train_rows + val_rows
    bad_rows = sorted({e["id"] for e in written if e["id"] in blocked})
    bad_refs = sorted({e["ref_id"] for e in written if e.get("ref_id")
                       and (e["ref_id"] in blocked or e["ref_id"] in screen["suspect"])})
    bad_cache = sorted(p.stem for p in cache.glob("*.wav") if p.stem in blocked)
    if bad_rows or bad_refs or bad_cache:
        raise SystemExit(f"speaker-screen violation: rows {bad_rows}, refs {bad_refs}, cached {bad_cache}")

    C.write_jsonl(out_dir / "train.jsonl", train_rows)
    manifest_hashes = {"train.jsonl": C.sha256_file(out_dir / "train.jsonl")}
    if args.with_val:
        stale_val = out_dir / "val.jsonl"
        if stale_val.exists():  # a val file here would be silently trained on twice by a hand-edited config
            stale_val.unlink()
    else:
        C.write_jsonl(out_dir / "val.jsonl", val_rows)
        manifest_hashes["val.jsonl"] = C.sha256_file(out_dir / "val.jsonl")
    with (out_dir / "features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "split", "duration", *FEATURES, "families", "text"])
        for cid, f in feats.items():
            writer.writerow([cid, by_id[cid]["split"], durations[cid], *(round(f[k], 4) for k in FEATURES),
                             "|".join(sorted(fam.get(cid, ()))), C.clean_text(by_id[cid]["text"])])

    # ---- config for this dataset ---------------------------------------------------------------------
    import yaml

    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    run = C.run_dir(dataset)
    template.update(
        train_manifest=str(out_dir / "train.jsonl"),
        val_manifest="" if args.with_val else str(out_dir / "val.jsonl"),
        save_path=str(run / "checkpoints"),
        tensorboard=str(run / "logs"),
    )
    yaml_path = out_dir / "train_lora_v2.yaml"
    yaml_path.write_text(
        f"# Generated by build_manifest.py from {args.template} for dataset {dataset} (--audio {variant}"
        f"{', --with-val: no validation set' if args.with_val else ''}).\n"
        "# Keys are exactly the official train() arguments; run it through train_lora_v2.py.\n"
        + yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # ---- report ------------------------------------------------------------------------------------------
    def dur_stats(manifest: list[dict]) -> dict:
        d = [e["duration"] for e in manifest]
        if not d:
            return {"rows": 0, "minutes": 0.0}
        return {"rows": len(d), "minutes": round(sum(d) / 60, 2), "min_s": min(d), "max_s": max(d),
                "median_s": round(float(np.median(d)), 3),
                "with_ref_audio": sum(1 for e in manifest if e.get("ref_audio")),
                "with_instruction": sum(1 for e in manifest if e.get("instruction")),
                "instruction_rows_with_ref": sum(1 for e in manifest if e.get("instruction") and e.get("ref_audio")),
                "asr_text_rows": sum(1 for e in manifest if e["text_source"] == "asr"),
                "from_val_split": sum(1 for e in manifest if e.get("split") == "val")}

    eff_batch = int(template["batch_size"]) * int(template["grad_accum_steps"])
    lead_cut = [t["lead_in"] - t["lead_out"] for t in trims.values() if t.get("lead_in") is not None]
    tail_cut = [t["tail_in"] - t["tail_out"] for t in trims.values() if t.get("tail_in") is not None]
    ref_durations = [p["ref_duration"] for p in pairs]
    report = {
        "variant": variant,
        "audio": variant,
        "dataset": dataset,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "builder": {"schema": C.MANIFEST_SCHEMA, "build_manifest_sha256": C.sha256_file(Path(__file__).resolve()),
                    "common_sha256": C.sha256_file(Path(C.__file__).resolve()), "character": C.SLUG,
                    "recipe_file": str(C.RECIPE_FILE), "recipe_sha256": C.RECIPE_SHA256},
        "manifest_sha256": manifest_hashes,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "train": dur_stats(train_rows),
        "val": dur_stats(val_rows),
        "schedule": {
            "effective_batch": eff_batch,
            "num_iters": template["num_iters"],
            "epochs": round(template["num_iters"] * eff_batch / max(1, len(train_rows)), 1),
            "steps_per_epoch": round(len(train_rows) / eff_batch, 2),
        },
        "excluded": {k: sorted(v) for k, v in sorted(excluded.items())},
        "excluded_because_they_speak_a_test_line": sorted(leak_ids),
        "test_overlap_detail": {k: v for k, v in sorted(leak_detail.items())},
        "speaker_screen": {
            "path": screen["path"],
            "sha256": screen["sha256"],
            "other_speaker_rows_excluded": {name: sorted(ids) for name, ids in sorted(
                {n: [i for i, s in other_speaker.items() if s == n and i in by_id and by_id[i]["split"] != "test"]
                 for n in set(other_speaker.values())}.items())},
            "uncertain_policy": "kept (--include-uncertain)" if args.include_uncertain else "excluded until listened to",
            "uncertain_excluded_rows": sorted(uncertain & {r["id"] for r in rows if r["split"] != "test"}),
            "suspect_in_training": sorted(screen["suspect"] & {e["id"] for e in written}),
            "suspect_policy": "kept for training, never used as ref_audio",
            "near_duplicate_of_test_excluded": sorted(screen_near_dup),
        },
        "stale_cache_files_removed": [p.stem for p in stale],
        "nonverbal_policy": "excluded from TTS rows and the ref pool (see build_manifest.py docstring)",
        "test_ids_never_read": sorted(test_set),
        "ref_pool_size": len(pool),
        "ref_pairing": {
            "mode": f"copied from {args.pairs_from}" if args.pairs_from else args.ref_pairing,
            "kinds": dict(Counter(p["pairing"] for p in pairs)),
            "median_prosody_distance": round(float(np.median([p["distance"] for p in pairs])), 3) if pairs else None,
            "ref_seconds": ({"min": min(ref_durations), "median": round(float(np.median(ref_durations)), 3),
                             "max": max(ref_durations)} if ref_durations else None),
            "distinct_refs": len(uses),
            "instruction_rows_without_ref": unpaired_instruction_rows,
            "note": "random = upstream-style same-speaker refs (timbre channel); nearest = prosody-matched",
        },
        "ref_pairs": pairs,
        "ref_uses": dict(uses.most_common()),
        "instructions": instructions,
        "trim": {
            "clips": len(trims),
            "trimmed_clips": sum(1 for t in trims.values() if t["trimmed"]),
            "median_lead_cut_s": round(float(np.median(lead_cut)), 3) if lead_cut else 0.0,
            "median_tail_cut_s": round(float(np.median(tail_cut)), 3) if tail_cut else 0.0,
            "max_lead_kept_s": max((t["lead_out"] or 0) for t in trims.values()),
            "max_tail_kept_s": max((t["tail_out"] or 0) for t in trims.values()),
            "rescaled_to_avoid_clipping": sorted(k for k, t in trims.items() if t["peak_rescaled"]),
        },
        "text_rewrites": [list(x) for x in C.TEXT_REWRITES],
        "outputs": {"cache": str(cache), "train": str(out_dir / "train.jsonl"),
                    "val": None if args.with_val else str(out_dir / "val.jsonl"),
                    "features": str(out_dir / "features.csv"), "config": str(yaml_path)},
        "seconds": round(time.time() - t0, 1),
    }
    C.atomic_write_json(out_dir / "manifest_report.json", report)
    print(json.dumps({k: report[k] for k in ("dataset", "train", "val", "schedule", "excluded",
                                             "excluded_because_they_speak_a_test_line", "ref_pool_size",
                                             "ref_pairing", "stale_cache_files_removed", "seconds")}, indent=2))
    print(f"instructions on {len(instructions)} rows; config -> {yaml_path}")


if __name__ == "__main__":
    main()
