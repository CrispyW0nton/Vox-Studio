"""Build a voice-lab character corpus (VOICE_LAB_CHARACTER, default baodur): one row per decoded game clip with the best transcript,
an ASR cross-check, a speech/nonverbal kind and a fixed train/val/test split.

Run with the Vox VoxCPM2 venv (it already has faster-whisper and the cached large-v3 model):
  %LOCALAPPDATA%/VoxStudio/engines/voxcpm2/.venv/Scripts/python.exe build_corpus.py
Only reads game-derived clips; writes BaoDurLab/corpus.jsonl and corpus_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "characters"))
import character  # noqa: E402  (VOICE_LAB_CHARACTER selects the character; default baodur)

CHAR = character.load()
DECODED = CHAR["source_audio"]
EXACT = CHAR["exact_manifest"]
# Only Bao-Dur had an earlier Vox training run whose DLG transcripts fill gaps in the exact manifest.
PREVIOUS = Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "voxcpm_training" / CHAR.get("vox_training_dir", "bao-dur")
WHISPER_CACHE = Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "engines" / "voxcpm2" / "cache" / "faster-whisper"
OUT = CHAR["lab"]

NONVERBAL_DIRECTION = re.compile(r"\b(grunt|effort|pain|death|dying|scream|laugh|chuckle|cough|sigh|gasp|groan|breath)", re.I)
INTERJECTION = re.compile(r"^(?:a+h+|u+h+|h+m+|m+h*m+|hn+h*|u+g+h+|a+r+g+h+|o+f+|he+h|ha+|n+g+h*|o+h+)$")
NAMES = CHAR["asr_prompt"]


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.casefold().replace("-", " ")).split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / len(ref)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exact = json.loads(EXACT.read_text(encoding="utf-8"))["entries"]
    previous: dict[str, str] = {}
    for name in ("train.jsonl", "validation.jsonl"):
        if CHAR["slug"] != "baodur" or not (PREVIOUS / name).is_file():
            continue
        for line in (PREVIOUS / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            previous[Path(row["audio"]).name.casefold()] = row["text"]

    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                         download_root=str(WHISPER_CACHE), local_files_only=True)

    rows = []
    for path in sorted(DECODED.glob("*.wav")):
        key = path.name.casefold()
        info = sf.info(path)
        entry = exact.get(path.name) or exact.get(key) or {}
        text, source = entry.get("text", ""), "tlk-exact"
        if not text and key in previous:
            text, source = previous[key], "dlg"
        segments, _ = model.transcribe(str(path), language="en", beam_size=5, vad_filter=False,
                                       condition_on_previous_text=False, initial_prompt=NAMES)
        asr = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            text, source = asr, "asr"
        directions = list(entry.get("performance_directions", []))
        spoken = words(text)
        nonverbal = not spoken or all(INTERJECTION.match(w) for w in spoken) or any(NONVERBAL_DIRECTION.search(d) for d in directions)
        rows.append({
            "id": path.stem, "audio": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration": round(info.frames / info.samplerate, 3), "text": text, "text_source": source,
            "asr": asr, "wer": round(wer(text, asr), 3) if source != "asr" else None,
            "directions": directions, "delivery_tags": list(entry.get("delivery_tags", [])),
            "kind": "nonverbal" if nonverbal else "speech",
        })
        print(f"{path.stem}\t{source}\t{rows[-1]['wer']}\t{text[:70]}", flush=True)

    # Held-out lines for the blind A/B: exact transcripts the actor actually spoke, 3-9 s,
    # spread evenly across the voice set. Validation is chosen the same way from what remains.
    def eligible(r):
        return r["kind"] == "speech" and r["text_source"] != "asr" and (r["wer"] or 0) <= 0.15 and 3 <= r["duration"] <= 9
    pool = [r for r in rows if eligible(r)]
    test = pool[:: max(1, len(pool) // 10)][:10]
    rest = [r for r in pool if r not in test]
    val = rest[:: max(1, len(rest) // 16)][:16]
    for r in rows:
        r["split"] = "test" if r in test else "val" if r in val else "train"
        # TTS training needs text the actor really said; VC training only needs clean audio.
        r["tts_usable"] = r["kind"] == "speech" and (r["text_source"] == "asr" or (r["wer"] or 0) <= 0.35) and 1.0 <= r["duration"] <= 20

    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "clips": len(rows), "seconds": round(sum(r["duration"] for r in rows), 1),
        "text_sources": {s: sum(r["text_source"] == s for r in rows) for s in ("tlk-exact", "dlg", "asr")},
        "nonverbal": sum(r["kind"] == "nonverbal" for r in rows),
        "transcript_mismatch_over_0_35": [r["id"] for r in rows if r["wer"] is not None and r["wer"] > 0.35],
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "tts_usable_train": sum(r["tts_usable"] and r["split"] == "train" for r in rows),
        "tts_usable_train_seconds": round(sum(r["duration"] for r in rows if r["tts_usable"] and r["split"] == "train"), 1),
        "test_ids": [r["id"] for r in test],
    }
    (OUT / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
