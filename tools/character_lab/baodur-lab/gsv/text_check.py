"""Second-opinion ASR on the few clips whose script line and corpus ASR differ on ordinary words (not names).

Evidence for <lab>/gsv/text_overrides.json (which export_dataset.py, infer.py and g2p_audit.py apply): each clip
is decoded twice with faster-whisper large-v3 on CPU (int8, cached model, local_files_only): once with no prompt,
once biased toward the script line (initial_prompt = the script text), with word probabilities. A word the free
decode hears with p >= 0.9 (or that both decodes agree on) overrides the script when the biased decode restores the
script word with p < 0.3; a script word restored with p >= 0.3 is ambiguous and the script stays. Neither decode
uses a names prompt: the check is about ordinary words. Run with the VoxCPM2 engine python (it has
faster-whisper); CPU only, one model, a handful of clips (~1 min per clip):

  CUDA_VISIBLE_DEVICES=-1 <VoxStudio voxcpm2 .venv python> text_check.py [--ids <id>,<id>,...]
Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters. The default clips are the
character's <lab>/gsv/text_check_ids.json ({"ids": [...]}, hand-picked: script-vs-ASR differences on ordinary
words in the exported list; names, homophones and spelling are not checked). Without that file and without --ids,
the candidates (exact-text train/val clips whose ASR disagrees, worst first) are listed and nothing is decoded.
Writes <lab>/gsv/text_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402

WHISPER_CACHE = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local")) / "VoxStudio" / "engines" \
    / "voxcpm2" / "cache" / "faster-whisper"


def default_ids(path: Path = C.TEXT_CHECK_IDS) -> str | None:
    """The character's hand-picked clips (<lab>/gsv/text_check_ids.json), as a comma list; None without the file."""
    if not path.exists():
        return None
    ids = [str(i).strip() for i in (C.read_json(path).get("ids") or []) if str(i).strip()]
    if not ids:
        raise SystemExit(f"{path} lists no 'ids'")
    return ",".join(ids)


def candidates(rows: list[dict], limit: int = 40) -> list[str]:
    """Exact-text train/val speech clips whose corpus ASR disagrees with the script (WER > 0), worst first."""
    pool = [r for r in rows if r.get("split") in ("train", "val") and r.get("kind") == "speech"
            and r.get("text_source") in C.EXACT_SOURCES and r.get("wer")]
    pool.sort(key=lambda r: (-float(r["wer"]), r["id"]))
    return [f"{r['id']} (wer {r['wer']}): script {r.get('text')!r} / asr {r.get('asr')!r}" for r in pool[:limit]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default=None, help=f"comma list (default: the ids in {C.TEXT_CHECK_IDS})")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(C.GSV_DATA / "text_check.json"))
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in ("", "-1"):
        raise SystemExit("CPU only: run with CUDA_VISIBLE_DEVICES=-1 (or empty)")
    rows = {r["id"]: r for r in C.load_corpus()}
    args.ids = args.ids or default_ids()
    if not args.ids:
        listing = "\n  ".join(candidates(list(rows.values()))) or "(none: no exact-text clip disagrees with its ASR)"
        raise SystemExit(f"{C.TEXT_CHECK_IDS} not found and no --ids given. Candidates for {C.DISPLAY_NAME} (pick the "
                         f"ordinary-word differences, not names; write them to that file as {{\"ids\": [...]}} or "
                         f"pass --ids):\n  {listing}")
    from faster_whisper import WhisperModel

    guard = C.HeldOut(list(rows.values()))
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    for clip in ids:
        if guard.reason(clip):
            raise SystemExit(f"{clip}: {guard.reason(clip)}")
    t0 = time.time()
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", download_root=str(WHISPER_CACHE),
                         local_files_only=True, cpu_threads=args.threads)
    C.log(f"faster-whisper large-v3 int8 loaded in {time.time() - t0:.1f}s")
    out = {"model": "large-v3 int8 cpu", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "clips": {}}
    for clip in ids:
        row = rows[clip]
        entry = {"script": row["text"], "corpus_asr": row.get("asr"), "corpus_wer": row.get("wer")}
        for mode, prompt in (("free", None), ("script_biased", row["text"])):
            segments, _info = model.transcribe(row["audio"], language="en", beam_size=5, temperature=0.0,
                                               condition_on_previous_text=False, initial_prompt=prompt,
                                               word_timestamps=True, vad_filter=False)
            words = [(w.word.strip(), round(float(w.probability), 3)) for s in segments for w in (s.words or [])]
            entry[mode] = {"text": " ".join(w for w, _ in words), "words": words}
        out["clips"][clip] = entry
        C.log(f"{clip}: script {row['text']!r}\n    free   {entry['free']['text']!r}\n"
              f"    biased {entry['script_biased']['text']!r}")
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(args.out, out)
    C.log(f"-> {args.out}")


if __name__ == "__main__":
    main()
