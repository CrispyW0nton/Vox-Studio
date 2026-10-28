"""CPU checks for the delivery task outputs (VOICE_LAB_CHARACTER, default baodur). Exit code 1 if anything fails.

  * delivery_map.json: shared format, required labels (content/labels.json "required"), 3+ references each,
    instruction text present, notes name the map's actual rank 1, no caution clip anywhere, rank 1 unique across
    labels, picks (instruction_picks.json) applied verbatim.
  * every reference: train/val (never test), speech, exact transcript identical to corpus.jsonl, duration identical
    and within 3-10 s, audio file present with the corpus sha256, restored copy present at restored/<id>.wav when the
    lab has a restored/ folder (Bao-Dur's does; the restored copies are an optional post-processing experiment, so a
    lab without one is checked on the original audio only), every copy inside GPT-SoVITS's 3-10 s prompt window at
    16 kHz, speaker screen not "other_speaker", not in exclude_other_speakers.json (confirmed, uncertain or
    near-duplicate), text not overlapping any held-out line.
  * takes_script.json: one new line per required delivery (within content/lines.json "new_line_limits": Bao-Dur
    10-18 words; 1-3 sentences, no "!", not close to any corpus line, no held-out overlap) plus every held-out line
    with its exact corpus text; takes_script_pairs.json: a natural read of every non-natural new line.
  * drift: delivery_map.json, label_stats.json, delivery_candidates.json, voxcpm_instruction_variants.json,
    exclude_other_speakers.json, both takes scripts, bible_facts.json and performance_bible.md must be identical to
    a fresh rebuild from the current code (into a temporary folder), so code and shipped data cannot diverge.
  * label_stats.json parses with score_delivery.label_targets() (documentation keys skipped).
Writes <lab>/delivery/validation_report.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import labcfg  # noqa: E402  (stdlib only)

os.environ.setdefault("NUMBA_CACHE_DIR", str(labcfg.NUMBA_CACHE))
import soundfile as sf  # noqa: E402

from analyze import syllables, tokens  # noqa: E402
from build_delivery_map import content, leaks_test, longest_common_run, test_overlap, word_list  # noqa: E402
from make_takes_script import PAIR_SUFFIX  # noqa: E402

LAB = labcfg.LAB
OUT = labcfg.DELIVERY
REF_KEYS = {"id", "audio", "restored_audio", "text", "duration", "why"}


def new_line_limits() -> tuple[tuple[int, int], tuple[float, float], str]:
    """(words, estimated seconds, note) a new line must stay within (content/lines.json "new_line_limits")."""
    limits = labcfg.content("lines")["new_line_limits"]
    words, seconds = limits["words"], limits["seconds"]
    return (int(words[0]), int(words[1])), (float(seconds[0]), float(seconds[1])), str(limits.get("words_note", ""))


def estimate_seconds(text: str, profile: dict) -> float:
    """His pace: articulation-rate syllables plus his typical pause at each sentence/clause break."""
    stats = profile["stats"]
    rate = stats["articulation_rate_syl_s"]["median"]
    pause = stats["pause_mean_s"]["median"] or 0.35
    syl = sum(syllables(w) for w in tokens(text))
    sentence_breaks = max(0, len(re.findall(r"[.!?]+(?:\s|$)", text.strip() + " ")) - 1)
    clause_breaks = len(re.findall(r",|\s-\s|\.\.\.|;", text))
    return syl / rate + sentence_breaks * pause + clause_breaks * 0.5 * pause


def run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, cwd=str(HERE))


def drift_checks(failures: list[str]) -> dict:
    """Rebuild every generated output into a temporary folder and compare bytes with the shipped files."""
    checked = {}
    with tempfile.TemporaryDirectory(prefix=f"{labcfg.SLUG}_delivery_") as tmp:
        tmp = Path(tmp)
        if (OUT / "instruction_picks.json").is_file():
            shutil.copy2(OUT / "instruction_picks.json", tmp / "instruction_picks.json")
        steps = [
            (["build_delivery_map.py", "--out-dir", str(tmp)],
             ["delivery_map.json", "label_stats.json", "delivery_candidates.json", "voxcpm_instruction_variants.json"]),
            (["exclusions.py", "--out", str(tmp / "exclude_other_speakers.json")], ["exclude_other_speakers.json"]),
            (["make_takes_script.py", "--out-dir", str(tmp)], ["takes_script.json", "takes_script_pairs.json"]),
            (["bible_facts.py", "--out-dir", str(tmp)], ["bible_facts.json"]),
            (["make_bible.py", "--map-dir", str(OUT), "--out", str(tmp / "performance_bible.md")],
             ["performance_bible.md"]),
        ]
        for command, names in steps:
            result = run(command)
            if result.returncode != 0:
                failures.append(f"rebuild step {command[0]} failed: {result.stderr.strip()[-400:]}")
                continue
            for name in names:
                shipped, fresh = OUT / name, tmp / name
                same = shipped.is_file() and fresh.is_file() and shipped.read_bytes() == fresh.read_bytes()
                checked[name] = same
                if not same:
                    failures.append(f"{name} differs from a fresh rebuild by {command[0]} (code and data drifted: rerun "
                                    "the pipeline)")
    return checked


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    REQUIRED = content().required
    NEW_WORDS, NEW_SECONDS, words_note = new_line_limits()
    restored_dir = LAB / "restored"
    restored_lab = restored_dir.is_dir()  # Bao-Dur's lab keeps restored copies; a lab without them is not checked for them
    corpus = {json.loads(l)["id"]: json.loads(l) for l in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    summary = json.loads((LAB / "corpus_summary.json").read_text(encoding="utf-8"))
    test_ids = set(summary["test_ids"])
    test_texts = [corpus[i]["text"] for i in sorted(test_ids)]
    tests = [(i, corpus[i]["text"]) for i in sorted(test_ids)]
    if {i for i, r in corpus.items() if r["split"] == "test"} != test_ids:
        failures.append("corpus_summary test_ids disagree with corpus.jsonl split column")
    speaker = json.loads((OUT / "speaker_check.json").read_text(encoding="utf-8"))["clips"]
    profile = json.loads((OUT / "voice_profile.json").read_text(encoding="utf-8"))
    exclusions = json.loads((OUT / "exclude_other_speakers.json").read_text(encoding="utf-8"))
    other_confirmed = set(exclusions["confirmed_ids"])
    other_uncertain = {item["id"] for item in exclusions["uncertain"]}
    near_test = set(exclusions.get("near_duplicate_of_test", {}))
    if "near_duplicate_of_test" not in exclusions.get("rule", ""):
        failures.append("exclude_other_speakers.json rule text does not name near_duplicate_of_test")
    candidates = json.loads((OUT / "delivery_candidates.json").read_text(encoding="utf-8"))["candidates"]
    cautions = {(label, c["id"]): c.get("cautions", []) for label, items in candidates.items() for c in items}

    # ---------------------------------------------------------------- delivery map
    document = json.loads((OUT / "delivery_map.json").read_text(encoding="utf-8"))
    if set(document) != {"deliveries"}:
        failures.append(f"delivery_map top-level keys {sorted(document)} != ['deliveries']")
    deliveries = document.get("deliveries", {})
    for label in REQUIRED:
        if label not in deliveries:
            failures.append(f"missing label {label}")
    picks_path = OUT / "instruction_picks.json"
    picks = json.loads(picks_path.read_text(encoding="utf-8")).get("picks", {}) if picks_path.is_file() else {}
    checked_files, ref_count = 0, 0
    sha_cache: dict[str, str] = {}
    rank1s: dict[str, str] = {}
    for label, entry in deliveries.items():
        if set(entry) != {"references", "voxcpm_instruction", "notes"}:
            failures.append(f"{label}: keys {sorted(entry)}")
        instruction = entry.get("voxcpm_instruction", "")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 300 or re.search(r"[()]", instruction):
            failures.append(f"{label}: voxcpm_instruction empty, too long, or contains parentheses")
        if label in picks and instruction != picks[label]["instruction"].strip():
            failures.append(f"{label}: instruction_picks.json pick is not applied verbatim")
        references = entry.get("references", [])
        if len(references) < 3:
            failures.append(f"{label}: only {len(references)} references")
        if references:
            rank1s[label] = references[0].get("id")
            if f"Rank 1: {references[0].get('id')}" not in entry.get("notes", ""):
                failures.append(f"{label}: notes do not name the map's rank 1 ({references[0].get('id')})")
        seen = set()
        for rank, ref in enumerate(references):
            ref_count += 1
            where = f"{label}[{rank}] {ref.get('id')}"
            if set(ref) != REF_KEYS:
                failures.append(f"{where}: keys {sorted(ref)}")
            clip = corpus.get(ref.get("id"))
            if clip is None:
                failures.append(f"{where}: id not in corpus")
                continue
            if ref["id"] in seen:
                failures.append(f"{where}: duplicate within label")
            seen.add(ref["id"])
            if ref["id"] in test_ids or clip["split"] == "test":
                failures.append(f"{where}: TEST CLIP USED AS REFERENCE")
            if clip["split"] not in ("train", "val"):
                failures.append(f"{where}: split {clip['split']}")
            if clip["kind"] != "speech" or clip["text_source"] not in ("tlk-exact", "dlg"):
                failures.append(f"{where}: kind/text_source {clip['kind']}/{clip['text_source']}")
            if ref["text"] != clip["text"]:
                failures.append(f"{where}: text differs from corpus")
            if abs(float(ref["duration"]) - clip["duration"]) > 1e-6 or not (3.0 <= clip["duration"] <= 10.0):
                failures.append(f"{where}: duration {ref['duration']} (corpus {clip['duration']})")
            if (clip.get("wer") or 0.0) > 0.15:
                failures.append(f"{where}: transcript vs ASR WER {clip['wer']} > 0.15")
            if not clip.get("tts_usable"):
                failures.append(f"{where}: corpus marks it tts_usable=false (gsv/infer.py would skip it silently)")
            if ref["audio"] != clip["audio"]:
                failures.append(f"{where}: audio path differs from corpus")
            flag = speaker.get(ref["id"], {}).get("flag")
            if flag == "other_speaker":
                failures.append(f"{where}: speaker screen says other speaker")
            elif flag == "suspect":
                failures.append(f"{where}: speaker screen 'suspect' clip in the map (cautions are kept out)")
            if ref["id"] in other_confirmed:
                failures.append(f"{where}: listed as another character in exclude_other_speakers.json")
            if ref["id"] in other_uncertain:
                failures.append(f"{where}: listed as uncertain speaker in exclude_other_speakers.json")
            if ref["id"] in near_test:
                failures.append(f"{where}: near-duplicate of a held-out line (exclude_other_speakers.json)")
            expected_restored = LAB / "restored" / f"{ref['id']}.wav"
            if Path(ref["restored_audio"]) != expected_restored:
                failures.append(f"{where}: restored_audio is not {expected_restored}")
            caution = cautions.get((label, ref["id"]))
            if caution is None:
                failures.append(f"{where}: not in delivery_candidates.json (map edited by hand?)")
            elif caution:
                failures.append(f"{where}: caution clip in the map: {'; '.join(caution)}")
            if "caution:" in ref.get("why", ""):
                failures.append(f"{where}: 'why' lists a caution")
            leak = leaks_test(ref["text"], test_texts)
            if leak:
                failures.append(f"{where}: text overlaps held-out line '{leak[:40]}'")
            for key in ("audio", "restored_audio") if restored_lab else ("audio",):
                path = Path(ref[key])
                if not path.is_file():
                    failures.append(f"{where}: {key} missing: {path}")
                    continue
                info = sf.info(str(path))
                samples_16k = int(info.frames * 16000 / info.samplerate)
                if info.channels != 1:
                    warnings.append(f"{where}: {key} has {info.channels} channels")
                if not (48000 <= samples_16k <= 160000):
                    failures.append(f"{where}: {key} outside GPT-SoVITS 3-10 s window ({samples_16k / 16000:.3f} s)")
                checked_files += 1
            if ref["audio"] not in sha_cache:
                sha_cache[ref["audio"]] = hashlib.sha256(Path(ref["audio"]).read_bytes()).hexdigest() if Path(ref["audio"]).is_file() else ""
            if sha_cache[ref["audio"]] != clip["sha256"]:
                failures.append(f"{where}: audio sha256 differs from corpus")
    if not restored_lab:
        warnings.append(f"no {restored_dir} folder: references were checked on the original audio only (restored "
                        "copies are an optional post-processing experiment, never training or prompt input)")
    duplicates = sorted({clip for clip in rank1s.values() if list(rank1s.values()).count(clip) > 1})
    if duplicates:
        failures.append(f"rank 1 shared between labels: {duplicates}")
    usage: dict[str, int] = {}
    for entry in deliveries.values():
        for ref in entry.get("references", []):
            usage[ref["id"]] = usage.get(ref["id"], 0) + 1
    if any(count > 2 for count in usage.values()):
        failures.append(f"clips in more than two labels: {sorted(k for k, n in usage.items() if n > 2)}")

    # ---------------------------------------------------------------- label stats reader
    from score_delivery import label_targets  # noqa: E402  (imports numpy only at module level)

    targets = label_targets(OUT / "label_stats.json")
    if set(targets) != set(deliveries):
        failures.append(f"label_stats.json labels {sorted(targets)} != delivery_map labels {sorted(deliveries)}")
    validated = sorted(k for k, v in targets.items() if v["kind"] in ("direction", "question_ending"))

    # ---------------------------------------------------------------- takes scripts
    script = json.loads((OUT / "takes_script.json").read_text(encoding="utf-8"))
    if set(script) != {"lines"}:
        failures.append(f"takes_script top-level keys {sorted(script)}")
    lines = script.get("lines", [])
    ids = [line.get("id") for line in lines]
    if len(ids) != len(set(ids)):
        failures.append("takes_script has duplicate ids")
    new = [line for line in lines if line.get("source") == "new"]
    heldout = [line for line in lines if line.get("source") == "heldout"]
    if len(new) != len(REQUIRED) or len(heldout) != len(test_ids) or len(lines) != len(REQUIRED) + len(test_ids):
        failures.append(f"takes_script counts new={len(new)} heldout={len(heldout)} total={len(lines)}")
    if sorted(line["delivery"] for line in new) != sorted(REQUIRED):
        failures.append(f"new-line deliveries {sorted(line['delivery'] for line in new)} != one per required label")
    all_texts = [r["text"] for r in corpus.values()]
    estimates = {}
    for line in lines:
        keys = set(line)
        allowed = {"id", "text", "delivery", "source"} | ({"heldout_id"} if line.get("source") == "heldout" else set())
        if keys != allowed:
            failures.append(f"{line.get('id')}: keys {sorted(keys)}")
        if line.get("delivery") not in deliveries:
            failures.append(f"{line.get('id')}: delivery '{line.get('delivery')}' not in delivery_map")
    for line in new:
        sentences = len(re.findall(r"[.!?]+(?:\s|$)", line["text"].strip() + " "))
        seconds = estimate_seconds(line["text"], profile)
        words = word_list(line["text"])
        estimates[line["id"]] = round(seconds, 1)
        if not (1 <= sentences <= 3):
            failures.append(f"{line['id']}: {sentences} sentences")
        if not (NEW_WORDS[0] <= len(words) <= NEW_WORDS[1]):
            note = f" ({words_note})" if words_note else ""
            failures.append(f"{line['id']}: {len(words)} words outside {NEW_WORDS}{note}")
        if not (NEW_SECONDS[0] <= seconds <= NEW_SECONDS[1]):
            failures.append(f"{line['id']}: estimated {seconds:.1f} s outside {NEW_SECONDS}")
        if "!" in line["text"]:
            failures.append(f"{line['id']}: '!' is out of character for him and cues engines to shout")
        if test_overlap(line["text"], tests):
            failures.append(f"{line['id']}: overlaps a held-out line")
        for text in all_texts:
            other = word_list(text)
            if SequenceMatcher(None, " ".join(words), " ".join(other)).ratio() >= 0.6 or longest_common_run(words, other) >= 6:
                failures.append(f"{line['id']}: too close to corpus line '{text[:50]}'")
    for line in heldout:
        clip = corpus.get(line.get("heldout_id", ""))
        if clip is None or clip["split"] != "test":
            failures.append(f"{line['id']}: heldout_id {line.get('heldout_id')} is not a test clip")
            continue
        if line["text"] != clip["text"]:
            failures.append(f"{line['id']}: text differs from corpus")
    if {line.get("heldout_id") for line in heldout} != test_ids:
        failures.append(f"held-out lines do not cover exactly the {len(test_ids)} test ids")
    pairs = json.loads((OUT / "takes_script_pairs.json").read_text(encoding="utf-8")).get("lines", [])
    expected = {f"{line['id']}{PAIR_SUFFIX}": line["text"] for line in new if line["delivery"] != "natural"}
    got = {line.get("id"): line.get("text") for line in pairs}
    if got != expected or any(line.get("delivery") != "natural" or line.get("source") != "new" for line in pairs):
        failures.append("takes_script_pairs.json is not exactly a natural read of every non-natural new line")

    # ---------------------------------------------------------------- drift
    drift = drift_checks(failures)

    report = {
        "ok": not failures, "failures": failures, "warnings": warnings,
        "labels": {label: [r["id"] for r in entry["references"]] for label, entry in deliveries.items()},
        "validated_labels": validated,
        "references_checked": ref_count, "audio_files_checked": checked_files,
        "new_line_estimated_seconds": estimates, "rebuild_identical": drift,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "failures", "warnings", "validated_labels", "references_checked",
                                              "audio_files_checked", "new_line_estimated_seconds", "rebuild_identical")},
                     indent=1))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
