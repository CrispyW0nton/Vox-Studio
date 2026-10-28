"""Every number cited in performance_bible.md, recomputed from the analysis outputs (CPU, a few seconds).

Reads features.jsonl, features.restored.jsonl, voice_profile.json, label_stats.json, delivery_map.json,
delivery_candidates.json, takes_script.json, exclude_other_speakers.json, the WavLM identity cache and
corpus.jsonl, plus (read only) the train/validation manifests of the Vox adapter trained for the character before
the lab existed (%LOCALAPPDATA%/VoxStudio/voxcpm_training/<name>; Bao-Dur's "July adapter"), when there is one;
writes <lab>/delivery/bible_facts.json (VOICE_LAB_CHARACTER, default baodur). No audio and no model is loaded.
features.jsonl holds train/val clips only, so held-out test clips are never read (the Vox manifests are only
scanned for which held-out ids they contain). The address-term facts (<term>_lines, ...) follow the character
config's address_term. make_bible.py renders performance_bible.md from these facts.

Run (VoxCPM2 runtime python; read-only venv, so never write bytecode into it):
  CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 <voxcpm2 .venv python> bible_facts.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import EXACT_SOURCES, profile_member, syllables, text_features, tokens  # noqa: E402
import labcfg  # noqa: E402

LAB = labcfg.LAB
OUT = labcfg.DELIVERY
JULY = labcfg.VOX_TRAINING  # Bao-Dur: %LOCALAPPDATA%/VoxStudio/voxcpm_training/bao-dur (the July adapter)
SOFTENERS = {"just": r"\bjust\b", "sure": r"\bsure\b", "maybe": r"\bmaybe\b", "all right": r"\ball right\b",
             "I think": r"\bi think\b", "you know": r"\byou know\b", "I don't know": r"\bi don't know\b",
             "I guess": r"\bi guess\b", "a little": r"\ba little\b", "perhaps": r"\bperhaps\b"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(values, q) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.percentile(values, q)), 3) if values else None


def spread(values) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": pct(values, 50), "p10": pct(values, 10), "p90": pct(values, 90),
            "p05": pct(values, 5), "p95": pct(values, 95), "p25": pct(values, 25), "p75": pct(values, 75),
            "mean": round(float(np.mean(values)), 3)}


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Recompute every figure {labcfg.NAME}'s performance bible cites (CPU).")
    parser.add_argument("--out-dir", type=Path, default=OUT, help="where bible_facts.json goes (inputs: the lab)")
    parser.add_argument("--map-dir", type=Path, default=OUT, help="folder with delivery_map/label_stats/candidates")
    args = parser.parse_args()
    corpus = {r["id"]: r for r in load_jsonl(LAB / "corpus.jsonl")}
    test_ids = {i for i, r in corpus.items() if r["split"] == "test"}
    rows = load_jsonl(OUT / "features.jsonl")
    restored = {r["id"]: r for r in load_jsonl(OUT / "features.restored.jsonl")} \
        if (OUT / "features.restored.jsonl").is_file() else {}
    profile = json.loads((OUT / "voice_profile.json").read_text(encoding="utf-8"))
    label_stats = {k: v for k, v in json.loads((args.map_dir / "label_stats.json").read_text(encoding="utf-8")).items()
                   if not k.startswith("_") and isinstance(v, dict)}
    delivery_map = json.loads((args.map_dir / "delivery_map.json").read_text(encoding="utf-8"))["deliveries"]
    candidates = json.loads((args.map_dir / "delivery_candidates.json").read_text(encoding="utf-8"))
    script = json.loads((args.map_dir / "takes_script.json").read_text(encoding="utf-8")) \
        if (args.map_dir / "takes_script.json").is_file() else json.loads((OUT / "takes_script.json").read_text(encoding="utf-8"))
    exclusions = json.loads((OUT / "exclude_other_speakers.json").read_text(encoding="utf-8"))
    other = set(exclusions["confirmed_ids"])
    if any(r["split"] == "test" for r in rows):
        raise SystemExit("features.jsonl contains test rows; the bible must be built from train/val only")

    members = [r for r in rows if profile_member(r)]
    # Text habits: all of his own exact-text train/val lines (any length), other characters removed.
    habit = [r for r in rows if r["kind"] == "speech" and r["text_source"] in EXACT_SOURCES and r["id"] not in other
             and r.get("speaker_flag") != "other_speaker"]
    facts: dict = {"profile_members": len(members), "profile_seconds": round(sum(r["duration"] for r in members), 1),
                   "profile_rule": profile["member_rule"], "habit_lines": len(habit),
                   "habit_seconds": round(sum(r["duration"] for r in habit), 1)}

    # ------------------------------------------------------------------ voice
    keys = ["f0_median_hz", "f0_range_st", "f0_std_st", "articulation_rate_syl_s", "speech_rate_syl_s",
            "words_per_min", "pause_count", "pause_mean_s", "pause_ratio", "energy_slope_db", "alpha_ratio_db",
            "hammarberg_db", "centroid_hz", "level_dbfs", "dynamic_db", "final_rise_st", "lead_silence_s",
            "tail_silence_s", "peak_dbfs", "low_f0_ratio"]
    facts["register_center_hz"] = profile["register"]["center_hz"]
    facts["voice"] = {k: spread([r.get(k) for r in members]) for k in keys}
    multi = [r for r in members if r["text_sentences"] >= 2 and r["pause_count"] >= 1]
    single = [r for r in members if r["text_sentences"] == 1 and r["pause_count"] >= 1]
    facts["pauses"] = {
        "longest_pause_in_multi_sentence_lines": spread([r["pause_max_s"] for r in multi]),
        "longest_pause_in_single_sentence_lines": spread([r["pause_max_s"] for r in single]),
        "lines_without_pause_share": round(float(np.mean([r["pause_count"] == 0 for r in members])), 3),
    }
    statements = [r for r in members if not r["text_has_question"] and r.get("final_rise_st") is not None]
    questions = [r for r in members if r["text_question"] and r.get("final_rise_st") is not None]
    habit_questions = [r for r in habit if r["text"].strip().endswith("?") and r.get("final_rise_st") is not None]
    facts["endings"] = {
        "statements": len(statements),
        "statement_final_rise_median_st": pct([r["final_rise_st"] for r in statements], 50),
        "statements_falling_more_than_0_5_st": sum(r["final_rise_st"] < -0.5 for r in statements),
        "statements_rising_more_than_1_st": sum(r["final_rise_st"] > 1.0 for r in statements),
        "questions": len(questions),
        "questions_rising_more_than_1_st": sum(r["final_rise_st"] > 1.0 for r in questions),
        "habit_questions": len(habit_questions),
        "habit_questions_rising_more_than_1_st": sum(r["final_rise_st"] > 1.0 for r in habit_questions),
    }
    creak = sorted((r for r in members if r.get("low_f0_ratio")), key=lambda r: -r["low_f0_ratio"])[:8]
    facts["creak_top"] = [{"id": r["id"], "low_f0_ratio": r["low_f0_ratio"], "text": r["text"][:70]} for r in creak]

    # Vox's own measurements and delivery reading of the same clips.
    facts["vox_features"] = {k: spread([r["vox"][k] for r in members if "vox" in r])
                             for k in ("pitch_range_semitones", "pitch_variation_semitones", "characters_per_second",
                                       "pause_ratio", "terminal_pitch_delta", "dynamic_db", "energy_slope_db")}
    facts["vox_readings"] = dict(Counter(r.get("vox_reading") or "" for r in members).most_common())

    # ------------------------------------------------------------------ text habits
    texts = [r["text"] for r in habit]
    tf = [text_features(t) for t in texts]
    words = [f["words"] for f in tf]
    sentences = [f["sentences"] for f in tf]
    term, key = labcfg.address_word(), labcfg.address_key()  # the term he addresses the player by ('General')
    facts["text"] = {
        "words_per_line": spread(words),
        "words_per_sentence_mean": round(sum(words) / max(1, sum(sentences)), 2),
        f"{key}_lines": sum(bool(re.search(labcfg.address_regex(), t)) for t in texts),
        f"{key}_closing_tag_lines": sum(bool(re.search(rf",\s*{term}\s*[.?!]", t)) for t in texts),
        f"{key}_opening_lines": sum(bool(re.search(rf"(?:^|[.?!]\s+){term},", t)) for t in texts),
        "softeners": {name: sum(bool(re.search(p, t, re.I)) for t in texts) for name, p in SOFTENERS.items()},
        "contractions": sum(f["contractions"] for f in tf),
        "formal_forms": sum(f["formal_forms"] for f in tf),
        "ellipsis_lines": sum(f["ellipsis"] for f in tf),
        "dash_lines": sum(f["dashes"] > 0 for f in tf),
        "exclamation_lines": [r["id"] for r in habit if "!" in r["text"]],
        "question_lines": sum(f["question"] for f in tf),
        "durations": spread([r["duration"] for r in habit]),
    }
    angry = [ref["text"] for ref in delivery_map.get("angry", {}).get("references", [])]
    angry_tf = [text_features(t) for t in angry]
    facts["text"]["angry_refs"] = len(angry)
    facts["text"]["angry_refs_contractions"] = sum(f["contractions"] for f in angry_tf)
    facts["text"]["angry_refs_formal_forms"] = sum(f["formal_forms"] for f in angry_tf)
    facts["text"]["angry_refs_lines_with_contractions"] = [
        ref["id"] for ref, f in zip(delivery_map.get("angry", {}).get("references", []), angry_tf) if f["contractions"]]

    # ------------------------------------------------------------------ restored vs original (information only)
    pairs = [(r, restored[r["id"]]) for r in members if r["id"] in restored]
    diff = {}
    for key in ("f0_median_hz", "f0_range_st", "articulation_rate_syl_s", "pause_ratio", "final_rise_st",
                "alpha_ratio_db", "hammarberg_db", "centroid_hz", "level_dbfs"):
        values = [abs(a[key] - b[key]) for a, b in pairs if a.get(key) is not None and b.get(key) is not None]
        diff[key] = {"median_abs_diff": pct(values, 50), "p95_abs_diff": pct(values, 95), "max_abs_diff": pct(values, 100)}
    facts["restored_vs_original"] = {"clips": len(pairs), "analysis_rate_hz": 16000, "abs_diff": diff}

    # ------------------------------------------------------------------ labels and references
    facts["labels"] = {}
    for label, entry in label_stats.items():
        refs = delivery_map.get(label, {}).get("references", [])
        facts["labels"][label] = {
            "kind": entry["kind"], "perm_p": entry["perm_p"], "validated": entry["validated"],
            "pool": len(entry["ids"]), "signature": entry["signature"],
            "deltas": {k: v["delta"] for k, v in entry["features"].items()},
            "references": [r["id"] for r in refs], "durations": [r["duration"] for r in refs],
            "rank1": refs[0]["id"] if refs else None, "rank1_text": refs[0]["text"] if refs else None,
            "rank1_duration": refs[0]["duration"] if refs else None,
            "rank2": refs[1]["id"] if len(refs) > 1 else None,
            "kept_out": [k["id"] for k in candidates.get("kept_out", {}).get(label, [])],
            "instruction": delivery_map.get(label, {}).get("voxcpm_instruction"),
            "question": entry.get("question"),
        }
    facts["reference_clips_unique"] = len({r["id"] for v in delivery_map.values() for r in v["references"]})
    facts["references_total"] = sum(len(v["references"]) for v in delivery_map.values())
    facts["eligible"], facts["excluded"] = candidates["eligible"], len(candidates["excluded"])

    # ------------------------------------------------------------------ identity calibration (WavLM cache)
    cache = OUT / ".cache" / "wavlm_centroid_original.json"
    if cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        identity = dict(data["calibration"])
        sims = (data.get("impostors") or {}).get("sims") or {}
        durations = (data.get("impostors") or {}).get("durations") or {}
        if sims:
            values = list(sims.values())
            longer = [v for c, v in sims.items() if durations.get(c, 0) >= 3.0]
            identity["impostors"] = {"n": len(values), "median": pct(values, 50), "p95": pct(values, 95),
                                     "max": round(max(values), 4), "max_clip": max(sims, key=sims.get),
                                     "max_clip_duration": durations.get(max(sims, key=sims.get)),
                                     "max_3s": round(max(longer), 4) if longer else None}
            identity["gate"] = round(max(identity["loo_p05"], identity["impostors"]["p95"],
                                         identity["impostors"]["max_3s"] or 0.0), 4)
        else:  # no other character in his files: the scorer's gate is his own p05 (score_delivery.py)
            identity["gate"] = round(identity["loo_p05"], 4)
        facts["identity"] = identity

    # ------------------------------------------------------------------ other speakers and held-out leaks
    pool = [r for r in corpus.values() if r["tts_usable"] and r["split"] == "train"]
    pool_s = sum(r["duration"] for r in pool)
    confirmed_pool_s = sum(r["duration"] for r in pool if r["id"] in other)
    near = exclusions.get("near_duplicate_of_test", {})
    near_pool_s = sum(r["duration"] for r in pool if r["id"] in near and r["id"] not in other)
    facts["other_speakers"] = {
        "confirmed": {name: len(items) for name, items in exclusions["confirmed"].items()},
        "confirmed_ids": {name: [i["id"] for i in items] for name, items in exclusions["confirmed"].items()},
        "confirmed_clips": len(other), "confirmed_seconds": exclusions["confirmed_seconds"],
        "confirmed_seconds_in_tts_pool": round(confirmed_pool_s, 1), "tts_pool_seconds": round(pool_s, 1),
        "share_of_tts_pool": round(confirmed_pool_s / pool_s, 3) if pool_s else None,
        "near_duplicate_seconds_in_tts_pool": round(near_pool_s, 1),
        "tts_pool_seconds_after": round(pool_s - confirmed_pool_s - near_pool_s, 1),
        "uncertain": [u["id"] for u in exclusions["uncertain"]],
        "near_duplicate_of_test": near,
    }
    july = {}
    for name in ("train.jsonl", "validation.jsonl"):
        path = JULY / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            found = {clip_id for clip_id in corpus if clip_id in text}  # which of his corpus clips it names
            july[name] = {"clips": len(found), "heldout": sorted(found & test_ids),
                          "near_duplicates": sorted(found & set(near))}
    facts["july_adapter"] = july

    # ------------------------------------------------------------------ takes script
    stats = profile["stats"]
    rate, pause = stats["articulation_rate_syl_s"]["median"], stats["pause_mean_s"]["median"] or 0.35
    new_lines = []
    for line in script["lines"]:
        if line["source"] != "new":
            continue
        syl = sum(syllables(w) for w in tokens(line["text"]))
        breaks = max(0, len(re.findall(r"[.!?]+(?:\s|$)", line["text"].strip() + " ")) - 1)
        clauses = len(re.findall(r",|\s-\s|\.\.\.|;", line["text"]))
        new_lines.append({"id": line["id"], "delivery": line["delivery"], "words": word_count(line["text"]),
                          "seconds": round(syl / rate + breaks * pause + clauses * 0.5 * pause, 1)})
    facts["script"] = {"new": new_lines, "heldout": {l["heldout_id"]: l["delivery"] for l in script["lines"]
                                                     if l["source"] == "heldout"},
                       "lines": len(script["lines"])}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bible_facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    brief = {k: facts[k] for k in ("profile_members", "profile_seconds", "habit_lines", "habit_seconds", "endings")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
