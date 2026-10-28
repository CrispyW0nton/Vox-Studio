"""Build <lab>/delivery/delivery_map.json (shared format) from features.jsonl (VOICE_LAB_CHARACTER, default baodur).

The character's content comes from <lab>/delivery/content/labels.json (label set, curated evidence, TLK note rules,
tag overrides, context-only and spoken-mismatch clips, prior prosody signatures, VoxCPM2 instruction variants A/B,
craft notes) and content/lexicon.json (g2p_risky tokens); everything below is the method, the same for every
character.

Every reference is a train/val speech clip with an exact transcript (tlk-exact|dlg, WER <= 0.15), 3.0-10.0 s long
(GPT-SoVITS rejects prompts outside 3-10 s), not flagged as another speaker by speaker_check.py, not listed in
exclude_other_speakers.json (confirmed, uncertain, or near-duplicate of a held-out line), and not overlapping any
held-out test line. Test clips are never candidates.

1. Candidates per label: score = 3.0 x curated text evidence (my reading of the line in its scene)
                                + 2.0 x TLK direction / delivery-tag evidence
                                + 0.8 x prosody fit (a hand-written prior over his own z-scores, tanh-bounded)
                                + quality (exact match, duration, speaker similarity, G2P-risky tokens, silence).
   Caution = speaker screen suspect, atypical pitch contour, the prior signature contradicted (fit <= -0.5 without
   a strong TLK note), a TLK text that differs from what he says (labels.json spoken_mismatch), or context-only
   (labels.json context_only).
2. References: a global greedy by score fills up to six CAUTION-FREE clips per label; a clip appears in at most
   two labels. Caution clips never enter delivery_map.json (every engine and the eval treat every listed reference
   as usable, and the eval takes each label's prosody target from all of them); they stay in
   delivery_candidates.json with their reasons, and the label's notes name them.
3. Target = what the label measurably does in HIS clips, from a larger pool than the six references: the label's
   evidence-backed candidates (curated or TLK evidence >= 0.6; no speaker-suspect or atypical-contour clip; outside
   questioning, no line that ends in a question, whose ending and range follow the question form), best 12 by
   score. target_z = median z over TARGET_KEYS. Signature = the target features with |median| >= 0.3 on whose sign
   >= 80% of the pool agree. perm_p = how often a random set of the same size drawn from his own statement clips
   is displaced at least as far (|median z|, 4000 draws, fixed seed). kind:
     direction    signature and perm_p <= 0.05 (VALIDATED: the scorers may score direction on the signature);
     descriptive  signature but perm_p > 0.05 (described, never scored: no more displaced than chance);
     none         no feature the pool agrees on.
   Questioning: the ending of his yes/no questions (profile clips) against his statements, scored on question
   lines only and from his medians. Natural: closeness to his medians.
4. Rank 1 is what every engine uses by default: caution-free, unique across labels (global greedy), best score
   (+0.3 if >= 5 s; VoxCPM2's docs advise 5-30 s of reference audio), and it must not run against what the label's
   own clips agree on: along_loo (projection onto the signature of the pool WITHOUT the clip; 1 = as far as the
   pool moved) >= 0.5 for validated labels and >= 0 for descriptive ones. Labels with no signature keep the
   evidence order (no measured direction exists to overrule the director's TLK note). Natural rank 1 needs prosody
   fit >= 0.3; questioning rank 1 must be a yes/no question that rises.
5. Order: rank 1; then the evidence-best clip if it is not rank 1 (the "intent alternative", usually the TLK-noted
   take, so --refs-per-line 2 A/Bs it by ear); then the rest by score.
6. Notes are generated from the result (kind, perm_p, signature in natural units, rank 1 and 2, the caution clips
   kept out) plus a short hand-written craft note, so the notes can never name a different rank 1 than the map.
VoxCPM2 instruction picks applied after a probe (score_delivery.py --apply-from) are stored in
<out-dir>/instruction_picks.json as {"picks": {label: {"variant", "run", "note", "instruction"}}}; every rebuild
re-applies the exact instruction string that was rendered and heard, so rebuilding never reverts a pick.

Run (CPU, any of the lab pythons):
  CUDA_VISIBLE_DEVICES=-1 python build_delivery_map.py [--features features.jsonl] [--top 6] [--out-dir DIR]
Writes delivery_map.json, voxcpm_instruction_variants.json, delivery_candidates.json (top-12 per label with
scores and along_loo) and label_stats.json (per label: target pool, feature deltas, signature, perm_p, kind,
flag bounds). The output is deterministic: validate.py rebuilds into a temporary folder and requires identical
bytes, so the shipped files always match this code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labcfg  # noqa: E402

LAB = labcfg.LAB
OUT = labcfg.DELIVERY
EXCLUSIONS = OUT / "exclude_other_speakers.json"
PICKS_NAME = "instruction_picks.json"  # read from the output folder (the lab, or a test folder)
EXACT_SOURCES = {"tlk-exact", "dlg"}
MIN_S, MAX_S = 3.0, 10.0
MAX_WER = 0.15
CONTRADICTION_FIT = -0.5   # prior prosody fit at or below this contradicts the label ...
STRONG_DIRECTION = 0.8     # ... unless a TLK note/tag for the label is at least this strong
# Features the scorers compare (score_delivery.py imports these). Level is left out on purpose: engines render
# at arbitrary gain and every take is mastered afterwards.
TARGET_KEYS = ["f0_median_st", "f0_range_st", "final_rise_st", "articulation_rate_syl_s", "pause_ratio",
               "alpha_ratio_db", "energy_slope_db"]
SIGNATURE_MIN_Z = 0.3      # a signature feature moves at least this far (median, his z units)
SIGNATURE_AGREE = 0.8      # ... and at least this share of the pool agrees on its sign
POOL_MIN_EVIDENCE = 0.6    # target pool: curated or TLK evidence at least this strong
POOL_MAX, POOL_MIN = 12, 4
PERM_DRAWS, PERM_SEED, VALIDATED_P = 4000, 20260930, 0.05
REP_MIN = 0.5              # rank-1 along_loo for validated labels: at least half as far as his other clips of it
DESCRIPTIVE_MIN = 0.0      # rank-1 along_loo for descriptive labels: at least not against what they agree on
NATURAL_RANK1_FIT = 0.3    # natural rank 1 must sit near his medians
QUESTION_RANK1_Z = 0.3     # questioning rank 1 must rise at least this far above his usual ending (z)
LONG_PROMPT_S, LONG_PROMPT_BONUS = 5.0, 0.3
NATURAL_KEYS = ["f0_median_st", "f0_range_st", "articulation_rate_syl_s", "pause_ratio", "effort", "final_rise_st"]
# Natural-unit wording for the signature features (deltas vs his medians).
FEATURE_WORDS = {
    "f0_median_st": lambda d: f"pitch {d:+.1f} st",
    "f0_range_st": lambda d: f"pitch range {d:+.1f} st",
    "final_rise_st": lambda d: f"ending {d:+.1f} st",
    "articulation_rate_syl_s": lambda d: f"{d:+.1f} syl/s",
    "pause_ratio": lambda d: f"pause share {100 * d:+.0f} pts",
    "alpha_ratio_db": lambda d: f"spectral tilt {d:+.1f} dB",
    "energy_slope_db": lambda d: f"energy {'holds up' if d > 0 else 'falls'} {abs(d):.1f} dB "
                                 f"{'more' if d < 0 else 'better'} toward the end",
}
NEVER = r"(?!x)x"  # a regex that matches nothing (an empty rule list or token list in the content files)


@lru_cache(maxsize=None)
def content() -> SimpleNamespace:
    """The character's label content (<lab>/delivery/content/labels.json, plus lexicon.json's g2p_risky), checked.

    labels: the label set in build order ('natural' is required: the identity anchor, scored as closeness);
    curated {label: {clip: (weight, rationale)}}: my reading of the line in its scene (1.0 = the words carry the
    emotion outright, 0.6-0.9 strong but shared, <= 0.5 a plausible secondary reading); direction_rules
    [(regex on a TLK note, {label: weight})] and performance_note (which notes are performance notes; staging notes
    such as 'remote N' or 'Fade in' carry no delivery information); tag_overrides {(clip, tag): weight} for delivery
    tags of uncertain origin; context_only {(label, clip): why} (right for a label only in one context: never near
    the top); spoken_mismatch {clip: why} (the TLK text differs from what he says, so the clip cannot be a prompt:
    a prompt transcript that does not match its audio misaligns GPT-SoVITS's and VoxCPM2's continuation);
    signatures: the hand-written prior prosody signature per label (weights on his own z-scored features,
    'effort' = flatter spectral tilt; it only feeds the candidate score and the prior-contradiction caution, while
    what a label measurably does in his clips is computed in target_from()); instructions / compact: VoxCPM2
    control text variants A (identity clause + delivery + one guard clause, Vox sidecar style) and B (compact,
    positive-only); craft: the hand-written directing note per label (no rank ids, no measured numbers)."""
    raw = {k: v for k, v in labcfg.content("labels").items() if not k.startswith("_")}

    def section(name: str) -> dict:
        return {k: v for k, v in (raw.get(name) or {}).items() if not k.startswith("_")}

    labels = [str(label) for label in raw["labels"]]
    if "natural" not in labels or len(set(labels)) != len(labels):
        raise SystemExit(f"{labcfg.CONTENT / 'labels.json'}: 'labels' must be unique and include 'natural' "
                         f"(the identity anchor): {labels}")
    instructions, compact, craft = section("voxcpm_instructions"), section("voxcpm_compact"), section("craft")
    missing = [f"{name}.{label}" for name, table in (("voxcpm_instructions", instructions),
                                                     ("voxcpm_compact", compact), ("craft", craft))
               for label in labels if not str(table.get(label) or "").strip()]
    if missing:
        raise SystemExit(f"{labcfg.CONTENT / 'labels.json'}: missing text for {missing}")
    lexicon = labcfg.content("lexicon")
    return SimpleNamespace(
        labels=labels,
        required=[str(label) for label in raw.get("required", labels)],
        curated={label: {clip: (float(entry[0]), str(entry[1])) for clip, entry in entries.items()}
                 for label, entries in section("curated").items()},
        direction_rules=[(str(pattern), {label: float(w) for label, w in weights.items()})
                         for pattern, weights in raw.get("direction_rules") or []],
        performance_note=re.compile(raw.get("performance_note") or NEVER, re.I),
        tag_overrides={(clip, tag): float(w) for clip, tags in section("tag_overrides").items()
                       for tag, w in tags.items()},
        context_only={(label, clip): str(why) for label, clips in section("context_only").items()
                      for clip, why in clips.items()},
        spoken_mismatch={clip: str(why) for clip, why in section("spoken_mismatch").items()},
        signatures={label: {key: float(w) for key, w in weights.items()}
                    for label, weights in section("prior_signatures").items()},
        instructions=instructions, compact=compact, craft=craft,
        g2p_risky=re.compile(lexicon.get("g2p_risky") or NEVER),
    )


def cfg_source() -> str:
    """Where the render step's default cfg_value comes from, for the variants' usage note."""
    return f" (Vox's {labcfg.NAME} profile)" if labcfg.CHARACTER.get("vox_profile_id") else ""


def write_json(path: Path, document) -> None:
    """Atomic write: other lab tasks read these files while they run."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


# --------------------------------------------------------------------------------------------- text overlap
def word_list(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.casefold().replace("-", " ")).split()


def common_run(a: list[str], b: list[str]) -> tuple[int, int]:
    """Length of the longest common run of words and its end index in a."""
    best, end, table = 0, 0, [0] * (len(b) + 1)
    for i, x in enumerate(a, 1):
        previous = 0
        for j, y in enumerate(b, 1):
            current = table[j]
            table[j] = previous + 1 if x == y else 0
            if table[j] > best:
                best, end = table[j], i
            previous = current
    return best, end


def longest_common_run(a: list[str], b: list[str]) -> int:
    return common_run(a, b)[0]


def test_overlap(text: str, tests: list[tuple[str, str]]) -> dict | None:
    """The held-out leak rule shared by the map, the exclusion list and validate.py: character ratio >= 0.6 or a
    run of >= 5 words in common with a held-out line. Returns the first hit."""
    words = word_list(text)
    for test_id, test_text in tests:
        other = word_list(test_text)
        ratio = SequenceMatcher(None, " ".join(words), " ".join(other)).ratio()
        run, end = common_run(words, other)
        if ratio >= 0.6 or run >= 5:
            return {"test_id": test_id, "ratio": round(ratio, 2), "run": run,
                    "phrase": " ".join(words[end - run:end])}
    return None


def leaks_test(text: str, test_texts: list[str]) -> str | None:
    hit = test_overlap(text, [(str(i), t) for i, t in enumerate(test_texts)])
    return test_texts[int(hit["test_id"])] if hit else None


# --------------------------------------------------------------------------------------------- evidence
def direction_scores(row: dict) -> tuple[dict[str, float], list[str]]:
    c = content()
    scores: dict[str, float] = {}
    notes = [d for d in row.get("directions", []) if c.performance_note.search(d)]
    for direction in row.get("directions", []):
        for pattern, labels in c.direction_rules:
            if re.search(pattern, direction.strip(), re.I):
                for label, weight in labels.items():
                    scores[label] = max(scores.get(label, 0.0), weight)
    for tag in row.get("delivery_tags", []):
        if tag in c.labels:
            scores[tag] = max(scores.get(tag, 0.0), c.tag_overrides.get((row["id"], tag), 1.0))
    return scores, notes


def derived_z(row: dict) -> dict[str, float]:
    z = dict(row.get("z", {}))
    if "alpha_ratio_db" in z and "hammarberg_db" in z:
        z["effort"] = (z["alpha_ratio_db"] - z["hammarberg_db"]) / 2.0
    if "low_f0_ratio" in z:
        z["creak"] = max(-1.0, min(3.0, z["low_f0_ratio"]))
    return z


def prosody_fit(label: str, z: dict[str, float]) -> float:
    if label == "natural":
        values = [min(abs(z[k]), 3.0) for k in NATURAL_KEYS if k in z]
        return math.tanh(1.0 - (sum(values) / max(len(values), 1)))
    signature = content().signatures.get(label, {})  # no prior for the label: fit 0 (neutral)
    total = sum(weight * max(-3.0, min(3.0, z.get(key, 0.0))) for key, weight in signature.items())
    return math.tanh(total / 1.5)


def quality(row: dict) -> tuple[float, list[str]]:
    score, notes = 0.0, []
    wer = row.get("wer") or 0.0
    if wer == 0.0:
        score += 0.3
    elif wer > 0.10:
        score -= 0.3
        notes.append(f"transcript/ASR mismatch {wer:.2f}")
    duration = row["duration"]  # longer prompts condition GPT-SoVITS/VoxCPM style more stably; >9.5 s nears the 10 s cap
    if 5.0 <= duration <= 8.0:
        score += 0.35
    elif 4.0 <= duration < 5.0:
        score += 0.2
    elif 8.0 < duration <= 9.5:
        score += 0.1
    elif duration > 9.5:
        score -= 0.3
    flag = row.get("speaker_flag")
    zr, zc = row.get("speaker_z_resemblyzer"), row.get("speaker_z_campplus")
    if flag == "suspect":
        score -= 0.6
        notes.append("speaker screen: suspect (voice-quality shift)")
    elif zr is not None and zc is not None and min(zr, zc) >= -1.5:
        score += 0.2
    if content().g2p_risky.search(row["text"]):  # the character's code-names/digits (content/lexicon.json)
        score -= 0.4
        notes.append("text has code-names/digits (respell for G2P)")
    if (row.get("tail_silence_s") or 0) > 0.35 or (row.get("lead_silence_s") or 0) > 0.3:
        score -= 0.1
    z = row.get("z", {})
    if max(abs(z.get("f0_range_st", 0.0)), abs(z.get("f0_std_st", 0.0)), abs(z.get("f0_median_st", 0.0))) > 3.5:
        score -= 0.4
        notes.append("atypical pitch contour for him (transfers idiosyncratically)")
    mismatch = content().spoken_mismatch
    if row["id"] in mismatch:  # a caution (prompt use), not a score penalty: its prosody is still his
        notes.append("spoken text differs: " + mismatch[row["id"]])
    if (row.get("snr_db") or 99) < 25:
        score -= 0.3
    if not row.get("restored_exists"):
        score -= 0.2
    return score, notes


def describe(label: str, row: dict, profile: dict) -> str:
    """Measured evidence in natural units relative to his own medians, most label-relevant first."""
    stats = profile["stats"]
    z = derived_z(row)
    parts: list[tuple[float, str]] = []

    def add(key: str, text: str, relevance: float = 1.0) -> None:
        if key in z:
            parts.append((abs(z[key]) * relevance, text))

    if "f0_median_st" in row:
        add("f0_median_st", f"pitch {row['f0_median_st']:+.1f} st vs his centre")
    if "f0_range_st" in row:
        add("f0_range_st", f"pitch range {row['f0_range_st']:.1f} st (his median {stats['f0_range_st']['median']:.1f})")
    if "articulation_rate_syl_s" in row:
        add("articulation_rate_syl_s", f"{row['articulation_rate_syl_s']:.1f} syl/s (his {stats['articulation_rate_syl_s']['median']:.1f})")
    if "pause_ratio" in row:
        add("pause_ratio", f"pauses {100 * row['pause_ratio']:.0f}% of the line (his {100 * stats['pause_ratio']['median']:.0f}%)")
    if "final_rise_st" in row:
        add("final_rise_st", f"ending {row['final_rise_st']:+.1f} st", 1.3 if label == "questioning" else 1.0)
    if "effort" in z:
        parts.append((abs(z["effort"]), "harder/brighter phonation" if z["effort"] > 0 else "softer, darker phonation"))
    if "creak" in z and z["creak"] > 1.0:
        parts.append((z["creak"], "creaky/very low phrase endings"))
    if "energy_slope_db" in row:
        add("energy_slope_db", "energy holds or builds to the end" if z.get("energy_slope_db", 0) > 0
            else "energy falls away", 0.8)
    if label == "natural":
        parts.sort(key=lambda item: item[0])
        near = [text for score, text in parts if score < 0.8][:3]
        return ("close to his medians: " + "; ".join(near)) if near else "; ".join(t for _, t in parts[:2])
    parts.sort(key=lambda item: -item[0])
    return "; ".join(text for _, text in parts[:3])


# --------------------------------------------------------------------------------------------- targets
def zvec(row: dict, profile: dict, keys: list[str]) -> list[float | None]:
    stats = profile["stats"]
    return [(row[k] - stats[k]["median"]) / stats[k]["scale"] if row.get(k) is not None and k in stats else None
            for k in keys]


def zclip(row: dict, profile: dict, keys: list[str]) -> list[float]:
    return [max(-3.0, min(3.0, v)) if v is not None else 0.0 for v in zvec(row, profile, keys)]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def ends_with_question(text: str) -> bool:
    """Its last sentence is a question: the ending and range then follow the question form, not the delivery."""
    return text.strip().endswith("?")


def yes_no_question(text: str) -> bool:
    """The last sentence is a question that is not a wh-question (yes/no or tag questions)."""
    last = re.split(r"(?<=[.!?])\s+", text.strip())[-1]
    return last.endswith("?") and not re.match(r"(?i)\W*(what|why|how|where|who|whom|whose|when|which)\b", last)


def measurement_caution(entry: dict) -> bool:
    """Cautions that make a clip's prosody untrustworthy as a measurement of the label."""
    return any(c.startswith(("speaker screen", "atypical")) for c in entry["cautions"])


def target_from(rows: list[dict], profile: dict) -> tuple[dict[str, float], dict[str, dict]]:
    """Median z per TARGET_KEY and the signature (|median| >= SIGNATURE_MIN_Z, >= SIGNATURE_AGREE share agree)."""
    target, signature = {}, {}
    for key in TARGET_KEYS:
        values = [v for v in (zvec(r, profile, [key])[0] for r in rows) if v is not None]
        if len(values) < POOL_MIN:
            continue
        mid = median([max(-3.0, min(3.0, v)) for v in values])
        target[key] = round(mid, 3)
        agree = sum(1 for v in values if v != 0 and (v > 0) == (mid > 0))
        if abs(mid) >= SIGNATURE_MIN_Z and agree >= SIGNATURE_AGREE * len(values):
            signature[key] = {"z": round(mid, 3), "agree": agree, "n": len(values)}
    return target, signature


def along(row: dict, target: dict[str, float], profile: dict) -> float | None:
    keys = list(target)
    values = zvec(row, profile, keys)
    pairs = [(max(-3.0, min(3.0, v)), target[k]) for v, k in zip(values, keys) if v is not None]
    norm = sum(t * t for _, t in pairs)
    return sum(v * t for v, t in pairs) / norm if pairs and norm > 1e-9 else None


def along_loo(row: dict, pool: list[dict], keys: list[str], profile: dict) -> float | None:
    """Projection of the row onto the median z of the pool WITHOUT the row, over the signature keys."""
    others = [m for m in pool if m["id"] != row["id"]]
    if not keys or len(others) < 2:
        return None
    target = {}
    for key in keys:
        values = [max(-3.0, min(3.0, v)) for v in (zvec(m, profile, [key])[0] for m in others) if v is not None]
        if values:
            target[key] = median(values)
    return along(row, target, profile)


def displacement_p(pool: list[dict], null_rows: list[dict], profile: dict) -> tuple[float, float]:
    """|median z| of the pool over TARGET_KEYS and the share of random same-size sets of his statement clips that
    are displaced at least as far (permutation p, +1 smoothed)."""
    import numpy as np

    matrix = np.array([zclip(r, profile, TARGET_KEYS) for r in null_rows])
    stat = float(np.linalg.norm(np.median(np.array([zclip(r, profile, TARGET_KEYS) for r in pool]), axis=0)))
    rng = np.random.default_rng(PERM_SEED + len(pool))
    draws = np.array([np.linalg.norm(np.median(matrix[rng.choice(len(matrix), len(pool), replace=False)], axis=0))
                      for _ in range(PERM_DRAWS)])
    return stat, float((np.sum(draws >= stat) + 1) / (PERM_DRAWS + 1))


def question_target(profile_rows: list[dict], profile: dict) -> dict:
    """His yes/no questions' ending (final_rise_st, z) against random same-size sets of his statements."""
    import numpy as np

    questions = [v for v in (zvec(r, profile, ["final_rise_st"])[0] for r in profile_rows if yes_no_question(r["text"]))
                 if v is not None]
    statements = [v for v in (zvec(r, profile, ["final_rise_st"])[0] for r in profile_rows
                              if not ends_with_question(r["text"])) if v is not None]
    if len(questions) < POOL_MIN or len(statements) < 2 * len(questions):
        return {}
    mid = median(questions)
    rng = np.random.default_rng(PERM_SEED)
    pool = np.array(statements)
    draws = np.array([np.median(pool[rng.choice(len(pool), len(questions), replace=False)]) for _ in range(PERM_DRAWS)])
    p = float((np.sum(draws >= mid) + 1) / (PERM_DRAWS + 1))
    natural_q = [r["final_rise_st"] for r in profile_rows if yes_no_question(r["text"]) and r.get("final_rise_st") is not None]
    natural_s = [r["final_rise_st"] for r in profile_rows
                 if not ends_with_question(r["text"]) and r.get("final_rise_st") is not None]
    return {"z": round(mid, 3), "agree": sum(1 for v in questions if v > 0), "n": len(questions), "perm_p": p,
            "source": "his yes/no questions (profile clips) vs his statements",
            "questions_median_st": round(median(natural_q), 2), "statements_median_st": round(median(natural_s), 2)}


def short_caution(text: str) -> str:
    """One short reason for the notes (the full caution stays in delivery_candidates.json)."""
    for prefix, short in (("speaker screen", "speaker screen suspect: voice-quality shift"),
                          ("atypical", "atypical pitch contour for him"),
                          ("measured prosody runs against", "measured prosody runs against the label"),
                          ("spoken text differs", "its TLK text differs from what he says, so it cannot be a prompt")):
        if text.startswith(prefix):
            return short
    return text


def auc(own: list[float], others: list[float]) -> float | None:
    if not own or not others:
        return None
    wins = sum((1.0 if o > x else 0.5 if o == x else 0.0) for o in own for x in others)
    return round(wins / (len(own) * len(others)), 3)


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build {labcfg.NAME}'s delivery map (CPU).")
    parser.add_argument("--features", type=Path, default=OUT / "features.jsonl")
    parser.add_argument("--profile", type=Path, default=OUT / "voice_profile.json")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=OUT, help="where to write the outputs (default: the lab)")
    args = parser.parse_args()
    out_dir = args.out_dir
    c = content()
    LABELS, CURATED, CONTEXT_ONLY, LABEL_CRAFT = c.labels, c.curated, c.context_only, c.craft

    corpus = [json.loads(l) for l in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_ids = {r["id"] for r in corpus if r["split"] == "test"}
    tests = [(r["id"], r["text"]) for r in corpus if r["split"] == "test"]
    rows = [json.loads(l) for l in args.features.read_text(encoding="utf-8").splitlines() if l.strip()]
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    confirmed_other, uncertain_other, near_test = set(), set(), {}
    if EXCLUSIONS.is_file():
        exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
        confirmed_other = set(exclusions.get("confirmed_ids", []))
        uncertain_other = {item["id"] for item in exclusions.get("uncertain", [])}
        near_test = dict(exclusions.get("near_duplicate_of_test", {}))
    else:
        print(f"note: {EXCLUSIONS.name} not found; run exclusions.py (speaker screen flags still apply)", flush=True)

    eligible, excluded = {}, {}
    for row in rows:
        reason = None
        if row["id"] in test_ids or row["split"] not in ("train", "val"):
            reason = "test split"
        elif row["kind"] != "speech":
            reason = "nonverbal"
        elif row["text_source"] not in EXACT_SOURCES:
            reason = "no exact transcript"
        elif (row.get("wer") or 0.0) > MAX_WER:
            reason = f"wer {row['wer']}"
        elif not (MIN_S <= row["duration"] <= MAX_S):
            reason = f"duration {row['duration']} s outside {MIN_S}-{MAX_S}"
        elif row.get("speaker_flag") == "other_speaker":
            reason = "speaker screen: other speaker"
        elif row["id"] in confirmed_other:
            reason = "exclude_other_speakers.json: confirmed other character"
        elif row["id"] in uncertain_other:
            reason = "exclude_other_speakers.json: uncertain (needs a listen)"
        elif row["id"] in near_test:
            reason = f"near-duplicate of a held-out line: {near_test[row['id']]}"
        elif "error" in row:
            reason = row["error"]
        else:
            hit = test_overlap(row["text"], tests)
            if hit:
                reason = f"text overlaps held-out {hit['test_id']}: '{hit['phrase']}'"
        if reason:
            excluded[row["id"]] = reason
        else:
            eligible[row["id"]] = row

    # 1. candidates
    scored: dict[str, list[dict]] = {}
    for label in LABELS:
        entries = []
        for clip_id, row in eligible.items():
            curated = CURATED.get(label, {}).get(clip_id)
            directions, notes = direction_scores(row)
            evidence = (curated[0] if curated else 0.0), directions.get(label, 0.0)
            if evidence == (0.0, 0.0):
                continue  # every reference needs text or TLK grounding
            z = derived_z(row)
            fit = prosody_fit(label, z)
            q, q_notes = quality(row)
            if any(n.startswith("atypical pitch") for n in q_notes):
                fit = min(fit, 0.0)  # the outlier pitch stats that would earn the fit are the untrusted part
            cautions = [n for n in q_notes if n.startswith(("speaker screen", "atypical", "spoken text differs"))]
            if fit <= CONTRADICTION_FIT and evidence[1] < STRONG_DIRECTION:
                cautions.append(f"measured prosody runs against the '{label}' prior (fit {fit:+.2f})")
            if (label, clip_id) in CONTEXT_ONLY:
                cautions.append(CONTEXT_ONLY[(label, clip_id)])
            entries.append({
                "id": clip_id, "score": round(3.0 * evidence[0] + 2.0 * evidence[1] + 0.8 * fit + q, 3),
                "curated": evidence[0], "direction": evidence[1], "prosody_fit": round(fit, 3), "quality": round(q, 3),
                "duration": row["duration"], "rationale": curated[1] if curated else "", "notes": notes,
                "quality_notes": q_notes, "cautions": cautions,
            })
        scored[label] = sorted(entries, key=lambda e: (-e["score"], e["id"]))

    # 2. reference sets: global greedy by score over CAUTION-FREE entries; a clip in at most two labels.
    members: dict[str, list[dict]] = {label: [] for label in LABELS}
    usage: dict[str, int] = {}
    for _, label, entry in sorted(((e["score"], label, e) for label in LABELS for e in scored[label] if not e["cautions"]),
                                  key=lambda item: (-item[0], LABELS.index(item[1]), item[2]["id"])):
        if len(members[label]) < args.top and usage.get(entry["id"], 0) < 2:
            members[label].append(entry)
            usage[entry["id"]] = usage.get(entry["id"], 0) + 1

    # 3. targets from a larger evidence pool (see the docstring); the scorers trust validated labels only.
    profile_rows = [r for r in rows if r["split"] in ("train", "val") and r["kind"] == "speech"
                    and r["text_source"] in EXACT_SOURCES and (r["wer"] or 0.0) <= 0.15 and r["duration"] >= 1.5
                    and r.get("speaker_flag") in (None, "ok", "suspect") and "error" not in r
                    and r["id"] not in confirmed_other]
    statements = [r for r in profile_rows if not ends_with_question(r["text"])]
    info: dict[str, dict] = {}
    for label in LABELS:
        pool = [eligible[e["id"]] for e in scored[label]
                if max(e["curated"], e["direction"]) >= POOL_MIN_EVIDENCE and not measurement_caution(e)
                and (label == "questioning" or not ends_with_question(eligible[e["id"]]["text"]))][:POOL_MAX]
        entry_info: dict = {"pool": pool, "signature": {}, "target_z": {}, "perm_p": None, "displacement": None}
        if label == "natural":
            kind = "closeness"
        elif label == "questioning":
            question = question_target(profile_rows, profile)
            if question:
                entry_info["signature"] = {"final_rise_st": {k: question[k] for k in ("z", "agree", "n", "source")}}
                entry_info["target_z"] = {"final_rise_st": question["z"]}
                entry_info["perm_p"] = round(question["perm_p"], 4)
                entry_info["question"] = {k: question[k] for k in ("questions_median_st", "statements_median_st",
                                                                   "agree", "n")}
            kind = "question_ending" if question and question["perm_p"] <= VALIDATED_P else "none"
        elif len(pool) >= POOL_MIN:
            target, signature = target_from(pool, profile)
            displacement, p_value = displacement_p(pool, statements, profile)
            entry_info.update(signature=signature, target_z=target, perm_p=round(p_value, 4),
                              displacement=round(displacement, 3))
            kind = "direction" if signature and p_value <= VALIDATED_P else ("descriptive" if signature else "none")
        else:
            kind = "none"
        entry_info["kind"] = kind
        keys = list(entry_info["signature"])
        for entry in scored[label]:  # every candidate, so kept-out clips show it in delivery_candidates.json too
            row = eligible[entry["id"]]
            entry["along_loo"] = None
            if label == "questioning" and keys:
                value = zvec(row, profile, ["final_rise_st"])[0]
                if value is not None:
                    entry["along_loo"] = round(value / entry_info["signature"]["final_rise_st"]["z"], 3)
            elif keys and label != "natural":
                value = along_loo(row, pool, keys, profile)
                entry["along_loo"] = round(value, 3) if value is not None else None
        info[label] = entry_info

    def qualifies(label: str, entry: dict) -> bool:
        if entry["cautions"]:
            return False
        if label == "natural":
            return entry["prosody_fit"] >= NATURAL_RANK1_FIT
        if label == "questioning":
            row = eligible[entry["id"]]
            rise = zvec(row, profile, ["final_rise_st"])[0]
            return yes_no_question(row["text"]) and rise is not None and rise >= QUESTION_RANK1_Z
        kind, value = info[label]["kind"], entry.get("along_loo")
        if kind == "direction":
            return value is not None and value >= REP_MIN
        if kind == "descriptive":  # not validated, but rank 1 must not run against what the label's clips agree on
            return value is None or value >= DESCRIPTIVE_MIN
        return True  # no signature: no measured direction to overrule the evidence order with

    def rank1_score(entry: dict) -> float:
        return entry["score"] + (LONG_PROMPT_BONUS if entry["duration"] >= LONG_PROMPT_S else 0.0)

    # 4. rank 1: unique across labels, global greedy over qualifying (label, clip) pairs
    primary_of: dict[str, str] = {}
    taken: set[str] = set()
    pairs = sorted(((rank1_score(e), label, e) for label in LABELS for e in members[label] if qualifies(label, e)),
                   key=lambda item: (-item[0], LABELS.index(item[1]), item[2]["id"]))
    for _, label, entry in pairs:
        if label in primary_of or entry["id"] in taken:
            continue
        primary_of[label] = entry["id"]
        taken.add(entry["id"])
    for label in LABELS:  # no qualifying clip: fall back to the best caution-free one (reported)
        if label not in primary_of:
            for entry in sorted(members[label], key=lambda e: (-rank1_score(e), e["id"])):
                if entry["id"] not in taken:
                    primary_of[label] = entry["id"]
                    taken.add(entry["id"])
                    info[label]["rank1_fallback"] = True
                    break

    # 5. order: rank 1, then by score (the first of those is the intent alternative when it outscores rank 1)
    chosen: dict[str, list[dict]] = {}
    for label in LABELS:
        clean = sorted(members[label], key=lambda e: (-e["score"], e["id"]))
        primary = next((e for e in clean if e["id"] == primary_of.get(label)), None)
        order = [primary] if primary else []
        for entry in clean:
            if entry is not primary:
                if primary and not order[1:] and entry["score"] > primary["score"]:
                    entry["intent_alternative"] = "rank 1 elsewhere" if entry["id"] in taken else "off target"
                order.append(entry)
        chosen[label] = order

    # Per-label statistics from the pool (natural units for the notes and the bible, z for the scorers).
    keys = ["f0_median_st", "f0_range_st", "f0_std_st", "final_rise_st", "articulation_rate_syl_s", "speech_rate_syl_s",
            "pause_ratio", "pause_mean_s", "alpha_ratio_db", "hammarberg_db", "low_f0_ratio", "energy_slope_db",
            "level_dbfs", "dynamic_db"]
    label_stats = {}
    for label in LABELS:
        pool = info[label]["pool"] or [eligible[e["id"]] for e in chosen[label]]
        stat = {}
        for key in keys:
            values = [m[key] for m in pool if m.get(key) is not None]
            if values:
                mid = median(values)
                his = profile["stats"].get(key, {}).get("median")
                stat[key] = {"median": round(mid, 3), "his_median": his, "delta": round(mid - (his or 0.0), 3)}
        signature, kind = info[label]["signature"], info[label]["kind"]
        sig_keys = list(signature)
        auc_value = None
        if sig_keys and label not in ("natural", "questioning"):
            own = [a for a in (along_loo(r, pool, sig_keys, profile) for r in pool) if a is not None]
            sig_target = {k: v["z"] for k, v in signature.items()}
            pool_ids = {r["id"] for r in pool}
            others = [a for a in (along(r, sig_target, profile) for r in statements if r["id"] not in pool_ids)
                      if a is not None]
            auc_value = auc(own, others)
        # His own clips of the label span these values: score_delivery.py never flags a take inside them.
        bounds = {}
        for key in ("f0_range_st", "articulation_rate_syl_s"):
            values = [m[key] for m in pool + [eligible[e["id"]] for e in chosen[label]] if m.get(key) is not None]
            if values:
                bounds[key] = [round(min(values), 3), round(max(values), 3)]
        label_stats[label] = {
            "ids": [m["id"] for m in pool],
            "reference_ids": [e["id"] for e in chosen[label]],
            "features": stat,
            "target_z": info[label]["target_z"],
            "signature": signature,
            "displacement": info[label]["displacement"],
            "perm_p": info[label]["perm_p"],
            "kind": kind,
            "validated": kind in ("direction", "question_ending", "closeness"),
            "basis": "his_medians" if kind in ("question_ending", "closeness") else "paired_baseline",
            "lines": "yes_no_questions" if label == "questioning" else "all",
            "rank1": chosen[label][0]["id"] if chosen[label] else None,
            "rank1_fallback": bool(info[label].get("rank1_fallback")),
            "along_loo": {e["id"]: e.get("along_loo") for e in chosen[label]},
            "auc_optimistic": auc_value,
            "flag_bounds": bounds,
        }
        if "question" in info[label]:
            label_stats[label]["question"] = info[label]["question"]

    # Clips the caution rule kept out of the map that would otherwise have been references.
    kept_out: dict[str, list[dict]] = {}
    for label in LABELS:
        floor = min((e["score"] for e in members[label]), default=-math.inf) if len(members[label]) >= args.top \
            else -math.inf
        kept_out[label] = [e for e in scored[label][:12] if e["cautions"] and e["score"] >= floor]

    def snippet(text: str, limit: int = 48) -> str:
        return text if len(text) <= limit else text[:limit].rstrip(" ,.;-") + "..."

    def ref_phrase(entry: dict) -> str:
        row = eligible[entry["id"]]
        tlk = f", TLK '{entry['notes'][0]}'" if entry["notes"] else ""
        return f"{entry['id']} '{snippet(row['text'])}' ({row['duration']:.1f} s{tlk})"

    def label_notes(label: str) -> str:
        kind, p_value, signature = info[label]["kind"], info[label]["perm_p"], info[label]["signature"]
        features = label_stats[label]["features"]
        sig_text = ", ".join(f"{FEATURE_WORDS[k](features[k]['delta'])} ({v['agree']} of {v['n']})"
                             for k, v in signature.items() if k in features and k in FEATURE_WORDS)
        if kind == "closeness":
            measured = "Scored as closeness to his own medians."
        elif kind == "question_ending":
            q = info[label]["question"]
            measured = (f"Validated (perm p {p_value:.4f}): his yes/no questions end {q['questions_median_st']:+.1f} st, "
                        f"his statements {q['statements_median_st']:+.1f} st ({q['agree']} of {q['n']} questions rise); "
                        "scored on question lines only, by the ending.")
        elif kind == "direction":
            measured = (f"Validated (perm p {p_value:.3f}): his {label} clips agree on {sig_text}; scored as direction "
                        "on these features.")
        elif kind == "descriptive":
            measured = (f"Not validated (perm p {p_value:.2f}: no more displaced than random sets of his lines): his "
                        f"{label} clips agree on {sig_text}. Described, never scored; judge by ear.")
        elif p_value is None:
            measured = "Too few evidence-backed clips to measure; judge by ear."
        else:
            measured = (f"No measurable signature (perm p {p_value:.2f}; no feature on which 80% of his {label} clips "
                        "agree): the delivery lives in the words and the voice. Judge by ear.")
        first = chosen[label][0]
        rank1 = f" Rank 1: {ref_phrase(first)}"
        if first.get("along_loo") is not None:
            rank1 += f", along {first['along_loo']:+.2f}"
        rank1 += " (fallback: no clip met the rank-1 rule)." if info[label].get("rank1_fallback") else "."
        rank2 = ""
        if len(chosen[label]) > 1:
            second = chosen[label][1]
            alternative = second.get("intent_alternative")
            along_text = f" (along {second['along_loo']:+.2f})" if second.get("along_loo") is not None else ""
            if alternative == "off target":
                if kind == "direction":
                    reason = "follows the validated target less"
                elif kind == "question_ending":
                    reason = "is not a rising yes/no question (the questioning rank-1 rule)"
                else:
                    reason = f"runs against what his other {label} clips agree on"
                reason += along_text
                rank2 = (f" Rank 2: {ref_phrase(second)}, the strongest text/TLK evidence, but it {reason}; A/B it "
                         "against rank 1 with --refs-per-line 2.")
            elif alternative == "rank 1 elsewhere":
                other = next(name for name in LABELS if primary_of.get(name) == second["id"])
                rank2 = (f" Rank 2: {ref_phrase(second)}, the strongest text/TLK evidence, but rank 1 of {other}; A/B "
                         "it against rank 1 with --refs-per-line 2.")
            else:
                rank2 = f" Rank 2: {ref_phrase(second)}."
        kept = kept_out[label]
        kept_text = (" Kept out of the map (delivery_candidates.json): "
                     + "; ".join(f"{e['id']} ({short_caution(e['cautions'][0])})" for e in kept) + ".") if kept else ""
        return f"{measured}{rank1}{rank2}{kept_text} {LABEL_CRAFT[label]}"

    picks_path = out_dir / PICKS_NAME
    picks = json.loads(picks_path.read_text(encoding="utf-8")).get("picks", {}) if picks_path.is_file() else {}
    for label, pick in picks.items():
        text = pick.get("instruction")
        if label not in LABELS or not isinstance(text, str) or not text.strip() or re.search(r"[()]", text):
            raise SystemExit(f"{picks_path}: bad pick for '{label}' (unknown label, or instruction empty/with "
                             f"parentheses): {pick}")

    rank1_rules = {
        "closeness": "rank 1: best evidence among clips close to his medians",
        "question_ending": "rank 1: best evidence among his rising yes/no questions",
        "direction": f"rank 1: best evidence among clips that follow the label's validated target (along >= {REP_MIN})",
        "descriptive": "rank 1: best evidence among clips that do not run against what the label's clips agree on "
                       "(that signature is not validated)",
        "none": "rank 1: best evidence (the label has no measurable signature, so the evidence order stands)",
    }
    deliveries = {}
    for label in LABELS:
        references = []
        kind = info[label]["kind"]
        for rank, entry in enumerate(chosen[label]):
            row = eligible[entry["id"]]
            why_parts = []
            if entry["rationale"]:
                why_parts.append(entry["rationale"])
            if entry["notes"]:
                why_parts.append("TLK: " + "; ".join(f"'{n}'" for n in entry["notes"]))
            if row.get("delivery_tags"):
                why_parts.append("tag: " + ", ".join(row["delivery_tags"]))
            why_parts.append("measured: " + describe(label, row, profile))
            if entry.get("along_loo") is not None:
                target_name = {"direction": "validated target", "question_ending": "validated question ending"}.get(
                    kind, "signature (not validated)")
                why_parts.append(f"along the label's {target_name} {entry['along_loo']:+.2f} (1 = like his other "
                                 "clips of it)")
            if rank == 0:
                why_parts.append("rank 1: no clip qualifies; best clip" if info[label].get("rank1_fallback")
                                 else rank1_rules[kind])
            elif entry.get("intent_alternative") == "off target":
                why_parts.append("intent alternative: strongest text/TLK evidence, but it does not meet the rank-1 rule; "
                                 "A/B it against rank 1 by ear")
            elif entry.get("intent_alternative"):
                why_parts.append("intent alternative: strongest text/TLK evidence for this label, but it is rank 1 of "
                                 "another label; A/B it against rank 1 by ear")
            source = "exact TLK text" if row["text_source"] == "tlk-exact" else "DLG transcript"
            why_parts.append(f"{source}, ASR WER {row['wer']:.2f}; {row['split']}")
            if entry["quality_notes"]:
                why_parts.append("note: " + "; ".join(dict.fromkeys(entry["quality_notes"])))
            references.append({
                "id": row["id"], "audio": row["audio"], "restored_audio": row["restored_audio"],
                "text": row["text"], "duration": row["duration"], "why": " | ".join(why_parts),
            })
        instruction, notes = c.instructions[label], label_notes(label)
        pick = picks.get(label)
        if pick:
            instruction = pick["instruction"].strip()
            notes = f"{notes} [probe {pick.get('run')}: {pick.get('note', '')}]"
        deliveries[label] = {"references": references, "voxcpm_instruction": instruction, "notes": notes}

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "delivery_map.json", {"deliveries": deliveries})
    write_json(out_dir / "voxcpm_instruction_variants.json", {
        "usage": "VoxCPM2 controlled cloning (Vox's production convention, the voxcpm render step's idref_instr mode): "
                 "text = '(' + instruction + ')' + line, reference_wav_path = ONE fixed identity clip (natural rank 1), "
                 "no prompt_wav/prompt_text. With a LoRA adapter Vox and the render step (--identity-clause auto) "
                 "prefix 'Keep the trained voice stable. ' and capitalise the instruction's first letter. The render "
                 f"step's default cfg_value is 1.5{cfg_source()}; probe_voxcpm.py uses the same. A = Vox-style "
                 "(identity words + delivery + one guard clause), B = compact positive-only descriptors. "
                 "delivery_map.json carries A until a probe pick is applied (instruction_picks.json).",
        "variants": {"A": c.instructions, "B": c.compact},
    })
    write_json(out_dir / "delivery_candidates.json", {
        "scoring": "3.0*curated + 2.0*direction + 0.8*prosody_fit + quality",
        "ordering": "members = up to six caution-free clips per label (global greedy by score, a clip in at most two "
                    "labels); rank 1 = best member (+0.3 if >= 5 s), unique across labels; validated labels need "
                    f"along_loo >= {REP_MIN}, descriptive ones >= {DESCRIPTIVE_MIN}; natural fit >= {NATURAL_RANK1_FIT}; "
                    "questioning: a rising yes/no question; then the rest by score. Caution clips never enter the map.",
        "eligible": len(eligible), "excluded": excluded,
        "candidates": {label: [{**e, "member": any(m["id"] == e["id"] for m in members[label])}
                               for e in scored[label][:12]] for label in LABELS},
        "kept_out": {label: [{"id": e["id"], "score": e["score"], "cautions": e["cautions"]} for e in kept_out[label]]
                     for label in LABELS},
    })
    write_json(out_dir / "label_stats.json", {
        "_about": "Per label: 'ids' = the target pool (evidence-backed candidates, see build_delivery_map.py), "
                  "'features' = pool medians in natural units vs his medians, 'target_z' = pool median z over the "
                  "scorer features, 'signature' = the features >= 80% of the pool agree on, 'perm_p' = share of random "
                  "same-size sets of his statements displaced as far, 'kind' = direction (validated) | descriptive | "
                  "none | question_ending | closeness, 'validated' = the scorers may use it, 'flag_bounds' = the range "
                  "his own clips of the label span (no drop flag inside it). 'auc_optimistic' ranks the pool against "
                  "his other statements on its own signature (optimistic: same clips chose it). Keys starting with "
                  "'_' are documentation; every reader skips them.",
        **label_stats})

    short = labcfg.short_id
    summary = {label: [(short(r["id"]), r["duration"]) for r in deliveries[label]["references"]] for label in LABELS}
    print(json.dumps({"eligible": len(eligible), "excluded": len(excluded),
                      "kinds": {label: (label_stats[label]["kind"], label_stats[label]["perm_p"]) for label in LABELS},
                      "references": summary,
                      "kept_out": {label: [short(e["id"]) for e in kept_out[label]] for label in LABELS if kept_out[label]},
                      "picks_applied": sorted(picks)}))


if __name__ == "__main__":
    main()
