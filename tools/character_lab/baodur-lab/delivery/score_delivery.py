"""Delivery scoring for a character's rendered takes (CPU; only --asr cuda touches the GPU; VOICE_LAB_CHARACTER,
default baodur: every path below is that character's lab).

For every take in a shared-format manifest ({"takes": [{take_id, engine, line_id, text, delivery, wav, ...}]};
relative wav names resolve against the manifest's folder):
  * prosody: analyze.clip_features() with his register centre, z-scored against voice_profile.json.
  * direction, ONLY for labels whose measured target is validated in label_stats.json (kind "direction": his
    clips of the label are displaced from his medians further than random sets of his lines, perm_p <= 0.05;
    for Bao-Dur warm and weary), on the signature features the label's clips agree on. Movement is measured from a
    paired neutral read of the same words when one exists (probe variant "none" averaged over seeds, or a
    natural take of the same text from this manifest or --baseline-manifest), else from his own medians over
    the 40 profile statements closest in syllable count (length-matched, because his long lines pause more and
    fall further). along = projection onto the signature (1 = as far as his clips of it moved, 2 = twice:
    overacting); direction = 1 - |1 - along|, clipped to [-1, 1]. Questioning (kind question_ending) is scored
    on question lines only, by the ending against his yes/no questions. Natural is closeness to his
    length-matched medians: tanh(1 - mean |z - base|). Every other label gets NO direction score: its target is
    no more coherent than chance, so its delivery is judged by ear.
  * identity: WavLM-SV (microsoft/wavlm-base-plus-sv, Vox's audition model) cosine to his centroid from
    train/val profile clips (never test). Calibrated on his clips (leave-one-out) and on the other-character clips
    in his files (exclude_other_speakers.json confirmed; Bao-Dur: 38 clips of HK-47, G0-T0, Atton).
    identity_units = (sim - his median) / (his median - his p05): 0 = his median, -1 = his p05. Gate = max(his p05,
    impostor p95, impostor max over clips >= 3 s); with no other-character clips, his p05. WavLM alone confuses a
    short Atton clip with Bao-Dur (0.937, above his p10), so the eval's three-encoder scorecard and the user's ears
    stay the identity authority; this gate only keeps take selection honest.
  * words: optional faster-whisper large-v3-turbo (local cache, temperature 0 = deterministic) WER and
    instruction-leak detection; --asr-prompt primes Whisper with the character config's asr_prompt (his names),
    off by default.
Objective (picks and reranks): direction (validated labels only) + identity_units (clipped to [-2, 0.5]) - WER
term (0 at WER <= 0.05, 1 at >= 0.15) - 1 if a gate fails (WER > 0.15, instruction spoken, identity < gate).

--pick-instructions (probe manifests): per label, every alternative (B, P, P2) is compared with A on paired
(line, seed) differences of the objective; it wins only when the 95% bootstrap CI of the mean difference
excludes 0 AND the mean gain is >= --margin. Otherwise A stays ("no detectable difference"). Writes
voxcpm_instruction_probe.json; nothing is applied. After listening (edit "winner"/"controlled_variant" there
to overrule by ear), apply with --apply-from: the exact instruction strings the probe rendered go into
instruction_picks.json and build_delivery_map.py rebuilds delivery_map.json (the previous map is kept once as
delivery_map.before_<run>.json), so every later rebuild keeps the picks.

Outputs (default <lab>/delivery/scores/<engine>_<run>/, or next to a probe manifest): delivery_scores.json
and .md (means with standard errors); reference_rerank.json when a delivery was rendered from more than one
reference clip (a proposal to audition). Per-take results are cached in .score_cache.jsonl as they are made, so
an interrupted run resumes where it stopped; one failing take is recorded and flagged, never fatal.

Run with the VoxCPM2 runtime python (read-only venv: no bytecode is written; numba's cache goes to
<lab>/delivery/.cache/numba):
  CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 <voxcpm2 .venv python> score_delivery.py --manifest <manifest.json>
    [--baseline-manifest <natural pairs manifest>] [--asr cpu|cuda|none] [--asr-prompt] [--pick-instructions]
    [--out-dir DIR] [--no-build-centroid]
  <voxcpm2 .venv python> score_delivery.py --apply-from <probe dir>/voxcpm_instruction_probe.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labcfg  # noqa: E402  (stdlib only)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(labcfg.NUMBA_CACHE))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from analyze import EXACT_SOURCES, measure, profile_member, warm_jit  # noqa: E402
from build_delivery_map import NATURAL_KEYS, TARGET_KEYS, derived_z, ends_with_question  # noqa: E402
import resources  # noqa: E402

SCORER_VERSION = 2
LAB = labcfg.LAB
DELIVERY = labcfg.DELIVERY
BUILD = Path(__file__).resolve().parent / "build_delivery_map.py"
ENGINE_CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "VoxStudio" / "engines" / "voxcpm2" / "cache"
REPORT_KEYS = sorted(set(TARGET_KEYS) | set(NATURAL_KEYS))
DEFAULT_MARGIN = 0.15   # an alternative must also gain at least this much on average (objective units)
LENGTH_NEIGHBOURS = 40  # his profile statements closest in syllable count form the length-matched baseline
ASR_GPU_GB = 3.0        # faster-whisper large-v3-turbo fp16 (CTranslate2: not capped by torch, so checked first)
IDENTITY_GPU_GB = 1.0   # WavLM-base-plus-sv on the GPU (capped through torch's allocator)
FAIL_PENALTY = 1.0


def write_json(path: Path, document) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ words
def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.casefold().replace("-", " ")).split()


def norm_text(text: str) -> str:
    return " ".join(words(text))


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


def instruction_leak(instruction: str, text: str, transcript: str) -> bool:
    """True when two consecutive instruction words that are not in the line appear in the transcript."""
    if not instruction:
        return False
    spoken, line = words(transcript), set(words(text))
    inst = [w for w in words(instruction) if w not in line and len(w) > 2]
    pairs = {(a, b) for a, b in zip(inst, inst[1:])}
    return any((a, b) in pairs for a, b in zip(spoken, spoken[1:]))


class Transcriber:
    """faster-whisper; prompt = the character's asr_prompt (names Whisper should spell his way) when --asr-prompt is
    given, else no prompt (the default, as every earlier score was made)."""

    def __init__(self, device: str, prompt: str | None = None) -> None:
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"
        self.prompt = prompt or None
        self.model = WhisperModel("large-v3-turbo", device=device, compute_type=compute,
                                  download_root=str(ENGINE_CACHE / "faster-whisper"), local_files_only=True,
                                  cpu_threads=max(1, min(8, (os.cpu_count() or 4) // 2)))

    def __call__(self, path: str) -> str:
        options = {"initial_prompt": self.prompt} if self.prompt else {}
        segments, _ = self.model.transcribe(path, language="en", beam_size=5, vad_filter=False, temperature=0.0,
                                            condition_on_previous_text=False, **options)
        return " ".join(s.text.strip() for s in segments).strip()


# ------------------------------------------------------------------------------------------------ identity
class Identity:
    """WavLM-SV x-vectors (first 12 s at 16 kHz, as in Vox's audition tool), on the CPU or (~0.5 GB) the GPU."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 4)))
        self.torch, self.device = torch, device
        self.extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv", cache_dir=str(ENGINE_CACHE),
                                                              local_files_only=True)
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv", cache_dir=str(ENGINE_CACHE),
                                                     local_files_only=True).eval().to(device)

    def embed(self, path: str) -> np.ndarray:
        import librosa
        import soundfile as sf

        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != 16000:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=16000, res_type="soxr_hq")
        inputs = self.extractor(mono[: 16000 * 12], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            vector = self.model(**inputs).embeddings[0].float()
        return self.torch.nn.functional.normalize(vector, dim=0).cpu().numpy().astype(np.float64)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def centroid_members(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["split"] in ("train", "val") and r["kind"] == "speech"
            and r["text_source"] in EXACT_SOURCES and r.get("speaker_flag") == "ok" and r["duration"] >= 2.0
            and (r["wer"] or 0) <= 0.15]


def identity_calibration(identity_factory, audio_choice: str, allow_build: bool) -> tuple[np.ndarray, dict]:
    """His WavLM centroid (train/val profile clips, never test) with leave-one-out calibration, plus the
    other-character clips in his files as impostors; cached per member set (building embeds ~185 + 38 clips)."""
    rows = load_jsonl(DELIVERY / "features.jsonl")
    members = centroid_members(rows)
    key = hashlib.sha256("|".join(sorted(r["id"] for r in members)).encode()).hexdigest()[:16]
    exclusions = json.loads((DELIVERY / "exclude_other_speakers.json").read_text(encoding="utf-8"))
    impostor_ids = sorted(exclusions["confirmed_ids"])
    character = {item["id"]: name for name, items in exclusions["confirmed"].items() for item in items}
    impostor_key = hashlib.sha256(("|".join(impostor_ids) + key).encode()).hexdigest()[:16]
    cache = DELIVERY / ".cache" / f"wavlm_centroid_{audio_choice}.json"
    data = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    if data and data.get("members_key") != key and not ("members_key" not in data
                                                           and data["calibration"].get("clips") == len(members)):
        data = {}  # stale: features.jsonl changed
    if data and "members_key" not in data:
        data["members_key"] = key  # written before keys existed, same rule over the same features: adopt it
    model = None
    if not data:
        if not allow_build:
            raise SystemExit(f"WavLM centroid cache {cache} is missing or stale and --no-build-centroid was given")
        model = identity_factory()
        vectors = np.stack([model.embed(r["audio"] if audio_choice == "original" else r["restored_audio"])
                            for r in members])
        total = vectors.sum(axis=0)
        loo = [float(v @ ((total - v) / np.linalg.norm(total - v))) for v in vectors]
        centroid = total / np.linalg.norm(total)
        data = {"centroid": centroid.tolist(), "members_key": key,
                "calibration": {"clips": len(members), "loo_p05": round(float(np.percentile(loo, 5)), 4),
                                "loo_p10": round(float(np.percentile(loo, 10)), 4),
                                "loo_median": round(float(np.median(loo)), 4)}}
    centroid = np.array(data["centroid"])
    if (data.get("impostors") or {}).get("key") != impostor_key:
        if not allow_build:
            raise SystemExit(f"impostor calibration in {cache} is missing or stale and --no-build-centroid was given")
        model = model or identity_factory()
        corpus = {r["id"]: r for r in load_jsonl(LAB / "corpus.jsonl")}
        sims = {}
        for clip_id in impostor_ids:
            path = corpus[clip_id]["audio"] if audio_choice == "original" else str(LAB / "restored" / f"{clip_id}.wav")
            sims[clip_id] = round(float(model.embed(path) @ centroid), 4)
        data["impostors"] = {"key": impostor_key, "sims": sims, "durations": {c: corpus[c]["duration"] for c in sims}}
    sims = data["impostors"]["sims"]
    durations = data["impostors"]["durations"]
    calibration = dict(data["calibration"])
    if sims:
        values = np.array(list(sims.values()))
        long_values = [v for c, v in sims.items() if durations[c] >= 3.0]
        impostors = {"n": len(values), "median": round(float(np.median(values)), 4),
                     "p95": round(float(np.percentile(values, 95)), 4), "max": round(float(values.max()), 4),
                     "max_3s": round(float(max(long_values)), 4) if long_values else None,
                     "max_clip": max(sims, key=sims.get),
                     "by_character": {name: round(float(max(v for c, v in sims.items() if character.get(c) == name)), 4)
                                      for name in sorted(set(character.values()))}}
        calibration["impostors"] = impostors
        calibration["gate"] = round(max(calibration["loo_p05"], impostors["p95"], impostors["max_3s"] or 0.0), 4)
    else:  # no other character in his files (exclude_other_speakers.json confirmed is empty): his p05 alone
        calibration["impostors"] = {"n": 0, "median": None, "p95": None, "max": None, "max_3s": None,
                                    "max_clip": None, "by_character": {}}
        calibration["gate"] = round(calibration["loo_p05"], 4)
    calibration["unit"] = round(calibration["loo_median"] - calibration["loo_p05"], 4)
    calibration["centroid_key"] = data["members_key"]
    if model is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache, data)
    return centroid, calibration


def identity_units(sim: float, calibration: dict) -> float:
    return (sim - calibration["loo_median"]) / max(calibration["unit"], 1e-6)


# ------------------------------------------------------------------------------------------------ targets
def zscores(features: dict, profile: dict) -> dict:
    z = {}
    for key, stat in profile["stats"].items():
        if features.get(key) is not None:
            z[key] = (features[key] - stat["median"]) / stat["scale"]
    return derived_z({"z": z})


def clip3(value: float) -> float:
    return max(-3.0, min(3.0, value))


def label_targets(path: Path | None = None) -> dict[str, dict]:
    """Per label from label_stats.json: kind, validated signature (z), lines rule and flag bounds. Keys that start
    with '_' (documentation) and anything that is not a label entry are skipped."""
    path = path or DELIVERY / "label_stats.json"
    stats = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    targets = {}
    for label, entry in stats.items():
        if label.startswith("_") or not isinstance(entry, dict) or "kind" not in entry:
            continue
        targets[label] = {"kind": entry["kind"], "validated": bool(entry.get("validated")),
                          "signature": {k: v["z"] for k, v in (entry.get("signature") or {}).items()},
                          "lines": entry.get("lines", "all"), "flag_bounds": entry.get("flag_bounds") or {}}
    return targets


class LengthBaseline:
    """His median z (clipped) over the profile statements closest in syllable count to the take's line."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = sorted((r for r in rows if profile_member(r) and not ends_with_question(r["text"])
                            and r.get("text_syllables") and r.get("z")), key=lambda r: (r["text_syllables"], r["id"]))
        self.cache: dict[int, dict] = {}

    def __call__(self, syllables: int) -> dict:
        if syllables not in self.cache:
            near = sorted(self.rows, key=lambda r: (abs(r["text_syllables"] - syllables), r["id"]))[:LENGTH_NEIGHBOURS]
            keys = {k for r in near for k in derived_z(r)}
            self.cache[syllables] = {k: float(np.median([clip3(derived_z(r)[k]) for r in near if k in derived_z(r)]))
                                     for k in keys}
        return self.cache[syllables]


def direction(label: str, text: str, z: dict, base: dict, basis: str, targets: dict) -> dict:
    """Direction only where the label's target is validated (see the module docstring)."""
    target = targets.get(label, {})
    kind = target.get("kind")
    if label == "natural" or kind == "closeness":
        values = [min(abs(clip3(z[k]) - base.get(k, 0.0)), 3.0) for k in NATURAL_KEYS if k in z]
        return {"direction": round(math.tanh(1.0 - sum(values) / max(len(values), 1)), 3), "along": None,
                "off_target": None, "basis": f"closeness to his {basis}", "scored": True}
    if kind == "question_ending":
        goal = target["signature"].get("final_rise_st")
        if not text.strip().endswith("?") or goal is None or "final_rise_st" not in z:
            return {"direction": None, "along": None, "off_target": None, "basis": "not a question line", "scored": False}
        along_value = clip3(z["final_rise_st"]) / goal
        return {"direction": round(max(-1.0, min(1.0, 1.0 - abs(1.0 - along_value))), 3),
                "along": round(along_value, 3), "off_target": None, "basis": "his yes/no question endings",
                "scored": True}
    signature = target.get("signature") or {}
    keys = [k for k in signature if k in z]
    if not keys:
        return {"direction": None, "along": None, "off_target": None, "basis": "no signature", "scored": False}
    move = np.array([clip3(z[k]) - base.get(k, 0.0) for k in keys])
    goal = np.array([signature[k] for k in keys])
    along_value = float(move @ goal / (goal @ goal))
    others = [k for k in TARGET_KEYS if k in z and k not in signature]
    off = float(np.sqrt(np.mean([(clip3(z[k]) - base.get(k, 0.0)) ** 2 for k in others]))) if others else None
    result = {"along": round(along_value, 3), "off_target": round(off, 3) if off is not None else None,
              "basis": basis}
    if kind == "direction":
        result.update(direction=round(max(-1.0, min(1.0, 1.0 - abs(1.0 - along_value))), 3), scored=True)
    else:  # descriptive: reported for curiosity, never scored (the target is no more coherent than chance)
        result.update(direction=None, scored=False, along_descriptive=result.pop("along"))
        result["along"] = None
    return result


def objective(item: dict, calibration: dict) -> float:
    """Per take: direction (validated labels only) + identity units - WER term - FAIL_PENALTY on a failed gate."""
    if item.get("error"):
        return -2.0 - FAIL_PENALTY
    units = max(-2.0, min(0.5, identity_units(item["identity"], calibration)))
    wer_value = item.get("wer")
    wer_term = 0.0 if wer_value is None else max(0.0, min(1.0, (wer_value - 0.05) / 0.10))
    failed = (wer_value is not None and wer_value > 0.15) or item.get("instruction_leak") \
        or item["identity"] < calibration["gate"]
    return (item.get("direction") or 0.0) + units - wer_term - (FAIL_PENALTY if failed else 0.0)


def mean_se(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    return round(mean, 3), (round(se, 3) if se is not None else None)


def summarize(items: list[dict], calibration: dict) -> dict:
    ok = [i for i in items if not i.get("error")]
    direction_mean, direction_se = mean_se([i.get("direction") for i in ok])
    units_mean, units_se = mean_se([identity_units(i["identity"], calibration) for i in ok])
    objective_mean, objective_se = mean_se([objective(i, calibration) for i in items])
    wers = [i["wer"] for i in ok if i.get("wer") is not None]
    alongs = [i["along"] for i in ok if i.get("along") is not None]
    return {
        "n": len(items), "errors": len(items) - len(ok),
        "direction": direction_mean, "direction_se": direction_se,
        "along": round(float(np.mean(alongs)), 3) if alongs else None,
        "identity": round(float(np.mean([i["identity"] for i in ok])), 4) if ok else None,
        "identity_units": units_mean, "identity_units_se": units_se,
        "wer": round(float(np.mean(wers)), 3) if wers else None,
        "leaks": sum(bool(i.get("instruction_leak")) for i in ok),
        "flagged": sum(bool(i["flags"]) for i in items),
        "objective": objective_mean, "objective_se": objective_se,
    }


def paired_test(diffs: list[float], margin: float, boot: int, seed: int) -> dict:
    """Mean of paired differences with a percentile bootstrap 95% CI; qualifies only if the CI excludes 0 and
    the mean gain is at least `margin`."""
    if len(diffs) < 4:
        return {"n": len(diffs), "mean": None, "se": None, "ci_low": None, "ci_high": None, "qualifies": False}
    values = np.array(diffs)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(boot, len(values)))].mean(axis=1)
    low, high = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    mean = float(values.mean())
    return {"n": len(diffs), "mean": round(mean, 3), "se": round(float(values.std(ddof=1) / math.sqrt(len(values))), 3),
            "ci_low": round(low, 3), "ci_high": round(high, 3), "qualifies": bool(low > 0 and mean >= margin)}


# ------------------------------------------------------------------------------------------------ picks
def pick_instructions(results: list[dict], calibration: dict, margin: float, boot: int, settings: dict) -> dict:
    by_cell: dict[tuple, dict[str, dict]] = {}
    instructions: dict[tuple, set] = {}
    for r in results:
        if r["variant"] not in ("A", "B", "P", "P2"):
            continue
        by_cell.setdefault((r["delivery"], r["line_id"], r["seed"], r.get("cfg")), {})[r["variant"]] = r
        if r["variant"] in ("A", "B"):
            instructions.setdefault((r["delivery"], r["variant"]), set()).add(r.get("instruction_core") or "")
    picks = {}
    for label in sorted({key[0] for key in by_cell}):
        cells = [v for k, v in sorted(by_cell.items()) if k[0] == label]
        paired = {}
        for index, variant in enumerate(("B", "P", "P2")):
            diffs = [objective(c[variant], calibration) - objective(c["A"], calibration)
                     for c in cells if "A" in c and variant in c]
            if diffs:
                paired[variant] = paired_test(diffs, margin, boot, seed=20260930 + index)
        texts = {}
        for variant in ("A", "B"):
            found = instructions.get((label, variant), set())
            if len(found) > 1:
                raise SystemExit(f"{label}: the probe rendered {len(found)} different variant {variant} instructions")
            if found:
                texts[variant] = next(iter(found))
        qualified = sorted((v for v, t in paired.items() if t["qualifies"]), key=lambda v: -paired[v]["mean"])
        winner = qualified[0] if qualified else "A"
        controlled = "B" if paired.get("B", {}).get("qualifies") else "A"
        summary = {v: summarize([c[v] for c in cells if v in c], calibration) for v in ("A", "B", "P", "P2")
                   if any(v in c for c in cells)}
        if winner == "A":
            reason = "no alternative beat the Vox-style instruction detectably (paired 95% CI and margin)"
        elif winner == "B":
            reason = "the compact instruction beat the Vox-style one"
        else:
            reason = (f"prompt continuation ({'reference = the prompt clip' if winner == 'P' else 'reference = the '
                      'natural identity clip'}) beat instruction control")
        picks[label] = {"winner": winner, "controlled_variant": controlled, "instructions": texts,
                        "reason": reason, "paired_vs_A": paired, "summary": summary,
                        "prompt_id": settings.get("prompt_ids", {}).get(label),
                        "direction_scored": any(c[v].get("direction") is not None for c in cells for v in c)}
    return picks


def pick_note(run: str, pick: dict) -> str:
    def stat(variant: str) -> str:
        t = pick["paired_vs_A"].get(variant)
        if not t or t["mean"] is None:
            return f"{variant} n/a"
        return f"{variant}-A {t['mean']:+.2f} [{t['ci_low']:+.2f}, {t['ci_high']:+.2f}] n={t['n']}"
    evidence = "; ".join(stat(v) for v in ("B", "P", "P2") if v in pick["paired_vs_A"])
    instruction = "compact (B)" if pick["controlled_variant"] == "B" else "Vox-style (A)"
    if pick["winner"] in ("P", "P2"):
        mode = "ref_prompt" + (" with --identity-ref natural" if pick["winner"] == "P2" else "")
        return (f"prompt continuation beat instruction control: prefer the render step's {mode} for this label; "
                f"{instruction} instruction kept for controlled mode ({evidence})")
    if pick["winner"] == "B":
        return f"compact instruction beat the Vox-style one ({evidence})"
    measured = [v for v, t in pick["paired_vs_A"].items() if t.get("qualifies")]
    if measured:
        return f"Vox-style instruction kept by ear over a measured win for {', '.join(measured)} ({evidence})"
    return f"Vox-style instruction kept, no detectable difference ({evidence})"


def apply_picks(document: dict, run: str, delivery_dir: Path) -> None:
    """Store the probe's picks (exact rendered instruction strings) in instruction_picks.json and rebuild the map."""
    current = json.loads((delivery_dir / "delivery_map.json").read_text(encoding="utf-8"))["deliveries"]
    settings = document.get("settings", {})
    if settings.get("identity_id") and current["natural"]["references"][0]["id"] != settings["identity_id"]:
        print(f"warning: the identity clip changed since the probe ({settings['identity_id']} -> "
              f"{current['natural']['references'][0]['id']})", flush=True)
    picks_path = delivery_dir / "instruction_picks.json"
    stored = json.loads(picks_path.read_text(encoding="utf-8")) if picks_path.is_file() else {"picks": {}}
    for label, pick in document["picks"].items():
        if label not in current:
            print(f"warning: {label} is not in delivery_map.json; skipped", flush=True)
            continue
        variant = pick.get("controlled_variant", "A")
        text = (pick.get("instructions") or {}).get(variant)
        if not text:
            raise SystemExit(f"{label}: the probe recorded no variant {variant} instruction to apply")
        prompt_now = current[label]["references"][0]["id"]
        note = pick_note(run, pick)
        if pick["winner"] in ("P", "P2") and pick.get("prompt_id") and pick["prompt_id"] != prompt_now:
            note += f" (that result was for prompt {pick['prompt_id']}; rank 1 is now {prompt_now})"
        stored["picks"][label] = {"variant": variant, "run": run, "note": note, "instruction": text,
                                  "winner": pick["winner"]}
    backup = delivery_dir / f"delivery_map.before_{run}.json"
    if not backup.exists():
        backup.write_bytes((delivery_dir / "delivery_map.json").read_bytes())
    write_json(picks_path, stored)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run([sys.executable, str(BUILD), "--out-dir", str(delivery_dir)], env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"rebuild failed after storing the picks:\n{result.stderr[-2000:]}")
    applied = json.loads(result.stdout.strip().splitlines()[-1]).get("picks_applied")
    print(f"applied {len(document['picks'])} picks from probe {run} ({picks_path.name}); map rebuilt with picks "
          f"{applied}; previous map kept as {backup.name}", flush=True)


# ------------------------------------------------------------------------------------------------ main
def resolve_wav(take: dict, manifest: Path) -> str:
    wav = Path(take["wav"])
    return str(wav if wav.is_absolute() else (manifest.parent / wav).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rendered takes for delivery, identity and words (CPU).")
    parser.add_argument("--manifest", type=Path, default=None, help="shared-format takes manifest to score")
    parser.add_argument("--baseline-manifest", type=Path, default=None,
                        help="natural reads of the same texts (e.g. a render of takes_script_pairs.json)")
    parser.add_argument("--asr", choices=("cpu", "cuda", "none"), default="cpu")
    parser.add_argument("--asr-prompt", action="store_true",
                        help="give Whisper the character's asr_prompt (voice-lab/characters/<slug>.json) so his names "
                             "are spelled as in the script; off by default (earlier scores were made without it)")
    parser.add_argument("--keep-free-gb", type=float, default=8.0, help="--asr cuda: VRAM left for other apps")
    parser.add_argument("--identity-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="WavLM device; auto = cuda when --asr cuda got its VRAM headroom, else cpu")
    parser.add_argument("--workers", type=int, default=4,
                        help="processes measuring prosody (pyin, ~2-7 s per take each); ~0.4 GB host memory per worker")
    parser.add_argument("--identity-audio", choices=("original", "restored"), default="original",
                        help="which copy of his clips builds the WavLM centroid (user decision: original)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pick-instructions", action="store_true", help="probe manifests: choose A/B/P/P2 per label")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--boot", type=int, default=4000, help="bootstrap resamples for the paired picks")
    parser.add_argument("--no-build-centroid", action="store_true",
                        help="fail instead of embedding his corpus when the identity cache is missing or stale")
    parser.add_argument("--apply-from", type=Path, default=None,
                        help="apply a voxcpm_instruction_probe.json (after listening) and rebuild the map; no scoring")
    parser.add_argument("--delivery-dir", type=Path, default=DELIVERY,
                        help="with --apply-from: the folder whose delivery_map.json is rebuilt (tests use a copy)")
    args = parser.parse_args()
    if args.apply_from:
        document = json.loads(args.apply_from.read_text(encoding="utf-8"))
        apply_picks(document, document.get("run") or args.apply_from.parent.name, args.delivery_dir)
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --apply-from is given")
    if args.asr == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"):
            raise SystemExit("--asr cuda needs a visible GPU: run without CUDA_VISIBLE_DEVICES=\"\" (e.g. set it to 0)")
        import torch

        ok, info = resources.vram_headroom(torch, ASR_GPU_GB + IDENTITY_GPU_GB, args.keep_free_gb, 0.5)
        if not ok:
            print(f"GPU ASR would leave less than {args.keep_free_gb} GB free ({json.dumps(info)}); using CPU ASR",
                  flush=True)
            args.asr = "cpu"
        else:
            torch.cuda.set_per_process_memory_fraction(min(1.0, IDENTITY_GPU_GB / info["total_gb"]), 0)
    if args.identity_device == "auto":
        args.identity_device = "cuda" if args.asr == "cuda" else "cpu"
    elif args.identity_device == "cuda" and args.asr != "cuda":
        import torch

        ok, info = resources.vram_headroom(torch, IDENTITY_GPU_GB, args.keep_free_gb, 0.5)
        if not ok:
            print(f"GPU identity would leave less than {args.keep_free_gb} GB free; using the CPU", flush=True)
            args.identity_device = "cpu"
        else:
            torch.cuda.set_per_process_memory_fraction(min(1.0, IDENTITY_GPU_GB / info["total_gb"]), 0)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    takes = [dict(t, _manifest=str(args.manifest), _role="take") for t in manifest["takes"]]
    if args.baseline_manifest:
        extra = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))["takes"]
        takes += [dict(t, _manifest=str(args.baseline_manifest), _role="baseline") for t in extra]
    profile = json.loads((DELIVERY / "voice_profile.json").read_text(encoding="utf-8"))
    center = profile["register"]["center_hz"]
    stats = profile["stats"]
    targets = label_targets()
    rows = load_jsonl(DELIVERY / "features.jsonl")
    length_base = LengthBaseline(rows)
    test_ids = set(json.loads((LAB / "corpus_summary.json").read_text(encoding="utf-8"))["test_ids"])
    if args.out_dir:
        out_dir = args.out_dir
    elif DELIVERY / "probe" in args.manifest.resolve().parents:
        out_dir = args.manifest.parent
    else:
        out_dir = DELIVERY / "scores" / f"{args.manifest.parent.parent.name}_{args.manifest.parent.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    identity_model = None

    def identity_factory():
        nonlocal identity_model
        identity_model = identity_model or Identity(args.identity_device)
        return identity_model

    centroid, calibration = identity_calibration(identity_factory, args.identity_audio, not args.no_build_centroid)
    asr_prompt = (labcfg.CHARACTER.get("asr_prompt") or "") if args.asr_prompt else ""
    transcriber = Transcriber(args.asr, asr_prompt) if args.asr != "none" else None
    asr_tag = f"{args.asr}+prompt" if asr_prompt and args.asr != "none" else args.asr  # cache key: never mix the two
    cache_path = out_dir / ".score_cache.jsonl"
    cached = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cached[item["_key"]] = item

    # Prosody runs in a process pool (pyin dominates); identity and ASR run here meanwhile, take by take.
    from concurrent.futures import ProcessPoolExecutor

    plan = []
    for take in takes:
        wav = resolve_wav(take, Path(take["_manifest"]))
        stat = os.stat(wav) if os.path.exists(wav) else None
        key = "|".join(map(str, (SCORER_VERSION, take["take_id"], take["_role"], wav, stat.st_size if stat else 0,
                                 stat.st_mtime_ns if stat else 0, asr_tag, calibration["centroid_key"])))
        plan.append((take, wav, key))
    todo = [(take, wav, key) for take, wav, key in plan if not (key in cached and not cached[key].get("error"))]
    print(f"{len(plan)} takes: {len(plan) - len(todo)} from the cache, {len(todo)} to score "
          f"({args.workers} prosody workers, identity on {args.identity_device}, ASR {asr_tag})", flush=True)
    if todo:
        warm_jit()  # numba's cache is written here once, never by several prosody workers at once
    fresh = {}
    with cache_path.open("a", encoding="utf-8") as cache_stream, \
            ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {key: pool.submit(measure, (wav, take["text"], center)) for take, wav, key in todo}
        for index, (take, wav, key) in enumerate(todo, 1):
            params = take.get("params") or {}
            result = {
                "_key": key, "take_id": take["take_id"], "role": take["_role"], "engine": take.get("engine"),
                "line_id": take["line_id"], "text": take["text"], "delivery": take["delivery"],
                "variant": params.get("variant") or params.get("mode") or "take", "seed": take.get("seed"),
                "cfg": params.get("cfg_value"), "reference_id": take.get("reference_id"),
                "prompt_id": params.get("prompt_id"), "instruction_core": params.get("instruction_core"),
                "wav": wav,
            }
            try:
                if not os.path.exists(wav):
                    raise FileNotFoundError(wav)
                identity_sim = round(float(identity_factory().embed(wav) @ centroid), 4)
                transcript = transcriber(wav) if transcriber else None
                features = futures[key].result()
                if "error" in features:
                    raise RuntimeError(features["error"])
                z = zscores(features, profile)
                result.update({
                    "identity": identity_sim,
                    "syllables": features.get("text_syllables"),
                    "features": {k: features.get(k) for k in ("duration_s", "f0_median_hz", "f0_median_st", "f0_range_st",
                                                               "final_rise_st", "articulation_rate_syl_s", "pause_ratio",
                                                               "alpha_ratio_db", "hammarberg_db", "low_f0_ratio",
                                                               "level_dbfs")},
                    "z": {k: round(v, 3) for k, v in z.items() if k in REPORT_KEYS},
                })
                if transcript is not None:
                    result.update(asr=transcript, wer=round(wer(take["text"], transcript), 3),
                                  instruction_leak=instruction_leak(params.get("instruction", ""), take["text"],
                                                                    transcript))
            except Exception as exc:  # one bad take never loses the run
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["trace"] = traceback.format_exc(limit=2)
            references = {take.get("reference_id"), params.get("prompt_id"), params.get("identity_ref_id")}
            if references & test_ids:
                result["reference_is_test_clip"] = sorted(references & test_ids)
            fresh[key] = result
            cache_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            cache_stream.flush()
            if index % 20 == 0:
                print(f"scored {index}/{len(todo)}", flush=True)
    results = [fresh.get(key) or cached[key] for _, _, key in plan]

    # Baselines: probe 'none' (mean over seeds per line and cfg) > a natural read of the same text > length-matched.
    none_base: dict[tuple, list[dict]] = {}
    natural_base: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get("error"):
            continue
        if r["variant"] == "none":
            none_base.setdefault((r["line_id"], r["cfg"]), []).append(r["z"])
        elif r["delivery"] == "natural":
            natural_base.setdefault((r["engine"], norm_text(r["text"])), []).append(r["z"])

    def mean_z(items: list[dict]) -> dict:
        keys = {k for item in items for k in item}
        return {k: float(np.mean([clip3(item[k]) for item in items if k in item])) for k in keys}

    for r in results:
        if r.get("error"):
            r.update(direction=None, along=None, off_target=None, basis=None, flags=[f"scoring failed: {r['error']}"])
            continue
        base, basis = None, None
        if r["delivery"] != "natural":
            if none_base.get((r["line_id"], r["cfg"])):
                base, basis = mean_z(none_base[(r["line_id"], r["cfg"])]), "paired probe 'none' reads"
            elif natural_base.get((r["engine"], norm_text(r["text"]))):
                base, basis = mean_z(natural_base[(r["engine"], norm_text(r["text"]))]), "paired natural reads"
        if base is None:
            base, basis = length_base(int(r.get("syllables") or 0)), "length-matched medians"
        r.update(direction(r["delivery"], r["text"], r["z"], base, basis, targets))
        r["identity_units"] = round(identity_units(r["identity"], calibration), 3)
        flags = []
        f = r["features"]
        bounds = targets.get(r["delivery"], {}).get("flag_bounds", {})
        range_lo, range_hi = bounds.get("f0_range_st", [math.inf, -math.inf])
        rate_lo, rate_hi = bounds.get("articulation_rate_syl_s", [math.inf, -math.inf])
        value = f.get("f0_range_st")
        if value is not None and value < min(stats["f0_range_st"]["p10"], range_lo):
            flags.append("flat pitch (below his p10 range and his own clips of this delivery)")
        if value is not None and value > max(stats["f0_range_st"]["p90"] * 1.5, range_hi):
            flags.append("sing-song pitch (above 1.5x his p90 range and his clips of this delivery)")
        rate = f.get("articulation_rate_syl_s")
        if rate is not None and rate > max(stats["articulation_rate_syl_s"]["p90"] + 0.5, rate_hi):
            flags.append("rushed")
        if rate is not None and rate < min(stats["articulation_rate_syl_s"]["p10"] - 0.5, rate_lo):
            flags.append("dragging")
        if r.get("along") is not None and r["along"] > 2.0 and targets.get(r["delivery"], {}).get("kind") == "direction":
            flags.append("overacted (more than twice his clips' movement)")
        if r.get("off_target") is not None and r["off_target"] > 1.5:
            flags.append("large movement unrelated to the delivery")
        if r["identity"] < calibration["gate"]:
            flags.append("identity below the gate (his p05 / other characters)")
        if r.get("wer") is not None and r["wer"] > 0.15:
            flags.append("words wrong (WER > 0.15)")
        if r.get("instruction_leak"):
            flags.append("instruction text spoken")
        if r.get("reference_is_test_clip"):
            flags.append("HELD-OUT TEST CLIP USED AS REFERENCE")
        r["flags"] = flags
        r["objective"] = round(objective(r, calibration), 3)
    for r in results:
        if r.get("error"):
            r["objective"] = round(objective(r, calibration), 3)

    scored = [r for r in results if r["role"] == "take"]
    groups: dict[tuple, list[dict]] = {}
    for r in scored:
        groups.setdefault((r["delivery"], r["variant"]), []).append(r)
    summary = [{"delivery": label, "variant": variant, **summarize(items, calibration)}
               for (label, variant), items in sorted(groups.items())]
    report = {"manifest": str(args.manifest), "baseline_manifest": str(args.baseline_manifest or ""),
              "scorer_version": SCORER_VERSION, "calibration": calibration, "profile_center_hz": center,
              "targets": {k: {"kind": v["kind"], "signature_z": v["signature"]} for k, v in targets.items()},
              "summary": summary, "takes": [{k: v for k, v in r.items() if not k.startswith("_")} for r in scored]}

    # Reference ranking: deliveries rendered from more than one reference clip (engine manifests, not probes).
    delivery_map = json.loads((DELIVERY / "delivery_map.json").read_text(encoding="utf-8"))["deliveries"]
    by_ref: dict[tuple, dict[str, list[dict]]] = {}
    for r in scored:
        if r["variant"] not in ("A", "B", "P", "P2", "none") and r.get("reference_id"):
            by_ref.setdefault((r["delivery"], r["variant"]), {}).setdefault(r.get("prompt_id") or r["reference_id"],
                                                                           []).append(r)
    rerank = {}
    for (label, variant), refs in sorted(by_ref.items()):
        if len(refs) < 2:
            continue
        rendered = sorted(({"reference_id": rid, **summarize(items, calibration)} for rid, items in refs.items()),
                          key=lambda s: -(s["objective"] if s["objective"] is not None else -99))
        current = [ref["id"] for ref in delivery_map.get(label, {}).get("references", [])]
        proposed = [item["reference_id"] for item in rendered if item["reference_id"] in current]
        proposed += [rid for rid in current if rid not in proposed]
        rerank[f"{label}/{variant}"] = {"current_order": current, "proposed_order": proposed, "rendered": rendered}
    if rerank:
        write_json(out_dir / "reference_rerank.json", {
            "note": "Proposal to audition, not a result: unpaired means over the takes of each reference (standard "
                    "errors in 'rendered'). Reorder delivery_map.json only after listening, by changing the evidence "
                    "in build_delivery_map.py and rebuilding.",
            "labels": rerank})
        report["reference_rerank"] = {k: v["proposed_order"][:3] for k, v in rerank.items()}

    picks = {}
    if args.pick_instructions:
        picks = pick_instructions(scored, calibration, args.margin, args.boot, manifest.get("settings", {}))
        settings = manifest.get("settings", {})
        write_json(out_dir / "voxcpm_instruction_probe.json", {
            "run": settings.get("run") or args.manifest.parent.name, "manifest": str(args.manifest),
            "rule": f"alternative wins only if the paired (line, seed) 95% bootstrap CI of its objective gain over A "
                    f"excludes 0 and the mean gain >= {args.margin}; otherwise A. Edit 'winner'/'controlled_variant' "
                    "after listening to overrule, then apply with score_delivery.py --apply-from <this file>.",
            "margin": args.margin, "boot": args.boot,
            "settings": {k: settings.get(k) for k in ("delivery_map_sha256", "variants_sha256", "identity_id",
                                                      "prompt_ids", "cfgs", "seeds", "lora", "snapshot")},
            "picks": picks})
        report["instruction_picks"] = {k: v["winner"] for k, v in picks.items()}

    write_json(out_dir / "delivery_scores.json", report)

    def fmt(value, se=None, digits=2, signed=True):
        if value is None:
            return "-"
        text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
        return f"{text} ± {se:.{digits}f}" if se is not None else text

    imp = calibration["impostors"]
    others = (f"the other characters in his files (n={imp['n']}) median {imp['median']}, p95 {imp['p95']}, max "
              f"{imp['max']} ({imp['max_clip']}; clips >= 3 s max {imp['max_3s']})" if imp["n"]
              else "no other character's clips in his files to calibrate against")
    lines = [f"# Delivery scores: {args.manifest}", "",
             f"Identity (WavLM-SV vs his train/val centroid): his clips leave-one-out p05 {calibration['loo_p05']}, "
             f"median {calibration['loo_median']} ({calibration['clips']} clips); {others}. Gate {calibration['gate']}. "
             "Identity units: 0 = his median, -1 = his p05. The eval's three-encoder scorecard is the identity "
             "authority.",
             "Direction is scored only where his clips of the label are measurably coherent (validated: "
             + ", ".join(sorted(k for k, v in targets.items() if v["kind"] in ("direction", "question_ending")))
             + "; natural = closeness). Every other delivery is judged by ear. Means ± standard error.", "",
             "| delivery | variant | n | direction | along | identity units | WER | leaks | flagged | objective |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['delivery']} | {s['variant']} | {s['n']} | {fmt(s['direction'], s['direction_se'])} | "
                     f"{fmt(s['along'])} | {fmt(s['identity_units'], s['identity_units_se'])} | "
                     f"{fmt(s['wer'], signed=False)} | {s['leaks']} | {s['flagged']} | "
                     f"{fmt(s['objective'], s['objective_se'])} |")
    if rerank:
        lines += ["", "## Reference ranking (proposal to audition)", ""] + [
            f"- {k}: " + ", ".join(f"{labcfg.short_id(i['reference_id'])} ({fmt(i['objective'], i['objective_se'])}, "
                                   f"n={i['n']})"
                                   for i in v["rendered"]) for k, v in rerank.items()]
    if picks:
        lines += ["", f"## VoxCPM2 instruction picks (paired by line and seed; switch only when the 95% CI excludes 0 "
                      f"and the mean gain >= {args.margin})", "",
                  "| label | winner | controlled mode | B - A | P - A | P2 - A |", "|---|---|---|---|---|---|"]
        for label, pick in picks.items():
            cells = []
            for variant in ("B", "P", "P2"):
                t = pick["paired_vs_A"].get(variant)
                cells.append("-" if not t or t["mean"] is None else
                             f"{t['mean']:+.2f} ± {t['se']:.2f} [{t['ci_low']:+.2f}, {t['ci_high']:+.2f}] n={t['n']}"
                             + (" (wins)" if t["qualifies"] else ""))
            lines.append(f"| {label} | {pick['winner']} | {pick['controlled_variant']} | " + " | ".join(cells) + " |")
    (out_dir / "delivery_scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "takes": len(scored), "errors": sum(1 for r in scored if r.get("error")),
                      "calibration_gate": calibration["gate"], "summary": summary[:40],
                      "picks": report.get("instruction_picks"), "reference_rerank": report.get("reference_rerank")},
                     indent=1))


if __name__ == "__main__":
    main()
