"""Aggregate metric outputs into <lab>/eval/<run>/scorecard.json and scorecard.md.

Standalone: <python with numpy> aggregate.py --run-dir <lab>/eval/<run> [options]
(run_eval.py calls it in-process after the metric steps.) <lab> is the lab of the character VOICE_LAB_CHARACTER
selects (default baodur: BaoDurLab); "the character" below is that voice (Bao-Dur's numbers are given as examples),
and the scorecard names him/her from the character config.

Purpose: throw out broken takes, measure identity fairly, and hand the listener a short list per line. The
user's ears pick the take; nothing here pretends to score emotion unless a signal proves it can.

Ranking per line: tiers first, then the composite (also printed at the top of scorecard.md).
  tier 0  passes every gate
  tier 1  identity gate: identity below the --identity-gate-q percentile (default p05) of the character's own
          clips, measured the same way (another voice, or a weak clone)
  tier 2  hard failure: WER > max(--wer-gate, one word) (subtitles show the text), 2+ consecutive missing or
          extra words, unreadable/silent/non-finite audio, > 0.5 % clipped samples, a full-scale one-sample click
  tier 3  leak: a held-out test clip (or a train clip repeating that held-out line) used as conditioning, or a
          model trained on it
  Unscored takes (a step hit a resource error, or a metric output predates the take) are not ranked: re-run
  the named step.
Identity, per speaker model (ERes2NetV2, Resemblyzer, WavLM-SV when present), on the run's primary variant:
  identity = (cos(take, character's train centroid) - impostor_mean) / (expected_real_cos - impostor_mean)
  expected_real_cos = least-squares fit on the character's kept train clips' leave-one-out cosines:
    b0 + b1 * L + b2 * L^2 + b3 * min(atypicality, p95),  L = log(clamp(speech span, 1, 12 s)); the span part is
    made non-decreasing (on Bao-Dur a log-linear form left a -0.22 rank trend with span for ERes2NetV2, this one -0.04)
  atypicality = mean robust |z| of F0 median, F0 IQR, energy IQR and speaking rate vs the train clips. On Bao-Dur's
  own clips raw similarity rises with length (Spearman +0.77) and falls with expressive delivery (-0.45); the
  fit removes both, so 1.0 = as close as the character's own clips of that length and expressiveness, 0.0 = the
  impostor voices of <lab>/eval/eval_character.json (Bao-Dur: other male game voices, Atton and Carth).
  Held-out (and train/val diagnostic) lines average in the paired identity: cos to the real recording of the
  line, anchored on real cross-line pairs vs impostor pairs.
  A take's identity is the mean over the models its engine is NOT conditioned on: GPT-SoVITS v2Pro/v2ProPlus feed
  ERes2NetV2.forward3(reference) into the generator (module/models.py sv_emb), so ERes2NetV2 is left out of
  their identity and only reported (also any model named in params.conditioning_encoders).
Composite (ranks within a tier):
  composite = min(identity, 1)
              - w_prosody   * prosody distance to the line's real recording (paired lines only; two-sided)
              - w_plausible * implausibility: robust sd beyond the character's own range, one-sided per feature
                              (monotone F0 IQR/range or flat dynamics below the p05; F0 median and rate beyond
                              p01/p99; pauses and creak above p99). Expressive takes inside that range cost nothing.
              - w_natural   * DNSMOS deficit: robust sd below the real clips' p05 (mean of SIG and OVRL)
              - w_artifact  * soft artifact/ASR flags
              + w_delivery  * delivery match, only for a delivery signal that passes leave-one-out validation on
                              the delivery map's own reference clips (prosody target and WavLM layer features are
                              tested every run; per-label prosody cues are shown either way, unweighted)
  Takes within --tie-margin of a line's best tier-0 take are ties: listen to all of them.
Engine comparison: groups (engine/run folders by default) are compared only on the lines both rendered, by the
expected best-of-k selection score at equal k (k = the smaller group's takes on that line), with a bootstrap 95 %
CI over lines; a winner needs a CI that excludes 0 and per-model identity differences that agree in sign.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402

SV_METRICS = (("sv_eres2net", "ERes2NetV2"), ("sv_resemblyzer", "Resemblyzer"), ("sv_wavlm", "WavLM-SV"))
SV_LABEL = dict(SV_METRICS)
SV_SHORT = {"sv_eres2net": "E", "sv_resemblyzer": "R", "sv_wavlm": "W"}
METRIC_NAMES = ("sv_eres2net", "sv_resemblyzer", "sv_wavlm", "asr", "prosody", "artifacts", "dnsmos")
OPTIONAL_METRICS = {"sv_wavlm", "dnsmos"}
# Opt-in steps from the eval-extra venv: UTMOSv2 naturalness MOS (metric_utmos.py) and emotion2vec+ classes/embeddings
# (metric_emotion.py). Loaded only when their output exists (no warning when absent), reported in their own columns,
# never part of tiers or the composite; a missing/stale entry is a note, not "unscored".
EXTRA_METRICS = ("utmos", "emotion")
# Names in model_version / conditioning_encoders that mean an engine's generator is conditioned on that model.
COUPLING = {"sv_eres2net": ("eres2net",), "sv_resemblyzer": ("resemblyzer", "ge2e"), "sv_wavlm": ("wavlm",)}
V2PRO_PREFIX = "v2pro"  # GPT-SoVITS v2Pro / v2ProPlus: sv_emb = ERes2NetV2.forward3(reference fbank)

# name, weight, floor for the robust scale, label, unit (held-out match distance and the prosody delivery signal)
PROSODY_FEATURES = [
    ("f0_median_st", 1.0, 0.5, "F0 median", "st"),
    ("f0_iqr_st", 1.0, 0.3, "F0 IQR", "st"),
    ("f0_range_st", 0.5, 0.5, "F0 range p5-p95", "st"),
    ("f0_end_st", 0.5, 0.5, "line-end F0 shift", "st"),
    ("rate_syl_s", 1.0, 0.25, "speaking rate", "syl/s"),
    ("pause_ratio", 0.75, 0.03, "pause ratio", ""),
    ("energy_iqr_db", 0.5, 1.0, "energy IQR", "dB"),
    ("creak_ratio", 0.5, 0.05, "creak ratio", ""),
    ("artic_rate_syl_s", 0.0, 0.25, "articulation rate", "syl/s"),
    ("voiced_ratio", 0.0, 0.05, "voiced ratio", ""),
    ("lufs", 0.0, 1.0, "loudness", "LUFS"),
]
FEATURE_LABEL = {n: (l, u) for n, _w, _f, l, u in PROSODY_FEATURES}
HE, HIS = ec.HE, ec.HIS  # the character's pronouns in report text (character config "pronouns", default he/his)
# feature, weight, low bound (quantile of the character's train clips >= 2 s), high bound, what a low / high value means
PLAUSIBILITY = [
    ("f0_median_st", 1.0, "p01", "p99", f"pitch below {HIS} range", f"pitch above {HIS} range"),
    ("f0_iqr_st", 1.0, "p05", "p99", "monotone (F0 IQR)", "erratic intonation (F0 IQR)"),
    ("f0_range_st", 0.5, "p05", "p99", "monotone (F0 range)", "erratic intonation (F0 range)"),
    ("rate_syl_s", 1.0, "p01", "p99", f"slower than {HE} ever speaks", f"faster than {HE} ever speaks"),
    ("pause_ratio", 0.75, None, "p99", None, f"longer pauses than any of {HIS} lines"),
    ("energy_iqr_db", 0.5, "p05", None, "flat dynamics (energy IQR)", None),
    ("creak_ratio", 0.5, None, "p99", None, f"more creak/fry than {HIS} clips"),
]
ATYPICALITY_FEATURES = ("f0_median_st", "f0_iqr_st", "energy_iqr_db", "rate_syl_s")
CUE_FEATURES = ("f0_median_st", "f0_iqr_st", "f0_range_st", "f0_end_st", "rate_syl_s", "pause_ratio",
                "energy_iqr_db", "creak_ratio")
Z_CAP = 4.0
SPAN_MIN_S, SPAN_MAX_S = 1.0, 12.0  # identity fit: log-span clamped here (few real clips are longer)
REAL_MIN_SPAN_S = 2.0               # real clips shorter than this are too noisy for thresholds
HARD_CLIP_FRACTION = 0.005
SOFT_CLIP_FRACTION = 1e-4
WRAP_SPIKE_REL = 0.6     # int16 wrap through the 32->48 kHz resample: spike_ratio 10-16, spike_rel_peak 0.68-0.78
CLICK_SPIKE_RATIO = 8.0  # real train clips: spike_ratio max 7.5 (original), 3.4 (restored)
DELIVERY_MIN_ACC, DELIVERY_CHANCE_X, DELIVERY_MAX_P, DELIVERY_MIN_N = 0.35, 3.0, 1e-3, 20

DEFAULTS = {"wer_gate": 0.05, "w_prosody": 0.05, "w_plausible": 0.15, "w_natural": 0.10, "w_artifact": 0.05,
            "w_delivery": 0.10, "identity_gate_q": 5.0, "tie_margin": 0.10, "group_by": "run", "top_n": 3,
            "bootstrap": 2000}


def add_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("selection")
    group.add_argument("--wer-gate", type=float, default=DEFAULTS["wer_gate"],
                       help=f"max WER (one word error is always allowed); {HIS} real test clips score 0")
    group.add_argument("--w-prosody", type=float, default=DEFAULTS["w_prosody"],
                       help="cost per unit prosody distance to the line's real recording (paired lines only)")
    group.add_argument("--w-plausible", type=float, default=DEFAULTS["w_plausible"],
                       help=f"cost per robust sd outside {HIS} own prosody range (one-sided)")
    group.add_argument("--w-natural", type=float, default=DEFAULTS["w_natural"],
                       help=f"cost per robust sd of DNSMOS below {HIS} real clips' p05")
    group.add_argument("--w-artifact", type=float, default=DEFAULTS["w_artifact"], help="cost per soft flag")
    group.add_argument("--w-delivery", type=float, default=DEFAULTS["w_delivery"],
                       help="weight of a VALIDATED delivery signal (0..1 match); unvalidated signals get none")
    group.add_argument("--identity-gate-q", type=float, default=DEFAULTS["identity_gate_q"],
                       help=f"identity gate = this percentile of {HIS} own clips' identity")
    group.add_argument("--tie-margin", type=float, default=DEFAULTS["tie_margin"],
                       help="composite gap below which takes are listed as ties for listening")
    group.add_argument("--group-by", choices=("run", "engine"), default=DEFAULTS["group_by"],
                       help="columns: engine/run folder (default; VoxCPM adapters and Seed-VC checkpoints apart) "
                            "or the manifest engine field")
    group.add_argument("--top-n", type=int, default=DEFAULTS["top_n"], help="takes listed per line")
    group.add_argument("--bootstrap", type=int, default=DEFAULTS["bootstrap"], help="bootstrap resamples")


def options_from_args(args) -> dict:
    return {k: getattr(args, k) for k in DEFAULTS}


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def _median(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else 0.5 * (v[mid - 1] + v[mid])


def fmt(value, digits=2, signed=False):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):  # average ties
        idx = values == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 5:
        return None
    a, b = (np.asarray(v, dtype=np.float64) for v in zip(*pairs))
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def expected_best_of_k(values: list[float], k: int) -> float | None:
    """Expected maximum of k values drawn without replacement from `values` (equalises takes per group)."""
    v = sorted(values)
    n = len(v)
    if not n:
        return None
    k = max(1, min(k, n))
    total = math.comb(n, k)
    return sum(v[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / total


def bootstrap_ci(diffs: list[float], reps: int, seed: int = 20260930) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=np.float64)
    if len(d) < 2:
        return float(d.mean()), float(d.mean())
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(max(200, reps), len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def normalize_sv(metric: dict, primary: str) -> dict:
    """Accept the flat speaker-metric schema written before per-variant output (baselines + takes at top level)."""
    baselines = metric.get("baselines") or {}
    if "test_vs_centroid" not in baselines:
        return metric
    variant = metric.get("audio_variant") or primary
    out = dict(metric)
    out["baselines"] = {variant: baselines}
    out["variants"] = [variant]
    takes = {}
    for key, entry in (metric.get("takes") or {}).items():
        if isinstance(entry, dict) and "error" not in entry and "by_variant" not in entry:
            takes[key] = {"sha256": entry.get("sha256"), "by_variant": {variant: {
                k: entry[k] for k in ("centroid_cos", "heldout_cos", "reference_cos") if k in entry}}}
        else:
            takes[key] = entry
    out["takes"] = takes
    return out


def load_metrics(run_dir: Path, run: dict, warnings: list[str]) -> tuple[dict, dict]:
    signature = ec.takes_signature(run["takes"])
    metrics, stale = {}, {}
    for name in METRIC_NAMES + EXTRA_METRICS:
        path = ec.metric_out_path(run_dir, name)
        if not path.exists():
            if name in EXTRA_METRICS:
                continue
            if name in OPTIONAL_METRICS:
                warnings.append(f"optional metric {name} not run: {path}")
            else:
                warnings.append(f"metric {name} missing (step failed or skipped): {path}")
            continue
        try:
            metric = ec.read_json(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"metric {name} unreadable ({exc}); left out")
            continue
        if name.startswith("sv_"):
            metric = normalize_sv(metric, run["audio_variant"])
        if metric.get("audio_variant") not in (None, run["audio_variant"]):
            warnings.append(f"{name}.json was computed for {metric['audio_variant']} audio, run is {run['audio_variant']}")
        if metric.get("takes_signature") and metric["takes_signature"] != signature:
            stale[name] = True
            warnings.append(f"{name}.json was computed for a different take set (manifests changed since); takes it "
                            f"lacks or that changed are unscored: re-run with --metrics {name}")
        metrics[name] = metric
    return metrics, stale


# ----------------------------------------------------------------------------------------------
# Real-clip models: speaker purity, prosody, identity anchors, naturalness
# ----------------------------------------------------------------------------------------------

# Other-speaker screen thresholds, per character (<lab>/eval/eval_character.json speaker_screen; Bao-Dur 0.35/0.0/2.5)
LIKELY_OTHER_IDENTITY = ec.EVAL_CHARACTER["speaker_screen"]["likely_other_identity"]      # clips >= SHORT_CLIP_S
POSSIBLE_OTHER_IDENTITY = ec.EVAL_CHARACTER["speaker_screen"]["possible_other_identity"]  # shorter, noisier clips
SHORT_CLIP_S = ec.EVAL_CHARACTER["speaker_screen"]["short_clip_s"]


def speaker_purity(metrics: dict, corpus: dict, variant: str) -> dict:
    """Train clips that are not the character: the delivery task's confirmed list plus the speaker models' consensus."""
    per_model = {}
    for m, _label in SV_METRICS:
        values = _get(metrics.get(m) or {}, "baselines", variant, "train_identity")
        if values:
            per_model[m] = values
    known, known_source = ec.known_other_speakers()
    train_ids = {i for i, r in corpus.items() if r["split"] == "train"}
    known &= train_ids
    likely, possible = [], []
    for clip_id in sorted(set().union(*[set(v) for v in per_model.values()]) if per_model else []):
        values = {m: v[clip_id] for m, v in per_model.items() if clip_id in v}
        mean = sum(values.values()) / len(values)
        row = corpus.get(clip_id, {})
        entry = {"id": clip_id, "identity": mean, "by_model": values, "duration": row.get("duration"),
                 "text_source": row.get("text_source"), "text": row.get("text"), "confirmed": clip_id in known}
        if (row.get("duration") or 0) >= SHORT_CLIP_S and mean < LIKELY_OTHER_IDENTITY:
            likely.append(entry)
        elif (row.get("duration") or 0) < SHORT_CLIP_S and mean < POSSIBLE_OTHER_IDENTITY:
            possible.append(entry)
    likely.sort(key=lambda e: e["identity"])
    possible.sort(key=lambda e: e["identity"])
    flagged = {e["id"] for e in likely + possible}
    identity_of = {}
    for clip_id in known:
        values = [v[clip_id] for v in per_model.values() if clip_id in v]
        identity_of[clip_id] = sum(values) / len(values) if values else None
    scored = {i for i, v in identity_of.items() if v is not None}
    return {"models": list(per_model), "variant": variant,
            "rule": f"mean train identity (leave-one-out, constant anchor) over speaker models < {LIKELY_OTHER_IDENTITY} "
                    f"for clips >= {SHORT_CLIP_S} s (likely another character), < {POSSIBLE_OTHER_IDENTITY} for shorter "
                    "clips (possible, low confidence); plus clips the delivery task confirmed as other characters",
            "likely_other_speaker": likely, "possible_other_speaker": possible, "confirmed_source": known_source,
            "confirmed_other_speaker": [{"id": i, "identity": identity_of[i], "duration": corpus[i].get("duration"),
                                         "text": corpus[i].get("text"), "models_agree": i in flagged}
                                        for i in sorted(known, key=lambda i: (identity_of[i] is None, identity_of[i] or 0))],
            "agreement": {"confirmed": len(known), "confirmed_scored": len(scored), "models_flagged": len(flagged),
                          "both": len(known & flagged), "confirmed_only": sorted(scored - flagged),
                          "confirmed_not_scored": sorted(known - scored), "models_only": sorted(flagged - known)},
            "exclude": sorted(flagged | known)}


def real_features(metrics: dict) -> dict:
    return {i: r for i, r in (_get(metrics.get("prosody") or {}, "real") or {}).items()
            if isinstance(r, dict) and "error" not in r}


def real_span(clip_id: str, feats: dict, corpus: dict) -> float | None:
    r = feats.get(clip_id)
    if r and r.get("span_s"):
        return float(r["span_s"])
    row = corpus.get(clip_id)
    return float(row["duration"]) if row and row.get("duration") else None


def take_span(key: str, metrics: dict) -> float | None:
    p = _get(metrics.get("prosody") or {}, "takes", key)
    if isinstance(p, dict) and p.get("span_s"):
        return float(p["span_s"])
    a = _get(metrics.get("artifacts") or {}, "takes", key)
    if isinstance(a, dict) and a.get("speech_detected") and a.get("duration_s") is not None:
        return max(0.0, a["duration_s"] - (a.get("lead_silence_s") or 0.0) - (a.get("tail_silence_s") or 0.0))
    return None


def zscore(value, name: str, pm: dict):
    c, s = pm["center"].get(name), pm["scale"].get(name)
    if value is None or c is None or not s:
        return None
    return (value - c) / s


def atypicality(features: dict | None, pm: dict | None):
    if not features or not pm:
        return None
    values = []
    for name in ATYPICALITY_FEATURES:
        z = zscore(features.get(name), name, pm)
        if z is None:
            return None
        values.append(abs(z))
    return float(np.mean(values))


def prosody_model(feats: dict, exclude: set[str], delivery_map: dict | None, tests: set[str],
                  warnings: list[str]) -> dict | None:
    train = {i: r for i, r in feats.items() if r.get("split") == "train" and i not in exclude}
    if len(train) < 5:
        return None
    center, scale = {}, {}
    for name, _w, floor, *_ in PROSODY_FEATURES:
        s = ec.stats([r.get(name) for r in train.values()])
        if s.get("n", 0) >= 5:
            center[name] = s["median"]
            scale[name] = max(floor, (s["p75"] - s["p25"]) / 1.349)
    long_train = [r for r in train.values() if (r.get("span_s") or 0) >= REAL_MIN_SPAN_S]
    ranges = {}
    for name, *_ in PLAUSIBILITY:
        values = [r[name] for r in long_train if r.get(name) is not None]
        if len(values) >= 10:
            ranges[name] = {f"p{q:02d}": float(np.percentile(values, q)) for q in (1, 5, 50, 95, 99)}
    pm = {"center": center, "scale": scale, "ranges": ranges, "real": feats, "n_train": len(train),
          "n_long": len(long_train)}
    atys = [a for a in (atypicality(r, pm) for r in long_train) if a is not None]
    pm["aty_cap"] = float(np.percentile(atys, 95)) if len(atys) >= 10 else None
    pm["aty_median"] = float(np.median(atys)) if atys else None
    targets = {}
    if delivery_map:
        for label, spec in (delivery_map.get("deliveries") or {}).items():
            ids = []
            for ref in spec.get("references", []):
                rid = ref.get("id")
                if rid in tests:
                    warnings.append(f"delivery map '{label}' lists held-out test clip {rid} as a reference; ignored")
                    continue
                if rid in exclude:
                    warnings.append(f"delivery map '{label}' reference {rid} is another speaker's line or scores like "
                                    "one; ignored")
                    continue
                if rid in feats:
                    ids.append(rid)
            if ids:
                targets[label.casefold()] = {"ids": ids, "features": {
                    name: _median([feats[i].get(name) for i in ids]) for name, *_ in PROSODY_FEATURES}}
    pm["targets"] = targets
    return pm


def prosody_distance(features: dict | None, target: dict | None, pm: dict):
    """Two-sided RMS of weighted robust z (used only against the line's own real recording)."""
    if not features or not target or "error" in features:
        return None, {}
    num = den = 0.0
    z = {}
    for name, weight, *_ in PROSODY_FEATURES:
        x, t, s = features.get(name), target.get(name), pm["scale"].get(name)
        if x is None or t is None or not s:
            continue
        z[name] = (x - t) / s
        if weight > 0:
            num += weight * min(abs(z[name]), Z_CAP) ** 2
            den += weight
    return (math.sqrt(num / den) if den else None), z


def plausibility(features: dict | None, pm: dict | None):
    """One-sided penalty (robust sd beyond the character's own p01/p05/p95/p99) and the matching prosody flags."""
    if not features or not pm or "error" in features:
        return None, [], {}
    penalty, flags, excess = 0.0, [], {}
    for name, weight, lo_q, hi_q, lo_text, hi_text in PLAUSIBILITY:
        x, rng, s = features.get(name), pm["ranges"].get(name), pm["scale"].get(name)
        if x is None or not rng or not s:
            continue
        label, unit = FEATURE_LABEL[name]
        if lo_q and x < rng[lo_q]:
            e = min((rng[lo_q] - x) / s, Z_CAP)
            excess[name] = -e
            penalty += weight * e
            flags.append(f"{lo_text}: {x:.2f}{(' ' + unit) if unit else ''} ({HIS} {lo_q} {rng[lo_q]:.2f})")
        elif hi_q and x > rng[hi_q]:
            e = min((x - rng[hi_q]) / s, Z_CAP)
            excess[name] = e
            penalty += weight * e
            flags.append(f"{hi_text}: {x:.2f}{(' ' + unit) if unit else ''} ({HIS} {hi_q} {rng[hi_q]:.2f})")
    return penalty, flags, excess


def identity_model(metric: dict, variant: str, feats: dict, corpus: dict, pm: dict | None,
                   exclude: set[str]) -> dict | None:
    """Expected real cosine as a function of speech span and expressiveness, fitted on the kept train clips."""
    b = _get(metric, "baselines", variant)
    if not b:
        return None
    model = {"variant": variant, "impostor_mean": _get(b, "impostor_vs_centroid", "mean"),
             "heldout_mean": _get(b, "heldout_real_vs_centroid", "mean"),
             "heldout_sd": _get(b, "heldout_real_vs_centroid", "std"),
             "pair_real": _get(b, "test_pairwise", "mean"), "pair_real_sd": _get(b, "test_pairwise", "std"),
             "pair_imp": _get(b, "impostor_vs_test_pairwise", "mean")}
    rows = []
    for clip_id, cos in (b.get("train_loo_cos") or {}).items():
        span = real_span(clip_id, feats, corpus)
        if clip_id in exclude or span is None or span < SPAN_MIN_S:
            continue
        rows.append((float(cos), span, atypicality(feats.get(clip_id), pm)))
    use_aty = bool(pm and pm.get("aty_cap") is not None and sum(r[2] is not None for r in rows) >= 30)
    data = [r for r in rows if r[2] is not None] if use_aty else rows
    if len(data) < 15:
        model.update(kind="constant", note="no train leave-one-out cosines (old metric output or tiny corpus): "
                                           "anchored on the held-out real mean")
        return model
    y = np.array([r[0] for r in data])
    L = np.log(np.clip([r[1] for r in data], SPAN_MIN_S, SPAN_MAX_S))
    columns = [np.ones(len(data)), L, L ** 2]
    if use_aty:
        columns.append(np.minimum([r[2] for r in data], pm["aty_cap"]))
    X = np.stack(columns, axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    # The span curve is made non-decreasing (running maximum on a log grid): more speech never lowers the anchor.
    grid = np.linspace(math.log(SPAN_MIN_S), math.log(SPAN_MAX_S), 60)
    curve = np.maximum.accumulate(coef[0] + coef[1] * grid + coef[2] * grid ** 2)
    model.update(kind="span+expressiveness" if use_aty else "span", coef=[float(c) for c in coef], n=len(data),
                 resid_sd=float(np.sqrt(np.sum(resid ** 2) / max(1, len(data) - X.shape[1]))),
                 span_grid_log=[round(float(v), 5) for v in grid], span_curve=[round(float(v), 6) for v in curve],
                 aty_cap=pm["aty_cap"] if use_aty else None, aty_median=pm["aty_median"] if use_aty else None)
    return model


def expected_cos(model: dict, span: float | None, aty: float | None) -> float | None:
    if model.get("kind") == "constant":
        return model.get("heldout_mean")
    L = math.log(min(max(span if span else 5.0, SPAN_MIN_S), SPAN_MAX_S))
    value = float(np.interp(L, model["span_grid_log"], model["span_curve"]))
    c = model["coef"]
    if len(c) > 3:
        a = aty if aty is not None else model["aty_median"]
        value += c[3] * min(a, model["aty_cap"])
    return value


def anchored(cos, real, imp, real_sd=None):
    """0 = impostor mean, 1 = the real anchor; z-scaled fallback when there is no impostor baseline."""
    if cos is None or real is None:
        return None
    if imp is not None and real - imp > 0.02:
        return (cos - imp) / (real - imp)
    return 1.0 + (cos - real) / (4 * max(real_sd or 0.0, 0.01))


def real_identities(metric: dict, model: dict, variant: str, feats: dict, corpus: dict, pm: dict | None,
                    exclude: set[str]) -> dict:
    """Identity of the character's own clips under the anchor model: kept train clips (leave-one-out) and val/test."""
    b = _get(metric, "baselines", variant) or {}
    out = {}
    sources = [("train", b.get("train_loo_cos") or {})] + \
              [(split, _get(b, f"{split}_vs_centroid", "per_clip") or {}) for split in ("val", "test")]
    for split, values in sources:
        for clip_id, cos in values.items():
            if clip_id in exclude:
                continue
            span = real_span(clip_id, feats, corpus)
            if split == "train" and (span is None or span < REAL_MIN_SPAN_S):
                continue
            aty = atypicality(feats.get(clip_id), pm)
            out[clip_id] = {"split": split, "span": span, "aty": aty,
                            "identity": anchored(cos, expected_cos(model, span, aty), model["impostor_mean"],
                                                 model.get("heldout_sd")),
                            "identity_constant": anchored(cos, model["heldout_mean"], model["impostor_mean"],
                                                          model.get("heldout_sd"))}
    return out


def dnsmos_model(metric: dict, exclude: set[str]) -> dict | None:
    real = metric.get("real") or {}
    train = [r for i, r in real.items() if r.get("split") == "train" and i not in exclude and "error" not in r]
    model = {}
    for key in ("SIG", "BAK", "OVRL", "P808"):
        values = [r[key] for r in train if r.get(key) is not None]
        if len(values) >= 10:
            q05, q25, q50, q75 = np.percentile(values, [5, 25, 50, 75])
            model[key] = {"n": len(values), "p05": float(q05), "median": float(q50), "min": float(min(values)),
                          "scale": max(0.05, float(q75 - q25) / 1.349)}
    return model or None


def naturalness(entry: dict | None, model: dict | None):
    if not model or not isinstance(entry, dict) or "error" in entry:
        return None, []
    deficits, flags = [], []
    for key in ("SIG", "OVRL"):
        x, m = entry.get(key), model.get(key)
        if x is None or not m:
            continue
        deficits.append(min(max(0.0, (m["p05"] - x) / m["scale"]), Z_CAP))
        if x < m["p05"]:
            flags.append(f"DNSMOS {key} {x:.2f} below {HIS} real p05 {m['p05']:.2f}")
    bak, m = entry.get("BAK"), model.get("BAK")
    if bak is not None and m and bak < m["p05"]:
        flags.append(f"DNSMOS BAK {bak:.2f} below {HIS} real p05 {m['p05']:.2f} (background noise)")
    return (_mean(deficits) if deficits else None), flags


# ----------------------------------------------------------------------------------------------
# Opt-in eval-extra metrics (separate columns, never in tiers or the composite)
# ----------------------------------------------------------------------------------------------

def _extra_entry(metrics: dict, name: str, take: dict) -> dict | None:
    """A take's usable entry of an extra metric (None if absent, failed, or computed for another wav version)."""
    entry = _get(metrics.get(name) or {}, "takes", take["key"])
    if not isinstance(entry, dict) or entry.get("error") or entry.get("unscored"):
        return None
    if entry.get("sha256") and take.get("wav_sha256") and entry["sha256"] != take["wav_sha256"]:
        return None
    return entry


def utmos_model(metric: dict, exclude: set[str]) -> dict | None:
    """The character's real train clips' UTMOSv2 distribution (other-speaker clips left out): the anchor for take
    percentiles."""
    real = metric.get("real") or {}
    values = sorted(float(r["mos"]) for i, r in real.items()
                    if isinstance(r, dict) and r.get("split") == "train" and i not in exclude and r.get("mos") is not None)
    if len(values) < 10:
        return None
    q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": len(values), "min": values[0], "p05": float(q05), "p25": float(q25), "median": float(q50),
            "p75": float(q75), "p95": float(q95), "max": values[-1], "values": values}


def utmos_value(entry: dict | None, model: dict | None) -> dict | None:
    if not entry or entry.get("mos") is None:
        return None
    out = {"mos": entry["mos"], "rep_sd": entry.get("rep_sd")}
    if model:
        # share of the character's train clips scoring at or below this take (50 = the median clip)
        out["pct_real"] = 100.0 * int(np.searchsorted(model["values"], entry["mos"], side="right")) / model["n"]
        out["vs_real_median"] = entry["mos"] - model["median"]
    return out


def emotion_signal(metric: dict, exclude: set[str], tests: set[str]) -> dict:
    """Leave-one-out nearest-centroid test of the emotion2vec+ embedding on the delivery map's reference clips (same
    rule as the delivery signals; embeddings centred on the reference mean like WavLM). Reported, never weighted."""
    real = metric.get("real") or {}
    by_label: dict[str, list[str]] = {}
    for label, ids in (metric.get("reference_labels") or {}).items():
        keep = [i for i in ids if i not in exclude and i not in tests and isinstance(_get(real, i, "embedding"), list)]
        if keep:
            by_label[label] = keep
    labels_of: dict[str, set[str]] = {}
    for label, ids in by_label.items():
        for i in ids:
            labels_of.setdefault(i, set()).add(label)
    raw = {i: np.asarray(real[i]["embedding"], dtype=np.float64) for i in labels_of}
    if len(raw) < 6:
        return {"name": "emotion2vec", "metric": "cosine to label centroid", "labels": len(by_label),
                "validated": False, "weighted": False, "reason": "fewer than 6 reference embeddings"}
    center = np.mean(list(raw.values()), axis=0)
    result = _loo_signal("emotion2vec", {i: ec.unit(v - center) for i, v in raw.items()}, by_label, labels_of, True)
    result["weighted"] = False
    return result


def emotion_value(entry: dict | None, labels: list[str]) -> dict | None:
    if not entry or entry.get("top") is None:
        return None
    out = {"top": entry["top"], "top_p": entry.get("top_p"), "delivery": entry.get("delivery"),
           "probs": dict(zip(labels, entry.get("probs") or []))}
    ref = entry.get("reference")
    if isinstance(ref, dict) and "error" not in ref:
        out["ref"] = {k: ref.get(k) for k in ("label", "n_refs", "mean_cos", "max_cos", "nearest_ref", "centroid_cos",
                                              "rank", "of", "match", "nearest_label", "centered")}
    elif isinstance(ref, dict):
        out["ref_error"] = ref.get("error")
    if isinstance(entry.get("paired"), dict):
        out["paired"] = entry["paired"]
    return out


def extra_models(metrics: dict, exclude: set[str], tests: set[str]) -> dict:
    """Real-clip anchors of the extra metrics that are present ({} when none ran)."""
    out = {}
    if "utmos" in metrics:
        out["utmos"] = utmos_model(metrics["utmos"], exclude)
    if "emotion" in metrics:
        out["emotion_labels"] = list(metrics["emotion"].get("labels") or [])
        out["emotion_signal"] = emotion_signal(metrics["emotion"], exclude, tests)
    return out


# ----------------------------------------------------------------------------------------------
# Delivery: self-validating signals and descriptive cues
# ----------------------------------------------------------------------------------------------

def _prosody_vector(features: dict | None, pm: dict):
    if not features or "error" in features:
        return None
    out = []
    for name, weight, *_ in PROSODY_FEATURES:
        if weight <= 0:
            continue
        z = zscore(features.get(name), name, pm)
        if z is None:
            return None
        out.append(max(-Z_CAP, min(Z_CAP, z)) * math.sqrt(weight))
    return np.asarray(out)


def delivery_signals(pm: dict | None, metrics: dict, warnings: list[str]) -> dict:
    """Leave-one-out nearest-centroid test of each candidate delivery signal on the delivery map's references."""
    if not pm or not pm.get("targets"):
        return {}
    by_label = {label: t["ids"] for label, t in pm["targets"].items()}
    labels_of: dict[str, set[str]] = {}
    for label, ids in by_label.items():
        for i in ids:
            labels_of.setdefault(i, set()).add(label)
    candidates = {"prosody": {i: _prosody_vector(pm["real"].get(i), pm) for i in labels_of}}
    candidates["prosody"] = {i: v for i, v in candidates["prosody"].items() if v is not None}
    side = metrics.get("sv_wavlm", {}).get("side") if "sv_wavlm" in metrics else None
    wavlm_mean = None
    if side and side.get("real"):
        raw = {i: np.asarray(v, dtype=np.float64) for i, v in side["real"].items() if i in labels_of}
        if len(raw) >= 6:
            wavlm_mean = np.mean(list(raw.values()), axis=0)
            candidates["wavlm"] = {i: ec.unit(v - wavlm_mean) for i, v in raw.items()}
    results = {}
    for name, vectors in candidates.items():
        cosine = name == "wavlm"
        result = _loo_signal(name, vectors, by_label, labels_of, cosine)
        if "centroids" in result:
            result["center"] = wavlm_mean if cosine else None
        results[name] = result
    return results


def _loo_signal(name: str, vectors: dict, by_label: dict[str, list[str]], labels_of: dict[str, set[str]],
                cosine: bool) -> dict:
    """Leave-one-out nearest-centroid recognition of the delivery labels of the reference clips from `vectors`."""
    labels = [l for l, ids in by_label.items() if sum(i in vectors for i in ids) >= 3]
    result = {"name": name, "metric": "cosine to label centroid" if cosine else "euclidean on weighted robust z",
              "labels": len(labels), "validated": False}
    if len(labels) < 3:
        result["reason"] = "fewer than 3 labels with >= 3 references"
        return result

    def centroid(ids, skip=None):
        vs = [vectors[j] for j in ids if j in vectors and j != skip]
        if len(vs) < 2:
            return None
        c = np.mean(vs, axis=0)
        return ec.unit(c) if cosine else c

    def distance(v, c):
        return float(1.0 - v @ c) if cosine else float(np.linalg.norm(v - c))

    hits = n = 0
    chance = 0.0
    confusion: dict[str, dict[str, int]] = {}
    for i in sorted(i for i in labels_of if i in vectors):
        true = labels_of[i] & set(labels)
        if not true:
            continue
        cents = {l: centroid(by_label[l], skip=i) for l in labels}
        cents = {l: c for l, c in cents.items() if c is not None}
        if not cents:
            continue
        pred = min(cents, key=lambda l: distance(vectors[i], cents[l]))
        hits += pred in true
        n += 1
        chance += len(true & set(cents)) / len(cents)
        for t in true:
            confusion.setdefault(t, {}).setdefault(pred, 0)
            confusion[t][pred] += 1
    if not n:
        result["reason"] = "no references with vectors"
        return result
    chance /= n
    acc = hits / n
    p = binom_tail(hits, n, chance)
    result.update(n=n, hits=hits, accuracy=acc, chance=chance, p_value=p, confusion=confusion,
                  rule=f"accuracy >= max({DELIVERY_MIN_ACC}, {DELIVERY_CHANCE_X} x chance), p < {DELIVERY_MAX_P}, "
                       f"n >= {DELIVERY_MIN_N}")
    result["validated"] = bool(n >= DELIVERY_MIN_N and acc >= max(DELIVERY_MIN_ACC, DELIVERY_CHANCE_X * chance)
                               and p < DELIVERY_MAX_P)
    result["centroids"] = {l: centroid(by_label[l]) for l in labels}
    result["cosine"] = cosine
    return result


def delivery_score(signal: dict, label: str, vector) -> float | None:
    cents = {l: c for l, c in (signal.get("centroids") or {}).items() if c is not None}
    if vector is None or label not in cents or len(cents) < 2:
        return None
    if signal["cosine"]:
        dist = {l: float(1.0 - vector @ c) for l, c in cents.items()}
    else:
        dist = {l: float(np.linalg.norm(vector - c)) for l, c in cents.items()}
    order = sorted(dist, key=dist.get)
    return 1.0 - order.index(label) / (len(order) - 1)


def _cue_met(z, sign) -> bool:
    return z is not None and z * sign >= 0.25


def delivery_cues(pm: dict | None) -> dict:
    """Per label: prosody directions the character's reference clips share (|median z| >= 0.5, >= 2/3 agree), with a
    leave-one-out check of how often a reference meets its own label's cues vs other labels' references do."""
    if not pm or not pm.get("targets"):
        return {}

    def derive(ids):
        out = []
        for f in CUE_FEATURES:
            zs = [zscore(pm["real"][i].get(f), f, pm) for i in ids]
            zs = [z for z in zs if z is not None]
            if len(zs) < 2:
                continue
            med = float(np.median(zs))
            sign = 1 if med > 0 else -1
            share = sum(1 for z in zs if z * sign > 0) / len(zs)
            if abs(med) >= 0.5 and share >= 2 / 3:
                out.append({"feature": f, "sign": sign, "median_z": med, "share": share})
        return out

    def met_fraction(clip_id, cues):
        if not cues:
            return None
        return sum(_cue_met(zscore(pm["real"][clip_id].get(c["feature"]), c["feature"], pm), c["sign"])
                   for c in cues) / len(cues)

    out = {}
    for label, t in pm["targets"].items():
        ids = t["ids"]
        if len(ids) < 3:
            continue
        cues = derive(ids)
        own = [met_fraction(i, derive([j for j in ids if j != i])) for i in ids]
        others = [met_fraction(j, cues) for l2, t2 in pm["targets"].items() if l2 != label
                  for j in t2["ids"] if j not in ids]
        out[label] = {"cues": cues, "own_hit_rate": _mean(own), "other_hit_rate": _mean(others),
                      "n_refs": len(ids)}
    return out


# ----------------------------------------------------------------------------------------------
# Per-take flags
# ----------------------------------------------------------------------------------------------

def error_runs(reference: str, hypothesis: str) -> tuple[list[str], list[str]]:
    """Longest run of consecutive missing reference words and of consecutive extra hypothesis words."""
    ref, hyp = ec.norm_words(reference), ec.norm_words(hypothesis)
    best_del, best_ins, cur_del, cur_ins = [], [], [], []
    for i, j in ec.align_words(ref, hyp):
        if j is None:
            cur_del, cur_ins = cur_del + [ref[i]], []
        elif i is None:
            cur_ins, cur_del = cur_ins + [hyp[j]], []
        else:
            cur_del, cur_ins = [], []
        if len(cur_del) > len(best_del):
            best_del = cur_del
        if len(cur_ins) > len(best_ins):
            best_ins = cur_ins
    return best_del, best_ins


def asr_flags(text: str, asr: dict, options: dict):
    hard, soft, notes = [], [], []
    n = asr.get("ref_words") or 0
    limit = max(options["wer_gate"], 1.0 / n) if n else options["wer_gate"]
    wer_ok = asr["wer"] <= limit + 1e-9
    if not wer_ok:
        hard.append(f"WER {asr['wer']:.2f} > gate {limit:.2f} ({asr.get('word_errors')} of {n} words; "
                    f"heard: \"{asr.get('asr', '')}\")")
    missing, extra = error_runs(text, asr.get("asr", ""))
    if len(missing) >= 2:
        hard.append(f"{len(missing)} consecutive words missing (\"{' '.join(missing)}\"): truncated or skipped")
    if len(extra) >= 2:
        hard.append(f"{len(extra)} consecutive extra words (\"{' '.join(extra)}\"): hallucination/babble")
    for r in (asr.get("probe") or {}).get("results") or []:
        if not r.get("ok"):
            message = (f"name '{r['stem']}' heard as \"{r.get('heard') or '(nothing)'}\" without the ASR prompt "
                       "(mispronounced?)")
            (soft if r.get("calibrated") else notes).append(message if r.get("calibrated") else
                                                             message + "; no real clip to calibrate against")
    if (asr.get("max_compression_ratio") or 0) > 2.4:
        notes.append("ASR compression ratio > 2.4 (repetition/babble?)")
    if asr.get("avg_logprob") is not None and asr["avg_logprob"] < -1.0:
        notes.append(f"low ASR confidence (avg logprob {asr['avg_logprob']:.2f})")
    if (asr.get("max_no_speech_prob") or 0) > 0.5:
        notes.append(f"Whisper no-speech probability {asr['max_no_speech_prob']:.2f} on a segment (silence/noise?)")
    return wer_ok, hard, soft, notes


def artifact_flags(a: dict | None, artifacts: dict) -> tuple[list[str], list[str], list[str]]:
    hard, soft, notes = [], [], []
    if not a:
        return hard, soft, notes
    if "error" in a:
        hard.append(f"unreadable: {a['error']}")
        return hard, soft, notes
    base = _get(artifacts, "baselines", "train_real") or {}
    restored_hb = _get(artifacts, "baselines", "restore_report", "restored_high_band_db", "p95")
    if a.get("nonfinite_samples"):
        hard.append(f"{a['nonfinite_samples']} non-finite samples")
    if a.get("speech_detected") is False:
        hard.append("no speech detected")
    clipped = a.get("clipped_fraction") or 0.0
    if clipped > HARD_CLIP_FRACTION:
        hard.append(f"clipped {clipped:.2%} of samples")
    elif clipped > SOFT_CLIP_FRACTION or (a.get("clip_runs") or 0) > 0:
        soft.append(f"clipping ({a.get('clip_runs', 0)} runs, {clipped:.3%})")
    # Isolation (spike_ratio) separates clicks from speech for any bandwidth: Bao-Dur's real clips stay below 7.6
    # (original) / 3.5 (full-band restored). Magnitude is absolute: full-band audio reaches spike_rel_peak 1.4
    # naturally, so it only splits a full-scale wrap (hard) from a small tick (soft) once isolation is shown.
    spike, rel, at = a.get("spike_ratio"), a.get("spike_rel_peak"), a.get("spike_at_s")
    real_spike = _get(base, "spike_ratio", "max")
    if spike is not None and spike > max(CLICK_SPIKE_RATIO, 1.1 * (real_spike or 0)):
        if rel is not None and rel > WRAP_SPIKE_REL:
            hard.append(f"full-scale one-sample click at {fmt(at, 3)} s (spike {rel:.2f} x peak, ratio {spike:.0f}; "
                        "int16 wrap?)")
        else:
            soft.append(f"isolated click at {fmt(at, 3)} s (spike ratio {spike:.1f}, real max {fmt(real_spike, 1)})")
    lead, tail = a.get("lead_silence_s"), a.get("tail_silence_s")
    lead_p95, tail_p95 = _get(base, "lead_silence_s", "p95"), _get(base, "tail_silence_s", "p95")
    if lead is not None and lead_p95 is not None and lead > lead_p95 + 0.3:
        soft.append(f"long lead silence {lead:.2f}s (real p95 {lead_p95:.2f}s)")
    if tail is not None and tail_p95 is not None and tail > tail_p95 + 0.5:
        soft.append(f"long tail silence {tail:.2f}s (real p95 {tail_p95:.2f}s)")
    onset, onset_max = a.get("onset_rel_db"), _get(base, "onset_rel_db", "max")
    if onset is not None and onset_max is not None and onset > max(-6.0, onset_max + 3.0):
        soft.append(f"onset cut mid-sound (first 10 ms at {onset:.0f} dB re the loud frames)")
    offset, offset_max = a.get("offset_rel_db"), _get(base, "offset_rel_db", "max")
    if offset is not None and offset_max is not None and offset > max(-30.0, offset_max + 5.0):
        soft.append(f"ending cut mid-sound (last 10 ms at {offset:.0f} dB re the loud frames; real max {offset_max:.0f})")
    hb = a.get("high_band_db")
    hb_p95 = _get(base, "high_band_db", "p95")
    ceilings = [v for v in (hb_p95, restored_hb) if v is not None]
    if hb is not None and ceilings and hb > max(ceilings) + 6:
        soft.append(f"high-band energy {hb:.1f} dB, more than any real clip (p95 {max(ceilings):.1f} dB"
                    + (" incl. restored" if restored_hb is not None else "") + "): hiss/aliasing?")
    elif hb is not None and hb_p95 is not None and hb > hb_p95 + 6:
        notes.append(f"more top end than the reference clips ({hb:.1f} dB vs p95 {hb_p95:.1f} dB above 11.5 kHz)")
    floor, floor_p99 = a.get("floor_p05_db"), _get(base, "floor_p05_db", "p99")
    flat, flat_med = a.get("quiet_flatness"), _get(base, "quiet_flatness", "median")
    nf, nf_p95 = a.get("noise_floor_db"), _get(base, "noise_floor_db", "p95")
    if floor is not None and floor_p99 is not None and floor > floor_p99 + 3 and (flat is None or flat_med is None
                                                                                 or flat > flat_med):
        soft.append(f"steady noise floor: quietest frames at {floor:.1f} dBFS (real p99 {floor_p99:.1f}), "
                    f"noise-like (flatness {fmt(flat)}): hiss/hum?")
    elif nf is not None and nf_p95 is not None and nf > nf_p95 + 10:
        soft.append(f"noise floor in silences {nf:.1f} dBFS (real p95 {nf_p95:.1f} dBFS)")
    if a.get("rate") != 48000 or a.get("channels") != 1 or a.get("subtype") != "PCM_16":
        notes.append(f"format {a.get('rate')} Hz/{a.get('channels')} ch/{a.get('subtype')} (contract: 48 kHz mono PCM_16)")
    if abs(a.get("dc_offset") or 0.0) > 0.01:
        notes.append(f"DC offset {a['dc_offset']:.3f}")
    return hard, soft, notes


def coupled_models(take: dict) -> set[str]:
    version = str(take.get("model_version") or "").lower()
    names = " ".join([version] + [str(x).lower() for x in _as_list(take.get("conditioning_encoders"))])
    out = {m for m, keys in COUPLING.items() if any(k in names for k in keys)}
    if version.startswith(V2PRO_PREFIX):
        out.add("sv_eres2net")
    return out


# ----------------------------------------------------------------------------------------------
# Main aggregation
# ----------------------------------------------------------------------------------------------

def aggregate(run_dir: Path | str, options: dict | None = None) -> dict:
    options = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None}}
    run_dir = Path(run_dir)
    run = ec.load_run(run_dir)
    variant = run["audio_variant"]
    corpus = ec.corpus_index()
    tests = {i for i, r in corpus.items() if r["split"] == "test"}
    warnings = list(run.get("warnings", []))
    metrics, stale = load_metrics(run_dir, run, warnings)
    status_path = run_dir / "run_status.json"
    steps = ec.read_json(status_path).get("steps", []) if status_path.exists() else []
    delivery_map = None
    if run.get("delivery_map") and Path(run["delivery_map"]).exists():
        delivery_map = ec.read_json(run["delivery_map"])

    purity = speaker_purity(metrics, corpus, variant)
    exclude = set(purity["exclude"])
    if exclude:
        agree = purity["agreement"]
        warnings.append(f"{len(exclude)} train clips are other characters' lines or score like one "
                        f"({agree['confirmed']} confirmed by the delivery task, {agree['models_flagged']} flagged by the "
                        f"speaker models, {agree['both']} both); they are excluded from every real baseline here and "
                        "should be dropped from training data (see speaker_purity)")
    feats = real_features(metrics)
    pm = prosody_model(feats, exclude, delivery_map, tests, warnings) if feats else None
    sv_present = [m for m, _l in SV_METRICS if m in metrics]
    id_models = {m: {v: identity_model(metrics[m], v, feats, corpus, pm, exclude)
                     for v in (metrics[m].get("baselines") or {})} for m in sv_present}
    for m in sv_present:
        if not id_models[m].get(variant):
            warnings.append(f"{m}: no baselines for the run's {variant} audio; left out of identity")
    sv_ranked = [m for m in sv_present if id_models[m].get(variant)]
    real_ident = {m: real_identities(metrics[m], id_models[m][variant], variant, feats, corpus, pm, exclude)
                  for m in sv_ranked}
    gate_cache: dict[frozenset, dict] = {}

    def gate_for(models: list[str]) -> dict:
        key = frozenset(models)
        if key not in gate_cache:
            common = set.intersection(*[set(real_ident[m]) for m in models]) if models else set()
            values = [_mean([real_ident[m][i]["identity"] for m in models]) for i in common]
            values = [v for v in values if v is not None]
            gate_cache[key] = {"models": sorted(models), "n": len(values),
                               "threshold": float(np.percentile(values, options["identity_gate_q"]))
                               if len(values) >= 10 else None,
                               "median": float(np.median(values)) if values else None}
        return gate_cache[key]

    dn_model = dnsmos_model(metrics["dnsmos"], exclude) if "dnsmos" in metrics else None
    signals = delivery_signals(pm, metrics, warnings)
    validated = {k: s for k, s in signals.items() if s.get("validated")}
    cues = delivery_cues(pm)
    wavlm_side_takes = _get(metrics.get("sv_wavlm") or {}, "side", "takes") or {}
    extra = extra_models(metrics, exclude, tests)

    rows = []
    for take in run["takes"]:
        rows.append(score_take(take, metrics, stale, id_models, variant, sv_ranked, gate_for, pm, dn_model,
                               validated, cues, wavlm_side_takes, options, extra))
    lines = rank_lines(rows, run, metrics, pm, id_models, variant, feats, corpus, real_ident, options)
    groups = group_stats(rows, lines, sv_ranked, extra)
    engines = engine_summary(rows)
    comparisons = compare_groups(rows, lines, sv_ranked, options)
    breakdown = settings_breakdown(rows)
    memorisation = paired_gaps(rows)
    baselines = build_baselines(metrics, variant, id_models, real_ident, gate_for, sv_ranked, pm, dn_model, feats,
                                corpus, delivery_map, exclude)
    if extra:
        baselines.update(extra_baselines(metrics, extra, exclude))
    unscored = [r for r in rows if r["unscored"]]
    if unscored:
        steps_needed = sorted({reason.split(":", 1)[0] for r in unscored for reason in r["unscored"]})
        warnings.append(f"{len(unscored)} take(s) unscored and left out of the ranking; re-run with "
                        f"--metrics {','.join(steps_needed)}")
    card = {
        "run": run["run"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "reference_audio": variant,
        "device": run.get("device"), "manifests": run["manifests"], "script": run.get("script"),
        "delivery_map": run.get("delivery_map"), "options": options, "metrics_present": list(metrics),
        "method": method_text(options, validated, sv_ranked),
        "steps": steps, "baselines": baselines, "speaker_purity": purity,
        "delivery_signals": {k: {x: y for x, y in s.items() if x not in ("centroids", "center", "cosine")}
                             for k, s in signals.items()},
        "delivery_cues": cues, "training_provenance": run.get("training_provenance") or {},
        "groups": groups, "engines": engines, "comparisons": comparisons, "settings_breakdown": breakdown,
        "memorisation": memorisation, "lines": lines, "takes": rows, "warnings": warnings,
    }
    if extra:
        card["method"]["separate_columns"] = (
            f"UTMOSv2 (fold 0, naturalness MOS 1-5, percentile vs {HIS} train clips) and emotion2vec+ base (top class, "
            "match of the take's embedding to its delivery label's reference clips, paired real clip) are reported in "
            "their own columns next to identity, intelligibility (WER) and DNSMOS; they never change tiers or the "
            "composite. The emotion2vec delivery check is reported, not weighted.")
    ec.write_json(run_dir / "scorecard.json", card)
    (run_dir / "scorecard.md").write_text(render_markdown(card), encoding="utf-8")
    if purity["models"]:
        # Corpus-level finding, independent of the takes: the latest per audio variant, for data-prep tasks.
        ec.write_json(ec.EVAL_ROOT / f"speaker_purity_{variant}.json",
                      {"from_run": run["run"], "created": card["created"], "audio_variant": variant, **purity})
    return card


def method_text(options: dict, validated: dict, sv_ranked: list[str]) -> dict:
    return {
        "identity": "per model: (cos to train centroid - impostor mean) / (expected real cos for the take's speech span "
                    f"and atypicality - impostor mean), expected cos fitted on {HIS} kept train clips' leave-one-out "
                    "cosines; paired lines average in the identity vs the real recording; mean over the models the "
                    f"take's engine is not conditioned on ({', '.join(SV_LABEL[m] for m in sv_ranked) or 'none'})",
        "tiers": "0 pass; 1 identity below the gate; 2 hard failure (WER, missing/extra word runs, unreadable, "
                 "silent, NaN, clipping, full-scale click); 3 leak; unscored takes are not ranked",
        "gate": f"WER <= max({options['wer_gate']}, 1/reference_words); identity >= p{options['identity_gate_q']:g} "
                f"of {HIS} own clips measured the same way",
        "composite": f"min(identity, 1) - {options['w_prosody']} x paired prosody distance - {options['w_plausible']} x "
                     f"implausibility - {options['w_natural']} x DNSMOS deficit - {options['w_artifact']} x soft flags"
                     + (f" + {options['w_delivery']} x delivery match ({', '.join(validated)})" if validated else
                        " (no delivery term: no signal passed validation)"),
        "ties": f"tier-0 takes within {options['tie_margin']} of the line's best are ties",
        "engine_comparison": "shared lines only; expected best-of-k selection score at equal k; bootstrap 95% CI over "
                             "lines; winner needs CI excluding 0 and per-model identity differences agreeing in sign",
        "prosody_features": [{"name": n, "weight": w, "floor": f, "label": l, "unit": u}
                             for n, w, f, l, u in PROSODY_FEATURES],
        "plausibility": [{"feature": n, "weight": w, "low": lo, "high": hi} for n, w, lo, hi, *_ in PLAUSIBILITY],
    }


def score_take(take, metrics, stale, id_models, variant, sv_ranked, gate_for, pm, dn_model, validated, cues,
               wavlm_side_takes, options, extra=None) -> dict:
    key = take["key"]
    extra = extra or {}
    row = {k: take.get(k) for k in ("key", "take_id", "engine", "run_label", "line_id", "text", "delivery",
                                    "heldout_id", "paired_id", "paired_split", "reference_id", "seed", "wav",
                                    "model_version")}
    row["params"] = take.get("params") or {}
    row["group"] = take["engine"] if options["group_by"] == "engine" else take["run_label"]
    row["violations"] = list(take.get("violations", []))
    row["provenance_status"] = (take.get("provenance") or {}).get("status")
    notes = list(take.get("notes", []))
    hard, soft, prosody_flags, quality_flags = [], [], [], []

    # 0. unscored / stale entries
    unscored = []
    for name, metric in metrics.items():
        entry = _get(metric, "takes", key)
        if name in EXTRA_METRICS:  # separate columns: a gap is a note, never "unscored"
            if entry is None or (isinstance(entry, dict) and entry.get("unscored")):
                notes.append(f"{name}: no result for this take (re-run --metrics {name})")
            elif (isinstance(entry, dict) and entry.get("sha256") and take.get("wav_sha256")
                  and entry["sha256"] != take["wav_sha256"]):
                notes.append(f"{name}: scored a different version of this wav (re-run --metrics {name})")
            elif isinstance(entry, dict) and entry.get("error"):
                notes.append(f"{name}: {entry['error']}")
            continue
        if entry is None:
            unscored.append(f"{name}: no result for this take (metric output is older than the take set)")
        elif isinstance(entry, dict) and entry.get("unscored"):
            unscored.append(f"{name}: {entry.get('error', 'resource error')}")
        elif (isinstance(entry, dict) and entry.get("sha256") and take.get("wav_sha256")
              and entry["sha256"] != take["wav_sha256"]):
            unscored.append(f"{name}: scored a different version of this wav (re-rendered since)")
        elif isinstance(entry, dict) and entry.get("error"):
            # asr and artifacts errors are reported in their own sections below; a silent take fails there too
            if name.startswith("sv_") or (name == "prosody" and not str(entry["error"]).startswith("no speech")):
                hard.append(f"{name}: {entry['error']}")
            elif name == "dnsmos":
                notes.append(f"dnsmos: {entry['error']}")
    row["unscored"] = unscored

    # 1. identity
    span = take_span(key, metrics)
    pros = _get(metrics.get("prosody") or {}, "takes", key)
    pros = pros if isinstance(pros, dict) and "error" not in pros else None
    aty = atypicality(pros, pm)
    row["span_s"], row["atypicality"] = span, aty
    detail = {}
    for m in [x for x, _l in SV_METRICS if x in metrics]:
        entry = _get(metrics[m], "takes", key)
        d = {"model": SV_LABEL[m]}
        if not isinstance(entry, dict) or "error" in entry:
            d["error"] = (entry or {}).get("error", "no result") if isinstance(entry, dict) or entry is None else "bad entry"
            detail[m] = d
            continue
        e = _get(entry, "by_variant", variant) or {}
        model = id_models.get(m, {}).get(variant)
        cos = e.get("centroid_cos")
        d.update(centroid_cos=cos, heldout_cos=e.get("heldout_cos"), reference_cos=e.get("reference_cos"))
        if model:
            exp = expected_cos(model, span, aty)
            ic = anchored(cos, exp, model["impostor_mean"], model.get("heldout_sd"))
            ip = anchored(e.get("heldout_cos"), model["pair_real"], model["pair_imp"], model.get("pair_real_sd")) \
                if e.get("heldout_cos") is not None else None
            d.update(expected_real_cos=exp, identity_centroid=ic, identity_paired=ip,
                     identity=_mean([ic, ip]) if ip is not None else ic,
                     identity_constant=anchored(cos, model["heldout_mean"], model["impostor_mean"], model.get("heldout_sd")))
            others = {}
            for v2, e2 in (entry.get("by_variant") or {}).items():
                m2 = id_models.get(m, {}).get(v2)
                if v2 != variant and m2 and isinstance(e2, dict):
                    others[v2] = anchored(e2.get("centroid_cos"), expected_cos(m2, span, aty), m2["impostor_mean"])
            if others:
                d["identity_other_variants"] = others
        detail[m] = d
    coupled = coupled_models(take)
    usable = [m for m in sv_ranked if detail.get(m, {}).get("identity") is not None]
    included = [m for m in usable if m not in coupled]
    if usable and not included:
        included = usable
        notes.append("every speaker model is coupled to this engine; identity uses them anyway")
    for m in coupled & set(detail):
        detail[m]["coupled"] = True
    row["similarity_detail"] = detail
    row["identity_models"] = included
    row["identity_coupled"] = sorted(coupled & set(detail))
    row["similarity"] = _mean([detail[m]["identity"] for m in included]) if included else None
    row["identity_gate_value"] = _mean([detail[m]["identity_centroid"] for m in included]) if included else None
    gate = gate_for(included) if included else {"threshold": None}
    row["identity_gate"] = gate.get("threshold")
    identity_ok = not (row["identity_gate_value"] is not None and gate.get("threshold") is not None
                       and row["identity_gate_value"] < gate["threshold"])
    if not identity_ok:
        quality_flags.append(f"identity {row['identity_gate_value']:.2f} below the gate {gate['threshold']:.2f} "
                             f"(p{options['identity_gate_q']:g} of {HIS} own clips)")
    spread = [detail[m]["identity_centroid"] for m in included if detail[m].get("identity_centroid") is not None]
    if len(spread) >= 2 and max(spread) - min(spread) > 0.4:
        notes.append("speaker models disagree: " + ", ".join(f"{SV_LABEL[m]} {fmt(detail[m]['identity_centroid'])}"
                                                            for m in included))

    # 2. intelligibility
    asr = _get(metrics.get("asr") or {}, "takes", key)
    row["asr"] = asr if isinstance(asr, dict) else None
    wer_ok = True
    row["wer"] = None
    if isinstance(asr, dict) and "error" not in asr:
        row["wer"] = asr["wer"]
        wer_ok, h, s, n = asr_flags(take.get("text") or "", asr, options)
        hard += h
        soft += s
        notes += n
    elif "asr" in metrics and not any(u.startswith("asr") for u in unscored):
        hard.append(f"ASR failed: {(asr or {}).get('error', 'no result') if isinstance(asr, dict) else 'no result'}")

    # 3. artifacts
    art = _get(metrics.get("artifacts") or {}, "takes", key)
    if not (isinstance(art, dict) and art.get("unscored")):
        h, s, n = artifact_flags(art, metrics.get("artifacts") or {})
        hard += h
        soft += s
        notes += n
    row["artifacts"] = {k: art.get(k) for k in ("peak_dbfs", "lead_silence_s", "tail_silence_s", "spike_ratio",
                                                "spike_rel_peak", "floor_p05_db", "quiet_flatness", "high_band_db",
                                                "rate", "subtype")} if isinstance(art, dict) and "error" not in art else None

    # 4. naturalness
    dn = _get(metrics.get("dnsmos") or {}, "takes", key)
    row["dnsmos"] = {k: dn.get(k) for k in ("SIG", "BAK", "OVRL", "P808")} if isinstance(dn, dict) and "error" not in dn else None
    deficit, qflags = naturalness(dn, dn_model)
    row["naturalness_deficit"] = deficit
    quality_flags += qflags
    if "utmos" in metrics:  # separate naturalness column (UTMOSv2), not in the composite
        row["utmos"] = utmos_value(_extra_entry(metrics, "utmos", take), extra.get("utmos"))
    if "emotion" in metrics:  # separate emotion column (emotion2vec+), not in the composite
        row["emotion"] = emotion_value(_extra_entry(metrics, "emotion", take), extra.get("emotion_labels") or [])

    # 5. prosody: plausibility (all lines), match to the line's real recording (paired lines), delivery
    row["prosody"] = pros
    penalty, pflags, excess = plausibility(pros, pm)
    row["plausibility_penalty"], row["plausibility_excess"] = penalty, excess
    prosody_flags += pflags
    row["prosody_distance"] = None
    if pm and pros and take.get("paired_id") and take["paired_id"] in pm["real"]:
        d, z = prosody_distance(pros, pm["real"][take["paired_id"]], pm)
        row["prosody_distance"], row["prosody_z"] = d, z
    label = (take.get("delivery") or "").casefold()
    row["delivery_cues"] = None
    if pm and pros and label in cues and cues[label]["cues"]:
        met = []
        for c in cues[label]["cues"]:
            z = zscore(pros.get(c["feature"]), c["feature"], pm)
            met.append({"feature": c["feature"], "sign": c["sign"], "z": z, "met": _cue_met(z, c["sign"])})
        own, other = cues[label].get("own_hit_rate"), cues[label].get("other_hit_rate")
        row["delivery_cues"] = {"met": sum(m["met"] for m in met), "of": len(met), "detail": met,
                                "informative": own is not None and other is not None and own - other >= 0.2}
    scores = {}
    for name, signal in validated.items():
        if name == "prosody":
            vec = _prosody_vector(pros, pm) if pm else None
        else:
            raw = wavlm_side_takes.get(key)
            vec = ec.unit(np.asarray(raw, dtype=np.float64) - signal["center"]) if raw is not None else None
        scores[name] = delivery_score(signal, label, vec)
    row["delivery_scores"] = scores
    row["delivery_score"] = _mean(list(scores.values())) if scores else None

    # tiers, composite
    hard = list(row["violations"]) + hard
    leak = bool(row["violations"])
    if row["unscored"]:
        tier, reason = None, "unscored"
    elif leak:
        tier, reason = 3, "leak"
    elif hard or not wer_ok:
        tier, reason = 2, "hard failure"
    elif not identity_ok:
        tier, reason = 1, "identity gate"
    else:
        tier, reason = 0, "pass"
    if take.get("heldout_id") and row["provenance_status"] == "unverified":
        notes.append("paired scores unverified: this model's training data could not be resolved")
    row["flags"] = {"hard": hard, "soft": soft, "prosody": prosody_flags, "quality": quality_flags,
                    "notes": list(dict.fromkeys(notes))}
    row["tier"], row["tier_reason"] = tier, reason
    row["gate_pass"] = tier == 0
    composite = min(row["similarity"], 1.0) if row["similarity"] is not None else 0.0
    if row["prosody_distance"] is not None:
        composite -= options["w_prosody"] * row["prosody_distance"]
    if penalty:
        composite -= options["w_plausible"] * penalty
    if deficit:
        composite -= options["w_natural"] * deficit
    composite -= options["w_artifact"] * len(soft)
    if row["delivery_score"] is not None:
        composite += options["w_delivery"] * row["delivery_score"]
    row["composite"] = composite
    row["selection_score"] = None if tier in (None, 3) else composite - (1.0 if tier in (1, 2) else 0.0)
    return row


def real_reference(clip_id: str, metrics: dict, pm: dict | None, real_ident: dict, dn_metric: dict | None) -> dict:
    ref = {"id": clip_id}
    for m, ident in real_ident.items():
        if clip_id in ident:
            ref[m] = {"identity": ident[clip_id]["identity"]}
    per_clip = _get(metrics.get("asr") or {}, "baselines", "test_real", "per_clip", clip_id)
    if per_clip:
        ref["asr"] = {"wer": per_clip.get("wer"), "asr": per_clip.get("asr")}
    if pm and clip_id in pm["real"]:
        real = pm["real"][clip_id]
        ref["prosody"] = {name: real.get(name) for name, *_ in PROSODY_FEATURES}
        ref["plausibility_penalty"] = plausibility(real, pm)[0]
    dn = _get(dn_metric or {}, "real", clip_id)
    if dn:
        ref["dnsmos"] = {k: dn.get(k) for k in ("SIG", "BAK", "OVRL", "P808")}
    u = _get(metrics.get("utmos") or {}, "real", clip_id)
    if isinstance(u, dict) and u.get("mos") is not None:
        ref["utmos"] = u["mos"]
    e = _get(metrics.get("emotion") or {}, "real", clip_id)
    if isinstance(e, dict) and e.get("top"):
        ref["emotion"] = {"top": e["top"], "top_p": e.get("top_p")}
    return ref


def rank_lines(rows, run, metrics, pm, id_models, variant, feats, corpus, real_ident, options) -> list[dict]:
    by_line: dict[str, list[dict]] = {}
    for row in rows:
        by_line.setdefault(row["line_id"], []).append(row)
    near_dups = run.get("near_duplicates_of_heldout") or {}
    lines = []
    for line_id, members in by_line.items():
        scored = [r for r in members if r["tier"] is not None]
        scored.sort(key=lambda r: (r["tier"], -r["composite"]))
        for rank, r in enumerate(scored, 1):
            r["rank_in_line"] = rank
        for r in members:
            r.setdefault("rank_in_line", None)
            r["tie"] = False
        passing = [r for r in scored if r["tier"] == 0]
        best = passing[0] if passing else (scored[0] if scored else None)
        shortlist = []
        if passing:
            top = passing[0]["composite"]
            for i, r in enumerate(passing):
                if i < options["top_n"] or r["composite"] >= top - options["tie_margin"]:
                    shortlist.append(r["key"])
                    r["tie"] = r["composite"] >= top - options["tie_margin"]
        by_group: dict[str, str] = {}
        for group in sorted({r["group"] for r in scored}):
            ranked = [r for r in scored if r["group"] == group]
            for rank, r in enumerate(ranked, 1):
                r["rank_in_group"] = rank
            by_group[group] = ranked[0]["key"]
        first = members[0]
        line = {"line_id": line_id, "text": first["text"], "delivery": first["delivery"],
                "heldout_id": first["heldout_id"], "paired_id": first.get("paired_id"),
                "paired_split": first.get("paired_split"), "n_takes": len(members), "n_scored": len(scored),
                "n_pass": len(passing), "best": best["key"] if best else None,
                "best_passes_gate": bool(best and best["tier"] == 0), "shortlist": shortlist, "top_by_group": by_group,
                "ranking": [r["key"] for r in scored], "unscored": [r["key"] for r in members if r["tier"] is None]}
        if first.get("paired_id"):
            line["real_reference"] = real_reference(first["paired_id"], metrics, pm, real_ident, metrics.get("dnsmos"))
        if first["heldout_id"] and near_dups.get(first["heldout_id"]):
            line["heldout_near_duplicates"] = near_dups[first["heldout_id"]]
        lines.append(line)
    lines.sort(key=lambda l: l["line_id"])
    return lines


def group_stats(rows, lines, sv_ranked, extra=None) -> dict:
    groups = {}
    lines_by_id = {l["line_id"]: l for l in lines}
    # equal-k expected best: per line, k = the fewest scored non-leaking takes any group has on that line
    per_line_k = {}
    for line_id in lines_by_id:
        counts = {}
        for r in rows:
            if r["line_id"] == line_id and r["selection_score"] is not None:
                counts[r["group"]] = counts.get(r["group"], 0) + 1
        per_line_k[line_id] = min(counts.values()) if counts else None
    ranked_groups = {r["group"] for r in rows if r["selection_score"] is not None}
    common = [l for l in lines_by_id if per_line_k[l] is not None and
              all(any(r["line_id"] == l and r["group"] == g and r["selection_score"] is not None for r in rows)
                  for g in ranked_groups)]
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group]
        scored = [r for r in members if r["tier"] is not None]
        ebk = {}
        for line_id, k in per_line_k.items():
            values = [r["selection_score"] for r in scored if r["line_id"] == line_id and r["selection_score"] is not None]
            if values and k:
                ebk[line_id] = expected_best_of_k(values, k)
        by_model = {m: _mean([r["similarity_detail"].get(m, {}).get("identity") for r in scored]) for m in sv_ranked}
        groups[group] = {
            "takes": len(members), "scored": len(scored), "unscored": len(members) - len(scored),
            "lines": len({r["line_id"] for r in scored}),
            "pass_rate": (sum(r["tier"] == 0 for r in scored) / len(scored)) if scored else None,
            "tiers": {str(t): sum(r["tier"] == t for r in scored) for t in (0, 1, 2, 3)},
            "identity_mean": _mean([r["similarity"] for r in scored if r["tier"] != 3]),
            "identity_by_model": by_model,
            "wer_mean": _mean([r["wer"] for r in scored]),
            "dnsmos_ovrl_mean": _mean([(r["dnsmos"] or {}).get("OVRL") for r in scored]),
            "plausibility_mean": _mean([r["plausibility_penalty"] for r in scored]),
            "soft_flag_rate": (sum(bool(r["flags"]["soft"]) for r in scored) / len(scored)) if scored else None,
            "expected_best_equal_k": _mean(list(ebk.values())),
            "expected_best_common_lines": _mean([ebk[l] for l in common if l in ebk]) if common else None,
            "common_lines": len(common),
            "lines_won_equal_k": 0,
        }
        if extra and "utmos" in extra:
            groups[group]["utmos_mean"] = _mean([(r.get("utmos") or {}).get("mos") for r in scored])
            groups[group]["utmos_pct_real_mean"] = _mean([(r.get("utmos") or {}).get("pct_real") for r in scored])
        if extra and "emotion_labels" in extra:
            groups[group]["emotion_match_mean"] = _mean([((r.get("emotion") or {}).get("ref") or {}).get("match")
                                                         for r in scored])
            groups[group]["emotion_ref_cos_mean"] = _mean([((r.get("emotion") or {}).get("ref") or {}).get("centroid_cos")
                                                           for r in scored])
    for line_id, k in per_line_k.items():
        if not k:
            continue
        best_group, best_value = None, None
        for group in groups:
            values = [r["selection_score"] for r in rows if r["group"] == group and r["line_id"] == line_id
                      and r["selection_score"] is not None]
            if values:
                v = expected_best_of_k(values, k)
                if best_value is None or v > best_value:
                    best_group, best_value = group, v
        if best_group:
            groups[best_group]["lines_won_equal_k"] += 1
    return groups


def engine_summary(rows) -> dict:
    out = {}
    for engine in sorted({r["engine"] for r in rows}):
        members = [r for r in rows if r["engine"] == engine and r["tier"] is not None]
        out[engine] = {"runs": sorted({r["run_label"] for r in rows if r["engine"] == engine}),
                       "takes": len(members),
                       "pass_rate": (sum(r["tier"] == 0 for r in members) / len(members)) if members else None,
                       "identity_mean": _mean([r["similarity"] for r in members if r["tier"] != 3])}
    return out


def compare_groups(rows, lines, sv_ranked, options) -> dict:
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["selection_score"] is not None:
            cells.setdefault((r["line_id"], r["group"]), []).append(r)
    groups = sorted({g for _l, g in cells})
    line_ids = [l["line_id"] for l in lines]
    pairs = []
    if len(groups) > 10:  # keep the table readable: compare everything with the group of best equal-k score
        best = max(groups, key=lambda g: _mean([expected_best_of_k([r["selection_score"] for r in cells[(l, g)]], 1)
                                               for l in line_ids if (l, g) in cells]) or -9)
        candidates = [(best, g) for g in groups if g != best]
    else:
        candidates = list(itertools.combinations(groups, 2))
    for a, b in candidates:
        shared = [l for l in line_ids if (l, a) in cells and (l, b) in cells]
        entry = {"a": a, "b": b, "shared_lines": len(shared)}
        if len(shared) < 3:
            entry["verdict"] = "too few shared lines"
            pairs.append(entry)
            continue
        diffs, by_model = [], {m: [] for m in sv_ranked}
        for l in shared:
            ra, rb = cells[(l, a)], cells[(l, b)]
            k = min(len(ra), len(rb))
            diffs.append(expected_best_of_k([r["selection_score"] for r in ra], k)
                         - expected_best_of_k([r["selection_score"] for r in rb], k))
            for m in sv_ranked:
                ia = _mean([r["similarity_detail"].get(m, {}).get("identity") for r in ra])
                ib = _mean([r["similarity_detail"].get(m, {}).get("identity") for r in rb])
                if ia is not None and ib is not None:
                    by_model[m].append(ia - ib)
        mean = float(np.mean(diffs))
        lo, hi = bootstrap_ci(diffs, options["bootstrap"])
        model_means = {m: (float(np.mean(v)) if v else None) for m, v in by_model.items()}
        signs = {np.sign(v) for v in model_means.values() if v is not None and abs(v) > 1e-9}
        agree = len(signs) <= 1 and (not signs or next(iter(signs)) == np.sign(mean))
        if lo > 0 and agree:
            verdict = f"{a} better"
        elif hi < 0 and agree:
            verdict = f"{b} better"
        elif (lo > 0 or hi < 0) and not agree:
            verdict = "composite differs but speaker models disagree: listen"
        else:
            verdict = "tie (CI includes 0)"
        entry.update(mean_diff=mean, ci95=[lo, hi], identity_diff_by_model=model_means, models_agree=agree,
                     verdict=verdict)
        pairs.append(entry)
    return {"rule": "a - b per shared line: expected best-of-k selection score (composite, -1 for gate failures; "
                    "leaks and unscored takes excluded), k = the smaller take count; bootstrap 95% CI over lines",
            "pairs": pairs}


def settings_breakdown(rows) -> dict:
    """Per group: params (and reference_id) with 2-12 distinct values, each used by >= 2 takes."""
    out = {}
    skip = ("seconds", "rtf", "duration", "peak", "rms", "trim", "text", "why", "note", "reason", "wav", "audio",
            "path", "sha", "time", "created", "instruction", "device")
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group and r["tier"] is not None and r["tier"] != 3]
        if len(members) < 4:
            continue
        keys = {"reference_id"} | {k for r in members for k, v in r["params"].items()
                                   if isinstance(v, (str, int, float, bool)) and not any(s in k.lower() for s in skip)}
        table = {}
        for k in sorted(keys):
            values: dict[str, list[dict]] = {}
            for r in members:
                v = r.get("reference_id") if k == "reference_id" else r["params"].get(k)
                if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 60:
                    values.setdefault(str(v), []).append(r)
            if not (2 <= len(values) <= 12) or any(len(v) < 2 for v in values.values()):
                continue
            table[k] = {v: {"n": len(rs), "pass_rate": sum(r["tier"] == 0 for r in rs) / len(rs),
                            "composite_mean": _mean([r["composite"] for r in rs]),
                            "identity_mean": _mean([r["similarity"] for r in rs])}
                        for v, rs in sorted(values.items())}
        if table:
            out[group] = table
    return out


def paired_gaps(rows) -> dict:
    """Per group: paired identity / prosody distance / WER on test (held-out) vs train/val diagnostic lines."""
    out = {}
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group and r["tier"] is not None and r["tier"] != 3
                   and r.get("paired_split")]
        if not members:
            continue
        by_split = {}
        for split in ("test", "val", "train"):
            rs = [r for r in members if r["paired_split"] == split]
            if rs:
                by_split[split] = {
                    "takes": len(rs),
                    "paired_identity": _mean([_mean([r["similarity_detail"][m].get("identity_paired")
                                                     for m in r["identity_models"]]) for r in rs]),
                    "prosody_distance": _mean([r["prosody_distance"] for r in rs]),
                    "wer": _mean([r["wer"] for r in rs])}
        entry = {"by_split": by_split}
        if "train" in by_split and "test" in by_split:
            t, h = by_split["train"], by_split["test"]
            entry["train_minus_test"] = {k: (t[k] - h[k]) if t[k] is not None and h[k] is not None else None
                                         for k in ("paired_identity", "prosody_distance", "wer")}
        out[group] = entry
    return out


def build_baselines(metrics, variant, id_models, real_ident, gate_for, sv_ranked, pm, dn_model, feats, corpus,
                    delivery_map, exclude) -> dict:
    baselines = {}
    for m in sv_ranked:
        b = _get(metrics[m], "baselines", variant) or {}
        model = id_models[m][variant]
        ident = real_ident[m]
        train = [v for v in ident.values() if v["split"] == "train"]
        held = [v for v in ident.values() if v["split"] in ("val", "test")]
        bias = {"n_train": len(train),
                "spearman_constant_vs_span": spearman([v["identity_constant"] for v in train], [v["span"] for v in train]),
                "spearman_adjusted_vs_span": spearman([v["identity"] for v in train], [v["span"] for v in train]),
                "spearman_constant_vs_atypicality": spearman([v["identity_constant"] for v in train], [v["aty"] for v in train]),
                "spearman_adjusted_vs_atypicality": spearman([v["identity"] for v in train], [v["aty"] for v in train])}
        for key, field in (("adjusted_median_by_span", "identity"), ("constant_median_by_span", "identity_constant")):
            bins = []
            for lo, hi in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 12)):
                values = [v[field] for v in train if v["span"] is not None and lo <= v["span"] < hi
                          and v[field] is not None]
                if len(values) >= 10:
                    bins.append({"span_s": f"{lo}-{hi}", "n": len(values), "median": float(np.median(values))})
            bias[key] = bins
            bias[key.replace("median_by_span", "span_spread")] = (max(x["median"] for x in bins) -
                                                                  min(x["median"] for x in bins)) if bins else None
        by_label = {}
        for label, spec in ((delivery_map or {}).get("deliveries") or {}).items():
            ids = [r.get("id") for r in spec.get("references", []) if r.get("id") in ident]
            if ids:
                by_label[label.casefold()] = {"n": len(ids),
                                              "constant_anchor": _median([ident[i]["identity_constant"] for i in ids]),
                                              "adjusted": _median([ident[i]["identity"] for i in ids]),
                                              "adjusted_min": min(ident[i]["identity"] for i in ids
                                                                  if ident[i]["identity"] is not None)
                                              if any(ident[i]["identity"] is not None for i in ids) else None}
        baselines[m] = {
            "model": SV_LABEL[m], "variant": variant,
            "centroid_train_clips": b.get("centroid_train_clips"), "train_suspects": len(b.get("train_suspects") or []),
            "test_vs_centroid": {k: v for k, v in (b.get("test_vs_centroid") or {}).items()},
            "val_vs_centroid": {k: v for k, v in (b.get("val_vs_centroid") or {}).items() if k != "per_clip"},
            "train_leave_one_out": {k: v for k, v in (b.get("train_leave_one_out") or {}).items()},
            "impostor_vs_centroid": b.get("impostor_vs_centroid"), "test_pairwise": b.get("test_pairwise"),
            "impostor_vs_test_pairwise": b.get("impostor_vs_test_pairwise"),
            "identity_model": {k: v for k, v in model.items()},
            "real_identity_heldout": ec.stats([v["identity"] for v in held]),
            "real_identity_train": ec.stats([v["identity"] for v in train]),
            "real_identity_per_clip": {i: v["identity"] for i, v in ident.items() if v["split"] in ("val", "test")},
            "bias_check": bias, "identity_by_delivery_label": by_label,
            "gate_alone": gate_for([m]),
        }
    if sv_ranked:
        baselines["identity_gate_all_models"] = gate_for(sv_ranked)
        if "sv_eres2net" in sv_ranked and len(sv_ranked) > 1:
            baselines["identity_gate_without_eres2net"] = gate_for([m for m in sv_ranked if m != "sv_eres2net"])
    if "asr" in metrics:
        baselines["asr"] = {k: v for k, v in (metrics["asr"].get("baselines") or {}).items()}
        baselines["asr"]["device"] = metrics["asr"].get("device")
        baselines["asr"]["fell_back"] = metrics["asr"].get("fell_back")
    if pm:
        heldout = [i for i, r in pm["real"].items() if r.get("split") in ("val", "test")]
        baselines["prosody"] = {
            "train_center": pm["center"], "train_scale": pm["scale"], "ranges_train_2s": pm["ranges"],
            "n_train": pm["n_train"], "n_train_2s": pm["n_long"], "atypicality_cap_p95": pm.get("aty_cap"),
            "heldout_real_implausibility": ec.stats([plausibility(pm["real"][i], pm)[0] for i in heldout]),
            "heldout_real_flagged": sum(1 for i in heldout if plausibility(pm["real"][i], pm)[1]),
            "n_heldout": len(heldout)}
    if dn_model:
        baselines["dnsmos"] = dn_model
    if "artifacts" in metrics:
        baselines["artifacts"] = metrics["artifacts"].get("baselines")
        real = metrics["artifacts"].get("real") or {}
        fired = {}
        for clip_id, a in real.items():
            if clip_id in exclude:
                continue
            h, s, _n = artifact_flags(a, metrics["artifacts"])
            if h or s:
                fired[clip_id] = h + s
        baselines["artifacts_flags_on_real_train"] = {"clips": len(real), "flagged": len(fired),
                                                      "examples": dict(list(fired.items())[:5])}
    return baselines


def extra_baselines(metrics: dict, extra: dict, exclude: set[str]) -> dict:
    """Real anchors of the opt-in metrics for the card (only the ones that ran)."""
    out = {}
    if "utmos" in metrics:
        m = metrics["utmos"]
        model = extra.get("utmos")
        out["utmos"] = {"settings": m.get("settings"), "device": m.get("device"), "real_scope": m.get("real_scope"),
                        "train_real": {k: v for k, v in model.items() if k != "values"} if model else None,
                        "note": None if model else "fewer than 10 real train clips scored (--real none?): takes "
                                                   "are shown without a percentile."}
    if "emotion" in metrics:
        m = metrics["emotion"]
        real = m.get("real") or {}
        by_label = {}
        for label, ids in (m.get("reference_labels") or {}).items():
            tops: dict[str, int] = {}
            for i in ids:
                top = _get(real, i, "top")
                if top and i not in exclude:
                    tops[top] = tops.get(top, 0) + 1
            by_label[label] = dict(sorted(tops.items(), key=lambda kv: -kv[1]))
        signal = extra.get("emotion_signal") or {}
        out["emotion"] = {"settings": m.get("settings"), "device": m.get("device"),
                          "references_scope": m.get("references_scope"), "center": m.get("center"),
                          "reference_top_classes": by_label,
                          "signal": {k: v for k, v in signal.items() if k not in ("centroids", "center", "cosine")}}
    return out


# ----------------------------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------------------------

def _short(path: str | None) -> str:
    if not path:
        return "-"
    try:
        return str(Path(path).resolve().relative_to(ec.LAB.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _flags_text(row: dict) -> str:
    f = row["flags"]
    parts = [f"FAIL: {h}" for h in f["hard"]] + f["soft"] + [f"prosody: {p}" for p in f["prosody"]] + f["quality"]
    return _cell("; ".join(parts)) if parts else "-"


def _identity_text(row: dict) -> str:
    detail = row["similarity_detail"]
    parts = []
    for m, _l in SV_METRICS:
        d = detail.get(m)
        if not d or d.get("identity") is None:
            continue
        text = f"{SV_SHORT[m]} {fmt(d['identity'])}"
        if m not in row["identity_models"]:
            text += " excl."
        parts.append(text)
    paired = [detail[m].get("identity_paired") for m in row["identity_models"] if detail.get(m)]
    paired = _mean(paired)
    extra = f"; pair {fmt(paired)}" if paired is not None else ""
    if paired is not None and row.get("heldout_id") and row.get("provenance_status") == "unverified":
        extra += " (unverified)"
    return f"{fmt(row['similarity'])} ({', '.join(parts)}{extra})" if parts else fmt(row["similarity"])


def _cues_text(row: dict) -> str:
    c = row.get("delivery_cues")
    if not c:
        return "-"
    arrows = []
    for d in c["detail"]:
        label = FEATURE_LABEL.get(d["feature"], (d["feature"], ""))[0]
        arrows.append(f"{label} {'up' if d['sign'] > 0 else 'down'} {'ok' if d['met'] else 'no'}")
    return f"{c['met']}/{c['of']} ({', '.join(arrows)})" + ("" if c.get("informative") else "; weak cues")


def _utmos_text(row: dict) -> str:
    u = row.get("utmos")
    if not u:
        return "-"
    return fmt(u["mos"]) + (f" (p{u['pct_real']:.0f})" if u.get("pct_real") is not None else "")


def _emotion_text(row: dict, short: bool = False) -> str:
    e = row.get("emotion")
    if not e:
        return "-"
    text = f"{e['top']} {fmt(e.get('top_p'))}"
    ref = e.get("ref")
    if ref and ref.get("match") is not None:
        text += f"; {ref['label']} match {fmt(ref['match'])}" + ("" if short else f" (rank {ref['rank']}/{ref['of']})")
    elif ref:
        text += f"; {ref['label']} refs cos {fmt(ref.get('mean_cos'))}"
    p = e.get("paired")
    if p and not short:
        text += f"; real {p.get('real_top')} (cos {fmt(p.get('cos'))})"
    return text


def _short_flags(row: dict, limit: int = 3) -> str:
    f = row["flags"]
    parts = ([f"FAIL {h.split(' (')[0][:70]}" for h in f["hard"]] + [s.split(" (")[0] for s in f["soft"]]
             + [p.split(":")[0] for p in f["prosody"]] + [q.split(" (")[0] for q in f["quality"]])
    parts = list(dict.fromkeys(parts))
    return _cell("; ".join(parts[:limit]) + (f"; +{len(parts) - limit}" if len(parts) > limit else "")) if parts else "-"


TIER_NAMES = {0: "pass", 1: "identity", 2: "fail", 3: "leak", None: "unscored"}


def render_markdown(card: dict) -> str:
    rows = {r["key"]: r for r in card["takes"]}
    o = card["options"]
    b = card["baselines"]
    out = [f"# {ec.NAME} take scorecard: {card['run']}", ""]
    out.append(f"Generated {card['created']}. Reference audio: **{card['reference_audio']}** real clips. "
               f"{len(card['takes'])} takes, {len(card['lines'])} lines, {len(card['groups'])} groups "
               f"({o['group_by']}), metrics: {', '.join(card['metrics_present']) or 'none'}. Manifests: "
               + ", ".join(f"`{_short(m)}`" for m in card["manifests"]) + ".")
    out.append("")
    sv = [m for m, _l in SV_METRICS if m in b]
    signals = card.get("delivery_signals") or {}
    validated = [k for k, s in signals.items() if s.get("validated")]
    out += ["## How to read this", "",
            "Your ears pick the take. This card removes broken takes, measures identity fairly and lists the takes "
            "worth hearing per line (the **shortlist**).", "",
            f"- **tier**: `pass` = every gate passed; `identity` = sounds less like {ec.NAME} than "
            f"{100 - o['identity_gate_q']:g} % of {HIS} own clips (measured the same way); `fail` = hard failure (WER "
            f"above max({o['wer_gate']:.2f}, one word), 2+ consecutive missing/extra words, silent/unreadable audio, "
            "heavy clipping, a full-scale one-sample click); `leak` = held-out data used as conditioning or in training. "
            "Tiers rank in that order; unscored takes are not ranked.",
            f"- **identity**: 1.00 = as close to {HIS} train centroid as {HIS} own clips of the same length and "
            f"expressiveness, 0.00 = {ec.IMPOSTOR_LABEL}. Mean over the speaker models the engine is "
            f"not conditioned on ({', '.join(SV_LABEL[m] for m in sv) or 'none present'}); `excl.` marks a model "
            "coupled to the engine (GPT-SoVITS v2Pro/v2ProPlus are conditioned on ERes2NetV2). Paired lines add the "
            "identity vs the real recording (`pair`). The ranking caps identity at 1.",
            f"- **composite** = min(identity, 1) - {o['w_prosody']} x prosody distance to the line's real recording "
            f"(paired lines) - {o['w_plausible']} x implausibility (robust sd outside {HIS} own range, one-sided: "
            f"monotone, flat, too fast/slow; expressive but in-range costs nothing) - {o['w_natural']} x DNSMOS deficit "
            f"(below {HIS} clips' p05) - {o['w_artifact']} x soft flags"
            + (f" + {o['w_delivery']} x delivery match ({', '.join(validated)})." if validated else
               ". No delivery term: no delivery signal passed leave-one-out validation (see below), so emotion is "
               "left to your ears; per-label prosody cues are shown as hints only."),
            f"- **ties**: passing takes within {o['tie_margin']} of a line's best are marked `tie`: listen to all.", ""]
    has_utmos = "utmos" in card["metrics_present"]
    has_emotion = "emotion" in card["metrics_present"]
    if has_utmos or has_emotion:
        out[-1:-1] = ["- **separate columns** (never in tiers or the composite): "
                      + "; ".join(([f"**UTMOS** = UTMOSv2 naturalness MOS 1-5 (fold 0, sarulab domain, mean of 5 random "
                                    f"crops); `pN` = the share of {HIS} own train clips scoring at or below it"]
                                   if has_utmos else [])
                                  + (["**emotion** = emotion2vec+ base top class and probability; `match` = rank of "
                                      "the take's delivery label among all labels' reference centroids (1 = nearest); "
                                      "`real` = the paired real recording's class"] if has_emotion else []))
                      + ". Identity, intelligibility (WER), naturalness and emotion stay separate."]
    xh = (" UTMOS |" if has_utmos else "") + (" emotion |" if has_emotion else "")
    xd = "---|" * (int(has_utmos) + int(has_emotion))

    def xcells(r: dict, short: bool = False) -> str:
        return ((f" {_utmos_text(r)} |" if has_utmos else "")
                + (f" {_cell(_emotion_text(r, short))} |" if has_emotion else ""))

    # shortlist
    out += ["## Listening shortlist", "",
            "Per line: passing takes in rank order (ties and the top "
            f"{o['top_n']}). Lines without a passing take show their best take with its tier.", "",
            "| line | delivery | take (group) | composite | identity | WER |" + xh + " flags | delivery cues | wav |",
            "|---|---|---|---|---|---|" + xd + "---|---|---|"]
    for line in card["lines"]:
        keys = line["shortlist"] or ([line["best"]] if line["best"] else [])
        for n, key in enumerate(keys):
            r = rows[key]
            tag = (f"{line['line_id']}" + (f" (held-out {line['heldout_id']})" if line["heldout_id"] else "")) if n == 0 else ""
            mark = " tie" if r.get("tie") and n > 0 else ""
            tier = "" if r["tier"] == 0 else f" [{TIER_NAMES[r['tier']]}]"
            out.append(f"| {tag} | {line['delivery'] if n == 0 else ''} | {r['take_id']} ({_cell(r['group'])}){mark}{tier} | "
                       f"{fmt(r['composite'])} | {fmt(r['similarity'])} | {fmt(r['wer'])} |{xcells(r, True)} "
                       f"{_short_flags(r)} | {_cell(_cues_text(r))} | `{_short(r['wav'])}` |")
        if not keys:
            out.append(f"| {line['line_id']} | {line['delivery']} | (no scored take) | | | |" + " |" * (xd.count("|"))
                       + " | | |")
    out.append("")

    # groups
    groups = card["groups"]
    out += ["## Groups", "",
            "`best (equal k)` = mean over lines of the expected best selection score when every group gets the same "
            "number of takes per line (the fewest any group rendered); it removes the advantage of rendering more "
            "candidates. Selection score = composite, minus 1 for gate failures.", "",
            "| group | takes (unscored) | lines | pass | identity | " + " | ".join(SV_LABEL[m] for m in sv)
            + " | WER | DNSMOS OVRL |" + (" UTMOS (pN) |" if has_utmos else "")
            + (" emotion match |" if has_emotion else "")
            + " implausibility | soft flags | best (equal k) | lines won (equal k) |",
            "|---|---|---|---|---|" + "---|" * len(sv) + "---|---|" + xd + "---|---|---|---|"]
    for g, s in groups.items():
        gx = ((f" {fmt(s.get('utmos_mean'))}" + (f" (p{s['utmos_pct_real_mean']:.0f})"
                                                  if s.get("utmos_pct_real_mean") is not None else "") + " |")
              if has_utmos else "") + (f" {fmt(s.get('emotion_match_mean'))} |" if has_emotion else "")
        out.append(f"| {_cell(g)} | {s['takes']} ({s['unscored']}) | {s['lines']} | "
                   f"{fmt(100 * s['pass_rate'], 0) + '%' if s['pass_rate'] is not None else '-'} | "
                   f"{fmt(s['identity_mean'])} | " + " | ".join(fmt(s["identity_by_model"].get(m)) for m in sv)
                   + f" | {fmt(s['wer_mean'])} | {fmt(s['dnsmos_ovrl_mean'])} |{gx} {fmt(s['plausibility_mean'])} | "
                   f"{fmt(100 * s['soft_flag_rate'], 0) + '%' if s['soft_flag_rate'] is not None else '-'} | "
                   f"{fmt(s['expected_best_equal_k'])} | {s['lines_won_equal_k']} |")
    out.append("")
    if card.get("engines") and o["group_by"] == "run":
        out += ["Engines (all runs pooled): " + "; ".join(
            f"{e} {s['takes']} takes in {len(s['runs'])} run(s), pass {fmt(100 * s['pass_rate'], 0) if s['pass_rate'] is not None else '-'}%, "
            f"identity {fmt(s['identity_mean'])}" for e, s in card["engines"].items()) + ".", ""]
    pairs = card["comparisons"]["pairs"]
    if pairs:
        out += ["### Paired comparisons (shared lines only)", "", card["comparisons"]["rule"] + ".", "",
                "| a | b | shared lines | mean a - b | 95% CI | identity a - b by model | verdict |",
                "|---|---|---|---|---|---|---|"]
        for p in pairs:
            models = ", ".join(f"{SV_SHORT[m]} {fmt(v, 2, True)}" for m, v in (p.get("identity_diff_by_model") or {}).items()
                               if v is not None)
            ci = p.get("ci95")
            out.append(f"| {_cell(p['a'])} | {_cell(p['b'])} | {p['shared_lines']} | {fmt(p.get('mean_diff'), 3, True)} | "
                       + (f"[{fmt(ci[0], 3, True)}, {fmt(ci[1], 3, True)}]" if ci else "-")
                       + f" | {models or '-'} | {p.get('verdict')} |")
        out.append("")
    if card.get("settings_breakdown"):
        out += ["### Settings breakdown", "",
                "Per group, parameters that vary across its takes (each value used by at least 2 takes):", ""]
        for g, table in card["settings_breakdown"].items():
            out.append(f"- **{_cell(g)}**: " + "; ".join(
                f"`{k}` " + ", ".join(f"{v} (n {s['n']}, pass {s['pass_rate']:.0%}, composite {fmt(s['composite_mean'])}, "
                                      f"identity {fmt(s['identity_mean'])})" for v, s in values.items())
                for k, values in table.items()))
        out.append("")
    if card.get("memorisation"):
        out += ["### Paired scores by line type (memorisation check)", "",
                "| group | split | takes | paired identity | prosody distance | WER |", "|---|---|---|---|---|---|"]
        for g, e in card["memorisation"].items():
            for split, s in e["by_split"].items():
                out.append(f"| {_cell(g)} | {split} | {s['takes']} | {fmt(s['paired_identity'])} | "
                           f"{fmt(s['prosody_distance'])} | {fmt(s['wer'])} |")
            if e.get("train_minus_test"):
                t = e["train_minus_test"]
                out.append(f"| {_cell(g)} | train - test | | {fmt(t['paired_identity'], 2, True)} | "
                           f"{fmt(t['prosody_distance'], 2, True)} | {fmt(t['wer'], 2, True)} |")
        out += ["", "A large train - test gap in paired identity means the model reproduces lines it trained on "
                "better than unseen ones (memorisation).", ""]

    # top take per line per group
    names = list(groups)
    out += ["## Top take per line per group", "",
            "Cell: take (composite / identity / WER). **Bold** = best for the line; ~~struck~~ = not tier 0.", "",
            "| line | delivery | " + " | ".join(_cell(n) for n in names) + " |", "|---|---|" + "---|" * len(names)]
    for line in card["lines"]:
        cells = []
        for g in names:
            key = line["top_by_group"].get(g)
            if not key:
                cells.append("-")
                continue
            r = rows[key]
            cell = f"{r['take_id']} ({fmt(r['composite'])} / {fmt(r['similarity'])} / {fmt(r['wer'])})"
            if r["tier"] != 0:
                cell = f"~~{cell}~~ {TIER_NAMES[r['tier']]}"
            if key == line["best"]:
                cell = f"**{cell}**"
            cells.append(_cell(cell))
        tag = f"{line['line_id']}" + (f" (held-out {line['heldout_id']})" if line["heldout_id"] else "")
        out.append(f"| {tag} | {line['delivery']} | " + " | ".join(cells) + " |")
    out.append("")

    # lines
    out += ["## Lines", ""]
    for line in card["lines"]:
        out += [f"### {line['line_id']} ({line['delivery']})", "", f"> {line['text']}", ""]
        ref = line.get("real_reference")
        if ref:
            parts = [f"{'held-out' if line['heldout_id'] else line.get('paired_split') + ' diagnostic'} real clip "
                     f"`{ref['id']}`"]
            for m in sv:
                if m in ref:
                    parts.append(f"{SV_LABEL[m]} identity {fmt(ref[m].get('identity'))}")
            if "asr" in ref:
                parts.append(f"real WER {fmt(ref['asr'].get('wer'))}")
            if "dnsmos" in ref:
                parts.append(f"DNSMOS SIG/OVRL {fmt(ref['dnsmos'].get('SIG'))}/{fmt(ref['dnsmos'].get('OVRL'))}")
            if "utmos" in ref:
                parts.append(f"UTMOS {fmt(ref['utmos'])}")
            if "emotion" in ref:
                parts.append(f"emotion {ref['emotion'].get('top')} {fmt(ref['emotion'].get('top_p'))}")
            if "prosody" in ref:
                pr = ref["prosody"]
                parts.append(f"F0 median {fmt(pr.get('f0_median_st'), 1)} st, IQR {fmt(pr.get('f0_iqr_st'), 1)} st, "
                             f"rate {fmt(pr.get('rate_syl_s'), 1)} syl/s, pauses {fmt(pr.get('pause_ratio'), 2)}")
            out += ["Real reference: " + "; ".join(parts) + ".", ""]
        if line.get("heldout_near_duplicates"):
            out += ["Caution: the corpus repeats this held-out line in "
                    + ", ".join(f"`{i}`" for i in line["heldout_near_duplicates"])
                    + ". A model trained on those clips has heard this line (its takes here are leaks), and prompting "
                    "with them on this line is a leak too.", ""]
        out += ["| rank | tier | group | take | composite | identity (models; pair) | WER | heard | DNSMOS SIG/OVRL |"
                + xh + " implausibility | paired prosody d | delivery cues | flags | wav |",
                "|---|---|---|---|---|---|---|---|---|" + xd + "---|---|---|---|---|"]
        shown = [rows[k] for k in line["ranking"]
                 if rows[k]["rank_in_line"] <= o["top_n"] or rows[k].get("rank_in_group") == 1 or rows[k].get("tie")]
        for r in shown:
            heard = (r["asr"] or {}).get("asr", "-") if isinstance(r["asr"], dict) else "-"
            best = " (best)" if r["key"] == line["best"] else (" tie" if r.get("tie") else "")
            dn = r.get("dnsmos") or {}
            out.append(f"| {r['rank_in_line']}{best} | {TIER_NAMES[r['tier']]} | {_cell(r['group'])} | {r['take_id']} | "
                       f"{fmt(r['composite'])} | {_cell(_identity_text(r))} | {fmt(r['wer'])} | {_cell(heard)} | "
                       f"{fmt(dn.get('SIG'))}/{fmt(dn.get('OVRL'))} |{xcells(r)} {fmt(r['plausibility_penalty'])} | "
                       f"{fmt(r['prosody_distance'])} | {_cell(_cues_text(r))} | {_flags_text(r)} | `{_short(r['wav'])}` |")
        notes = [f"{r['take_id']}: " + "; ".join(r["flags"]["notes"]) for r in shown if r["flags"]["notes"]]
        if notes:
            out += ["", "Notes: " + _cell(" | ".join(notes))]
        if line["unscored"]:
            out += ["", "Unscored (re-run the step): " + "; ".join(
                f"{rows[k]['take_id']} ({'; '.join(rows[k]['unscored'])})" for k in line["unscored"])]
        out.append("")

    # real baselines
    out += ["## Real-vs-real baselines", ""]
    if sv:
        out += ["| model | centroid clips | test vs centroid mean (min) | impostors vs centroid (max) | identity fit | "
                f"{HIS} held-out clips: identity median (p05, min) | gate alone | Spearman identity vs span / atypicality "
                "(constant anchor -> fitted) | spread of span-bin medians |", "|---|---|---|---|---|---|---|---|---|"]
        for m in sv:
            x = b[m]
            t, imp = x["test_vs_centroid"], x.get("impostor_vs_centroid") or {}
            im = x["identity_model"]
            fit = (f"{im.get('kind')}: " + ", ".join(fmt(c, 3, True) for c in im.get("coef", [])) +
                   f" (n {im.get('n', '-')}, resid sd {fmt(im.get('resid_sd'), 3)})") if im.get("coef") else im.get("kind")
            h = x["real_identity_heldout"]
            bias = x["bias_check"]
            out.append(f"| {x['model']} | {x.get('centroid_train_clips')} | {fmt(t.get('mean'), 3)} ({fmt(t.get('min'), 3)}) | "
                       f"{fmt(imp.get('mean'), 3)} ({fmt(imp.get('max'), 3)}) | {fit} | {fmt(h.get('median'))} "
                       f"({fmt(h.get('p05'))}, {fmt(h.get('min'))}) | {fmt(x['gate_alone'].get('threshold'))} | "
                       f"{fmt(bias['spearman_constant_vs_span'], 2, True)} -> {fmt(bias['spearman_adjusted_vs_span'], 2, True)} / "
                       f"{fmt(bias['spearman_constant_vs_atypicality'], 2, True)} -> "
                       f"{fmt(bias['spearman_adjusted_vs_atypicality'], 2, True)} | "
                       f"{fmt(bias.get('constant_span_spread'))} -> {fmt(bias.get('adjusted_span_spread'))} |")
        out.append("")
        g_all = b.get("identity_gate_all_models") or {}
        g_wo = b.get("identity_gate_without_eres2net")
        out.append(f"Identity gate (p{o['identity_gate_q']:g} of {HIS} own clips, mean over models): "
                   f"{fmt(g_all.get('threshold'))} with all models (n {g_all.get('n')})"
                   + (f", {fmt(g_wo.get('threshold'))} without ERes2NetV2 (GPT-SoVITS v2Pro takes)" if g_wo else "") + ".")
        out.append("")
        labels = sorted({l for m in sv for l in b[m]["identity_by_delivery_label"]})
        if labels:
            out += ["Expressiveness check, median identity of each delivery label's reference clips "
                    "(constant anchor -> fitted anchor); the fitted values should sit near 1 for every label:", "",
                    "| model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
            for m in sv:
                d = b[m]["identity_by_delivery_label"]
                out.append(f"| {SV_LABEL[m]} | " + " | ".join(
                    f"{fmt(d[l]['constant_anchor'])} -> {fmt(d[l]['adjusted'])}" if l in d else "-" for l in labels) + " |")
            out.append("")
        ids = sorted(b[sv[0]].get("real_identity_per_clip") or {})
        tests = [i for i in ids if i in (b[sv[0]]["test_vs_centroid"].get("per_clip") or {})]
        if tests:
            out += ["Held-out test clips (identity under the fitted anchor; real WER):", "",
                    "| test clip | " + " | ".join(SV_LABEL[m] for m in sv) + " | real WER |", "|---|" + "---|" * (len(sv) + 1)]
            for i in tests:
                wer = _get(b, "asr", "test_real", "per_clip", i, "wer")
                out.append(f"| {i} | " + " | ".join(fmt(b[m]['real_identity_per_clip'].get(i)) for m in sv) + f" | {fmt(wer)} |")
            out.append("")
    if "asr" in b:
        t = b["asr"].get("test_real") or {}
        out.append(f"ASR ({b['asr'].get('device') or '?'}{', fell back to CPU: ' + str(b['asr']['fell_back']) if b['asr'].get('fell_back') else ''}): "
                   f"real held-out WER mean {fmt(t.get('mean'), 3)}, max {fmt(t.get('max'), 3)} over {t.get('n', 0)} clips.")
        out.append("")
    if "prosody" in b:
        p = b["prosody"]
        out += [f"Prosody range of {HIS} train clips >= 2 s (implausibility bounds): " + "; ".join(
            f"{FEATURE_LABEL[n][0]} {fmt(r['p01'])}/{fmt(r['p05'])}-{fmt(r['p95'])}/{fmt(r['p99'])}"
            for n, r in p["ranges_train_2s"].items()) + " (p01/p05 - p95/p99). "
            f"{HIS.capitalize()} held-out clips: implausibility median {fmt(p['heldout_real_implausibility'].get('median'))}, "
            f"{p['heldout_real_flagged']} of {p['n_heldout']} get a prosody flag.", ""]
    if "dnsmos" in b:
        out += [f"DNSMOS of {HIS} train clips: " + "; ".join(
            f"{k} median {fmt(v['median'])} (p05 {fmt(v['p05'])}, n {v['n']})" for k, v in b["dnsmos"].items()) + ".", ""]
    if "utmos" in b:
        u = b["utmos"]
        t = u.get("train_real")
        st = u.get("settings") or {}
        out += [f"UTMOSv2 ({st.get('config')} fold {st.get('fold')}, domain {st.get('predict_dataset')}, "
                f"{st.get('repetitions')} repetitions, {u.get('device')}) of {HIS} train clips: "
                + (f"median {fmt(t['median'])} (p05 {fmt(t['p05'])}, p95 {fmt(t['p95'])}, n {t['n']})." if t
                   else str(u.get("note"))), ""]
    if "emotion" in b:
        e = b["emotion"]
        sig = e.get("signal") or {}
        tops = "; ".join(f"{label} " + ", ".join(f"{c} {n}" for c, n in counts.items())
                         for label, counts in (e.get("reference_top_classes") or {}).items() if counts)
        out += [f"emotion2vec+ base on the delivery references ({(e.get('center') or {}).get('n_refs', 0)} clips, "
                f"{'centred' if (e.get('center') or {}).get('centered') else 'raw'} cosines): top class per label: "
                + (tops or "-") + ". Leave-one-out label recognition from the embedding: "
                + (f"accuracy {fmt(sig.get('accuracy'))} vs chance {fmt(sig.get('chance'))} (n {sig.get('n')}, "
                   f"p {fmt(sig.get('p_value'), 4)}), {'passes' if sig.get('validated') else 'fails'} the delivery-signal "
                   "rule; reported only, never weighted." if sig.get("n") else f"not tested ({sig.get('reason')})."), ""]
    if "artifacts" in b and b["artifacts"]:
        a = b["artifacts"].get("train_real") or {}
        fired = b.get("artifacts_flags_on_real_train") or {}
        out += [f"Real train clips: lead silence p95 {fmt(_get(a, 'lead_silence_s', 'p95'))} s, tail p95 "
                f"{fmt(_get(a, 'tail_silence_s', 'p95'))} s, high band >11.5 kHz p95 {fmt(_get(a, 'high_band_db', 'p95'), 1)} dB, "
                f"quietest-frame floor p99 {fmt(_get(a, 'floor_p05_db', 'p99'), 1)} dBFS, spike ratio max "
                f"{fmt(_get(a, 'spike_ratio', 'max'), 1)}. Artifact flags fire on {fired.get('flagged', '-')} of "
                f"{fired.get('clips', '-')} real train clips.", ""]

    # delivery
    if signals or card.get("delivery_cues"):
        out += ["## Delivery signals", "",
                f"A delivery signal earns composite weight only if it recognises the label of {ec.NAME}'s own reference "
                "clips in a leave-one-out test (rule below).", "",
                "| signal | refs | accuracy | chance | p | validated |", "|---|---|---|---|---|---|"]
        for name, s in signals.items():
            out.append(f"| {name} ({s.get('metric')}) | {s.get('n', '-')} | {fmt(s.get('accuracy'))} | {fmt(s.get('chance'))} | "
                       f"{fmt(s.get('p_value'), 4)} | {'yes' if s.get('validated') else 'no' + (': ' + s['reason'] if s.get('reason') else '')} |")
        rule = next((s.get("rule") for s in signals.values() if s.get("rule")), None)
        out += ["", f"Rule: {rule}." if rule else "", ""]
        cues = card.get("delivery_cues") or {}
        if cues:
            out += [f"Per-label prosody cues (directions {HIS} reference clips share; hints for listening, not scored). "
                    "`own` = how often a reference meets its own label's cues (derived without it), `others` = how "
                    "often other labels' references meet them:", "",
                    "| label | cues | own | others |", "|---|---|---|---|"]
            for label, c in cues.items():
                text = ", ".join(f"{FEATURE_LABEL[x['feature']][0]} {'up' if x['sign'] > 0 else 'down'} "
                                 f"(z {fmt(x['median_z'], 1, True)})" for x in c["cues"]) or "none consistent"
                out.append(f"| {label} | {text} | {fmt(c['own_hit_rate'])} | {fmt(c['other_hit_rate'])} |")
            out.append("")

    purity = card.get("speaker_purity") or {}
    if purity.get("exclude"):
        likely, possible = purity["likely_other_speaker"], purity["possible_other_speaker"]
        agree = purity.get("agreement") or {}
        out += ["## Training-set speaker purity", "",
                f"{len(purity['exclude'])} train clips are left out of every real baseline: {agree.get('confirmed', 0)} "
                f"confirmed as {ec.other_speakers_label() or 'other characters'} by the delivery task"
                + (f" (`{_short(purity.get('confirmed_source'))}`)" if purity.get("confirmed_source") else "")
                + f", {len(likely)} clips >= {SHORT_CLIP_S} s the speaker models score like another character and "
                f"{len(possible)} short clips that possibly are ({agree.get('both', 0)} flagged clips are also confirmed). "
                "Rule: " + purity["rule"] + ". Drop them from TTS/VC training data.", ""]
        if agree.get("models_only"):
            out += ["Flagged by the speaker models only (listen before dropping): "
                    + ", ".join(f"`{i}`" for i in agree["models_only"]) + ".", ""]
        if agree.get("confirmed_only"):
            ident = {e["id"]: e.get("identity") for e in purity.get("confirmed_other_speaker") or []}
            out += ["Confirmed by the delivery task but above the speaker-model thresholds (identity in brackets): "
                    + ", ".join(f"`{i}` ({fmt(ident.get(i))})" for i in agree["confirmed_only"]) + ".", ""]

    prov = card.get("training_provenance") or {}
    if prov:
        out += ["## Training provenance per run", "",
                "| run | status | clips trained on | test clips in training | near-duplicates in training | "
                "other-speaker clips in training | source |", "|---|---|---|---|---|---|---|"]
        for label, p in prov.items():
            src = "; ".join(p.get("sources") or []) or "; ".join(p.get("notes") or []) or "-"
            out.append(f"| {_cell(label)} | {p.get('status')} | {p.get('clips_trained_on')} | "
                       f"{', '.join(p.get('test_clips_in_training') or []) or '-'} | "
                       f"{', '.join(p.get('near_duplicates_in_training') or []) or '-'} | "
                       f"{len(p.get('other_speaker_clips_in_training') or [])} | {_cell(src)[:160]} |")
        out += ["", "`unverified` = the harness could not resolve what the model trained on; its held-out scores are "
                "marked unverified, not clean.", ""]

    if card["steps"]:
        out += ["## Steps", "", "| step | device | exit | seconds | GPU guard | log |", "|---|---|---|---|---|---|"]
        for s in card["steps"]:
            guard = s.get("gpu_guard")
            if isinstance(guard, dict):
                guard = guard.get("killed") or (f"free {guard.get('free_at_start_gb')} GB at start, min "
                                                f"{guard.get('min_free_gb', '-')} GB")
            out.append(f"| {s['metric']} | {s.get('device', '-')} | {s['exit_code']} | {s['seconds']} | "
                       f"{_cell(guard or '-')} | `{_short(s['log'])}` |")
        out.append("")
    if card["warnings"]:
        out += ["## Warnings", ""] + [f"- {_cell(w)}" for w in card["warnings"]] + [""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    add_options(parser)
    args = parser.parse_args()
    card = aggregate(args.run_dir, options_from_args(args))
    ec.log(f"wrote {Path(args.run_dir) / 'scorecard.json'} and scorecard.md ({len(card['takes'])} takes)")


if __name__ == "__main__":
    main()
