"""Self-test for choosing a Seed-VC checkpoint: val clips and other voices, never a held-out test clip.

The character comes from VOICE_LAB_CHARACTER (voice-lab/characters/<slug>.json, default baodur); <lab> is its lab
folder. Character content comes from <lab>/seedvc/settings.json: impostor_dirs (the other voices; Bao-Dur: Atton +
Carth), promptsens_labels (Bao-Dur: calm, angry, sad), neutral_label and the pYIN range (f0).
Source sets (selftest.py sources, CPU) -> <lab>/seedvc/selftest/sources/<set>/manifest.json
  up4, down3   every val speech clip pitch/formant-shifted +4 / -3 st (librosa resampling + phase vocoder): the
               character's own words and timing in a "different" voice; ground truth = the untouched val clip. Each
               take carries the delivery label under which the delivery map lists its clip (e.g. gblbaodur237 ->
               troubled).
  ood          12 lines per impostor voice (RvcDatasets/*Decoded, 2.5-8 s; never trained on): other voices, words
               and phrasing - the production situation (a TTS take or a human read). No ground truth: scored
               on identity, WER against the source's own transcript (ASR of the source), how much of the source's
               pitch range and contour survive, and high-band noise.
  promptsens   6 down3 val clips, each converted under the promptsens_labels prompts (three contrasting deliveries):
               the spread of output mean F0 and range across the three prompts shows whether a checkpoint still lets
               the prompt carry register and dynamics (emotion control), or whether fine-tuning collapsed it. Skipped
               (with a hint) for a character whose settings name no labels yet.
  tts          once the engines have rendered them: up to 2 takes per engine of each 'new' line of takes_script.json
               (never a held-out line), i.e. the real production input. Scored like ood, with WER against the text.
Conversions (GPU): convert.py ... --ref-select rank --f0-report off --out-root <lab>/seedvc/selftest/converted
(rank: the val sources are pitch-shifted on purpose, so their pitch must not steer the prompt).
Identity and WER (GPU, recommended): eval/run_eval.py on the source and converted manifests (ERes2NetV2 + Resemblyzer,
each anchored 0 = impostors, 1 = the character's real held-out clips; faster-whisper with its allocator
bound and GPU watchdog), read back with --eval-run <name>. Without it: Resemblyzer anchored the same way locally
(level-normalised first, as the eval does) and --asr cpu|cuda through asr_helper.py (bounded allocator + watchdog,
the character's asr_prompt).

Metrics per converted take (22.05 kHz, frame-aligned: timbre-mode conversion keeps the timing)
  val:  identity, spk_centroid/spk_clip (Resemblyzer cosines), mcd (c1-c24 of an 80-band mel cepstrum, active
        frames), f0_rmse_st, f0_corr, f0_median_err_st, f0_range_dev (mean |log2| of the p10-p90 range ratio vs the
        ground truth: 0 = his dynamics, 1 = half or double), voicing_agree, dur_ratio, hf_tilt_dev (4-10.5 kHz vs
        0-4 kHz energy minus the ground truth's, dB: residual diffusion noise / hiss), wer (vs the transcript)
  ood:  identity, f0_corr_src, f0_range_dev_src (vs the source), abs_median_shift_st, hf_tilt_dev (vs the median of
        his real val clips), wer_src (vs the ASR of the source)
  tts:  as ood, with wer against the line's text
  promptsens: spread_mean_st, spread_range_st (max - min across the three prompts, mean over clips; mean rather than
        median F0 because pYIN's 0.2 st bins quantise a median)

Selection (score --train-run RUN): candidates = pretrained + RUN's raw and EMA checkpoints rendered with the
self-test defaults. Gates, each relative to pretrained on the same takes (MCD and F0 correlation are reported but
not used: a 3-step, noisier render scores BETTER on both than a 30-step one, 3.79 vs 4.03 dB and 0.58 vs 0.46):
  WER (val; ood vs source ASR; tts)            <= pretrained + 0.02
  hf_tilt_dev (val; ood; tts)                  <= pretrained + 1.0 dB
  f0_range_dev (val; ood and tts vs source)    <= pretrained + 0.10
  prompt spread (mean F0; range)               >= 0.6 x pretrained (when pretrained's spread is >= 0.3 st)
Among the candidates that pass (pretrained always does): highest identity (mean of val, ood, tts); candidates within
0.005 of the top go to the earliest step, so pretrained wins unless fine-tuning clearly helps. Writes
runs/RUN/selection.json (best, beats_pretrained, shortlist = best two fine-tuned + pretrained, whether the best is
the last step) and a blind listening pack (selftest/listening/RUN_<time>/: ground truth, source, pretrained and the
top checkpoints per line under random letters, key in _key.json). `selftest.py confirm --train-run RUN --tag TAG`
records the pick by ear; convert.py best:/shortlist: prefer it over the proxy.
A/B mode (score --baseline-tag TAG): the same gates relative to TAG for every other tag in the scored set (e.g. the
CFG/steps sweep, --ref-select variants, a content-perturbation run scored with --subsets ood).
Settings that differ from the self-test defaults (rank, timbre mode, 30 steps, CFG 0.7/0.7, length 1, no suffix)
are appended to the tag (e.g. cfm_original_s01000+scfg=0.5), so variants never mix with the plain checkpoints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.hide_cuda()  # this process runs Resemblyzer and pYIN on the CPU; the ASR subprocess gets its own device

SOURCES = C.SELFTEST / "sources"
CONVERTED = C.SELFTEST / "converted"
SCORES = C.SELFTEST / "scores"
CACHE = C.SELFTEST / "cache"
LISTENING = C.SELFTEST / "listening"
VOX_PYTHON = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local")) / "VoxStudio" / "engines" / \
    "voxcpm2" / ".venv" / "Scripts" / "python.exe"
CT2_CACHE_CONFIG = "4,3,12,209715200"   # as eval/run_eval.py: keep at most 200 MB of freed CTranslate2 blocks
FEATURE_VERSION = 2                      # bump when a cached feature's definition changes
EMBED_LEVEL_DB = -20.0                   # Resemblyzer is level-sensitive: every wave is levelled to this first
SELFTEST_DEFAULTS = {"reference_select": "rank", "diffusion_steps": 30, "intelligibility_cfg_rate": 0.7,
                     "similarity_cfg_rate": 0.7, "length_adjust": 1.0, "run_suffix": ""}
SHORT = {"reference_select": "ref", "diffusion_steps": "steps", "intelligibility_cfg_rate": "icfg",
         "similarity_cfg_rate": "scfg", "length_adjust": "len", "run_suffix": "sfx"}
GATES = {"wer": 0.02, "hf_tilt_db": 1.0, "f0_range_dev": 0.10, "spread_ratio": 0.6, "spread_min_st": 0.3,
         "tie": 0.005}


def parse_args():
    p = argparse.ArgumentParser(description="Seed-VC checkpoint self-test (val clips + other voices)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--variants", default="up4:4,down3:-3", help="name:semitones,...")
    s.add_argument("--sets", default="val,ood,promptsens,tts", help="which manifests to write")
    s.add_argument("--tts-per-line", type=int, default=2, help="tts set: takes per (new line, engine)")
    s.add_argument("--ood-per-speaker", type=int, default=12)
    s.add_argument("--ood-min-sec", type=float, default=2.5)
    s.add_argument("--ood-max-sec", type=float, default=8.0)
    s.add_argument("--promptsens-clips", type=int, default=6)
    labels = C.SETTINGS["promptsens_labels"]
    s.add_argument("--promptsens-labels", default=",".join(labels) if labels else None,
                   help=f"comma-separated delivery labels (default: promptsens_labels in {C.SETTINGS_PATH})")
    s.add_argument("--promptsens-variant", default="down3")
    s.add_argument("--out", type=Path, default=SOURCES)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--rerender", action="store_true", help="re-render WAVs that already exist")
    c = sub.add_parser("score")
    c.add_argument("--train-run", action="append", default=[],
                   help="write runs/<run>/selection.json (gated) for this run; repeatable")
    c.add_argument("--baseline-tag", default=None,
                   help="A/B mode: gates relative to this tag (or best:<run> = that run's selected tag), no selection.json")
    c.add_argument("--converted-root", type=Path, default=CONVERTED)
    c.add_argument("--sources-root", type=Path, default=SOURCES)
    c.add_argument("--runs", default=None, help="regex on converted run folder names to score (default all)")
    c.add_argument("--subsets", default="val,ood,promptsens,tts")
    c.add_argument("--name", default=None, help="scores file name (default: train run, baseline tag or 'all')")
    c.add_argument("--runs-root", type=Path, default=C.RUNS)
    c.add_argument("--scores-dir", type=Path, default=SCORES)
    c.add_argument("--no-f0", action="store_true", help="skip pYIN (no pitch metrics, gates or prompt spread)")
    c.add_argument("--eval-run", default=None, help=f"{C.LAB.name}/eval/<name> (run_eval.py on these manifests): "
                                                     "identity = its anchored similarity, WER/ASR from its asr step")
    c.add_argument("--asr", choices=["off", "cpu", "cuda"], default="off",
                   help="WER via asr_helper.py when no --eval-run (cuda: ~4-5 GB, bounded allocator + watchdog)")
    c.add_argument("--asr-need-gb", type=float, default=5.0)
    c.add_argument("--listen-lines", type=int, default=6)
    c.add_argument("--listen-ood", type=int, default=4)
    c.add_argument("--no-listening-pack", action="store_true")
    c.add_argument("--listening-root", type=Path, default=LISTENING)
    c.add_argument("--plain-diffusion-steps", type=int, default=SELFTEST_DEFAULTS["diffusion_steps"],
                   help="diffusion steps of the plain self-test conversions (other step counts become +steps= tags)")
    k = sub.add_parser("confirm", help="record the checkpoint chosen by listening")
    k.add_argument("--train-run", required=True)
    k.add_argument("--tag", required=True, help="a candidate tag from selection.json (or 'pretrained')")
    k.add_argument("--note", default="")
    k.add_argument("--runs-root", type=Path, default=C.RUNS)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- sources

def delivery_labels() -> dict:
    """val clip id -> first delivery label whose reference list contains it (map order)."""
    if not C.DELIVERY_MAP.exists():
        return {}
    labels = {}
    for label, entry in C.read_json(C.DELIVERY_MAP)["deliveries"].items():
        for ref in entry.get("references") or []:
            labels.setdefault(ref.get("id"), label)
    return labels


def map_labels() -> list[str]:
    """Delivery labels that have references in the character's delivery map (map order)."""
    if not C.DELIVERY_MAP.exists():
        return []
    return [label for label, entry in C.read_json(C.DELIVERY_MAP)["deliveries"].items() if entry.get("references")]


def make_sources(args) -> int:
    import librosa
    rows = C.load_corpus()
    excluded = C.speaker_exclusions(required=True, rows=rows)
    val_rows = [r for r in rows if r["split"] == "val" and r["kind"] == "speech"
                and r["id"] not in excluded["confirmed"] and r["id"] not in excluded["uncertain"]
                and r["id"] not in excluded["near_test"]]
    val_rows = val_rows[: args.limit or None]
    labels = delivery_labels()
    sets = {s.strip() for s in args.sets.split(",") if s.strip()}
    root = args.out.resolve()
    shifted: dict[str, dict] = {}
    for spec in args.variants.split(","):
        name, _, steps = spec.partition(":")
        steps = float(steps)
        out = root / name
        takes = []
        for row in val_rows:
            path = out / f"{row['id']}_{name}.wav"
            if args.rerender or not path.exists():
                wave, sr = C.load_mono(row["audio"])
                wave = librosa.effects.pitch_shift(wave, sr=sr, n_steps=steps, res_type="soxr_hq")
                peak = float(abs(wave).max())
                if peak > 0.97:
                    wave = wave * (0.97 / peak)
                C.write_pcm16(path, wave, sr)
            shifted.setdefault(name, {})[row["id"]] = (path, steps)
            takes.append({"take_id": f"{row['id']}_{name}", "engine": f"selftest_{name}", "line_id": row["id"],
                          "text": row["text"], "delivery": labels.get(row["id"], C.NEUTRAL_LABEL), "wav": str(path),
                          "reference_id": None, "seed": None,
                          "params": {"source_clip_id": row["id"], "pitch_shift_semitones": steps,
                                     "delivery_from": "delivery_map" if row["id"] in labels
                                     else f"default {C.NEUTRAL_LABEL}",
                                     "method": "librosa.effects.pitch_shift (formants shift with pitch)"}})
        if "val" in sets:
            C.write_json(out / "manifest.json", {"takes": takes})
            print(f"wrote {out / 'manifest.json'} ({len(takes)} takes; deliveries "
                  f"{sorted({t['delivery'] for t in takes})})")
    if "ood" in sets:
        clips = C.impostor_clips(per_dir=args.ood_per_speaker, min_s=args.ood_min_sec, max_s=args.ood_max_sec,
                                 offset=0.5)   # half a step off evalcommon's impostor grid
        takes = [{"take_id": f"ood_{c['speaker']}_{c['stem']}", "engine": "selftest_ood",
                  "line_id": f"ood_{c['speaker']}_{c['stem']}", "text": "", "delivery": C.NEUTRAL_LABEL,
                  "wav": c["path"], "reference_id": None, "seed": None,
                  "params": {"speaker": c["speaker"], "seconds": c["seconds"],
                             "method": f"{C.IMPOSTOR_VOICE} ({Path(c['path']).parent.parent.name}), never trained on; "
                                       "text unknown, WER is measured against the ASR of this source"}} for c in clips]
        C.write_json(root / "ood" / "manifest.json", {"takes": takes})
        print(f"wrote {root / 'ood' / 'manifest.json'} ({len(takes)} takes: "
              f"{sorted({t['params']['speaker'] for t in takes})})")
    if "tts" in sets:
        make_tts_sources(root, args.tts_per_line)
    if "promptsens" in sets and not args.promptsens_labels:
        available = map_labels()
        print(f"promptsens: skipped, no prompt labels for {C.NAME}: set \"promptsens_labels\" (three contrasting "
              f"delivery labels) in {C.SETTINGS_PATH} or pass --promptsens-labels; "
              + (f"{C.DELIVERY_MAP.name} labels: {available}" if available else f"no {C.DELIVERY_MAP} yet"))
    elif "promptsens" in sets:
        base = shifted.get(args.promptsens_variant)
        if not base:
            raise SystemExit(f"--promptsens-variant {args.promptsens_variant} is not among --variants")
        pool = [r for r in val_rows if r["duration"] >= 2.5 and r["id"] in base]
        step = max(1, len(pool) // max(1, args.promptsens_clips))
        picked = pool[::step][: args.promptsens_clips]
        prompt_labels = [s.strip() for s in args.promptsens_labels.split(",") if s.strip()]
        available = map_labels()
        missing = [label for label in prompt_labels if label not in available]
        if missing:
            print(f"WARNING: promptsens labels {missing} have no references in {C.DELIVERY_MAP}; convert.py renders "
                  f"them with the {C.NEUTRAL_LABEL!r} prompt, so the prompt-spread gate loses meaning")
        takes = []
        for row in picked:
            path, steps = base[row["id"]]
            for label in prompt_labels:
                takes.append({"take_id": f"{row['id']}_{args.promptsens_variant}_ps_{label}",
                              "engine": "selftest_promptsens", "line_id": row["id"], "text": row["text"],
                              "delivery": label, "wav": str(path), "reference_id": None, "seed": None,
                              "params": {"source_clip_id": row["id"], "pitch_shift_semitones": steps,
                                         "prompt_label": label}})
        C.write_json(root / "promptsens" / "manifest.json", {"takes": takes})
        print(f"wrote {root / 'promptsens' / 'manifest.json'} ({len(takes)} takes: {[r['id'] for r in picked]} x "
              f"{prompt_labels})")
    return 0


def make_tts_sources(root: Path, per_line: int) -> None:
    """Engine takes of the takes script's 'new' lines (never held-out lines) as a production-like source set."""
    script = C.LAB / "delivery" / "takes_script.json"
    new_lines = {str(line["id"]): line for line in (C.read_json(script).get("lines", []) if script.exists() else [])
                 if line.get("source") == "new"}
    takes, per_key = [], {}
    for manifest in sorted(C.TAKES.glob("*/*/manifest.json")):
        if manifest.parent.parent.name.startswith("seedvc_from_"):
            continue
        for take in C.read_json(manifest).get("takes", []):
            line = new_lines.get(str(take.get("line_id")))
            if line is None or take.get("heldout_id") or (take.get("params") or {}).get("heldout_id"):
                continue
            key = (line["id"], take.get("engine"))
            if per_key.get(key, 0) >= per_line:
                continue
            wav = Path(str(take["wav"]))
            wav = wav if wav.is_absolute() else (manifest.parent / wav).resolve()
            if not wav.exists():
                continue
            per_key[key] = per_key.get(key, 0) + 1
            take_id = C.safe_name(f"tts_{manifest.parent.parent.name}_{manifest.parent.name}_{take['take_id']}", 150)
            takes.append({"take_id": take_id, "engine": "selftest_tts", "line_id": line["id"],
                          "text": take.get("text") or line["text"], "delivery": take.get("delivery") or line.get("delivery"),
                          "wav": str(wav), "reference_id": take.get("reference_id"), "seed": None,
                          "params": {"from_manifest": str(manifest), "from_take_id": take.get("take_id"),
                                     "from_engine": take.get("engine")}})
    if not takes:
        print(f"tts: no engine takes of the 'new' lines yet under {C.LAB.name}/takes; rerun 'sources --sets tts' later")
        return
    out = root / "tts" / "manifest.json"
    C.write_json(out, {"takes": takes})
    print(f"wrote {out} ({len(takes)} takes of {len({t['line_id'] for t in takes})} new lines from "
          f"{sorted({t['params']['from_engine'] for t in takes})})")


# ------------------------------------------------------------------------------------------- features

def cepstra(wave22):
    import librosa
    import numpy as np
    from scipy.fft import dct
    mel = librosa.feature.melspectrogram(y=wave22, sr=C.MODEL_SR, n_fft=1024, hop_length=256, n_mels=80,
                                         fmin=0, fmax=C.MODEL_SR / 2, center=True)
    # Conventional cepstral scale, ln|X| = c0 + 2 sum c_d cos(d w): c_d = DCT-II_ortho(ln amplitude) / sqrt(2N).
    cep = dct(0.5 * np.log(mel + 1e-10), type=2, axis=0, norm="ortho") / np.sqrt(2 * mel.shape[0])
    return cep[1:25], 10 * np.log10(mel.sum(axis=0) + 1e-10)


class Features:
    """Per-WAV features cached by content hash (selftest/cache/features_v<N>/<sha256>.npz)."""

    def __init__(self, use_f0: bool):
        C.shim_pkg_resources()
        from resemblyzer import VoiceEncoder, preprocess_wav
        self.encoder = VoiceEncoder("cpu", verbose=False)
        self.preprocess = preprocess_wav
        self.use_f0 = use_f0
        self.dir = CACHE / f"features_v{FEATURE_VERSION}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.memo: dict = {}
        self.computed = 0

    def embed(self, wave22):
        level = C.active_rms_db(wave22, C.MODEL_SR)
        wave = wave22 * (10 ** ((EMBED_LEVEL_DB - level) / 20))
        return self.encoder.embed_utterance(self.preprocess(wave, source_sr=C.MODEL_SR))

    def get(self, path, f0: bool | None = None) -> dict:
        import numpy as np
        want_f0 = self.use_f0 if f0 is None else (f0 and self.use_f0)
        key = C.norm_path(path)
        hit = self.memo.get(key)
        if hit is not None and (not want_f0 or "f0" in hit):
            return hit
        sha = C.sha256_file(path)
        cache_path = self.dir / f"{sha}.npz"
        data = {}
        if cache_path.exists():
            with np.load(cache_path) as z:
                data = {k: z[k] for k in z.files}
        if not data or (want_f0 and "f0" not in data):
            wave = C.load_mono(path, C.MODEL_SR)[0]
            if not data:
                cep, energy = cepstra(wave)
                tilt = C.hf_tilt_db(wave)
                data = {"emb": self.embed(wave), "cep": cep, "energy": energy, "n": np.array(len(wave)),
                        "tilt": np.array(np.nan if tilt is None else tilt)}
            if want_f0:
                data["f0"], data["voiced"] = C.f0_track(wave)
            tmp = cache_path.with_name(f"{sha}.{os.getpid()}.tmp.npz")
            np.savez(tmp, **data)
            os.replace(tmp, cache_path)
            self.computed += 1
        out = {k: (v.item() if np.ndim(v) == 0 else v) for k, v in data.items()}
        self.memo[key] = out
        return out


def unit(v):
    import numpy as np
    return v / (np.linalg.norm(v) + 1e-12)


def compare_val(gt: dict, hyp: dict) -> dict:
    """Converted take vs the untouched val clip (frame-aligned)."""
    import numpy as np
    n = min(gt["cep"].shape[1], hyp["cep"].shape[1])
    active = gt["energy"][:n] > gt["energy"][:n].max() - 40
    diff = gt["cep"][:, :n] - hyp["cep"][:, :n]
    mcd = (10 / math.log(10)) * np.sqrt(2 * (diff ** 2).sum(axis=0))
    out = {"spk_clip": float(np.dot(unit(hyp["emb"]), unit(gt["emb"]))), "mcd": float(mcd[active].mean()),
           "dur_ratio": hyp["n"] / gt["n"]}
    if np.isfinite(hyp["tilt"]) and np.isfinite(gt["tilt"]):
        out["hf_tilt_dev"] = float(hyp["tilt"] - gt["tilt"])
    if "f0" in gt and "f0" in hyp:
        pair = C.f0_pair((gt["f0"], gt["voiced"]), (hyp["f0"], hyp["voiced"]))
        m = min(len(gt["f0"]), len(hyp["f0"]))
        rv, hv = gt["voiced"][:m].astype(bool), hyp["voiced"][:m].astype(bool)
        out["voicing_agree"] = float((rv == hv).mean())
        both = rv & hv & np.isfinite(gt["f0"][:m]) & np.isfinite(hyp["f0"][:m])
        if both.sum() >= 5:
            a, b = np.log2(gt["f0"][:m][both]), np.log2(hyp["f0"][:m][both])
            out["f0_rmse_st"] = float(np.sqrt(np.mean((12 * (b - a)) ** 2)))
        if pair["f0_corr"] is not None:
            out["f0_corr"] = pair["f0_corr"]
        if pair["median_shift_st"] is not None:
            out["f0_median_err_st"] = pair["median_shift_st"]
            out["f0_range_ratio"] = pair["range_ratio"]
    return out


def compare_src(src: dict, hyp: dict, val_tilt: float | None) -> dict:
    """Converted take vs its own source (out-of-domain: no ground truth)."""
    import numpy as np
    out = {"dur_ratio": hyp["n"] / src["n"]}
    if np.isfinite(hyp["tilt"]) and val_tilt is not None:
        out["hf_tilt_dev"] = float(hyp["tilt"] - val_tilt)
    if "f0" in src and "f0" in hyp:
        pair = C.f0_pair((src["f0"], src["voiced"]), (hyp["f0"], hyp["voiced"]))
        for key, name in (("f0_corr", "f0_corr_src"), ("range_ratio", "f0_range_ratio_src"),
                          ("median_shift_st", "median_shift_st")):
            if pair[key] is not None:
                out[name] = pair[key]
    return out


def train_centroid(feats, rows, excluded) -> tuple:
    """Resemblyzer centroid of the character's screened train speech (cached per id list and feature version)."""
    import numpy as np
    train = [r for r in rows if r["split"] == "train" and r["kind"] == "speech" and r["duration"] >= 1.0
             and r["id"] not in excluded["confirmed"] and r["id"] not in excluded["uncertain"]]
    ids = sorted(r["id"] for r in train)
    key = hashlib.sha256((",".join(ids) + f"|v{FEATURE_VERSION}").encode()).hexdigest()[:12]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"resemblyzer_train_centroid_{key}.npy"
    if path.exists():
        centroid = np.load(path)
    else:
        centroid = np.stack([unit(feats.get(r["audio"], f0=False)["emb"]) for r in train]).mean(axis=0)
        np.save(path, centroid)
        C.write_json(path.with_suffix(".json"), {"ids": ids, "clips": len(ids), "feature_version": FEATURE_VERSION,
                                                 "excluded_by": excluded["path"]})
    return unit(centroid), str(path)


def local_anchors(feats, centroid, val_rows) -> dict:
    """Anchors for local identity, as the eval: 0 = impostor voices (C.IMPOSTOR_DIRS), 1 = the character's real val
    clips."""
    import numpy as np
    imps = C.impostor_clips(per_dir=15)
    imp = [float(np.dot(unit(feats.get(c["path"], f0=False)["emb"]), centroid)) for c in imps]
    val = [float(np.dot(unit(feats.get(r["audio"], f0=False)["emb"]), centroid)) for r in val_rows]
    out = {"impostor_mean": float(np.mean(imp)) if imp else None, "val_mean": float(np.mean(val)) if val else None,
           "n_impostors": len(imp), "n_val": len(val)}
    out["usable"] = bool(out["impostor_mean"] is not None and out["val_mean"] is not None
                         and out["val_mean"] - out["impostor_mean"] > 0.02)
    return out


# ----------------------------------------------------------------------------------------------- ASR

def gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return min(float(v) for v in out.split()) / 1024
    except Exception:
        return None


def watchdog(process, reserve_gb: float, stop: threading.Event, guard: dict) -> None:
    """Kill our own ASR child (never anything else) if the GPU's free memory falls below the reserve."""
    while not stop.wait(0.5):
        free = gpu_free_gb()
        if free is None:
            continue
        guard["min_free_gb"] = round(min(free, guard.get("min_free_gb", free)), 2)
        if free < reserve_gb and process.poll() is None:
            guard["killed"] = f"killed asr_helper pid {process.pid}: GPU free {free:.1f} GB < reserve {reserve_gb:.0f} GB"
            process.kill()
            return


def run_asr(items: list[dict], device: str, need_gb: float) -> dict:
    """wav path -> {"wer", "hyp"} via asr_helper.py under the VoxCPM2 python (cached by sha256/model/compute type)."""
    if device == "cuda":
        free = gpu_free_gb()
        if free is None or free - need_gb < C.PAINTER_RESERVE_GB:
            raise SystemExit(f"ERROR: --asr cuda needs {need_gb} GB + {C.PAINTER_RESERVE_GB:.0f} GB Painter reserve; "
                             f"nvidia-smi reports {free} GB free. Use --asr cpu, --eval-run, or wait.")
    if not VOX_PYTHON.exists():
        raise SystemExit(f"ERROR: VoxCPM2 python not found at {VOX_PYTHON}")
    CACHE.mkdir(parents=True, exist_ok=True)
    items_path = CACHE / "asr_items.json"
    C.write_json(items_path, [{"wav": i["wav"], "text": i["text"]} for i in items])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",   # the Vox venv is hash-locked
               VOICE_LAB_CHARACTER=C.CHARACTER["slug"])                             # asr_helper: this asr_prompt
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"   # an empty value does not hide the GPU on Windows
    else:
        if env.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
            env.pop("CUDA_VISIBLE_DEVICES")   # this process hid CUDA for itself only
        env["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = CT2_CACHE_CONFIG
    process = subprocess.Popen([str(VOX_PYTHON), str(HERE / "asr_helper.py"), "--items", str(items_path),
                                "--cache", str(CACHE / "asr_large-v3.json"), "--device", device], env=env)
    guard: dict = {}
    stop = threading.Event()
    if device == "cuda":
        threading.Thread(target=watchdog, args=(process, C.PAINTER_RESERVE_GB, stop, guard), daemon=True).start()
    try:
        code = process.wait()
    finally:
        stop.set()
    if guard.get("killed"):
        raise SystemExit(f"ERROR: {guard['killed']}; rerun later or with --asr cpu")
    if code:
        raise SystemExit(f"ERROR: asr_helper.py exited with {code}")
    return {C.norm_path(i["wav"]): {"wer": i["wer"], "hyp": i["hyp"]} for i in C.read_json(items_path)}


def load_eval(name: str) -> tuple[dict, dict]:
    path = C.EVAL_ROOT / name / "scorecard.json"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found (run eval/run_eval.py --run {name} on the self-test manifests)")
    card = C.read_json(path)
    rows = {}
    for row in card.get("takes", []):
        if not row.get("wav"):
            continue
        asr = row.get("asr") if isinstance(row.get("asr"), dict) else {}
        rows[C.norm_path(row["wav"])] = {"identity": row.get("similarity"), "wer": row.get("wer"),
                                         "hyp": asr.get("asr"), "unscored": row.get("unscored")}
    return rows, {"scorecard": str(path), "similarity": (card.get("method") or {}).get("similarity"),
                  "steps": [{k: s.get(k) for k in ("metric", "device", "exit_code")} for s in card.get("steps") or []]}


# ------------------------------------------------------------------------------------------------ score

def tag_of(params: dict) -> str:
    if not params.get("cfm_tag"):
        return "source"
    tag = params["cfm_tag"]
    if params.get("convert_style"):
        ar = params.get("ar_checkpoint") or "pretrained"
        tag += "+style" + ("" if str(ar).startswith("pretrained") else f"-{C.checkpoint_tag(ar)}")
    for key, default in SELFTEST_DEFAULTS.items():
        value = params.get(key)
        if value is None:
            value = default
        if value != default:
            tag += f"+{SHORT[key]}={value}"
    return tag


def classify(take: dict, by_id: dict) -> tuple[str | None, dict | None, str]:
    """(subset, val row or None, variant) for a take; None subset for anything else (never a test clip)."""
    variant = str(take.get("engine", "")).replace("seedvc_from_", "").replace("selftest_", "")
    line = str(take.get("line_id") or "")
    if variant == "ood" or line.startswith("ood_"):
        return "ood", None, "ood"
    if variant == "tts":
        return "tts", None, "tts"
    row = by_id.get(line)
    if row is None or row["split"] != "val":
        return None, None, variant
    return ("promptsens" if variant == "promptsens" else "val"), row, variant


def step_of(tag: str) -> int:
    match = re.search(r"_s(\d+)$", tag)
    return int(match.group(1)) if match else 0


def mean(values):
    import numpy as np
    values = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.mean(values)), 4) if values else None


def summarize(groups: dict) -> dict:
    """tag -> {subset: metrics, identity} from per-take rows."""
    import numpy as np
    out = {}
    for tag, subsets in groups.items():
        entry = {"tag": tag, "checkpoint": None}
        for subset, takes in subsets.items():
            entry["checkpoint"] = entry["checkpoint"] or next((t["checkpoint"] for t in takes if t.get("checkpoint")), None)
            if subset == "promptsens":
                by_line: dict = {}
                for t in takes:
                    if t.get("mean_st") is not None:
                        by_line.setdefault(t["line_id"], []).append(t)
                spreads_m = [max(x["mean_st"] for x in v) - min(x["mean_st"] for x in v)
                             for v in by_line.values() if len(v) >= 2]
                spreads_r = [max(x["range_st"] for x in v) - min(x["range_st"] for x in v)
                             for v in by_line.values() if len(v) >= 2]
                entry[subset] = {"lines": len(spreads_m), "spread_mean_st": mean(spreads_m),
                                 "spread_range_st": mean(spreads_r), "identity": mean([t.get("identity") for t in takes])}
                continue
            metrics = {"n": len(takes)}
            keys = ["identity", "spk_centroid", "spk_clip", "mcd", "f0_rmse_st", "f0_corr", "f0_median_err_st",
                    "voicing_agree", "dur_ratio", "hf_tilt_dev", "wer", "f0_corr_src", "wer_src"]
            for key in keys:
                value = mean([t.get(key) for t in takes])
                if value is not None:
                    metrics[key] = value
            ratios = [t.get("f0_range_ratio") for t in takes if (t.get("f0_range_ratio") or 0) > 0]
            if ratios:
                metrics["f0_range_dev"] = round(float(np.mean(np.abs(np.log2(ratios)))), 4)
            ratios = [t.get("f0_range_ratio_src") for t in takes if (t.get("f0_range_ratio_src") or 0) > 0]
            if ratios:
                metrics["f0_range_dev_src"] = round(float(np.mean(np.abs(np.log2(ratios)))), 4)
            shifts = [abs(t["median_shift_st"]) for t in takes if t.get("median_shift_st") is not None]
            if shifts:
                metrics["abs_median_shift_st"] = round(float(np.mean(shifts)), 3)
            entry[subset] = metrics
        entry["identity"] = mean([entry.get(s, {}).get("identity") for s in ("val", "ood", "tts")])
        out[tag] = entry
    return out


def gate(entry: dict, base: dict) -> tuple[list[str], dict]:
    failed, checks = [], {}

    def check(name, subset, key, mode, margin):
        value, ref = entry.get(subset, {}).get(key), base.get(subset, {}).get(key)
        if value is None or ref is None:
            checks[name] = "not evaluated"
            return
        if mode == "le":
            limit = ref + margin
            ok = value <= limit + 1e-9
        else:
            if ref < GATES["spread_min_st"]:
                checks[name] = f"not evaluated (baseline spread {ref} < {GATES['spread_min_st']} st)"
                return
            limit = ref * margin
            ok = value >= limit - 1e-9
        checks[name] = {"value": value, "baseline": ref, "limit": round(limit, 4), "pass": ok}
        if not ok:
            failed.append(name)

    check("val.wer", "val", "wer", "le", GATES["wer"])
    check("ood.wer_src", "ood", "wer_src", "le", GATES["wer"])
    check("val.hf_tilt_dev", "val", "hf_tilt_dev", "le", GATES["hf_tilt_db"])
    check("ood.hf_tilt_dev", "ood", "hf_tilt_dev", "le", GATES["hf_tilt_db"])
    check("val.f0_range_dev", "val", "f0_range_dev", "le", GATES["f0_range_dev"])
    check("ood.f0_range_dev_src", "ood", "f0_range_dev_src", "le", GATES["f0_range_dev"])
    check("tts.wer", "tts", "wer", "le", GATES["wer"])
    check("tts.hf_tilt_dev", "tts", "hf_tilt_dev", "le", GATES["hf_tilt_db"])
    check("tts.f0_range_dev_src", "tts", "f0_range_dev_src", "le", GATES["f0_range_dev"])
    check("promptsens.spread_mean_st", "promptsens", "spread_mean_st", "ge", GATES["spread_ratio"])
    check("promptsens.spread_range_st", "promptsens", "spread_range_st", "ge", GATES["spread_ratio"])
    return failed, checks


def is_run_candidate(tag: str, run: str) -> bool:
    return tag == "pretrained" or bool(re.fullmatch(rf"{re.escape(run)}(_ema)?_s\d+", tag))


def listening_pack(run: str, picks: list[str], takes_by_tag: dict, val_rows: dict, args) -> str | None:
    """Blind pack: per line the ground truth (val), the source, pretrained and the picks under random letters."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.listening_root.resolve() / f"{run}_{stamp}"
    versions = ["pretrained"] + [p for p in picks if p != "pretrained"]
    plain = {t: v for t, v in takes_by_tag.items() if t in versions}
    common_val = set.intersection(*[{k for k, e in v.items() if e["subset"] == "val" and e["variant"] == "up4"}
                                    for v in plain.values()]) if plain else set()
    common_ood = set.intersection(*[{k for k, e in v.items() if e["subset"] == "ood"}
                                    for v in plain.values()]) if plain else set()
    val_keys = sorted(common_val)
    by_delivery: dict = {}
    for key in val_keys:
        by_delivery.setdefault(plain["pretrained"][key]["delivery"], []).append(key)
    chosen_val = []
    while len(chosen_val) < min(args.listen_lines, len(val_keys)):
        for group in by_delivery.values():
            if group and len(chosen_val) < args.listen_lines:
                chosen_val.append(group.pop(0))
    ood_keys = sorted(common_ood)
    chosen_ood = []   # --listen-ood split across the impostor voices, the remainder to the first ones
    speakers = C.IMPOSTOR_SPEAKERS
    for i, speaker in enumerate(speakers):
        quota = args.listen_ood // len(speakers) + (1 if i < args.listen_ood % len(speakers) else 0)
        chosen_ood += [k for k in ood_keys if f"|ood_{speaker}_" in k][:quota]
    common_tts = set.intersection(*[{k for k, e in v.items() if e["subset"] == "tts"}
                                    for v in plain.values()]) if plain else set()
    tts_lines: set = set()
    for key in sorted(common_tts):   # one take per new line, up to 4 lines
        line = key.split("|")[2]
        if line not in tts_lines and len(tts_lines) < 4:
            tts_lines.add(line)
            chosen_ood.append(key)
    key_doc, index = {}, [f"# Blind listening pack: {run} ({stamp})", "",
                          f"Per folder: rank the letters by how much they sound like {C.NAME} AND how natural the "
                          "delivery is (timing, emotion, no hiss or warble). Then open _key.json and record the "
                          f"winner: selftest.py confirm --train-run {run} --tag <tag>.", ""]
    for n, key in enumerate(chosen_val + chosen_ood, 1):
        first = plain["pretrained"][key]
        folder = out / f"{n:02d}_{C.safe_name(first['line_id'], 40)}_{first['subset']}"
        candidates = {tag: plain[tag][key]["wav"] for tag in versions if key in plain.get(tag, {})}
        candidates["source"] = first["source_wav"]
        if first["subset"] == "val":
            candidates["ground truth"] = val_rows[first["line_id"]]["audio"]
        rng = random.Random(f"{run}|{key}")
        names = list(candidates)
        rng.shuffle(names)
        letters = {}
        for i, name in enumerate(names):
            letter = chr(ord("A") + i)
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[name], folder / f"{letter}.wav")
            letters[letter] = name
        key_doc[folder.name] = letters
        text = first.get("text") or "(other speaker; text unknown)"
        index.append(f"- {folder.name}: {first['delivery']}: \"{text}\" -> {', '.join(sorted(letters))}")
    if not key_doc:
        return None
    C.write_json(out / "_key.json", {"run": run, "versions": versions, "folders": key_doc})
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return str(out)


def score(args) -> int:
    import numpy as np
    SELFTEST_DEFAULTS["diffusion_steps"] = args.plain_diffusion_steps
    if args.baseline_tag and args.baseline_tag.startswith("best:"):
        args.baseline_tag = C.selected_checkpoint(args.runs_root.resolve(), args.baseline_tag[5:])[0]
        print(f"baseline tag: {args.baseline_tag}")
    rows = C.load_corpus()
    by_id = {r["id"]: r for r in rows}
    excluded = C.speaker_exclusions(required=True, rows=rows)
    subsets = {s.strip() for s in args.subsets.split(",") if s.strip()}
    run_filter = re.compile(args.runs) if args.runs else None
    feats = Features(use_f0=not args.no_f0)
    centroid, centroid_path = train_centroid(feats, rows, excluded)
    val_rows = {r["id"]: r for r in rows if r["split"] == "val" and r["kind"] == "speech"
                and r["id"] not in excluded["confirmed"] and r["id"] not in excluded["uncertain"]}
    anchors = local_anchors(feats, centroid, list(val_rows.values()))
    val_tilts = [feats.get(r["audio"], f0=False)["tilt"] for r in val_rows.values()]
    val_tilt = float(np.nanmedian(val_tilts)) if val_tilts else None

    manifests = sorted(args.sources_root.resolve().glob("*/manifest.json"))
    converted = sorted(args.converted_root.resolve().glob("*/*/manifest.json"))
    manifests += [m for m in converted if not run_filter or run_filter.search(m.parent.name)]
    entries = []
    for manifest in manifests:
        for take in C.read_json(manifest)["takes"]:
            subset, row, variant = classify(take, by_id)
            if subset is None or subset not in subsets:
                continue
            wav = Path(take["wav"]) if Path(take["wav"]).is_absolute() else manifest.parent / take["wav"]
            if not wav.exists():
                print(f"missing {wav}")
                continue
            params = take.get("params") or {}
            tag = tag_of(params)
            if tag == "source" and subset == "promptsens":
                continue
            source_wav = params.get("source_wav") or str(wav)
            entries.append({"tag": tag, "checkpoint": params.get("cfm_checkpoint"), "take": take, "row": row,
                            "subset": subset, "variant": variant, "wav": str(wav), "source_wav": source_wav,
                            "manifest": str(manifest), "line_id": str(take.get("line_id")),
                            "delivery": take.get("delivery")})
    if not entries:
        raise SystemExit("ERROR: nothing to score")
    print(f"{len(entries)} takes in {len(manifests)} manifests; subsets {sorted(subsets)}", flush=True)

    eval_rows, eval_info = load_eval(args.eval_run) if args.eval_run else ({}, None)
    asr = {}
    if not eval_rows and args.asr != "off":
        wavs = {}
        for e in entries:
            text = (e["row"] or {}).get("text", "") if e["subset"] != "tts" else (e["take"].get("text") or "")
            wavs[C.norm_path(e["wav"])] = {"wav": e["wav"], "text": text}
            if e["subset"] == "ood":
                wavs.setdefault(C.norm_path(e["source_wav"]), {"wav": e["source_wav"], "text": ""})
        for r in {e["row"]["id"]: e["row"] for e in entries if e["row"]}.values():
            wavs.setdefault(C.norm_path(r["audio"]), {"wav": r["audio"], "text": r["text"]})
        asr = run_asr(list(wavs.values()), args.asr, args.asr_need_gb)
    ec = C.lab_module(C.EVAL_CODE, "evalcommon")

    def heard(path) -> dict | None:
        key = C.norm_path(path)
        if eval_rows:
            return eval_rows.get(key)
        return asr.get(key)

    def wer_vs(hyp: str | None, ref: str | None):
        if hyp is None or ref is None:
            return None
        ref_words = ec.norm_words(ref)
        return ec.edit_distance(ref_words, ec.norm_words(hyp)) / max(1, len(ref_words)) if ref_words else None

    groups: dict = {}
    takes_by_tag: dict = {}
    not_in_eval = 0
    for n, e in enumerate(entries, 1):
        out = feats.get(e["wav"])
        cos = float(np.dot(unit(out["emb"]), centroid))
        metrics = {"spk_centroid": round(cos, 4)}
        if anchors["usable"]:
            metrics["identity"] = round((cos - anchors["impostor_mean"]) / (anchors["val_mean"] - anchors["impostor_mean"]), 4)
        hit = heard(e["wav"])
        if eval_rows:
            metrics["identity"] = hit.get("identity") if hit else None
            not_in_eval += hit is None
        if e["subset"] == "val":
            gt_audio = (e["take"].get("params") or {}).get("reference_audio") or "original"
            gt_path = C.clip_path(e["row"], gt_audio if gt_audio in ("original", "restored") else "original")
            metrics.update(compare_val(feats.get(gt_path), out))
            if hit:
                metrics["wer"] = hit.get("wer") if hit.get("wer") is not None else wer_vs(hit.get("hyp"), e["row"]["text"])
        elif e["subset"] in ("ood", "tts"):
            if e["tag"] != "source":
                metrics.update(compare_src(feats.get(e["source_wav"]), out, val_tilt))
                src_hit = heard(e["source_wav"])
                if e["subset"] == "ood" and hit and src_hit:
                    metrics["wer_src"] = wer_vs(hit.get("hyp"), src_hit.get("hyp"))
            elif np.isfinite(out["tilt"]) and val_tilt is not None:
                metrics["hf_tilt_dev"] = float(out["tilt"] - val_tilt)
            if e["subset"] == "tts" and hit:
                metrics["wer"] = hit.get("wer") if hit.get("wer") is not None else \
                    wer_vs(hit.get("hyp"), e["take"].get("text"))
        else:  # promptsens
            stats = C.f0_summary(out["f0"], out["voiced"]) if "f0" in out else None
            if stats:
                voiced = out["f0"][out["voiced"].astype(bool) & np.isfinite(out["f0"])]
                metrics["mean_st"] = round(float(np.mean(12 * np.log2(voiced / 100.0))), 3)
                metrics["range_st"] = stats["range_st"]
        row = {"take_id": e["take"]["take_id"], "line_id": e["line_id"], "variant": e["variant"],
               "delivery": e["delivery"], "checkpoint": e["checkpoint"], "wav": e["wav"], **metrics}
        groups.setdefault(e["tag"], {}).setdefault(e["subset"], []).append(row)
        takes_by_tag.setdefault(e["tag"], {})[f"{e['subset']}|{e['variant']}|{e['line_id']}|{e['delivery']}"] = \
            {**row, "subset": e["subset"], "source_wav": e["source_wav"], "text": e["take"].get("text")}
        if n % 100 == 0:
            print(f"  {n}/{len(entries)} takes ({feats.computed} new feature files)", flush=True)

    if not_in_eval:
        print(f"WARNING: {not_in_eval} takes are missing from eval run {args.eval_run} (identity/WER left empty); "
              "rerun eval/run_eval.py on the current manifests", flush=True)
    summary = summarize(groups)
    ceiling = {"identity_method": eval_info["similarity"] if eval_info else
               (f"local Resemblyzer, anchored 0 = {C.IMPOSTOR_LABEL}, 1 = real val" if anchors["usable"]
                else "unanchored"),
               "anchors": anchors, "val_hf_tilt_median_db": val_tilt}
    if "val" in subsets:
        ceiling["spk_centroid_original_val"] = mean([float(np.dot(unit(feats.get(r["audio"], f0=False)["emb"]), centroid))
                                                     for r in val_rows.values()])
        if eval_rows or asr:
            ceiling["wer_original_val"] = mean([(heard(r["audio"]) or {}).get("wer") for r in val_rows.values()])

    comparisons = {}
    baselines = [("pretrained", run) for run in args.train_run]
    if args.baseline_tag:
        baselines.append((args.baseline_tag, None))
    for base_tag, run in baselines:
        if base_tag not in summary:
            print(f"WARNING: baseline tag {base_tag} not among the scored tags; skipped")
            continue
        rows_out = []
        for tag, entry in summary.items():
            if tag == "source" or (run and not is_run_candidate(tag, run)):
                continue
            failed, checks = gate(entry, summary[base_tag])
            rows_out.append({"tag": tag, "checkpoint": entry["checkpoint"], "identity": entry["identity"],
                             "identity_delta": None if entry["identity"] is None or summary[base_tag]["identity"] is None
                             else round(entry["identity"] - summary[base_tag]["identity"], 4),
                             "passes": not failed, "failed": failed, "checks": checks})
        rows_out.sort(key=lambda r: (not r["passes"], -(r["identity"] if r["identity"] is not None else -9),
                                     step_of(r["tag"])))
        comparisons[run or f"vs:{base_tag}"] = {"baseline": base_tag, "rows": rows_out}

    name = args.name or "_".join(args.train_run) or (f"vs_{C.safe_name(args.baseline_tag)}" if args.baseline_tag else "all")
    out_path = args.scores_dir.resolve() / f"{name}.json"
    C.write_json(out_path, {"scored_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                            "centroid": centroid_path, "eval": eval_info, "asr": "eval" if eval_rows else args.asr,
                            "gates": GATES, "ceiling": ceiling, "manifests": [str(m) for m in manifests],
                            "summary": summary, "comparisons": comparisons,
                            "takes": {t: g for t, g in groups.items()}})
    header = ["tag", "identity", "val.wer", "val.hf_tilt_dev", "val.f0_range_dev", "val.mcd", "val.f0_corr",
              "ood.identity", "ood.wer_src", "ood.hf_tilt_dev", "ood.f0_range_dev_src", "ood.f0_corr_src",
              "tts.identity", "tts.wer", "tts.f0_range_dev_src",
              "promptsens.spread_mean_st", "promptsens.spread_range_st"]

    def cell(entry, col):
        if col in ("tag", "identity"):
            return entry.get(col)
        subset, key = col.split(".")
        return entry.get(subset, {}).get(key)
    order = sorted(summary.values(), key=lambda e: (e["tag"] != "source", e["tag"] != "pretrained", "+" in e["tag"],
                                                     re.sub(r"_s\d+$", "", e["tag"]), step_of(e["tag"])))
    print(" | ".join(header))
    for entry in order:
        print(" | ".join("" if cell(entry, h) is None else str(cell(entry, h)) for h in header))
    print(f"ceiling/anchors: {ceiling}")
    for label, comp in comparisons.items():
        print(f"\n[{label}] gates relative to {comp['baseline']}:")
        for r in comp["rows"]:
            print(f"  {r['tag']}: identity {r['identity']} ({r['identity_delta']:+}) "
                  f"{'PASS' if r['passes'] else 'FAIL ' + ','.join(r['failed'])}"
                  if r["identity_delta"] is not None else f"  {r['tag']}: identity n/a, failed {r['failed']}")
    print(f"wrote {out_path}")

    for run in args.train_run:
        comp = comparisons.get(run)
        if not comp:
            continue
        candidates = [r for r in comp["rows"] if r["identity"] is not None]
        passing = [r for r in candidates if r["passes"]]
        if not passing:
            print(f"no scored candidates with identity for {run}; selection.json not written")
            continue
        top = max(r["identity"] for r in passing)
        near = [r for r in passing if r["identity"] >= top - GATES["tie"]]
        best = min(near, key=lambda r: (step_of(r["tag"]), "_ema_" in r["tag"]))
        finetuned = sorted((r for r in passing if r["tag"] != "pretrained"),
                           key=lambda r: (-r["identity"], step_of(r["tag"])))
        shortlist = ([best] if best["tag"] != "pretrained" else []) + \
                    [r for r in finetuned if r["tag"] != best["tag"]][: 2 - (best["tag"] != "pretrained")]
        shortlist = shortlist[:2] + [r for r in candidates if r["tag"] == "pretrained"]
        steps = [step_of(r["tag"]) for r in candidates if r["tag"] != "pretrained"]
        last = bool(steps) and best["tag"] != "pretrained" and step_of(best["tag"]) == max(steps)
        selection_path = args.runs_root.resolve() / run / "selection.json"
        previous = C.read_json(selection_path) if selection_path.exists() else {}
        pack = None if args.no_listening_pack else listening_pack(
            run, [r["tag"] for r in shortlist], takes_by_tag, val_rows, args)
        selection = {
            "run": run, "method": "gated: pass every gate relative to pretrained, then highest identity "
                                  f"(ties within {GATES['tie']} to the earliest step)",
            "gates": GATES, "identity_source": ceiling["identity_method"],
            "tag": best["tag"], "cfm_checkpoint": best["checkpoint"] if best["tag"] != "pretrained" else None,
            "beats_pretrained": best["tag"] != "pretrained",
            "shortlist": [{"tag": r["tag"], "cfm_checkpoint": r["checkpoint"] if r["tag"] != "pretrained" else None,
                           "identity": r["identity"]} for r in shortlist],
            "best_is_last_step": last,
            "advice": ("the best checkpoint is the run's last step: continue training from it "
                       f"(train.py --init-cfm <its path> --run-name {run}_cont --max-steps 1500) and score again"
                       if last else None),
            "candidates": comp["rows"], "metrics": summary.get(best["tag"]), "ceiling": ceiling,
            "listening_pack": pack, "confirmed": previous.get("confirmed"),
            "scores": str(out_path), "note": "objective proxy from selftest.py; confirm by listening "
                                             "(selftest.py confirm), the user's ears are the final judge",
        }
        if previous.get("confirmed"):
            print(f"NOTE: keeping the earlier listening confirmation {previous['confirmed'].get('tag')} in {selection_path}")
        C.write_json(selection_path, selection)
        print(f"selected {best['tag']} (beats_pretrained {selection['beats_pretrained']}), shortlist "
              f"{[r['tag'] for r in shortlist]} -> {selection_path}" + (f"; listening pack {pack}" if pack else ""))
        if last:
            print("NOTE:", selection["advice"])
    return 0


def confirm(args) -> int:
    path = args.runs_root.resolve() / args.train_run / "selection.json"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found; run selftest.py score --train-run {args.train_run} first")
    selection = C.read_json(path)
    candidates = {r["tag"]: r for r in selection.get("candidates", [])}
    if args.tag not in candidates:
        raise SystemExit(f"ERROR: {args.tag} is not a candidate; choose from {sorted(candidates)}")
    pick = candidates[args.tag]
    selection["confirmed"] = {"tag": args.tag, "cfm_checkpoint": pick["checkpoint"] if args.tag != "pretrained" else None,
                              "by": "listening", "note": args.note, "passes_gates": pick["passes"],
                              "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    C.write_json(path, selection)
    print(f"confirmed {args.tag} for {args.train_run} -> {path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.cmd == "sources":
        return make_sources(args)
    if args.cmd == "confirm":
        return confirm(args)
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
