"""Per-clip prosody analysis for a character's lab corpus (CPU only; VOICE_LAB_CHARACTER, default baodur).

For every train/val clip this measures pitch (librosa.pyin), level/dynamics, pauses, speaking and
articulation rate, spectral effort (alpha ratio, Hammarberg index, centroid) and the exact
Vox-sidecar style features (port of vox_voxcpm_server.style_features) plus the Vox delivery
reading, joined with text, TLK directions, delivery tags and the speaker screen
(speaker_check.json, when present). It also writes the character's baseline voice profile and a
robust z-score for every numeric feature relative to that profile. His pitch range (pyin search band and frame)
comes from <lab>/delivery/content/voice.json, his syllable exceptions from content/lexicon.json, and the
"addresses the player" text feature from the character config's address_term (text_addresses_<term>).

Held-out test clips are skipped unless --include-test is given; they then go to a separate
eval-only file (features_test*.jsonl) and never into the profile or any reference choice.

Run with the VoxCPM2 runtime python (read-only venv: numpy/scipy/librosa/soundfile are there; no bytecode is
written and numba's JIT cache goes to <lab>/delivery/.cache/numba, never into the venv):
  CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
    %LOCALAPPDATA%/VoxStudio/engines/voxcpm2/.venv/Scripts/python.exe analyze.py \
    [--audio original|restored] [--workers 6] [--include-test] [--out-dir DIR]
Outputs in <lab>/delivery/ (or --out-dir, e.g. a scratch dry-run; the per-clip measurement cache is then read from
the lab and only written next to the outputs): features{,.restored}.jsonl/.csv and voice_profile{,.restored}.json.
clip_features(path, text) is importable for scoring rendered takes the same way.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True  # CUDA is hidden in main() only: importers (score_delivery --asr cuda) keep the GPU
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labcfg  # noqa: E402  (stdlib only: safe before numpy/librosa)

# librosa's pyin/yin are numba-jitted with cache=True: without this, numba writes .nbi/.nbc files next to librosa's
# sources inside the hash-locked VoxCPM2 venv. Keep its cache in the lab instead (set before librosa is imported).
os.environ.setdefault("NUMBA_CACHE_DIR", str(labcfg.NUMBA_CACHE))

import numpy as np  # noqa: E402

LAB = labcfg.LAB
OUT_DIR = labcfg.DELIVERY
VOX_SIDECAR = Path(r"C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\AI-VoiceActing-GameDevelopment"
                   r"\third_party\voxcpm_sidecar")
EXACT_SOURCES = {"tlk-exact", "dlg"}
ANALYSIS_VERSION = 1  # bump when clip_features() changes so cached measurements are recomputed

SR = 16000         # analysis rate: prosody and <8 kHz spectra are identical for original and restored input
HOP = 160           # 10 ms frames everywhere
RMS_FRAME = 400     # 25 ms
# His pitch range (content/voice.json): pyin search band and frame (Bao-Dur: 60-400 Hz, 1536 = 96 ms, a 48 ms YIN
# window that stays stable on his low register without octave-up errors).
_PITCH = labcfg.content("voice")["pitch_tracking"]
PYIN_FRAME = int(_PITCH["pyin_frame"])
FMIN, FMAX = float(_PITCH["fmin_hz"]), float(_PITCH["fmax_hz"])
LOW_LIMIT_ST, HIGH_LIMIT_ST = -9.0, 15.0  # pitch frames outside this band (re his register centre) are excluded
MIN_PAUSE_S = 0.18  # silent gaps shorter than this are stop closures, not pauses

# Tokens the vowel-group syllable counter gets wrong, as spoken in the game (content/lexicon.json).
SPECIAL_SYLLABLES: dict[str, int] = {k.casefold(): int(v) for k, v in
                                     labcfg.content("lexicon")["special_syllables"].items() if not k.startswith("_")}
ADDRESS_FEATURE = f"addresses_{labcfg.address_key()}"  # text feature: the line addresses the player by his term


def settings_key() -> str:
    """Fingerprint of the character settings a cached measurement depends on (pitch range, syllable exceptions)."""
    import hashlib

    blob = json.dumps({"pitch": [FMIN, FMAX, PYIN_FRAME], "syllables": sorted(SPECIAL_SYLLABLES.items())})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- text features
def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*'?|'[A-Za-z]+", text)


def syllables(word: str) -> int:
    lowered = word.casefold()
    if lowered in SPECIAL_SYLLABLES:
        return SPECIAL_SYLLABLES[lowered]
    if re.fullmatch(r"\d+", lowered):
        return max(1, len(lowered))
    core = re.sub(r"[^a-z']", "", lowered)
    stem = core.replace("'", "")
    if not stem:
        return 0
    count = len(re.findall(r"[aeiouy]+", stem))
    if stem.endswith("e") and not stem.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    if stem.endswith("ed") and not re.search(r"[td]ed$", stem) and count > 1:
        count -= 1
    if core.endswith("n't") and core[:-3] and core[:-3][-1] not in "aeiouy":
        count += 1  # didn't, wasn't, couldn't (syllabic n); don't, can't, won't, haven't stay as counted
    return max(1, count)


def text_features(text: str) -> dict:
    words = tokens(text)
    lowered = text.casefold()
    return {
        "words": len(words),
        "syllables": sum(syllables(w) for w in words),
        "sentences": max(1, len(re.findall(r"[.!?]+(?:\s|$)", text.strip() + " "))),
        "question": text.strip().endswith("?"),
        "has_question": "?" in text,
        "exclamation": "!" in text,
        "ellipsis": "..." in text or "\u2026" in text,
        "dashes": len(re.findall(r"\s[-\u2013\u2014]\s|[-\u2013\u2014]$", text)),
        "contractions": len(re.findall(r"\b[A-Za-z]+'(?:t|re|ve|ll|d|m)\b", text)) + len(
            re.findall(r"\b(?:it|that|there|what|he|she|who|let|here)'s\b", lowered)),
        "formal_forms": len(re.findall(r"\b(?:cannot|can not|do not|does not|did not|is not|are not|will not|"
                                        r"i am|it is|i do not|i will)\b", lowered)),
        ADDRESS_FEATURE: bool(re.search(labcfg.address_regex(), text)),
        "first_person": len(re.findall(r"\b(?:i|me|my|myself|i'm|i've|i'd|i'll)\b", lowered)),
    }


# ----------------------------------------------------------------------------- audio features
def _load(path: str):
    import librosa
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    clip_ratio = float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0
    native = mono
    if rate != SR:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=SR, res_type="soxr_hq")
    return mono.astype(np.float32), native, int(rate), peak, clip_ratio


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    runs, start = [], None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def vox_style_features(native: np.ndarray, rate: int, transcript: str) -> dict:
    """Exact port of vox_voxcpm_server.style_features (so readings match Vox's own analysis)."""
    import librosa
    from scipy.signal import medfilt

    mono, _ = librosa.effects.trim(native, top_db=38)
    duration = max(len(mono) / rate, 0.1)
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=480)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)
    active_rms = rms_db[rms_db > -30.0]
    dynamic_db = float(np.percentile(active_rms, 90) - np.percentile(active_rms, 10)) if active_rms.size else 0.0
    f0 = librosa.yin(mono, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=rate,
                     frame_length=2048, hop_length=480)
    frame_count = min(len(f0), len(rms_db))
    frame_positions = np.linspace(0.0, 1.0, frame_count)
    analysis_rms = rms_db[:frame_count]
    voiced = (np.isfinite(f0[:frame_count]) & (f0[:frame_count] < librosa.note_to_hz("C6") * 0.98)
              & (analysis_rms > -38.0))
    finite_f0 = f0[:frame_count][voiced]
    voiced_positions = frame_positions[voiced]
    variation = pitch_range = slope = terminal = 0.0
    if finite_f0.size >= 3:
        raw = 12.0 * np.log2(finite_f0)
        kernel = 5 if raw.size >= 5 else 3
        local = medfilt(raw, kernel_size=kernel)
        semis = raw - (12.0 * np.round((raw - local) / 12.0))
        semis = medfilt(semis, kernel_size=kernel)
        center = float(np.median(semis))
        plausible = np.abs(semis - center) < 10.0
        semis = semis[plausible] - center
        voiced_positions = voiced_positions[plausible]
        if semis.size >= 3:
            variation = float(np.std(semis))
            pitch_range = float(np.percentile(semis, 90) - np.percentile(semis, 10))
            slope = float(np.polyfit(voiced_positions, semis, 1)[0])
            opening = semis[voiced_positions <= 0.35]
            ending = semis[voiced_positions >= 0.65]
            terminal = float(np.median(ending) - np.median(opening)) if opening.size and ending.size else 0.0
    pause_ratio = float(np.mean(rms_db < -32.0)) if rms_db.size else 0.0
    first = analysis_rms[frame_positions <= 0.35]
    last = analysis_rms[frame_positions >= 0.65]
    energy_slope = float(np.median(last) - np.median(first)) if first.size and last.size else 0.0
    return {
        "dynamic_db": dynamic_db, "pitch_variation_semitones": variation, "pitch_range_semitones": pitch_range,
        "pitch_slope_semitones": slope, "terminal_pitch_delta": terminal, "pause_ratio": pause_ratio,
        "energy_slope_db": energy_slope,
        "characters_per_second": len(re.sub(r"[^A-Za-z0-9]", "", transcript)) / duration,
    }


_VOX_DELIVERY = None


def vox_reading(features: dict, transcript: str) -> str:
    """Vox's own delivery label for the clip (vox_delivery.detect_delivery, imported read-only)."""
    global _VOX_DELIVERY
    if _VOX_DELIVERY is None:
        source = VOX_SIDECAR / "vox_delivery.py"
        if not source.is_file():
            _VOX_DELIVERY = False
        else:
            spec = importlib.util.spec_from_file_location("vox_delivery_readonly", source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
            spec.loader.exec_module(module)  # sys.dont_write_bytecode keeps the Vox repo untouched
            _VOX_DELIVERY = module
    if _VOX_DELIVERY is False:
        return ""
    return _VOX_DELIVERY.detect_delivery(features, transcript, ()).label


def register_center(frames_hz) -> float:
    """Centre of the speaker's modal register: median of the densest one-octave window of voiced frames.
    Robust to the sub-octave cluster pyin reports on his creaky, low phrase endings."""
    st = 12.0 * np.log2(np.asarray(frames_hz, dtype=float) / FMIN)
    if st.size == 0:
        return float("nan")
    if st.max() - st.min() <= 12.0:
        return float(FMIN * 2.0 ** (np.median(st) / 12.0))
    starts = np.arange(st.min(), st.max() - 12.0 + 0.25, 0.25)
    counts = [np.count_nonzero((st >= s) & (st < s + 12.0)) for s in starts]
    best = starts[int(np.argmax(counts))]
    window = st[(st >= best) & (st < best + 12.0)]
    return float(FMIN * 2.0 ** (np.median(window) / 12.0))


def pitch_stats(f0_hz, times, center_hz: float) -> dict:
    """Pitch statistics relative to his register centre. Frames > 9 st below it (creak or sub-octave
    tracking) and > 15 st above it (octave-up errors) are counted, not used."""
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    out: dict = {"f0_frames": int(f0.size)}
    if f0.size < 10 or not center_hz or not np.isfinite(center_hz):
        return out
    st = 12.0 * np.log2(f0 / center_hz)
    low, high = st < LOW_LIMIT_ST, st > HIGH_LIMIT_ST
    keep = ~low & ~high
    out.update({"low_f0_ratio": round(float(low.mean()), 3), "high_f0_ratio": round(float(high.mean()), 3)})
    s, tt = st[keep], t[keep]
    if s.size < 10:
        return out
    median = float(np.median(s))
    n = s.size
    tail = max(5, int(n * 0.15))
    body = s[int(n * 0.40):max(int(n * 0.40) + 1, n - tail)]
    to_hz = lambda value: center_hz * 2.0 ** (value / 12.0)  # noqa: E731
    out.update({
        "f0_median_hz": round(to_hz(median), 1),
        "f0_median_st": round(median, 2),
        "f0_p10_hz": round(to_hz(float(np.percentile(s, 10))), 1),
        "f0_p90_hz": round(to_hz(float(np.percentile(s, 90))), 1),
        "f0_range_st": round(float(np.percentile(s, 90) - np.percentile(s, 10)), 2),
        "f0_iqr_st": round(float(np.percentile(s, 75) - np.percentile(s, 25)), 2),
        "f0_std_st": round(float(np.std(s)), 2),
        "f0_slope_st_per_s": round(float(np.polyfit(tt, s, 1)[0]), 2) if np.ptp(tt) > 0 else 0.0,
        "final_rise_st": round(float(np.median(s[-tail:]) - np.median(body)), 2) if body.size else 0.0,
        "f0_peak_st": round(float(np.percentile(s, 97) - median), 2),
    })
    return out


def clip_features(path: str, text: str, center_hz: float | None = None) -> dict:
    """All measurements for one clip. Pass center_hz (voice_profile.json register.center_hz) to get pitch
    statistics directly; without it the raw voiced track is returned under _f0_hz/_f0_t."""
    import librosa
    from scipy.signal import medfilt

    y, native, rate, peak, clip_ratio = _load(path)
    duration = len(y) / SR
    result: dict = {"duration_s": round(duration, 3), "peak_dbfs": round(20 * math.log10(max(peak, 1e-6)), 2),
                    "clip_ratio": round(clip_ratio, 5), "native_rate": rate}

    rms = librosa.feature.rms(y=y, frame_length=RMS_FRAME, hop_length=HOP, center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    noise_floor = float(np.percentile(db, 5))
    reference = float(np.percentile(db, 95))
    threshold = max(reference - 30.0, noise_floor + 8.0)
    active = medfilt((db > threshold).astype(float), kernel_size=5) > 0.5
    if not active.any():
        return {**result, "error": "no speech detected"}
    first, last = int(np.argmax(active)), int(len(active) - np.argmax(active[::-1]))
    span_s = (last - first) * HOP / SR
    result.update({
        "lead_silence_s": round(first * HOP / SR, 3),
        "tail_silence_s": round(max(0.0, duration - last * HOP / SR), 3),
        "speech_span_s": round(span_s, 3),
        "noise_floor_db": round(noise_floor, 2),
    })

    inside = active[first:last]
    pause_runs = [(a, b) for a, b in _runs(~inside) if (b - a) * HOP / SR >= MIN_PAUSE_S]
    pauses = [(b - a) * HOP / SR for a, b in pause_runs]
    pause_total = float(sum(pauses))
    speaking_s = max(span_s - pause_total, 0.05)
    result.update({
        "pause_count": len(pauses),
        "pause_total_s": round(pause_total, 3),
        "pause_ratio": round(pause_total / max(span_s, 1e-3), 4),
        "pause_mean_s": round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "pause_max_s": round(float(np.max(pauses)), 3) if pauses else 0.0,
    })

    active_db = db[first:last][inside]
    level = 10.0 * math.log10(float(np.mean(10.0 ** (active_db / 10.0))))
    third = max(1, int(len(active_db) * 0.35))
    result.update({
        "level_dbfs": round(level, 2),
        "snr_db": round(level - noise_floor, 2),
        "dynamic_db": round(float(np.percentile(active_db, 90) - np.percentile(active_db, 10)), 2),
        "energy_slope_db": round(float(np.median(active_db[-third:]) - np.median(active_db[:third])), 2),
    })

    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=PYIN_FRAME,
                                                hop_length=HOP, center=True)
    count = min(len(f0), len(active))
    frame_active = np.zeros(count, dtype=bool)
    frame_active[first:min(last, count)] = active[first:min(last, count)]
    voiced = frame_active & voiced_flag[:count] & np.isfinite(f0[:count])
    result["voiced_ratio"] = round(float(voiced.sum() / max(frame_active.sum(), 1)), 3)
    result["voicing_prob"] = round(float(np.nanmean(voiced_prob[:count][frame_active])), 3)
    times = np.arange(count) * HOP / SR
    raw = f0[:count][voiced]
    vtimes = times[voiced]
    if raw.size >= 5:
        semis = 12.0 * np.log2(raw / FMIN)
        local = medfilt(semis, kernel_size=5)
        raw = FMIN * 2.0 ** ((semis - 12.0 * np.round((semis - local) / 12.0)) / 12.0)  # fold isolated octave jumps
    # Raw voiced track; pitch_stats() turns it into statistics once his register centre is known.
    result["_f0_hz"] = [round(float(v), 2) for v in raw]
    result["_f0_t"] = [round(float(v), 3) for v in vtimes]
    if center_hz:
        result.update(pitch_stats(result.pop("_f0_hz"), result.pop("_f0_t"), center_hz))
    if voiced.sum() >= 10:
        # Spectral effort on voiced, active frames (<8 kHz, so original and restored stay comparable).
        spectrum = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, center=True)) ** 2
        frames = np.where(voiced)[0]
        frames = frames[frames < spectrum.shape[1]]
        freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
        voiced_spec = spectrum[:, frames]
        ltas = voiced_spec.mean(axis=1) + 1e-12
        low = ltas[(freqs >= 50) & (freqs < 1000)].sum()
        high = ltas[(freqs >= 1000) & (freqs < 5000)].sum()
        ltas_db = 10 * np.log10(ltas)
        centroid = librosa.feature.spectral_centroid(S=np.sqrt(voiced_spec), sr=SR, n_fft=512)[0]
        flatness = librosa.feature.spectral_flatness(S=np.sqrt(voiced_spec))[0]
        result.update({
            "alpha_ratio_db": round(10 * math.log10(high / low), 2),
            "hammarberg_db": round(float(ltas_db[freqs < 2000].max() - ltas_db[(freqs >= 2000) & (freqs < 5000)].max()), 2),
            "centroid_hz": round(float(np.median(centroid)), 1),
            "flatness": round(float(np.median(flatness)), 5),
        })

    tf = text_features(text)
    result.update({f"text_{k}": v for k, v in tf.items()})
    result.update({
        "speech_rate_syl_s": round(tf["syllables"] / max(span_s, 0.1), 2),
        "articulation_rate_syl_s": round(tf["syllables"] / speaking_s, 2),
        "words_per_min": round(60.0 * tf["words"] / max(span_s, 0.1), 1),
    })
    vox = vox_style_features(native, rate, text)
    result["vox"] = {k: round(v, 4) for k, v in vox.items()}
    result["vox_reading"] = vox_reading(vox, text)
    return result


def _work(item: tuple[str, str, str]) -> tuple[str, dict]:
    clip_id, path, text = item
    try:
        return clip_id, clip_features(path, text)
    except Exception as exception:  # keep going; the row records the failure
        return clip_id, {"error": f"{type(exception).__name__}: {exception}"}


def warm_jit() -> None:
    """Compile librosa's numba kernels in THIS process (and write numba's cache once) before a process pool starts.
    numba's on-disk cache is not safe against several workers compiling the same function at once: a cold cache filled
    that way (JoleeLab, 2026-09-30) made every later pyin call crash with an access violation. Measurements are
    unchanged; the kernels are only compiled earlier."""
    import tempfile

    import soundfile as sf

    rate = 22050
    t = np.arange(int(rate * 0.6), dtype=np.float32) / rate
    tone = 0.2 * np.sin(2 * np.pi * 120.0 * t) * np.hanning(t.size)
    gap = np.zeros(int(rate * 0.2), dtype=np.float32)
    signal = np.concatenate([gap, tone, gap, 0.7 * tone, gap]).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="delivery_jit_") as folder:
        path = str(Path(folder) / "warm.wav")
        sf.write(path, signal, rate, subtype="FLOAT")
        clip_features(path, "warm up the pitch tracker", center_hz=120.0)


def measure(item: tuple[str, str, float]) -> dict:
    """Process-pool job for score_delivery.py: clip_features(path, text, center_hz), errors returned, not raised."""
    path, text, center_hz = item
    try:
        return clip_features(path, text, center_hz=center_hz)
    except Exception as exception:
        return {"error": f"{type(exception).__name__}: {exception}"}


# ----------------------------------------------------------------------------- profile
NUMERIC = [
    "f0_median_hz", "f0_median_st", "f0_p10_hz", "f0_p90_hz", "f0_range_st", "f0_iqr_st", "f0_std_st",
    "f0_slope_st_per_s", "final_rise_st", "f0_peak_st", "low_f0_ratio", "level_dbfs", "dynamic_db", "energy_slope_db", "snr_db", "noise_floor_db",
    "pause_ratio", "pause_count", "pause_mean_s", "pause_max_s", "speech_rate_syl_s", "articulation_rate_syl_s",
    "words_per_min", "alpha_ratio_db", "hammarberg_db", "centroid_hz", "flatness", "voiced_ratio", "voicing_prob",
    "lead_silence_s", "tail_silence_s", "speech_span_s",
]


def profile_member(row: dict) -> bool:
    """Clips that define his baseline: his own exact-text speech, speaker screen not flagged, >= 1.5 s."""
    speaker_ok = row.get("speaker_flag") in (None, "ok", "suspect")
    return (row["split"] in ("train", "val") and row["kind"] == "speech" and row["text_source"] in EXACT_SOURCES
            and (row["wer"] or 0.0) <= 0.15 and row["duration"] >= 1.5 and speaker_ok and "error" not in row)


def build_profile(rows: list[dict], center_hz: float) -> dict:
    members = [r for r in rows if profile_member(r)]
    stats = {}
    for key in NUMERIC:
        values = np.array([r[key] for r in members if r.get(key) is not None], dtype=float)
        if values.size < 5:
            continue
        q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
        # Robust spread; the std floor keeps mostly-zero features (e.g. low_f0_ratio) from exploding.
        scale = max((q75 - q25) / 1.349, 0.5 * float(np.std(values)), 1e-3)
        stats[key] = {"median": round(float(q50), 4), "p10": round(float(q10), 4), "p25": round(float(q25), 4),
                      "p75": round(float(q75), 4), "p90": round(float(q90), 4),
                      "scale": round(float(scale), 6), "n": int(values.size)}
    f0 = np.array([r["f0_median_hz"] for r in members if r.get("f0_median_hz")], dtype=float)
    return {
        "members": len(members),
        "member_rule": "train/val speech, text tlk-exact|dlg, wer <= 0.15, >= 1.5 s, speaker screen not other_speaker",
        "seconds": round(sum(r["duration"] for r in members), 1),
        "register": {
            "center_hz": round(center_hz, 2),
            "center_rule": "median of the densest one-octave window of all voiced pyin frames of the member clips",
            "pitch_band_st": [LOW_LIMIT_ST, HIGH_LIMIT_ST],
            "f0_median_of_clip_medians_hz": round(float(np.median(f0)), 1) if f0.size else None,
            "clip_median_spread_st": round(float(12 * np.log2(np.percentile(f0, 90) / np.percentile(f0, 10))), 2) if f0.size else None,
        },
        "stats": stats,
    }


def add_z(rows: list[dict], profile: dict) -> None:
    for row in rows:
        z = {}
        for key, stat in profile["stats"].items():
            value = row.get(key)
            if value is not None:
                z[key] = round((value - stat["median"]) / stat["scale"], 3)
        row["z"] = z


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=f"{labcfg.NAME} per-clip prosody analysis (CPU).")
    parser.add_argument("--audio", choices=("original", "restored"), default="original")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--include-test", action="store_true",
                        help="also measure held-out test clips into features_test*.jsonl (evaluation use only)")
    parser.add_argument("--no-cache", action="store_true", help="re-measure every clip")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="where the outputs go (default: the lab's delivery folder; inputs always come from the lab)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only ("-1": on Windows an empty value deletes the variable)
    out_dir = args.out_dir

    corpus = [json.loads(line) for line in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    speaker = {}
    speaker_path = OUT_DIR / "speaker_check.json"
    if speaker_path.is_file():
        speaker = json.loads(speaker_path.read_text(encoding="utf-8"))["clips"]
    else:
        print("note: speaker_check.json not found; run speaker_check.py first for identity flags", flush=True)

    def audio_for(row: dict) -> str:
        return row["audio"] if args.audio == "original" else str(LAB / "restored" / f"{row['id']}.wav")

    wanted = [r for r in corpus if r["split"] != "test" or args.include_test]
    # Per-clip measurements are cached by (analysis version, audio choice, clip id, sha256, text). The lab's cache is
    # always read; a run into another folder writes its additions next to its outputs, never into the lab.
    cache_name = f"clips_{args.audio}_v{ANALYSIS_VERSION}.json"
    cache_path = OUT_DIR / ".cache" / cache_name
    in_lab = out_dir.resolve() == OUT_DIR.resolve()
    write_cache_path = cache_path if in_lab else out_dir / ".cache" / cache_name
    # The cache also records which pitch range and syllable exceptions measured it (content/voice.json and
    # content/lexicon.json are data now, so an edit there must invalidate it); a cache written before the stamp
    # existed is adopted and, in the lab, stamped with the current settings.
    settings = settings_key()
    cache, adopted = {}, False
    for path in dict.fromkeys((cache_path, write_cache_path)):
        if path.is_file() and not args.no_cache:
            data = json.loads(path.read_text(encoding="utf-8"))
            stamp = data.pop("_settings", None)
            if stamp is not None and stamp != settings:
                print(f"note: {path} was measured with other pitch/lexicon settings: re-measuring", flush=True)
                continue
            adopted = adopted or stamp is None
            cache.update(data)

    def cache_key(row: dict) -> str:
        return f"{row['id']}|{row['sha256']}|{row['text']}"

    results: dict[str, dict] = {r["id"]: cache[cache_key(r)] for r in wanted if cache_key(r) in cache}
    jobs = [(r["id"], audio_for(r), r["text"]) for r in wanted if r["id"] not in results]
    if jobs:
        warm_jit()  # numba's cache is written by this process alone, never by several workers at once
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for index, (clip_id, features) in enumerate(pool.map(_work, jobs, chunksize=4), 1):
                results[clip_id] = features
                if index % 50 == 0:
                    print(f"analyzed {index}/{len(jobs)}", flush=True)
        cache.update({cache_key(r): results[r["id"]] for r in wanted if "error" not in results[r["id"]]})
    if jobs or (adopted and in_lab):
        write_cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_cache_path.write_text(json.dumps({**cache, "_settings": settings}), encoding="utf-8")
    print(f"clips: {len(wanted)} ({len(jobs)} measured, {len(wanted) - len(jobs)} from cache)", flush=True)
    results = {k: json.loads(json.dumps(v)) for k, v in results.items()}  # rows below must not alias the cache

    rows, test_rows = [], []
    for base in wanted:
        row = {k: base[k] for k in ("id", "split", "kind", "text", "text_source", "wer", "duration", "tts_usable",
                                    "directions", "delivery_tags")}
        row["audio"] = base["audio"]
        restored = LAB / "restored" / f"{base['id']}.wav"
        row["restored_audio"] = str(restored)
        row["restored_exists"] = restored.is_file()
        row["analyzed_audio"] = args.audio
        check = speaker.get(base["id"], {})
        row["speaker_flag"] = check.get("flag")
        row["speaker_sim_resemblyzer"] = check.get("resemblyzer_sim")
        row["speaker_sim_campplus"] = check.get("campplus_sim")
        row["speaker_z_resemblyzer"] = check.get("resemblyzer_z")
        row["speaker_z_campplus"] = check.get("campplus_z")
        row.update(results.get(base["id"], {"error": "not analyzed"}))
        (test_rows if base["split"] == "test" else rows).append(row)

    # His register centre comes from his own exact-text clips only (never from test clips).
    member_frames = [v for r in rows if profile_member(r) for v in r.get("_f0_hz", [])]
    center_hz = register_center(member_frames)
    for row in rows + test_rows:
        row.update(pitch_stats(row.pop("_f0_hz", []), row.pop("_f0_t", []), center_hz))
    profile = build_profile(rows, center_hz)
    add_z(rows, profile)
    add_z(test_rows, profile)
    suffix = "" if args.audio == "original" else ".restored"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"features{suffix}.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if test_rows:
        with (out_dir / f"features_test{suffix}.jsonl").open("w", encoding="utf-8") as stream:
            for row in test_rows:
                stream.write(json.dumps({**row, "eval_only": True}, ensure_ascii=False) + "\n")
    columns = ["id", "split", "kind", "text_source", "wer", "duration", "tts_usable", "speaker_flag",
               "speaker_sim_resemblyzer", "speaker_sim_campplus", "vox_reading", *NUMERIC, "clip_ratio", "peak_dbfs",
               "text_words", "text_syllables", "text_question", "text_exclamation", "text_ellipsis",
               "text_contractions", "text_formal_forms", f"text_{ADDRESS_FEATURE}", "directions", "delivery_tags",
               "text"]
    with (out_dir / f"features{suffix}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["; ".join(row.get(c) or []) if c in ("directions", "delivery_tags") else row.get(c)
                             for c in columns])
    profile["audio"] = args.audio
    (out_dir / f"voice_profile{suffix}.json").write_text(json.dumps(profile, indent=1), encoding="utf-8")
    errors = [r["id"] for r in rows if "error" in r]
    print(json.dumps({"rows": len(rows), "test_rows": len(test_rows), "errors": errors,
                      "profile_members": profile["members"], "profile_seconds": profile["seconds"],
                      "register": profile["register"]}, indent=1))


if __name__ == "__main__":
    main()
