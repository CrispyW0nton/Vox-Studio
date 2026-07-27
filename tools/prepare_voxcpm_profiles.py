from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


SAMPLE_RATE = 48000
WINDOW_SECONDS = 4.0
REFERENCE_SECONDS = 24.0
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}


@dataclass(frozen=True)
class ProfileSpec:
    voice_ids: tuple[str, ...]
    name: str
    sources: tuple[Path, ...]
    primary_prompt: Path | None = None
    primary_prompt_text: str = ""
    exclude_name_patterns: tuple[str, ...] = ()


def default_specs() -> tuple[ProfileSpec, ...]:
    documents = Path.home() / "Documents"
    voices = documents / "KotorMods" / "Voices"
    kotor_project = Path.home() / "Kotor.vox"
    local_app_data = Path(os.environ["LOCALAPPDATA"]) / "VoxStudio"
    alan_takes = tuple((kotor_project / "takes").rglob("*.opus"))
    return (
        ProfileSpec(
            ("jQ0cqfnHLpPATTWLuRCy",),
            "Xaria",
            (local_app_data / "voxcpm_seed_audio" / "xaria_reference.wav",),
        ),
        ProfileSpec(
            ("QD3JDvJNNRSe5ZhVM9hR", "i3UvU0lrwOz34jmla36c"),
            "Carth",
            (voices / "RvcDatasets" / "CarthDecoded",),
            voices / "RvcDatasets" / "CarthDecoded" / "nm02aacarb01000_.wav",
            (
                "Bastila, you're alive! Finally, things are looking up. "
                "Now we just need to figure out a way to get off this planet."
            ),
            ("nm01aacart*", "n_m1bncart*"),
        ),
        ProfileSpec(
            ("zsJfu6NHUhZIGZKxw0w0",),
            "Kreia",
            (
                voices / "KreiaTrainingData" / "kreia_voice_hq_ivc.mp3",
                voices / "RvcDatasets" / "Kreia",
            ),
        ),
        ProfileSpec(
            ("VqtR5ry1ddcv59m6Wvqg",),
            "Atton",
            (voices / "AttonTrainingData" / "atton_voice_cleaned.mp3",),
        ),
        ProfileSpec(
            ("0KRk8sPqojm2YNRCGKqu",),
            "Bao-Dur",
            (voices / "BaoDurTrainingData" / "baodur_voice_cleaned.mp3",),
        ),
        ProfileSpec(
            ("ubAJyJphmwzPmKNsS9W6",),
            "Alan Watts",
            alan_takes,
        ),
    )


def source_files(paths: tuple[Path, ...],
                 exclude_name_patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            files.extend(
                file
                for file in path.rglob("*")
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
            )
    return sorted(
        {
            file
            for file in files
            if not any(file.match(pattern) for pattern in exclude_name_patterns)
        }
    )


def load_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, sample_rate // divisor)
    return mono.astype(np.float32, copy=False)


def candidate_windows(files: list[Path]) -> list[tuple[float, np.ndarray, Path]]:
    window_frames = int(WINDOW_SECONDS * SAMPLE_RATE)
    candidates: list[tuple[float, np.ndarray, Path]] = []
    for path in files:
        try:
            audio = load_mono(path)
        except Exception as exception:
            print(f"skip {path}: {exception}")
            continue
        if len(audio) < SAMPLE_RATE:
            continue

        if len(audio) <= window_frames:
            starts = [0]
        else:
            count = min(18, max(3, int(len(audio) / window_frames)))
            starts = np.linspace(0, len(audio) - window_frames, count, dtype=int).tolist()

        for start in starts:
            clip = audio[start : start + window_frames]
            if len(clip) < SAMPLE_RATE:
                continue
            rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
            clipping = float(np.mean(np.abs(clip) > 0.985))
            active = float(np.mean(np.abs(clip) > 0.008))
            if rms < 0.008 or active < 0.25 or clipping > 0.01:
                continue
            target_rms = 10.0 ** (-22.0 / 20.0)
            score = abs(math.log(max(rms, 1e-6) / target_rms)) + clipping * 25.0
            candidates.append((score, clip.copy(), path))
    return candidates


def build_reference(
    files: list[Path],
    primary_prompt: Path | None,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray, Path]]]:
    candidates = candidate_windows(files)
    if not candidates:
        raise RuntimeError("No usable speech was found in the configured source files.")

    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[float, np.ndarray, Path]] = []
    used_paths: set[Path] = set()
    target_count = int(REFERENCE_SECONDS / WINDOW_SECONDS)
    if primary_prompt is not None and primary_prompt.exists():
        primary_audio = load_mono(primary_prompt)
        primary_audio, _ = librosa.effects.trim(primary_audio, top_db=38)
        selected.append((-1.0, primary_audio[: int(12.0 * SAMPLE_RATE)], primary_prompt))
        used_paths.add(primary_prompt)
    for candidate in candidates:
        if candidate[2] in used_paths:
            continue
        selected.append(candidate)
        used_paths.add(candidate[2])
        if len(selected) == target_count:
            break
    for candidate in candidates:
        if len(selected) == target_count:
            break
        if any(candidate is existing for existing in selected):
            continue
        selected.append(candidate)

    gap = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    for _, clip, _ in selected:
        peak = float(np.max(np.abs(clip)))
        if peak > 0.0:
            clip = clip * min(0.92 / peak, 2.5)
        parts.extend((clip, gap))
    reference = np.concatenate(parts[:-1])
    return reference, selected


_transcriber: WhisperModel | None = None


def transcriber() -> WhisperModel:
    global _transcriber
    if _transcriber is None:
        engine_root = Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "engines" / "voxcpm2"
        cache_root = engine_root / "cache" / "faster-whisper"
        try:
            _transcriber = WhisperModel(
                "base.en",
                device="cuda",
                compute_type="float16",
                download_root=str(cache_root),
            )
        except Exception:
            _transcriber = WhisperModel(
                "base.en",
                device="cpu",
                compute_type="int8",
                download_root=str(cache_root),
            )
    return _transcriber


def transcribe_clip(clip: np.ndarray) -> str:
    divisor = math.gcd(SAMPLE_RATE, 16000)
    audio_16k = resample_poly(clip, 16000 // divisor, SAMPLE_RATE // divisor).astype(
        np.float32,
        copy=False,
    )
    segments, _ = transcriber().transcribe(
        audio_16k,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return re.sub(r"\s+", " ", " ".join(segment.text.strip() for segment in segments)).strip()


def style_features(clip: np.ndarray, transcript: str) -> dict[str, float]:
    duration = max(len(clip) / SAMPLE_RATE, 0.1)
    rms = librosa.feature.rms(y=clip, frame_length=2048, hop_length=480)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)
    active_rms = rms_db[rms_db > -45.0]
    dynamic_db = (
        float(np.percentile(active_rms, 90) - np.percentile(active_rms, 10))
        if active_rms.size
        else 0.0
    )
    f0 = librosa.yin(
        clip,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=SAMPLE_RATE,
        frame_length=2048,
        hop_length=480,
    )
    finite_f0 = f0[np.isfinite(f0)]
    pitch_variation = (
        float(np.std(finite_f0) / max(np.mean(finite_f0), 1.0))
        if finite_f0.size
        else 0.0
    )
    spoken_characters = len(re.sub(r"[^A-Za-z0-9]", "", transcript))
    return {
        "duration_seconds": round(duration, 4),
        "dynamic_db": round(dynamic_db, 4),
        "pitch_variation": round(pitch_variation, 6),
        "characters_per_second": round(spoken_characters / duration, 4),
    }


def write_profile(root: Path, spec: ProfileSpec) -> None:
    files = source_files(spec.sources, spec.exclude_name_patterns)
    if not files:
        print(f"skip {spec.name}: no local source audio")
        return

    reference, selected = build_reference(files, spec.primary_prompt)
    style_anchors: list[dict] = []
    style_clips: list[np.ndarray] = []
    for index, (_, clip, source) in enumerate(selected):
        prompt_text = (
            spec.primary_prompt_text
            if index == 0 and spec.primary_prompt is not None and source == spec.primary_prompt
            else transcribe_clip(clip)
        )
        if not prompt_text:
            continue
        style_anchors.append(
            {
                "audio": f"styles/style_{index + 1:02d}.wav",
                "prompt_text": prompt_text,
                "source": str(source),
                "features": style_features(clip, prompt_text),
            }
        )
        style_clips.append(clip)
    total_source_seconds = 0.0
    for path in files:
        try:
            total_source_seconds += float(sf.info(path).duration)
        except Exception:
            pass

    for voice_id in spec.voice_ids:
        profile_root = root / voice_id
        profile_root.mkdir(parents=True, exist_ok=True)
        (profile_root / "styles").mkdir(parents=True, exist_ok=True)
        sf.write(profile_root / "reference.wav", reference, SAMPLE_RATE, subtype="PCM_16")
        for anchor, clip in zip(style_anchors, style_clips, strict=True):
            sf.write(
                profile_root / anchor["audio"],
                clip,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
        profile = {
            "format_version": 2,
            "voice_id": voice_id,
            "name": spec.name,
            "engine": "VoxCPM2",
            "reference_audio": "reference.wav",
            "source_file_count": len(files),
            "source_duration_seconds": round(total_source_seconds, 2),
            "selected_sources": [str(item[2]) for item in selected],
            "style_anchors": style_anchors,
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }
        (profile_root / "profile.json").write_text(
            json.dumps(profile, indent=2),
            encoding="utf-8",
        )
        print(
            f"built {spec.name} [{voice_id}] from {len(files)} files "
            f"({total_source_seconds:.1f}s available)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "voxcpm_profiles",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for spec in default_specs():
        try:
            write_profile(arguments.output, spec)
        except Exception as exception:
            print(f"skip {spec.name}: {exception}")


if __name__ == "__main__":
    main()
