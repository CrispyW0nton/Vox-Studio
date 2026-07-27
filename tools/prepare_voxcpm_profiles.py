from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import medfilt, resample_poly


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
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    style_anchor_count: int = 6
    control_instruction: str = ""
    use_controlled_cloning: bool = False
    controlled_cfg_value: float = 0.0
    lora_training_name: str = ""
    use_stable_character_identity: bool = False


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
            style_anchor_count=316,
            control_instruction=(
                "Keep Carth's clear, earnest military timbre. Match the "
                "performer without adding intensity."
            ),
            use_controlled_cloning=True,
            controlled_cfg_value=2.0,
            lora_training_name="carth",
            use_stable_character_identity=True,
        ),
        ProfileSpec(
            ("zsJfu6NHUhZIGZKxw0w0",),
            "Kreia",
            (voices / "RvcDatasets" / "KreiaDecoded101",),
            voices / "RvcDatasets" / "KreiaDecoded101" / "101101kreia030.wav",
            (
                "I am Kreia, and I am your rescuer, as you are mine. "
                "Tell me, do you recall what happened?"
            ),
            cfg_value=2.5,
        ),
        ProfileSpec(
            ("VqtR5ry1ddcv59m6Wvqg",),
            "Atton",
            (voices / "RvcDatasets" / "AttonConversational",),
            voices / "RvcDatasets" / "AttonConversational" / "101101atton001.wav",
            (
                "Nice outfit. What, you miners change regulation uniforms "
                "while I've been in here?"
            ),
            cfg_value=1.5,
            style_anchor_count=68,
            control_instruction=(
                "Keep Atton's guarded youthful timbre and casual drawl. Match "
                "the performer without adding sarcasm or tension."
            ),
            use_controlled_cloning=True,
            controlled_cfg_value=3.0,
        ),
        ProfileSpec(
            ("0KRk8sPqojm2YNRCGKqu",),
            "Bao-Dur",
            (voices / "RvcDatasets" / "BaoDurDecoded",),
            voices / "RvcDatasets" / "BaoDurDecoded" / "gblbaodur003.wav",
            (
                "Uh, General, I think you've got more important things to worry "
                "about right now than talking to me."
            ),
            cfg_value=1.5,
            style_anchor_count=328,
            control_instruction=(
                "Keep Bao-Dur's low, breathy, gentle timbre. Match the performer "
                "without flattening emphasis or adding fatigue."
            ),
            use_controlled_cloning=True,
            controlled_cfg_value=1.5,
            lora_training_name="bao-dur",
            use_stable_character_identity=True,
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


def candidate_windows(files: list[Path]) -> list[tuple[float, int, int, Path]]:
    window_frames = int(WINDOW_SECONDS * SAMPLE_RATE)
    complete_line_frames = int(12.0 * SAMPLE_RATE)
    candidates: list[tuple[float, int, int, Path]] = []
    for path in files:
        try:
            audio = load_mono(path)
        except Exception as exception:
            print(f"skip {path}: {exception}")
            continue
        if len(audio) < SAMPLE_RATE:
            continue

        if len(audio) <= complete_line_frames:
            starts = [0]
            candidate_frames = len(audio)
        else:
            count = min(18, max(3, int(len(audio) / window_frames)))
            starts = np.linspace(0, len(audio) - window_frames, count, dtype=int).tolist()
            candidate_frames = window_frames

        for start in starts:
            clip = audio[start : start + candidate_frames]
            if len(clip) < SAMPLE_RATE:
                continue
            rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
            clipping = float(np.mean(np.abs(clip) > 0.985))
            active = float(np.mean(np.abs(clip) > 0.008))
            if rms < 0.008 or active < 0.25 or clipping > 0.01:
                continue
            target_rms = 10.0 ** (-22.0 / 20.0)
            score = abs(math.log(max(rms, 1e-6) / target_rms)) + clipping * 25.0
            candidates.append((score, start, len(clip), path))
    return candidates


def build_reference(
    files: list[Path],
    primary_prompt: Path | None,
    style_anchor_count: int,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray, Path]]]:
    candidates = candidate_windows(files)
    if not candidates:
        raise RuntimeError("No usable speech was found in the configured source files.")

    candidates.sort(key=lambda item: item[0])
    selected_candidates: list[tuple[float, int, int, Path]] = []
    used_paths: set[Path] = set()
    target_count = max(1, style_anchor_count)
    selected: list[tuple[float, np.ndarray, Path]] = []
    if primary_prompt is not None and primary_prompt.exists():
        primary_audio = load_mono(primary_prompt)
        primary_audio, _ = librosa.effects.trim(primary_audio, top_db=38)
        selected.append((-1.0, primary_audio[: int(12.0 * SAMPLE_RATE)], primary_prompt))
        used_paths.add(primary_prompt)
    for candidate in candidates:
        if candidate[3] in used_paths:
            continue
        selected_candidates.append(candidate)
        used_paths.add(candidate[3])
        if len(selected) + len(selected_candidates) == target_count:
            break
    for candidate in candidates:
        if len(selected) + len(selected_candidates) == target_count:
            break
        if candidate in selected_candidates:
            continue
        selected_candidates.append(candidate)

    for score, start, frame_count, path in selected_candidates:
        audio = load_mono(path)
        clip, _ = librosa.effects.trim(audio[start : start + frame_count], top_db=38)
        selected.append((score, clip, path))

    gap = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
    parts: list[np.ndarray] = []
    reference_count = int(REFERENCE_SECONDS / WINDOW_SECONDS)
    for _, clip, _ in selected[:reference_count]:
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
    clip, _ = librosa.effects.trim(clip, top_db=38)
    duration = max(len(clip) / SAMPLE_RATE, 0.1)
    rms = librosa.feature.rms(y=clip, frame_length=2048, hop_length=480)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)
    active_rms = rms_db[rms_db > -30.0]
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
    frame_count = min(len(f0), len(rms_db))
    frame_positions = np.linspace(0.0, 1.0, frame_count)
    analysis_rms = rms_db[:frame_count]
    voiced = (
        np.isfinite(f0[:frame_count])
        & (f0[:frame_count] < librosa.note_to_hz("C6") * 0.98)
        & (analysis_rms > -38.0)
    )
    finite_f0 = f0[:frame_count][voiced]
    voiced_positions = frame_positions[voiced]
    pitch_variation_semitones = 0.0
    pitch_range = 0.0
    pitch_slope = 0.0
    terminal_pitch_delta = 0.0
    if finite_f0.size >= 3:
        raw_semitones = 12.0 * np.log2(finite_f0)
        kernel_size = 5 if raw_semitones.size >= 5 else 3
        local_median = medfilt(raw_semitones, kernel_size=kernel_size)
        semitones = raw_semitones - (
            12.0 * np.round((raw_semitones - local_median) / 12.0)
        )
        semitones = medfilt(semitones, kernel_size=kernel_size)
        center = float(np.median(semitones))
        plausible = np.abs(semitones - center) < 10.0
        semitones = semitones[plausible] - center
        voiced_positions = voiced_positions[plausible]
        if semitones.size >= 3:
            pitch_variation_semitones = float(np.std(semitones))
            pitch_range = float(
                np.percentile(semitones, 90) - np.percentile(semitones, 10)
            )
            pitch_slope = float(np.polyfit(voiced_positions, semitones, 1)[0])
            opening = semitones[voiced_positions <= 0.35]
            ending = semitones[voiced_positions >= 0.65]
            terminal_pitch_delta = (
                float(np.median(ending) - np.median(opening))
                if opening.size and ending.size
                else 0.0
            )
    pause_ratio = float(np.mean(rms_db < -32.0)) if rms_db.size else 0.0
    first_energy = analysis_rms[frame_positions <= 0.35]
    last_energy = analysis_rms[frame_positions >= 0.65]
    energy_slope = (
        float(np.median(last_energy) - np.median(first_energy))
        if first_energy.size and last_energy.size
        else 0.0
    )
    spoken_characters = len(re.sub(r"[^A-Za-z0-9]", "", transcript))
    return {
        "duration_seconds": round(duration, 4),
        "dynamic_db": round(dynamic_db, 4),
        "pitch_variation_semitones": round(pitch_variation_semitones, 4),
        "pitch_range_semitones": round(pitch_range, 4),
        "pitch_slope_semitones": round(pitch_slope, 4),
        "terminal_pitch_delta": round(terminal_pitch_delta, 4),
        "pause_ratio": round(pause_ratio, 6),
        "energy_slope_db": round(energy_slope, 4),
        "characters_per_second": round(spoken_characters / duration, 4),
    }


def transcript_cache(root: Path, spec: ProfileSpec) -> dict[str, str]:
    cached: dict[str, str] = {}
    for voice_id in spec.voice_ids:
        profile_path = root / voice_id / "profile.json"
        if not profile_path.exists():
            continue
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for anchor in profile.get("style_anchors", []):
            source = str(anchor.get("source", "")).strip()
            prompt_text = str(anchor.get("prompt_text", "")).strip()
            if source and prompt_text:
                cached[source.casefold()] = prompt_text
    return cached


def write_profile(root: Path, spec: ProfileSpec) -> None:
    files = source_files(spec.sources, spec.exclude_name_patterns)
    if not files:
        print(f"skip {spec.name}: no local source audio")
        return

    reference, selected = build_reference(
        files,
        spec.primary_prompt,
        spec.style_anchor_count,
    )
    cached_transcripts = transcript_cache(root, spec)
    style_anchors: list[dict] = []
    style_clips: list[np.ndarray] = []
    for index, (_, clip, source) in enumerate(selected):
        prompt_text = (
            spec.primary_prompt_text
            if index == 0 and spec.primary_prompt is not None and source == spec.primary_prompt
            else cached_transcripts.get(str(source).casefold()) or transcribe_clip(clip)
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
        adapter_installed = False
        if spec.lora_training_name:
            checkpoints_root = (
                Path(os.environ["LOCALAPPDATA"])
                / "VoxStudio"
                / "voxcpm_training"
                / spec.lora_training_name
                / "checkpoints"
            )
            checkpoint_root = next(
                (
                    candidate
                    for candidate in (
                        checkpoints_root / "best",
                        checkpoints_root / "latest",
                    )
                    if (candidate / "lora_weights.safetensors").exists()
                    and (candidate / "lora_config.json").exists()
                ),
                checkpoints_root / "latest",
            )
            weights_path = checkpoint_root / "lora_weights.safetensors"
            config_path = checkpoint_root / "lora_config.json"
            if weights_path.exists() and config_path.exists():
                adapter_root = profile_root / "lora"
                adapter_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(weights_path, adapter_root / weights_path.name)
                shutil.copy2(config_path, adapter_root / config_path.name)
                adapter_installed = True
        profile_reference = selected[0][1] if adapter_installed else reference
        if adapter_installed:
            profile_reference = profile_reference.copy()
            reference_peak = float(np.max(np.abs(profile_reference)))
            if reference_peak > 0.0:
                profile_reference *= min(0.92 / reference_peak, 2.5)
        sf.write(
            profile_root / "reference.wav",
            profile_reference,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        for anchor, clip in zip(style_anchors, style_clips, strict=True):
            sf.write(
                profile_root / anchor["audio"],
                clip,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
        profile = {
            "format_version": 5,
            "voice_id": voice_id,
            "name": spec.name,
            "engine": "VoxCPM2",
            "reference_audio": "reference.wav",
            "source_file_count": len(files),
            "source_duration_seconds": round(total_source_seconds, 2),
            "selected_sources": [str(item[2]) for item in selected],
            "style_anchors": style_anchors,
            "cfg_value": spec.cfg_value,
            "inference_timesteps": spec.inference_timesteps,
            "control_instruction": spec.control_instruction,
            "use_controlled_cloning": spec.use_controlled_cloning,
            "controlled_cfg_value": spec.controlled_cfg_value,
            "lora_adapter": "lora" if adapter_installed else "",
            "use_stable_character_identity": (
                spec.use_stable_character_identity and adapter_installed
            ),
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
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help="Build only the named profile; may be supplied more than once.",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    requested_voices = {name.casefold() for name in arguments.voice}
    specs = tuple(
        spec
        for spec in default_specs()
        if not requested_voices or spec.name.casefold() in requested_voices
    )
    if arguments.voice and not specs:
        parser.error(f"no matching profile for: {', '.join(arguments.voice)}")
    for spec in specs:
        try:
            write_profile(arguments.output, spec)
        except Exception as exception:
            print(f"skip {spec.name}: {exception}")


if __name__ == "__main__":
    main()
