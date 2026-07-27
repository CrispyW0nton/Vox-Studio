from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
import threading
import time
import urllib.parse
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from faster_whisper import WhisperModel
from scipy.signal import medfilt
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig
from vox_delivery import (
    DeliveryReading,
    apply_pronunciations,
    build_transcription_hotwords,
    canonicalize_transcription,
    delivery_distance,
    detect_delivery,
    load_pronunciations,
    specialize_delivery,
    text_delivery_target,
)
from vox_profiles import (
    activate_lora,
    control_identity_instruction,
    generation_options,
    resolve_profile_asset,
    synthesis_inputs,
    text_identity_instruction,
)
from vox_text import (
    STORYTELLING_MODE,
    delivery_instruction,
    normalized_delivery_tag,
    normalized_performance_mode,
    plan_story_performance,
    split_text_for_synthesis,
    story_beat_instruction,
    synthesis_seed,
)


ENGINE_ROOT = Path(
    os.environ.get("VOX_VOXCPM_ENGINE_ROOT")
    or Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "engines" / "voxcpm2"
)
PROFILE_ROOT = Path(
    os.environ.get("VOX_VOXCPM_PROFILE_ROOT")
    or Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "voxcpm_profiles"
)

os.environ.setdefault("HF_HOME", str(ENGINE_ROOT / "cache"))

app = FastAPI(title="Vox Studio VoxCPM2", version="1.2")
model_lock = threading.Lock()
transcriber_lock = threading.Lock()
delivery_lock = threading.Lock()
text_render_lock = asyncio.Lock()
model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VoxCPM2")
tts_model: VoxCPM | None = None
active_lora: dict[str, object] = {"loaded_signature": None, "enabled": False}
transcriber: WhisperModel | None = None
transcriber_model_name = ""
delivery_history: deque[dict[str, float]] = deque(maxlen=24)
pronunciations = load_pronunciations(Path(__file__).with_name("pronunciations.json"))
transcription_hotwords = build_transcription_hotwords(
    pronunciations,
    (
        "Star Wars",
        "Knights of the Old Republic",
        "KOTOR",
        "Ebon Hawk",
    ),
)


@dataclass(frozen=True)
class TextSynthesisSection:
    text: str
    prompt_path: Path
    prompt_text: str
    reference_path: Path
    instruction: str
    pause_after: float
    seed: int


torch.set_float32_matmul_precision("high")


def profile_directories() -> list[Path]:
    if not PROFILE_ROOT.exists():
        return []
    return [
        path
        for path in PROFILE_ROOT.iterdir()
        if path.is_dir() and (path / "profile.json").exists()
    ]


def load_profile(voice_id: str) -> tuple[dict, Path, Path]:
    safe_voice_id = re.sub(r"[^A-Za-z0-9_-]", "", voice_id)
    if not safe_voice_id or safe_voice_id != voice_id:
        raise HTTPException(status_code=400, detail="Invalid character profile id.")

    profile_dir = PROFILE_ROOT / safe_voice_id
    profile_path = profile_dir / "profile.json"
    if not profile_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No local VoxCPM2 profile is installed for this character. "
                "Build or import a clean character reference first."
            ),
        )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reference_path = (profile_dir / profile["reference_audio"]).resolve()
    if profile_dir.resolve() not in reference_path.parents or not reference_path.exists():
        raise HTTPException(status_code=500, detail="Character reference audio is missing.")
    return profile, profile_dir, reference_path


def get_tts_model() -> VoxCPM:
    global tts_model
    if tts_model is None:
        tts_model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            load_denoiser=False,
            cache_dir=str(ENGINE_ROOT / "cache"),
            optimize=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lora_config=LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
                r=32,
                alpha=32,
                dropout=0.0,
            ),
        )
    return tts_model


def get_transcriber() -> WhisperModel:
    global transcriber, transcriber_model_name
    with transcriber_lock:
        if transcriber is None:
            cache_root = ENGINE_ROOT / "cache" / "faster-whisper"
            cache_root.mkdir(parents=True, exist_ok=True)
            preferred_model = (
                os.environ.get("VOXSTUDIO_WHISPER_MODEL", "turbo").strip()
                or "turbo"
            )
            attempts: list[tuple[str, str, str]] = []
            if torch.cuda.is_available():
                attempts.extend(
                    (
                        (preferred_model, "cuda", "float16"),
                        ("base.en", "cuda", "float16"),
                    )
                )
            attempts.append(("base.en", "cpu", "int8"))
            last_error: Exception | None = None
            tried: set[tuple[str, str, str]] = set()
            for model_name, device, compute_type in attempts:
                attempt = (model_name, device, compute_type)
                if attempt in tried:
                    continue
                tried.add(attempt)
                try:
                    transcriber = WhisperModel(
                        model_name,
                        device=device,
                        compute_type=compute_type,
                        download_root=str(cache_root),
                    )
                    transcriber_model_name = model_name
                    break
                except Exception as exception:
                    last_error = exception
                    print(
                        f"Whisper {model_name} on {device} unavailable: {exception}",
                        flush=True,
                    )
            if transcriber is None:
                raise RuntimeError(
                    "No Whisper transcription model could load."
                ) from last_error
    return transcriber


def warm_models() -> None:
    try:
        get_transcriber()
        with model_lock:
            get_tts_model()
    except Exception as exception:
        print(f"VoxCPM2 warmup failed: {exception}", flush=True)


@app.on_event("startup")
def start_warmup() -> None:
    model_executor.submit(warm_models)


def write_pcm_wave(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def transcribe_performance(path: Path) -> str:
    segments, _ = get_transcriber().transcribe(
        str(path),
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        hotwords=transcription_hotwords,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return canonicalize_transcription(re.sub(r"\s+", " ", text), pronunciations)


def style_features(path: Path, transcript: str) -> dict[str, float]:
    clip, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = clip.mean(axis=1)
    mono, _ = librosa.effects.trim(mono, top_db=38)
    duration = max(len(mono) / sample_rate, 0.1)
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=480)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)
    active_rms = rms_db[rms_db > -30.0]
    dynamic_db = (
        float(np.percentile(active_rms, 90) - np.percentile(active_rms, 10))
        if active_rms.size
        else 0.0
    )
    f0 = librosa.yin(
        mono,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sample_rate,
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
        "dynamic_db": dynamic_db,
        "pitch_variation_semitones": pitch_variation_semitones,
        "pitch_range_semitones": pitch_range,
        "pitch_slope_semitones": pitch_slope,
        "terminal_pitch_delta": terminal_pitch_delta,
        "pause_ratio": pause_ratio,
        "energy_slope_db": energy_slope,
        "characters_per_second": spoken_characters / duration,
    }


def performance_delivery(
    features: dict[str, float],
    transcript: str,
    delivery_profile: str = "",
) -> DeliveryReading:
    with delivery_lock:
        reading = detect_delivery(features, transcript, tuple(delivery_history))
        delivery_history.append(dict(features))
    return specialize_delivery(delivery_profile, reading, features, transcript)


def select_style_anchor(
    profile: dict,
    profile_dir: Path,
    performance: dict[str, float],
    delivery: DeliveryReading,
    transcript: str,
) -> tuple[Path, str, str] | None:
    anchors = profile.get("style_anchors", [])
    if not anchors:
        return None

    def distance(anchor: dict) -> float:
        features = anchor.get("features", {})
        anchor_text = str(anchor.get("prompt_text", ""))
        anchor_delivery = detect_delivery(features, anchor_text)
        return delivery_distance(
            performance,
            features,
            delivery,
            anchor_delivery,
            transcript,
            anchor_text,
        )

    selected = min(anchors, key=distance)
    selected_path = (profile_dir / selected["audio"]).resolve()
    if profile_dir.resolve() not in selected_path.parents or not selected_path.exists():
        raise HTTPException(status_code=500, detail="Character style anchor is missing.")
    return (
        selected_path,
        str(selected["prompt_text"]),
        str(selected.get("source", selected_path.name)),
    )


def build_control_instruction(
    profile: dict,
    performance: dict[str, float],
    delivery: DeliveryReading,
) -> str:
    base = control_identity_instruction(profile)
    if not profile.get("use_controlled_cloning", False) or not base:
        return ""

    delivery_controls = {
        "calm": "Calm, even delivery.",
        "measured": "Measured, deliberate delivery.",
        "neutral": "Natural, neutral delivery.",
        "reflective": "Warm, reflective restraint; preserve the sense of memory.",
        "wry": "Dry skepticism and restrained exasperation; never playful.",
        "guarded": "Guarded suspicion with clipped restraint; do not add anger.",
        "wounded": "Contained hurt and vulnerability; avoid melodrama.",
        "warm": "Earnest, understated warmth and relief.",
        "disapproving": "Firm moral disapproval with controlled frustration.",
        "resolute": "Steady protective resolve and military focus.",
        "emphatic": "Use only the performer's emphasis.",
        "urgent": "Match the performer's urgency without exceeding it.",
        "questioning": "Preserve the performer's questioning cadence.",
        "sarcastic": "Dry, controlled sarcasm; no broad comedy.",
    }
    details = [delivery_controls.get(delivery.label, "Match the performer's delivery.")]
    if performance["pitch_slope_semitones"] > 3.0:
        details.append(
            "Gradual rise through the line."
        )
    elif performance["pitch_slope_semitones"] < -3.0:
        details.append("Settling cadence toward the ending.")
    if performance["pause_ratio"] >= 0.18:
        details.append("Keep the meaningful pauses.")
    elif performance["pause_ratio"] <= 0.05:
        details.append("Connected phrasing; do not invent pauses.")
    pace = performance["characters_per_second"]
    if pace >= 20.0:
        details.append("Fast pace.")
    elif pace >= 16.0:
        details.append("Brisk pace.")
    elif pace <= 10.0:
        details.append("Slow pace.")
    elif pace <= 13.0:
        details.append("Deliberate pace.")
    else:
        details.append("Moderate pace.")
    pitch_range = performance["pitch_range_semitones"]
    if pitch_range >= 10.0:
        details.append("Wide pitch movement.")
    elif pitch_range >= 6.0:
        details.append("Moderate pitch movement.")
    else:
        details.append("Restrained pitch movement.")
    if performance["dynamic_db"] >= 16.0:
        details.append("Strong dynamic contrast.")
    elif performance["dynamic_db"] >= 10.0:
        details.append("Natural dynamic contrast.")
    elif performance["dynamic_db"] <= 8.0:
        details.append("Even volume.")
    if performance["energy_slope_db"] >= 3.0:
        details.append("Build energy toward the ending.")
    elif performance["energy_slope_db"] <= -3.0:
        details.append("Settle the energy toward the ending.")
    if performance["terminal_pitch_delta"] >= 2.5:
        details.append("Rising ending.")
    elif performance["terminal_pitch_delta"] <= -2.5:
        details.append("Falling ending.")
    return " ".join((base, *details))


def synthesize(
    spoken_text: str,
    prompt_path: Path,
    prompt_text: str,
    reference_path: Path,
    profile: dict,
    control_instruction: str,
    lora_path: Path | None,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    with model_lock:
        model = get_tts_model()
        activate_lora(model, lora_path, active_lora)
        torch.manual_seed(seed)
        options = generation_options(
            spoken_text,
            prompt_path,
            prompt_text,
            reference_path,
            profile,
            control_instruction,
        )
        rendered = model.generate(**options)
        return rendered, int(model.tts_model.sample_rate)


def text_control_instruction(profile: dict, delivery_tag: str) -> str:
    return " ".join(
        value
        for value in (
            text_identity_instruction(profile),
            delivery_instruction(delivery_tag),
        )
        if value
    )


def text_synthesis_section(
    profile: dict,
    profile_dir: Path,
    reference_path: Path,
    text: str,
    delivery_tag: str,
    instruction: str,
    pause_after: float,
    seed: int,
) -> TextSynthesisSection:
    target_features, target_delivery = text_delivery_target(delivery_tag)
    style_anchor = select_style_anchor(
        profile,
        profile_dir,
        target_features,
        target_delivery,
        text,
    )
    if style_anchor is None:
        return TextSynthesisSection(
            text=text,
            prompt_path=reference_path,
            prompt_text="",
            reference_path=reference_path,
            instruction=instruction,
            pause_after=pause_after,
            seed=seed,
        )

    anchor_path, anchor_text, _ = style_anchor
    return TextSynthesisSection(
        text=text,
        prompt_path=anchor_path,
        prompt_text=anchor_text,
        reference_path=anchor_path,
        instruction="",
        pause_after=pause_after,
        seed=seed,
    )


def trim_text_audio_silence(
    audio: np.ndarray,
    sample_rate: int,
    frame_milliseconds: int = 10,
    leading_padding_milliseconds: int = 20,
    trailing_padding_milliseconds: int = 32,
) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_samples = max(1, int(sample_rate * frame_milliseconds / 1000))
    frame_count = len(output) // frame_samples
    if frame_count < 3:
        return output.copy()

    framed = output[: frame_count * frame_samples].reshape(frame_count, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(framed), axis=1))
    reference_rms = float(np.percentile(frame_rms, 90))
    threshold = max(0.0008, min(0.006, reference_rms * 0.04))
    active_frames = np.flatnonzero(frame_rms >= threshold)
    if active_frames.size == 0:
        return output.copy()

    leading_padding = int(sample_rate * leading_padding_milliseconds / 1000)
    trailing_padding = int(sample_rate * trailing_padding_milliseconds / 1000)
    start = max(0, int(active_frames[0]) * frame_samples - leading_padding)
    end = min(
        len(output),
        (int(active_frames[-1]) + 1) * frame_samples + trailing_padding,
    )
    return output[start:end].copy()


def finished_text_audio(
    audio: np.ndarray,
    sample_rate: int,
    fade_milliseconds: int = 8,
) -> np.ndarray:
    output = trim_text_audio_silence(audio, sample_rate)
    fade_samples = min(
        int(sample_rate * fade_milliseconds / 1000),
        len(output) // 2,
    )
    if fade_samples > 0:
        output[:fade_samples] *= np.linspace(
            0.0,
            1.0,
            fade_samples,
            dtype=np.float32,
        )
        output[-fade_samples:] *= np.linspace(
            1.0,
            0.0,
            fade_samples,
            dtype=np.float32,
        )
    return output


def text_pause_seconds(section: str) -> float:
    stripped = section.rstrip()
    if stripped.endswith("."):
        return 0.16
    if stripped.endswith("?"):
        return 0.13
    if stripped.endswith("!"):
        return 0.10
    return 0.08


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "engine": "VoxCPM2",
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": tts_model is not None,
            "transcriber_loaded": transcriber is not None,
            "transcriber_model": transcriber_model_name,
            "profile_count": len(profile_directories()),
            "message": "VoxCPM2 phrase-live service is ready.",
        }
    )


@app.post("/render_performance")
def render_performance(
    audio: UploadFile = File(...),
    voice_id: str = Form(...),
    sample_rate: int = Form(16000),
    channels: int = Form(1),
    transcript: str = Form(""),
) -> Response:
    if sample_rate < 8000 or sample_rate > 192000 or channels != 1:
        raise HTTPException(status_code=400, detail="Performance audio must be mono PCM16.")

    pcm = audio.file.read()
    if len(pcm) < sample_rate // 5 * 2 or len(pcm) % 2:
        raise HTTPException(status_code=400, detail="Performance phrase is too short.")

    profile, profile_dir, reference_path = load_profile(voice_id)
    try:
        lora_path = resolve_profile_asset(profile, profile_dir, "lora_adapter")
    except ValueError as exception:
        raise HTTPException(status_code=500, detail=str(exception)) from exception
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="voxstudio_voxcpm_") as temp_dir:
        prompt_path = Path(temp_dir) / "performance.wav"
        write_pcm_wave(prompt_path, pcm, sample_rate, channels)

        spoken_text = re.sub(r"\s+", " ", transcript).strip()
        if not spoken_text:
            spoken_text = transcribe_performance(prompt_path)
        if not spoken_text:
            raise HTTPException(
                status_code=422,
                detail="No speech was recognized. Speak a complete line and try again.",
            )

        performance = style_features(prompt_path, spoken_text)
        delivery = performance_delivery(
            performance,
            spoken_text,
            str(profile.get("delivery_profile", "")),
        )
        style_anchor = (
            None
            if profile.get("use_stable_character_identity", False)
            else select_style_anchor(
                profile,
                profile_dir,
                performance,
                delivery,
                spoken_text,
            )
        )
        inputs = synthesis_inputs(
            profile,
            prompt_path,
            spoken_text,
            reference_path,
            style_anchor,
        )

        control_instruction = build_control_instruction(profile, performance, delivery)
        if (
            style_anchor is not None
            and not profile.get("use_stable_character_identity", False)
            and delivery.label in {
                "emphatic",
                "urgent",
                "questioning",
                "sarcastic",
            }
        ):
            control_instruction = ""
        pronunciation = apply_pronunciations(spoken_text, pronunciations)
        rendered, output_sample_rate = model_executor.submit(
            synthesize,
            pronunciation.text,
            inputs.prompt_path,
            inputs.prompt_text,
            inputs.reference_path,
            profile,
            control_instruction,
            lora_path,
        ).result()

    output = np.clip(rendered, -1.0, 1.0)
    output_pcm = (output * 32767.0).astype("<i2").tobytes()
    latency_ms = int((time.perf_counter() - started) * 1000.0)
    headers = {
        "X-Vox-Sample-Rate": str(output_sample_rate),
        "X-Vox-Latency-Ms": str(latency_ms),
        "X-Vox-Character": urllib.parse.quote(str(profile.get("name", voice_id))),
        "X-Vox-Transcript": urllib.parse.quote(spoken_text),
        "X-Vox-Style-Source": urllib.parse.quote(inputs.style_source),
        "X-Vox-Delivery": delivery.label,
        "X-Vox-Pronunciations": urllib.parse.quote(
            ", ".join(pronunciation.matched_terms)
        ),
        "X-Vox-Synthesis-Mode": "controlled" if control_instruction else "continuation",
        "X-Vox-Adapter": "trained" if lora_path else "base",
        "Cache-Control": "no-store",
    }
    return Response(output_pcm, media_type="audio/L16", headers=headers)


@app.post("/render_text")
async def render_text(
    request: Request,
    voice_id: str = Form(...),
    text: str = Form(...),
    delivery: str = Form("natural"),
    mode: str = Form("standard"),
) -> Response:
    spoken_text = re.sub(r"\s+", " ", text).strip()
    if not spoken_text:
        raise HTTPException(status_code=400, detail="Enter text to synthesize.")
    if len(spoken_text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Text-to-speech captures are limited to 10,000 characters.",
        )

    profile, profile_dir, reference_path = load_profile(voice_id)
    try:
        lora_path = resolve_profile_asset(profile, profile_dir, "lora_adapter")
    except ValueError as exception:
        raise HTTPException(status_code=500, detail=str(exception)) from exception

    delivery_tag = normalized_delivery_tag(delivery)
    performance_mode = normalized_performance_mode(mode)
    story_plan = (
        plan_story_performance(spoken_text, delivery_tag)
        if performance_mode == STORYTELLING_MODE
        else None
    )
    if story_plan is not None:
        synthesis_sections = tuple(
            text_synthesis_section(
                profile=profile,
                profile_dir=profile_dir,
                reference_path=reference_path,
                text=beat.text,
                delivery_tag=beat.delivery,
                instruction=" ".join(
                    value
                    for value in (
                        text_identity_instruction(profile),
                        story_beat_instruction(beat, delivery_tag),
                    )
                    if value
                ),
                pause_after=beat.pause_after,
                seed=synthesis_seed(index, beat.text),
            )
            for index, beat in enumerate(story_plan.beats)
        )
    else:
        sections = split_text_for_synthesis(spoken_text)
        instruction = text_control_instruction(profile, delivery_tag)
        synthesis_sections = tuple(
            text_synthesis_section(
                profile=profile,
                profile_dir=profile_dir,
                reference_path=reference_path,
                text=section,
                delivery_tag=delivery_tag,
                instruction=instruction,
                pause_after=text_pause_seconds(section),
                seed=synthesis_seed(index, section),
            )
            for index, section in enumerate(sections)
        )

    if text_render_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Another text performance is still rendering.",
        )

    started = time.perf_counter()
    output_pcm = bytearray()
    matched_pronunciations: list[str] = []
    output_sample_rate = 0

    async with text_render_lock:
        for index, section in enumerate(synthesis_sections):
            if await request.is_disconnected():
                raise HTTPException(
                    status_code=499,
                    detail="Text performance cancelled because the client disconnected.",
                )
            pronunciation = apply_pronunciations(section.text, pronunciations)
            for term in pronunciation.matched_terms:
                if term not in matched_pronunciations:
                    matched_pronunciations.append(term)
            rendered, section_sample_rate = await asyncio.wrap_future(
                model_executor.submit(
                    synthesize,
                    pronunciation.text,
                    section.prompt_path,
                    section.prompt_text,
                    section.reference_path,
                    profile,
                    section.instruction,
                    lora_path,
                    section.seed,
                )
            )
            if output_sample_rate and output_sample_rate != section_sample_rate:
                raise HTTPException(
                    status_code=500,
                    detail="VoxCPM2 returned inconsistent section sample rates.",
                )
            output_sample_rate = section_sample_rate
            finished = np.clip(
                finished_text_audio(rendered, output_sample_rate),
                -1.0,
                1.0,
            )
            output_pcm.extend((finished * 32767.0).astype("<i2").tobytes())
            if index + 1 < len(synthesis_sections):
                output_pcm.extend(
                    b"\x00\x00"
                    * int(output_sample_rate * section.pause_after)
                )

    if not output_pcm or output_sample_rate <= 0:
        raise HTTPException(status_code=500, detail="VoxCPM2 returned no text audio.")

    latency_ms = int((time.perf_counter() - started) * 1000.0)
    headers = {
        "X-Vox-Sample-Rate": str(output_sample_rate),
        "X-Vox-Latency-Ms": str(latency_ms),
        "X-Vox-Character": urllib.parse.quote(str(profile.get("name", voice_id))),
        "X-Vox-Transcript": urllib.parse.quote(spoken_text[:400]),
        "X-Vox-Delivery": delivery_tag,
        "X-Vox-Pronunciations": urllib.parse.quote(
            ", ".join(matched_pronunciations)
        ),
        "X-Vox-Adapter": "trained" if lora_path else "base",
        "X-Vox-Section-Count": str(len(synthesis_sections)),
        "X-Vox-Performance-Mode": performance_mode,
        "X-Vox-Style-Mode": (
            "performance matched"
            if any(section.prompt_text for section in synthesis_sections)
            else "controlled reference"
        ),
        "Cache-Control": "no-store",
    }
    return Response(bytes(output_pcm), media_type="audio/L16", headers=headers)


@app.post("/analyze_story")
def analyze_story(
    text: str = Form(...),
    delivery: str = Form("natural"),
) -> JSONResponse:
    spoken_text = re.sub(r"\s+", " ", text).strip()
    if not spoken_text:
        raise HTTPException(status_code=400, detail="Enter a story to analyze.")
    if len(spoken_text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Storytelling captures are limited to 10,000 characters.",
        )
    plan = plan_story_performance(spoken_text, normalized_delivery_tag(delivery))
    return JSONResponse(asdict(plan))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18990)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    arguments = parse_args()
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")
