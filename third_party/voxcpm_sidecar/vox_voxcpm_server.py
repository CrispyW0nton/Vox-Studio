from __future__ import annotations

import argparse
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
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from faster_whisper import WhisperModel
from scipy.signal import medfilt
from voxcpm import VoxCPM
from vox_delivery import (
    DeliveryReading,
    apply_pronunciations,
    build_transcription_hotwords,
    canonicalize_transcription,
    delivery_distance,
    detect_delivery,
    load_pronunciations,
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

app = FastAPI(title="Vox Studio VoxCPM2", version="1.0")
model_lock = threading.Lock()
transcriber_lock = threading.Lock()
delivery_lock = threading.Lock()
model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VoxCPM2")
tts_model: VoxCPM | None = None
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
) -> DeliveryReading:
    with delivery_lock:
        reading = detect_delivery(features, transcript, tuple(delivery_history))
        delivery_history.append(dict(features))
    return reading


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
    base = str(profile.get("control_instruction", "")).strip()
    if not profile.get("use_controlled_cloning", False) or not base:
        return ""

    details = [
        "Treat the reference clip as voice identity only, not as an emotional "
        "performance.",
        delivery.instruction,
    ]
    if performance["pitch_slope_semitones"] > 4.0:
        details.append(
            "Follow the performer's gradual rise through the line without exceeding it."
        )
    elif performance["pitch_slope_semitones"] < -4.0:
        details.append("Follow the performer's settling cadence toward the ending.")
    if performance["pause_ratio"] >= 0.18:
        details.append("Retain the performer's meaningful pauses.")
    details.append("Never raise the emotional intensity above the performer.")
    return " ".join((base, *details))


def synthesize(
    spoken_text: str,
    prompt_path: Path,
    prompt_text: str,
    reference_path: Path,
    profile: dict,
    control_instruction: str,
) -> tuple[np.ndarray, int]:
    with model_lock:
        model = get_tts_model()
        torch.manual_seed(42)
        cfg_value = float(profile.get("cfg_value", 2.0))
        controlled_cfg_value = float(profile.get("controlled_cfg_value", 0.0))
        if control_instruction and controlled_cfg_value > 0.0:
            cfg_value = controlled_cfg_value
        options = {
            "text": (
                f"({control_instruction}){spoken_text}" if control_instruction else spoken_text
            ),
            "reference_wav_path": str(reference_path),
            "cfg_value": cfg_value,
            "inference_timesteps": int(profile.get("inference_timesteps", 10)),
            "normalize": False,
        }
        if not control_instruction:
            options["prompt_wav_path"] = str(prompt_path)
            options["prompt_text"] = prompt_text
        rendered = model.generate(**options)
        return rendered, int(model.tts_model.sample_rate)


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
        delivery = performance_delivery(performance, spoken_text)
        style_anchor = select_style_anchor(
            profile,
            profile_dir,
            performance,
            delivery,
            spoken_text,
        )
        if style_anchor is None:
            synthesis_prompt_path = prompt_path
            synthesis_prompt_text = spoken_text
            synthesis_reference_path = reference_path
            style_source = "performer"
        else:
            synthesis_prompt_path, synthesis_prompt_text, style_source = style_anchor
            synthesis_reference_path = synthesis_prompt_path

        control_instruction = build_control_instruction(profile, performance, delivery)
        if style_anchor is not None and delivery.label in {
            "emphatic",
            "urgent",
            "questioning",
            "sarcastic",
        }:
            control_instruction = ""
        pronunciation = apply_pronunciations(spoken_text, pronunciations)
        rendered, output_sample_rate = model_executor.submit(
            synthesize,
            pronunciation.text,
            synthesis_prompt_path,
            synthesis_prompt_text,
            synthesis_reference_path,
            profile,
            control_instruction,
        ).result()

    output = np.clip(rendered, -1.0, 1.0)
    output_pcm = (output * 32767.0).astype("<i2").tobytes()
    latency_ms = int((time.perf_counter() - started) * 1000.0)
    headers = {
        "X-Vox-Sample-Rate": str(output_sample_rate),
        "X-Vox-Latency-Ms": str(latency_ms),
        "X-Vox-Character": urllib.parse.quote(str(profile.get("name", voice_id))),
        "X-Vox-Transcript": urllib.parse.quote(spoken_text),
        "X-Vox-Style-Source": urllib.parse.quote(style_source),
        "X-Vox-Delivery": delivery.label,
        "X-Vox-Pronunciations": urllib.parse.quote(
            ", ".join(pronunciation.matched_terms)
        ),
        "X-Vox-Synthesis-Mode": "controlled" if control_instruction else "continuation",
        "Cache-Control": "no-store",
    }
    return Response(output_pcm, media_type="audio/L16", headers=headers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18990)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    arguments = parse_args()
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")
