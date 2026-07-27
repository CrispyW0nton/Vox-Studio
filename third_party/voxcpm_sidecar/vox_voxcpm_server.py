from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import tempfile
import threading
import time
import urllib.parse
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from faster_whisper import WhisperModel
from voxcpm import VoxCPM


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
model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VoxCPM2")
tts_model: VoxCPM | None = None
transcriber: WhisperModel | None = None
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
    global transcriber
    with transcriber_lock:
        if transcriber is None:
            cache_root = ENGINE_ROOT / "cache" / "faster-whisper"
            cache_root.mkdir(parents=True, exist_ok=True)
            try:
                transcriber = WhisperModel(
                    "base.en",
                    device="cuda",
                    compute_type="float16",
                    download_root=str(cache_root),
                )
            except Exception:
                transcriber = WhisperModel(
                    "base.en",
                    device="cpu",
                    compute_type="int8",
                    download_root=str(cache_root),
                )
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
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return re.sub(r"\s+", " ", text)


def normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def style_features(path: Path, transcript: str) -> dict[str, float]:
    clip, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = clip.mean(axis=1)
    duration = max(len(mono) / sample_rate, 0.1)
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=480)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)
    active_rms = rms_db[rms_db > -45.0]
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
    finite_f0 = f0[np.isfinite(f0)]
    pitch_variation = (
        float(np.std(finite_f0) / max(np.mean(finite_f0), 1.0))
        if finite_f0.size
        else 0.0
    )
    spoken_characters = len(re.sub(r"[^A-Za-z0-9]", "", transcript))
    return {
        "dynamic_db": dynamic_db,
        "pitch_variation": pitch_variation,
        "characters_per_second": spoken_characters / duration,
    }


def select_style_anchor(
    profile: dict,
    profile_dir: Path,
    performance_path: Path,
    transcript: str,
) -> tuple[Path, str, str] | None:
    anchors = profile.get("style_anchors", [])
    if not anchors:
        return None

    target_text = normalized_text(transcript)
    for anchor in anchors:
        anchor_text = normalized_text(str(anchor.get("prompt_text", "")))
        if target_text and anchor_text:
            similarity = difflib.SequenceMatcher(None, target_text, anchor_text).ratio()
            if similarity >= 0.92:
                path = (profile_dir / anchor["audio"]).resolve()
                if profile_dir.resolve() not in path.parents or not path.exists():
                    raise HTTPException(
                        status_code=500,
                        detail="Character style anchor is missing.",
                    )
                return path, str(anchor["prompt_text"]), str(anchor.get("source", path.name))

    performance = style_features(performance_path, transcript)

    def distance(anchor: dict) -> float:
        features = anchor.get("features", {})
        rate = max(float(features.get("characters_per_second", 1.0)), 0.1)
        performance_rate = max(performance["characters_per_second"], 0.1)
        return (
            abs(float(features.get("dynamic_db", 0.0)) - performance["dynamic_db"]) / 12.0
            + abs(
                float(features.get("pitch_variation", 0.0))
                - performance["pitch_variation"]
            )
            * 5.0
            + abs(math.log(rate / performance_rate))
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


def synthesize(
    spoken_text: str,
    prompt_path: Path,
    prompt_text: str,
    reference_path: Path,
    profile: dict,
) -> tuple[np.ndarray, int]:
    with model_lock:
        model = get_tts_model()
        torch.manual_seed(42)
        rendered = model.generate(
            text=spoken_text,
            prompt_wav_path=str(prompt_path),
            prompt_text=prompt_text,
            reference_wav_path=str(reference_path),
            cfg_value=float(profile.get("cfg_value", 2.0)),
            inference_timesteps=int(profile.get("inference_timesteps", 10)),
            normalize=False,
        )
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

        style_anchor = select_style_anchor(
            profile,
            profile_dir,
            prompt_path,
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

        rendered, output_sample_rate = model_executor.submit(
            synthesize,
            spoken_text,
            synthesis_prompt_path,
            synthesis_prompt_text,
            synthesis_reference_path,
            profile,
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
