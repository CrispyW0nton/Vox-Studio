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
from pathlib import Path

import numpy as np
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


def load_profile(voice_id: str) -> tuple[dict, Path]:
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
    return profile, reference_path


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
        with model_lock:
            get_tts_model()
        get_transcriber()
    except Exception as exception:
        print(f"VoxCPM2 warmup failed: {exception}", flush=True)


@app.on_event("startup")
def start_warmup() -> None:
    threading.Thread(target=warm_models, name="VoxCPM2Warmup", daemon=True).start()


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

    profile, reference_path = load_profile(voice_id)
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

        with model_lock:
            model = get_tts_model()
            torch.manual_seed(42)
            rendered = model.generate(
                text=spoken_text,
                prompt_wav_path=str(prompt_path),
                prompt_text=spoken_text,
                reference_wav_path=str(reference_path),
                cfg_value=float(profile.get("cfg_value", 2.0)),
                inference_timesteps=int(profile.get("inference_timesteps", 10)),
                normalize=False,
            )
            output_sample_rate = int(model.tts_model.sample_rate)

    output = np.clip(rendered, -1.0, 1.0)
    output_pcm = (output * 32767.0).astype("<i2").tobytes()
    latency_ms = int((time.perf_counter() - started) * 1000.0)
    headers = {
        "X-Vox-Sample-Rate": str(output_sample_rate),
        "X-Vox-Latency-Ms": str(latency_ms),
        "X-Vox-Character": urllib.parse.quote(str(profile.get("name", voice_id))),
        "X-Vox-Transcript": urllib.parse.quote(spoken_text),
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
