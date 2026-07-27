from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    parser = argparse.ArgumentParser(description="Vox Studio real-time RVC sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument(
        "--rvc-root",
        type=Path,
        default=local_app_data / "VoxStudio/training/RVC-WebUI",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=local_app_data / "VoxStudio/rvc_models",
    )
    return parser.parse_args()


def load_runtime(rvc_root: Path) -> dict[str, Any]:
    resolved_root = rvc_root.resolve()
    if not (resolved_root / "infer/rtrvc.py").is_file():
        raise FileNotFoundError(f"Official RVC runtime not found at {resolved_root}")

    os.chdir(resolved_root)
    sys.path.insert(0, str(resolved_root))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchaudio.transforms as audio_transforms

    from configs.config import Config
    from infer.rtrvc import RVC

    return {
        "np": np,
        "torch": torch,
        "functional": functional,
        "audio_transforms": audio_transforms,
        "Config": Config,
        "RVC": RVC,
    }


class RealtimeRvc:
    def __init__(
        self,
        runtime: dict[str, Any],
        pth_path: Path,
        index_path: Path | None,
        pitch_shift: int,
        sample_rate: int = 48000,
    ) -> None:
        self.runtime = runtime
        self.np = runtime["np"]
        self.torch = runtime["torch"]
        self.functional = runtime["functional"]
        self.sample_rate = sample_rate
        self.pitch_shift = pitch_shift
        saved_arguments = sys.argv
        try:
            sys.argv = [saved_arguments[0]]
            self.config = runtime["Config"]()
        finally:
            sys.argv = saved_arguments
        self.device = self.config.device
        self.block_time = 0.25
        self.crossfade_time = 0.05
        self.extra_time = 2.5
        self.index_rate = 0.75 if index_path is not None else 0.0

        self.rvc = runtime["RVC"](
            pitch_shift,
            0.0,
            str(pth_path),
            "" if index_path is None else str(index_path),
            self.index_rate,
            self.config,
        )
        self._initialize_buffers()
        self._prewarm()

    def _initialize_buffers(self) -> None:
        torch = self.torch
        np = self.np
        transforms = self.runtime["audio_transforms"]

        self.zc = self.sample_rate // 100
        self.block_frame = (
            int(np.round(self.block_time * self.sample_rate / self.zc)) * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.crossfade_time * self.sample_rate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.extra_time * self.sample_rate / self.zc)) * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.device,
            dtype=torch.float32,
        )
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.sola_den_kernel = torch.ones(
            1,
            1,
            self.sola_buffer_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.device)
        self.output_resampler = (
            transforms.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.sample_rate,
                dtype=torch.float32,
            ).to(self.device)
            if self.rvc.tgt_sr != self.sample_rate
            else None
        )

    def _prewarm(self) -> None:
        phase = self.np.arange(self.block_frame, dtype=self.np.float32)
        probe = 0.02 * self.np.sin(
            2 * self.np.pi * 220.0 * phase / float(self.sample_rate)
        )
        probe_pcm = (
            (self.np.clip(probe, -1.0, 1.0) * 32767.0)
            .round()
            .astype("<i2")
            .tobytes()
        )
        self.convert(probe_pcm, self.pitch_shift)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)
        self._reset_state()

    def _reset_state(self) -> None:
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.sola_buffer.zero_()
        self.rvc.cache_pitch.zero_()
        self.rvc.cache_pitchf.zero_()

    def convert(self, pcm16_audio: bytes, pitch_shift: int) -> bytes:
        if len(pcm16_audio) % 2:
            raise ValueError("PCM audio must contain complete 16-bit samples")

        np = self.np
        torch = self.torch
        original_samples = len(pcm16_audio) // 2
        if original_samples == 0:
            raise ValueError("Audio chunk is empty")
        if original_samples > self.block_frame:
            raise ValueError(
                f"Audio chunk contains {original_samples} samples; "
                f"the real-time block limit is {self.block_frame}"
            )

        samples = np.frombuffer(pcm16_audio, dtype="<i2").astype(np.float32) / 32768.0
        if original_samples < self.block_frame:
            samples = np.pad(samples, (0, self.block_frame - original_samples))

        if pitch_shift != self.pitch_shift:
            self.pitch_shift = pitch_shift
            self.rvc.change_key(pitch_shift)

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(samples).to(self.device)
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k:
        ].clone()
        resample_input = self.input_wav[-self.block_frame - 2 * self.zc:]
        self.input_wav_res[-self.block_frame_16k - 160:] = self.resampler(
            resample_input
        )[160:]

        inferred = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            "rmvpe",
        )
        if self.output_resampler is not None:
            inferred = self.output_resampler(inferred)

        conv_input = inferred[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        correlation = self.functional.conv1d(
            conv_input,
            self.sola_buffer[None, None, :],
        )
        denominator = torch.sqrt(
            self.functional.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8
        )
        sola_offset = int(torch.argmax(correlation[0, 0] / denominator[0, 0]).item())
        inferred = inferred[sola_offset:]
        inferred[:self.sola_buffer_frame] *= self.fade_in_window
        inferred[:self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = inferred[
            self.block_frame:self.block_frame + self.sola_buffer_frame
        ]

        output = (
            inferred[:original_samples]
            .clamp(-1.0, 1.0)
            .mul(32767.0)
            .round()
            .short()
            .cpu()
            .numpy()
            .astype("<i2", copy=False)
        )
        return output.tobytes()


class RvcService:
    def __init__(self, model_root: Path) -> None:
        self.runtime: dict[str, Any] | None = None
        self.model_root = model_root.resolve()
        self.engine: RealtimeRvc | None = None
        self.loaded_model_id = ""
        self.last_latency_ms = -1
        self.lock = threading.Lock()
        self.runtime_ready = threading.Event()
        self.runtime_error = ""

    def start_runtime_load(self, rvc_root: Path) -> None:
        threading.Thread(
            target=self._load_runtime,
            args=(rvc_root,),
            name="vox-rvc-runtime-loader",
            daemon=True,
        ).start()

    def health(self) -> dict[str, Any]:
        if not self.runtime_ready.is_set():
            return {
                "ok": True,
                "engine": "rvc-realtime",
                "cuda_available": False,
                "cuda_version": "loading",
                "loaded_model_id": "",
                "last_latency_ms": -1,
                "message": "Real-time RVC is warming up.",
            }
        if self.runtime is None:
            return {
                "ok": False,
                "engine": "rvc-realtime",
                "cuda_available": False,
                "cuda_version": "n/a",
                "loaded_model_id": "",
                "last_latency_ms": -1,
                "message": self.runtime_error or "RVC runtime failed to load.",
            }

        torch = self.runtime["torch"]
        cuda_available = bool(torch.cuda.is_available())
        return {
            "ok": True,
            "engine": "rvc-realtime",
            "cuda_available": cuda_available,
            "cuda_version": str(torch.version.cuda or "n/a"),
            "loaded_model_id": self.loaded_model_id,
            "last_latency_ms": self.last_latency_ms,
            "message": (
                "Real-time RVC is ready."
                if self.loaded_model_id
                else "Real-time RVC is ready; select a model to begin."
            ),
        }

    def convert(
        self,
        model_id: str,
        pcm16_audio: bytes,
        sample_rate: int,
        channels: int,
        pitch_shift: int,
    ) -> bytes:
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("Invalid RVC model id")
        if sample_rate != 48000 or channels != 1:
            raise ValueError("Real-time RVC requires 48 kHz mono PCM audio")

        with self.lock:
            started_at = time.perf_counter()
            self._ensure_model(model_id, pitch_shift)
            converted = self.engine.convert(pcm16_audio, pitch_shift)
            self.last_latency_ms = int((time.perf_counter() - started_at) * 1000)
            return converted

    def load_model(self, model_id: str, pitch_shift: int) -> dict[str, Any]:
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("Invalid RVC model id")
        with self.lock:
            started_at = time.perf_counter()
            self._ensure_model(model_id, pitch_shift)
            self.last_latency_ms = int((time.perf_counter() - started_at) * 1000)
            return self.health()

    def _ensure_model(self, model_id: str, pitch_shift: int) -> None:
        if not self.runtime_ready.wait(timeout=20.0):
            raise TimeoutError("RVC runtime did not finish warming up")
        if self.runtime is None:
            raise RuntimeError(self.runtime_error or "RVC runtime failed to load")
        if self.engine is None or self.loaded_model_id != model_id:
            self.engine = self._load_engine(model_id, pitch_shift)
            self.loaded_model_id = model_id

    def _load_engine(self, model_id: str, pitch_shift: int) -> RealtimeRvc:
        model_dir = (self.model_root / model_id).resolve()
        if model_dir.parent != self.model_root:
            raise ValueError("RVC model path escaped the model root")
        manifest_path = model_dir / "model.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)

        pth_path = Path(manifest["pth_path"]).resolve()
        index_text = str(manifest.get("index_path", "")).strip()
        index_path = Path(index_text).resolve() if index_text else None
        if not pth_path.is_file():
            raise FileNotFoundError(f"RVC weights not found: {pth_path}")
        if index_path is not None and not index_path.is_file():
            raise FileNotFoundError(f"RVC index not found: {index_path}")
        return RealtimeRvc(
            self.runtime,
            pth_path,
            index_path,
            pitch_shift,
        )

    def _load_runtime(self, rvc_root: Path) -> None:
        try:
            self.runtime = load_runtime(rvc_root)
        except BaseException as exception:
            self.runtime_error = str(exception)
            traceback.print_exc()
        finally:
            self.runtime_ready.set()


def parse_multipart(content_type: str, body: bytes) -> dict[str, bytes]:
    message = BytesParser(policy=email_policy).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("ascii")
        + body
    )
    fields: dict[str, bytes] = {}
    if not message.is_multipart():
        return fields
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = part.get_payload(decode=True) or b""
    return fields


class RequestHandler(BaseHTTPRequestHandler):
    service: RvcService
    server_version = "VoxStudioRvc/1.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(200, self.service.health())
            return
        self._write_json(404, {"ok": False, "message": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/load_model":
            self._load_model()
            return
        if self.path != "/convert_chunk":
            self._write_json(404, {"ok": False, "message": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            fields = parse_multipart(
                self.headers.get("Content-Type", ""),
                self.rfile.read(content_length),
            )
            converted = self.service.convert(
                self._text_field(fields, "model_id"),
                fields.get("audio", b""),
                int(self._text_field(fields, "sample_rate")),
                int(self._text_field(fields, "channels")),
                int(self._text_field(fields, "pitch_shift")),
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16")
            self.send_header("Content-Length", str(len(converted)))
            self.send_header("X-RVC-Latency-Ms", str(self.service.last_latency_ms))
            self.end_headers()
            self.wfile.write(converted)
        except Exception as exception:
            traceback.print_exc()
            self._write_json(500, {"ok": False, "message": str(exception)})

    def _load_model(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length) or b"{}")
            health = self.service.load_model(
                str(request.get("model_id", "")),
                int(request.get("pitch_shift", 0)),
            )
            self._write_json(200, health)
        except Exception as exception:
            traceback.print_exc()
            self._write_json(500, {"ok": False, "message": str(exception)})

    def log_message(self, format_text: str, *args: Any) -> None:
        sys.stdout.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format_text % args)
        )
        sys.stdout.flush()

    @staticmethod
    def _text_field(fields: dict[str, bytes], name: str) -> str:
        if name not in fields:
            raise ValueError(f"Missing multipart field: {name}")
        return fields[name].decode("utf-8").strip()

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    args = parse_args()
    service = RvcService(args.model_root)
    RequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Vox Studio real-time RVC listening on http://{args.host}:{args.port}")
    print(f"RVC runtime: {args.rvc_root.resolve()}")
    print(f"Model root: {args.model_root.resolve()}")
    service.start_runtime_load(args.rvc_root)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
