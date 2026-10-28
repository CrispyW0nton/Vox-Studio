from __future__ import annotations

import argparse
import importlib.util
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
from types import SimpleNamespace
from typing import Any, Iterable


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1


def parse_args() -> argparse.Namespace:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    parser = argparse.ArgumentParser(description="Vox Studio Performance Mirror sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18910)
    parser.add_argument(
        "--seed-vc-root",
        type=Path,
        default=local_app_data / "VoxStudio/engines/seed-vc",
    )
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=local_app_data / "VoxStudio/voxcpm_profiles",
    )
    return parser.parse_args()


def retain_latest_blocks(blocks: Iterable[bytes], maximum: int) -> tuple[list[bytes], int]:
    retained = list(blocks)
    bounded_maximum = max(0, int(maximum))
    dropped = max(0, len(retained) - bounded_maximum)
    return retained[dropped:], dropped


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_voice_reference(profiles_root: Path, voice_id: str) -> Path:
    if VOICE_ID_PATTERN.fullmatch(voice_id) is None:
        raise ValueError("Voice id contains unsupported characters")

    resolved_profiles = profiles_root.resolve()
    profile_root = (resolved_profiles / voice_id).resolve()
    if not _is_within(profile_root, resolved_profiles):
        raise ValueError("Voice profile escaped the profile directory")
    if not profile_root.is_dir():
        raise FileNotFoundError(f"Character profile not found for {voice_id}")

    relative_reference = Path("reference.wav")
    profile_path = profile_root / "profile.json"
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        configured = str(profile.get("performance_mirror_reference", "")).strip()
        if configured:
            relative_reference = Path(configured)

    reference_path = (profile_root / relative_reference).resolve()
    if not _is_within(reference_path, profile_root):
        raise ValueError("Performance Mirror reference escaped the voice profile")
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Performance Mirror reference is missing for {voice_id}: {reference_path}"
        )
    return reference_path


def parse_multipart(content_type: str, body: bytes) -> dict[str, bytes]:
    message = (
        b"Content-Type: "
        + content_type.encode("ascii", errors="strict")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    parsed = BytesParser(policy=email_policy).parsebytes(message)
    if not parsed.is_multipart():
        raise ValueError("Request must use multipart/form-data")

    fields: dict[str, bytes] = {}
    for part in parsed.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if field_name:
            fields[field_name] = part.get_payload(decode=True) or b""
    return fields


def _decode_field(fields: dict[str, bytes], name: str, default: str = "") -> str:
    value = fields.get(name)
    return default if value is None else value.decode("utf-8", errors="strict").strip()


class SeedVcRealtimeEngine:
    """Stateful adapter around the official Seed-VC tiny real-time model."""

    def __init__(self, seed_vc_root: Path) -> None:
        self.seed_vc_root = seed_vc_root.resolve()
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.channels = DEFAULT_CHANNELS
        self.block_time = 0.36
        self.crossfade_time = 0.04
        self.extra_time_ce = 1.5
        self.extra_time_dit = 0.1
        self.extra_time_right = 0.02
        self.diffusion_steps = 6
        # Upstream documents CFG 0 as roughly 1.5x faster with only a subtle
        # real-time quality difference. Speaker identity still comes from the
        # reference condition while this keeps each block ahead of capture.
        self.inference_cfg_rate = 0.0
        self.max_prompt_length = 3.0
        self.silence_rms = 0.0015
        self.reference_path: Path | None = None
        self.reference_wav: Any = None

        self._load_upstream()
        self._initialize_buffers()

    def _load_upstream(self) -> None:
        if not (self.seed_vc_root / "real-time-gui.py").is_file():
            raise FileNotFoundError(
                f"Official Seed-VC runtime not found at {self.seed_vc_root}"
            )

        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.chdir(self.seed_vc_root)
        if str(self.seed_vc_root) not in sys.path:
            sys.path.insert(0, str(self.seed_vc_root))

        import librosa
        import numpy as np
        import torch
        import torch.nn.functional as functional
        import torchaudio.transforms as audio_transforms

        spec = importlib.util.spec_from_file_location(
            "vox_seed_vc_realtime",
            self.seed_vc_root / "real-time-gui.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to import the official Seed-VC real-time module")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)

        if not torch.cuda.is_available():
            raise RuntimeError("Performance Mirror requires an NVIDIA CUDA GPU")
        upstream.device = torch.device("cuda")
        model_args = SimpleNamespace(
            checkpoint_path=None,
            config_path=None,
            fp16=True,
        )

        self.librosa = librosa
        self.np = np
        self.torch = torch
        self.functional = functional
        self.audio_transforms = audio_transforms
        self.upstream = upstream
        self.model_set = upstream.load_models(model_args)
        upstream.print = lambda *args, **kwargs: None
        self.device = upstream.device
        self.model_sample_rate = int(self.model_set[-1]["sampling_rate"])

    def _initialize_buffers(self) -> None:
        torch = self.torch
        np = self.np
        self.zc = self.sample_rate // 50
        self.block_frame = (
            int(np.round(self.block_time * self.sample_rate / self.zc)) * self.zc
        )
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.crossfade_time * self.sample_rate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.extra_time_ce * self.sample_rate / self.zc)) * self.zc
        )
        self.extra_frame_right = (
            int(np.round(self.extra_time_right * self.sample_rate / self.zc)) * self.zc
        )
        total_samples = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right
        )
        self.input_wav = torch.zeros(
            total_samples,
            device=self.device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            320 * total_samples // self.zc,
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
        self.skip_tail = self.extra_frame_right // self.zc
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
        self.input_resampler = self.audio_transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.device)
        self.output_resampler = self.audio_transforms.Resample(
            orig_freq=self.model_sample_rate,
            new_freq=self.sample_rate,
            dtype=torch.float32,
        ).to(self.device)

    def set_reference(self, reference_path: Path) -> None:
        resolved = reference_path.resolve()
        reference, _ = self.librosa.load(
            resolved,
            sr=self.model_sample_rate,
            mono=True,
        )
        reference, _ = self.librosa.effects.trim(reference, top_db=38)
        if reference.size < int(self.model_sample_rate * 0.8):
            raise ValueError("Character reference must contain at least 0.8 seconds of speech")

        peak = float(self.np.max(self.np.abs(reference)))
        if peak > 0.98:
            reference = reference * (0.98 / peak)
        self.reference_path = resolved
        self.reference_wav = reference.astype(self.np.float32, copy=False)
        self.reset_stream()
        self._prewarm_reference()

    def reset_stream(self) -> None:
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.sola_buffer.zero_()

    def _prewarm_reference(self) -> None:
        phase = self.np.arange(self.block_frame, dtype=self.np.float32)
        probe = 0.025 * self.np.sin(
            2.0 * self.np.pi * 180.0 * phase / float(self.sample_rate)
        )
        probe_pcm = (
            (self.np.clip(probe, -1.0, 1.0) * 32767.0)
            .round()
            .astype("<i2")
            .tobytes()
        )
        self.convert(probe_pcm, 0)
        self.torch.cuda.synchronize(self.device)
        self.reset_stream()

    def _append_input(self, samples: Any) -> None:
        torch = self.torch
        self.input_wav[:-self.block_frame] = self.input_wav[
            self.block_frame:
        ].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(samples).to(self.device)
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k:
        ].clone()
        resample_input = self.input_wav[-self.block_frame - 2 * self.zc:]
        self.input_wav_res[-self.block_frame_16k - 320:] = self.input_resampler(
            resample_input
        )[320:]

    def _match_source_energy(self, inferred: Any, source: Any) -> Any:
        torch = self.torch
        source_tensor = torch.from_numpy(source).to(self.device)
        output_tensor = inferred[: self.block_frame]
        envelope_window = max(self.zc * 2, 1)
        envelope_stride = max(self.zc // 2, 1)

        def envelope(audio: Any) -> Any:
            pooled = self.functional.avg_pool1d(
                audio.square()[None, None, :],
                kernel_size=envelope_window,
                stride=envelope_stride,
                padding=envelope_window // 2,
                count_include_pad=False,
            ).clamp_min(1.0e-8).sqrt()
            return self.functional.interpolate(
                pooled,
                size=self.block_frame,
                mode="linear",
                align_corners=True,
            )[0, 0]

        source_envelope = envelope(source_tensor)
        output_envelope = envelope(output_tensor)
        gain = (source_envelope / output_envelope).clamp(0.35, 2.5)
        gain = self.functional.avg_pool1d(
            gain[None, None, :],
            kernel_size=max(self.zc // 2, 1),
            stride=1,
            padding=max(self.zc // 4, 0),
        )
        gain = gain[0, 0, : self.block_frame]
        if gain.shape[0] < self.block_frame:
            gain = self.functional.pad(
                gain,
                (0, self.block_frame - int(gain.shape[0])),
                mode="replicate",
            )
        inferred[: self.block_frame] *= gain
        return inferred

    def convert(self, pcm16_audio: bytes, pitch_shift: int = 0) -> bytes:
        del pitch_shift
        if self.reference_path is None or self.reference_wav is None:
            raise RuntimeError("Select a character before converting audio")
        if len(pcm16_audio) % 2:
            raise ValueError("PCM audio must contain complete 16-bit samples")

        original_samples = len(pcm16_audio) // 2
        if original_samples <= 0:
            raise ValueError("Audio chunk is empty")
        if original_samples > self.block_frame:
            raise ValueError(
                f"Audio chunk contains {original_samples} samples; "
                f"the Performance Mirror block limit is {self.block_frame}"
            )

        samples = (
            self.np.frombuffer(pcm16_audio, dtype="<i2").astype(self.np.float32)
            / 32768.0
        )
        source_samples = samples.copy()
        if original_samples < self.block_frame:
            samples = self.np.pad(samples, (0, self.block_frame - original_samples))
        self._append_input(samples)

        source_rms = float(
            self.np.sqrt(self.np.mean(source_samples * source_samples) + 1.0e-9)
        )
        source_peak = float(self.np.max(self.np.abs(source_samples)))
        if source_rms < self.silence_rms and source_peak < 0.008:
            self.sola_buffer.zero_()
            return bytes(original_samples * 2)

        inferred = self.upstream.custom_infer(
            self.model_set,
            self.reference_wav,
            str(self.reference_path),
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.skip_tail,
            self.return_length,
            self.diffusion_steps,
            self.inference_cfg_rate,
            self.max_prompt_length,
            self.extra_time_ce - self.extra_time_dit,
        )
        inferred = self.output_resampler(inferred)

        search_input = inferred[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        correlation = self.functional.conv1d(
            search_input,
            self.sola_buffer[None, None, :],
        )
        denominator = self.torch.sqrt(
            self.functional.conv1d(search_input**2, self.sola_den_kernel) + 1.0e-8
        )
        sola_offset = int(
            self.torch.argmax(correlation[0, 0] / denominator[0, 0]).item()
        )
        inferred = inferred[sola_offset:]
        inferred[: self.sola_buffer_frame] *= self.fade_in_window
        inferred[: self.sola_buffer_frame] += (
            self.sola_buffer * self.fade_out_window
        )
        self.sola_buffer[:] = inferred[
            self.block_frame: self.block_frame + self.sola_buffer_frame
        ]
        inferred = self._match_source_energy(inferred, samples)

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


class PerformanceMirrorService:
    def __init__(self, seed_vc_root: Path, profiles_root: Path) -> None:
        self.seed_vc_root = seed_vc_root.resolve()
        self.profiles_root = profiles_root.resolve()
        self.engine: SeedVcRealtimeEngine | None = None
        self.loaded_voice_id = ""
        self.last_latency_ms = -1
        self.runtime_error = ""
        self.lock = threading.Lock()
        self.runtime_ready = threading.Event()
        if self.runtime_installed:
            threading.Thread(
                target=self._prepare_runtime,
                name="performance-mirror-warmup",
                daemon=True,
            ).start()

    @property
    def runtime_installed(self) -> bool:
        return (self.seed_vc_root / "real-time-gui.py").is_file()

    def _prepare_runtime(self) -> None:
        try:
            engine = SeedVcRealtimeEngine(self.seed_vc_root)
            with self.lock:
                self.engine = engine
        except Exception as exception:
            self.runtime_error = str(exception)
            traceback.print_exc()
        finally:
            self.runtime_ready.set()

    def health(self) -> dict[str, Any]:
        installed = self.runtime_installed
        loading = installed and not self.runtime_ready.is_set()
        ready = self.engine is not None
        if not installed:
            message = "Install the Performance Mirror engine to enable live character conversion."
        elif loading:
            message = "Performance Mirror is warming up on the GPU."
        elif self.runtime_error:
            message = self.runtime_error
        elif self.loaded_voice_id:
            message = "Performance Mirror is ready."
        else:
            message = "Performance Mirror is ready for a character."
        return {
            "ok": True,
            "engine": "performance-mirror",
            "runtime_installed": installed,
            "runtime_ready": ready,
            "cuda_available": ready,
            "cuda_version": "loading" if loading else "",
            "loaded_model_id": self.loaded_voice_id,
            "last_latency_ms": self.last_latency_ms,
            "message": message,
        }

    def load_voice(self, voice_id: str) -> dict[str, Any]:
        reference = resolve_voice_reference(self.profiles_root, voice_id)
        if not self.runtime_installed:
            raise FileNotFoundError(
                "Performance Mirror is not installed. Run its one-time setup first."
            )
        if not self.runtime_ready.is_set():
            raise RuntimeError("Performance Mirror is still warming up")
        if self.engine is None:
            raise RuntimeError(self.runtime_error or "Performance Mirror failed to load")

        with self.lock:
            if voice_id != self.loaded_voice_id:
                self.engine.set_reference(reference)
                self.loaded_voice_id = voice_id
        return self.health()

    def convert(
        self,
        voice_id: str,
        audio: bytes,
        sample_rate: int,
        channels: int,
        pitch_shift: int,
    ) -> bytes:
        if sample_rate != DEFAULT_SAMPLE_RATE or channels != DEFAULT_CHANNELS:
            raise ValueError("Performance Mirror requires 48 kHz mono PCM input")
        if voice_id != self.loaded_voice_id:
            self.load_voice(voice_id)
        if self.engine is None:
            raise RuntimeError("Performance Mirror is not ready")

        started = time.perf_counter()
        with self.lock:
            converted = self.engine.convert(audio, pitch_shift)
        self.last_latency_ms = int((time.perf_counter() - started) * 1000.0)
        return converted


class PerformanceMirrorHandler(BaseHTTPRequestHandler):
    server_version = "VoxStudioPerformanceMirror/1.0"

    @property
    def service(self) -> PerformanceMirrorService:
        return self.server.service  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, exception: Exception) -> None:
        self._send_json(status, {"ok": False, "message": str(exception)})

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"ok": False, "message": "Not found"})
            return
        self._send_json(200, self.service.health())

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            if self.path == "/load_model":
                request = json.loads(body.decode("utf-8"))
                voice_id = str(request.get("model_id", "")).strip()
                self._send_json(200, self.service.load_voice(voice_id))
                return
            if self.path == "/convert_chunk":
                fields = parse_multipart(self.headers.get("Content-Type", ""), body)
                voice_id = _decode_field(fields, "model_id")
                sample_rate = int(
                    _decode_field(fields, "sample_rate", str(DEFAULT_SAMPLE_RATE))
                )
                channels = int(
                    _decode_field(fields, "channels", str(DEFAULT_CHANNELS))
                )
                pitch_shift = int(_decode_field(fields, "pitch_shift", "0"))
                converted = self.service.convert(
                    voice_id,
                    fields.get("audio", b""),
                    sample_rate,
                    channels,
                    pitch_shift,
                )
                self.send_response(200)
                self.send_header("Content-Type", "audio/L16")
                self.send_header("Content-Length", str(len(converted)))
                self.end_headers()
                self.wfile.write(converted)
                return
            self._send_json(404, {"ok": False, "message": "Not found"})
        except (FileNotFoundError, RuntimeError) as exception:
            self._send_error_json(503, exception)
        except (ValueError, KeyError, json.JSONDecodeError) as exception:
            self._send_error_json(400, exception)
        except Exception as exception:
            traceback.print_exc()
            self._send_error_json(500, exception)

    def log_message(self, format_text: str, *args: Any) -> None:
        print(
            f"[PerformanceMirror] {self.address_string()} "
            f"{format_text % args}",
            flush=True,
        )


class PerformanceMirrorHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: PerformanceMirrorService,
    ) -> None:
        super().__init__(address, PerformanceMirrorHandler)
        self.service = service


def main() -> int:
    args = parse_args()
    service = PerformanceMirrorService(args.seed_vc_root, args.profiles_root)
    server = PerformanceMirrorHttpServer((args.host, args.port), service)
    print(
        f"Vox Studio Performance Mirror listening on "
        f"http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
