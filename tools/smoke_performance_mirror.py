from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf


SAMPLE_RATE = 48000
BLOCK_SAMPLES = int(SAMPLE_RATE * 0.36)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a WAV file through Vox Studio Performance Mirror"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18910")
    return parser.parse_args()


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return (
        np.clip(samples, -1.0, 1.0)
        .__mul__(32767.0)
        .round()
        .astype("<i2")
        .tobytes()
    )


def convert_block(
    endpoint: str,
    voice_id: str,
    block: bytes,
) -> tuple[bytes, float]:
    started = time.perf_counter()
    response = requests.post(
        f"{endpoint}/convert_chunk",
        files={"audio": ("mirror.pcm", block, "application/octet-stream")},
        data={
            "model_id": voice_id,
            "sample_rate": str(SAMPLE_RATE),
            "channels": "1",
            "pitch_shift": "0",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.content, (time.perf_counter() - started) * 1000.0


def main() -> int:
    args = parse_args()
    source, source_rate = sf.read(args.source, dtype="float32", always_2d=False)
    if source.ndim > 1:
        source = source.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        source = librosa.resample(
            source,
            orig_sr=source_rate,
            target_sr=SAMPLE_RATE,
        )

    output_blocks: list[bytes] = []
    latencies: list[float] = []
    for offset in range(0, len(source), BLOCK_SAMPLES):
        source_block = source[offset: offset + BLOCK_SAMPLES]
        original_samples = len(source_block)
        if original_samples < BLOCK_SAMPLES:
            source_block = np.pad(
                source_block,
                (0, BLOCK_SAMPLES - original_samples),
            )
        converted, latency = convert_block(
            args.endpoint,
            args.voice_id,
            pcm16_bytes(source_block),
        )
        output_blocks.append(converted[: original_samples * 2])
        latencies.append(latency)

    silence, silence_latency = convert_block(
        args.endpoint,
        args.voice_id,
        bytes(BLOCK_SAMPLES * 2),
    )
    silence_peak = int(np.abs(np.frombuffer(silence, dtype="<i2")).max(initial=0))

    output = np.frombuffer(b"".join(output_blocks), dtype="<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, SAMPLE_RATE, subtype="PCM_16")

    print(f"output={args.output}")
    print(f"blocks={len(latencies)}")
    print(f"latency_median_ms={statistics.median(latencies):.1f}")
    print(f"latency_p95_ms={np.percentile(latencies, 95):.1f}")
    print(f"latency_max_ms={max(latencies):.1f}")
    print(f"silence_latency_ms={silence_latency:.1f}")
    print(f"silence_peak_pcm16={silence_peak}")
    return 0 if silence_peak == 0 and output.size > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
