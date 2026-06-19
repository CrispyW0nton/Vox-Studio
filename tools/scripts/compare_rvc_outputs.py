#!/usr/bin/env python3
"""Compare sidecar and native RVC WAV renders for Pass 9b validation."""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WavAudio:
    path: Path
    sample_rate: int
    channels: int
    samples: list[float]


def existing_wav(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a file")
    if path.suffix.lower() != ".wav":
        raise argparse.ArgumentTypeError(f"{path} is not a .wav file")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B compare sidecar and native RVC WAV outputs"
    )
    parser.add_argument("--sidecar", required=True, type=existing_wav)
    parser.add_argument("--native", required=True, type=existing_wav)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    parser.add_argument("--max-spectral-db", type=float, default=2.0)
    parser.add_argument("--max-lag-ms", type=float, default=100.0)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--silence-floor-db", type=float, default=-80.0)
    parser.add_argument(
        "--max-analysis-seconds",
        type=float,
        default=30.0,
        help="Cap spectral analysis length after alignment; use 0 for all audio",
    )
    return parser.parse_args()


def read_wav(path: Path) -> WavAudio:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        frame_count = reader.getnframes()
        compression = reader.getcomptype()
        raw = reader.readframes(frame_count)

    if compression != "NONE":
        raise SystemExit(f"{path} is compressed; expected PCM WAV")
    if channels <= 0:
        raise SystemExit(f"{path} has no channels")
    if sample_width != 2:
        raise SystemExit(f"{path} must be 16-bit PCM; got {sample_width * 8}-bit")

    value_count = len(raw) // sample_width
    values = struct.unpack(f"<{value_count}h", raw)
    samples: list[float] = []
    for offset in range(0, len(values), channels):
        frame = values[offset : offset + channels]
        samples.append(sum(frame) / (channels * 32768.0))

    return WavAudio(path=path, sample_rate=sample_rate, channels=channels, samples=samples)


def validate_fft_size(frame_size: int, hop_size: int) -> None:
    if frame_size <= 0 or frame_size & (frame_size - 1) != 0:
        raise SystemExit("--frame-size must be a positive power of two")
    if hop_size <= 0:
        raise SystemExit("--hop-size must be positive")
    if hop_size > frame_size:
        raise SystemExit("--hop-size must be less than or equal to --frame-size")


def normalized_correlation(
    reference: list[float],
    candidate: list[float],
    lag: int,
    min_overlap: int,
) -> float:
    reference_start = 0 if lag >= 0 else -lag
    candidate_start = lag if lag >= 0 else 0
    overlap = min(len(reference) - reference_start, len(candidate) - candidate_start)
    if overlap < min_overlap:
        return float("-inf")

    dot = 0.0
    reference_energy = 0.0
    candidate_energy = 0.0
    for index in range(overlap):
        reference_value = reference[reference_start + index]
        candidate_value = candidate[candidate_start + index]
        dot += reference_value * candidate_value
        reference_energy += reference_value * reference_value
        candidate_energy += candidate_value * candidate_value

    if reference_energy <= 1.0e-12 or candidate_energy <= 1.0e-12:
        return float("-inf")
    return dot / math.sqrt(reference_energy * candidate_energy)


def estimate_native_lag(
    sidecar: list[float],
    native: list[float],
    sample_rate: int,
    max_lag_ms: float,
) -> tuple[int, float]:
    if not sidecar or not native:
        return 0, 0.0

    stride = max(1, min(len(sidecar), len(native)) // 16000)
    reference = sidecar[::stride]
    candidate = native[::stride]
    max_lag = max(0, int(sample_rate * max_lag_ms / 1000.0 / stride))
    max_lag = min(max_lag, max(len(reference), len(candidate)) - 1)
    min_overlap = max(16, min(len(reference), len(candidate)) // 8)

    best_lag = 0
    best_score = float("-inf")
    for lag in range(-max_lag, max_lag + 1):
        score = normalized_correlation(reference, candidate, lag, min_overlap)
        if score > best_score:
            best_lag = lag
            best_score = score

    return best_lag * stride, best_score if math.isfinite(best_score) else 0.0


def aligned_samples(
    sidecar: list[float],
    native: list[float],
    native_lag_samples: int,
    sample_rate: int,
    max_analysis_seconds: float,
) -> tuple[list[float], list[float]]:
    sidecar_start = 0 if native_lag_samples >= 0 else -native_lag_samples
    native_start = native_lag_samples if native_lag_samples >= 0 else 0
    overlap = min(len(sidecar) - sidecar_start, len(native) - native_start)
    if overlap <= 0:
        raise SystemExit("The WAV files do not overlap after lag alignment")

    if max_analysis_seconds > 0.0:
        overlap = min(overlap, int(sample_rate * max_analysis_seconds))

    return (
        sidecar[sidecar_start : sidecar_start + overlap],
        native[native_start : native_start + overlap],
    )


def fft(values: list[complex]) -> list[complex]:
    size = len(values)
    result = values[:]
    swap_index = 0
    for index in range(1, size):
        bit = size >> 1
        while swap_index & bit:
            swap_index ^= bit
            bit >>= 1
        swap_index ^= bit
        if index < swap_index:
            result[index], result[swap_index] = result[swap_index], result[index]

    length = 2
    while length <= size:
        angle = -2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        for start in range(0, size, length):
            rotation = 1.0 + 0.0j
            half = length // 2
            for offset in range(half):
                even = result[start + offset]
                odd = result[start + offset + half] * rotation
                result[start + offset] = even + odd
                result[start + offset + half] = even - odd
                rotation *= step
        length *= 2

    return result


def padded_frame(samples: list[float], start: int, frame_size: int) -> list[float]:
    frame = samples[start : start + frame_size]
    if len(frame) < frame_size:
        frame = frame + [0.0] * (frame_size - len(frame))
    return frame


def frame_starts(sample_count: int, frame_size: int, hop_size: int) -> list[int]:
    if sample_count <= frame_size:
        return [0]
    return list(range(0, sample_count - frame_size + 1, hop_size))


def spectral_difference(
    sidecar: list[float],
    native: list[float],
    frame_size: int,
    hop_size: int,
    silence_floor_db: float,
) -> tuple[float, int, int]:
    window = [
        0.5 - (0.5 * math.cos(2.0 * math.pi * index / (frame_size - 1)))
        for index in range(frame_size)
    ]
    difference_sum = 0.0
    compared_bins = 0
    starts = frame_starts(min(len(sidecar), len(native)), frame_size, hop_size)

    for start in starts:
        sidecar_frame = padded_frame(sidecar, start, frame_size)
        native_frame = padded_frame(native, start, frame_size)
        sidecar_fft = fft(
            [complex(sidecar_frame[index] * window[index], 0.0) for index in range(frame_size)]
        )
        native_fft = fft(
            [complex(native_frame[index] * window[index], 0.0) for index in range(frame_size)]
        )

        for bin_index in range((frame_size // 2) + 1):
            sidecar_db = 20.0 * math.log10(max(abs(sidecar_fft[bin_index]), 1.0e-12))
            native_db = 20.0 * math.log10(max(abs(native_fft[bin_index]), 1.0e-12))
            if max(sidecar_db, native_db) < silence_floor_db:
                continue
            difference_sum += abs(sidecar_db - native_db)
            compared_bins += 1

    if compared_bins == 0:
        return 0.0, len(starts), 0
    return difference_sum / compared_bins, len(starts), compared_bins


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    validate_fft_size(args.frame_size, args.hop_size)
    sidecar = read_wav(args.sidecar)
    native = read_wav(args.native)
    if sidecar.sample_rate != native.sample_rate:
        raise SystemExit(
            "Sample rates differ: "
            f"{sidecar.sample_rate} Hz sidecar vs {native.sample_rate} Hz native"
        )

    lag_samples, correlation = estimate_native_lag(
        sidecar.samples, native.samples, sidecar.sample_rate, args.max_lag_ms
    )
    sidecar_aligned, native_aligned = aligned_samples(
        sidecar.samples,
        native.samples,
        lag_samples,
        sidecar.sample_rate,
        args.max_analysis_seconds,
    )
    mean_difference, frames, bins = spectral_difference(
        sidecar_aligned,
        native_aligned,
        args.frame_size,
        args.hop_size,
        args.silence_floor_db,
    )
    passed = mean_difference <= args.max_spectral_db

    return {
        "sidecar": str(sidecar.path),
        "native": str(native.path),
        "sample_rate": sidecar.sample_rate,
        "sidecar_channels": sidecar.channels,
        "native_channels": native.channels,
        "native_lag_samples": lag_samples,
        "native_lag_ms": (lag_samples * 1000.0) / sidecar.sample_rate,
        "alignment_correlation": correlation,
        "samples_compared": len(sidecar_aligned),
        "seconds_compared": len(sidecar_aligned) / sidecar.sample_rate,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "frames_compared": frames,
        "spectral_bins_compared": bins,
        "mean_abs_spectral_db": mean_difference,
        "max_spectral_db": args.max_spectral_db,
        "passed": passed,
    }


def write_report(report: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(report, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_report(report, args.out)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
