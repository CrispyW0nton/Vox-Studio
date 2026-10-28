#!/usr/bin/env python3
"""Build a clean RVC dataset from KotOR's wrapped dialogue audio.

KotOR stores some MP3 streams behind a short, misleading RIFF header. Passing
those files directly to a normal WAV decoder produces noise instead of speech.
The wrapper detection here follows the open-source implementations in PyKotor
and KotOR.js:

https://github.com/OpenKotOR/PyKotor
https://github.com/KotORPublicDomain/KotOR.js
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import shutil
import struct
import subprocess
import sys
import wave

from dataclasses import asdict, dataclass
from pathlib import Path


SFX_MAGIC = b"\xff\xf3\x60\xc4"
SFX_HEADER_SIZE = 470
MP3_IN_WAV_RIFF_SIZE = 50
MP3_IN_WAV_HEADER_SIZE = 58
VO_HEADER_SIZE = 20


@dataclass(frozen=True)
class AudioPayload:
    data: bytes
    input_format: str | None
    wrapper: str


@dataclass(frozen=True)
class AudioStats:
    duration_seconds: float
    peak: float
    rms: float
    zero_crossing_rate: float


def unwrap_kotor_audio(data: bytes) -> AudioPayload:
    """Return the playable payload inside a KotOR audio container."""
    if data.startswith(SFX_MAGIC):
        if len(data) <= SFX_HEADER_SIZE:
            raise ValueError("truncated KotOR SFX wrapper")
        payload = data[SFX_HEADER_SIZE:]
        input_format = "wav" if payload.startswith(b"RIFF") else None
        return AudioPayload(payload, input_format, "sfx-470")

    if data.startswith(b"RIFF"):
        if len(data) >= VO_HEADER_SIZE + 4 and data[VO_HEADER_SIZE : VO_HEADER_SIZE + 4] == b"RIFF":
            return AudioPayload(data[VO_HEADER_SIZE:], "wav", "voice-20")

        if len(data) >= 8 and struct.unpack("<I", data[4:8])[0] == MP3_IN_WAV_RIFF_SIZE:
            if len(data) <= MP3_IN_WAV_HEADER_SIZE:
                raise ValueError("truncated KotOR MP3-in-WAV wrapper")
            return AudioPayload(data[MP3_IN_WAV_HEADER_SIZE:], "mp3", "mp3-in-wav-58")

        return AudioPayload(data, "wav", "standard-wav")

    return AudioPayload(data, None, "unwrapped")


def decode_audio(
    ffmpeg: str,
    source: Path,
    output: Path,
    *,
    sample_rate: int,
) -> str:
    payload = unwrap_kotor_audio(source.read_bytes())
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if payload.input_format is not None:
        command.extend(["-f", payload.input_format])
    command.extend(
        [
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        input=payload.data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"FFmpeg exited with {result.returncode}")
    return payload.wrapper


def inspect_pcm_wav(path: Path) -> AudioStats:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit PCM output")
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    if not samples:
        raise ValueError("decoded audio is empty")

    scale = 32768.0
    peak = max(abs(value) for value in samples) / scale
    rms = math.sqrt(sum(value * value for value in samples) / len(samples)) / scale
    crossings = sum(
        1
        for previous, current in zip(samples, samples[1:])
        if (previous < 0 <= current) or (previous >= 0 > current)
    )
    zero_crossing_rate = crossings / max(1, len(samples) - 1)
    return AudioStats(
        duration_seconds=len(samples) / sample_rate,
        peak=peak,
        rms=rms,
        zero_crossing_rate=zero_crossing_rate,
    )


def source_index(source_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in {".wav", ".mp3"}:
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def selection_names(selection_dir: Path, excluded_globs: list[str]) -> tuple[list[str], list[str]]:
    names = {
        path.name.casefold()
        for path in selection_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in {".wav", ".mp3"}
    }
    if not names:
        raise ValueError(f"{selection_dir} does not contain any audio files")
    excluded = sorted(
        name
        for name in names
        if any(fnmatch.fnmatch(name, pattern.casefold()) for pattern in excluded_globs)
    )
    return sorted(names - set(excluded)), excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode selected KotOR dialogue into a clean RVC training dataset"
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--selection-dir",
        required=True,
        type=Path,
        help="Directory whose filenames identify the desired dialogue lines",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Report path (defaults beside the output directory, not inside it)",
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--exclude-name-glob",
        action="append",
        default=[],
        help="Case-insensitive filename glob to omit; may be supplied more than once",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    if not args.source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {args.source_root}")
    if not args.selection_dir.is_dir():
        raise SystemExit(f"Selection directory does not exist: {args.selection_dir}")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive")
    if args.min_seconds < 0:
        raise SystemExit("--min-seconds cannot be negative")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise SystemExit(f"FFmpeg was not found: {args.ffmpeg}")
    return ffmpeg


def main() -> int:
    args = parse_args()
    ffmpeg = validate_args(args)
    sources = source_index(args.source_root)
    selected, excluded = selection_names(args.selection_dir, args.exclude_name_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    failures: list[str] = []
    for index, name in enumerate(selected, start=1):
        matches = sources.get(name, [])
        if len(matches) != 1:
            failures.append(f"{name}: expected one source, found {len(matches)}")
            continue

        source = matches[0]
        output = args.output_dir / f"{Path(name).stem}.wav"
        if output.exists() and not args.overwrite:
            failures.append(f"{name}: output already exists (use --overwrite)")
            continue

        try:
            wrapper = decode_audio(ffmpeg, source, output, sample_rate=args.sample_rate)
            stats = inspect_pcm_wav(output)
            if stats.duration_seconds < args.min_seconds:
                raise ValueError(
                    f"only {stats.duration_seconds:.3f}s, below {args.min_seconds:.3f}s minimum"
                )
            if stats.rms > 0.35 and stats.zero_crossing_rate > 0.25:
                raise ValueError(
                    "decoded output looks like full-scale broadband noise "
                    f"(rms={stats.rms:.3f}, zcr={stats.zero_crossing_rate:.3f})"
                )
            manifest.append(
                {
                    "name": name,
                    "source": str(source),
                    "output": str(output),
                    "wrapper": wrapper,
                    **asdict(stats),
                }
            )
            print(f"[{index}/{len(selected)}] {name}: {stats.duration_seconds:.2f}s ({wrapper})")
        except (OSError, RuntimeError, ValueError, wave.Error) as exc:
            output.unlink(missing_ok=True)
            failures.append(f"{name}: {exc}")

    report = {
        "source_root": str(args.source_root.resolve()),
        "selection_dir": str(args.selection_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "sample_rate": args.sample_rate,
        "selected": len(selected),
        "excluded": excluded,
        "decoded": len(manifest),
        "failed": len(failures),
        "duration_seconds": sum(float(item["duration_seconds"]) for item in manifest),
        "files": manifest,
        "failures": failures,
    }
    report_path = args.manifest or (
        args.output_dir.parent / f"{args.output_dir.name}.dataset_manifest.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"Decoded {len(manifest)}/{len(selected)} files "
        f"({report['duration_seconds'] / 60.0:.1f} minutes)."
    )
    print(f"Manifest: {report_path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
