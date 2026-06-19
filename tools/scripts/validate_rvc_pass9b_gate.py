#!/usr/bin/env python3
"""Validate Pass 9b native RVC acceptance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path


def existing_file(value: str) -> Path:
    path = existing_path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a file")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Vox Studio Pass 9b RVC evidence")
    parser.add_argument("--ab-report", required=True, type=existing_file)
    parser.add_argument("--native-report", required=True, type=existing_file)
    parser.add_argument("--sidecar-report", type=existing_file)
    parser.add_argument("--sidecar-latency-ms", type=float)
    parser.add_argument("--sidecar-footprint", required=True, action="append", type=existing_path)
    parser.add_argument("--native-footprint", required=True, action="append", type=existing_path)
    parser.add_argument("--max-spectral-db", type=float, default=2.0)
    parser.add_argument("--min-size-reduction-mb", type=float, default=300.0)
    parser.add_argument("--out", type=Path, help="Optional validation JSON report path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def numeric_value(report: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    current: Any = report
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def sidecar_latency_ms(args: argparse.Namespace) -> float:
    if args.sidecar_latency_ms is not None:
        return float(args.sidecar_latency_ms)
    if args.sidecar_report is None:
        raise SystemExit("Provide --sidecar-report or --sidecar-latency-ms")

    report = load_json(args.sidecar_report)
    for keys in (("latency_ms",), ("last_latency_ms",), ("status", "last_latency_ms")):
        latency = numeric_value(report, keys)
        if latency is not None:
            return latency
    raise SystemExit(f"{args.sidecar_report} does not contain sidecar latency")


def native_latency_ms(native_report: dict[str, Any]) -> float:
    latency = numeric_value(native_report, ("latency_ms",))
    if latency is None:
        raise SystemExit("Native report does not contain latency_ms")
    return latency


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def total_size(paths: list[Path]) -> int:
    return sum(size_bytes(path) for path in paths)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    ab_report = load_json(args.ab_report)
    native_report = load_json(args.native_report)

    spectral_db = numeric_value(ab_report, ("mean_abs_spectral_db",))
    if spectral_db is None:
        raise SystemExit("A/B report does not contain mean_abs_spectral_db")
    spectral_passed = bool(ab_report.get("passed", False)) and spectral_db <= args.max_spectral_db

    native_ms = native_latency_ms(native_report)
    sidecar_ms = sidecar_latency_ms(args)
    latency_passed = native_ms <= sidecar_ms

    sidecar_bytes = total_size(args.sidecar_footprint)
    native_bytes = total_size(args.native_footprint)
    reduction_bytes = sidecar_bytes - native_bytes
    min_reduction_bytes = int(args.min_size_reduction_mb * 1024.0 * 1024.0)
    footprint_passed = reduction_bytes >= min_reduction_bytes

    report = {
        "passed": spectral_passed and latency_passed and footprint_passed,
        "spectral": {
            "passed": spectral_passed,
            "mean_abs_spectral_db": spectral_db,
            "max_spectral_db": args.max_spectral_db,
            "ab_report_passed": bool(ab_report.get("passed", False)),
        },
        "latency": {
            "passed": latency_passed,
            "native_latency_ms": native_ms,
            "sidecar_latency_ms": sidecar_ms,
        },
        "footprint": {
            "passed": footprint_passed,
            "sidecar_bytes": sidecar_bytes,
            "native_bytes": native_bytes,
            "reduction_bytes": reduction_bytes,
            "min_reduction_bytes": min_reduction_bytes,
        },
    }
    return report


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
