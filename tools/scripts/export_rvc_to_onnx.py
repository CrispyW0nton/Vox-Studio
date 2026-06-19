#!/usr/bin/env python3
"""Prepare a Vox Studio native ONNX RVC bundle manifest.

This script is the Pass 9b conversion entry point. It intentionally keeps generated
`.onnx` files out of git and writes only into a user-selected output directory.

The actual RVC export graph varies by source checkpoint and upstream fork. `--dry-run`
records the bundle layout and graph-contract template Vox Studio expects; pass a contract
from `inspect_onnx_contract.py --required` once a real ONNX export has been inspected.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a file")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an RVC model to Vox Studio ONNX layout")
    parser.add_argument("--model-id", required=True, help="Stable Vox Studio model id")
    parser.add_argument("--pth", required=True, type=existing_file, help="Source RVC .pth file")
    parser.add_argument("--index", type=existing_file, help="Optional RVC .index file")
    parser.add_argument("--generator-onnx", type=existing_file, help="Existing generator ONNX")
    parser.add_argument("--hubert-onnx", type=existing_file, help="Existing HuBERT ONNX model")
    parser.add_argument("--rmvpe-onnx", type=existing_file, help="Existing RMVPE ONNX model")
    parser.add_argument("--graph-contract", type=existing_file, help="Existing graph contract JSON")
    parser.add_argument(
        "--require-contract",
        action="store_true",
        help="Require the written graph_contract to pass native validation",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output bundle directory")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only native_manifest.json and do not invoke PyTorch export",
    )
    return parser.parse_args()


def validate_extensions(args: argparse.Namespace) -> None:
    if args.pth.suffix.lower() != ".pth":
        raise SystemExit("--pth must point to an RVC .pth file")
    if args.index is not None and args.index.suffix.lower() != ".index":
        raise SystemExit("--index must point to an RVC .index file")
    for label in ("generator_onnx", "hubert_onnx", "rmvpe_onnx"):
        value = getattr(args, label)
        if value is not None and value.suffix.lower() != ".onnx":
            raise SystemExit(f"--{label.replace('_', '-')} must point to an .onnx file")
    if args.graph_contract is not None and args.graph_contract.suffix.lower() != ".json":
        raise SystemExit("--graph-contract must point to a JSON file")


def default_graph_contract(required: bool) -> dict[str, Any]:
    return {
        "required": required,
        "generator": {"inputs": [], "outputs": []},
        "hubert": {"inputs": [], "outputs": []},
        "f0": {"inputs": [], "outputs": []},
    }


def load_graph_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.graph_contract is None:
        if args.require_contract:
            raise SystemExit("--require-contract needs --graph-contract")
        return default_graph_contract(False)

    contract = json.loads(args.graph_contract.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise SystemExit("--graph-contract must contain a JSON object")
    if args.require_contract:
        contract["required"] = True
    else:
        contract.setdefault("required", False)
    return contract


def write_manifest(args: argparse.Namespace) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": args.model_id,
        "generator_onnx": "generator.onnx",
        "hubert_onnx": "hubert.onnx",
        "f0_onnx": "rmvpe.onnx",
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "source_pth": str(args.pth),
        "source_index": "" if args.index is None else str(args.index),
        "graph_contract": load_graph_contract(args),
    }
    path = args.out / "native_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def copy_if_different(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def assemble_existing_onnx_bundle(args: argparse.Namespace) -> None:
    missing = [
        option
        for option, value in (
            ("--generator-onnx", args.generator_onnx),
            ("--hubert-onnx", args.hubert_onnx),
            ("--rmvpe-onnx", args.rmvpe_onnx),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(
            "Non-dry-run assembly requires existing ONNX artifacts: " + ", ".join(missing)
        )

    args.out.mkdir(parents=True, exist_ok=True)
    copy_if_different(args.generator_onnx, args.out / "generator.onnx")
    copy_if_different(args.hubert_onnx, args.out / "hubert.onnx")
    copy_if_different(args.rmvpe_onnx, args.out / "rmvpe.onnx")


def main() -> int:
    args = parse_args()
    validate_extensions(args)
    if args.dry_run:
        manifest_path = write_manifest(args)
        print(f"Wrote {manifest_path}")
        return 0

    assemble_existing_onnx_bundle(args)
    manifest_path = write_manifest(args)
    print(f"Assembled {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
