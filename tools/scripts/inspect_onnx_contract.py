#!/usr/bin/env python3
"""Inspect ONNX graph I/O and emit a Vox Studio native RVC graph contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TYPE_MAP = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int8)": "int8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(uint8)": "uint8",
    "tensor(uint16)": "uint16",
    "tensor(uint32)": "uint32",
    "tensor(uint64)": "uint64",
    "tensor(bool)": "bool",
    "tensor(string)": "string",
}


def existing_onnx(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a file")
    if path.suffix.lower() != ".onnx":
        raise argparse.ArgumentTypeError(f"{path} is not an .onnx file")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ONNX I/O for Vox Studio RVC")
    parser.add_argument("--generator", type=existing_onnx, help="RVC generator ONNX model")
    parser.add_argument("--hubert", type=existing_onnx, help="HuBERT encoder ONNX model")
    parser.add_argument("--f0", type=existing_onnx, help="RMVPE/Crepe F0 ONNX model")
    parser.add_argument("--out", type=Path, help="Output graph contract JSON path")
    parser.add_argument("--required", action="store_true", help="Mark the contract strict")
    parser.add_argument(
        "--template",
        action="store_true",
        help="Write an empty contract template without importing onnxruntime",
    )
    return parser.parse_args()


def tensor_type(value: str) -> str:
    return TYPE_MAP.get(value, value)


def tensor_shape(value: Any) -> tuple[list[int], list[str]]:
    dimensions: list[int] = []
    symbolic_dimensions: list[str] = []
    if value is None:
        return dimensions, symbolic_dimensions
    for dimension in value:
        if isinstance(dimension, int):
            dimensions.append(dimension)
            symbolic_dimensions.append("")
        elif dimension is None:
            dimensions.append(-1)
            symbolic_dimensions.append("")
        else:
            dimensions.append(-1)
            symbolic_dimensions.append(str(dimension))
    return dimensions, symbolic_dimensions


def tensor_to_contract(tensor: Any) -> dict[str, Any]:
    dimensions, symbolic_dimensions = tensor_shape(tensor.shape)
    result: dict[str, Any] = {
        "name": tensor.name,
        "element_type": tensor_type(tensor.type),
        "dimensions": dimensions,
        "role": "",
    }
    if any(symbolic_dimensions):
        result["symbolic_dimensions"] = symbolic_dimensions
    return result


def empty_session_contract() -> dict[str, list[dict[str, Any]]]:
    return {"inputs": [], "outputs": []}


def inspect_model(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Install onnxruntime in the active Python environment.") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "inputs": [tensor_to_contract(tensor) for tensor in session.get_inputs()],
        "outputs": [tensor_to_contract(tensor) for tensor in session.get_outputs()],
    }


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.template:
        if args.required:
            raise SystemExit("--template cannot be combined with --required")
        return {
            "required": args.required,
            "generator": empty_session_contract(),
            "hubert": empty_session_contract(),
            "f0": empty_session_contract(),
        }

    missing = [
        name
        for name, value in (
            ("--generator", args.generator),
            ("--hubert", args.hubert),
            ("--f0", args.f0),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("Missing required model paths: " + ", ".join(missing))

    return {
        "required": args.required,
        "generator": inspect_model(args.generator),
        "hubert": inspect_model(args.hubert),
        "f0": inspect_model(args.f0),
    }


def write_contract(contract: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(contract, indent=2)
    if out is None:
        print(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def main() -> int:
    args = parse_args()
    write_contract(build_contract(args), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
