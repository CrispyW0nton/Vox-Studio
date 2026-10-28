"""Per-character settings shared by every voice-lab recipe (gsv, seedvc, voxcpm, eval, delivery, indextts).

Select a character with the environment variable VOICE_LAB_CHARACTER=<slug> (default "baodur", so existing Bao-Dur
commands behave exactly as before). Each <slug>.json in this folder describes one character; paths are absolute.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = "baodur"
PATH_KEYS = ("lab", "source_audio", "exact_manifest")


def slug() -> str:
    return os.environ.get("VOICE_LAB_CHARACTER", "").strip() or DEFAULT


def load(name: str | None = None) -> dict:
    name = name or slug()
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid character slug: {name!r}")
    path = ROOT / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No voice-lab character config {path}; known: {sorted(p.stem for p in ROOT.glob('*.json'))}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["slug"] = name
    for key in PATH_KEYS:
        if data.get(key):
            data[key] = Path(data[key])
    return data
