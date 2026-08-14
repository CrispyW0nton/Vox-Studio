from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TRAINING_SAMPLE_RATE = 16_000
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 15.0
MIN_WORDS = 3
VALIDATION_BUCKET = 0
SPLIT_BUCKETS = 10


@dataclass(frozen=True)
class VoiceSource:
    name: str
    audio_root: Path
    tlk_path: Path
    dlg_path: Path | None = None
    voiceover_prefix: str = ""
    excluded_globs: tuple[str, ...] = ()
    dialogue_manifest: Path | None = None


@dataclass(frozen=True)
class TrainingSample:
    source_audio: Path
    text: str
    duration_seconds: float


def default_sources() -> dict[str, VoiceSource]:
    steam = Path(r"C:\Program Files (x86)\Steam\steamapps\common")
    voices = Path.home() / "Documents" / "KotorMods" / "Voices" / "RvcDatasets"
    return {
        "carth": VoiceSource(
            name="Carth",
            audio_root=voices / "CarthDecoded",
            tlk_path=steam / "swkotor" / "dialog.tlk",
            excluded_globs=("nm01aacart*", "n_m1bncart*"),
        ),
        "bao-dur": VoiceSource(
            name="Bao-Dur",
            audio_root=voices / "BaoDurDecoded",
            tlk_path=steam / "Knights of the Old Republic II" / "dialog.tlk",
            dlg_path=(
                steam
                / "Knights of the Old Republic II"
                / "Override"
                / "baodur.dlg"
            ),
            voiceover_prefix="gblbaodur",
        ),
        "kreia": VoiceSource(
            name="Kreia",
            audio_root=voices / "KreiaDecoded",
            tlk_path=steam / "Knights of the Old Republic II" / "dialog.tlk",
            dialogue_manifest=voices / "KreiaDecoded" / "dialogue_manifest.json",
        ),
    }


def clean_dialogue_text(text: str) -> str:
    cleaned = re.sub(r"\{[^{}]*\}", " ", text)
    cleaned = re.sub(r"\[[^\[\]]*\]", " ", cleaned)
    cleaned = re.sub(r"::.*?::", " ", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\"")
    return cleaned


def is_usable_dialogue(text: str) -> bool:
    if not text or "::" in text:
        return False
    words = re.findall(r"[A-Za-z0-9']+", text)
    if len(words) < MIN_WORDS:
        return False
    return any(re.search(r"[A-Za-z]", word) for word in words)


def is_excluded_name(path: Path, globs: tuple[str, ...]) -> bool:
    name = path.name.casefold()
    return any(Path(name).match(pattern.casefold()) for pattern in globs)


def split_name(source_audio: Path) -> str:
    digest = hashlib.sha256(source_audio.stem.casefold().encode("utf-8")).digest()
    return (
        "validation"
        if int.from_bytes(digest[:4], "big") % SPLIT_BUCKETS == VALIDATION_BUCKET
        else "train"
    )


def read_dialogue_mapping(source: VoiceSource) -> dict[str, str]:
    if source.dialogue_manifest is not None:
        if not source.dialogue_manifest.is_file():
            raise RuntimeError(
                f"Character dialogue manifest is missing: {source.dialogue_manifest}"
            )
        manifest = json.loads(source.dialogue_manifest.read_text(encoding="utf-8"))
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            raise RuntimeError("Character dialogue manifest entries are invalid.")
        return {
            Path(str(filename)).stem.casefold(): clean_dialogue_text(
                str(value.get("text", ""))
            )
            for filename, value in entries.items()
            if isinstance(value, dict) and str(value.get("text", "")).strip()
        }

    try:
        from pykotor.resource.formats.tlk import read_tlk
    except ImportError as exception:
        roaming_site_packages = (
            Path(os.environ.get("APPDATA", ""))
            / "Python"
            / "Python314"
            / "site-packages"
        )
        if roaming_site_packages.exists():
            sys.path.append(str(roaming_site_packages))
        try:
            from pykotor.resource.formats.tlk import read_tlk
        except ImportError:
            raise RuntimeError(
                "PyKotor is required to read the installed KOTOR dialogue files."
            ) from exception

    with contextlib.redirect_stdout(io.StringIO()):
        tlk = read_tlk(source.tlk_path)

    if source.dlg_path is None:
        return {
            str(entry.voiceover).casefold(): clean_dialogue_text(entry.text)
            for _, entry in tlk
            if str(entry.voiceover).strip() and str(entry.text).strip()
        }

    try:
        from pykotor.resource.generics.dlg import read_dlg
    except ImportError as exception:
        raise RuntimeError(
            "PyKotor is required to read the installed KOTOR dialogue files."
        ) from exception

    with contextlib.redirect_stdout(io.StringIO()):
        dlg = read_dlg(source.dlg_path)

    mapping: dict[str, str] = {}
    for node in dlg.all_entries():
        voiceover = str(node.vo_resref).strip()
        if not voiceover.casefold().startswith(source.voiceover_prefix.casefold()):
            continue
        stringref = int(node.text.stringref)
        if stringref < 0:
            continue
        text = clean_dialogue_text(tlk.get(stringref).text)
        if text:
            mapping[voiceover.casefold()] = text
    return mapping


def load_and_prepare_audio(path: Path) -> tuple[np.ndarray, float] | None:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if not mono.size or not np.all(np.isfinite(mono)):
        return None

    threshold = max(float(np.max(np.abs(mono))) * 0.01, 1e-4)
    active = np.flatnonzero(np.abs(mono) >= threshold)
    if active.size:
        padding = int(sample_rate * 0.12)
        start = max(0, int(active[0]) - padding)
        end = min(len(mono), int(active[-1]) + padding + 1)
        mono = mono[start:end]

    duration = len(mono) / float(sample_rate)
    if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
        return None

    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    clipping_ratio = float(np.mean(np.abs(mono) >= 0.995))
    active_ratio = float(np.mean(np.abs(mono) >= max(rms * 0.35, 0.004)))
    if rms < 0.004 or clipping_ratio > 0.01 or active_ratio < 0.18:
        return None

    if sample_rate != TRAINING_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TRAINING_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TRAINING_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    return mono.astype(np.float32, copy=False), duration


def collect_samples(
    source: VoiceSource,
    mapping: dict[str, str],
) -> tuple[list[TrainingSample], dict[str, int]]:
    samples: list[TrainingSample] = []
    stats = {
        "audio_files": 0,
        "missing_transcript": 0,
        "excluded": 0,
        "unusable_text": 0,
        "unusable_audio": 0,
    }
    for path in sorted(source.audio_root.glob("*.wav")):
        stats["audio_files"] += 1
        if is_excluded_name(path, source.excluded_globs):
            stats["excluded"] += 1
            continue
        text = mapping.get(path.stem.casefold(), "")
        if not text:
            stats["missing_transcript"] += 1
            continue
        if not is_usable_dialogue(text):
            stats["unusable_text"] += 1
            continue
        prepared = load_and_prepare_audio(path)
        if prepared is None:
            stats["unusable_audio"] += 1
            continue
        _, duration = prepared
        samples.append(TrainingSample(path, text, duration))
    return samples, stats


def write_dataset(
    source: VoiceSource,
    output_root: Path,
    mapping: dict[str, str],
) -> dict:
    samples, stats = collect_samples(source, mapping)
    if not samples:
        raise RuntimeError(f"No usable {source.name} training samples were found.")

    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict]] = {"train": [], "validation": []}
    for sample in samples:
        prepared = load_and_prepare_audio(sample.source_audio)
        if prepared is None:
            continue
        audio, duration = prepared
        target = audio_root / f"{sample.source_audio.stem}.wav"
        sf.write(target, audio, TRAINING_SAMPLE_RATE, subtype="PCM_16")
        split = split_name(sample.source_audio)
        manifests[split].append(
            {
                "audio": str(target.resolve()),
                "text": sample.text,
                "duration": round(duration, 4),
                "dataset_id": 0,
            }
        )

    for split, entries in manifests.items():
        manifest_path = output_root / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    summary = {
        "voice": source.name,
        "sample_rate": TRAINING_SAMPLE_RATE,
        "train_samples": len(manifests["train"]),
        "validation_samples": len(manifests["validation"]),
        "total_duration_seconds": round(
            sum(entry["duration"] for entries in manifests.values() for entry in entries),
            2,
        ),
        "source": {
            "audio_root": str(source.audio_root),
            "tlk_path": str(source.tlk_path),
            "dlg_path": str(source.dlg_path) if source.dlg_path else "",
        },
        "filter_stats": stats,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def write_training_config(
    output_root: Path,
    engine_root: Path,
    max_steps: int,
) -> Path:
    snapshots_root = engine_root / "cache" / "models--openbmb--VoxCPM2" / "snapshots"
    snapshots = sorted(path for path in snapshots_root.glob("*") if path.is_dir())
    if not snapshots:
        raise RuntimeError(f"The local VoxCPM2 model was not found under {snapshots_root}.")
    config_path = output_root / "train_lora.yaml"
    values = {
        "pretrained_path": str(snapshots[-1].resolve()),
        "train_manifest": str((output_root / "train.jsonl").resolve()),
        "val_manifest": str((output_root / "validation.jsonl").resolve()),
        "save_path": str((output_root / "checkpoints").resolve()),
        "tensorboard": str((output_root / "logs").resolve()),
    }
    config = "\n".join(
        (
            f"pretrained_path: {json.dumps(values['pretrained_path'])}",
            f"train_manifest: {json.dumps(values['train_manifest'])}",
            f"val_manifest: {json.dumps(values['val_manifest'])}",
            "sample_rate: 16000",
            "out_sample_rate: 48000",
            "batch_size: 1",
            "grad_accum_steps: 8",
            "num_workers: 0",
            f"num_iters: {max_steps}",
            "log_interval: 10",
            "valid_interval: 250",
            "save_interval: 250",
            "learning_rate: 0.0001",
            "weight_decay: 0.01",
            "warmup_steps: 100",
            f"max_steps: {max_steps}",
            "max_batch_tokens: 4096",
            "max_grad_norm: 1.0",
            f"save_path: {json.dumps(values['save_path'])}",
            f"tensorboard: {json.dumps(values['tensorboard'])}",
            "lambdas:",
            "  loss/diff: 1.0",
            "  loss/stop: 1.0",
            "lora:",
            "  enable_lm: true",
            "  enable_dit: true",
            "  enable_proj: false",
            "  r: 32",
            "  alpha: 32",
            "  dropout: 0.0",
            "",
        )
    )
    config_path.write_text(config, encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare exact KOTOR dialogue/audio pairs for VoxCPM2 fine-tuning."
    )
    sources = default_sources()
    parser.add_argument("--voice", choices=tuple(sources), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "voxcpm_training",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    arguments = parser.parse_args()
    source = sources[arguments.voice]
    output_root = arguments.output_root / arguments.voice
    output_root.mkdir(parents=True, exist_ok=True)
    mapping = read_dialogue_mapping(source)
    summary = write_dataset(source, output_root, mapping)
    engine_root = Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "engines" / "voxcpm2"
    summary["training_config"] = str(
        write_training_config(output_root, engine_root, arguments.max_steps)
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
