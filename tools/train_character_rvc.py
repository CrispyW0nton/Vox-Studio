from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    parser = argparse.ArgumentParser(
        description="Train and install a character-specific RVC model from a VoxCPM profile."
    )
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=local_app_data / "VoxStudio/voxcpm_profiles",
    )
    parser.add_argument(
        "--rvc-root",
        type=Path,
        default=local_app_data / "VoxStudio/training/RVC-WebUI",
    )
    parser.add_argument(
        "--training-source-root",
        type=Path,
        default=local_app_data / "VoxStudio/rvc_training_sources",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=local_app_data / "VoxStudio/rvc_models",
    )
    return parser.parse_args()


def safe_identifier(value: str, label: str) -> str:
    if not value or any(not (character.isalnum() or character in "_-") for character in value):
        raise ValueError(f"{label} must contain only letters, numbers, underscores, or hyphens")
    return value


def load_profile(profiles_root: Path, voice_id: str) -> dict:
    profile_path = profiles_root.resolve() / safe_identifier(voice_id, "Voice id") / "profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Character profile not found: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if str(profile.get("voice_id", "")) != voice_id:
        raise ValueError("Character profile voice id does not match the requested voice")
    return profile


def prepare_training_sources(
    profile: dict,
    training_source_root: Path,
    model_id: str,
) -> Path:
    dataset_root = training_source_root.resolve() / safe_identifier(model_id, "Model id")
    dataset_root.mkdir(parents=True, exist_ok=True)
    sources = [Path(value).resolve() for value in profile.get("selected_sources", [])]
    available_sources = [path for path in sources if path.is_file()]
    if not available_sources:
        raise FileNotFoundError("The character profile has no available training audio")

    expected_targets = {
        dataset_root / f"line_{index:04d}{source.suffix.lower()}"
        for index, source in enumerate(available_sources, start=1)
    }
    for stale_target in dataset_root.glob("line_*"):
        if stale_target.is_file() and stale_target not in expected_targets:
            stale_target.unlink()

    for index, source in enumerate(available_sources, start=1):
        target = dataset_root / f"line_{index:04d}{source.suffix.lower()}"
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return dataset_root


def run_command(command: list[str], working_directory: Path) -> None:
    print(f"Running: {' '.join(command[1:])}", flush=True)
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(working_directory)
        if not existing_python_path
        else str(working_directory) + os.pathsep + existing_python_path
    )
    subprocess.run(command, cwd=working_directory, env=environment, check=True)


def prepare_filelist(rvc_root: Path, experiment: str) -> None:
    experiment_root = rvc_root / "logs" / experiment
    gt_wavs = experiment_root / "0_gt_wavs"
    features = experiment_root / "3_feature768"
    f0 = experiment_root / "2a_f0"
    f0nsf = experiment_root / "2b-f0nsf"
    names = (
        {path.stem for path in gt_wavs.glob("*.wav")}
        & {path.stem for path in features.glob("*.npy")}
        & {path.name.removesuffix(".wav.npy") for path in f0.glob("*.wav.npy")}
        & {path.name.removesuffix(".wav.npy") for path in f0nsf.glob("*.wav.npy")}
    )
    if not names:
        raise RuntimeError("RVC preprocessing did not produce a complete training example")

    rows = [
        "|".join(
            (
                (gt_wavs / f"{name}.wav").as_posix(),
                (features / f"{name}.npy").as_posix(),
                (f0 / f"{name}.wav.npy").as_posix(),
                (f0nsf / f"{name}.wav.npy").as_posix(),
                "0",
            )
        )
        for name in sorted(names)
    ]
    mute_root = rvc_root / "logs" / "mute"
    mute_row = "|".join(
        (
            (mute_root / "0_gt_wavs/mute40k.wav").as_posix(),
            (mute_root / "3_feature768/mute.npy").as_posix(),
            (mute_root / "2a_f0/mute.wav.npy").as_posix(),
            (mute_root / "2b-f0nsf/mute.wav.npy").as_posix(),
            "0",
        )
    )
    rows.extend((mute_row, mute_row))
    random.Random(1234).shuffle(rows)
    (experiment_root / "filelist.txt").write_text(
        "\n".join(rows),
        encoding="utf-8",
    )

    config_path = experiment_root / "config.json"
    if not config_path.exists():
        shutil.copy2(rvc_root / "configs/v1/40k.json", config_path)


def run_training(arguments: argparse.Namespace, dataset_root: Path) -> None:
    rvc_root = arguments.rvc_root.resolve()
    if not (rvc_root / "train/train.py").is_file():
        raise FileNotFoundError(f"RVC training runtime not found: {rvc_root}")

    experiment = safe_identifier(arguments.experiment, "Experiment")
    experiment_root = rvc_root / "logs" / experiment
    experiment_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    workers = max(1, min(os.cpu_count() or 1, 16))

    run_command(
        [
            python,
            "-m",
            "train.preprocess",
            str(dataset_root),
            "40000",
            str(workers),
            str(experiment_root),
            "False",
            "3.7",
        ],
        rvc_root,
    )
    (experiment_root / "extract_f0_feature.log").write_text("", encoding="utf-8")
    run_command(
        [
            python,
            "-m",
            "train.dataset.extract_f0",
            "cuda",
            "1",
            "0",
            "0",
            str(experiment_root),
            "True",
        ],
        rvc_root,
    )
    run_command(
        [
            python,
            "-m",
            "train.dataset.extract_hubert_feature",
            "cuda:0",
            "1",
            "0",
            "0",
            str(experiment_root),
            "v2",
            "True",
        ],
        rvc_root,
    )
    prepare_filelist(rvc_root, experiment)
    run_command(
        [
            python,
            "-m",
            "train.train",
            "-e",
            experiment,
            "-sr",
            "40k",
            "-f0",
            "1",
            "-bs",
            str(max(1, arguments.batch_size)),
            "-g",
            "0",
            "-te",
            str(max(1, arguments.epochs)),
            "-se",
            str(max(1, arguments.save_every)),
            "-pg",
            "assets/pretrained_v2/f0G40k.pth",
            "-pd",
            "assets/pretrained_v2/f0D40k.pth",
            "-l",
            "1",
            "-c",
            "1",
            "-sw",
            "1",
            "-v",
            "v2",
        ],
        rvc_root,
    )
    run_command(
        [
            python,
            "-m",
            "train.train_index",
            experiment,
            "v2",
            str(rvc_root / "assets/indices"),
            str(workers),
        ],
        rvc_root,
    )


def newest_matching(root: Path, pattern: str) -> Path:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Training output is missing: {root / pattern}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def install_model(arguments: argparse.Namespace) -> Path:
    rvc_root = arguments.rvc_root.resolve()
    experiment = safe_identifier(arguments.experiment, "Experiment")
    model_id = safe_identifier(arguments.model_id, "Model id")
    weights = newest_matching(rvc_root / "assets/weights", f"{experiment}.pth")
    index = newest_matching(rvc_root / "logs" / experiment, "added_*.index")

    target_root = arguments.model_root.resolve() / model_id
    target_root.mkdir(parents=True, exist_ok=True)
    target_weights = target_root / f"{experiment}.pth"
    target_index = target_root / f"{experiment}.index"
    shutil.copy2(weights, target_weights)
    shutil.copy2(index, target_index)

    manifest = {
        "id": model_id,
        "display_name": arguments.display_name,
        "pth_path": str(target_weights),
        "index_path": str(target_index),
        "sample_rate": 40000,
        "notes": (
            f"Character-trained RVC v2 model built from "
            f"{arguments.epochs} epochs of clean profile dialogue"
        ),
        "imported_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "character_voice_id": arguments.voice_id,
    }
    (target_root / "model.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return target_root


def main() -> int:
    arguments = parse_args()
    profile = load_profile(arguments.profiles_root, arguments.voice_id)
    dataset_root = prepare_training_sources(
        profile,
        arguments.training_source_root,
        arguments.model_id,
    )
    print(f"Prepared {len(profile.get('selected_sources', []))} source lines in {dataset_root}")
    run_training(arguments, dataset_root)
    installed = install_model(arguments)
    print(f"Installed character model at {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
