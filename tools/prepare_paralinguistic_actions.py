from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


FRAME_MILLISECONDS = 10
MIN_REACTION_SECONDS = 0.18
MAX_REACTION_SECONDS = 3.5


@dataclass(frozen=True)
class ActionRecipe:
    name: str
    dia_tag: str
    tags: tuple[str, ...]
    intensity: float


ACTION_RECIPES = (
    ActionRecipe("sigh", "sighs", ("reflective", "weary", "restrained"), 0.35),
    ActionRecipe("laugh", "laughs", ("warm", "amused"), 0.55),
    ActionRecipe("chuckle", "chuckle", ("dry", "restrained"), 0.4),
    ActionRecipe("cough", "coughs", ("pained",), 0.55),
    ActionRecipe("throat_clear", "clears throat", ("restrained",), 0.35),
    ActionRecipe("sob", "sobs", ("wounded",), 0.65),
    ActionRecipe("whimper", "groans", ("wounded", "restrained"), 0.35),
    ActionRecipe("inhale", "inhales", ("startled",), 0.4),
    ActionRecipe("exhale", "exhales", ("weary", "restrained"), 0.35),
    ActionRecipe("choke", "coughs", ("pained", "forceful"), 0.75),
)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


def trim_to_activity(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not output.size:
        return output
    frame_size = max(1, int(sample_rate * FRAME_MILLISECONDS / 1000))
    frame_count = len(output) // frame_size
    if frame_count < 2:
        return output
    frames = output[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = max(0.0008, float(np.percentile(rms, 90)) * 0.035)
    active = np.flatnonzero(rms >= threshold)
    if not active.size:
        return output
    padding = int(sample_rate * 0.04)
    start = max(0, int(active[0]) * frame_size - padding)
    end = min(len(output), (int(active[-1]) + 1) * frame_size + padding)
    return output[start:end]


def extract_leading_reaction(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    output = trim_to_activity(audio, sample_rate)
    frame_size = max(1, int(sample_rate * FRAME_MILLISECONDS / 1000))
    frame_count = len(output) // frame_size
    if frame_count < 4:
        return output
    frames = output[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = max(0.0008, float(np.percentile(rms, 85)) * 0.025)
    quiet = rms < threshold
    minimum_quiet_frames = max(8, int(0.12 * sample_rate / frame_size))
    earliest_split = int(MIN_REACTION_SECONDS * sample_rate / frame_size)
    latest_split = min(
        frame_count - minimum_quiet_frames,
        int(MAX_REACTION_SECONDS * sample_rate / frame_size),
    )
    split_frame = 0
    cursor = earliest_split
    while cursor <= latest_split:
        if not quiet[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < frame_count and quiet[end]:
            end += 1
        if end - cursor >= minimum_quiet_frames:
            split_frame = cursor
            break
        cursor = end
    if split_frame:
        output = output[: split_frame * frame_size]
    output = trim_to_activity(output, sample_rate)
    return output


def normalize_reaction(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    output = trim_to_activity(audio, sample_rate).copy()
    if not output.size:
        return output
    rms = float(np.sqrt(np.mean(np.square(output)) + 1e-12))
    target_rms = 10.0 ** (-20.0 / 20.0)
    if rms > 0.0:
        output *= min(target_rms / rms, 2.0)
    peak = float(np.max(np.abs(output)))
    if peak > 0.95:
        output *= 0.95 / peak
    fade = min(len(output) // 2, int(0.012 * sample_rate))
    if fade:
        output[:fade] *= np.linspace(0.0, 1.0, fade)
        output[-fade:] *= np.linspace(1.0, 0.0, fade)
    return output


def select_clone_prompt(profile: dict, profile_root: Path) -> tuple[Path, str]:
    candidates = []
    for anchor in profile.get("style_anchors", []):
        text = str(anchor.get("prompt_text", "")).strip()
        relative = str(anchor.get("audio", "")).strip()
        duration = float(anchor.get("features", {}).get("duration_seconds", 0.0))
        path = (profile_root / relative).resolve()
        if text and path.is_file() and 4.0 <= duration <= 10.0:
            candidates.append((abs(duration - 7.0), path, text))
    if not candidates:
        raise RuntimeError("No 4-10 second character prompt is available.")
    _, path, text = min(candidates, key=lambda item: item[0])
    return path, text


def preserved_manifest_entries(
    character_root: Path,
    refreshed_actions: set[str],
) -> list[dict]:
    manifest_path = character_root / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and str(entry.get("name", "")).casefold() not in refreshed_actions
        and (character_root / str(entry.get("audio", ""))).is_file()
    ]


def publish_action_bank(
    character_root: Path,
    stage_root: Path,
    manifest: dict,
    replace_file=os.replace,
) -> None:
    character_root.mkdir(parents=True, exist_ok=True)
    for staged in sorted(stage_root.glob("*.wav")):
        replace_file(staged, character_root / staged.name)

    manifest_path = character_root / "manifest.json"
    staged_manifest = character_root / ".manifest.json.tmp"
    try:
        staged_manifest.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        replace_file(staged_manifest, manifest_path)
    finally:
        staged_manifest.unlink(missing_ok=True)

    referenced = {
        str(entry.get("audio", ""))
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict)
    }
    for stale in character_root.glob("*.wav"):
        if stale.name not in referenced:
            stale.unlink()


def build_action_bank(
    voice_id: str,
    output_root: Path,
    variants: int,
    selected_actions: set[str],
    model=None,
    torch_module=None,
) -> dict:
    if model is None or torch_module is None:
        try:
            import torch
            from dia.model import Dia
        except ImportError as exception:
            raise RuntimeError(
                "Dia is not installed. Run tools/install_dia_actions.ps1 first."
            ) from exception
        torch_module = torch
        model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            compute_dtype="float16",
        )

    profile_root = (
        Path(os.environ["LOCALAPPDATA"])
        / "VoxStudio"
        / "voxcpm_profiles"
        / voice_id
    )
    profile_path = profile_root / "profile.json"
    if not profile_path.is_file():
        raise RuntimeError(f"Character profile is missing: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    prompt_path, prompt_text = select_clone_prompt(profile, profile_root)

    character_root = output_root / voice_id
    character_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    recipes = tuple(
        recipe
        for recipe in ACTION_RECIPES
        if not selected_actions or recipe.name in selected_actions
    )
    with tempfile.TemporaryDirectory(prefix="_staging-", dir=character_root) as stage:
        stage_root = Path(stage)
        for recipe in recipes:
            for variant in range(variants):
                seed = 7100 + variant + sum(ord(char) for char in recipe.name)
                torch_module.manual_seed(seed)
                target_text = f"[S1] ({recipe.dia_tag})"
                generated = model.generate(
                    f"[S1] {prompt_text} {target_text}",
                    audio_prompt=str(prompt_path),
                    max_tokens=1024,
                    use_torch_compile=False,
                    verbose=False,
                    cfg_scale=4.0,
                    temperature=1.55 + variant * 0.12,
                    top_p=0.9,
                    cfg_filter_top_k=50,
                )
                raw_path = stage_root / f"_{recipe.name}_{variant + 1:02d}_raw.wav"
                model.save_audio(str(raw_path), generated)
                raw, sample_rate = load_mono(raw_path)
                reaction = normalize_reaction(
                    extract_leading_reaction(raw, sample_rate),
                    sample_rate,
                )
                duration = len(reaction) / float(sample_rate)
                raw_path.unlink(missing_ok=True)
                if not MIN_REACTION_SECONDS <= duration < MAX_REACTION_SECONDS:
                    continue
                target = stage_root / f"{recipe.name}_{variant + 1:02d}.wav"
                sf.write(target, reaction, sample_rate, subtype="PCM_16")
                entries.append(
                    {
                        "name": recipe.name,
                        "audio": target.name,
                        "tags": list(recipe.tags),
                        "intensity": recipe.intensity,
                        "source": "Dia voice clone from licensed character reference",
                        "generator": "nari-labs/Dia-1.6B-0626",
                        "seed": seed,
                        "duration_seconds": round(duration, 3),
                    }
                )
        refreshed_actions = {str(entry["name"]) for entry in entries}
        entries = (
            preserved_manifest_entries(character_root, refreshed_actions) + entries
        )
        manifest = {
            "format_version": 1,
            "voice_id": voice_id,
            "character": profile.get("name", voice_id),
            "clone_prompt": str(prompt_path),
            "entries": entries,
        }
        publish_action_bank(character_root, stage_root, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cached character vocalizations with Dia."
    )
    parser.add_argument("--voice-id", action="append", required=True)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(os.environ["LOCALAPPDATA"])
            / "VoxStudio"
            / "generated_vocal_actions"
        ),
    )
    arguments = parser.parse_args()
    if arguments.variants < 1 or arguments.variants > 4:
        parser.error("--variants must be between 1 and 4")
    selected_actions = {
        re.sub(r"[^a-z_]", "", action.casefold()) for action in arguments.action
    }
    try:
        import torch
        from dia.model import Dia
    except ImportError as exception:
        raise RuntimeError(
            "Dia is not installed. Run tools/install_dia_actions.ps1 first."
        ) from exception
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B-0626",
        compute_dtype="float16",
    )
    for voice_id in arguments.voice_id:
        manifest = build_action_bank(
            voice_id,
            arguments.output,
            arguments.variants,
            selected_actions,
            model=model,
            torch_module=torch,
        )
        print(
            f"built {manifest['character']}: {len(manifest['entries'])} "
            "cached vocalizations"
        )


if __name__ == "__main__":
    main()
