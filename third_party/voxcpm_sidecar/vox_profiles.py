from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SynthesisInputs:
    prompt_path: Path
    prompt_text: str
    reference_path: Path
    style_source: str


def control_identity_instruction(profile: dict) -> str:
    if str(profile.get("lora_adapter", "")).strip():
        return "Keep the trained voice stable."
    return str(profile.get("control_instruction", "")).strip()


def text_identity_instruction(profile: dict) -> str:
    explicit = str(profile.get("text_control_instruction", "")).strip()
    if explicit:
        return explicit

    instruction = control_identity_instruction(profile)
    if not instruction:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", instruction)
    identity_sentences: list[str] = []
    for sentence in sentences:
        if re.search(
            r"\b(performer|performance|microphone|recording|source voice)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            break
        identity_sentences.append(sentence)
    return " ".join(identity_sentences).strip()


def resolve_profile_asset(
    profile: dict,
    profile_dir: Path,
    key: str,
) -> Path | None:
    relative = str(profile.get(key, "")).strip()
    if not relative:
        return None
    resolved_root = profile_dir.resolve()
    resolved = (profile_dir / relative).resolve()
    if resolved_root not in resolved.parents or not resolved.exists():
        raise ValueError(f"Character profile asset '{key}' is missing or invalid.")
    return resolved


def synthesis_inputs(
    profile: dict,
    performer_path: Path,
    performer_text: str,
    reference_path: Path,
    style_anchor: tuple[Path, str, str] | None,
) -> SynthesisInputs:
    if profile.get("use_stable_character_identity", False) or style_anchor is None:
        return SynthesisInputs(
            performer_path,
            performer_text,
            reference_path,
            "performer",
        )
    anchor_path, anchor_text, source = style_anchor
    return SynthesisInputs(anchor_path, anchor_text, anchor_path, source)


def generation_options(
    spoken_text: str,
    prompt_path: Path,
    prompt_text: str,
    reference_path: Path,
    profile: dict,
    control_instruction: str,
) -> dict:
    cfg_value = float(profile.get("cfg_value", 2.0))
    controlled_cfg_value = float(profile.get("controlled_cfg_value", 0.0))
    if control_instruction and controlled_cfg_value > 0.0:
        cfg_value = controlled_cfg_value
    options = {
        "text": (
            f"({control_instruction}){spoken_text}"
            if control_instruction
            else spoken_text
        ),
        "reference_wav_path": str(reference_path),
        "cfg_value": cfg_value,
        "inference_timesteps": int(profile.get("inference_timesteps", 10)),
        "normalize": False,
    }
    if not control_instruction:
        options["prompt_wav_path"] = str(prompt_path)
        options["prompt_text"] = prompt_text
    return options


def activate_lora(
    model,
    adapter_path: Path | None,
    state: dict[str, object],
) -> tuple[int, int]:
    if adapter_path is None:
        model.set_lora_enabled(False)
        state["enabled"] = False
        return 0, 0

    resolved_path = adapter_path.resolve()
    weights = next(
        (
            candidate
            for candidate in (
                resolved_path / "lora_weights.safetensors",
                resolved_path / "lora_weights.ckpt",
            )
            if candidate.exists()
        ),
        None,
    )
    if weights is None:
        raise RuntimeError(f"No LoRA weights were found in {resolved_path}.")
    weights_stat = weights.stat()
    signature = (
        str(resolved_path),
        weights_stat.st_mtime_ns,
        weights_stat.st_size,
    )
    if state.get("loaded_signature") == signature:
        model.set_lora_enabled(True)
        state["enabled"] = True
        return 0, 0

    model.set_lora_enabled(False)
    state["enabled"] = False
    loaded, skipped = model.load_lora(str(resolved_path))
    if not loaded:
        raise RuntimeError(
            f"No compatible LoRA weights were found in {resolved_path}."
        )
    model.set_lora_enabled(True)
    state["loaded_signature"] = signature
    state["enabled"] = True
    return len(loaded), len(skipped)
