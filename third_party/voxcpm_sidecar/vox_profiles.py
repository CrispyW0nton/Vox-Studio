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


@dataclass(frozen=True)
class VocalActionAsset:
    path: Path
    tags: tuple[str, ...]
    intensity: float
    source: str
    mode: str = "direct"
    prompt_text: str = ""


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


def resolve_profile_vocal_actions(
    profile: dict,
    profile_dir: Path,
    action_name: str,
) -> tuple[VocalActionAsset, ...]:
    actions = profile.get("vocal_actions", {})
    if not isinstance(actions, dict):
        raise ValueError("Character profile vocal actions are invalid.")
    configured = actions.get(action_name, ())
    if isinstance(configured, (str, dict)):
        configured = (configured,)
    if not isinstance(configured, (list, tuple)):
        raise ValueError(f"Character vocal action '{action_name}' is invalid.")

    resolved_root = profile_dir.resolve()
    assets: list[VocalActionAsset] = []
    for item in configured:
        metadata = item if isinstance(item, dict) else {"audio": item}
        relative = str(metadata.get("audio", "")).strip()
        if not relative:
            continue
        resolved = (profile_dir / relative).resolve()
        if resolved_root not in resolved.parents or not resolved.is_file():
            raise ValueError(
                f"Character vocal action '{action_name}' is missing or invalid."
            )
        tags = tuple(
            dict.fromkeys(
                str(tag).strip().casefold()
                for tag in metadata.get("tags", [])
                if str(tag).strip()
            )
        )
        try:
            intensity = float(metadata.get("intensity", 0.5))
        except (TypeError, ValueError):
            intensity = 0.5
        assets.append(
            VocalActionAsset(
                path=resolved,
                tags=tags,
                intensity=max(0.0, min(1.0, intensity)),
                source=str(metadata.get("source", relative)).strip() or relative,
                mode=(
                    "prompt"
                    if str(metadata.get("mode", "direct")).casefold() == "prompt"
                    else "direct"
                ),
                prompt_text=str(metadata.get("prompt_text", "")).strip(),
            )
        )
    return tuple(assets)


def vocal_action_target(cue: str, delivery_label: str) -> tuple[set[str], float]:
    text = f"{cue} {delivery_label}".casefold()
    tags = {
        tag
        for tag, pattern in (
            ("restrained", r"\b(?:quiet|soft|subtle|small|restrained|gentle)\b"),
            ("pained", r"\b(?:pain|pained|hurt|wounded|dying|agony)\b"),
            ("weary", r"\b(?:weary|tired|exhausted|spent|relieved)\b"),
            ("dry", r"\b(?:dry|wry|rueful|sarcastic|bitter)\b"),
            ("forceful", r"\b(?:hard|forceful|loud|strong|violent|intense|urgent)\b"),
            ("fearful", r"\b(?:fear|afraid|shocked|startled|terrified)\b"),
            ("reflective", r"\b(?:reflective|wistful|sad|sorrowful)\b"),
        )
        if re.search(pattern, text)
    }
    if re.search(r"\b(?:quiet|soft|subtle|small|restrained|gentle)\b", text):
        intensity = 0.25
    elif re.search(r"\b(?:hard|forceful|loud|violent|intense|scream|dying)\b", text):
        intensity = 0.9
    elif re.search(r"\b(?:urgent|pained|wounded|fearful)\b", text):
        intensity = 0.7
    else:
        intensity = 0.5
    return tags, intensity


def select_vocal_action_asset(
    assets: tuple[VocalActionAsset, ...],
    cue: str,
    delivery_label: str,
    seed: int,
    avoid_path: Path | None = None,
) -> VocalActionAsset | None:
    if not assets:
        return None
    target_tags, target_intensity = vocal_action_target(cue, delivery_label)

    def score(asset: VocalActionAsset) -> float:
        overlap = len(target_tags.intersection(asset.tags))
        mismatch = len(target_tags.difference(asset.tags)) if asset.tags else 0
        return (
            overlap * 2.0
            - mismatch * 0.15
            - abs(asset.intensity - target_intensity)
            + (0.35 if asset.mode == "direct" else 0.0)
            - (0.8 if avoid_path is not None and asset.path == avoid_path else 0.0)
        )

    ranked = sorted(assets, key=lambda asset: (-score(asset), str(asset.path)))
    best_score = score(ranked[0])
    close = tuple(asset for asset in ranked if score(asset) >= best_score - 0.25)
    return close[seed % len(close)]


def anchor_direction_adjustment(anchor: dict, delivery_label: str) -> float:
    tags = {
        str(tag).strip().casefold()
        for tag in anchor.get("delivery_tags", [])
        if str(tag).strip()
    }
    target = delivery_label.strip().casefold()
    if not tags or target in {"", "natural", "neutral", "measured", "calm"}:
        return 0.0
    if target in tags:
        return -1.0
    related = {
        "sarcastic": {"wry"},
        "wry": {"sarcastic"},
        "wounded": {"reflective"},
        "reflective": {"wounded"},
        "urgent": {"emphatic"},
        "emphatic": {"urgent", "resolute"},
        "resolute": {"emphatic"},
    }
    if tags.intersection(related.get(target, set())):
        return -0.45
    return 0.35


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
