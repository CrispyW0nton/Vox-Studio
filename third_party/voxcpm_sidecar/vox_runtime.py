"""Explicit runtime choices for the interactive engine and directed narration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeOptions:
    optimize: bool
    allocate_lora: bool


def runtime_options(name: str) -> RuntimeOptions:
    if name == "default":
        return RuntimeOptions(optimize=True, allocate_lora=True)
    if name == "narration":
        # Short isolated narration workers do not benefit from compilation warmup
        # or unused adapter layers. Preserve the native model and sample rate.
        return RuntimeOptions(optimize=False, allocate_lora=False)
    raise ValueError("Unknown Vox runtime profile; use default or narration.")


def validate_runtime_adapter(name: str, has_lora: bool) -> None:
    options = runtime_options(name)
    if has_lora and not options.allocate_lora:
        raise ValueError("A trained LoRA voice requires the default Vox runtime.")
