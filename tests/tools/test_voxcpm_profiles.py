from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "voxcpm_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from vox_profiles import (  # noqa: E402
    activate_lora,
    control_identity_instruction,
    generation_options,
    resolve_profile_asset,
    synthesis_inputs,
    text_identity_instruction,
)


class SynthesisInputTests(unittest.TestCase):
    def test_trained_profile_uses_performer_delivery_and_stable_reference(self) -> None:
        performer = Path("performance.wav")
        reference = Path("character.wav")
        anchor = (Path("angry-game-line.wav"), "Move now!", "angry-game-line")

        selected = synthesis_inputs(
            {"use_stable_character_identity": True},
            performer,
            "I understand.",
            reference,
            anchor,
        )

        self.assertEqual(selected.prompt_path, performer)
        self.assertEqual(selected.prompt_text, "I understand.")
        self.assertEqual(selected.reference_path, reference)
        self.assertEqual(selected.style_source, "performer")

    def test_legacy_profile_keeps_anchor_behavior(self) -> None:
        anchor = (Path("game-line.wav"), "Hello there.", "game-line")

        selected = synthesis_inputs(
            {},
            Path("performance.wav"),
            "Hello.",
            Path("reference.wav"),
            anchor,
        )

        self.assertEqual(selected.prompt_path, anchor[0])
        self.assertEqual(selected.reference_path, anchor[0])
        self.assertEqual(selected.style_source, "game-line")


class ControlInstructionTests(unittest.TestCase):
    def test_trained_adapter_uses_neutral_identity_instruction(self) -> None:
        instruction = control_identity_instruction(
            {
                "lora_adapter": "lora",
                "control_instruction": "Keep Carth's earnest military timbre.",
            }
        )

        self.assertEqual(instruction, "Keep the trained voice stable.")

    def test_legacy_profile_keeps_its_character_instruction(self) -> None:
        instruction = control_identity_instruction(
            {"control_instruction": "Keep Atton's casual drawl."}
        )

        self.assertEqual(instruction, "Keep Atton's casual drawl.")

    def test_text_mode_removes_performer_dependent_directions(self) -> None:
        instruction = text_identity_instruction(
            {
                "control_instruction": (
                    "Keep Atton's guarded youthful timbre and casual drawl. "
                    "Follow the performer's pace and emotional intensity exactly. "
                    "Do not add tension unless it is present."
                )
            }
        )

        self.assertEqual(
            instruction,
            "Keep Atton's guarded youthful timbre and casual drawl.",
        )

    def test_text_mode_accepts_a_dedicated_identity_instruction(self) -> None:
        instruction = text_identity_instruction(
            {
                "control_instruction": "Follow the performer's pace.",
                "text_control_instruction": "Keep the character voice stable.",
            }
        )

        self.assertEqual(instruction, "Keep the character voice stable.")


class ProfileAssetTests(unittest.TestCase):
    def test_resolves_adapter_inside_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "lora" / "lora_weights.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"weights")

            resolved = resolve_profile_asset(
                {"lora_adapter": "lora"},
                root,
                "lora_adapter",
            )

        self.assertEqual(resolved, adapter.parent.resolve())

    def test_rejects_adapter_outside_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "profile"
            root.mkdir()
            outside = root.parent / "outside"
            outside.mkdir()

            with self.assertRaises(ValueError):
                resolve_profile_asset(
                    {"lora_adapter": "..\\outside"},
                    root,
                    "lora_adapter",
                )


class GenerationOptionTests(unittest.TestCase):
    def test_controlled_generation_never_uses_performer_as_speaker_prompt(self) -> None:
        options = generation_options(
            "Trust me.",
            Path("performer.wav"),
            "Trust me.",
            Path("carth.wav"),
            {"cfg_value": 2.0, "controlled_cfg_value": 2.5},
            "Use a calm, measured delivery.",
        )

        self.assertEqual(options["reference_wav_path"], "carth.wav")
        self.assertNotIn("prompt_wav_path", options)
        self.assertNotIn("prompt_text", options)
        self.assertEqual(options["cfg_value"], 2.5)
        self.assertTrue(options["text"].startswith("(Use a calm"))

    def test_legacy_generation_uses_selected_prompt(self) -> None:
        options = generation_options(
            "Trust me.",
            Path("anchor.wav"),
            "I understand.",
            Path("anchor.wav"),
            {},
            "",
        )

        self.assertEqual(options["prompt_wav_path"], "anchor.wav")
        self.assertEqual(options["prompt_text"], "I understand.")


class FakeModel:
    def __init__(self) -> None:
        self.enabled: list[bool] = []
        self.loads: list[str] = []

    def set_lora_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def load_lora(self, path: str) -> tuple[list[str], list[str]]:
        self.loads.append(path)
        return ["layer.lora_A", "layer.lora_B"], []


class AdapterActivationTests(unittest.TestCase):
    def test_loads_once_and_reenables_matching_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir)
            (adapter / "lora_weights.safetensors").write_bytes(b"weights")
            model = FakeModel()
            state: dict[str, object] = {}

            first = activate_lora(model, adapter, state)
            second = activate_lora(model, adapter, state)

        self.assertEqual(first, (2, 0))
        self.assertEqual(second, (0, 0))
        self.assertEqual(len(model.loads), 1)
        self.assertTrue(model.enabled[-1])

    def test_disables_adapter_for_untrained_profile(self) -> None:
        model = FakeModel()
        state = {"loaded_signature": ("old", 1, 1), "enabled": True}

        activate_lora(model, None, state)

        self.assertFalse(model.enabled[-1])
        self.assertFalse(state["enabled"])


if __name__ == "__main__":
    unittest.main()
