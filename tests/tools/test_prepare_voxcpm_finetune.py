from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "prepare_voxcpm_finetune.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_voxcpm_finetune", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DialogueCleaningTests(unittest.TestCase):
    def test_removes_acting_directions_but_preserves_dialogue(self) -> None:
        text = (
            "{Incredulous, during a fight}Uh, General, I think you've got "
            "more important things to worry about. [Bao-Dur turns away]"
        )

        self.assertEqual(
            MODULE.clean_dialogue_text(text),
            "Uh, General, I think you've got more important things to worry about.",
        )

    def test_rejects_nonverbal_and_too_short_lines(self) -> None:
        self.assertFalse(MODULE.is_usable_dialogue("::Laughs::"))
        self.assertFalse(MODULE.is_usable_dialogue("Yes, General."))
        self.assertTrue(MODULE.is_usable_dialogue("Yes, General, I understand."))

    def test_split_is_deterministic(self) -> None:
        path = Path("nm40accart02012_.wav")

        self.assertEqual(MODULE.split_name(path), MODULE.split_name(path))
        self.assertIn(MODULE.split_name(path), {"train", "validation"})

    def test_training_config_uses_local_model_and_windows_safe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "training data"
            snapshot = (
                root
                / "engine"
                / "cache"
                / "models--openbmb--VoxCPM2"
                / "snapshots"
                / "model"
            )
            output.mkdir(parents=True)
            snapshot.mkdir(parents=True)

            path = MODULE.write_training_config(output, root / "engine", 750)
            config = path.read_text(encoding="utf-8")

        self.assertIn(f"pretrained_path: {MODULE.json.dumps(str(snapshot))}", config)
        self.assertIn("batch_size: 1", config)
        self.assertIn("num_workers: 0", config)
        self.assertIn("max_steps: 750", config)

    def test_reads_exact_dialogue_manifest_for_character_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "dialogue_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "entries": {
                            "003003kreia006.wav": {
                                "text": "Save your pity. I am here to save you."
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            source = MODULE.VoiceSource(
                name="Kreia",
                audio_root=root,
                tlk_path=root / "dialog.tlk",
                dialogue_manifest=manifest,
            )

            mapping = MODULE.read_dialogue_mapping(source)

        self.assertEqual(
            mapping["003003kreia006"],
            "Save your pity. I am here to save you.",
        )


class AudioPreparationTests(unittest.TestCase):
    def test_resamples_and_trims_clean_speech_like_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "line.wav"
            sample_rate = 48_000
            silence = np.zeros(sample_rate // 2, dtype=np.float32)
            time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
            speech = 0.12 * np.sin(2.0 * np.pi * 180.0 * time)
            sf.write(path, np.concatenate((silence, speech, silence)), sample_rate)

            prepared = MODULE.load_and_prepare_audio(path)

        self.assertIsNotNone(prepared)
        audio, duration = prepared
        self.assertAlmostEqual(len(audio) / MODULE.TRAINING_SAMPLE_RATE, duration, delta=0.1)
        self.assertLess(duration, 2.5)

    def test_rejects_static_and_clipped_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "static.wav"
            sf.write(path, np.ones(32_000, dtype=np.float32), 16_000)

            prepared = MODULE.load_and_prepare_audio(path)

        self.assertIsNone(prepared)


if __name__ == "__main__":
    unittest.main()
