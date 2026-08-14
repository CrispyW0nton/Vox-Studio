from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "prepare_paralinguistic_actions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_paralinguistic_actions",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReactionExtractionTests(unittest.TestCase):
    def test_rejects_click_length_chuckle_fragments(self) -> None:
        chuckle = next(
            recipe for recipe in MODULE.ACTION_RECIPES if recipe.name == "chuckle"
        )

        self.assertFalse(MODULE.reaction_duration_is_valid(chuckle, 0.25))
        self.assertTrue(MODULE.reaction_duration_is_valid(chuckle, 0.8))
        self.assertLess(MODULE.generation_token_limit(chuckle), 1024)

    def test_preserves_valid_previous_action_when_refresh_yields_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "chuckle_01.wav").write_bytes(b"wave")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "name": "chuckle",
                                "audio": "chuckle_01.wav",
                                "duration_seconds": 0.8,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            chuckle = next(
                recipe for recipe in MODULE.ACTION_RECIPES if recipe.name == "chuckle"
            )

            preserved = MODULE.preserved_manifest_entries(
                root,
                refreshed_actions=set(),
                selected_recipes={"chuckle": chuckle},
            )

        self.assertEqual(len(preserved), 1)

    def test_discards_invalid_previous_action_when_refresh_yields_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "chuckle_01.wav").write_bytes(b"wave")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "name": "chuckle",
                                "audio": "chuckle_01.wav",
                                "duration_seconds": 0.18,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            chuckle = next(
                recipe for recipe in MODULE.ACTION_RECIPES if recipe.name == "chuckle"
            )

            preserved = MODULE.preserved_manifest_entries(
                root,
                refreshed_actions=set(),
                selected_recipes={"chuckle": chuckle},
            )

        self.assertEqual(preserved, [])

    def test_extracts_reaction_before_silence_and_spoken_phrase(self) -> None:
        sample_rate = 48_000
        time = np.arange(int(0.55 * sample_rate), dtype=np.float32) / sample_rate
        reaction = 0.14 * np.sin(2.0 * np.pi * 170.0 * time)
        silence = np.zeros(int(0.25 * sample_rate), dtype=np.float32)
        speech = 0.12 * np.sin(
            2.0
            * np.pi
            * 230.0
            * np.arange(sample_rate, dtype=np.float32)
            / sample_rate
        )

        extracted = MODULE.extract_leading_reaction(
            np.concatenate((reaction, silence, speech)),
            sample_rate,
        )

        self.assertGreater(len(extracted) / sample_rate, 0.45)
        self.assertLess(len(extracted) / sample_rate, 0.70)

    def test_normalizes_without_clipping(self) -> None:
        sample_rate = 48_000
        audio = np.full(sample_rate // 2, 0.8, dtype=np.float32)

        normalized = MODULE.normalize_reaction(audio, sample_rate)

        self.assertLessEqual(float(np.max(np.abs(normalized))), 0.95)
        self.assertAlmostEqual(float(normalized[0]), 0.0, places=5)
        self.assertAlmostEqual(float(normalized[-1]), 0.0, places=5)

    def test_leaves_overlong_unsplit_reaction_for_quality_rejection(self) -> None:
        sample_rate = 48_000
        time = np.arange(int(4.2 * sample_rate), dtype=np.float32) / sample_rate
        reaction = 0.14 * np.sin(2.0 * np.pi * 170.0 * time)

        extracted = MODULE.extract_leading_reaction(reaction, sample_rate)

        self.assertGreater(len(extracted) / sample_rate, MODULE.MAX_REACTION_SECONDS)

    def test_preserves_unrelated_generated_actions_on_partial_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "laugh_01.wav").write_bytes(b"wave")
            (root / "sigh_01.wav").write_bytes(b"wave")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {"name": "laugh", "audio": "laugh_01.wav"},
                            {"name": "sigh", "audio": "sigh_01.wav"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            preserved = MODULE.preserved_manifest_entries(root, {"sigh"})

        self.assertEqual(preserved, [{"name": "laugh", "audio": "laugh_01.wav"}])

    def test_failed_publication_keeps_old_manifest_assets_resolvable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "bank"
            stage = Path(temp_dir) / "stage"
            root.mkdir()
            stage.mkdir()
            (root / "sigh_01.wav").write_bytes(b"old")
            old_manifest = {
                "entries": [{"name": "sigh", "audio": "sigh_01.wav"}]
            }
            (root / "manifest.json").write_text(
                json.dumps(old_manifest),
                encoding="utf-8",
            )
            (stage / "sigh_01.wav").write_bytes(b"new")
            (stage / "laugh_01.wav").write_bytes(b"new laugh")

            def fail_after_first(source, target) -> None:
                if Path(source).name == "sigh_01.wav":
                    Path(source).replace(target)
                    return
                raise OSError("simulated interruption")

            with self.assertRaises(OSError):
                MODULE.publish_action_bank(
                    root,
                    stage,
                    {
                        "entries": [
                            {"name": "sigh", "audio": "sigh_01.wav"},
                            {"name": "laugh", "audio": "laugh_01.wav"},
                        ]
                    },
                    replace_file=fail_after_first,
                )

            current = json.loads((root / "manifest.json").read_text())
            referenced = [root / entry["audio"] for entry in current["entries"]]
            self.assertEqual(current, old_manifest)
            self.assertTrue(all(path.is_file() for path in referenced))


if __name__ == "__main__":
    unittest.main()
