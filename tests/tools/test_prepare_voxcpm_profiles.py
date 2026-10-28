from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "prepare_voxcpm_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_voxcpm_profiles", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AdapterCorpusTests(unittest.TestCase):
    def test_carth_profile_uses_exact_calm_prompt_pair(self) -> None:
        carth = next(spec for spec in MODULE.default_specs() if spec.name == "Carth")

        self.assertEqual(carth.sources[0].name, "CarthExact")
        self.assertEqual(carth.primary_prompt.name, "nm02aacart02001_.wav")
        self.assertEqual(carth.style_anchor_count, 212)
        self.assertTrue(carth.primary_prompt_text.startswith("Don't worry"))
        self.assertIn("kolto packs", carth.primary_prompt_text)

    def test_rejects_adapter_without_training_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current = Path(temp_dir) / "CarthExact"
            current.mkdir()

            matches = MODULE.training_matches_profile_sources(
                current.parent / "missing-summary.json",
                (current,),
            )

        self.assertFalse(matches)

    def test_rejects_adapter_trained_from_a_superseded_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "CarthExact"
            old = root / "CarthDecoded"
            current.mkdir()
            summary = root / "summary.json"
            summary.write_text(
                json.dumps({"source": {"audio_root": str(old)}}),
                encoding="utf-8",
            )

            matches = MODULE.training_matches_profile_sources(summary, (current,))

        self.assertFalse(matches)

    def test_accepts_adapter_trained_from_the_current_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current = Path(temp_dir) / "CarthExact"
            current.mkdir()
            summary = current.parent / "summary.json"
            summary.write_text(
                json.dumps({"source": {"audio_root": str(current)}}),
                encoding="utf-8",
            )

            matches = MODULE.training_matches_profile_sources(summary, (current,))

        self.assertTrue(matches)

    def test_rejects_malformed_training_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current = Path(temp_dir) / "CarthExact"
            current.mkdir()
            summary = current.parent / "summary.json"
            summary.write_text("{not-json", encoding="utf-8")

            matches = MODULE.training_matches_profile_sources(summary, (current,))

        self.assertFalse(matches)

    def test_rejects_checkpoint_older_than_prepared_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "latest"
            checkpoint.mkdir()
            weights = checkpoint / "lora_weights.safetensors"
            config = checkpoint / "lora_config.json"
            summary = root / "summary.json"
            weights.write_bytes(b"old weights")
            config.write_text("{}", encoding="utf-8")
            summary.write_text("{}", encoding="utf-8")
            now = time.time_ns()
            old = now - 10_000_000_000
            os.utime(weights, ns=(old, old))
            os.utime(config, ns=(old, old))
            os.utime(summary, ns=(now, now))

            matches = MODULE.checkpoint_matches_training_summary(
                checkpoint,
                summary,
            )

        self.assertFalse(matches)

    def test_accepts_checkpoint_written_after_prepared_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "latest"
            checkpoint.mkdir()
            summary = root / "summary.json"
            summary.write_text("{}", encoding="utf-8")
            now = time.time_ns()
            os.utime(summary, ns=(now, now))
            for name in ("lora_weights.safetensors", "lora_config.json"):
                artifact = checkpoint / name
                artifact.write_bytes(b"new")
                newer = now + 10_000_000_000
                os.utime(artifact, ns=(newer, newer))

            matches = MODULE.checkpoint_matches_training_summary(
                checkpoint,
                summary,
            )

        self.assertTrue(matches)


if __name__ == "__main__":
    unittest.main()
