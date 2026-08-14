from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "prepare_kotor_character_corpus.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_kotor_character_corpus",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PerformanceDirectionTests(unittest.TestCase):
    def test_preserves_directing_notes_separately_from_spoken_text(self) -> None:
        raw = (
            "{Sighs, faint wistfulness, like for a lost son}"
            "Where Revan wanders now, I do not know."
        )

        self.assertEqual(
            MODULE.clean_dialogue_text(raw),
            "Where Revan wanders now, I do not know.",
        )
        self.assertEqual(
            MODULE.performance_directions(raw),
            ("Sighs, faint wistfulness, like for a lost son",),
        )

    def test_maps_kreia_direction_to_emotional_anchor_tags(self) -> None:
        tags = MODULE.delivery_tags(
            ("Bitter, a little pissed, but with controlled restraint",)
        )

        self.assertIn("wounded", tags)
        self.assertIn("urgent", tags)

    def test_maps_dry_and_wistful_directions_without_flattening_them(self) -> None:
        self.assertEqual(
            MODULE.delivery_tags(("Rueful, dry amusement",)),
            ("wry",),
        )
        self.assertEqual(
            MODULE.delivery_tags(("Sighs, faint wistfulness",)),
            ("reflective",),
        )

    def test_rejects_explicit_other_speakers_but_accepts_default_speaker(self) -> None:
        self.assertTrue(MODULE.speaker_matches_character("", "kreia"))
        self.assertTrue(MODULE.speaker_matches_character("Kreia", "kreia"))
        self.assertFalse(MODULE.speaker_matches_character("Visas", "kreia"))


class CorpusPublicationTests(unittest.TestCase):
    @staticmethod
    def write_valid_wave(path: Path) -> None:
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(48000)
            audio.writeframes(b"\x00\x00" * 480)

    def test_empty_scan_preserves_existing_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            existing = output / "existing.wav"
            self.write_valid_wave(existing)
            old_manifest = {"entries": {"existing.wav": {"text": "Old"}}}
            (output / "dialogue_manifest.json").write_text(
                json.dumps(old_manifest),
                encoding="utf-8",
            )

            with mock.patch.object(MODULE, "read_dialogue_records", return_value={}):
                with self.assertRaises(RuntimeError):
                    MODULE.prepare_corpus(Path("game"), "kreia", output, "ffmpeg")

            self.assertTrue(existing.is_file())
            self.assertEqual(
                json.loads((output / "dialogue_manifest.json").read_text()),
                old_manifest,
            )

    def test_rejects_partial_wave_left_by_interrupted_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "partial.wav"
            path.write_bytes(b"RIFF")

            self.assertFalse(MODULE.is_valid_corpus_wave(path))


if __name__ == "__main__":
    unittest.main()
