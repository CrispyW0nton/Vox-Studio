#!/usr/bin/env python3
"""Tests for KotOR dialogue wrapper detection and PCM validation."""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
import tempfile
import unittest
import wave

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "tools" / "scripts" / "build_kotor_rvc_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_kotor_rvc_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class KotorAudioDatasetTests(unittest.TestCase):
    def test_unwraps_mp3_hidden_behind_fake_wav_header(self) -> None:
        header = b"RIFF" + struct.pack("<I", 50) + bytes(50)
        payload = b"\xff\xfb\x38\xc4mp3-data"

        result = MODULE.unwrap_kotor_audio(header + payload)

        self.assertEqual("mp3-in-wav-58", result.wrapper)
        self.assertEqual("mp3", result.input_format)
        self.assertEqual(payload, result.data)

    def test_unwraps_sfx_header(self) -> None:
        payload = b"RIFF" + bytes(64)
        wrapped = MODULE.SFX_MAGIC + bytes(MODULE.SFX_HEADER_SIZE - 4) + payload

        result = MODULE.unwrap_kotor_audio(wrapped)

        self.assertEqual("sfx-470", result.wrapper)
        self.assertEqual("wav", result.input_format)
        self.assertEqual(payload, result.data)

    def test_preserves_standard_wav(self) -> None:
        payload = b"RIFF" + struct.pack("<I", 128) + b"WAVE" + bytes(64)

        result = MODULE.unwrap_kotor_audio(payload)

        self.assertEqual("standard-wav", result.wrapper)
        self.assertEqual(payload, result.data)

    def test_inspects_mono_pcm_statistics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voxstudio_kotor_audio_") as temp:
            path = Path(temp) / "speech.wav"
            sample_rate = 16000
            samples = [
                int(0.25 * 32767 * math.sin(2 * math.pi * 220 * i / sample_rate))
                for i in range(sample_rate)
            ]
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(sample_rate)
                writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))

            stats = MODULE.inspect_pcm_wav(path)

            self.assertAlmostEqual(1.0, stats.duration_seconds, places=3)
            self.assertAlmostEqual(0.25, stats.peak, places=2)
            self.assertGreater(stats.rms, 0.15)
            self.assertLess(stats.zero_crossing_rate, 0.05)

    def test_selection_excludes_matching_dialogue_names(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voxstudio_kotor_selection_") as temp:
            directory = Path(temp)
            (directory / "nm01aacart01000_.wav").touch()
            (directory / "nm40aacart05000_.wav").touch()

            selected, excluded = MODULE.selection_names(directory, ["nm01aacart*.wav"])

            self.assertEqual(["nm40aacart05000_.wav"], selected)
            self.assertEqual(["nm01aacart01000_.wav"], excluded)


if __name__ == "__main__":
    unittest.main()
