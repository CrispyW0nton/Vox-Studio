from __future__ import annotations

import sys
import unittest

from pathlib import Path


SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "voxcpm_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from vox_delivery import (  # noqa: E402
    apply_pronunciations,
    delivery_distance,
    detect_delivery,
    load_pronunciations,
)


class DeliveryDetectionTests(unittest.TestCase):
    def test_detects_calm_without_inventing_emotion(self) -> None:
        reading = detect_delivery(
            {
                "characters_per_second": 14.0,
                "pitch_range_semitones": 4.0,
                "pitch_variation_semitones": 1.5,
                "dynamic_db": 9.0,
                "pause_ratio": 0.08,
                "energy_slope_db": -1.0,
                "terminal_pitch_delta": -1.0,
            },
            "I understand.",
        )

        self.assertEqual(reading.label, "calm")
        self.assertIn("Do not add urgency", reading.instruction)

    def test_detects_urgent_and_questioning_delivery(self) -> None:
        urgent = detect_delivery(
            {
                "characters_per_second": 21.0,
                "pitch_range_semitones": 12.0,
                "pitch_variation_semitones": 4.0,
                "dynamic_db": 18.0,
                "pause_ratio": 0.03,
                "energy_slope_db": 5.0,
                "terminal_pitch_delta": 1.0,
            },
            "Get to the ship now!",
        )
        questioning = detect_delivery(
            {
                "characters_per_second": 15.0,
                "pitch_range_semitones": 7.0,
                "pitch_variation_semitones": 2.5,
                "dynamic_db": 12.0,
                "pause_ratio": 0.08,
                "energy_slope_db": 0.0,
                "terminal_pitch_delta": 4.0,
            },
            "Are you certain",
        )

        self.assertEqual(urgent.label, "urgent")
        self.assertEqual(questioning.label, "questioning")

    def test_detects_sarcasm_without_marking_sincere_praise(self) -> None:
        features = {
            "characters_per_second": 15.0,
            "pitch_range_semitones": 6.0,
            "pitch_variation_semitones": 2.2,
            "dynamic_db": 13.0,
            "pause_ratio": 0.08,
            "energy_slope_db": 0.0,
            "terminal_pitch_delta": -1.0,
        }

        sarcastic = detect_delivery(
            features,
            "Nice outfit. You really expect me to believe that?",
        )
        sincere = detect_delivery(features, "It is nice to see you again.")

        self.assertEqual(sarcastic.label, "sarcastic")
        self.assertNotEqual(sincere.label, "sarcastic")

    def test_anchor_distance_penalizes_wrong_delivery(self) -> None:
        calm_features = {
            "characters_per_second": 14.0,
            "pitch_range_semitones": 4.0,
            "pitch_variation_semitones": 1.5,
            "dynamic_db": 9.0,
            "pause_ratio": 0.08,
            "energy_slope_db": -1.0,
            "terminal_pitch_delta": -1.0,
            "pitch_slope_semitones": -1.0,
        }
        urgent_features = {
            **calm_features,
            "characters_per_second": 21.0,
            "pitch_range_semitones": 12.0,
            "pitch_variation_semitones": 4.0,
            "dynamic_db": 18.0,
            "pause_ratio": 0.03,
            "energy_slope_db": 5.0,
        }
        calm = detect_delivery(calm_features, "I understand.")
        urgent = detect_delivery(urgent_features, "Move now!")

        matching = delivery_distance(
            calm_features,
            calm_features,
            calm,
            calm,
            "I understand.",
            "I understand.",
        )
        mismatched = delivery_distance(
            calm_features,
            urgent_features,
            calm,
            urgent,
            "I understand.",
            "Move now!",
        )

        self.assertLess(matching, mismatched)


class PronunciationTests(unittest.TestCase):
    def test_applies_longest_game_terms_without_changing_display_text(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")
        original = "Bao-Dur met Carth on Telos before leaving for Nar Shaddaa."

        result = apply_pronunciations(original, entries)

        self.assertEqual(
            result.text,
            "Bow-dur met Karth on Tee-lohs before leaving for Nar shuh-dah.",
        )
        self.assertEqual(
            result.matched_terms,
            ("Nar Shaddaa", "Bao-Dur", "Telos", "Carth"),
        )
        self.assertEqual(
            original,
            "Bao-Dur met Carth on Telos before leaving for Nar Shaddaa.",
        )

    def test_does_not_replace_inside_unrelated_words(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")
        result = apply_pronunciations("The carton is open.", entries)

        self.assertEqual(result.text, "The carton is open.")
        self.assertEqual(result.matched_terms, ())


if __name__ == "__main__":
    unittest.main()
