from __future__ import annotations

import sys
import unittest

from pathlib import Path


SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "voxcpm_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from vox_delivery import (  # noqa: E402
    DeliveryReading,
    apply_pronunciations,
    build_transcription_hotwords,
    canonicalize_transcription,
    delivery_distance,
    detect_delivery,
    load_pronunciations,
    specialize_delivery,
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

    def test_detects_reflective_nostalgia_without_calling_it_emphatic(self) -> None:
        reading = detect_delivery(
            {
                "characters_per_second": 15.3,
                "pitch_range_semitones": 9.0,
                "pitch_variation_semitones": 3.6,
                "pitch_slope_semitones": -4.7,
                "dynamic_db": 15.5,
                "pause_ratio": 0.08,
                "energy_slope_db": -3.0,
                "terminal_pitch_delta": -0.7,
            },
            (
                "I wish you the best of luck. I hope you find the happiness "
                "I once knew myself."
            ),
        )

        self.assertEqual(reading.label, "reflective")
        self.assertIn("without adding agitation", reading.instruction)

    def test_history_cannot_change_the_current_phrase_label(self) -> None:
        features = {
            "characters_per_second": 16.2,
            "pitch_range_semitones": 9.0,
            "pitch_variation_semitones": 3.3,
            "pitch_slope_semitones": 0.0,
            "dynamic_db": 15.0,
            "pause_ratio": 0.08,
            "energy_slope_db": 0.0,
            "terminal_pitch_delta": 0.0,
        }
        quiet_history = (
            {
                "characters_per_second": 9.0,
                "pitch_range_semitones": 3.0,
                "dynamic_db": 8.0,
            },
        ) * 8

        without_history = detect_delivery(features, "We should keep moving.")
        with_history = detect_delivery(
            features,
            "We should keep moving.",
            quiet_history,
        )

        self.assertEqual(without_history.label, "neutral")
        self.assertEqual(with_history.label, without_history.label)

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


class CharacterDeliveryTests(unittest.TestCase):
    FEATURES = {
        "characters_per_second": 15.0,
        "pitch_range_semitones": 8.0,
        "pitch_variation_semitones": 2.8,
        "pitch_slope_semitones": -1.0,
        "dynamic_db": 13.0,
        "pause_ratio": 0.08,
        "energy_slope_db": -1.0,
        "terminal_pitch_delta": -1.0,
    }

    def carth_delivery(self, text: str):
        reading = detect_delivery(self.FEATURES, text)
        return specialize_delivery("carth", reading, self.FEATURES, text)

    def test_maps_carth_sarcasm_to_dry_wry_skepticism(self) -> None:
        reading = self.carth_delivery(
            "So it's a popularity contest, basically? Wonderful."
        )

        self.assertEqual(reading.label, "wry")
        self.assertIn("rather than playful", reading.instruction)

    def test_keeps_carth_concern_distinct_from_sarcasm(self) -> None:
        reading = self.carth_delivery(
            (
                "Are you sure about this? I want to get Bastila back, "
                "but what if they find out?"
            )
        )

        self.assertEqual(reading.label, "guarded")
        self.assertIn("without turning the question into sarcasm", reading.instruction)

    def test_maps_carth_guarded_distrust(self) -> None:
        reading = specialize_delivery(
            "carth",
            DeliveryReading("reflective", "", 0.5, 0.5, 0.5),
            self.FEATURES,
            "I've seen an evasion or two in my time, enough to know when I hear it."
        )

        self.assertEqual(reading.label, "guarded")

    def test_maps_carth_contained_grief(self) -> None:
        reading = self.carth_delivery(
            "My family was destroyed that day and my wife died in the bombardment."
        )

        self.assertEqual(reading.label, "wounded")

    def test_maps_carth_understated_warmth(self) -> None:
        reading = specialize_delivery(
            "carth",
            DeliveryReading("emphatic", "", 0.5, 0.7, 0.7),
            {**self.FEATURES, "energy_slope_db": 6.8},
            "I'm proud of you. Not everyone could have done that."
        )

        self.assertEqual(reading.label, "warm")

    def test_maps_carth_moral_disapproval(self) -> None:
        reading = self.carth_delivery(
            "This isn't right. We should be helping people, not hurting them."
        )

        self.assertEqual(reading.label, "disapproving")

    def test_maps_carth_protective_resolve(self) -> None:
        reading = self.carth_delivery(
            "We have to rescue Bastila and stop them before they escape."
        )

        self.assertEqual(reading.label, "resolute")

    def test_does_not_apply_carth_palette_to_other_profiles(self) -> None:
        reading = detect_delivery(self.FEATURES, "We have to rescue Bastila.")

        specialized = specialize_delivery(
            "atton",
            reading,
            self.FEATURES,
            "We have to rescue Bastila.",
        )

        self.assertEqual(specialized, reading)


class PronunciationTests(unittest.TestCase):
    def test_builds_unique_game_vocabulary_for_transcription(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")

        hotwords = build_transcription_hotwords(
            entries,
            ("Star Wars", "Ebon Hawk", "carth"),
        )

        self.assertIn("Star Wars", hotwords)
        self.assertIn("Ebon Hawk", hotwords)
        self.assertIn("Carth", hotwords)
        self.assertIn("Bao-Dur", hotwords)
        self.assertIn("Nar Shaddaa", hotwords)
        self.assertEqual(hotwords.casefold().count("carth"), 1)
        self.assertNotIn("Karth", hotwords)
        self.assertNotIn("Bao Dur", hotwords)

    def test_canonicalizes_known_asr_variants_only(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")

        result = canonicalize_transcription(
            "Karth met Bao Dur on Tilos beside the carton.",
            entries,
        )

        self.assertEqual(
            result,
            "Carth met Bao-Dur on Telos beside the carton.",
        )

    def test_applies_longest_game_terms_without_changing_display_text(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")
        original = "Bao-Dur met Carth on Telos before leaving for Nar Shaddaa."

        result = apply_pronunciations(original, entries)

        self.assertEqual(
            result.text,
            "Bae-oh Dure met Karth on TEE-los before leaving for Narr Sha-Da.",
        )
        self.assertEqual(
            result.matched_terms,
            ("Nar Shaddaa", "Bao-Dur", "Telos", "Carth"),
        )
        self.assertEqual(
            original,
            "Bao-Dur met Carth on Telos before leaving for Nar Shaddaa.",
        )

    def test_applies_rodian_pronunciation_without_changing_display_text(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")
        original = "The Rodian returned to Telos."

        result = apply_pronunciations(original, entries)

        self.assertEqual(result.text, "The Rode-ian returned to TEE-los.")
        self.assertEqual(result.matched_terms, ("Telos", "Rodian"))
        self.assertEqual(original, "The Rodian returned to Telos.")

    def test_applies_story_game_terms_for_pazaak_scene(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")

        result = apply_pronunciations(
            "The Rodian called pazaak a Sabacc game while a Gamorrean watched.",
            entries,
        )

        self.assertIn("puh Zack", result.text)
        self.assertIn("sah back", result.text)
        self.assertIn("ga-maw-ree-uhn", result.text)
        self.assertIn("Pazaak", result.matched_terms)
        self.assertIn("Sabacc", result.matched_terms)
        self.assertIn("Gamorrean", result.matched_terms)

        pazaak = next(entry for entry in entries if entry.term == "Pazaak")
        self.assertEqual(pazaak.guide, "puh-ZACK")
        self.assertIn("Pazak", pazaak.aliases)

        sabacc = next(entry for entry in entries if entry.term == "Sabacc")
        self.assertEqual(sabacc.guide, "sah-BAK")

    def test_applies_plural_gamorrean_pronunciation(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")

        result = apply_pronunciations("Two Gamorreans entered.", entries)

        self.assertEqual(result.text, "Two ga-maw-ree-uhns entered.")
        self.assertEqual(result.matched_terms, ("Gamorreans",))

    def test_does_not_replace_inside_unrelated_words(self) -> None:
        entries = load_pronunciations(SIDECAR_ROOT / "pronunciations.json")
        result = apply_pronunciations("The carton is open.", entries)

        self.assertEqual(result.text, "The carton is open.")
        self.assertEqual(result.matched_terms, ())


if __name__ == "__main__":
    unittest.main()
