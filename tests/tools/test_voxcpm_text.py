from __future__ import annotations

import sys
import unittest
from pathlib import Path


SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "voxcpm_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from vox_text import (  # noqa: E402
    delivery_instruction,
    normalized_delivery_tag,
    split_text_for_synthesis,
    supported_delivery_tags,
)


class DeliveryTagTests(unittest.TestCase):
    def test_supports_full_requested_emotional_spectrum(self) -> None:
        tags = supported_delivery_tags()

        self.assertIn("natural", tags)
        self.assertIn("reflective", tags)
        self.assertIn("wry", tags)
        self.assertIn("wounded", tags)
        self.assertIn("sarcastic", tags)

    def test_unknown_delivery_falls_back_to_natural(self) -> None:
        self.assertEqual(normalized_delivery_tag("not-a-tag"), "natural")
        self.assertIn("Natural", delivery_instruction("not-a-tag"))


class TextSegmentationTests(unittest.TestCase):
    def test_packs_complete_sentences_into_bounded_sections(self) -> None:
        sections = split_text_for_synthesis(
            "The first sentence is calm. The second sentence asks a question? "
            "The final sentence resolves the thought.",
            maximum_characters=50,
        )

        self.assertEqual(
            sections,
            (
                "The first sentence is calm.",
                "The second sentence asks a question?",
                "The final sentence resolves the thought.",
            ),
        )

    def test_splits_a_long_sentence_without_losing_words(self) -> None:
        text = (
            "This deliberately long sentence keeps moving through several ideas, "
            "while preserving every written word for exact local synthesis."
        )

        sections = split_text_for_synthesis(text, maximum_characters=60)

        self.assertGreater(len(sections), 1)
        self.assertEqual(" ".join(sections), text)
        self.assertTrue(all(len(section) <= 60 for section in sections))


if __name__ == "__main__":
    unittest.main()
