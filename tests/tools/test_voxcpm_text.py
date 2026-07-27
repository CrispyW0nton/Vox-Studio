from __future__ import annotations

import sys
import unittest
from pathlib import Path


SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "voxcpm_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from vox_text import (  # noqa: E402
    STORYTELLING_MODE,
    delivery_instruction,
    normalized_delivery_tag,
    plan_story_performance,
    split_text_for_synthesis,
    supported_delivery_tags,
)

ATTON_STORY = (
    "So there I was, sitting in a smoke-filled spacer dive on Nar Shaddaa, "
    "trying to win enough credits for a bowl of actual warm food, when a "
    "loudmouth Rodian decided he was going to teach me a lesson in math. "
    "He kept hitting me with side-bets and smug little clicks of his mandible "
    "every time he drew a plus-two, thinking I was just another broke smuggler "
    "sweating over a twenty-side deck. I told him, nice and easy, that pazaak "
    "is all about knowing when to fold your hand and when to let the cards "
    "speak for themselves-mostly because my own deck was heavily modified "
    "with a few choice +/- cards I lifted off a gullible protocol droid on "
    "Onderon. Well, the green guy flips a natural twenty, slams his greasy "
    "claws on the table, and calls me a cheap Sabacc cheat who doesn't even "
    "know the rules of a real spacer's game. I just smiled, tapped my side-deck, "
    "and told him the house always wins-except when the house is carrying a "
    "heavily customized dual-ion blaster under the table. He reached for his "
    "sidearm, I cleared leather faster than a Republic cruiser jumping to "
    "lightspeed, and the whole cantina erupted into a lovely, glowing mess of "
    "blaster bolts, shattered glass, and screaming gamorreans before I even "
    "finished my drink."
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


class StoryPerformanceTests(unittest.TestCase):
    def test_atton_story_beats_preserve_every_word_and_build_an_arc(self) -> None:
        plan = plan_story_performance(ATTON_STORY, "natural")

        self.assertEqual(plan.mode, STORYTELLING_MODE)
        self.assertGreaterEqual(len(plan.beats), 7)
        self.assertLessEqual(len(plan.beats), 14)
        self.assertEqual(" ".join(beat.text for beat in plan.beats), ATTON_STORY)
        self.assertEqual(plan.beats[0].role, "hook")
        self.assertEqual(plan.beats[-1].role, "payoff")
        self.assertIn("escalation", {beat.role for beat in plan.beats})
        self.assertIn("climax", {beat.role for beat in plan.beats})
        self.assertIn("wry", {beat.delivery for beat in plan.beats})
        self.assertIn("urgent", {beat.delivery for beat in plan.beats})
        self.assertEqual(
            plan.beats[-1].text,
            "before I even finished my drink.",
        )
        house_reveal = next(
            beat for beat in plan.beats if "house always wins" in beat.text
        )
        self.assertEqual(house_reveal.role, "turn")
        self.assertEqual(house_reveal.delivery, "wry")

    def test_story_plan_directs_intent_emphasis_and_thought_change(self) -> None:
        plan = plan_story_performance(ATTON_STORY, "wry")

        self.assertTrue(all(beat.direction for beat in plan.beats))
        self.assertTrue(all(beat.emphasis for beat in plan.beats))
        self.assertTrue(all(0.08 <= beat.pause_after <= 0.55 for beat in plan.beats))
        self.assertIn("one listener", plan.summary)
        self.assertIn("land", plan.beats[-1].direction.casefold())
        self.assertNotEqual(plan.beats[0].direction, plan.beats[-1].direction)

    def test_short_story_remains_a_single_complete_beat(self) -> None:
        text = "I found the map, and now we know where to go."

        plan = plan_story_performance(text, "reflective")

        self.assertEqual(len(plan.beats), 1)
        self.assertEqual(plan.beats[0].text, text)
        self.assertEqual(plan.beats[0].role, "resolution")
        self.assertEqual(plan.beats[0].delivery, "reflective")

    def test_reflective_story_does_not_invent_an_action_climax(self) -> None:
        text = (
            "I remember the first winter in that little house. "
            "Morning light gathered on the old wooden floor. "
            "My mother would hum while the kettle warmed. "
            "We rarely spoke because the quiet already felt complete. "
            "Years later I returned and found the rooms empty. "
            "Even so, the light through the window felt like a welcome."
        )

        plan = plan_story_performance(text, "reflective")

        self.assertNotIn("climax", {beat.role for beat in plan.beats})
        self.assertNotIn("urgent", {beat.delivery for beat in plan.beats})
        self.assertEqual(plan.beats[-1].role, "resolution")
        self.assertEqual(plan.beats[-1].delivery, "reflective")
        self.assertTrue(all(beat.text for beat in plan.beats))
        self.assertEqual(" ".join(beat.text for beat in plan.beats), text)

    def test_wounded_ending_keeps_its_emotional_truth(self) -> None:
        text = (
            "I waited beside the road until the lanterns went dark. "
            "He had promised to return before winter. "
            "By morning I understood that the promise had died with him."
        )

        plan = plan_story_performance(text, "wounded")

        self.assertEqual(plan.beats[-1].role, "resolution")
        self.assertEqual(plan.beats[-1].delivery, "wounded")
        self.assertNotIn("wry", {beat.delivery for beat in plan.beats})

    def test_poignant_smile_does_not_turn_grief_into_a_punchline(self) -> None:
        text = (
            "She held the old photograph against the window. "
            "For one quiet moment she smiled at the life they had shared."
        )

        plan = plan_story_performance(text, "wounded")

        self.assertEqual(plan.beats[-1].role, "resolution")
        self.assertEqual(plan.beats[-1].delivery, "wounded")

    def test_punctuation_poor_story_preserves_words_and_builds_manageable_beats(
        self,
    ) -> None:
        text = (
            "we crossed the ridge before sunrise and followed the river through "
            "the valley while the fog lifted around us and every familiar tree "
            "looked strange in the pale light but nobody wanted to turn back "
            "because the village was finally visible beyond the last hill"
        )

        plan = plan_story_performance(text, "natural")

        self.assertGreaterEqual(len(plan.beats), 2)
        self.assertLessEqual(len(plan.beats), 6)
        self.assertTrue(all(beat.text for beat in plan.beats))
        self.assertEqual(" ".join(beat.text for beat in plan.beats), text)

    def test_oversized_token_is_never_cut_or_rewritten(self) -> None:
        token = "https://example.test/" + ("a" * 240)
        text = f"I found this address {token} and wrote it down."

        plan = plan_story_performance(text, "natural")

        self.assertIn(token, {word for beat in plan.beats for word in beat.text.split()})
        self.assertEqual(" ".join(beat.text for beat in plan.beats), text)


if __name__ == "__main__":
    unittest.main()
