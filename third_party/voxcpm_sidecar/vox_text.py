from __future__ import annotations

import re
from dataclasses import dataclass


STANDARD_MODE = "standard"
STORYTELLING_MODE = "storytelling"


DELIVERY_INSTRUCTIONS = {
    "natural": "Natural, conversational delivery with believable variation.",
    "calm": "Calm, even delivery with relaxed pacing and restrained intensity.",
    "measured": "Measured, deliberate delivery with clear phrasing.",
    "reflective": "Warm, reflective restraint with thoughtful pauses.",
    "warm": "Earnest, understated warmth without becoming sentimental.",
    "wry": "Dry, restrained wit with subtle skepticism, never broad comedy.",
    "guarded": "Guarded suspicion with clipped restraint and controlled tension.",
    "wounded": "Contained hurt and vulnerability without melodrama.",
    "resolute": "Steady resolve, focus, and controlled confidence.",
    "urgent": "Genuine urgency with forward momentum, without shouting.",
    "questioning": "Curious, searching delivery with a natural questioning cadence.",
    "sarcastic": "Dry, controlled sarcasm with precise emphasis, never cartoonish.",
}

_STORY_ROLE_DIRECTIONS = {
    "hook": (
        "Invite the listener into a lived memory. Use an unhurried opening, "
        "and let the specific image create the interest."
    ),
    "setup": (
        "Paint the scene clearly at an easy conversational pace. Let each new "
        "image arrive before moving to the next thought."
    ),
    "escalation": (
        "Tighten the focus and build momentum gradually. Keep the words clear "
        "and do not rush ahead to the climax."
    ),
    "turn": (
        "Recognize the change in circumstances, take a real thought beat, and "
        "enjoy the reveal without overplaying it."
    ),
    "climax": (
        "Quicken slightly through the immediate action, but keep the thought "
        "clear and every word intelligible."
    ),
    "payoff": (
        "Ease back and land the final image as the natural payoff. Leave room "
        "for it to register without announcing the joke."
    ),
    "resolution": (
        "Slow enough for the final thought to settle with honest simplicity. "
        "Preserve the story's established feeling instead of manufacturing a "
        "punchline."
    ),
}

_ROLE_PAUSES = {
    "hook": 0.30,
    "setup": 0.24,
    "escalation": 0.22,
    "turn": 0.28,
    "climax": 0.18,
    "payoff": 0.40,
    "resolution": 0.34,
}

_OPERATIVE_STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "another",
    "because",
    "before",
    "being",
    "could",
    "does",
    "enough",
    "even",
    "every",
    "from",
    "going",
    "have",
    "into",
    "just",
    "little",
    "more",
    "most",
    "much",
    "only",
    "other",
    "over",
    "really",
    "should",
    "some",
    "that",
    "their",
    "there",
    "these",
    "they",
    "themselves",
    "this",
    "those",
    "through",
    "under",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "wins",
    "would",
    "your",
}

_VIVID_WORDS = {
    "blaster",
    "credits",
    "erupted",
    "finished",
    "lightspeed",
    "mandible",
    "pazaak",
    "rodian",
    "sabacc",
    "screaming",
    "shattered",
    "smiled",
}


@dataclass(frozen=True)
class StoryBeat:
    index: int
    role: str
    text: str
    delivery: str
    direction: str
    emphasis: tuple[str, ...]
    pause_after: float


@dataclass(frozen=True)
class StoryPlan:
    mode: str
    summary: str
    beats: tuple[StoryBeat, ...]


def supported_delivery_tags() -> tuple[str, ...]:
    return tuple(DELIVERY_INSTRUCTIONS)


def normalized_delivery_tag(value: str) -> str:
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    return normalized if normalized in DELIVERY_INSTRUCTIONS else "natural"


def delivery_instruction(value: str) -> str:
    return DELIVERY_INSTRUCTIONS[normalized_delivery_tag(value)]


def normalized_performance_mode(value: str) -> str:
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    return STORYTELLING_MODE if normalized == STORYTELLING_MODE else STANDARD_MODE


def _split_oversized_section(section: str, maximum_characters: int) -> list[str]:
    chunks: list[str] = []
    remaining = section.strip()
    minimum_boundary = max(1, int(maximum_characters * 0.45))
    while len(remaining) > maximum_characters:
        window = remaining[: maximum_characters + 1]
        candidates = [
            match.end()
            for match in re.finditer(r"[,;:]\s+|\s+", window)
            if match.end() >= minimum_boundary
        ]
        boundary = candidates[-1] if candidates else maximum_characters
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def split_text_for_synthesis(
    text: str,
    maximum_characters: int = 260,
) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ()
    if maximum_characters < 40:
        raise ValueError("maximum_characters must be at least 40")

    sentences = [
        match.group(0).strip()
        for match in re.finditer(r".+?(?:[.!?]+(?=\s|$)|$)", normalized)
        if match.group(0).strip()
    ]
    sections: list[str] = []
    pending = ""
    for sentence in sentences:
        sentence_parts = _split_oversized_section(sentence, maximum_characters)
        for part in sentence_parts:
            combined = f"{pending} {part}".strip()
            if pending and len(combined) > maximum_characters:
                sections.append(pending)
                pending = part
            else:
                pending = combined
    if pending:
        sections.append(pending)
    return tuple(sections)


def _story_thoughts(
    text: str,
    maximum_characters: int = 280,
) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ()

    sentences = [
        match.group(0).strip()
        for match in re.finditer(r".+?(?:[.!?]+(?=\s|$)|$)", normalized)
        if match.group(0).strip()
    ]
    thoughts: list[str] = []
    for sentence in sentences:
        remaining = sentence
        sentence_limit = (
            min(maximum_characters, 220)
            if len(sentences) == 1 and not re.search(r"[.!?]", sentence)
            else maximum_characters
        )
        while len(remaining) > sentence_limit:
            window = remaining[: sentence_limit + 1]
            preferred = [
                (match.start(), match.end())
                for match in re.finditer(
                    r"[,;:]\s+(?=(?:and|as|before|but|except|he|i|she|"
                    r"they|because|that|thinking|until|well|when|while)\b)"
                    r"|\s+(?=and\s+when\b)"
                    r"|\s+(?=because\b)",
                    window,
                    flags=re.IGNORECASE,
                )
                if match.end() >= 70
                and not remaining[: match.start()].casefold().endswith("mostly")
            ]
            secondary = (
                []
                if preferred
                else [
                    (match.start(), match.end())
                    for match in re.finditer(r"[,;:]\s+", window)
                    if match.end() >= 85
                ]
            )
            fallback = (
                []
                if preferred or secondary
                else [
                    (match.start(), match.end())
                    for match in re.finditer(r"\s+", window)
                    if match.end() >= 120
                ]
            )
            candidates = preferred or secondary or fallback
            if candidates:
                finishing = [
                    end
                    for _, end in candidates
                    if len(remaining) - end <= sentence_limit
                ]
                boundary = finishing[0] if finishing else candidates[-1][1]
            else:
                next_boundary = re.search(r"\s+", remaining[sentence_limit:])
                if next_boundary is None:
                    thoughts.append(remaining)
                    remaining = ""
                    break
                boundary = sentence_limit + next_boundary.end()
            thoughts.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            thoughts.append(remaining)

    return tuple(thoughts)


def _story_direction(role: str, text: str) -> str:
    lowered = text.casefold()
    if role == "payoff" and re.search(
        r"\b(reached|drew|fired|erupted|exploded|shattered|screaming|"
        r"attack|charged)\b",
        lowered,
    ):
        return (
            "Carry conversational momentum through the action, then ease into "
            "the final understated clause and land it without announcing the joke."
        )
    return _STORY_ROLE_DIRECTIONS[role]


def _story_role(index: int, count: int, text: str) -> str:
    lowered = text.casefold()
    if count == 1 or index == count - 1:
        has_understated_payoff = bool(
            re.search(
                r"\b(apparently|naturally|of course|before (?:i|we|they) even|"
                r"without so much as|as if nothing|ridiculous|absurd)\b",
                lowered,
            )
        )
        return "payoff" if has_understated_payoff else "resolution"
    if index == 0:
        return "hook"

    position = index / max(1, count - 1)
    if position >= 0.58 and re.search(
        r"\b(reached|drew|fired|erupted|exploded|shattered|screaming|"
        r"attack|charged)\b",
        lowered,
    ):
        return "climax"
    if re.search(r"^(well|but|except|until|then)\b", lowered):
        return "turn"
    if position < 0.28:
        return "setup"
    if position < 0.62:
        return "escalation"
    return "turn"


def _story_delivery(role: str, text: str, overall: str, count: int) -> str:
    base = normalized_delivery_tag(overall)
    if count == 1:
        return base

    lowered = text.casefold()
    has_dry_humor = bool(
        re.search(
            r"\b(absurd|apparently|cheap|gullible|laughed|lovely|"
            r"loudmouth|ridiculous|smiled|smirking|surely)\b",
            lowered,
        )
    )
    if role == "climax":
        return "urgent"
    if role == "payoff":
        return "sarcastic" if base == "sarcastic" else "wry"
    if role == "resolution":
        return "natural" if base in {"urgent", "sarcastic"} else base
    if role == "turn":
        if has_dry_humor or base in {"wry", "sarcastic"}:
            return "wry"
        return "natural" if base == "natural" else base
    if role == "escalation":
        return "wry" if has_dry_humor else base
    if role == "hook":
        return "wry" if has_dry_humor or base in {"wry", "sarcastic"} else "reflective"
    if role == "setup" and base in {"urgent", "sarcastic"}:
        return "natural"
    if base not in {"natural", "urgent", "sarcastic"}:
        return base
    return "natural"


def _operative_words(text: str, maximum_words: int = 3) -> tuple[str, ...]:
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for order, match in enumerate(
        re.finditer(r"[A-Za-z0-9]+(?:[+'/-][A-Za-z0-9]+)*", text)
    ):
        word = match.group(0)
        normalized = word.casefold().strip("'+/-")
        transition_suffix = re.search(
            r"-(?:but|except|mostly|then)$",
            normalized,
        )
        if transition_suffix is not None:
            normalized = normalized[: transition_suffix.start()]
            word = word[: transition_suffix.start()]
        if (
            len(normalized) < 4
            or normalized in _OPERATIVE_STOP_WORDS
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        score = len(normalized)
        if normalized in _VIVID_WORDS:
            score += 20
        if word[:1].isupper() and order > 0:
            score += 5
        candidates.append((score, -order, word))
    candidates.sort(reverse=True)
    selected = sorted(
        candidates[:maximum_words],
        key=lambda candidate: -candidate[1],
    )
    return tuple(candidate[2] for candidate in selected)


def story_beat_instruction(beat: StoryBeat, overall: str = "natural") -> str:
    emphasis = ", ".join(beat.emphasis)
    baseline = normalized_delivery_tag(overall)
    continuity = (
        "Begin directly, as if the listener is already with you."
        if beat.index == 1
        else (
            "Keep this as one continuous conversation. Carry the rhythm and "
            "intention forward instead of restarting the performance."
        )
    )
    return " ".join(
        (
            "Tell this as a lived event to one specific listener.",
            continuity,
            delivery_instruction(beat.delivery),
            beat.direction,
            f"Give clean, natural emphasis to: {emphasis}." if emphasis else "",
            (
                f"Let {baseline} be the overall color, but follow the changing "
                "thought rather than forcing one emotion across the story."
            ),
            "Keep the character identity and vocal placement consistent. "
            "Let the pace move naturally between roughly 145 and 165 words per "
            "minute: ease around reveals and move a little faster through action. "
            "Use brief human clause breaths, but do not pause at every comma. "
            "Complete this thought before moving on. Do not add words, announce "
            "punctuation, or over-act.",
        )
    ).strip()


def plan_story_performance(
    text: str,
    overall_delivery: str = "natural",
) -> StoryPlan:
    thoughts = _story_thoughts(text)
    count = len(thoughts)
    beats: list[StoryBeat] = []
    for index, thought in enumerate(thoughts):
        role = _story_role(index, count, thought)
        delivery = _story_delivery(role, thought, overall_delivery, count)
        emphasis = _operative_words(thought)
        beats.append(
            StoryBeat(
                index=index + 1,
                role=role,
                text=thought,
                delivery=delivery,
                direction=_story_direction(role, thought),
                emphasis=emphasis,
                pause_after=_ROLE_PAUSES[role],
            )
        )

    arc = " -> ".join(dict.fromkeys(beat.role for beat in beats))
    summary = (
        f"{count} directed beat{'s' if count != 1 else ''} for one listener"
        + (f": {arc}." if arc else ".")
    )
    return StoryPlan(
        mode=STORYTELLING_MODE,
        summary=summary,
        beats=tuple(beats),
    )
