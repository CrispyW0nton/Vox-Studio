from __future__ import annotations

import re


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


def supported_delivery_tags() -> tuple[str, ...]:
    return tuple(DELIVERY_INSTRUCTIONS)


def normalized_delivery_tag(value: str) -> str:
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    return normalized if normalized in DELIVERY_INSTRUCTIONS else "natural"


def delivery_instruction(value: str) -> str:
    return DELIVERY_INSTRUCTIONS[normalized_delivery_tag(value)]


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
