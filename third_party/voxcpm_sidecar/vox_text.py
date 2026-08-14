from __future__ import annotations

import hashlib
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
    "hook": 0.22,
    "setup": 0.12,
    "escalation": 0.10,
    "turn": 0.18,
    "climax": 0.08,
    "payoff": 0.28,
    "resolution": 0.24,
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
class VocalAction:
    name: str
    display: str
    synthesis_text: str
    direction: str
    pause_after: float


@dataclass(frozen=True)
class PerformanceScriptSegment:
    text: str
    action: VocalAction | None = None


_VOCAL_ACTIONS = (
    (
        VocalAction(
            name="death",
            display="Death gasp",
            synthesis_text="Ah... ngh...",
            direction=(
                "Perform one brief, believable in-character death gasp: a startled "
                "vocal catch that weakens into a final breath. It is a nonverbal "
                "human sound, not spoken dialogue."
            ),
            pause_after=0.42,
        ),
        (r"\bdies?\b", r"\bdying\b", r"\bdeath(?:\s+gasp|\s+sound)?\b"),
    ),
    (
        VocalAction(
            name="throat_clear",
            display="Clears throat",
            synthesis_text="Ahem.",
            direction=(
                "Clear the throat once, naturally and briefly, in the character's "
                "voice. It is a nonverbal vocal action, not a spoken word."
            ),
            pause_after=0.20,
        ),
        (r"\bclear(?:s|ing)?(?:\s+(?:his|her|their|the))?\s+throat\b",),
    ),
    (
        VocalAction(
            name="chuckle",
            display="Chuckles",
            synthesis_text="Heh... heh.",
            direction=(
                "Give one quiet, natural in-character chuckle with restrained breath "
                "and amusement. Do not turn it into spoken dialogue."
            ),
            pause_after=0.22,
        ),
        (r"\bchuckl(?:e|es|ed|ing)\b", r"\bsnicker(?:s|ed|ing)?\b"),
    ),
    (
        VocalAction(
            name="laugh",
            display="Laughs",
            synthesis_text="Ha, ha!",
            direction=(
                "Give one short, believable in-character laugh. Match the character's "
                "natural vocal placement and do not say the name of the action."
            ),
            pause_after=0.24,
        ),
        (r"\blaugh(?:s|ed|ing)?\b", r"\bgiggl(?:e|es|ed|ing)\b"),
    ),
    (
        VocalAction(
            name="sigh",
            display="Sighs",
            synthesis_text="Haaah...",
            direction=(
                "Produce one natural audible in-character sigh: a voiced breath that "
                "releases and fades. Keep it subtle unless the surrounding performance "
                "is intense."
            ),
            pause_after=0.28,
        ),
        (r"\bsigh(?:s|ed|ing)?\b",),
    ),
    (
        VocalAction(
            name="gasp",
            display="Gasps",
            synthesis_text="Ah!",
            direction=(
                "Produce one brief involuntary in-character gasp with a sharp intake "
                "of breath. It must sound like a reaction, not a spoken word."
            ),
            pause_after=0.16,
        ),
        (r"\bgasp(?:s|ed|ing)?\b",),
    ),
    (
        VocalAction(
            name="cough",
            display="Coughs",
            synthesis_text="Ahem. Ahem.",
            direction=(
                "Cough naturally once or twice in the character's voice. Keep it brief "
                "and nonverbal; do not announce the action."
            ),
            pause_after=0.22,
        ),
        (r"\bcough(?:s|ed|ing)?\b",),
    ),
    (
        VocalAction(
            name="sob",
            display="Sobs",
            synthesis_text="Hh... hh...",
            direction=(
                "Give one restrained, believable in-character sob with an unsteady "
                "breath. It is a nonverbal emotional sound, not dialogue."
            ),
            pause_after=0.30,
        ),
        (
            r"\bsob(?:s|bed|bing)?\b",
            r"\b(?:cry|cries|cried|crying)\b(?!\s+out\s+in\s+pain)",
        ),
    ),
    (
        VocalAction(
            name="whimper",
            display="Whimpers",
            synthesis_text="Nnh...",
            direction=(
                "Make one soft, involuntary in-character whimper. Keep it human, "
                "brief, and entirely nonverbal."
            ),
            pause_after=0.22,
        ),
        (r"\bwhimper(?:s|ed|ing)?\b",),
    ),
    (
        VocalAction(
            name="groan",
            display="Groans",
            synthesis_text="Nngh...",
            direction=(
                "Make one natural in-character groan that carries the surrounding "
                "emotion. Do not pronounce or announce the stage direction."
            ),
            pause_after=0.24,
        ),
        (r"\bgroan(?:s|ed|ing)?\b", r"\bmoan(?:s|ed|ing)?\b"),
    ),
    (
        VocalAction(
            name="grunt",
            display="Grunts",
            synthesis_text="Hnh.",
            direction=(
                "Give one short, natural in-character grunt as a nonverbal reaction. "
                "Do not add dialogue."
            ),
            pause_after=0.14,
        ),
        (r"\bgrunt(?:s|ed|ing)?\b",),
    ),
    (
        VocalAction(
            name="effort",
            display="Effort sound",
            synthesis_text="Hnh!",
            direction=(
                "Give one natural in-character exertion sound matching the "
                "requested physical effort and intensity. Do not add words."
            ),
            pause_after=0.12,
        ),
        (
            r"\beffort(?:\s+sound)?\b",
            r"\bexert(?:s|ed|ing|ion)?\b",
            r"\bstrain(?:s|ed|ing)?\b",
            r"\bheav(?:e|es|ed|ing)\b",
        ),
    ),
    (
        VocalAction(
            name="pain",
            display="Pain reaction",
            synthesis_text="Ah!",
            direction=(
                "Give one involuntary in-character pain reaction with intensity "
                "appropriate to the direction. Keep it brief and nonverbal."
            ),
            pause_after=0.18,
        ),
        (
            r"\bpain(?:ful|ed)?\s+(?:reaction|cry|sound)\b",
            r"\b(?:cry|cries|cried|crying)\s+out\s+in\s+pain\b",
            r"\breacts?\s+in\s+pain\b",
            r"\btakes?\s+a\s+hit\b",
        ),
    ),
    (
        VocalAction(
            name="scream",
            display="Screams",
            synthesis_text="Aah!",
            direction=(
                "Perform one brief in-character cry or scream appropriate to the "
                "moment. Preserve the character identity and do not say the action."
            ),
            pause_after=0.24,
        ),
        (
            r"\bscream(?:s|ed|ing)?\b",
            r"\bshriek(?:s|ed|ing)?\b",
            r"\byelp(?:s|ed|ing)?\b",
        ),
    ),
    (
        VocalAction(
            name="choke",
            display="Chokes",
            synthesis_text="Kh... ngh!",
            direction=(
                "Make one brief, believable in-character choking vocalization with "
                "a caught breath. Keep it nonverbal and do not add words."
            ),
            pause_after=0.24,
        ),
        (r"\bchok(?:e|es|ed|ing)\b",),
    ),
    (
        VocalAction(
            name="inhale",
            display="Inhales",
            synthesis_text="Hh...",
            direction=(
                "Take one clearly audible, natural in-character breath in. Do not "
                "speak or name the action."
            ),
            pause_after=0.10,
        ),
        (r"\binhal(?:e|es|ed|ing)\b", r"\bbreathes?\s+in\b"),
    ),
    (
        VocalAction(
            name="exhale",
            display="Exhales",
            synthesis_text="Haaah...",
            direction=(
                "Release one clearly audible, natural in-character breath out. Do not "
                "speak or name the action."
            ),
            pause_after=0.18,
        ),
        (r"\bexhal(?:e|es|ed|ing)\b", r"\bbreathes?\s+out\b"),
    ),
)

_STAGE_DIRECTION_PATTERN = re.compile(
    r"(?<!\*)(?:\\)?(?P<marker>\*{1,2})\s*(?P<body>[^*\r\n]{1,80}?)"
    r"\s*(?:\\)?(?P=marker)(?!\*)"
)


def vocal_action_for_cue(cue: str) -> VocalAction | None:
    normalized = re.sub(r"[^a-z\s'-]", " ", cue.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for action, patterns in _VOCAL_ACTIONS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return action
    return None


def vocal_action_by_name(name: str) -> VocalAction | None:
    normalized = re.sub(r"[^a-z_]", "", name.casefold())
    return next(
        (action for action, _ in _VOCAL_ACTIONS if action.name == normalized),
        None,
    )


def vocal_action_instruction(action: VocalAction) -> str:
    return " ".join(
        (
            action.direction,
            "Render only this vocal action once.",
            "Do not say its label, describe it, add words, or repeat it.",
        )
    )


def parse_performance_script(text: str) -> tuple[PerformanceScriptSegment, ...]:
    segments: list[PerformanceScriptSegment] = []
    pending = ""
    cursor = 0

    def flush_pending() -> None:
        nonlocal pending
        normalized = re.sub(r"\s+", " ", pending).strip()
        if normalized:
            segments.append(PerformanceScriptSegment(text=normalized))
        pending = ""

    for match in _STAGE_DIRECTION_PATTERN.finditer(text):
        pending += text[cursor : match.start()]
        cue = match.group("body").strip()
        action = vocal_action_for_cue(cue)
        if action is None:
            pending += cue
        else:
            flush_pending()
            segments.append(PerformanceScriptSegment(text=cue, action=action))
        cursor = match.end()

    pending += text[cursor:]
    flush_pending()
    return tuple(segments)


@dataclass(frozen=True)
class StoryBeat:
    index: int
    role: str
    text: str
    delivery: str
    direction: str
    emphasis: tuple[str, ...]
    pause_after: float
    action: str = ""
    action_cue: str = ""


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


def synthesis_seed(index: int, text: str, base_seed: int = 42) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    text_value = int.from_bytes(digest, byteorder="little", signed=False)
    return (base_seed + (max(0, index) * 104729) + text_value) % (2**31)


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
    if beat.action:
        action = vocal_action_by_name(beat.action)
        return vocal_action_instruction(action) if action is not None else beat.direction

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
            "Keep the character identity and vocal placement consistent. Vary phrase "
            "length, pitch movement, and energy with the thought instead of falling "
            "into a repeated cadence. Use brief human clause breaths, but do not pause "
            "at every comma. Complete this thought before moving on. Do not add words, "
            "announce punctuation, or over-act.",
        )
    ).strip()


def plan_story_performance(
    text: str,
    overall_delivery: str = "natural",
) -> StoryPlan:
    script_items: list[str | PerformanceScriptSegment] = []
    for segment in parse_performance_script(text):
        if segment.action is not None:
            script_items.append(segment)
            continue
        script_items.extend(_story_thoughts(segment.text))

    count = sum(isinstance(item, str) for item in script_items)
    beats: list[StoryBeat] = []
    speech_index = 0
    for item in script_items:
        if isinstance(item, PerformanceScriptSegment):
            action = item.action
            if action is None:
                continue
            beats.append(
                StoryBeat(
                    index=len(beats) + 1,
                    role="action",
                    text=f"*{action.display}*",
                    delivery=normalized_delivery_tag(overall_delivery),
                    direction=action.direction,
                    emphasis=(),
                    pause_after=action.pause_after,
                    action=action.name,
                    action_cue=item.text,
                )
            )
            continue

        thought = item
        role = _story_role(speech_index, count, thought)
        delivery = _story_delivery(role, thought, overall_delivery, count)
        emphasis = _operative_words(thought)
        beats.append(
            StoryBeat(
                index=len(beats) + 1,
                role=role,
                text=thought,
                delivery=delivery,
                direction=_story_direction(role, thought),
                emphasis=emphasis,
                pause_after=_ROLE_PAUSES[role],
            )
        )
        speech_index += 1

    arc = " -> ".join(dict.fromkeys(beat.role for beat in beats))
    action_count = sum(bool(beat.action) for beat in beats)
    summary = (
        f"{len(beats)} directed beat{'s' if len(beats) != 1 else ''} for one listener"
        + (
            f", including {action_count} vocal action"
            f"{'s' if action_count != 1 else ''}"
            if action_count
            else ""
        )
        + (f": {arc}." if arc else ".")
    )
    return StoryPlan(
        mode=STORYTELLING_MODE,
        summary=summary,
        beats=tuple(beats),
    )
