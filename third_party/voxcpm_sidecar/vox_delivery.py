from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DeliveryReading:
    label: str
    instruction: str
    pace: float
    expressiveness: float
    intensity: float


_TEXT_DELIVERY_SHAPES = {
    "natural": (
        "neutral",
        0.50,
        0.46,
        0.42,
        {
            "characters_per_second": 15.0,
            "pitch_range_semitones": 7.0,
            "pitch_variation_semitones": 2.6,
            "dynamic_db": 12.0,
            "pause_ratio": 0.08,
            "pitch_slope_semitones": 0.0,
            "terminal_pitch_delta": -0.5,
            "energy_slope_db": 0.0,
        },
    ),
    "calm": (
        "calm",
        0.32,
        0.28,
        0.24,
        {
            "characters_per_second": 12.8,
            "pitch_range_semitones": 5.0,
            "pitch_variation_semitones": 1.8,
            "dynamic_db": 9.5,
            "pause_ratio": 0.11,
            "pitch_slope_semitones": -0.5,
            "terminal_pitch_delta": -1.0,
            "energy_slope_db": -0.5,
        },
    ),
    "measured": (
        "measured",
        0.22,
        0.36,
        0.30,
        {
            "characters_per_second": 10.8,
            "pitch_range_semitones": 6.0,
            "pitch_variation_semitones": 2.1,
            "dynamic_db": 10.5,
            "pause_ratio": 0.21,
            "pitch_slope_semitones": -0.8,
            "terminal_pitch_delta": -1.2,
            "energy_slope_db": -0.3,
        },
    ),
    "reflective": (
        "reflective",
        0.30,
        0.42,
        0.32,
        {
            "characters_per_second": 11.8,
            "pitch_range_semitones": 6.5,
            "pitch_variation_semitones": 2.3,
            "dynamic_db": 11.0,
            "pause_ratio": 0.16,
            "pitch_slope_semitones": -2.0,
            "terminal_pitch_delta": -1.8,
            "energy_slope_db": -0.8,
        },
    ),
    "warm": (
        "calm",
        0.38,
        0.45,
        0.36,
        {
            "characters_per_second": 13.2,
            "pitch_range_semitones": 6.2,
            "pitch_variation_semitones": 2.3,
            "dynamic_db": 11.5,
            "pause_ratio": 0.10,
            "pitch_slope_semitones": 0.2,
            "terminal_pitch_delta": -0.8,
            "energy_slope_db": 0.5,
        },
    ),
    "wry": (
        "sarcastic",
        0.50,
        0.60,
        0.44,
        {
            "characters_per_second": 15.0,
            "pitch_range_semitones": 8.0,
            "pitch_variation_semitones": 3.0,
            "dynamic_db": 12.5,
            "pause_ratio": 0.09,
            "pitch_slope_semitones": -1.0,
            "terminal_pitch_delta": -2.2,
            "energy_slope_db": 0.0,
        },
    ),
    "guarded": (
        "calm",
        0.42,
        0.40,
        0.48,
        {
            "characters_per_second": 14.0,
            "pitch_range_semitones": 5.8,
            "pitch_variation_semitones": 2.0,
            "dynamic_db": 10.5,
            "pause_ratio": 0.10,
            "pitch_slope_semitones": -0.8,
            "terminal_pitch_delta": -1.5,
            "energy_slope_db": 0.4,
        },
    ),
    "wounded": (
        "reflective",
        0.26,
        0.50,
        0.40,
        {
            "characters_per_second": 11.2,
            "pitch_range_semitones": 7.0,
            "pitch_variation_semitones": 2.7,
            "dynamic_db": 12.0,
            "pause_ratio": 0.18,
            "pitch_slope_semitones": -2.2,
            "terminal_pitch_delta": -2.0,
            "energy_slope_db": -1.2,
        },
    ),
    "resolute": (
        "emphatic",
        0.58,
        0.62,
        0.62,
        {
            "characters_per_second": 16.0,
            "pitch_range_semitones": 8.5,
            "pitch_variation_semitones": 3.1,
            "dynamic_db": 14.0,
            "pause_ratio": 0.07,
            "pitch_slope_semitones": 0.6,
            "terminal_pitch_delta": -1.8,
            "energy_slope_db": 1.8,
        },
    ),
    "urgent": (
        "urgent",
        0.82,
        0.72,
        0.84,
        {
            "characters_per_second": 19.0,
            "pitch_range_semitones": 10.0,
            "pitch_variation_semitones": 3.8,
            "dynamic_db": 16.0,
            "pause_ratio": 0.04,
            "pitch_slope_semitones": 1.2,
            "terminal_pitch_delta": -0.5,
            "energy_slope_db": 3.5,
        },
    ),
    "questioning": (
        "questioning",
        0.50,
        0.58,
        0.48,
        {
            "characters_per_second": 15.0,
            "pitch_range_semitones": 8.0,
            "pitch_variation_semitones": 3.0,
            "dynamic_db": 12.5,
            "pause_ratio": 0.08,
            "pitch_slope_semitones": 1.0,
            "terminal_pitch_delta": 4.5,
            "energy_slope_db": 0.5,
        },
    ),
    "sarcastic": (
        "sarcastic",
        0.54,
        0.66,
        0.52,
        {
            "characters_per_second": 15.5,
            "pitch_range_semitones": 8.5,
            "pitch_variation_semitones": 3.3,
            "dynamic_db": 13.5,
            "pause_ratio": 0.08,
            "pitch_slope_semitones": -0.8,
            "terminal_pitch_delta": -2.5,
            "energy_slope_db": 0.8,
        },
    ),
}


def text_delivery_target(value: str) -> tuple[dict[str, float], DeliveryReading]:
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    label, pace, expressiveness, intensity, features = _TEXT_DELIVERY_SHAPES.get(
        normalized,
        _TEXT_DELIVERY_SHAPES["natural"],
    )
    return (
        dict(features),
        DeliveryReading(
            label=label,
            instruction="",
            pace=pace,
            expressiveness=expressiveness,
            intensity=intensity,
        ),
    )


@dataclass(frozen=True)
class PronunciationEntry:
    term: str
    spoken: str
    aliases: tuple[str, ...] = ()
    guide: str = ""


@dataclass(frozen=True)
class PronunciationResult:
    text: str
    matched_terms: tuple[str, ...]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _value(features: Mapping[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default))
    except (TypeError, ValueError):
        return default


def _history_median(
    history: Sequence[Mapping[str, float]],
    key: str,
    fallback: float,
) -> float:
    values = [_value(item, key, fallback) for item in history]
    return median(values) if values else fallback


def _looks_like_question(text: str, terminal_pitch_delta: float) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    if terminal_pitch_delta < 2.5:
        return False
    return bool(
        re.match(
            r"(?i)^(who|what|when|where|why|how|is|are|am|do|does|did|can|"
            r"could|would|will|should|have|has|was|were)\b",
            stripped,
        )
    )


def _sounds_sarcastic(
    text: str,
    expressiveness: float,
    terminal_pitch_delta: float,
    question: bool,
) -> bool:
    lowered = re.sub(r"\s+", " ", text.casefold()).strip()
    strong_patterns = (
        r"\b(?:yeah|oh),?\s+right\b",
        r"\bas if\b",
        r"\bwhat a surprise\b",
        r"\b(?:oh|well),?\s+(?:great|wonderful|perfect|brilliant)\b",
        r"\b(?:sure|right),?\s+(?:because|like)\b",
        r"\bnice\s+(?:outfit|try|excuse|story)\b",
        r"\byou really (?:think|expect|believe)\b",
    )
    if any(re.search(pattern, lowered) for pattern in strong_patterns):
        return True

    soft_cue = bool(
        re.search(
            r"\b(?:obviously|brilliant|wonderful|perfect|seriously|really|sure)\b",
            lowered,
        )
    )
    return soft_cue and (
        (question and expressiveness >= 0.42)
        or (terminal_pitch_delta <= -1.5 and expressiveness >= 0.48)
    )


def _sounds_reflective(
    text: str,
    pace: float,
    expressiveness: float,
    pause_ratio: float,
    pitch_slope: float,
    energy_slope: float,
    terminal_pitch_delta: float,
) -> bool:
    lowered = re.sub(r"\s+", " ", text.casefold()).strip()
    memory_cue = bool(
        re.search(
            r"\b(?:remember|used to|once|years ago|back then|in those days|"
            r"miss|wish|hope|home|family|wife|husband|son|daughter|"
            r"father|mother|happiness|the past)\b",
            lowered,
        )
    )
    restrained = (
        pace <= 0.62
        and expressiveness <= 0.72
        and energy_slope <= 1.0
        and terminal_pitch_delta <= 2.0
    )
    reflective_shape = pause_ratio >= 0.10 and pitch_slope <= -1.5
    return restrained and (memory_cue or reflective_shape)


def detect_delivery(
    features: Mapping[str, float],
    transcript: str,
    history: Sequence[Mapping[str, float]] = (),
) -> DeliveryReading:
    rate = _value(features, "characters_per_second", 15.0)
    pitch_range = _value(features, "pitch_range_semitones", 7.0)
    pitch_variation = _value(
        features,
        "pitch_variation_semitones",
        max(1.0, pitch_range / 3.0),
    )
    dynamic_range = _value(features, "dynamic_db", 12.0)
    pause_ratio = _value(features, "pause_ratio", 0.08)
    energy_slope = _value(features, "energy_slope_db", 0.0)
    pitch_slope = _value(features, "pitch_slope_semitones", 0.0)
    terminal_pitch_delta = _value(features, "terminal_pitch_delta", 0.0)

    absolute_pace = _clamp((rate - 9.0) / 12.0)
    pitch_expression = _clamp((pitch_range - 3.0) / 10.0)
    variation_expression = _clamp((pitch_variation - 1.0) / 3.5)
    dynamic_expression = _clamp((dynamic_range - 8.0) / 12.0)
    phrase_expressiveness = (
        (pitch_expression * 0.45)
        + (variation_expression * 0.30)
        + (dynamic_expression * 0.25)
    )
    phrase_pace = absolute_pace
    pace = phrase_pace
    expressiveness = phrase_expressiveness

    if len(history) >= 4:
        baseline_rate = _history_median(history, "characters_per_second", rate)
        baseline_pitch = _history_median(history, "pitch_range_semitones", pitch_range)
        baseline_dynamic = _history_median(history, "dynamic_db", dynamic_range)
        relative_pace = _clamp(0.5 + ((rate - baseline_rate) / 8.0))
        relative_expression = _clamp(
            0.5
            + ((pitch_range - baseline_pitch) / 12.0)
            + ((dynamic_range - baseline_dynamic) / 24.0)
        )
        pace = (absolute_pace * 0.72) + (relative_pace * 0.28)
        expressiveness = (expressiveness * 0.75) + (relative_expression * 0.25)

    rising_energy = _clamp((energy_slope + 1.0) / 8.0)
    intensity = _clamp(
        (phrase_expressiveness * 0.58)
        + (phrase_pace * 0.27)
        + (rising_energy * 0.15)
    )
    question = _looks_like_question(transcript, terminal_pitch_delta)
    sarcastic = _sounds_sarcastic(
        transcript,
        phrase_expressiveness,
        terminal_pitch_delta,
        question,
    )
    reflective = _sounds_reflective(
        transcript,
        phrase_pace,
        phrase_expressiveness,
        pause_ratio,
        pitch_slope,
        energy_slope,
        terminal_pitch_delta,
    )

    if sarcastic:
        label = "sarcastic"
        instruction = (
            "Preserve the performer's dry ironic emphasis and controlled contrast. "
            "Do not turn it into broad comedy, anger, or extra amusement."
        )
    elif question:
        label = "questioning"
        instruction = (
            "Preserve the performer's questioning shape and ending lift without "
            "adding surprise or urgency."
        )
    elif (
        (phrase_pace >= 0.70 and phrase_expressiveness >= 0.50)
        or (phrase_pace >= 0.58 and energy_slope >= 3.0)
    ):
        label = "urgent"
        instruction = (
            "Match the performer's quick, urgent delivery without making it louder "
            "or more emotional than the performance."
        )
    elif reflective:
        label = "reflective"
        instruction = (
            "Preserve the performer's warm, reflective restraint and sense of "
            "memory. Keep the pauses and tenderness without adding agitation or "
            "melodrama."
        )
    elif phrase_expressiveness >= 0.68:
        label = "emphatic"
        instruction = (
            "Follow the performer's stronger emphasis and pitch movement exactly, "
            "without exaggerating either."
        )
    elif phrase_pace <= 0.34 or pause_ratio >= 0.20:
        label = "measured"
        instruction = (
            "Keep the performer's slow, deliberate pacing and clear pauses. Do not "
            "fill the spaces or add intensity."
        )
    elif phrase_expressiveness <= 0.36 and energy_slope < 2.0:
        label = "calm"
        instruction = (
            "Keep the delivery calm and low-intensity. Do not add urgency, anger, "
            "sadness, or excitement."
        )
    else:
        label = "neutral"
        instruction = (
            "Use the performer's neutral conversational delivery. Do not introduce "
            "an emotional tone that is not present."
        )

    return DeliveryReading(
        label=label,
        instruction=instruction,
        pace=pace,
        expressiveness=expressiveness,
        intensity=intensity,
    )


def specialize_delivery(
    delivery_profile: str,
    reading: DeliveryReading,
    features: Mapping[str, float],
    transcript: str,
) -> DeliveryReading:
    if delivery_profile.casefold() != "carth":
        return reading

    lowered = re.sub(r"\s+", " ", transcript.casefold()).strip()
    energy_slope = _value(features, "energy_slope_db", 0.0)
    pause_ratio = _value(features, "pause_ratio", 0.08)

    def specialized(label: str, instruction: str) -> DeliveryReading:
        return DeliveryReading(
            label=label,
            instruction=instruction,
            pace=reading.pace,
            expressiveness=reading.expressiveness,
            intensity=reading.intensity,
        )

    genuine_concern = bool(
        re.search(
            r"(?:\bwhat if\b|\bshouldn't free\b|\bshould not free\b|"
            r"\bwant to get .+ back\b|\bhelp get .+ acquitted\b)",
            lowered,
        )
    )
    strong_wry_cue = bool(
        re.search(
            r"(?:\byeah,?\s+right\b|\b(?:oh|well),?\s+great\b|"
            r"\bwonderful\b|\bpopularity contest\b|\bdo we really have "
            r"another choice\b|\bsort of funny\b|\bor did you miss that\b|"
            r"\bjust let .+ suffer,?\s+right\b|\bi'm sure you are\b|"
            r"\bstupidity and ignorance will never go out of style\b)",
            lowered,
        )
    )
    wry_cue = strong_wry_cue or (
        reading.label == "sarcastic" and not genuine_concern
    )
    if wry_cue:
        return specialized(
            "wry",
            (
                "Use dry skepticism and restrained exasperation. Keep the irony "
                "morally pointed rather than playful, smug, or broadly comic."
            ),
        )

    wounded_cue = bool(
        re.search(
            r"(?:\bmy wife\b|\bmy family\b|\bmy son\b|\byour mother\b|"
            r"\bmy home ?world\b|\bi failed you\b|\bi have to confess\b|"
            r"\bthe pain\b|\bheld her while\b|\bdidn't abandon you\b)",
            lowered,
        )
    )
    if wounded_cue and (energy_slope <= 2.0 or pause_ratio >= 0.10):
        return specialized(
            "wounded",
            (
                "Keep the hurt contained and personal. Preserve hesitations and "
                "vulnerability without turning grief into shouting or melodrama."
            ),
        )

    warm_cue = bool(
        re.search(
            r"(?:\bi'm proud of you\b|\bi am proud of you\b|\bi'm glad\b|"
            r"\bgood to see you\b|\bbest of luck\b|\bthank you\b|"
            r"\byou're alive\b|\byou are alive\b|\bthings are looking up\b)",
            lowered,
        )
    )
    if warm_cue:
        return specialized(
            "warm",
            (
                "Use earnest, understated warmth and relief. Keep it grounded and "
                "sincere rather than sentimental or overly bright."
            ),
        )

    guarded_cue = bool(
        re.search(
            r"(?:\bwouldn't trust\b|\bdo not trust\b|\bdon't trust\b|"
            r"\bbetray you\b|\bbetrayed us\b|\bthis could be a trap\b|"
            r"\bwatch yourself\b|\bcareful\b|\bevasion\b|\bhiding\b|"
            r"\bexpect some answers\b|\bdon't believe\b|\bdoesn't believe\b)",
            lowered,
        )
    )
    if guarded_cue and reading.label != "urgent":
        return specialized(
            "guarded",
            (
                "Use guarded suspicion and clipped restraint. Let distrust sit "
                "under the words without adding anger or theatrical menace."
            ),
        )

    disapproval_cue = bool(
        re.search(
            r"(?:\bdon't approve\b|\bdo not approve\b|\bshouldn't have\b|"
            r"\bshould not have\b|\bthis isn't right\b|\bthis is not right\b|"
            r"\bdoesn't deserve\b|\bdoes not deserve\b|"
            r"\bcan't believe you're\b|\bcannot believe you are\b|"
            r"\btwisted .+ credits\b|\btaking contracts to kill\b)",
            lowered,
        )
    )
    if disapproval_cue:
        return specialized(
            "disapproving",
            (
                "Use firm moral disapproval with controlled frustration. Keep the "
                "judgment earnest and protective rather than self-righteous."
            ),
        )

    resolve_cue = bool(
        re.search(
            r"(?:\bwe have to\b|\bwe need to\b|\byou have to\b|"
            r"\bwe must\b|\bi swear\b|\bprotect\b|\brescue\b|\bsave\b|"
            r"\bdestroy the\b|\bstop them\b|\blet's move\b|\blet us move\b)",
            lowered,
        )
    )
    if resolve_cue and reading.label in {"neutral", "emphatic", "urgent", "measured"}:
        return specialized(
            "resolute",
            (
                "Use steady protective resolve and military focus. Preserve the "
                "performer's intensity without making the line agitated."
            ),
        )

    if reading.label == "reflective":
        return reading

    if reading.label == "sarcastic" and genuine_concern:
        return specialized(
            "guarded",
            (
                "Use cautious, guarded concern. Preserve the performer's doubt "
                "without turning the question into sarcasm or accusation."
            ),
        )

    return reading


_ADJACENT_DELIVERIES = {
    frozenset(("calm", "neutral")),
    frozenset(("calm", "measured")),
    frozenset(("calm", "reflective")),
    frozenset(("neutral", "measured")),
    frozenset(("neutral", "reflective")),
    frozenset(("measured", "reflective")),
    frozenset(("emphatic", "urgent")),
    frozenset(("emphatic", "sarcastic")),
    frozenset(("questioning", "sarcastic")),
    frozenset(("wry", "sarcastic")),
    frozenset(("guarded", "questioning")),
    frozenset(("wounded", "reflective")),
    frozenset(("warm", "calm")),
    frozenset(("disapproving", "emphatic")),
    frozenset(("resolute", "urgent")),
}


def delivery_distance(
    performance_features: Mapping[str, float],
    anchor_features: Mapping[str, float],
    performance_delivery: DeliveryReading,
    anchor_delivery: DeliveryReading,
    transcript: str,
    anchor_text: str,
) -> float:
    def difference(key: str, scale: float, default: float = 0.0) -> float:
        return abs(
            _value(anchor_features, key, default)
            - _value(performance_features, key, default)
        ) / scale

    label_pair = frozenset((performance_delivery.label, anchor_delivery.label))
    if performance_delivery.label == anchor_delivery.label:
        delivery_penalty = 0.0
    elif label_pair in _ADJACENT_DELIVERIES:
        delivery_penalty = 0.35
    elif "questioning" in label_pair:
        delivery_penalty = 1.5
    else:
        delivery_penalty = 1.1

    rate = max(_value(anchor_features, "characters_per_second", 15.0), 0.1)
    performance_rate = max(
        _value(performance_features, "characters_per_second", 15.0),
        0.1,
    )
    text_similarity = 0.0
    normalized_target = re.sub(r"[^a-z0-9]", "", transcript.casefold())
    normalized_anchor = re.sub(r"[^a-z0-9]", "", anchor_text.casefold())
    if normalized_target and normalized_anchor:
        text_similarity = SequenceMatcher(
            None,
            normalized_target,
            normalized_anchor,
        ).ratio()

    return (
        difference("dynamic_db", 10.0, 12.0)
        + difference("pitch_variation_semitones", 3.5, 2.5)
        + difference("pitch_range_semitones", 8.0, 7.0)
        + difference("pitch_slope_semitones", 8.0)
        + difference("terminal_pitch_delta", 6.0)
        + difference("pause_ratio", 0.16, 0.08)
        + difference("energy_slope_db", 10.0)
        + abs(math.log(rate / performance_rate))
        + abs(anchor_delivery.pace - performance_delivery.pace) * 0.8
        + abs(anchor_delivery.expressiveness - performance_delivery.expressiveness)
        * 0.9
        + abs(anchor_delivery.intensity - performance_delivery.intensity) * 1.1
        + delivery_penalty
        - (text_similarity * 0.08)
    )


def load_pronunciations(path: Path) -> tuple[PronunciationEntry, ...]:
    if not path.exists():
        return ()
    document = json.loads(path.read_text(encoding="utf-8"))
    entries: list[PronunciationEntry] = []
    for value in document.get("entries", []):
        term = str(value.get("term", "")).strip()
        spoken = str(value.get("spoken", "")).strip()
        if not term or not spoken:
            continue
        aliases = tuple(
            alias
            for alias in (str(item).strip() for item in value.get("aliases", []))
            if alias
        )
        guide = str(value.get("guide", "")).strip()
        entries.append(PronunciationEntry(term, spoken, aliases, guide))
    return tuple(entries)


def build_transcription_hotwords(
    entries: Iterable[PronunciationEntry],
    context: Iterable[str] = (),
) -> str:
    entry_list = tuple(entries)
    canonical_keys = {entry.term.casefold() for entry in entry_list}
    words: list[str] = []
    seen: set[str] = set()
    context_values: list[str] = []
    for item in context:
        value = str(item).strip()
        if value and value.casefold() not in canonical_keys:
            context_values.append(value)
    for value in (*context_values, *(entry.term for entry in entry_list)):
        key = value.casefold()
        if value and key not in seen:
            words.append(value)
            seen.add(key)
    return " ".join(words)


def canonicalize_transcription(
    text: str,
    entries: Iterable[PronunciationEntry],
) -> str:
    result = text
    ordered = sorted(
        entries,
        key=lambda entry: max(
            (len(entry.term), *(len(alias) for alias in entry.aliases))
        ),
        reverse=True,
    )
    for entry in ordered:
        alternatives = sorted(
            (entry.term, *entry.aliases),
            key=len,
            reverse=True,
        )
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:"
            + "|".join(re.escape(value) for value in alternatives)
            + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        result = pattern.sub(entry.term, result)
    return result


def apply_pronunciations(
    text: str,
    entries: Iterable[PronunciationEntry],
) -> PronunciationResult:
    result = text
    matched: list[str] = []
    ordered = sorted(
        entries,
        key=lambda entry: max(
            (len(entry.term), *(len(alias) for alias in entry.aliases))
        ),
        reverse=True,
    )
    for entry in ordered:
        alternatives = sorted(
            (entry.term, *entry.aliases),
            key=len,
            reverse=True,
        )
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:"
            + "|".join(re.escape(value) for value in alternatives)
            + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        result, count = pattern.subn(entry.spoken, result)
        if count and entry.term not in matched:
            matched.append(entry.term)
    return PronunciationResult(result, tuple(matched))
