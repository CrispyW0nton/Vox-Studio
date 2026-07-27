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


@dataclass(frozen=True)
class PronunciationEntry:
    term: str
    spoken: str
    aliases: tuple[str, ...] = ()


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
    terminal_pitch_delta = _value(features, "terminal_pitch_delta", 0.0)

    absolute_pace = _clamp((rate - 9.0) / 12.0)
    pitch_expression = _clamp((pitch_range - 3.0) / 10.0)
    variation_expression = _clamp((pitch_variation - 1.0) / 3.5)
    dynamic_expression = _clamp((dynamic_range - 8.0) / 12.0)
    expressiveness = (
        (pitch_expression * 0.45)
        + (variation_expression * 0.30)
        + (dynamic_expression * 0.25)
    )
    pace = absolute_pace

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
        (expressiveness * 0.58) + (pace * 0.27) + (rising_energy * 0.15)
    )
    question = _looks_like_question(transcript, terminal_pitch_delta)
    sarcastic = _sounds_sarcastic(
        transcript,
        expressiveness,
        terminal_pitch_delta,
        question,
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
        (pace >= 0.70 and expressiveness >= 0.50)
        or (pace >= 0.58 and energy_slope >= 3.0)
    ):
        label = "urgent"
        instruction = (
            "Match the performer's quick, urgent delivery without making it louder "
            "or more emotional than the performance."
        )
    elif expressiveness >= 0.68:
        label = "emphatic"
        instruction = (
            "Follow the performer's stronger emphasis and pitch movement exactly, "
            "without exaggerating either."
        )
    elif pace <= 0.34 or pause_ratio >= 0.20:
        label = "measured"
        instruction = (
            "Keep the performer's slow, deliberate pacing and clear pauses. Do not "
            "fill the spaces or add intensity."
        )
    elif expressiveness <= 0.36 and energy_slope < 2.0:
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


_ADJACENT_DELIVERIES = {
    frozenset(("calm", "neutral")),
    frozenset(("calm", "measured")),
    frozenset(("neutral", "measured")),
    frozenset(("emphatic", "urgent")),
    frozenset(("emphatic", "sarcastic")),
    frozenset(("questioning", "sarcastic")),
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
        entries.append(PronunciationEntry(term, spoken, aliases))
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
