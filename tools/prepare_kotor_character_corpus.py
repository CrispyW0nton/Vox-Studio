from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    from tools.scripts.build_kotor_rvc_dataset import decode_audio
except ModuleNotFoundError:
    from scripts.build_kotor_rvc_dataset import decode_audio


@dataclass(frozen=True)
class DialogueRecord:
    voiceover: str
    speaker: str
    text: str
    performance_directions: tuple[str, ...]


def clean_dialogue_text(text: str) -> str:
    cleaned = re.sub(r"\{[^{}]*\}", " ", text)
    cleaned = re.sub(r"\[[^\[\]]*\]", " ", cleaned)
    cleaned = re.sub(r"::.*?::", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" \t\"")


def performance_directions(text: str) -> tuple[str, ...]:
    values = re.findall(r"\{([^{}]+)\}|\[([^\[\]]+)\]|::(.*?)::", text)
    directions: list[str] = []
    for groups in values:
        value = next((group for group in groups if group), "")
        normalized = re.sub(r"\s+", " ", value).strip()
        if normalized and normalized.casefold() not in {
            direction.casefold() for direction in directions
        }:
            directions.append(normalized)
    return tuple(directions)


def delivery_tags(directions: tuple[str, ...]) -> tuple[str, ...]:
    text = " ".join(directions).casefold()
    tags: list[str] = []
    patterns = (
        ("sarcastic", r"\b(?:sarcastic|sarcasm|mocking|mockery)\b"),
        ("wry", r"\b(?:wry|dry|rueful|miffed|amused|skepti)"),
        ("wounded", r"\b(?:bitter|hurt|pain|grief|lost|sad|sorrow|weary)\b"),
        ("reflective", r"\b(?:reflect|wistful|nostalg|quiet|sigh|memory|rememb)"),
        ("urgent", r"\b(?:angry|pissed|urgent|furious|rage|shout|yell)\b"),
        ("emphatic", r"\b(?:emphatic|forceful|command|stern|sharp)\b"),
        ("warm", r"\b(?:warm|kind|gentle|affection|relief|conciliatory)\b"),
        ("guarded", r"\b(?:guarded|suspicious|distrust|cautious)\b"),
        ("questioning", r"\b(?:question|incredulous|confus|uncertain)\b"),
    )
    for tag, pattern in patterns:
        if re.search(pattern, text):
            tags.append(tag)
    return tuple(tags)


def speaker_matches_character(speaker: str, character_token: str) -> bool:
    normalized_speaker = re.sub(r"[^a-z0-9]", "", speaker.casefold())
    normalized_character = re.sub(r"[^a-z0-9]", "", character_token.casefold())
    return not normalized_speaker or normalized_character in normalized_speaker


def ensure_pykotor() -> None:
    try:
        import pykotor  # noqa: F401

        return
    except ImportError:
        pass
    roaming_site_packages = (
        Path(os.environ.get("APPDATA", ""))
        / "Python"
        / "Python314"
        / "site-packages"
    )
    if roaming_site_packages.exists():
        sys.path.append(str(roaming_site_packages))
    try:
        import pykotor  # noqa: F401
    except ImportError as exception:
        raise RuntimeError(
            "PyKotor is required to match installed KOTOR dialogue and audio."
        ) from exception


def read_dialogue_records(
    game_root: Path,
    character_token: str,
    voiceover_token: str | None = None,
) -> dict[str, DialogueRecord]:
    ensure_pykotor()
    from pykotor.extract.installation import Installation
    from pykotor.resource.generics.dlg import read_dlg
    from pykotor.resource.type import ResourceType

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        installation = Installation(game_root)

    records: dict[str, DialogueRecord] = {}
    seen_resources: set[tuple[str, int, int]] = set()
    token = (voiceover_token or character_token).casefold()
    for module_name in installation.modules_list():
        try:
            resources = installation.module_resources(module_name)
        except Exception:
            continue
        for resource in resources:
            if resource.restype() != ResourceType.DLG:
                continue
            identity = (
                str(resource.filepath()),
                int(resource.offset()),
                int(resource.size()),
            )
            if identity in seen_resources:
                continue
            seen_resources.add(identity)
            try:
                with contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(io.StringIO()):
                    dialogue = read_dlg(resource.data())
            except Exception:
                continue
            for node in dialogue.all_entries():
                voiceover = str(node.vo_resref).strip()
                if not voiceover or token not in voiceover.casefold():
                    continue
                speaker = str(node.speaker).strip()
                if not speaker_matches_character(speaker, character_token):
                    continue
                stringref = int(node.text.stringref)
                if stringref < 0:
                    continue
                raw_text = installation.talktable().string(stringref)
                spoken_text = clean_dialogue_text(raw_text)
                if not spoken_text:
                    continue
                records.setdefault(
                    voiceover.casefold(),
                    DialogueRecord(
                        voiceover=voiceover,
                        speaker=speaker,
                        text=spoken_text,
                        performance_directions=performance_directions(raw_text),
                    ),
                )
    return records


def source_audio_index(stream_voice_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in stream_voice_root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in {".wav", ".mp3"}:
            index.setdefault(path.stem.casefold(), path)
    return index


def game_audio_index(game_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    seen_roots: set[str] = set()
    for relative in ("StreamVoice", "streamvoice", "streamwaves"):
        root = game_root / relative
        root_key = os.path.normcase(str(root.resolve()))
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        if not root.is_dir():
            continue
        for stem, path in source_audio_index(root).items():
            index.setdefault(stem, path)
    return index


def is_valid_corpus_wave(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getframerate() == 48000
                and audio.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False


def atomic_write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as staged:
        staged.write(json.dumps(value, indent=2))
        staged_path = Path(staged.name)
    try:
        os.replace(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def prepare_corpus(
    game_root: Path,
    character_token: str,
    output_root: Path,
    ffmpeg: str,
    voiceover_token: str | None = None,
) -> dict:
    records = read_dialogue_records(game_root, character_token, voiceover_token)
    if not records:
        raise RuntimeError(
            f"No dialogue records were found for character '{character_token}'."
        )
    sources = game_audio_index(game_root)
    output_root.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    failures: list[dict[str, str]] = []
    for stem, record in sorted(records.items()):
        source = sources.get(stem)
        if source is None:
            continue
        target = output_root / f"{record.voiceover}.wav"
        try:
            if not is_valid_corpus_wave(target):
                with tempfile.NamedTemporaryFile(
                    dir=output_root,
                    prefix=f".{target.stem}.",
                    suffix=".tmp.wav",
                    delete=False,
                ) as staged:
                    staged_path = Path(staged.name)
                try:
                    decode_audio(
                        ffmpeg,
                        source,
                        staged_path,
                        sample_rate=48000,
                    )
                    if not is_valid_corpus_wave(staged_path):
                        raise RuntimeError("Decoded audio is not mono 48 kHz PCM16.")
                    os.replace(staged_path, target)
                finally:
                    staged_path.unlink(missing_ok=True)
        except Exception as exception:
            failures.append({"source": str(source), "error": str(exception)})
            continue
        entries[target.name.casefold()] = {
            "voiceover": record.voiceover,
            "speaker": record.speaker,
            "text": record.text,
            "performance_directions": list(record.performance_directions),
            "delivery_tags": list(delivery_tags(record.performance_directions)),
            "source": str(source),
        }

    if not entries:
        raise RuntimeError(
            f"No usable dialogue audio matched character '{character_token}'."
        )

    manifest = {
        "format_version": 1,
        "character": character_token,
        "voiceover_token": voiceover_token or character_token,
        "game_root": str(game_root),
        "matched_dialogue_records": len(entries),
        "available_dialogue_records": len(records),
        "entries": entries,
        "failures": failures,
    }
    atomic_write_json(output_root / "dialogue_manifest.json", manifest)
    expected_names = set(entries)
    for stale in output_root.glob("*.wav"):
        if stale.name.casefold() not in expected_names:
            stale.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exact local KOTOR character dialogue corpus."
    )
    parser.add_argument("--character", required=True)
    parser.add_argument(
        "--voiceover-token",
        help="Optional filename token when it differs from the character name.",
    )
    parser.add_argument(
        "--game-root",
        type=Path,
        default=(
            Path(r"C:\Program Files (x86)\Steam\steamapps\common")
            / "Knights of the Old Republic II"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    arguments = parser.parse_args()
    ffmpeg = shutil.which(arguments.ffmpeg)
    if ffmpeg is None:
        parser.error(f"FFmpeg was not found: {arguments.ffmpeg}")
    manifest = prepare_corpus(
        arguments.game_root,
        arguments.character,
        arguments.output,
        ffmpeg,
        arguments.voiceover_token,
    )
    print(
        json.dumps(
            {
                "character": manifest["character"],
                "matched_dialogue_records": manifest["matched_dialogue_records"],
                "available_dialogue_records": manifest[
                    "available_dialogue_records"
                ],
                "failures": len(manifest["failures"]),
                "output": str(arguments.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
