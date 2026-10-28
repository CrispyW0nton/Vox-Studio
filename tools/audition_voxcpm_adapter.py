from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMForXVector
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig


SPEAKER_SAMPLE_RATE = 16_000


def load_mono(path: Path, sample_rate: int = SPEAKER_SAMPLE_RATE) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor)
    return mono.astype(np.float32, copy=False)


def speaker_embedding(
    path: Path,
    extractor,
    model: WavLMForXVector,
) -> np.ndarray:
    audio = load_mono(path)[: SPEAKER_SAMPLE_RATE * 12]
    inputs = extractor(
        audio,
        sampling_rate=SPEAKER_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        embedding = model(**inputs).embeddings[0]
    embedding = torch.nn.functional.normalize(embedding, dim=0)
    return embedding.cpu().numpy()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def normalized_contour(values: np.ndarray, size: int = 100) -> np.ndarray:
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        return np.zeros(size, dtype=np.float32)
    source = np.linspace(0.0, 1.0, len(values))[finite]
    target = np.linspace(0.0, 1.0, size)
    contour = np.interp(target, source, values[finite])
    deviation = float(np.std(contour))
    return (
        ((contour - float(np.mean(contour))) / deviation).astype(np.float32)
        if deviation > 1e-5
        else np.zeros(size, dtype=np.float32)
    )


def prosody_similarity(prompt_path: Path, rendered_path: Path) -> dict[str, float]:
    prompt = load_mono(prompt_path)
    rendered = load_mono(rendered_path)
    hop = 320

    def contours(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energy = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=hop,
        )[0]
        pitch = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=SPEAKER_SAMPLE_RATE,
            frame_length=1024,
            hop_length=hop,
        )
        pitch[pitch >= librosa.note_to_hz("C6") * 0.98] = np.nan
        return normalized_contour(pitch), normalized_contour(energy)

    prompt_pitch, prompt_energy = contours(prompt)
    rendered_pitch, rendered_energy = contours(rendered)
    pitch_correlation = float(np.corrcoef(prompt_pitch, rendered_pitch)[0, 1])
    energy_correlation = float(np.corrcoef(prompt_energy, rendered_energy)[0, 1])
    if not np.isfinite(pitch_correlation):
        pitch_correlation = 0.0
    if not np.isfinite(energy_correlation):
        energy_correlation = 0.0
    duration_ratio = min(len(prompt), len(rendered)) / max(len(prompt), len(rendered))
    return {
        "pitch_contour_correlation": round(pitch_correlation, 5),
        "energy_contour_correlation": round(energy_correlation, 5),
        "duration_similarity": round(float(duration_ratio), 5),
    }


def manifest_paths(path: Path, limit: int = 24) -> list[Path]:
    return [
        Path(json.loads(line)["audio"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render matched base/LoRA auditions and compare speaker identity."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--control-text", default="")
    parser.add_argument("--target-audio-root", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--contrast-audio-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-audio", action="store_true")
    arguments = parser.parse_args()
    arguments.output_root.mkdir(parents=True, exist_ok=True)

    engine_root = Path(os.environ["LOCALAPPDATA"]) / "VoxStudio" / "engines" / "voxcpm2"
    baseline_path = arguments.output_root / "base.wav"
    adapted_path = arguments.output_root / "adapted.wav"
    loaded: list[str] = []
    skipped: list[str] = []
    if not arguments.reuse_audio:
        model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            load_denoiser=False,
            cache_dir=str(engine_root / "cache"),
            optimize=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            lora_config=LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
                r=32,
                alpha=32,
                dropout=0.0,
            ),
        )
        generation = {
            "text": (
                f"({arguments.control_text}){arguments.text}"
                if arguments.control_text
                else arguments.text
            ),
            "reference_wav_path": str(arguments.reference_audio),
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "normalize": False,
        }
        if not arguments.control_text:
            generation["prompt_wav_path"] = str(arguments.prompt_audio)
            generation["prompt_text"] = arguments.prompt_text

        torch.manual_seed(42)
        model.set_lora_enabled(False)
        baseline = model.generate(**generation)
        sf.write(baseline_path, baseline, model.tts_model.sample_rate)

        loaded, skipped = model.load_lora(str(arguments.checkpoint))
        if not loaded:
            raise RuntimeError("The checkpoint did not contain compatible LoRA weights.")
        model.set_lora_enabled(True)
        torch.manual_seed(42)
        adapted = model.generate(**generation)
        sf.write(adapted_path, adapted, model.tts_model.sample_rate)
    elif not baseline_path.exists() or not adapted_path.exists():
        parser.error("--reuse-audio requires existing base.wav and adapted.wav")

    wavlm_cache = engine_root / "cache"
    extractor = AutoFeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus-sv",
        cache_dir=str(wavlm_cache),
        local_files_only=True,
    )
    speaker_model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv",
        cache_dir=str(wavlm_cache),
        local_files_only=True,
    ).eval()
    if arguments.target_manifest:
        target_paths = manifest_paths(arguments.target_manifest)
    elif arguments.target_audio_root:
        target_paths = sorted(arguments.target_audio_root.glob("*.wav"))[:24]
    else:
        parser.error("one of --target-manifest or --target-audio-root is required")
    target_embeddings = [
        speaker_embedding(path, extractor, speaker_model) for path in target_paths
    ]
    centroid = np.mean(target_embeddings, axis=0)
    centroid /= np.linalg.norm(centroid)
    base_embedding = speaker_embedding(baseline_path, extractor, speaker_model)
    adapted_embedding = speaker_embedding(adapted_path, extractor, speaker_model)
    prompt_embedding = speaker_embedding(arguments.prompt_audio, extractor, speaker_model)
    metrics = {
        "loaded_lora_keys": len(loaded),
        "skipped_lora_keys": len(skipped),
        "target_clip_count": len(target_embeddings),
        "base_speaker_similarity": round(
            cosine(base_embedding, centroid),
            5,
        ),
        "adapted_speaker_similarity": round(
            cosine(adapted_embedding, centroid),
            5,
        ),
        "prompt_speaker_similarity": round(
            cosine(
                prompt_embedding, centroid,
            ),
            5,
        ),
        "base_delivery": prosody_similarity(arguments.prompt_audio, baseline_path),
        "adapted_delivery": prosody_similarity(arguments.prompt_audio, adapted_path),
        "base_audio": str(baseline_path),
        "adapted_audio": str(adapted_path),
    }
    if arguments.contrast_audio_root:
        contrast_paths = sorted(arguments.contrast_audio_root.glob("*.wav"))[:24]
        contrast_embeddings = [
            speaker_embedding(path, extractor, speaker_model)
            for path in contrast_paths
        ]
        contrast_centroid = np.mean(contrast_embeddings, axis=0)
        contrast_centroid /= np.linalg.norm(contrast_centroid)
        base_contrast = cosine(base_embedding, contrast_centroid)
        adapted_contrast = cosine(adapted_embedding, contrast_centroid)
        prompt_contrast = cosine(prompt_embedding, contrast_centroid)
        metrics.update(
            {
                "contrast_clip_count": len(contrast_embeddings),
                "base_contrast_similarity": round(base_contrast, 5),
                "base_identity_margin": round(
                    metrics["base_speaker_similarity"] - base_contrast,
                    5,
                ),
                "adapted_contrast_similarity": round(adapted_contrast, 5),
                "adapted_identity_margin": round(
                    metrics["adapted_speaker_similarity"] - adapted_contrast,
                    5,
                ),
                "prompt_contrast_similarity": round(prompt_contrast, 5),
                "prompt_identity_margin": round(
                    metrics["prompt_speaker_similarity"] - prompt_contrast,
                    5,
                ),
            }
        )
    (arguments.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
