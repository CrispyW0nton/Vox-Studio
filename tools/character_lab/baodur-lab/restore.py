"""Bandwidth-restore Bao-Dur's 22 kHz MP3-sourced game clips with ClearerVoice MossFormer2_SR_48K.

Run from voice-lab/clearvoice with its venv (checkpoints/ is resolved relative to that folder):
  .venv/Scripts/python.exe ../baodur-lab/restore.py
Writes BaoDurLab/restored/<id>.wav (48 kHz mono) and restore_report.json; originals are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

LAB = Path(r"C:\Users\NewAdmin\Documents\KotorMods\Voices\BaoDurLab")


def high_band_db(path: Path, cutoff: float = 11500.0) -> float:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / rate)
    return float(10 * np.log10(power[freqs > cutoff].sum() / power.sum() + 1e-12))


def main() -> None:
    if torch.cuda.is_available():
        # Shared GPU: Substance Painter and other sessions need headroom; stay within ~8 GB.
        torch.cuda.set_per_process_memory_fraction(0.25)
    from clearvoice import ClearVoice

    rows = [json.loads(line) for line in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    out_dir = LAB / "restored"
    out_dir.mkdir(exist_ok=True)
    model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
    report = []
    for row in rows:
        source, target = Path(row["audio"]), out_dir / f"{row['id']}.wav"
        if not target.exists():
            restored = model(input_path=str(source), online_write=False)
            model.write(restored, output_path=str(target))
        info = sf.info(target)
        report.append({"id": row["id"], "rate": info.samplerate, "duration": round(info.frames / info.samplerate, 3),
                       "source_duration": row["duration"], "high_band_db_before": round(high_band_db(source), 1),
                       "high_band_db_after": round(high_band_db(target), 1)})
        print(json.dumps(report[-1]), flush=True)
    before = np.median([r["high_band_db_before"] for r in report])
    after = np.median([r["high_band_db_after"] for r in report])
    drift = [r["id"] for r in report if abs(r["duration"] - r["source_duration"]) > 0.05]
    summary = {"clips": len(report), "median_high_band_db_before": float(before), "median_high_band_db_after": float(after),
               "duration_drift_over_50ms": drift, "model": "alibabasglab/MossFormer2_SR_48K",
               "review": "Listen before training: super-resolution can invent sibilance or alter timbre."}
    (LAB / "restore_report.json").write_text(json.dumps({"summary": summary, "clips": report}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
