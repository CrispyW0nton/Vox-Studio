"""Speaker-identity screen for a character's lab corpus (CPU only; VOICE_LAB_CHARACTER, default baodur).

A character's conversation resources can also carry other speakers' lines (Bao-Dur's gblbaodur* files hold
Atton, HK-47 and G0-T0), and such clips can enter corpus.jsonl with Whisper transcripts (text_source == "asr"),
so text provenance alone cannot separate them from the character's own untranscribed barks. This script embeds
every train/val clip with two independent speaker encoders:
  * Resemblyzer GE2E (256-d, English read speech),
  * CAM++ (192-d, the encoder Seed-VC uses for its style vector; weights already on disk),
builds the character's centroid from clips whose text came from his own TLK/DLG lines, and flags clips
that sit far from it with both encoders. Held-out test clips are never read. The anchor rule and the flag
thresholds are the character's (<lab>/delivery/content/screen.json).

Run with the Seed-VC venv (it has resemblyzer, torch, torchaudio, librosa):
  CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
    ../../seed-vc/.venv/Scripts/python.exe speaker_check.py [--audio original|restored] [--out PATH]
Writes <lab>/delivery/speaker_check.json (and never touches the corpus or audio).
The Encoders class is reusable for scoring rendered takes (see probe_voxcpm.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only ("-1": on Windows an empty value deletes the variable)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labcfg  # noqa: E402

LAB = labcfg.LAB
SEEDVC = Path(r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab\seed-vc")
OUT = labcfg.DELIVERY / "speaker_check.json"
EXACT_SOURCES = {"tlk-exact", "dlg"}


def screen_settings() -> dict:
    """The character's screen thresholds (content/screen.json), as floats."""
    screen = labcfg.content("screen")
    z = screen["other_speaker_z"]
    return {"max_wer": float(screen["anchor"]["max_wer"]), "min_s": float(screen["anchor"]["min_duration_s"]),
            "drop": float(screen["anchor_drop_z"]), "both": float(z["both"]), "either": float(z["either"]),
            "campplus_only": float(z["campplus_only"]), "suspect": float(screen["suspect_z"]),
            "group": float(screen["group_similarity"])}


def shim_pkg_resources() -> None:
    """webrtcvad (a resemblyzer dependency) imports pkg_resources only to read its own version.
    The Seed-VC venv has no setuptools, so provide a tiny in-process stand-in (nothing is installed)."""
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("pkg_resources")

        class _Distribution:
            version = "0"

        shim.get_distribution = lambda name: _Distribution()  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = shim


def load_16k(path: str) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != 16000:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=16000)
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


class Encoders:
    """Resemblyzer + CAM++ on CPU; embed() returns L2-normalised vectors (resemblyzer may be None)."""

    def __init__(self) -> None:
        import torch

        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 4)))
        shim_pkg_resources()
        from resemblyzer import VoiceEncoder, preprocess_wav

        self.torch = torch
        self.preprocess_wav = preprocess_wav
        self.resemblyzer = VoiceEncoder(device="cpu", verbose=False)
        sys.path.insert(0, str(SEEDVC))
        from modules.campplus.DTDNN import CAMPPlus

        weights = SEEDVC / "campplus_cn_common.bin"
        if not weights.is_file():
            raise SystemExit(f"CAM++ weights missing: {weights}")
        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(torch.load(str(weights), map_location="cpu"))
        self.campplus.eval()

    def embed(self, wav16k: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        import torchaudio

        torch = self.torch
        processed = self.preprocess_wav(wav16k, source_sr=16000)
        res = None
        if processed.size >= int(16000 * 0.3):
            res = self.resemblyzer.embed_utterance(processed).astype(np.float64)
            res /= np.linalg.norm(res) + 1e-9
        with torch.no_grad():
            tensor = torch.from_numpy(wav16k).unsqueeze(0)
            feat = torchaudio.compliance.kaldi.fbank(tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            cam = self.campplus(feat.unsqueeze(0))[0].numpy().astype(np.float64)
        cam /= np.linalg.norm(cam) + 1e-9
        return res, cam


def robust_stats(reference: np.ndarray) -> tuple[float, float]:
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median))) * 1.4826
    return median, max(mad, 1e-3)


def centroid_similarities(embeddings: dict[str, np.ndarray], anchors: list[str]) -> dict[str, float]:
    """Cosine similarity to the anchor centroid; anchors are scored leave-one-out."""
    members = [i for i in anchors if i in embeddings]
    total = np.sum([embeddings[i] for i in members], axis=0)
    result: dict[str, float] = {}
    for clip_id, vector in embeddings.items():
        centroid = total - vector if clip_id in members else total
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        result[clip_id] = float(vector @ centroid)
    return result


def group_flagged(ids: list[str], embeddings: dict[str, np.ndarray], threshold: float) -> list[list[str]]:
    """Single-link grouping of flagged clips so the report shows which ones share a voice."""
    groups: list[list[str]] = []
    for clip_id in ids:
        if clip_id not in embeddings:
            groups.append([clip_id])
            continue
        placed: list[str] | None = None
        for group in groups:
            if any(o in embeddings and float(embeddings[clip_id] @ embeddings[o]) >= threshold for o in group):
                if placed is None:
                    group.append(clip_id)
                    placed = group
                else:
                    placed.extend(group)
                    group.clear()
        groups = [g for g in groups if g]
        if placed is None:
            groups.append([clip_id])
    return sorted(groups, key=len, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{labcfg.NAME} corpus speaker-identity screen (CPU).")
    parser.add_argument("--audio", choices=("original", "restored"), default="original",
                        help="which copy to embed (identity is judged on the original game audio by default)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    s = screen_settings()

    rows = [json.loads(line) for line in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["split"] in ("train", "val")]  # held-out test clips are never read
    by_id = {r["id"]: r for r in rows}
    ids = [r["id"] for r in rows]
    encoders = Encoders()

    res_emb: dict[str, np.ndarray] = {}
    cam_emb: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows, 1):
        path = row["audio"] if args.audio == "original" else str(LAB / "restored" / f"{row['id']}.wav")
        res, cam = encoders.embed(load_16k(path))
        if res is not None:
            res_emb[row["id"]] = res
        cam_emb[row["id"]] = cam
        if index % 40 == 0:
            print(f"embedded {index}/{len(rows)}", flush=True)

    anchors = [r["id"] for r in rows
               if r["kind"] == "speech" and r["text_source"] in EXACT_SOURCES
               and (r["wer"] or 0.0) <= s["max_wer"] and r["duration"] >= s["min_s"] and r["id"] in res_emb]

    def score(anchor_ids: list[str]):
        res_sim = centroid_similarities(res_emb, anchor_ids)
        cam_sim = centroid_similarities(cam_emb, anchor_ids)
        res_med, res_mad = robust_stats(np.array([res_sim[i] for i in anchor_ids]))
        cam_med, cam_mad = robust_stats(np.array([cam_sim[i] for i in anchor_ids]))
        z_res = {i: (res_sim[i] - res_med) / res_mad if i in res_sim else float("nan") for i in ids}
        z_cam = {i: (cam_sim[i] - cam_med) / cam_mad for i in ids}
        return res_sim, cam_sim, z_res, z_cam, (res_med, res_mad, cam_med, cam_mad)

    # Anchors that disagree strongly with the rest (both encoders) leave the centroid; re-score until stable.
    for _ in range(4):
        res_sim, cam_sim, z_res, z_cam, stats = score(anchors)
        kept = [i for i in anchors if not (z_res[i] < s["drop"] and z_cam[i] < s["drop"])]
        if kept == anchors:
            break
        anchors = kept
    else:
        res_sim, cam_sim, z_res, z_cam, stats = score(anchors)

    report: dict[str, dict] = {}
    flagged, suspect = [], []
    for clip_id in ids:
        row = by_id[clip_id]
        zr, zc = z_res[clip_id], z_cam[clip_id]
        has_res = bool(np.isfinite(zr))
        if ((has_res and zr < s["both"] and zc < s["both"]) or (has_res and min(zr, zc) < s["either"])
                or (not has_res and zc < s["campplus_only"])):
            flag = "other_speaker"
            flagged.append(clip_id)
        elif (has_res and zr < s["suspect"]) or zc < s["suspect"]:
            flag = "suspect"
            suspect.append(clip_id)
        else:
            flag = "ok"
        report[clip_id] = {
            "flag": flag,
            "resemblyzer_sim": round(res_sim[clip_id], 4) if clip_id in res_sim else None,
            "campplus_sim": round(cam_sim[clip_id], 4),
            "resemblyzer_z": round(float(zr), 2) if has_res else None,
            "campplus_z": round(float(zc), 2),
            "centroid_anchor": clip_id in anchors,
            "short_clip": row["duration"] < 1.5,
            "duration": row["duration"], "text_source": row["text_source"], "split": row["split"],
            "tts_usable": row["tts_usable"], "text": row["text"],
        }

    groups = group_flagged(flagged, res_emb, s["group"])
    summary = {
        "audio": args.audio,
        "clips_screened": len(ids),
        "centroid_anchors": len(anchors),
        "anchor_rule": f"speech, text_source tlk-exact|dlg, wer <= {s['max_wer']:g}, duration >= {s['min_s']:g} s, "
                       "split train|val",
        "flag_rule": (f"other_speaker: robust z < {s['both']:g} on BOTH encoders (or < {s['either']:g} on either); "
                      f"suspect: z < {s['suspect']:g} on either; z = (sim - median) / (1.4826 * MAD) of leave-one-out "
                      "anchor sims"),
        "anchor_similarity": {
            "resemblyzer_median": round(stats[0], 4), "resemblyzer_mad": round(stats[1], 4),
            "campplus_median": round(stats[2], 4), "campplus_mad": round(stats[3], 4),
        },
        "other_speaker": flagged,
        "suspect": suspect,
        "other_speaker_seconds": round(sum(by_id[i]["duration"] for i in flagged), 1),
        "other_speaker_tts_usable_train": [i for i in flagged if by_id[i]["tts_usable"] and by_id[i]["split"] == "train"],
        "other_speaker_groups": [
            {"size": len(g), "ids": g, "examples": [by_id[i]["text"][:80] for i in g[:4]]} for g in groups
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "clips": report}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
