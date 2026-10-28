"""CPU check: the Seed-VC V2 config and every prefetched checkpoint resolve offline, weights really load, and a
tiny conversion runs end to end (content tokens -> CFM -> BigVGAN) on the CPU.

  [VOICE_LAB_CHARACTER=<slug>] ../../seed-vc/.venv/Scripts/python.exe check_offline.py [--style] [--register-probe] [--out DIR]

CPU only: CUDA is hidden from this process before torch is imported (common.hide_cuda), so the report's
cuda_visible must read false. Character data (VOICE_LAB_CHARACTER, default baodur): the source is a val clip of
<lab>/corpus.jsonl (--source-id, default settings check_source_id, else the first val speech clip of 3-6 s), the
prompt the first neutral fallback reference of <lab>/seedvc/data/original/prepare_report.json (run prepare.py first);
the report goes to <lab>/seedvc/checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.hide_cuda()  # CPU only

REQUIRED = [  # (repo, file, cache_dir relative to seed-vc or None for HF_HOME)
    ("Plachta/Seed-VC", "v2/cfm_small.pth", "./checkpoints"),
    ("Plachta/Seed-VC", "v2/ar_base.pth", "./checkpoints"),
    ("Plachta/ASTRAL-quantization", "bsq32/bsq32_light.pth", "./checkpoints"),
    ("Plachta/ASTRAL-quantization", "bsq2048/bsq2048_light.pth", "./checkpoints"),
    ("funasr/campplus", "campplus_cn_common.bin", "./checkpoints"),
    ("facebook/hubert-large-ll60k", "config.json", None),
    ("facebook/hubert-large-ll60k", "preprocessor_config.json", None),
    ("facebook/hubert-large-ll60k", "pytorch_model.bin", None),
    ("openai/whisper-small", "tokenizer.json", None),
    ("openai/whisper-small", "tokenizer_config.json", None),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "config.json", None),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt", None),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=C.DATA / "checks")
    p.add_argument("--style", action="store_true", help="also run the AR style-conversion path")
    p.add_argument("--source-id", default=C.SETTINGS["check_source_id"],
                   help="val clip used as the source (default: settings check_source_id, else the first val speech "
                        "clip of 3-6 s)")
    p.add_argument("--diffusion-steps", type=int, default=3)
    p.add_argument("--register-probe", action="store_true",
                   help="also convert the source shifted -4/0/+4 semitones and report median F0 in vs out")
    p.add_argument("--probe-steps", type=int, default=30)
    args = p.parse_args()
    out_dir = args.out.resolve()

    C.setup_seedvc()
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from svc_core import Converter

    report = {"cuda_visible": torch.cuda.is_available(), "hf_home": str(C.HF_HOME), "files": {}}
    for repo, name, cache in REQUIRED:
        path = hf_hub_download(repo_id=repo, filename=name, cache_dir=cache, local_files_only=True)
        report["files"][f"{repo}/{name}"] = path
    t0 = time.time()
    wrapper = C.build_wrapper()
    report["instantiate_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    C.load_weights(wrapper)
    report["load_checkpoints_s"] = round(time.time() - t0, 1)

    # Upstream loads with strict=False; prove the pretrained tensors actually match the config.
    keys = {}
    for part, module, regulator, file in (("cfm", wrapper.cfm, wrapper.cfm_length_regulator, "v2/cfm_small.pth"),
                                          ("ar", wrapper.ar, wrapper.ar_length_regulator, "v2/ar_base.pth")):
        ckpt = torch.load(report["files"][f"Plachta/Seed-VC/{file}"], map_location="cpu", weights_only=True)
        for label, mod, state in ((part, module, ckpt["net"][part]), (f"{part}_length_regulator", regulator,
                                                                      ckpt["net"]["length_regulator"])):
            missing, unexpected = mod.load_state_dict(C._strip(state), strict=False)
            keys[label] = {"tensors": len(state), "missing": missing, "unexpected_ignored": unexpected}
    report["pretrained_key_match"] = keys
    report["params_m"] = {name: round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
                          for name, m in wrapper.named_children()}

    corpus = C.load_corpus()
    rows = {r["id"]: r for r in corpus}
    args.source_id = args.source_id or C.check_source_default(corpus)
    source = rows[args.source_id]
    assert source["split"] == "val", "the check uses a val clip, never a test clip"
    fallback = C.read_json(C.PREPARED / "original" / "prepare_report.json")["fallback_references"][0]
    assert rows[fallback["id"]]["split"] in ("train", "val")
    source22, _ = C.load_mono(source["audio"], C.MODEL_SR)
    ref22, _ = C.load_mono(fallback["audio"], C.MODEL_SR)
    device = torch.device("cpu")
    for style in ([False, True] if args.style else [False]):
        converter = Converter(wrapper, device, fp16=False, convert_style=style)
        ref = converter.reference((fallback["id"], style), ref22)
        t0 = time.time()
        y22 = converter.convert(source22, ref, diffusion_steps=args.diffusion_steps, seed=1234)
        elapsed = time.time() - t0
        source_db = C.active_rms_db(source22, C.MODEL_SR)
        y48, gain_db, ceiling_db = C.finalize(y22, source_db)
        name = f"check_{'style' if style else 'timbre'}_{args.source_id}_ref_{fallback['id']}_{args.diffusion_steps}steps.wav"
        C.write_pcm16(out_dir / name, y48, C.OUT_SR)
        report[f"conversion_{'style' if style else 'timbre'}"] = {
            "source_s": round(len(source22) / C.MODEL_SR, 3), "output_s": round(len(y48) / C.OUT_SR, 3),
            "diffusion_steps": args.diffusion_steps, "cpu_seconds": round(elapsed, 1),
            "finite": bool(np.isfinite(y22).all()),
            "raw_peak_dbfs": round(float(20 * np.log10(np.abs(y22).max() + 1e-9)), 2),
            "raw_active_db": round(C.active_rms_db(y22, C.MODEL_SR), 2), "source_active_db": round(source_db, 2),
            "gain_db": round(gain_db, 2), "ceiling_gain_db": round(ceiling_db, 2), "wav": str(out_dir / name)}
    if args.register_probe:
        import librosa

        def median_f0(wave22) -> float:
            stats = C.f0_stats(wave22)
            return stats["median_hz"] if stats else float("nan")

        converter = Converter(wrapper, device, fp16=False, convert_style=False)
        ref = converter.reference((fallback["id"], "probe"), ref22)
        probe = {"reference_f0_hz": median_f0(ref22), "f0_settings": C.F0_KW, "reference_active_db": round(C.active_rms_db(ref22, C.MODEL_SR), 2),
                 "steps": args.probe_steps, "model": "pretrained cfm_small", "shifts": []}
        for shift in (-4, 0, 4):
            src = source22 if shift == 0 else librosa.effects.pitch_shift(source22, sr=C.MODEL_SR, n_steps=shift,
                                                                          res_type="soxr_hq")
            t0 = time.time()
            y22 = converter.convert(src, ref, diffusion_steps=args.probe_steps, seed=1234)
            pair = C.f0_pair(C.f0_track(src), C.f0_track(y22))
            y48, _, _ = C.finalize(y22, C.active_rms_db(src, C.MODEL_SR))
            C.write_pcm16(out_dir / f"probe_{args.source_id}_shift{shift:+d}_{args.probe_steps}steps.wav", y48, C.OUT_SR)
            probe["shifts"].append({"semitones": shift, "source_f0_hz": (pair["source_f0"] or {}).get("median_hz"),
                                    "output_f0_hz": (pair["output_f0"] or {}).get("median_hz"),
                                    "range_ratio": pair["range_ratio"], "f0_corr": pair["f0_corr"],
                                    "output_hf_tilt_db": round(C.hf_tilt_db(y22), 2),
                                    "raw_active_db": round(C.active_rms_db(y22, C.MODEL_SR), 2),
                                    "raw_peak_dbfs": round(float(20 * np.log10(np.abs(y22).max() + 1e-9)), 2),
                                    "cpu_seconds": round(time.time() - t0, 1)})
            print(probe["shifts"][-1], flush=True)
        report["register_probe"] = probe
    print(json.dumps(report, indent=1))
    ok = all(not v["missing"] for v in keys.values()) and \
        all(report[k]["finite"] for k in report if k.startswith("conversion_"))
    report["ok"] = ok
    C.write_json(out_dir / "check_offline_report.json", report)
    print("OK" if ok else "CHECK FAILED", "->", out_dir / "check_offline_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
