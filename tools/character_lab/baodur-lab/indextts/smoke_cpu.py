"""CPU smoke test for the IndexTTS-2.5 install: offline imports, path resolution and checkpoint compatibility
WITHOUT loading the full model or running inference (the shared machine's RAM/commit is near its limit and the GPU
belongs to other jobs).

  set CUDA_VISIBLE_DEVICES=-1
  [set VOICE_LAB_CHARACTER=jolee]       (voice-lab\\characters; unset = baodur)
  ...\\voice-lab\\index-tts\\.venv\\Scripts\\python.exe smoke_cpu.py [--clips <id>,<id>] [--lines <id>,<id>] [--no-rehash]
                                                                    [--skip-weights]

What it checks (report: <lab>\\indextts\\smoke\\cpu_smoke.json, <lab> = the character's data folder):
  1. environment: CUDA hidden (torch.cuda.is_available() is False), HF/ModelScope offline, a socket guard that
     blocks any non-loopback connection; every model class the 2.5 code path uses imports.
  2. paths: every file IndexTTS2(cfg_path, model_dir) of infer_v2_5.py opens resolves inside the install, matches
     the size + sha256 recorded when it was fetched (PROVENANCE.json), and the auxiliary-model downloader
     (ensure_models_available) would NOT be triggered.
  3. weights: each checkpoint is opened with torch.load(mmap=True, weights_only=True) - the torch 2.8 default the
     upstream loaders rely on - and its tensor names/shapes are compared with the corresponding module built on the
     META device (no memory): GPT UnifiedVoice(spk_cond_mode="campplus"), semantic codec, s2mel (CFM + length
     regulator), BigVGAN, w2v-bert-2.0 (safetensors header) and QwenEmotion (Qwen3, safetensors header). CAMPPlus
     (28 MB) is loaded for real. Character-independent; --skip-weights leaves these audits out (they peak at
     ~7 GB host commit) when only a new character's references and lines need checking.
  4. front ends on 1-2 TRAIN clips (never test clips): reference trimming, librosa load, 16 k / 22.05 k resampling,
     SeamlessM4T features, Kaldi fbank -> CAMPPlus speaker-style embedding, the s2mel mel; the text front end
     (wetext normalizer, multilingual tiktoken tokenizer, pronunciation annotations, Japanese G2P init) on 2 script
     lines. Nothing here runs the GPT, w2v-bert, s2mel, codec or vocoder forward passes.
     Default clips/lines: the character's <lab>\\indextts\\smoke_defaults.json (Bao-Dur: gblbaodur248 identity +
     gblbaodur229 angry; new_07_angry + heldout_gblbaodur276), else derived from the delivery step's outputs
     (default_selection).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" would be deleted on Windows; -1 hides every GPU

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REPORT = C.DATA_ROOT / "smoke" / "cpu_smoke.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default="",
                   help="1-2 TRAIN clip ids for the reference front-end check (default: the character's smoke "
                        "defaults: identity clip + one delivery reference)")
    p.add_argument("--lines", default="", help="script lines for the text check (default: the character's smoke "
                                               "defaults: a new line + a held-out line)")
    p.add_argument("--no-rehash", action="store_true", help="compare sizes only (skip re-hashing ~8.3 GB)")
    p.add_argument("--skip-weights", action="store_true",
                   help="leave out the checkpoint-vs-module audits of step 3 (character-independent; ~7 GB host "
                        "commit peak)")
    p.add_argument("--report", type=Path, default=REPORT)
    args = p.parse_args()
    if not args.clips or not args.lines:
        chosen = default_selection(args.lines)
        args.clips = args.clips or ",".join(chosen["clips"])
        args.lines = args.lines or ",".join(chosen["lines"])
    return args


def default_selection(lines: str = "") -> dict[str, list[str]]:
    """The character's default smoke clips and lines: C.SMOKE_DEFAULTS_FILE ({"clips": [...], "lines": [...]}) where it
    lists them, else derived from the corpus/delivery outputs. Derived lines: the first new line whose delivery is not
    the base delivery (C.base_delivery; else the first new line) + the first held-out line. Derived clips: the identity
    clip render.py uses by default (top usable base-delivery reference) + the top usable reference of the first line's
    delivery (usable = a TRAIN speech clip that passes the speaker screen, exactly what step 4a accepts)."""
    doc = json.loads(C.SMOKE_DEFAULTS_FILE.read_text(encoding="utf-8")) if C.SMOKE_DEFAULTS_FILE.exists() else {}
    out = {"clips": [str(x) for x in doc.get("clips") or []], "lines": [str(x) for x in doc.get("lines") or []]}
    if out["clips"] and out["lines"]:
        return out
    missing = [str(p) for p in (C.CORPUS, C.TAKES_SCRIPT, C.DELIVERY_MAP, C.SPEAKER_SCREEN) if not p.exists()]
    if missing:
        raise SystemExit(f"no smoke defaults for {C.CHAR['display_name']}: {C.SMOKE_DEFAULTS_FILE} does not list them "
                         f"and the corpus/delivery outputs to derive them from are missing ({missing}); run those "
                         "steps for this character first or pass --clips and --lines")
    script = json.loads(C.TAKES_SCRIPT.read_text(encoding="utf-8")).get("lines", [])
    base = C.base_delivery()
    if not out["lines"]:
        new = [ln for ln in script if ln.get("source") != "heldout"]
        first = next((ln for ln in new if str(ln.get("delivery") or base).casefold() != base), new[0] if new else None)
        held = next((ln for ln in script if ln.get("source") == "heldout"), None)
        out["lines"] = [str(ln["id"]) for ln in (first, held) if ln is not None]
    if not out["clips"]:
        by_id = {str(ln["id"]): ln for ln in script}
        wanted = [x.strip() for x in lines.split(",") if x.strip()] or out["lines"]
        first_line = (by_id.get(wanted[0]) if wanted else None) or {}
        label = str(first_line.get("delivery") or base).casefold()
        corpus = {r["id"]: r for r in C.load_corpus()}
        screen = C.load_speaker_screen()
        deliveries = {k.casefold(): v for k, v in
                      json.loads(C.DELIVERY_MAP.read_text(encoding="utf-8"))["deliveries"].items()}

        def top(name: str, exclude: set) -> str | None:
            for ref in deliveries.get(name, {}).get("references", []):
                cid = str(ref.get("id", ""))
                row = corpus.get(cid)
                if (row is not None and row["split"] == "train" and row["kind"] == "speech" and cid not in exclude
                        and cid not in screen["confirmed"] and cid not in screen["uncertain"]):
                    return cid
            return None

        identity = top(base, set())
        second = top(label, {identity})
        if second is None:  # no other usable clip for that delivery: the first other delivery that has one
            second = next((c for c in (top(name, {identity}) for name in deliveries if name != base) if c), None)
        out["clips"] = [c for c in (identity, second) if c]
    return out


def emotion_probe_text(script: dict[str, dict], line_ids: list[str]) -> str:
    """The text of the QwenEmotion tokenizer check: what render.py's id_emo_text mode would hand QwenEmotion for the
    first smoke line - its delivery's description in the character's emotion controls, else the line itself."""
    line = script.get(line_ids[0]) if line_ids else None
    if line is None:
        return "Calm and neutral."
    label = str(line.get("delivery") or C.base_delivery()).casefold()
    return C.load_emotion_controls(None).get(label, {}).get("text") or C.clean_text(str(line["text"]))


class Report:
    def __init__(self):
        self.doc = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checks": {}, "problems": [], "notes": []}

    def check(self, name: str, ok: bool, **info):
        self.doc["checks"][name] = {"ok": bool(ok), **info}
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  {json.dumps(info, default=str)[:300]}" if info else ""),
              flush=True)
        if not ok:
            self.doc["problems"].append(name)


def compare_state(model_sd: dict, ckpt: dict, strip: str = "") -> dict:
    """Names/shapes of a module's state_dict vs a checkpoint dict (tensors may be mmap/meta)."""
    ck = {}
    for k, v in ckpt.items():
        if hasattr(v, "shape"):
            ck[k[len(strip):] if strip and k.startswith(strip) else k] = v
    missing = sorted(k for k in model_sd if k not in ck)
    unexpected = sorted(k for k in ck if k not in model_sd)
    mismatch = sorted(k for k in model_sd if k in ck and tuple(model_sd[k].shape) != tuple(ck[k].shape))
    loaded = [k for k in model_sd if k in ck and k not in mismatch]
    params = sum(int(model_sd[k].numel()) for k in loaded)
    dtypes = sorted({str(ck[k].dtype).replace("torch.", "") for k in loaded})
    return {"model_tensors": len(model_sd), "checkpoint_tensors": len(ck), "loaded": len(loaded),
            "missing": len(missing), "missing_all": missing, "unexpected": len(unexpected),
            "shape_mismatch": len(mismatch),
            "missing_examples": missing[:6], "unexpected_examples": unexpected[:6], "mismatch_examples": mismatch[:6],
            "loaded_params_m": round(params / 1e6, 2), "checkpoint_dtypes": dtypes}


def main() -> None:
    args = parse_args()
    missing = [str(p) for p in (C.CORPUS, C.SPEAKER_SCREEN, C.TAKES_SCRIPT) if not p.exists()]
    if missing:  # fail before the slow model checks, not after them
        raise SystemExit(f"{C.CHAR['display_name']}: missing {missing} (written by the corpus and delivery steps; run "
                         "them for this character first)")
    rep = Report()
    C.offline_env()
    blocked = C.net_guard()
    C.enter_index_root()
    rep.doc["env"] = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                    "HF_HOME", "MODELSCOPE_CACHE")}
    rep.doc["cwd"] = os.getcwd()
    t0 = time.perf_counter()
    commit0 = C.process_commit()

    # ------------------------------------------------------------------ 1. imports
    import numpy as np
    import torch
    import torchaudio
    import transformers

    rep.check("cuda_hidden", not torch.cuda.is_available(), device_count=torch.cuda.device_count())
    imported, failed = {}, {}
    targets = {
        "indextts.infer_v2_5": ("IndexTTS2", "QwenEmotion", "find_most_similar_cosine",
                                "apply_pronunciation_annotations"),
        "indextts.gpt.model_v2": ("UnifiedVoice",),
        "indextts.codec.models": ("EnhancedCodec",),
        "indextts.s2mel.modules.commons": ("MyModel", "load_checkpoint2"),
        "indextts.s2mel.modules.bigvgan.bigvgan": ("BigVGAN", "load_hparams_from_json"),
        "indextts.s2mel.modules.campplus.DTDNN": ("CAMPPlus",),
        "indextts.s2mel.modules.audio": ("mel_spectrogram",),
        "indextts.utils.front": ("TextNormalizer",),
        "indextts.utils.tokenizer": ("get_tokenizer", "lang_to_token"),
        "indextts.utils.ja_g2p": ("JapaneseG2PProcessor",),
        "indextts.utils.checkpoint": ("load_checkpoint",),
        "transformers": ("SeamlessM4TFeatureExtractor", "Wav2Vec2BertModel", "Wav2Vec2BertConfig", "AutoConfig",
                         "AutoTokenizer", "AutoModelForCausalLM"),
        "modelscope": ("AutoModelForCausalLM",),
    }
    import importlib

    mods = {}
    for module, names in targets.items():
        ts = time.perf_counter()
        try:
            mod = importlib.import_module(module)
            for name in names:
                getattr(mod, name)
            mods[module] = mod
            imported[module] = round(time.perf_counter() - ts, 1)
        except Exception as exc:  # noqa: BLE001
            failed[module] = f"{type(exc).__name__}: {exc}"
    import huggingface_hub

    rep.check("imports", not failed, seconds=imported, failed=failed)
    rep.doc["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "torchaudio": torchaudio.__version__, "transformers": transformers.__version__,
                           "huggingface_hub": huggingface_hub.__version__, "numpy": np.__version__,
                           "cuda_build": torch.version.cuda}
    hub_cache = Path(huggingface_hub.constants.HF_HUB_CACHE).resolve()
    rep.check("hf_hub_cache_inside_install", hub_cache == C.AUX_DIR.resolve(), hf_hub_cache=str(hub_cache),
              hf_home=str(Path(huggingface_hub.constants.HF_HOME).resolve()),
              offline=bool(huggingface_hub.constants.HF_HUB_OFFLINE))

    # ------------------------------------------------------------------ 2. paths
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(C.CFG_PATH))
    paths = C.checkpoint_paths()
    rep.check("config_version", str(cfg.get("version")) == "2.5", version=str(cfg.get("version")))
    agree = {k: str(MODEL) for k, MODEL in (("gpt", C.MODEL_DIR / cfg.gpt_checkpoint),
                                            ("s2mel", C.MODEL_DIR / cfg.s2mel_checkpoint),
                                            ("w2v_stat", C.MODEL_DIR / cfg.w2v_stat),
                                            ("emo_matrix", C.MODEL_DIR / cfg.emo_matrix),
                                            ("spk_matrix", C.MODEL_DIR / cfg.spk_matrix))
             if Path(MODEL) != paths[k]}
    rep.check("paths_match_omegaconf", not agree, differ=agree)
    rep.check("aux_downloader_not_triggered", not C.downloader_would_run(paths), reasons=C.downloader_would_run(paths))
    verified = C.verified_files()
    resolved, bad = {}, {}
    for key, path in paths.items():
        inside = str(path.resolve()).casefold().startswith(str(C.INDEX_ROOT).casefold())
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
        if not files or not inside:
            bad[key] = f"missing or outside install: {path}"
            continue
        total = 0
        for f in files:
            total += f.stat().st_size
            entry = verified.get(str(f).casefold())
            if entry is None:
                if f.name not in ("pinyin.vocab",):
                    bad[f"{key}:{f.name}"] = "not in the fetch log"
                continue
            if f.stat().st_size != entry["size"]:
                bad[f"{key}:{f.name}"] = f"size {f.stat().st_size} != {entry['size']}"
            elif not args.no_rehash and C.sha256_file(f) != entry["sha256"]:
                bad[f"{key}:{f.name}"] = "sha256 differs from the fetch log"
        resolved[key] = {"path": str(path), "files": len(files), "bytes": total}
    rep.check("checkpoint_files_resolve_and_verify", not bad, rehashed=not args.no_rehash, problems=bad)
    rep.doc["resolved_paths"] = resolved

    # ------------------------------------------------------------------ 3. weights vs modules (meta device)
    weights = {}
    if args.skip_weights:
        rep.doc["notes"].append("--skip-weights: checkpoint-vs-module audits not run (character-independent; see a "
                                "full smoke report); CAMPPlus and the emotion tables are still loaded")

    import gc

    def audit(name: str, build, load, strip: str = "", allow_missing: tuple[str, ...] = (), note: str = ""):
        if args.skip_weights:
            return
        ts = time.perf_counter()
        module = ckpt = sd = None
        try:
            with torch.device("meta"):
                module = build()
            sd = module.state_dict()
            ckpt = load()
            info = compare_state(sd, ckpt, strip)
            real_missing = [k for k in info.pop("missing_all") if k not in allow_missing]
            ok = not real_missing and info["shape_mismatch"] == 0
            info["seconds"] = round(time.perf_counter() - ts, 1)
            info["commit_now"] = C.process_commit()
            if allow_missing and info["missing"]:
                info["allowed_missing"] = list(allow_missing)
                if note:
                    rep.doc["notes"].append(note)
            weights[name] = info
            rep.check(f"weights_{name}", ok, **{k: v for k, v in info.items() if not k.endswith("examples") or v})
        except Exception as exc:  # noqa: BLE001
            weights[name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
            rep.check(f"weights_{name}", False, error=weights[name]["error"])
        finally:
            del module, ckpt, sd
            gc.collect()  # drop each memory-mapped checkpoint before the next one (Windows commits COW maps)

    def tload(path):
        # the upstream loaders call torch.load(..., map_location="cpu") with torch 2.8's default weights_only=True;
        # mmap=True reads only the tensor metadata here (the data pages are never touched)
        return torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)

    UnifiedVoice = mods["indextts.gpt.model_v2"].UnifiedVoice
    gpt_ckpt = {}

    def gpt_load():
        sd = tload(paths["gpt"])
        gpt_ckpt["top_keys"] = list(sd.keys())[:3] if "model" in sd else None
        return sd["model"] if "model" in sd else sd

    audit("gpt", lambda: UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus"), gpt_load)

    EnhancedCodec = mods["indextts.codec.models"].EnhancedCodec
    audit("semantic_codec", lambda: EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec),
          lambda: tload(paths["semantic_codec"])["model"])

    MyModel = mods["indextts.s2mel.modules.commons"].MyModel
    s2 = {}

    def s2mel_load():
        state = tload(paths["s2mel"])
        s2["state_keys"] = sorted(state.keys())
        net = state["net"]
        s2["net_keys"] = sorted(net.keys())
        flat = {}
        for key, sub in net.items():
            for k, v in sub.items():
                flat[f"models.{key}.{k[len('module.'):] if k.startswith('module.') else k}"] = v
        return flat

    audit("s2mel", lambda: MyModel(cfg.s2mel), s2mel_load)
    if "s2mel" in weights:  # always, unless --skip-weights
        weights["s2mel"].update({"file_top_keys": s2.get("state_keys"), "net_modules": s2.get("net_keys")})

    bv = mods["indextts.s2mel.modules.bigvgan.bigvgan"]
    h = bv.load_hparams_from_json(str(paths["bigvgan_config"]))
    audit("bigvgan", lambda: bv.BigVGAN(h, use_cuda_kernel=False), lambda: tload(paths["bigvgan_weights"])["generator"])

    from safetensors import safe_open

    def safetensors_header(path):
        out = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                out[k] = torch.empty(tuple(sl.get_shape()), device="meta", dtype=_st_dtype(sl.get_dtype()))
        return out

    tf = mods["transformers"]
    w2v_cfg = tf.Wav2Vec2BertConfig.from_pretrained(str(paths["w2v_bert_dir"]), local_files_only=True)
    audit("w2v_bert", lambda: tf.Wav2Vec2BertModel(w2v_cfg), lambda: safetensors_header(paths["w2v_bert_weights"]))
    if "w2v_bert" in weights:  # always, unless --skip-weights
        weights["w2v_bert"]["hidden_layers"] = w2v_cfg.num_hidden_layers

    qcfg = tf.AutoConfig.from_pretrained(str(paths["qwen_emo"]), local_files_only=True)
    audit("qwen_emo", lambda: tf.AutoModelForCausalLM.from_config(qcfg),
          lambda: safetensors_header(paths["qwen_emo"] / "model.safetensors"),
          allow_missing=("lm_head.weight",) if getattr(qcfg, "tie_word_embeddings", False) else (),
          note="qwen_emo: lm_head.weight is tied to the input embeddings (tie_word_embeddings=true), so it is "
               "absent from model.safetensors by design")

    CAMPPlus = mods["indextts.s2mel.modules.campplus.DTDNN"].CAMPPlus
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    try:
        result = campplus.load_state_dict(torch.load(str(paths["campplus"]), map_location="cpu"))
        rep.check("weights_campplus_real_load", True, result=str(result))
    except Exception as exc:  # noqa: BLE001
        rep.check("weights_campplus_real_load", False, error=f"{type(exc).__name__}: {exc}")
    campplus.eval()

    stat = torch.load(str(paths["w2v_stat"]), map_location="cpu")
    emo = torch.load(str(paths["emo_matrix"]), map_location="cpu")
    spk = torch.load(str(paths["spk_matrix"]), map_location="cpu")
    emo_num = list(cfg.emo_num)
    rep.check("emotion_tables", emo.shape[0] == sum(emo_num) == spk.shape[0] and len(emo_num) == 8,
              emo_matrix=list(emo.shape), spk_matrix=list(spk.shape), emo_num=emo_num,
              w2v_stat={k: list(v.shape) for k, v in stat.items()})
    rep.doc["weights"] = weights

    # ------------------------------------------------------------------ 4a. reference front end (1-2 TRAIN clips)
    corpus = {r["id"]: r for r in C.load_corpus()}
    screen = C.load_speaker_screen()
    clips = [c.strip() for c in args.clips.split(",") if c.strip()][:2]
    embeddings = {}
    fe = tf.SeamlessM4TFeatureExtractor.from_pretrained(str(paths["w2v_bert_dir"]), local_files_only=True)
    mel_fn_args = {"n_fft": cfg.s2mel.preprocess_params.spect_params.n_fft,
                   "win_size": cfg.s2mel.preprocess_params.spect_params.win_length,
                   "hop_size": cfg.s2mel.preprocess_params.spect_params.hop_length,
                   "num_mels": cfg.s2mel.preprocess_params.spect_params.n_mels,
                   "sampling_rate": cfg.s2mel.preprocess_params.sr, "fmin": 0, "fmax": None, "center": False}
    mel_spectrogram = mods["indextts.s2mel.modules.audio"].mel_spectrogram
    import librosa

    for cid in clips:
        row = corpus.get(cid)
        ok = (row is not None and row["split"] == "train" and row["kind"] == "speech"
              and cid not in screen["confirmed"] and cid not in screen["uncertain"])
        if not ok:
            rep.check(f"reference_{cid}", False, error="not a usable TRAIN speech clip (refused; never test clips)")
            continue
        try:
            ref_path, meta = C.prepare_reference(row, "original", True)
            audio, sr = librosa.load(str(ref_path))  # exactly _load_and_cut_audio (librosa default 22050)
            audio = torch.tensor(audio).unsqueeze(0)[:, : int(15 * sr)]
            a22 = torchaudio.transforms.Resample(sr, 22050)(audio)
            a16 = torchaudio.transforms.Resample(sr, 16000)(audio)
            feats = fe(a16, sampling_rate=16000, return_tensors="pt")
            mel = mel_spectrogram(a22.float(), **mel_fn_args)
            fb = torchaudio.compliance.kaldi.fbank(a16, num_mel_bins=80, dither=0, sample_frequency=16000)
            fb = fb - fb.mean(dim=0, keepdim=True)
            with torch.no_grad():
                style = campplus(fb.unsqueeze(0))
            embeddings[cid] = style[0]
            rep.check(f"reference_{cid}", tuple(style.shape) == (1, 192) and mel.shape[1] == 80,
                      split=row["split"], source=str(C.source_audio(row, "original")), ref=str(ref_path),
                      native_sr=meta.get("rate"), trimmed_s=meta.get("duration_s"), librosa_sr=sr,
                      w2v_input_features=list(feats["input_features"].shape), ref_mel=list(mel.shape),
                      fbank=list(fb.shape), campplus_style=list(style.shape))
        except Exception as exc:  # noqa: BLE001
            rep.check(f"reference_{cid}", False, error=f"{type(exc).__name__}: {exc}",
                      trace=traceback.format_exc(limit=3))
    if len(embeddings) == 2:
        a, b = list(embeddings.values())
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        rep.doc["campplus_cosine_between_clips"] = round(cos, 4)
        rep.doc["notes"].append(f"CAMPPlus cosine between {', '.join(embeddings)} (same speaker, different "
                                f"delivery): {cos:.3f}")

    # ------------------------------------------------------------------ 4b. text front end (2 script lines)
    script = {ln["id"]: ln for ln in json.loads(C.TAKES_SCRIPT.read_text(encoding="utf-8"))["lines"]}
    try:
        tok = mods["indextts.utils.tokenizer"].get_tokenizer(multilingual=True, model_dir=str(C.MODEL_DIR))
        tn = mods["indextts.utils.front"].TextNormalizer(enable_glossary=True)
        tn.load()
        mods["indextts.utils.ja_g2p"].JapaneseG2PProcessor(g2p_ratio=0)
        annotate = mods["indextts.infer_v2_5"].apply_pronunciation_annotations
        texts = {}
        for lid in [x.strip() for x in args.lines.split(",") if x.strip()][:2]:
            raw = C.clean_text(script[lid]["text"])
            t = tn.clean_pattern.sub(lambda x: tn.char_rep_map[x.group()], raw)
            t = tn.normalize(t).lower()
            t = annotate(t)
            ids = tok.encode("<|en|> " + t, allowed_special="all")
            texts[lid] = {"input": raw, "normalized": t, "tokens": len(ids)}
        demo = annotate("<minute|M IH1 . N AH0 T>".lower())
        rep.check("text_front_end", all(v["tokens"] > 0 for v in texts.values()) and "SPECIAL_TOKEN_1" in demo,
                  lines=texts, annotation_demo=demo, lang_token_en=mods["indextts.utils.tokenizer"].lang_to_token("en"))
    except Exception as exc:  # noqa: BLE001
        rep.check("text_front_end", False, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc(limit=3))
    try:
        qtok = tf.AutoTokenizer.from_pretrained(str(paths["qwen_emo"]), local_files_only=True)
        probe = emotion_probe_text(script, [x.strip() for x in args.lines.split(",") if x.strip()][:2])
        prompt = qtok.apply_chat_template([{"role": "system", "content": "文本情感分类"},
                                           {"role": "user", "content": probe}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rep.check("qwen_emo_tokenizer", len(qtok(prompt)["input_ids"]) > 0, prompt_tokens=len(qtok(prompt)["input_ids"]))
    except (Exception, SystemExit) as exc:  # noqa: BLE001  (SystemExit: an invalid emotion controls file)
        rep.check("qwen_emo_tokenizer", False, error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ wrap up
    stray = [str(p) for p in (C.INDEX_ROOT / ".hf_home", C.INDEX_ROOT / ".modelscope_cache") if p.exists()]
    stray += [str(p) for p in C.AUX_DIR.glob("models--*")]
    rep.check("no_network_and_no_new_caches", not blocked and not stray, blocked=list(blocked), new_cache_dirs=stray)
    rep.doc["process_commit"] = {"start": commit0, "end": C.process_commit()}
    rep.doc["seconds"] = round(time.perf_counter() - t0, 1)
    rep.doc["vram_estimate_gb"] = {"render_default": C.vram_estimate(), "with_qwen_emo": C.vram_estimate(qwen_emo=True)}
    rep.doc["remaining_on_gpu"] = [
        "full IndexTTS2(...) construction with real weights (torch.load of gpt/s2mel/codec, w2v-bert from_pretrained, "
        "BigVGAN remove_weight_norm) and the move to cuda:0 under the allocator cap",
        "first inference per mode (spk_ref, id, id_emo_ref, id_emo_vec, id_emo_text) on 1-2 lines: audio sanity, "
        "no runaway lengths (max_mel_tokens warnings), stop-token early cut-offs",
        "bf16 GPT on the RTX 5090 (sm_120) with torch 2.8.0+cu128; BigVGAN torch activation path (CUDA kernel off)",
        "measured peak VRAM / host commit vs the estimates (manifest gpu.peak_reserved_gb, gpu.host)",
        "QwenEmotion on English emotion descriptions (vectors recorded per take) and its VRAM",
        "seed determinism of repeated takes and the 22.05 kHz -> 48 kHz output path",
        "listening + eval harness scoring (identity, WER, prosody, DNSMOS) of the donor takes; Seed-VC conversion",
    ]
    rep.doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rep.doc["ok"] = not rep.doc["problems"]
    C.atomic_write_json(args.report, rep.doc)
    print(json.dumps({"ok": rep.doc["ok"], "problems": rep.doc["problems"], "seconds": rep.doc["seconds"],
                      "process_commit": rep.doc["process_commit"], "report": str(args.report)}, indent=2))


def _st_dtype(name: str):
    import torch

    return {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
            "I32": torch.int32, "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
            "F64": torch.float64}.get(name, torch.float32)


if __name__ == "__main__":
    main()
