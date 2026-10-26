"""Evaluate a voice-lab character's takes: speaker similarity, intelligibility, prosody/delivery and artifacts.

The character is VOICE_LAB_CHARACTER=<slug> (default baodur; voice-lab/characters/<slug>.json). <lab> below is its
lab folder (BaoDurLab, JoleeLab, ...): corpus, delivery files, eval runs and caches all live there, and the
character's evaluation content (impostor voices, other-speaker thresholds, name probes, pitch range, self-test
clips) is <lab>/eval/eval_character.json (evalchar.py). Every metric step inherits the variable.

Usage (any Python works; without numpy the driver re-launches itself under the seed-vc venv):
  python run_eval.py --manifest <lab>/takes/<engine>/<run>/manifest.json [--manifest ...]
                     --run <name> [--audio original|restored] [--device cpu|cuda] [--asr-device cuda]
  python run_eval.py --takes-root <lab>/takes [--skip REGEX] --run <name> ...
                     (every <root>/<engine>/<run>/manifest.json; --skip drops engine/run labels matching REGEX)
  set VOICE_LAB_CHARACTER=jolee  (before any of the above) to score Jolee Bindo's takes against JoleeLab

Inputs: take manifests in the shared format ({"takes": [{take_id, engine, line_id, text, delivery, wav,
reference_id, seed, params}]}, WAVs relative to the manifest or absolute), the takes script
(<lab>/delivery/takes_script.json, for held-out ids) and the delivery map
(<lab>/delivery/delivery_map.json, for per-delivery targets). Both are optional. Optional take/manifest
fields the harness reads: params.model_version / conditioning_encoders (speaker-model coupling),
trained_on_ids / training_data (provenance.py), real_id (train/val diagnostic lines, see below).

Steps, each in the venv that has its dependency, all CPU unless --device cuda:
  sv_eres2net     GPT-SoVITS venv  ERes2NetV2 (GPT-SoVITS sv checkpoint) cosine vs the character's train centroid
  sv_resemblyzer  seed-vc venv     Resemblyzer cosine vs centroid (takes level-normalised first)
  sv_wavlm        seed-vc venv     WavLM-base-plus-SV (Vox Studio's local copy), the model no engine conditions on;
                                   also writes layer-6 features the aggregator tests as a delivery signal
  asr             VoxCPM2 python   faster-whisper WER vs intended text, plus a prompt-free pass on proper nouns
  prosody         seed-vc venv     pyin F0 median/IQR/range, rate, pauses, loudness (--workers in parallel)
  artifacts       seed-vc venv     clipping (absolute and renormalised), one-sample clicks, silences, hiss, cut endings
  dnsmos          GPT-SoVITS venv  DNSMOS P.835 SIG/BAK/OVRL + P.808 (naturalness / vocoder quality), CPU onnxruntime
  utmos           eval-extra venv  optional, not in the default --metrics: UTMOSv2 fold-0 naturalness MOS (sarulab
                                   domain, 5 repetitions); --extra-real all|train|none picks the real clips scored
  emotion         eval-extra venv  optional, not in the default --metrics: emotion2vec+ base 9-way classes + utterance
                                   embedding, compared with the delivery map's reference clips (--emotion-refs)
  aggregate       (this python)    gated ranking -> <lab>/eval/<run>/scorecard.json + scorecard.md
utmos and emotion (voice-lab/eval-extra, its own uv venv and local weights) are separate scorecard columns; they never
change tiers or the composite. Add them with --metrics <defaults>,utmos,emotion. The first utmos run scores every
real clip (Bao-Dur ~306; 5 repetitions each, ~1 min per clip on 4 CPU threads, so hours on CPU: prefer --device cuda
when the GPU is free, or --extra-real train|none); emotion embeds the delivery references (Bao-Dur ~73, ~1 s each on
CPU). Both are cached.
The first run of each model step embeds/scores the character's real clips once (cached by SHA-256 in
<lab>/eval/_cache; Bao-Dur: sv_wavlm ~340 clips, ~12 min on CPU, ~1 min on cuda; dnsmos ~306 clips, ~5 min CPU).
--audio picks the primary real-clip variant (the user chose original game audio for training and references, for
every character); --sv-variants both additionally scores the speaker models against the restored real clips when a
character has them (reported, never ranked). Held-out TEST clips are used only as ground truth: never in a centroid
or train distribution.
Diagnostic lines: a script line with "source": "train"|"val" and "real_id" (a clip of that split) pairs its
takes with that real recording, so the scorecard can report the train-vs-held-out gap per run (memorisation);
conditioning on that very clip is then a leak for the line.

Leak checks (hard failures that rank below every other take of the line):
- conditioning: a take whose reference_id or params name a test clip by exact id, or point at an audio file
  (any key) whose SHA-256 equals a test clip (original or restored copy) or whose file name carries its id;
  on a held-out line, the same for train clips that repeat that very line (evalcommon.near_duplicates_of_test);
  free-text fields (why/note/reason/text...) only produce notes;
- the take's own audio a byte copy of a test clip;
- training: the take's model trained on that held-out test clip or on a near-duplicate of it (provenance.py:
  contract fields, engine self-reports such as VoxCPM's params.adapter_trained_on_heldout, or the engines'
  training lists); Seed-VC takes inherit their source take's findings (params.source_manifest/source_take_id).
  Takes whose training data cannot be resolved are marked "provenance unverified" on held-out lines.
Real-clip results are cached in <lab>/eval/_cache, so later runs only process new takes.

A take a step could not score because of a resource error (memory, CUDA, file in use; retried once) is
"unscored": it is excluded from ranking, the step exits 3 and the run exits 1; re-run that step alone with
--metrics <name>. Takes missing from a metric (manifests changed between partial re-runs) are unscored too.

With --device cuda each torch step caps its own memory (ERes2NetV2 2 GB, Resemblyzer 1 GB, WavLM 2 GB) and the
ASR step runs CTranslate2 float16 (~4-5 GB for large-v3; CTranslate2 has no hard cap, its cache is bounded with
CT2_CUDA_CACHING_ALLOCATOR_CONFIG); prosody, artifacts and dnsmos always run on CPU. GPU guard: each cuda step
starts only if nvidia-smi reports at least its need (ERes2NetV2 2, Resemblyzer 1, WavLM 2, ASR 5 GB; override
with --gpu-need-gb) + --gpu-reserve-gb (8 GB) free, and while it runs a watchdog (0.5 s polls) kills that step's
own process (never anything else) if free memory drops below --gpu-reserve-gb, keeping >= 8 GB for Substance
Painter. A refused or killed step is reported in run_status.json and the scorecard; re-run it alone later with
--metrics <name> (partial re-runs keep the other steps' status).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE_LAB = Path(os.environ.get("BAODUR_VOICE_LAB", r"C:\Users\NewAdmin\Documents\GDeveloper\External\voice-lab"))
LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local"))
PYTHONS = {
    "gptsovits": VOICE_LAB / "GPT-SoVITS" / ".venv" / "Scripts" / "python.exe",
    "seedvc": VOICE_LAB / "seed-vc" / ".venv" / "Scripts" / "python.exe",
    "voxcpm2": LOCALAPPDATA / "VoxStudio" / "engines" / "voxcpm2" / ".venv" / "Scripts" / "python.exe",
    "evalextra": VOICE_LAB / "eval-extra" / ".venv" / "Scripts" / "python.exe",
}
METRICS = [  # name, venv, script
    ("sv_eres2net", "gptsovits", "metric_sv_eres2net.py"),
    ("sv_resemblyzer", "seedvc", "metric_sv_resemblyzer.py"),
    ("sv_wavlm", "seedvc", "metric_sv_wavlm.py"),
    ("asr", "voxcpm2", "metric_asr.py"),
    ("prosody", "seedvc", "metric_prosody.py"),
    ("artifacts", "seedvc", "metric_artifacts.py"),
    ("dnsmos", "gptsovits", "metric_dnsmos.py"),
]
EXTRA_METRICS = [  # opt-in (not in the default --metrics): separate scorecard columns, never in the composite
    ("utmos", "evalextra", "metric_utmos.py"),
    ("emotion", "evalextra", "metric_emotion.py"),
]
ALL_METRICS = METRICS + EXTRA_METRICS
STEP_GPU_GB = {"sv_eres2net": 2.0, "sv_resemblyzer": 1.0, "sv_wavlm": 2.0, "asr": 5.0,  # per-step need on cuda
               "utmos": 3.0, "emotion": 1.5}
REQUIRED_TAKE_KEYS = ("take_id", "engine", "line_id", "wav")
CT2_CACHE_CONFIG = "4,3,12,209715200"  # CTranslate2 CUB caching allocator: keep at most 200 MB of freed blocks

try:
    sys.path.insert(0, str(HERE))
    import evalcommon as ec  # needs numpy + soundfile
    import evalchar
    import provenance as pv
except ImportError:
    if os.environ.get("BAODUR_EVAL_REEXEC"):
        raise
    env = dict(os.environ, BAODUR_EVAL_REEXEC="1")
    sys.exit(subprocess.call([str(PYTHONS["seedvc"]), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


# Keys whose values may legitimately name a held-out line or clip without being conditioning (exact names,
# plus anything mentioning held-out), free-text keys (reported as notes only), and conditioning-like keys
# (any other string under them that contains a clip id counts as conditioning).
BOOKKEEPING_KEY = re.compile(r"^(line_id|take_id|source_take_id|run|adapter|engine|source_engine)$|held_?out", re.I)
FREE_TEXT_KEY = re.compile(r"(why|notes?|reasons?|comments?|description|text|transcripts?|captions?|summary|"
                           r"instructions?|labels?|warnings?)$", re.I)
REF_KEY = re.compile(r"ref|prompt|speaker|spk|aux|voice|cond|timbre|style|src|source|input|audio|wav|clip|target|"
                     r"identity", re.I)
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def iter_strings(value, key: str = ""):
    """(innermost key, string) for every string in a nested params structure."""
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from iter_strings(v, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_strings(v, key)


class LeakChecker:
    """Finds held-out test clips (and, per held-out line, train clips repeating that line) among a take's
    conditioning inputs: exact clip ids, audio paths (by SHA-256 and by the id in the file name) and ids in
    other strings under conditioning-like keys. Mentions in free text are returned as notes."""

    def __init__(self, corpus: dict[str, dict], line_ids: list[str], paired_ids: set[str] | None = None):
        self.corpus = corpus
        self.tests = {i for i, r in corpus.items() if r["split"] == "test"}
        self.near = {t: [e["id"] for e in v] for t, v in ec.near_duplicates_of_test(corpus).items()}
        ids = sorted(corpus, key=len, reverse=True)
        self.canon = {i.lower(): i for i in corpus}
        self.id_pattern = re.compile(r"(?<![0-9a-z])(" + "|".join(map(re.escape, ids)) + r")(?![0-9])", re.I)
        # names that embed a clip id without being conditioning (script line ids such as heldout_<test id>);
        # a bare clip id is never stripped, since a take may be conditioned on exactly that clip
        self.line_ids = sorted({l for l in line_ids if self._composite(l)}, key=len, reverse=True)
        # train/val clips that diagnostic lines are paired with: conditioning on them is a leak on that line
        watched = set(self.tests) | {c for v in self.near.values() for c in v} | set(paired_ids or ())
        self.sha_to_clip: dict[str, str] = {}
        for clip in watched:
            row = corpus[clip]
            if row.get("sha256"):
                self.sha_to_clip[row["sha256"].lower()] = clip
            restored = ec.RESTORED_DIR / f"{clip}.wav"
            if restored.exists():
                self.sha_to_clip[ec.file_sha256(restored)] = clip
        self._file_sha: dict[str, str | None] = {}

    def _composite(self, name: str) -> bool:
        """True for a name that embeds a clip id but is not itself one (e.g. heldout_<test id>-s0)."""
        return bool(name) and name.lower() not in self.canon and bool(self.id_pattern.search(name))

    def sha_of(self, path: Path) -> str | None:
        key = str(path)
        if key not in self._file_sha:
            try:
                self._file_sha[key] = ec.file_sha256(path) if path.is_file() else None
            except OSError:
                self._file_sha[key] = None
        return self._file_sha[key]

    def _audio_path(self, text: str, base: Path) -> Path | None:
        candidate = text.strip().strip('"')
        if len(candidate) > 400 or Path(candidate).suffix.lower() not in AUDIO_SUFFIXES:
            return None
        path = Path(candidate)
        return path if path.is_absolute() else base / path

    def _ids(self, text: str, strip: list[str]) -> set[str]:
        cleaned = text
        for token in strip:
            cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.I)
        return {self.canon[m.group(1).lower()] for m in self.id_pattern.finditer(cleaned)}

    def check(self, take: dict, base: Path, heldout: str | None,
              paired: str | None = None) -> tuple[list[str], list[str]]:
        strip = sorted(set(self.line_ids + [s for s in (str(take.get("line_id") or ""), str(take.get("take_id") or ""))
                                            if self._composite(s)]), key=len, reverse=True)
        named: set[str] = set()
        hashed: set[str] = set()
        mentioned: dict[str, set[str]] = {}
        fields = {"reference_id": take.get("reference_id") or "", "params": take.get("params") or {}}
        for key, text in iter_strings(fields):
            value = text.strip().strip('"')
            if not value:
                continue
            audio = self._audio_path(value, base)
            if audio is not None:  # any key: hash it (a renamed copy of a test clip is caught here)
                sha = self.sha_of(audio)
                if sha and sha in self.sha_to_clip:
                    hashed.add(self.sha_to_clip[sha])
            if BOOKKEEPING_KEY.search(key):
                continue
            if FREE_TEXT_KEY.search(key):
                for clip in self._ids(value, strip):
                    mentioned.setdefault(clip, set()).add(key)
                continue
            if value.lower() in self.canon:
                named.add(self.canon[value.lower()])
            elif audio is not None:
                named |= self._ids(audio.stem, strip)
            elif REF_KEY.search(key):
                named |= self._ids(value, strip)
            else:
                for clip in self._ids(value, strip):
                    mentioned.setdefault(clip, set()).add(key)
        violations, notes = [], []
        for clip in sorted((named | hashed) & self.tests):
            how = "by id" if clip in named else "by audio hash"
            violations.append(f"held-out test clip {clip} used as conditioning ({how})")
        near = set(self.near.get(heldout, [])) if heldout else set()
        for clip in sorted((named | hashed) & near):
            violations.append(f"conditioning clip {clip} repeats held-out line {heldout} (near-duplicate): leak")
        own = {paired} if paired and paired not in self.tests else set()
        for clip in sorted((named | hashed) & own):
            violations.append(f"conditioning clip {clip} is this diagnostic line's own real recording: leak")
        for clip, keys in sorted(mentioned.items()):
            if clip in self.tests or clip in near or clip in own:
                notes.append(f"params {', '.join(sorted(keys))} mention {clip} (free text, not treated as conditioning)")
        return violations, notes


def discover_manifests(roots: list[str], skip: str | None) -> list[str]:
    """<root>/<engine>/<run>/manifest.json under each root, minus engine/run labels matching skip."""
    pattern = re.compile(skip, re.I) if skip else None
    found = []
    for root in roots:
        for manifest in sorted(Path(root).glob("*/*/manifest.json")):
            label = f"{manifest.parent.parent.name}/{manifest.parent.name}"
            if pattern and pattern.search(label):
                continue
            found.append(str(manifest))
    return found


class TakeAuditor:
    """Conditioning leaks + training provenance for a take, following Seed-VC takes to their source take."""

    def __init__(self, leaks: LeakChecker, corpus: dict[str, dict]):
        self.leaks = leaks
        self.prov = pv.Provenance(corpus, leaks.id_pattern)
        self._docs: dict[str, dict | None] = {}

    def doc(self, path: Path) -> dict | None:
        key = str(path)
        if key not in self._docs:
            try:
                self._docs[key] = ec.read_json(path) if path.exists() else None
            except (OSError, ValueError):
                self._docs[key] = None
        return self._docs[key]

    def audit(self, raw: dict, manifest: Path, doc: dict, heldout: str | None, depth: int = 0,
              paired: str | None = None):
        violations, notes = self.leaks.check(raw, manifest.parent, heldout, paired)
        prov = self.prov.resolve(raw, doc, manifest)
        violations += pv.leak_findings(prov, heldout, self.leaks.near)
        source = prov.get("source_take")
        if source and depth < 3:
            src_path = Path(source["manifest"])
            if not src_path.is_absolute():
                src_path = (manifest.parent / src_path).resolve()
            src_doc = self.doc(src_path)
            src_take = next((t for t in (src_doc or {}).get("takes", []) if str(t.get("take_id")) == source["take_id"]),
                            None)
            if src_take is None:
                notes.append(f"source take {source['take_id']} not found in {src_path}; its training data is unverified")
                prov["status"] = "unverified"
            else:
                sv, snotes, sprov = self.audit(src_take, src_path, src_doc, heldout, depth + 1, paired)
                violations += [f"via source take {source['take_id']}: {v}" for v in sv]
                notes += [f"via source take {source['take_id']}: {n}" for n in snotes]
                prov["source_provenance"] = {k: sprov.get(k) for k in ("status", "sources", "notes")}
                if sprov["status"] == "unverified":
                    prov["status"] = "unverified"
                prov["trained_ids"] = sorted(set(prov["trained_ids"]) | set(sprov["trained_ids"]))
        return violations, notes, prov


def build_takes(args, run_dir: Path) -> dict:
    corpus = ec.corpus_index()
    warnings: list[str] = []

    lines: dict[str, dict] = {}
    script_path = Path(args.script) if args.script else None
    if script_path and script_path.exists():
        for line in ec.read_json(script_path).get("lines", []):
            lines[str(line["id"])] = line
    elif script_path:
        warnings.append(f"takes script not found: {script_path} (held-out ids come from the takes themselves)")
    paired_ids = {str(l["real_id"]) for l in lines.values()
                  if l.get("source") in ("train", "val") and l.get("real_id") and str(l["real_id"]) in corpus}
    leaks = LeakChecker(corpus, list(lines), paired_ids)
    auditor = TakeAuditor(leaks, corpus)
    for line in lines.values():
        if line.get("source") == "heldout" and line.get("heldout_id") in leaks.near:
            warnings.append(f"script line {line['id']}: train/val clips repeat held-out {line['heldout_id']} "
                            f"({', '.join(leaks.near[line['heldout_id']])}); takes from models trained on them are "
                            "leaks on this line")

    delivery_path = Path(args.delivery_map) if args.delivery_map else None
    if delivery_path and not delivery_path.exists():
        warnings.append(f"delivery map not found: {delivery_path} (prosody falls back to {ec.NAME}'s global distribution)")
        delivery_path = None

    known_other, _src = ec.known_other_speakers()
    takes: list[dict] = []
    seen: dict[str, int] = {}
    manifests = []
    provenance_by_run: dict[str, dict] = {}
    for manifest_arg in args.manifest:
        manifest = Path(manifest_arg).resolve()
        if not manifest.exists():
            raise SystemExit(f"manifest not found: {manifest}")
        manifests.append(str(manifest))
        doc = auditor.doc(manifest) or {}
        run_label = f"{manifest.parent.parent.name}/{manifest.parent.name}"
        for index, raw in enumerate(doc.get("takes", [])):
            missing = [k for k in REQUIRED_TAKE_KEYS if not raw.get(k)]
            if missing:
                warnings.append(f"{manifest}: take #{index} lacks {missing}; skipped")
                continue
            wav = Path(raw["wav"])
            if not wav.is_absolute():
                wav = (manifest.parent / wav).resolve()
            if not wav.exists():
                warnings.append(f"{manifest}: take {raw['take_id']} wav missing ({wav}); skipped")
                continue
            params = raw.get("params") or {}
            line_id = str(raw["line_id"])
            line = lines.get(line_id, {})
            text = (raw.get("text") or line.get("text") or "").strip()
            if not text:
                warnings.append(f"{run_label}/{raw['take_id']}: no text in take or script; WER and rate will be empty")
            delivery = (raw.get("delivery") or line.get("delivery") or "natural").strip()
            heldout = (raw.get("heldout_id") or params.get("heldout_id")
                       or (line.get("heldout_id") if line.get("source") == "heldout" else None))
            if not heldout and line_id in corpus and corpus[line_id]["split"] == "test":
                heldout = line_id
            real_id = None if heldout else (raw.get("real_id") or params.get("real_id")
                                            or (line.get("real_id") if line.get("source") in ("train", "val") else None))
            if real_id:
                real_id = str(real_id)
                if real_id not in corpus:
                    warnings.append(f"{run_label}/{raw['take_id']}: real_id {real_id} is not in the corpus; ignored")
                    real_id = None
                elif corpus[real_id]["split"] == "test":  # a test clip makes it a held-out line, with every leak rule
                    heldout, real_id = real_id, None
                elif text and ec.norm_words(text) != ec.norm_words(corpus[real_id]["text"]):
                    warnings.append(f"{run_label}/{raw['take_id']}: text differs from paired clip {real_id}'s transcript")
            if heldout and heldout not in corpus:
                warnings.append(f"{run_label}/{raw['take_id']}: heldout_id {heldout} is not in the corpus; ignored")
                heldout = None
            if heldout and corpus[heldout]["split"] != "test":
                warnings.append(f"{run_label}/{raw['take_id']}: heldout_id {heldout} is a {corpus[heldout]['split']} clip, not a test clip")
            if heldout and text and ec.norm_words(text) != ec.norm_words(corpus[heldout]["text"]):
                warnings.append(f"{run_label}/{raw['take_id']}: text differs from held-out clip {heldout}'s transcript")
            paired = heldout or real_id
            violations, notes, prov = auditor.audit(raw, manifest, doc, heldout, paired=paired)
            wav_sha = leaks.sha_of(wav)
            copied = leaks.sha_to_clip.get(wav_sha or "")
            if copied in leaks.tests:
                violations.append(f"take audio is a byte copy of held-out test clip {copied}: leak")
            if heldout and prov["status"] == "unverified":
                notes.append("training data unverified: this held-out line may have been heard in training "
                             + ("; ".join(prov.get("notes") or []) or "(no provenance fields or known engine layout)"))
            summary = provenance_by_run.setdefault(run_label, {"statuses": set(), "sources": set(), "trained": set(),
                                                               "notes": set(), "self_report_heldout": set()})
            summary["statuses"].add(prov["status"])
            summary["sources"].update(prov.get("sources") or [])
            summary["trained"].update(prov.get("trained_ids") or [])
            summary["notes"].update(prov.get("notes") or [])
            summary["self_report_heldout"].update(str(i) for i in (prov.get("self_report") or {}).get(
                "heldout_clips_in_its_training") or [])
            key = f"{run_label}/{raw['take_id']}"
            if key in seen:
                seen[key] += 1
                key = f"{key}#{seen[key]}"
            else:
                seen[key] = 1
            takes.append({
                "key": key, "take_id": str(raw["take_id"]), "engine": str(raw["engine"]), "run_label": run_label,
                "manifest": str(manifest), "line_id": line_id, "text": text, "delivery": delivery,
                "wav": str(wav), "wav_sha256": wav_sha, "reference_id": raw.get("reference_id"), "seed": raw.get("seed"),
                "params": params, "heldout_id": heldout, "paired_id": paired,
                "paired_split": corpus[paired]["split"] if paired else None, "violations": violations, "notes": notes,
                "provenance": {k: prov.get(k) for k in ("status", "sources", "notes", "self_report", "source_take",
                                                        "source_provenance")},
                "conditioning_encoders": params.get("conditioning_encoders") or doc.get("conditioning_encoders"),
                "model_version": params.get("model_version") or doc.get("model_version"),
            })
    if not takes:
        raise SystemExit("no usable takes in the given manifest(s)")
    training = {}
    for label, s in provenance_by_run.items():
        trained = s["trained"]
        status = "unverified" if "unverified" in s["statuses"] else (
            "pretrained" if s["statuses"] == {"pretrained"} else "verified")
        training[label] = {
            "status": status, "sources": sorted(s["sources"]), "notes": sorted(s["notes"]),
            "clips_trained_on": len(trained),
            "test_clips_in_training": sorted((trained & leaks.tests) | (s["self_report_heldout"] & leaks.tests)),
            "near_duplicates_in_training": sorted(trained & {c for v in leaks.near.values() for c in v}),
            "other_speaker_clips_in_training": sorted(trained & known_other),
        }
        if training[label]["test_clips_in_training"]:
            warnings.append(f"{label}: model trained on held-out test clip(s) {training[label]['test_clips_in_training']}; "
                            "its takes on those lines are leaks")
        if training[label]["other_speaker_clips_in_training"]:
            others = ec.other_speakers_label()
            warnings.append(f"{label}: model trained on {len(training[label]['other_speaker_clips_in_training'])} clips the "
                            "delivery task confirmed as other characters" + (f" ({others})" if others else ""))
    heldouts = sorted({t["heldout_id"] for t in takes if t["heldout_id"]})
    if args.sv_variants == "both":
        sv_variants = [args.audio] + [v for v in ec.available_variants() if v != args.audio]
    else:
        sv_variants = [args.audio]
    run = {
        "run": run_dir.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "audio_variant": args.audio,
        "sv_variants": sv_variants, "device": args.device, "manifests": manifests,
        "script": str(script_path) if script_path and script_path.exists() else None,
        "delivery_map": str(delivery_path) if delivery_path else None,
        "near_duplicates_of_heldout": {h: leaks.near[h] for h in heldouts if h in leaks.near},
        "training_provenance": training,
        "takes": takes, "warnings": warnings,
    }
    ec.write_json(run_dir / "takes.json", run)
    for w in warnings:
        ec.log(f"warning: {w}")
    ec.log(f"{len(takes)} takes from {len(manifests)} manifest(s) -> {run_dir / 'takes.json'} "
           f"(speaker references: {', '.join(sv_variants)})")
    return run


def gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return min(float(v) for v in out.split()) / 1024
    except Exception:
        return None


CPU_ONLY_STEPS = {"prosody", "artifacts", "dnsmos"}  # numpy/librosa/onnxruntime-CPU only


def step_device(name: str, args) -> str:
    if name in CPU_ONLY_STEPS:
        return "cpu"
    if name == "asr" and args.asr_device:
        return args.asr_device
    return args.device


def run_metric(name: str, venv: str, script: str, run_dir: Path, args) -> dict:
    python = PYTHONS[venv]
    device = step_device(name, args)
    threads = args.asr_threads if name == "asr" else args.threads
    cmd = [str(python), str(HERE / script), "--run-dir", str(run_dir), "--device", device, "--threads", str(threads)]
    if name == "asr":
        cmd += ["--model", args.asr_model]
        if args.no_name_probe:
            cmd += ["--no-name-probe"]
    if name == "prosody":
        cmd += ["--workers", str(args.workers)]
    if name == "utmos":
        cmd += ["--real", args.extra_real]
    if name == "emotion":
        cmd += ["--refs", args.emotion_refs, "--max-refs-per-label", str(args.emotion_max_refs)]
    env = dict(os.environ)
    # PYTHONDONTWRITEBYTECODE: never write .pyc files into the venvs (the VoxCPM2 one is hash-locked).
    env.update(PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HOME=str(VOICE_LAB / "hf-cache"), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    if venv == "evalextra":  # its weights and caches stay inside voice-lab/eval-extra
        env["HF_HOME"] = str(VOICE_LAB / "eval-extra" / "hf-home")
    if device == "cpu":
        # On this Windows driver an empty value still exposes the GPU (probed: '' -> 1 device, '-1' -> 0),
        # and torch 2.8 DataLoaders pin memory whenever an accelerator is visible.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if env.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
            env.pop("CUDA_VISIBLE_DEVICES")  # an inherited CPU-only setting would silently defeat --device cuda
        if name == "asr":
            env["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = CT2_CACHE_CONFIG
    log_path = run_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    guard: dict = {}
    if device == "cuda":
        free = gpu_free_gb()
        step_need = args.gpu_need_gb if args.gpu_need_gb is not None else STEP_GPU_GB.get(name, 5.0)
        need = step_need + args.gpu_reserve_gb
        if free is None or free < need:
            message = (f"refused: GPU has {'unknown' if free is None else f'{free:.1f}'} GB free, needs {need:.1f} GB "
                       f"({step_need} for this step + {args.gpu_reserve_gb} reserve); re-run later with "
                       f"--metrics {name}")
            log_path.write_text(message + "\n", encoding="utf-8")
            ec.log(f"!! {name} {message}")
            return {"metric": name, "device": device, "exit_code": 75, "seconds": 0.0, "log": str(log_path),
                    "gpu_guard": message}
        guard["free_at_start_gb"] = round(free, 1)
        guard["need_gb"] = step_need
    ec.log(f"== {name}: {python.parent.parent.parent.name} venv, {device}, {threads} threads")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        stop = threading.Event()
        if device == "cuda":
            threading.Thread(target=gpu_watchdog, args=(process, args.gpu_reserve_gb, stop, guard), daemon=True).start()
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                if not line.startswith(("  0%", "100%")):
                    sys.stdout.write("   " + line)
                    sys.stdout.flush()
            code = process.wait()
        finally:
            stop.set()
        if guard.get("killed"):
            log_file.write(f"GPU guard: {guard['killed']}\n")
    status = {"metric": name, "device": device, "exit_code": code, "seconds": round(time.time() - started, 1),
              "log": str(log_path)}
    metric_json = ec.metric_out_path(run_dir, name)
    if metric_json.exists() and metric_json.stat().st_mtime >= started:
        try:
            status["take_errors"] = ec.read_json(metric_json).get("take_errors", 0)
        except (OSError, ValueError):
            pass
    if guard:
        if "min_free_gb" in guard and "free_at_start_gb" in guard:
            # Approximate: other processes' allocations move the same number.
            guard["approx_peak_use_gb"] = round(guard["free_at_start_gb"] - guard["min_free_gb"], 1)
        status["gpu_guard"] = guard
    if code != 0:
        why = guard.get("killed") or (f"{status.get('take_errors')} take(s) unscored (resource errors)"
                                      if code == 3 else "")
        ec.log(f"!! {name} failed (exit {code}){'; ' + why if why else ''}; see {log_path}")
    return status


def gpu_watchdog(process: subprocess.Popen, reserve_gb: float, stop: threading.Event, guard: dict) -> None:
    """Kill this step's own child process if the GPU's free memory falls below the reserve (0.5 s polls)."""
    lowest = None
    while not stop.wait(0.5):
        free = gpu_free_gb()
        if free is None:
            continue
        lowest = free if lowest is None else min(lowest, free)
        guard["min_free_gb"] = round(lowest, 1)
        if free < reserve_gb and process.poll() is None:
            guard["killed"] = f"killed pid {process.pid}: GPU free memory {free:.1f} GB < reserve {reserve_gb:.1f} GB"
            process.kill()
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", default=[], help="takes manifest.json (repeatable)")
    parser.add_argument("--takes-root", action="append", default=[],
                        help=f"add every <root>/<engine>/<run>/manifest.json (repeatable), e.g. {ec.LAB.name}/takes")
    parser.add_argument("--skip", default=None, help="regex on 'engine/run' labels of discovered manifests to leave out")
    parser.add_argument("--run", required=True, help=f"output folder name under {ec.LAB.name}/eval "
                                                     f"(character {ec.SLUG}: VOICE_LAB_CHARACTER)")
    parser.add_argument("--audio", choices=ec.VARIANTS, default="original",
                        help="primary real-clip variant (prosody, ASR baseline, artifacts, speaker-purity report)")
    parser.add_argument("--sv-variants", choices=("both", "primary"), default="primary",
                        help="speaker models score against the primary reference variant only (ranked), or also "
                             "against the other one when its real set is complete (reported, not ranked)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--script", default=str(ec.DEFAULT_SCRIPT))
    parser.add_argument("--delivery-map", default=str(ec.DEFAULT_DELIVERY_MAP))
    parser.add_argument("--metrics", default=",".join(m[0] for m in METRICS),
                        help="comma list of steps to run (aggregation always runs); the optional eval-extra steps "
                             + ",".join(m[0] for m in EXTRA_METRICS) + " run only when listed")
    parser.add_argument("--extra-real", choices=("all", "train", "none"), default="all",
                        help="utmos: real clips scored as its baseline (none = takes only, e.g. smoke tests)")
    parser.add_argument("--emotion-refs", choices=("all", "run", "none"), default="all",
                        help="emotion: delivery reference clips embedded (all labels, the run's labels, none)")
    parser.add_argument("--emotion-max-refs", type=int, default=0, help="emotion: first N references per label (0 = all)")
    parser.add_argument("--aggregate-only", action="store_true", help="reuse existing metric outputs")
    parser.add_argument("--asr-device", choices=("cpu", "cuda"), default=None,
                        help="device for the whisper step only (default: --device). CPU large-v3 needs 1-3 min per "
                             "clip on a loaded machine and ~4.5 GB RAM, so run ASR on cuda")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-threads", type=int, default=8)
    parser.add_argument("--no-name-probe", action="store_true", help="skip the prompt-free proper-noun ASR pass")
    parser.add_argument("--threads", type=int, default=8, help="torch CPU threads per step (shared machine)")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes for the prosody step (pyin)")
    parser.add_argument("--gpu-need-gb", type=float, default=None,
                        help="GPU memory a cuda step needs before it starts (default per step: "
                             + ", ".join(f"{k} {v:g}" for k, v in STEP_GPU_GB.items()) + " GB)")
    parser.add_argument("--gpu-reserve-gb", type=float, default=8.0,
                        help="free GPU memory kept for other apps (Substance Painter needs 8 GB)")
    import aggregate
    aggregate.add_options(parser)
    args = parser.parse_args()
    ec.log(f"character {evalchar.describe()}")
    discovered = discover_manifests(args.takes_root, args.skip)
    args.manifest = list(dict.fromkeys(args.manifest + discovered))
    if args.takes_root:
        ec.log(f"{len(discovered)} manifest(s) under {', '.join(args.takes_root)}"
               + (f" (skipping /{args.skip}/)" if args.skip else ""))
    if not args.manifest and not args.aggregate_only:
        parser.error("give at least one --manifest or a --takes-root that contains manifests")

    run_dir = ec.EVAL_ROOT / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        ec.log(f"aggregate only: {run_dir}")
    else:
        wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
        unknown = set(wanted) - {m[0] for m in ALL_METRICS}
        if unknown:
            raise SystemExit(f"unknown metrics: {sorted(unknown)}")
        missing_venv = sorted({PYTHONS[v] for n, v, _s in EXTRA_METRICS if n in wanted and not PYTHONS[v].exists()})
        if missing_venv:
            raise SystemExit(f"eval-extra venv not found: {missing_venv[0]} (install voice-lab/eval-extra first)")
        build_takes(args, run_dir)
        statuses = [run_metric(name, venv, script, run_dir, args) for name, venv, script in ALL_METRICS if name in wanted]
        status_path = run_dir / "run_status.json"
        if status_path.exists() and set(wanted) != {m[0] for m in ALL_METRICS}:  # partial: keep other steps' status
            earlier = {s["metric"]: s for s in ec.read_json(status_path).get("steps", []) if s["metric"] not in wanted}
            statuses = [earlier.get(name) or next(s for s in statuses if s["metric"] == name)
                        for name, _v, _s in ALL_METRICS if name in earlier or name in wanted]
        ec.write_json(status_path, {"steps": statuses, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    card = aggregate.aggregate(run_dir, aggregate.options_from_args(args))
    ec.log(f"scorecard: {run_dir / 'scorecard.md'}")
    failed = [s for s in (card.get("steps") or []) if s.get("exit_code")]
    unscored = sum(1 for r in card.get("takes", []) if r.get("unscored"))
    if unscored:
        ec.log(f"!! {unscored} take(s) unscored; see the scorecard's warnings for the step to re-run")
    sys.exit(1 if failed or unscored else 0)


if __name__ == "__main__":
    main()
