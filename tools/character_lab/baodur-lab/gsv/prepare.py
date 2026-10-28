"""Headless GPT-SoVITS dataset formatting for one character (the webui's "1A one-click triple", step by step).

Character: VOICE_LAB_CHARACTER (default baodur) through voice-lab/characters; <lab> is its lab folder and
<speaker> its config "speaker".
Steps (each upstream script runs through gsv_run_step.py with the environment variables webui.py would set):
  text      GPT_SoVITS/prepare_datasets/1-get-text.py           -> 2-name2text.txt (English needs no BERT feats)
  hubert    GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py  -> 4-cnhubert/<wav>.pt, 5-wav32k/<wav>
  sv        GPT_SoVITS/prepare_datasets/2-get-sv.py             -> 7-sv_cn/<wav>.pt   (v2Pro / v2ProPlus only)
  semantic  GPT_SoVITS/prepare_datasets/3-get-semantic.py       -> 6-name2semantic.tsv
into <lab>/gsv/exp/<speaker>_<version>_<audio>[_trval]/ (the upstream "logs/<exp_name>" folder).
Original game audio only (<lab>/DECISIONS.md): a list holding restored copies is refused unless
--allow-restored. Before each step the host commit guard checks that the step's estimated commit plus a 2 GB
margin is available (waits up to --wait-min, else exit 5).

Defaults to --device cpu and needs no VRAM: measured on this machine (4-clip smoke run) each step spends
15-110 s on imports/model load and well under 1 s per clip, so a ~230-clip set (Bao-Dur's) takes roughly 10
minutes; peak working set 2.7 GB (step 1 loads BERT-large even for English), the other steps 1.2-1.4 GB, one step
at a time.
With --device cuda each step caps its allocator at --mem-gb (default 3 GB) and refuses to start unless
--reserve-gb stays free.
The list is refused if it holds a held-out test clip or a line the speaker screen gives to another character.
Incremental: per-clip features are skipped when present; step 1 is redone when the list or lexicon changed,
the semantic table when the clip set changed. Superseded outputs are archived (renamed), never deleted.
Features of clips that dropped out of the list stay on disk unused (both trainers read only listed clips).

Usage:
  python prepare.py                                  # <speaker>_original.list (train split) on CPU
  python prepare.py --splits train,val               # <speaker>_original_trval.list -> <speaker>_v2ProPlus_original_trval
  python prepare.py --version v2Pro                  # v2Pro fallback: semantic tokens from s2Gv2Pro
  python prepare.py --limit 8 --exp-root <scratch>/exp   # smoke test on 8 clips
  python prepare.py --steps none --allow-missing --exp-root <scratch>/exp   # plan/screen only, no models
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsv_common as C  # noqa: E402

STEP_ORDER = ("text", "hubert", "sv", "semantic")
STEP_SCRIPTS = {
    "text": "GPT_SoVITS/prepare_datasets/1-get-text.py",
    "hubert": "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
    "sv": "GPT_SoVITS/prepare_datasets/2-get-sv.py",
    "semantic": "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
}


def read_list(path: Path) -> list[list[str]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 4:
            raise SystemExit(f"{path}:{n}: expected wav|speaker|lang|text, got {line!r}")
        rows.append(parts)
    return rows


def parse_name2text(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                out[parts[0]] = parts[1]
    return out


def parse_semantic(path: Path) -> set[str]:
    names = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) == 2 and parts[1].strip():
                names.add(parts[0])
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", choices=C.AUDIO_MODES, default="original")
    ap.add_argument("--allow-restored", action="store_true",
                    help="permit restored audio (DECISIONS.md: the user chose original audio for training)")
    ap.add_argument("--splits", default=",".join(C.DEFAULT_SPLITS),
                    help="which exported list/experiment: train (default) or train,val")
    ap.add_argument("--version", choices=C.VERSIONS, default="v2ProPlus")
    ap.add_argument("--list", default=None,
                    help=f"training list (default {C.LISTS_DIR}\\{C.SPEAKER}_<audio>[_trval].list)")
    ap.add_argument("--name", default=None,
                    help=f"experiment name (default {C.SPEAKER}_<version>_<audio>[_trval])")
    ap.add_argument("--exp-root", default=str(C.EXP_ROOT))
    ap.add_argument("--steps", default=",".join(STEP_ORDER))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-gb", type=float, default=3.0, help="GPU allocator cap per step")
    ap.add_argument("--reserve-gb", type=float, default=8.0, help="VRAM that must stay free for other users")
    ap.add_argument("--wait-min", "--wait-vram-min", dest="wait_min", type=float, default=0.0,
                    help="wait up to N minutes for enough free VRAM / host commit")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads per step")
    ap.add_argument("--limit", type=int, default=0, help="only the first N list rows (smoke tests)")
    ap.add_argument("--lexicon", default=str(C.LEXICON))
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--force-text", action="store_true", help="redo step 1 even if inputs are unchanged")
    ap.add_argument("--force-semantic", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="do not fail when a clip drops out of a step")
    ap.add_argument("--exclusions", default=str(C.EXCLUSIONS), help="speaker-screen exclusions JSON")
    ap.add_argument("--no-exclusions", action="store_true",
                    help="DANGER: do not refuse lists that contain other characters' lines")
    args = ap.parse_args()
    for key in ("list", "exp_root", "exclusions", "lexicon"):
        setattr(args, key, C.abspath(getattr(args, key)))

    device = C.normalize_device(args.device)
    if device != "cpu":
        C.gpu_visible_devices(device)  # fail fast if the caller hid the GPU
    C.require_original_audio(args.audio, args.allow_restored, "preparing features")
    splits = C.parse_splits(args.splits)
    version = args.version
    name = args.name or C.exp_name(version, args.audio, splits)
    exp_dir = args.exp_root / name
    list_path = args.list or C.default_list_path(args.audio, splits)
    if not list_path.exists():
        raise SystemExit(f"{list_path} not found; run export_dataset.py --audio {args.audio} first")
    if not C.GSV_PY.exists():
        raise SystemExit(f"GPT-SoVITS venv python not found: {C.GSV_PY}")

    rows = read_list(list_path)
    if args.limit:
        rows = rows[: args.limit]
    corpus = C.load_corpus()
    guard = C.HeldOut(corpus)
    screen = C.screen_training_ids([Path(w).stem for w, *_ in rows], corpus,
                                   None if args.no_exclusions else args.exclusions)
    if screen["heldout"]:
        raise SystemExit(f"refusing to prepare {list_path}: held-out test clips {screen['heldout']}")
    if screen["other_speaker"]:
        raise SystemExit(f"refusing to prepare {list_path}: it contains other characters' lines "
                         f"{screen['other_speaker']}; re-run export_dataset.py (or pass --no-exclusions)")
    if args.no_exclusions:
        C.log("WARNING: --no-exclusions: the list was not checked against the speaker screen")
    for clip, why in screen["screened"].items():
        C.log(f"note: {clip} is in the list although the default screen would leave it out: {why}")
    for wav, _spk, lang, _text in rows:
        if lang != C.LANG:
            raise SystemExit(f"unexpected language {lang!r} in {list_path}")
        reason = guard.reason(Path(wav).stem, wav)
        if reason:
            raise SystemExit(f"refusing to prepare {list_path}: {reason}")
        if not Path(wav).exists():
            raise SystemExit(f"missing audio {wav}")
    names = [Path(wav).name for wav, *_ in rows]
    if len(set(n.casefold() for n in names)) != len(names):
        raise SystemExit("duplicate wav basenames in the list (upstream keys features by basename)")
    audio_kind = {"restored" if Path(w).parent.resolve() == C.RESTORED_DIR.resolve() else "original" for w, *_ in rows}
    audio_kind = audio_kind.pop() if len(audio_kind) == 1 else "mixed"
    C.require_original_audio(audio_kind, args.allow_restored, f"preparing {list_path.name}")
    if audio_kind != args.audio and not args.name:
        raise SystemExit(f"{list_path} holds {audio_kind} audio but --audio is {args.audio}; the default experiment "
                         f"name would be misleading (pass the matching --audio or an explicit --name)")

    body = "".join("|".join(r) + "\n" for r in rows)
    exp_dir.mkdir(parents=True, exist_ok=True)
    input_list = exp_dir / "input.list"
    if input_list.exists() and input_list.read_text(encoding="utf-8") != body:
        C.archive(input_list)
    if not input_list.exists():
        input_list.write_text(body, encoding="utf-8")

    stamp_path = exp_dir / "prep_stamp.json"
    stamp = C.read_json(stamp_path) if stamp_path.exists() else {}
    has_features = (exp_dir / "4-cnhubert").exists() and any((exp_dir / "4-cnhubert").iterdir())
    if has_features and stamp.get("audio_kind") not in (None, audio_kind):
        raise SystemExit(f"{exp_dir} holds features from {stamp['audio_kind']} audio but the list is {audio_kind}; "
                         f"use another --name")
    if stamp.get("version") not in (None, version):
        raise SystemExit(f"{exp_dir} was prepared for {stamp['version']}; use another --name")
    stamp.update({"audio_kind": audio_kind, "version": version, "list": str(list_path)})

    lexicon_path = C.require_lexicon(None if args.no_lexicon else args.lexicon)
    lexicon_sha = C.sha256_file(lexicon_path) if lexicon_path else None
    steps = [s for s in STEP_ORDER if s in {x.strip() for x in args.steps.split(",") if x.strip()}]
    if "Pro" not in version and "sv" in steps:
        steps.remove("sv")
    base = {"inp_text": input_list, "inp_wav_dir": "", "exp_name": name, "opt_dir": exp_dir, "i_part": "0",
            "all_parts": "1", "is_half": "False" if device == "cpu" else "True", "version": version}
    step_cmd = [C.GSV_PY, C.LAB_CODE / "gsv_run_step.py", "--device", device, "--mem-gb", args.mem_gb,
                "--reserve-gb", args.reserve_gb, "--wait-min", args.wait_min, "--threads", args.threads]
    report = {"exp": name, "exp_dir": str(exp_dir), "list": str(list_path), "rows": len(rows), "device": device,
              "version": version, "audio_kind": audio_kind, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "exclusions": None if args.no_exclusions else str(args.exclusions),
              "screened_but_included": screen["screened"], "steps": {}}
    problems: list[str] = []

    def run_step(step: str, extra: dict, lexicon: str | None = None) -> None:
        env = C.gsv_env(device, args.threads, {**base, **extra})
        cmd = step_cmd + ["--script", STEP_SCRIPTS[step]] + (["--lexicon", lexicon] if lexicon else [])
        commit = C.require_commit(C.COMMIT_GB[f"prep_{step}"] + (1.5 if device != "cpu" else 0.0),
                                  f"prep step {step}", args.wait_min)
        report.setdefault("host_commit", {})[step] = commit
        t0 = time.time()
        C.log(f"step {step}: {STEP_SCRIPTS[step]} on {device} ({len(rows)} clips)")
        rc = C.run_logged(cmd, env, exp_dir / "logs" / f"prep_{step}.log")
        report["steps"][step] = {"seconds": round(time.time() - t0, 1), "exit_code": rc}
        if rc:
            C.write_json(exp_dir / "prep_report.json", report)
            raise SystemExit(f"step {step} failed with exit code {rc} (log: {exp_dir / 'logs' / f'prep_{step}.log'})")

    def check(step: str, missing: list[str]) -> None:
        report["steps"].setdefault(step, {})["missing"] = missing
        if missing:
            problems.append(f"{step}: {len(missing)} clips missing, e.g. {missing[:5]}")

    if "text" in steps:
        final, part = exp_dir / "2-name2text.txt", exp_dir / "2-name2text-0.txt"
        key = {"list_sha256": C.sha256_text(body), "lexicon_sha256": lexicon_sha}
        have = parse_name2text(final)
        if final.exists() and stamp.get("text") == key and set(names) <= set(have) and not args.force_text:
            C.log("step text: up to date, skipped")
            report["steps"]["text"] = {"skipped": True}
        else:
            C.archive(final)
            C.archive(part)
            run_step("text", {"bert_pretrained_dir": C.BERT_DIR}, lexicon_path)
            if not part.exists():
                raise SystemExit(f"step text produced no {part.name}")
            os.replace(part, final)  # single part: identical to webui's merge
            stamp["text"] = key
            have = parse_name2text(final)
        phones = [p for n in names if n in have for p in have[n].split()]
        report["steps"]["text"].update({"lines": len(have), "phones": len(phones), "unk_phones": phones.count("UNK")})
        check("text", [n for n in names if n not in have])

    if "hubert" in steps:
        run_step("hubert", {"cnhubert_base_dir": C.HUBERT_DIR, "sv_path": C.SV_CKPT})
        check("hubert", [n for n in names if not (exp_dir / "4-cnhubert" / f"{n}.pt").exists()
                         or not (exp_dir / "5-wav32k" / n).exists()])

    if "sv" in steps:
        run_step("sv", {"cnhubert_base_dir": C.HUBERT_DIR, "sv_path": C.SV_CKPT})
        check("sv", [n for n in names if not (exp_dir / "7-sv_cn" / f"{n}.pt").exists()])

    if "semantic" in steps:
        final, part = exp_dir / "6-name2semantic.tsv", exp_dir / "6-name2semantic-0.tsv"
        names_sha = C.sha256_text("\n".join(sorted(names)))
        have = parse_semantic(final)
        if final.exists() and stamp.get("semantic_names_sha256") == names_sha and set(names) <= have \
                and not args.force_semantic:
            C.log("step semantic: up to date, skipped")
            report["steps"]["semantic"] = {"skipped": True}
        else:
            C.archive(final)
            C.archive(part)
            run_step("semantic", {"pretrained_s2G": C.S2_PRETRAINED[version][0],
                                  "s2config_path": C.S2_BASE_CONFIG[version]})
            if not part.exists():
                raise SystemExit(f"step semantic produced no {part.name}")
            lines = [ln for ln in part.read_text(encoding="utf-8").strip("\n").split("\n") if ln.strip()]
            final.write_text("\n".join(["item_name\tsemantic_audio"] + lines) + "\n", encoding="utf-8")
            C.archive(part, "merged")
            stamp["semantic_names_sha256"] = names_sha
            have = parse_semantic(final)
        check("semantic", [n for n in names if n not in have])

    # Verify every output for every clip, whichever steps ran in this invocation.
    have_text = parse_name2text(exp_dir / "2-name2text.txt")
    have_sem = parse_semantic(exp_dir / "6-name2semantic.tsv")
    coverage = {
        "text": [n for n in names if n not in have_text],
        "hubert": [n for n in names if not (exp_dir / "4-cnhubert" / f"{n}.pt").exists()],
        # upstream writes 5-wav32k in place (no temp + move) and skips existing files, so a crash mid-write
        # would leave a truncated wav behind; anything under 1 kB (< 16 ms at 32 kHz) is treated as missing
        "wav32k": [n for n in names if not (exp_dir / "5-wav32k" / n).exists()
                   or (exp_dir / "5-wav32k" / n).stat().st_size < 1024],
        "semantic": [n for n in names if n not in have_sem],
    }
    if "Pro" in version:
        coverage["sv"] = [n for n in names if not (exp_dir / "7-sv_cn" / f"{n}.pt").exists()]
    text_current = stamp.get("text") == {"list_sha256": C.sha256_text(body), "lexicon_sha256": lexicon_sha}
    if not text_current and "text" not in steps:
        problems.append("text: 2-name2text.txt predates the current list/lexicon; run the text step")
    report["coverage_missing"] = {k: len(v) for k, v in coverage.items()}
    # Features of clips that are no longer listed (e.g. from an earlier, unscreened list) stay on disk but are
    # never read: both trainers only use clips present in 2-name2text.txt / 6-name2semantic.tsv.
    listed = set(names)
    report["unlisted_feature_files"] = sorted(p.name[:-3] for p in (exp_dir / "4-cnhubert").glob("*.pt")
                                              if p.name[:-3] not in listed) if (exp_dir / "4-cnhubert").exists() else []
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["problems"] = problems
    report["complete"] = text_current and not any(coverage.values()) and not problems
    C.write_json(stamp_path, stamp)
    C.write_json(exp_dir / "prep_report.json", report)
    for p in problems:
        C.log(f"PROBLEM {p}")
    if problems and not args.allow_missing:
        raise SystemExit(f"prep incomplete ({len(problems)} problems); see {exp_dir / 'prep_report.json'}")
    timings = ", ".join(f"{step} {info.get('seconds', 0)}s" for step, info in report["steps"].items())
    C.log(f"prep done -> {exp_dir} ({timings})")
    if report["complete"]:
        C.log(f"experiment complete for training: {len(names)} clips with text, hubert, wav32k, "
              f"{'sv, ' if 'Pro' in version else ''}semantic")
    else:
        C.log(f"not yet trainable, missing per output: {report['coverage_missing']} (run the remaining steps)")


if __name__ == "__main__":
    main()
