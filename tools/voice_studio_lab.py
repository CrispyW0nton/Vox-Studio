"""Desktop catalog and bounded launcher for the existing local voice laboratory.

No weights or lab sources are copied. Cataloging never imports a model. Render
requires a successful preview of unchanged inputs, an exclusive shared GPU lock,
and conservative host/GPU admission. Existing renderer provenance checks remain.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def roots():
    home = Path.home()
    data = Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "VoxStudio"
    config = data / "voice_lab.json"
    settings = read(config) if config.is_file() else {}
    return {"data": data,
            "lab": Path(settings.get("labRoot", home / "Documents/GDeveloper/External/voice-lab")),
            "voices": Path(settings.get("voicesRoot", home / "Documents/KotorMods/Voices"))}


def python_at(directory):
    return directory / ".venv/Scripts/python.exe"


def catalog(r):
    items, presets, warnings = [], [], []

    def item(group, name, path, detail, links=()):
        path = Path(path)
        items.append(dict(group=group, name=name, path=str(path),
                          status="Present" if path.exists() else "Missing", detail=detail,
                          links=[str(p) for p in links if Path(p).exists()]))

    engines = r["data"] / "engines"
    descriptions = {
        "voxcpm2": "VoxCPM2: native TTS, narration and character LoRA laboratory.",
        "qwen3-tts-audition": "Qwen3-TTS: reference-conditioned auditions; import a narration recipe below.",
        "chatterbox": "Chatterbox: local auditions with native Perth watermarking; import a narration recipe below.",
        "seed-vc": "Seed-VC: Performance Mirror fallback and reference voice conversion.",
        "dia-actions": "Dia: offline nonverbal reaction generation.",
        "latentsync": "LatentSync: supporting lip-sync runtime.",
        "musetalk": "MuseTalk: supporting talking-face runtime.",
        "music-compose": "Music composition: supporting audio runtime.",
    }
    for directory in sorted(engines.iterdir()) if engines.exists() else []:
        if not directory.is_dir():
            continue
        models = list((directory / "cache").glob("models--*/snapshots/*"))
        item("Engines", directory.name, directory,
             descriptions.get(directory.name, "Local runtime") +
             f"\nPython: {python_at(directory)} ({'present' if python_at(directory).is_file() else 'missing'})" +
             "\nPresence is an inventory check; use Preview to validate a render.", models)
        for model in models:
            item("Models", f"{model.parent.parent.name} / {model.name[:12]}", model,
                 "Installed model snapshot; revision: " + model.name)
    for name, folder in (("GPT-SoVITS", "GPT-SoVITS"), ("IndexTTS 2.5", "index-tts"),
                         ("Seed-VC laboratory", "seed-vc"), ("ClearerVoice (post-processing)", "clearvoice"),
                         ("Evaluation models", "eval-extra")):
        path = r["lab"] / folder
        item("Engines", name, path, f"External laboratory runtime. Python: {python_at(path)}",
             [path / "PROVENANCE.json", path / "LICENSE_NOTES.md", path / "checkpoints"])
    item("Engines", "RVC", r["data"] / "training/RVC-WebUI", "Imported RVC and trained Performance Mirror voices.",
         [r["data"] / "rvc_models", r["data"] / "rvc_onnx_models"])

    pilot = r["voices"] / "VOICE_DELIVERY02/pilot_gpu_v001"
    results = pilot / "scores/DIRECTOR_RESULTS_v001.md"
    choices = {
        "atris": (400, "idref_instr", "October 8 blind pilot: step 400, cfg 1.5, instruction B. Scores 82/78 with mastering. Further lines require listening."),
        "atton": (800, "idref_instr", "October 8 blind pilot: step 800, instruction B. One favored take; provisional across other lines."),
        "baodur": (400, "ref_prompt", "October 8 blind pilot: continuation step 400. Prompt gblbaodur227 failed; use the reviewed prompt map. GPT-SoVITS remains available."),
    }
    for config in sorted((r["lab"] / "characters").glob("*.json")):
        try:
            char = read(config)
            slug, path = config.stem, Path(char["lab"])
            name = char["display_name"]
            selected = choices.get(slug) if results.is_file() else None
            detail = selected[2] if selected else "Laboratory character; no desktop selection recorded."
            detail += "\nRaw synthesis is not mastered game audio. The listening result applies to the complete recipe and mastering chain."
            links = [path / "DECISIONS.md", path / "delivery/performance_bible.md", path / "delivery",
                     path / "takes", pilot / f"{slug}_blind_pilot_v001.html", results,
                     pilot / "chained" / slug, r["voices"] / "VOICE_DELIVERY02/WRITING_GUIDE_VOXSTUDIO_v001.md"]
            item("Characters", name, path, detail, links)
            checkpoints = sorted((path / "voxcpm/runs").glob("*/checkpoints/step_*"))
            for checkpoint in checkpoints:
                item("Character models", f"{name} / {checkpoint.parent.parent.name} / {checkpoint.name}", checkpoint,
                     "Trained checkpoint. A training step alone does not establish listening approval.")
            if selected:
                step, mode, note = selected
                checkpoint = path / f"voxcpm/runs/lora_v2_original/checkpoints/step_{step:07d}"
                script = pilot / "inputs" / f"{slug}_v001_script.json"
                if slug == "atris":
                    script = path / "delivery/takes_script_v005_exact_v001.json"
                delivery_map = path / "delivery/delivery_map.json"
                if slug == "baodur":
                    # Retain the actual pilot map, including line-specific intent tags;
                    # exclude its documented losing prompt before exposing the preset.
                    script = pilot / "inputs/baodur_pilot_c20_script.json"
                    delivery_map = pilot / "inputs/baodur_pilot_map.json"
                line_ids = ""
                if slug == "baodur" and script.is_file() and delivery_map.is_file():
                    mapping = read(delivery_map).get("deliveries", {})
                    line_ids = ",".join(line["id"] for line in read(script)["lines"]
                                        if "gblbaodur227" not in json.dumps(mapping.get(line["delivery"], {})))
                presets.append(dict(id=slug + "-voxcpm", name=name + " — VoxCPM2 pilot recipe", backend="voxcpm",
                                    character=slug, script=str(script), checkpoint=str(checkpoint), mode=mode,
                                    deliveryMap=str(delivery_map), cfg=2.0 if slug == "baodur" else 1.5,
                                    note=note, seeds=1, lines=line_ids))
            if (path / "indextts/emotion_controls.json").is_file():
                presets.append(dict(id=slug + "-indextts", name=name + " — IndexTTS 2.5 audition", backend="indextts",
                                    character=slug, script=str(path / "delivery/takes_script.json"), seeds=1,
                                    note="Experimental IndexTTS performance donor; original references only."))
            if slug == "baodur" and (path / "gsv/weights/baodur_v2ProPlus_original").is_dir():
                presets.append(dict(id="baodur-gsv", name="Bao-Dur — GPT-SoVITS g15/s8/aux2", backend="gsv",
                                    character=slug, script=str(path / "delivery/takes_script.json"), seeds=1,
                                    note="Previously accepted g15/s8/aux2 recipe; new text still needs listening."))
        except (ValueError, OSError, KeyError) as exc:
            warnings.append(f"{config}: {exc}")
    for profile in sorted((r["data"] / "voxcpm_profiles").glob("*/profile.json")):
        try:
            p = read(profile)
            item("Desktop voices", p.get("name", profile.parent.name), profile.parent,
                 f"Voice ID: {p.get('voice_id', profile.parent.name)}\nEngine: {p.get('engine', 'VoxCPM2')}", [profile])
        except (ValueError, OSError) as exc:
            warnings.append(f"{profile}: {exc}")
    for model in sorted((r["data"] / "rvc_models").glob("*")):
        item("RVC models", model.name, model, "Installed conversion model or model manifest.")
    for config in sorted((r["lab"] / "characters").glob("*.json")):
        try:
            char = read(config)
            for model in sorted((Path(char["lab"]) / "gsv/weights").glob("*/*/*")):
                if model.suffix in {".ckpt", ".pth"}:
                    item("Character models", char["display_name"] + " / GPT-SoVITS / " + model.name,
                         model, "Trained GPT-SoVITS weights; use the recorded epoch pair in the render recipe.")
        except (OSError, ValueError, KeyError):
            pass  # Already reported in character discovery.
    return dict(items=items, presets=presets, warnings=warnings, roots={k: str(v) for k, v in r.items()})


def build_command(job, r, output):
    """Fixed backend adapters; never execute a command supplied in a recipe."""
    backend = job["backend"]
    if backend not in {"voxcpm", "gsv", "indextts"}:
        raise ValueError("Unknown laboratory backend")
    slug = job["character"]
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", slug):
        raise ValueError("Invalid character")
    charfile = r["lab"] / "characters" / (slug + ".json")
    char = read(charfile)
    lab = Path(char["lab"])
    seeds = job.get("seeds", 1)
    if type(seeds) is not int or not 1 <= seeds <= 6:
        raise ValueError("Choose one to six seeds")
    script = Path(job["script"])
    lines = read(script).get("lines", [])
    selected = job.get("lines", "").strip()
    ids = set(selected.split(",")) if selected else set()
    if ids and not ids <= {x["id"] for x in lines}:
        raise ValueError("Selected line IDs are absent from the script")
    chosen = [x for x in lines if not ids or x["id"] in ids]
    if not 1 <= len(chosen) <= 40 or any(not 1 <= len(x.get("text", "")) <= 1200 for x in chosen):
        raise ValueError("Select 1–40 script lines of 1–1200 characters")
    # The losing Bao-Dur continuation reference must not become a default.
    delivery_map = Path(job.get("deliveryMap", lab / "delivery/delivery_map.json"))
    if slug == "baodur" and backend == "voxcpm" and job.get("mode") == "ref_prompt":
        mapping = read(delivery_map)
        for line in chosen:
            if "gblbaodur227" in json.dumps(mapping.get("deliveries", mapping).get(line.get("delivery"), {})):
                raise ValueError("This line uses rejected prompt gblbaodur227; select a reviewed prompt/line")
    code = r["lab"] / "baodur-lab"
    common = ["--script", str(script), "--seeds", str(seeds), "--out-root", str(output)]
    if selected:
        common += ["--lines", selected]
    dependencies = [charfile, script, delivery_map, lab / "corpus.jsonl"]
    if backend == "voxcpm":
        checkpoint = Path(job["checkpoint"]).resolve()
        if not checkpoint.is_relative_to((lab / "voxcpm/runs").resolve()) or not checkpoint.is_dir():
            raise ValueError("Select a checkpoint from this character's laboratory")
        mode = job.get("mode", "idref_instr")
        cfg = job.get("cfg", 1.5)
        if mode not in {"idref_instr", "ref_prompt"} or type(cfg) not in {float, int} or not 1 <= cfg <= 3:
            raise ValueError("Invalid VoxCPM recipe settings")
        python = python_at(r["data"] / "engines/voxcpm2")
        worker = code / "voxcpm/render_lowmem.py"
        args = ["--lora", str(checkpoint), "--modes", mode, "--cfg", str(cfg),
                "--instruction-variant", "B", "--timesteps", "10", "--ref-audio", "original",
                "--delivery-map", str(delivery_map), "--output", "both", "--run-prefix", "desktop_",
                "--reserve-gb", "8", "--stop-ram-gb", "3", "--stop-vram-gb", "1"]
        dependencies += [p for p in checkpoint.iterdir() if p.name.startswith("lora_")]
    elif backend == "gsv":
        if slug != "baodur":
            raise ValueError("Only Bao-Dur has a recorded desktop GPT-SoVITS selection")
        python = python_at(r["lab"] / "GPT-SoVITS")
        worker = code / "gsv/infer.py"
        args = ["--exp", "baodur_v2ProPlus_original", "--gpt-epoch", "15", "--sovits-epoch", "8",
                "--aux-refs", "2", "--source", "new", "--device", "cuda", "--strict-refs", "--run", "desktop"]
        dependencies += list((lab / "gsv/weights/baodur_v2ProPlus_original").rglob("*.ckpt"))
        dependencies += list((lab / "gsv/weights/baodur_v2ProPlus_original").rglob("*.pth"))
    else:
        python = python_at(r["lab"] / "index-tts")
        worker = code / "indextts/render.py"
        args = ["--modes", "spk_ref", "--ref-audio", "original", "--output", "both", "--run-prefix", "desktop_"]
        dependencies += [r["lab"] / "index-tts/PROVENANCE.json"]
    if not python.is_file() or not worker.is_file():
        raise ValueError(f"Runtime or worker missing: {python}; {worker}")
    dependencies += list(code.rglob("*.py")) + list((r["lab"] / "characters").glob("*.py"))
    for folder in (lab / "delivery", lab / "voxcpm", lab / "gsv", lab / "indextts"):
        dependencies += list(folder.glob("*.json"))
    # Original reference recordings must not change between preview and render.
    dependencies += list((lab / "source/audio").glob("*.wav"))
    env = dict(os.environ, VOICE_LAB_CHARACTER=slug, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return [str(python), "-B", str(worker), *common, *args], env, dependencies


def memory_available():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("total", "free", "commitTotal", "commitFree", "virtualTotal", "virtualFree", "extended")]
    mem = Memory(); mem.length = ctypes.sizeof(mem)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise RuntimeError("Windows memory admission unavailable")
    return mem.free, mem.commitFree


def resource_gpu():
    """Select by GPU UUID, never assume CUDA and nvidia-smi indexes agree."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.free,display_active", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=20, check=True, creationflags=flags)
    rows = [tuple(p.strip() for p in line.split(",")) for line in result.stdout.splitlines() if line.strip()]
    candidates = [row for row in rows if row[3].lower() == "disabled" and float(row[2]) >= 18000]
    if not candidates or any(float(row[2]) < 4096 for row in rows if row[3].lower() == "enabled"):
        raise RuntimeError("Need an idle compute GPU with 18,000 MiB free and 4 GiB free on each display GPU")
    free, commit = memory_available()
    if free < 10 * 1024**3 or commit < 17 * 1024**3:
        raise RuntimeError("Need 10 GiB free physical RAM and 17 GiB free commit before loading a model")
    return candidates[0][0]


def execute(command, env, stop=None, log_path=None):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen(command, env=env, creationflags=flags, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    output = []
    def drain():
        log = Path(log_path).open("w", encoding="utf-8") if log_path else None
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                if log:
                    log.write(line); log.flush()
                output.append(line)
                if len(output) > 5000:
                    del output[:1000]
        finally:
            if log:
                log.close()
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        while child.poll() is None:
            if stop and stop.exists():
                child.terminate()
                child.wait(timeout=20)
                raise RuntimeError("Stopped; partial output retained")
            if stop and min(memory_available()) < 3 * 1024**3:
                raise RuntimeError("Stopped at the 3 GiB host memory reserve; partial output retained")
            time.sleep(0.25)
        if child.returncode:
            raise RuntimeError(f"Worker exited with code {child.returncode}; inspect the output above")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        reader.join(timeout=10)
    return "".join(output)


def operation(jobfile, action, expected, r):
    job = read(jobfile)
    work = Path(jobfile).parent
    receipt = work / "preview.json"
    output = work / "takes"
    if job.get("backend") in {"qwen", "chatterbox", "indextts25"}:
        return audition_operation(job, work, action, expected, r)
    command, env, dependencies = build_command(job, r, output)
    files = sorted(set([Path(__file__), *dependencies]))
    hashes = {str(p): digest(p) for p in files if p.is_file()}
    revision = hashlib.sha256(json.dumps(dict(job=job, files=hashes), sort_keys=True).encode()).hexdigest()
    print(json.dumps(dict(stage=action, revision=revision, command=command, output=str(output))), flush=True)
    if action == "preview":
        # A failed re-preview invalidates any previously successful receipt.
        receipt.unlink(missing_ok=True)
        preview_output = execute([*command, "--dry-run"], env, log_path=work / "preview.log")
        if job["backend"] == "gsv":
            report = json.loads(preview_output)
            if not report.get("gpt") or not report.get("sovits") or report.get("failures") or not report.get("takes_planned"):
                raise RuntimeError("GPT-SoVITS preview has missing weights, no takes or reference failures")
        elif "dry run" not in preview_output.lower() and "dry-run" not in preview_output.lower():
            raise RuntimeError("Worker returned without a dry-run validation report")
        receipt.write_text(json.dumps(dict(revision=revision, job=job, files=hashes), indent=2), encoding="utf-8")
    else:
        if not receipt.is_file() or expected != revision or read(receipt)["revision"] != revision:
            raise ValueError("Preview missing or inputs changed; preview again")
        claim = work / "render.claim.json"
        if claim.exists():
            raise ValueError("This job was already claimed; create a new preview for a new render")
        if (work / "STOP").exists():
            raise ValueError("This job was stopped; create a new preview")
        gpu = resource_gpu()
        lock = r["voices"] / "gpu_train.lock"
        owner = str(uuid.uuid4())
        # Never remove or replace another session's lock, even if it looks stale.
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(dict(owner=owner, token=owner, pid=os.getpid(), purpose="Vox Studio desktop",
                           created=time.strftime("%Y-%m-%d %H:%M:%S")), stream)
        try:
            with claim.open("x", encoding="utf-8") as stream:
                json.dump(dict(revision=revision, gpu=gpu), stream)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            execute(command, env, work / "STOP", work / "render.log")
            (work / "render.complete.json").write_text(json.dumps(dict(revision=revision, output=str(output)), indent=2), encoding="utf-8")
        finally:
            if lock.exists() and read(lock).get("owner") == owner:
                lock.unlink()
    print(json.dumps(dict(stage="complete", revision=revision, output=str(output))), flush=True)


def audition_operation(job, work, action, expected, r):
    """Retain the audition worker's model/reference hash binding and watermarking."""
    recipe_path = Path(job["script"])
    recipe = read(recipe_path)
    if recipe.get("backend") != job["backend"]:
        raise ValueError("Imported recipe backend changed")
    if job["backend"] == "indextts25":
        runtime_root = r["lab"] / "index-tts"
        if Path(recipe["modelPath"]).resolve() != (runtime_root / "checkpoints").resolve():
            raise ValueError("IndexTTS audition must use the installed laboratory checkpoints")
    else:
        runtime = "qwen3-tts-audition" if job["backend"] == "qwen" else "chatterbox"
        runtime_root = r["data"] / "engines" / runtime
    python = python_at(runtime_root)
    worker = Path(__file__).with_name("narration_audition.py")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = work / "preview.json"
    if action == "preview":
        receipt.unlink(missing_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run([str(python), "-B", str(worker), "preview", str(recipe_path)],
                            env=env, capture_output=True, text=True, encoding="utf-8", creationflags=flags)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    evidence = json.loads(result.stdout)
    revision = evidence["revision"]
    print(json.dumps(dict(stage=action, revision=revision, recipe=recipe)), flush=True)
    if action == "preview":
        receipt.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    else:
        if not receipt.is_file() or read(receipt)["revision"] != revision or expected != revision:
            raise ValueError("Audition inputs changed; preview again")
        if (work / "STOP").exists() or (work / "render.claim.json").exists():
            raise ValueError("This audition job was stopped or claimed; create a new preview")
        gpu = resource_gpu()
        owner = str(uuid.uuid4())
        lock = r["voices"] / "gpu_train.lock"
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(dict(owner=owner, token=owner, pid=os.getpid(), purpose="Vox Studio audition",
                           created=time.strftime("%Y-%m-%d %H:%M:%S")), stream)
        try:
            with (work / "render.claim.json").open("x", encoding="utf-8") as stream:
                json.dump(dict(revision=revision, gpu=gpu), stream)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = gpu
            execute([str(python), "-B", str(worker), "render", str(recipe_path), "--revision", revision], env, work / "STOP", work / "render.log")
            (work / "render.complete.json").write_text(json.dumps(dict(revision=revision, output=recipe["outputRoot"]), indent=2), encoding="utf-8")
        finally:
            if lock.exists() and read(lock).get("owner") == owner:
                lock.unlink()
    print(json.dumps(dict(stage="complete", revision=revision, output=recipe["outputRoot"])), flush=True)


def configure_stdio():
    # The Qt owner and worker logs use UTF-8. A legacy Windows console must not
    # kill the reader thread when a model prints Unicode progress bars.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


def main():
    configure_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("catalog", "preview", "render"))
    ap.add_argument("--job", type=Path)
    ap.add_argument("--revision", default="")
    args = ap.parse_args()
    try:
        r = roots()
        if args.action == "catalog":
            print(json.dumps(catalog(r), ensure_ascii=False))
        else:
            if not args.job:
                ap.error("--job is required")
            operation(args.job, args.action, args.revision, r)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Voice Lab: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
