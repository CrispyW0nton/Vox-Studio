"""Training provenance of a take's model: which corpus clips it was trained on, for the held-out leak check.

A take on a held-out line is only an honest test when the model that rendered it never trained on that
test clip, nor on a train clip that repeats the same line (evalcommon.near_duplicates_of_test). Conditioning
leaks (a test clip used as reference/prompt) are checked separately in run_eval.LeakChecker.

Resolution (per take; a Seed-VC take also inherits its source take's provenance in run_eval):
1. Contract fields, any engine, take params first, then the manifest top level:
     "trained_on_ids": [clip ids]          ([] = trained on none of the character's clips, e.g. a pretrained model)
     "training_data": path | [paths]       training lists (jsonl / .list / .txt / .tsv / Seed-VC prepare_report.json)
2. Engine self-reports (VoxCPM render.py): params.adapter_trained_on_heldout (bool) and the manifest's
   adapter.heldout_clips_in_its_training (test ids). A positive self-report is always a leak.
3. Engine resolvers for this lab's three engines:
     voxcpm   adapter_kind "none" -> base model only; "baseline" -> Vox Studio's own adapter for the character:
              %LOCALAPPDATA%/VoxStudio/voxcpm_training/<folder>/train.jsonl (evalchar.VOX_TRAINING_DIR; Bao-Dur:
              bao-dur, the July adapter; label from <lab>/eval/eval_character.json provenance);
              "lora" -> <lora_path>/../../run_config.json -> config.train_manifest
     gsv      zero_shot -> pretrained only; else the experiment folder of gpt_weights / sovits_weights
              (<root>/exp/<exp>/input.list, 2-name2text.txt and 6-name2semantic.tsv, union of all three)
     seedvc*  cfm_checkpoint / ar_checkpoint "pretrained..." -> pretrained; else <checkpoint dir>/run_meta.json ->
              <data>/prepare_report.json items[].sources (the train items, not the exclusion list)
4. Anything else: "unverified" (held-out scores are then reported as unverified, not as clean).
Only the stdlib is needed (evalchar included), so run_eval.py can use it from any Python.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import evalchar  # stdlib only: the character VOICE_LAB_CHARACTER selects (default baodur)

LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\NewAdmin\AppData\Local"))
# Vox Studio's own VoxCPM adapter for the character (adapter_kind "baseline"; Bao-Dur: the July adapter)
BASELINE_TRAIN_MANIFEST = LOCALAPPDATA / "VoxStudio" / "voxcpm_training" / evalchar.VOX_TRAINING_DIR / "train.jsonl"
BASELINE_LABEL = evalchar.SETTINGS["provenance"]["voxcpm_baseline_label"]
JULY_TRAIN_MANIFEST = BASELINE_TRAIN_MANIFEST  # name kept for older callers
TEXT_KEYS = re.compile(r"text|transcript|instruction|caption|why|note|reason", re.I)


class Provenance:
    def __init__(self, corpus_ids, id_pattern: re.Pattern):
        self.ids = set(corpus_ids)
        self.id_re = id_pattern
        self.canon = {i.lower(): i for i in self.ids}
        self._files: dict[str, tuple[set[str] | None, str | None]] = {}
        self._json: dict[str, dict | None] = {}

    # ------------------------------------------------------------------ parsing
    def _ids_in(self, text: str) -> set[str]:
        return {self.canon[m.group(1).lower()] for m in self.id_re.finditer(text or "")}

    def _read_json(self, path: Path) -> dict | None:
        key = str(path)
        if key not in self._json:
            try:
                self._json[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._json[key] = None
        return self._json[key]

    def training_ids(self, path: Path | str) -> tuple[set[str] | None, str | None]:
        """Clip ids in one training list file; (None, reason) when the file cannot be used."""
        path = Path(path)
        key = str(path)
        if key in self._files:
            return self._files[key]
        result: tuple[set[str] | None, str | None]
        if not path.exists():
            result = (None, f"training list not found: {path}")
        elif path.is_dir():
            result = ({i for p in path.rglob("*") if p.is_file() for i in self._ids_in(p.stem)}, None)
        else:
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                result = (None, f"unreadable training list {path}: {exc}")
            else:
                suffix = path.suffix.lower()
                found: set[str] = set()
                if suffix == ".json":
                    data = self._read_json(path) or {}
                    items = data.get("items") if isinstance(data, dict) else None
                    if isinstance(items, list):  # Seed-VC prepare_report.json: train items and their source clips
                        for item in items:
                            for source in item.get("sources") or []:
                                found |= self._ids_in(str(source))
                            found |= self._ids_in(Path(str(item.get("file") or "")).stem)
                        result = (found, None)
                    else:
                        result = (None, f"unknown training-list JSON structure: {path}")
                elif suffix == ".jsonl":
                    for line in body.splitlines():
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        found |= self._row_ids(row)
                    result = (found, None)
                else:  # .list / .txt / .tsv: first field per line is the audio file or item name
                    for line in body.splitlines():
                        head = re.split(r"[|\t]", line, maxsplit=1)[0].strip()
                        if head:
                            found |= self._ids_in(Path(head).stem if re.search(r"[\\/]|\.\w+$", head) else head)
                    result = (found, None)
        self._files[key] = result
        return result

    def _row_ids(self, value, key: str = "") -> set[str]:
        if isinstance(value, str):
            if TEXT_KEYS.search(key):
                return set()
            return self._ids_in(Path(value).stem if re.search(r"[\\/]", value) else value)
        if isinstance(value, dict):
            out: set[str] = set()
            for k, v in value.items():
                out |= self._row_ids(v, str(k))
            return out
        if isinstance(value, list):
            out = set()
            for v in value:
                out |= self._row_ids(v, key)
            return out
        return set()

    # ------------------------------------------------------------------ resolution
    def resolve(self, take: dict, manifest: dict, manifest_path: Path) -> dict:
        params = take.get("params") or {}
        engine = str(take.get("engine") or manifest.get("engine") or "").lower()
        result = {"status": "unverified", "trained_ids": [], "sources": [], "notes": [],
                  "self_report": {}, "source_take": None}
        adapter = manifest.get("adapter") if isinstance(manifest.get("adapter"), dict) else {}
        if "adapter_trained_on_heldout" in params:
            result["self_report"]["adapter_trained_on_heldout"] = bool(params["adapter_trained_on_heldout"])
        if adapter.get("heldout_clips_in_its_training") is not None:
            result["self_report"]["heldout_clips_in_its_training"] = list(adapter["heldout_clips_in_its_training"])
        if params.get("source_manifest") and params.get("source_take_id"):
            result["source_take"] = {"manifest": str(params["source_manifest"]), "take_id": str(params["source_take_id"])}

        trained: set[str] = set()
        statuses: list[str] = []

        def use_files(paths, label: str) -> None:
            ok = True
            for p in ([paths] if isinstance(paths, (str, Path)) else list(paths or [])):
                p = Path(str(p))
                if not p.is_absolute():
                    p = (manifest_path.parent / p).resolve()
                ids, problem = self.training_ids(p)
                if ids is None:
                    result["notes"].append(problem)
                    ok = False
                    continue
                trained.update(ids)
                result["sources"].append(f"{label}: {p}")
            statuses.append("verified" if ok and result["sources"] else "unverified")

        # 1. contract fields
        for scope, holder in (("params", params), ("manifest", manifest)):
            if isinstance(holder.get("trained_on_ids"), list):
                ids = {self.canon[str(i).lower()] for i in holder["trained_on_ids"] if str(i).lower() in self.canon}
                trained |= ids
                result["sources"].append(f"{scope}.trained_on_ids ({len(ids)} clips)")
                statuses.append("verified" if ids else "pretrained")
                break
            if holder.get("training_data"):
                use_files(holder["training_data"], f"{scope}.training_data")
                break
        else:
            # 3. engine resolvers
            if engine.startswith("seedvc") or "cfm_checkpoint" in params:
                self._seedvc(params, use_files, statuses, result)
            elif engine.startswith("voxcpm") or "adapter_kind" in params:
                self._voxcpm(params, adapter, use_files, statuses, result)
            elif engine.startswith("gsv") or "gpt_weights" in params or "sovits_weights" in params:
                self._gsv(params, manifest, use_files, statuses, result)
        if statuses:
            if "unverified" in statuses:
                result["status"] = "unverified"
            elif all(s == "pretrained" for s in statuses):
                result["status"] = "pretrained"
            else:
                result["status"] = "verified"
        result["trained_ids"] = sorted(trained)
        return result

    def _voxcpm(self, params, adapter, use_files, statuses, result) -> None:
        kind = params.get("adapter_kind") or adapter.get("kind")
        if kind == "none":
            statuses.append("pretrained")
            result["sources"].append("voxcpm base model (no adapter)")
        elif kind == "baseline":
            use_files([BASELINE_TRAIN_MANIFEST], BASELINE_LABEL)
        elif kind == "lora":
            lora = params.get("lora_path") or adapter.get("path")
            run_config = Path(str(lora)).parent.parent / "run_config.json" if lora else None
            data = self._read_json(run_config) if run_config and run_config.exists() else None
            manifest = ((data or {}).get("config") or {}).get("train_manifest")
            if manifest:
                use_files([manifest], f"voxcpm {run_config.parent.name} train_manifest")
            else:
                statuses.append("unverified")
                result["notes"].append(f"voxcpm LoRA {lora}: no run_config.json with config.train_manifest")
        else:
            statuses.append("unverified")
            result["notes"].append(f"voxcpm adapter kind {kind!r} unknown")

    def _gsv(self, params, manifest, use_files, statuses, result) -> None:
        if params.get("zero_shot") or manifest.get("zero_shot"):
            statuses.append("pretrained")
            result["sources"].append("gsv pretrained base (zero-shot)")
            return
        exp_dirs = []
        for key in ("gpt_weights", "sovits_weights"):
            weights = params.get(key) or manifest.get(key)
            if weights:
                w = Path(str(weights))
                if len(w.parents) > 3:  # <root>/weights/<exp>/<GPT|SoVITS>/<file>
                    exp_dirs.append(w.parents[3] / "exp" / w.parents[1].name)
        files = []
        for exp in dict.fromkeys(exp_dirs):
            files += [p for p in (exp / "input.list", exp / "2-name2text.txt", exp / "6-name2semantic.tsv") if p.exists()]
        if files:
            use_files(files, "gsv experiment")
        else:
            statuses.append("unverified")
            result["notes"].append(f"gsv: no experiment lists found for {[str(e) for e in exp_dirs] or 'unknown weights'}")

    def _seedvc(self, params, use_files, statuses, result) -> None:
        any_ckpt = False
        for key in ("cfm_checkpoint", "ar_checkpoint"):
            spec = params.get(key)
            if not spec:
                continue
            any_ckpt = True
            spec = str(spec)
            if spec.startswith("pretrained"):
                statuses.append("pretrained")
                result["sources"].append(f"seedvc {key}: {spec}")
                continue
            meta_path = Path(spec).parent / "run_meta.json"
            meta = self._read_json(meta_path) if meta_path.exists() else None
            data_dir = (meta or {}).get("data")
            report = Path(data_dir) / "prepare_report.json" if data_dir else None
            if report and report.exists():
                use_files([report], f"seedvc {key} training data")
            else:
                statuses.append("unverified")
                result["notes"].append(f"seedvc {key} {spec}: no run_meta.json/prepare_report.json to verify")
        if not any_ckpt:
            statuses.append("unverified")
            result["notes"].append("seedvc take without cfm_checkpoint")


def leak_findings(prov: dict, heldout: str | None, near: dict[str, list[str]]) -> list[str]:
    """Violations for a take on a held-out line, from its resolved provenance."""
    if not heldout:
        return []
    out = []
    report = prov.get("self_report") or {}
    if report.get("adapter_trained_on_heldout"):
        out.append(f"engine reports its model was trained on held-out line {heldout} "
                   "(params.adapter_trained_on_heldout): leak")
    listed = {str(i).lower() for i in report.get("heldout_clips_in_its_training") or []}
    trained = set(prov.get("trained_ids") or [])
    if heldout in trained or heldout.lower() in listed:
        out.append(f"model trained on held-out test clip {heldout} (this line's ground truth): leak")
    for clip in sorted(trained & set(near.get(heldout, []))):
        out.append(f"model trained on {clip}, which repeats held-out line {heldout}: leak")
    return out
