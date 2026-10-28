import importlib.util
import os
import subprocess
import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("desktop_lab", Path(__file__).resolve().parents[2] / "tools/voice_studio_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class DesktopLabTests(unittest.TestCase):
    def test_legacy_console_drains_unicode_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            logfile = Path(directory) / 'unicode.log'
            script = ("import importlib.util,os,sys; from pathlib import Path; "
                      "s=importlib.util.spec_from_file_location('lab',sys.argv[1]); "
                      "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                      "m.configure_stdio(); m.execute([sys.executable,'-c','print(chr(0x258f))'], "
                      "dict(os.environ,PYTHONIOENCODING='utf-8'),log_path=Path(sys.argv[2]))")
            result = subprocess.run([sys.executable, '-c', script, str(Path(lab.__file__)), str(logfile)],
                                    env=dict(os.environ, PYTHONIOENCODING='cp1252', PYTHONUTF8='0'),
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', errors='replace'))
            self.assertIn('\u258f', result.stdout.decode('utf-8'))
            self.assertIn('\u258f', logfile.read_text(encoding='utf-8'))
            self.assertNotIn(b'UnicodeEncodeError', result.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.r = {k: base / k for k in ("data", "lab", "voices")}
        for root in self.r.values():
            root.mkdir()
        self.character = self.r["voices"] / "AtrisLab"
        self.character.mkdir()
        self.put(self.r["lab"] / "characters/atris.json", {"lab": str(self.character), "display_name": "Atris"})
        self.put(self.character / "delivery/delivery_map.json", {})
        self.put(self.character / "corpus.jsonl", {})
        self.put(self.character / "source/audio/original.wav", {})
        self.script = self.character / "delivery/script.json"
        self.put(self.script, {"lines": [{"id": "E198", "text": "Take it.", "delivery": "guarded", "source": "new"}]})
        self.checkpoint = self.character / "voxcpm/runs/original/checkpoints/step_0000400"
        self.put(self.checkpoint / "lora_weights.safetensors", {})
        self.put(self.r["lab"] / "baodur-lab/voxcpm/render_lowmem.py", {})
        self.put(lab.python_at(self.r["data"] / "engines/voxcpm2"), {})
        self.job = dict(backend="voxcpm", character="atris", script=str(self.script), checkpoint=str(self.checkpoint), mode="idref_instr", seeds=1)
        self.work = base / "job"
        self.work.mkdir()
        self.jobfile = self.work / "job.json"
        self.put(self.jobfile, self.job)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def preview(self):
        with mock.patch.object(lab, "execute", return_value="dry run: inputs validated") as execute:
            lab.operation(self.jobfile, "preview", "", self.r)
            self.assertIn("--dry-run", execute.call_args.args[0])
        return lab.read(self.work / "preview.json")["revision"]

    def test_no_model_import_or_preview_from_inventory(self):
        with mock.patch.object(lab.subprocess, "Popen", side_effect=AssertionError("must not run")):
            result = lab.catalog(self.r)
        self.assertTrue(any(x["name"] == "Atris" for x in result["items"]))
        self.assertEqual(result["presets"], [])  # No listening report => no claimed selected recipe.

    def test_failed_preview_invalidates_success(self):
        self.preview()
        with mock.patch.object(lab, "execute", side_effect=RuntimeError("bad reference")):
            with self.assertRaisesRegex(RuntimeError, "bad reference"):
                lab.operation(self.jobfile, "preview", "", self.r)
        self.assertFalse((self.work / "preview.json").exists())

    def test_changed_audio_invalidates_before_gpu_work(self):
        revision = self.preview()
        self.put(self.character / "source/audio/original.wav", "changed")
        with mock.patch.object(lab, "resource_gpu", side_effect=AssertionError("GPU touched")):
            with self.assertRaisesRegex(ValueError, "changed"):
                lab.operation(self.jobfile, "render", revision, self.r)

    def test_foreign_lock_is_preserved(self):
        revision = self.preview()
        lock = self.r["voices"] / "gpu_train.lock"
        self.put(lock, {"owner": "another job"})
        before = lock.read_bytes()
        with mock.patch.object(lab, "resource_gpu", return_value="GPU-test"), mock.patch.object(lab, "execute") as execute:
            with self.assertRaises(FileExistsError):
                lab.operation(self.jobfile, "render", revision, self.r)
            execute.assert_not_called()
        self.assertEqual(lock.read_bytes(), before)
        self.assertFalse((self.work / "render.claim.json").exists())

    def test_render_is_single_use_and_cleans_only_own_lock(self):
        revision = self.preview()
        with mock.patch.object(lab, "resource_gpu", return_value="GPU-test"), mock.patch.object(lab, "execute") as execute:
            lab.operation(self.jobfile, "render", revision, self.r)
            self.assertEqual(execute.call_args.args[1]["CUDA_VISIBLE_DEVICES"], "GPU-test")
            self.assertNotIn("--dry-run", execute.call_args.args[0])
            with self.assertRaisesRegex(ValueError, "claimed"):
                lab.operation(self.jobfile, "render", revision, self.r)
        self.assertFalse((self.r["voices"] / "gpu_train.lock").exists())

    def test_reject_other_character_checkpoint(self):
        self.job["checkpoint"] = str(self.r["voices"])
        with self.assertRaisesRegex(ValueError, "this character"):
            lab.build_command(self.job, self.r, self.work / "takes")

    def test_reject_unknown_lines_and_unbounded_seeds(self):
        for fields in ({"lines": "unknown"}, {"seeds": 0}, {"seeds": True}, {"seeds": 7}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                lab.build_command(dict(self.job, **fields), self.r, self.work / "takes")

    def test_stopped_job_does_not_start(self):
        revision = self.preview()
        (self.work / "STOP").touch()
        with mock.patch.object(lab, "resource_gpu", side_effect=AssertionError("GPU touched")):
            with self.assertRaisesRegex(ValueError, "stopped"):
                lab.operation(self.jobfile, "render", revision, self.r)

    def test_compute_gpu_selected_by_uuid(self):
        result = mock.Mock(stdout="GPU-display, RTX 5090, 9000, Enabled\nGPU-compute, RTX 3090, 22000, Disabled\n")
        with mock.patch.object(lab.subprocess, "run", return_value=result), mock.patch.object(lab, "memory_available", return_value=(20 * 1024**3, 30 * 1024**3)):
            self.assertEqual(lab.resource_gpu(), "GPU-compute")

    def test_display_reserve_and_host_memory_refuse(self):
        result = mock.Mock(stdout="GPU-display, RTX 5090, 3000, Enabled\nGPU-compute, RTX 3090, 22000, Disabled\n")
        with mock.patch.object(lab.subprocess, "run", return_value=result), self.assertRaisesRegex(RuntimeError, "display"):
            lab.resource_gpu()
        result.stdout = "GPU-compute, RTX 3090, 22000, Disabled\n"
        with mock.patch.object(lab.subprocess, "run", return_value=result), mock.patch.object(lab, "memory_available", return_value=(5 * 1024**3, 30 * 1024**3)), self.assertRaisesRegex(RuntimeError, "physical RAM"):
            lab.resource_gpu()


if __name__ == "__main__":
    unittest.main()
