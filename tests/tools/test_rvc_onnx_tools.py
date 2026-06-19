#!/usr/bin/env python3
"""Tests for Pass 9b RVC ONNX tooling."""

from __future__ import annotations

import json
import math
import subprocess
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


def write_sine_wav(
    path: Path,
    *,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = 16000,
    seconds: float = 0.25,
    leading_silence_samples: int = 0,
) -> None:
    sample_count = int(sample_rate * seconds)
    values = [0] * leading_silence_samples
    for index in range(sample_count):
        sample = amplitude * math.sin((2.0 * math.pi * frequency * index) / sample_rate)
        values.append(max(-32768, min(32767, int(sample * 32767.0))))

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))


def write_sized_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([0x5A]) * size)


class RvcOnnxToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.inspect_script = self.repo_root / "tools" / "scripts" / "inspect_onnx_contract.py"
        self.export_script = self.repo_root / "tools" / "scripts" / "export_rvc_to_onnx.py"
        self.compare_script = self.repo_root / "tools" / "scripts" / "compare_rvc_outputs.py"
        self.gate_script = (
            self.repo_root / "tools" / "scripts" / "validate_rvc_pass9b_gate.py"
        )
        self.temp_dir = tempfile.TemporaryDirectory(prefix="voxstudio_rvc_tools_")
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_tool(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=self.repo_root,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def test_inspect_template_writes_non_required_contract(self) -> None:
        contract_path = self.workspace / "graph_contract.json"

        result = self.run_tool(str(self.inspect_script), "--template", "--out", str(contract_path))

        self.assertIn("Wrote", result.stdout)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertFalse(contract["required"])
        self.assertEqual([], contract["generator"]["inputs"])
        self.assertEqual([], contract["hubert"]["outputs"])

    def test_inspect_template_rejects_required_contract(self) -> None:
        result = self.run_tool(
            str(self.inspect_script),
            "--template",
            "--required",
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("--template cannot be combined with --required", result.stdout)

    def test_export_dry_run_writes_contract_template(self) -> None:
        pth_path = self.workspace / "hero.pth"
        out_dir = self.workspace / "bundle"
        pth_path.write_text("placeholder", encoding="utf-8")

        self.run_tool(
            str(self.export_script),
            "--model-id",
            "hero",
            "--pth",
            str(pth_path),
            "--out",
            str(out_dir),
            "--dry-run",
        )

        manifest = json.loads((out_dir / "native_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual("hero", manifest["model_id"])
        self.assertFalse(manifest["graph_contract"]["required"])
        self.assertEqual([], manifest["graph_contract"]["f0"]["inputs"])

    def test_export_requires_contract_file_for_strict_mode(self) -> None:
        pth_path = self.workspace / "hero.pth"
        pth_path.write_text("placeholder", encoding="utf-8")

        result = self.run_tool(
            str(self.export_script),
            "--model-id",
            "hero",
            "--pth",
            str(pth_path),
            "--out",
            str(self.workspace / "bundle"),
            "--dry-run",
            "--require-contract",
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("--require-contract needs --graph-contract", result.stdout)

    def test_export_embeds_strict_graph_contract(self) -> None:
        pth_path = self.workspace / "hero.pth"
        contract_path = self.workspace / "graph_contract.json"
        out_dir = self.workspace / "bundle"
        pth_path.write_text("placeholder", encoding="utf-8")
        contract_path.write_text(
            json.dumps(
                {
                    "required": False,
                    "generator": {
                        "inputs": [{"name": "audio", "element_type": "float32"}],
                        "outputs": [{"name": "wave", "element_type": "float32"}],
                    },
                    "hubert": {"inputs": [], "outputs": []},
                    "f0": {"inputs": [], "outputs": []},
                }
            ),
            encoding="utf-8",
        )

        self.run_tool(
            str(self.export_script),
            "--model-id",
            "hero",
            "--pth",
            str(pth_path),
            "--graph-contract",
            str(contract_path),
            "--require-contract",
            "--out",
            str(out_dir),
            "--dry-run",
        )

        manifest = json.loads((out_dir / "native_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["graph_contract"]["required"])
        self.assertEqual("audio", manifest["graph_contract"]["generator"]["inputs"][0]["name"])

    def test_export_assembles_existing_onnx_bundle(self) -> None:
        pth_path = self.workspace / "hero.pth"
        generator_path = self.workspace / "source_generator.onnx"
        hubert_path = self.workspace / "source_hubert.onnx"
        rmvpe_path = self.workspace / "source_rmvpe.onnx"
        contract_path = self.workspace / "graph_contract.json"
        out_dir = self.workspace / "bundle"
        pth_path.write_text("placeholder", encoding="utf-8")
        generator_path.write_text("generator-onnx", encoding="utf-8")
        hubert_path.write_text("hubert-onnx", encoding="utf-8")
        rmvpe_path.write_text("rmvpe-onnx", encoding="utf-8")
        contract_path.write_text(
            json.dumps(
                {
                    "required": False,
                    "generator": {
                        "inputs": [{"name": "phone", "element_type": "float32"}],
                        "outputs": [{"name": "wave", "element_type": "float32"}],
                    },
                    "hubert": {"inputs": [], "outputs": []},
                    "f0": {"inputs": [], "outputs": []},
                }
            ),
            encoding="utf-8",
        )

        result = self.run_tool(
            str(self.export_script),
            "--model-id",
            "hero",
            "--pth",
            str(pth_path),
            "--generator-onnx",
            str(generator_path),
            "--hubert-onnx",
            str(hubert_path),
            "--rmvpe-onnx",
            str(rmvpe_path),
            "--graph-contract",
            str(contract_path),
            "--require-contract",
            "--out",
            str(out_dir),
        )

        self.assertIn("Assembled", result.stdout)
        self.assertEqual("generator-onnx", (out_dir / "generator.onnx").read_text())
        self.assertEqual("hubert-onnx", (out_dir / "hubert.onnx").read_text())
        self.assertEqual("rmvpe-onnx", (out_dir / "rmvpe.onnx").read_text())
        manifest = json.loads((out_dir / "native_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual("generator.onnx", manifest["generator_onnx"])
        self.assertEqual("hubert.onnx", manifest["hubert_onnx"])
        self.assertEqual("rmvpe.onnx", manifest["f0_onnx"])
        self.assertTrue(manifest["graph_contract"]["required"])

    def test_export_requires_all_onnx_inputs_for_assembly(self) -> None:
        pth_path = self.workspace / "hero.pth"
        pth_path.write_text("placeholder", encoding="utf-8")

        result = self.run_tool(
            str(self.export_script),
            "--model-id",
            "hero",
            "--pth",
            str(pth_path),
            "--out",
            str(self.workspace / "bundle"),
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("Non-dry-run assembly requires existing ONNX artifacts", result.stdout)
        self.assertIn("--generator-onnx", result.stdout)

    def test_compare_accepts_aligned_matching_wavs(self) -> None:
        sidecar_path = self.workspace / "sidecar.wav"
        native_path = self.workspace / "native.wav"
        report_path = self.workspace / "report.json"
        write_sine_wav(sidecar_path)
        write_sine_wav(native_path, leading_silence_samples=32)

        result = self.run_tool(
            str(self.compare_script),
            "--sidecar",
            str(sidecar_path),
            "--native",
            str(native_path),
            "--out",
            str(report_path),
            "--max-spectral-db",
            "0.1",
        )

        self.assertIn('"passed": true', result.stdout)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(32, report["native_lag_samples"])
        self.assertLessEqual(report["mean_abs_spectral_db"], 0.1)
        self.assertGreater(report["frames_compared"], 0)

    def test_compare_rejects_spectral_mismatch(self) -> None:
        sidecar_path = self.workspace / "sidecar.wav"
        native_path = self.workspace / "native.wav"
        write_sine_wav(sidecar_path, amplitude=0.5)
        write_sine_wav(native_path, amplitude=0.1)

        result = self.run_tool(
            str(self.compare_script),
            "--sidecar",
            str(sidecar_path),
            "--native",
            str(native_path),
            "--max-spectral-db",
            "2.0",
            check=False,
        )

        self.assertEqual(2, result.returncode)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertGreater(report["mean_abs_spectral_db"], 2.0)

    def test_pass9b_gate_accepts_complete_evidence(self) -> None:
        ab_report_path = self.workspace / "ab.json"
        native_report_path = self.workspace / "native.json"
        sidecar_dir = self.workspace / "sidecar"
        native_dir = self.workspace / "native"
        gate_report_path = self.workspace / "gate.json"
        ab_report_path.write_text(
            json.dumps({"passed": True, "mean_abs_spectral_db": 1.25}),
            encoding="utf-8",
        )
        native_report_path.write_text(json.dumps({"latency_ms": 120}), encoding="utf-8")
        write_sized_file(sidecar_dir / "payload.bin", 4096)
        write_sized_file(native_dir / "runtime.bin", 1024)

        result = self.run_tool(
            str(self.gate_script),
            "--ab-report",
            str(ab_report_path),
            "--native-report",
            str(native_report_path),
            "--sidecar-latency-ms",
            "160",
            "--sidecar-footprint",
            str(sidecar_dir),
            "--native-footprint",
            str(native_dir),
            "--min-size-reduction-mb",
            "0.001",
            "--out",
            str(gate_report_path),
        )

        self.assertIn('"passed": true', result.stdout)
        report = json.loads(gate_report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["spectral"]["passed"])
        self.assertTrue(report["latency"]["passed"])
        self.assertTrue(report["footprint"]["passed"])

    def test_pass9b_gate_rejects_latency_regression(self) -> None:
        ab_report_path = self.workspace / "ab.json"
        native_report_path = self.workspace / "native.json"
        sidecar_report_path = self.workspace / "sidecar.json"
        sidecar_dir = self.workspace / "sidecar"
        native_dir = self.workspace / "native"
        ab_report_path.write_text(
            json.dumps({"passed": True, "mean_abs_spectral_db": 1.25}),
            encoding="utf-8",
        )
        native_report_path.write_text(json.dumps({"latency_ms": 240}), encoding="utf-8")
        sidecar_report_path.write_text(
            json.dumps({"status": {"last_latency_ms": 160}}),
            encoding="utf-8",
        )
        write_sized_file(sidecar_dir / "payload.bin", 4096)
        write_sized_file(native_dir / "runtime.bin", 1024)

        result = self.run_tool(
            str(self.gate_script),
            "--ab-report",
            str(ab_report_path),
            "--native-report",
            str(native_report_path),
            "--sidecar-report",
            str(sidecar_report_path),
            "--sidecar-footprint",
            str(sidecar_dir),
            "--native-footprint",
            str(native_dir),
            "--min-size-reduction-mb",
            "0.001",
            check=False,
        )

        self.assertEqual(2, result.returncode)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertFalse(report["latency"]["passed"])


if __name__ == "__main__":
    unittest.main()
