import importlib.util
import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = REPO_ROOT / "third_party/rvc_sidecar/vox_rvc_server.py"


def load_sidecar_module():
    spec = importlib.util.spec_from_file_location("vox_rvc_server", SIDECAR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the Vox Studio RVC sidecar module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RvcSidecarProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sidecar = load_sidecar_module()

    def test_multipart_parser_preserves_binary_audio_and_fields(self):
        boundary = "voxstudio-test-boundary"
        audio = b"\x00\x01\xff\x00voice\r\nbytes"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model_id"\r\n\r\n'
            "carth_rvc_hq\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="chunk.pcm"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("ascii") + audio + f"\r\n--{boundary}--\r\n".encode("ascii")

        fields = self.sidecar.parse_multipart(
            f"multipart/form-data; boundary={boundary}",
            body,
        )

        self.assertEqual(b"carth_rvc_hq", fields["model_id"])
        self.assertEqual(audio, fields["audio"])

    def test_model_id_pattern_rejects_path_traversal(self):
        pattern = self.sidecar.MODEL_ID_PATTERN
        self.assertIsNotNone(pattern.fullmatch("carth_rvc_hq"))
        self.assertIsNone(pattern.fullmatch("../outside"))
        self.assertIsNone(pattern.fullmatch("carth/model"))

    def test_health_is_available_while_runtime_warms(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.sidecar.RvcService(Path(directory))
            health = service.health()

        self.assertTrue(health["ok"])
        self.assertEqual("loading", health["cuda_version"])
        self.assertIn("warming up", health["message"])

    def test_source_envelope_gain_preserves_performance_dynamics(self):
        gain = self.sidecar.source_envelope_gain(0.2, 0.1, 0.0)
        neutral_gain = self.sidecar.source_envelope_gain(0.2, 0.1, 1.0)
        blended_gain = self.sidecar.source_envelope_gain(0.2, 0.1, 0.5)

        self.assertAlmostEqual(2.0, gain)
        self.assertAlmostEqual(1.0, neutral_gain)
        self.assertAlmostEqual(math.sqrt(2.0), blended_gain)

    def test_source_envelope_gain_clamps_mix_rate(self):
        self.assertAlmostEqual(
            2.0,
            self.sidecar.source_envelope_gain(0.2, 0.1, -1.0),
        )
        self.assertAlmostEqual(
            1.0,
            self.sidecar.source_envelope_gain(0.2, 0.1, 2.0),
        )

    def test_silence_gate_rejects_room_noise_but_not_speech(self):
        self.assertTrue(self.sidecar.should_gate_silence(0.0008, 0.004))
        self.assertFalse(self.sidecar.should_gate_silence(0.003, 0.02))

    def test_model_switch_reuses_the_loaded_content_and_pitch_models(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory)
            model_dir = model_root / "voice_two"
            model_dir.mkdir()
            weights = model_dir / "voice.pth"
            index = model_dir / "voice.index"
            weights.write_bytes(b"weights")
            index.write_bytes(b"index")
            (model_dir / "model.json").write_text(
                json.dumps(
                    {
                        "pth_path": str(weights),
                        "index_path": str(index),
                    }
                ),
                encoding="utf-8",
            )
            service = self.sidecar.RvcService(model_root)
            service.runtime = {"runtime": "ready"}
            previous = object()
            service.engine = previous

            with mock.patch.object(self.sidecar, "RealtimeRvc") as realtime:
                service._load_engine("voice_two", 0)

        realtime.assert_called_once_with(
            service.runtime,
            weights.resolve(),
            index.resolve(),
            0,
            previous,
        )


if __name__ == "__main__":
    unittest.main()
