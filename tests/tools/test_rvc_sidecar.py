import importlib.util
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
