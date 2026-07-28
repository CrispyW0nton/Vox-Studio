import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = (
    REPO_ROOT
    / "third_party/performance_mirror_sidecar/vox_performance_mirror_server.py"
)


def load_sidecar_module():
    spec = importlib.util.spec_from_file_location(
        "vox_performance_mirror_server",
        SIDECAR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the Performance Mirror sidecar module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PerformanceMirrorSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sidecar = load_sidecar_module()

    def test_voice_id_rejects_path_traversal(self):
        pattern = self.sidecar.VOICE_ID_PATTERN
        self.assertIsNotNone(pattern.fullmatch("VqtR5ry1ddcv59m6Wvqg"))
        self.assertIsNone(pattern.fullmatch("../outside"))
        self.assertIsNone(pattern.fullmatch("voice/profile"))

    def test_profile_reference_prefers_explicit_mirror_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles_root = Path(directory)
            profile_root = profiles_root / "voice_test"
            styles_root = profile_root / "styles"
            styles_root.mkdir(parents=True)
            (profile_root / "reference.wav").write_bytes(b"default")
            (styles_root / "mirror.wav").write_bytes(b"mirror")
            (profile_root / "profile.json").write_text(
                json.dumps(
                    {
                        "voice_id": "voice_test",
                        "performance_mirror_reference": "styles/mirror.wav",
                    }
                ),
                encoding="utf-8",
            )

            reference = self.sidecar.resolve_voice_reference(
                profiles_root,
                "voice_test",
            )

        self.assertEqual("mirror.wav", reference.name)

    def test_profile_reference_falls_back_to_reference_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles_root = Path(directory)
            profile_root = profiles_root / "voice_test"
            profile_root.mkdir(parents=True)
            expected = profile_root / "reference.wav"
            expected.write_bytes(b"default")

            reference = self.sidecar.resolve_voice_reference(
                profiles_root,
                "voice_test",
            )

        self.assertEqual(expected.resolve(), reference)

    def test_profile_reference_cannot_escape_profile_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles_root = Path(directory)
            profile_root = profiles_root / "voice_test"
            profile_root.mkdir(parents=True)
            (profiles_root / "outside.wav").write_bytes(b"outside")
            (profile_root / "profile.json").write_text(
                json.dumps(
                    {
                        "voice_id": "voice_test",
                        "performance_mirror_reference": "../outside.wav",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                self.sidecar.resolve_voice_reference(
                    profiles_root,
                    "voice_test",
                )

    def test_queue_policy_keeps_only_the_newest_live_blocks(self):
        blocks = [b"one", b"two", b"three", b"four"]

        retained, dropped = self.sidecar.retain_latest_blocks(blocks, 2)

        self.assertEqual([b"three", b"four"], retained)
        self.assertEqual(2, dropped)

    def test_health_reports_runtime_install_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self.sidecar.PerformanceMirrorService(
                root / "missing-seed-vc",
                root / "profiles",
            )

            health = service.health()

        self.assertTrue(health["ok"])
        self.assertEqual("performance-mirror", health["engine"])
        self.assertFalse(health["runtime_installed"])
        self.assertIn("Install", health["message"])


if __name__ == "__main__":
    unittest.main()
