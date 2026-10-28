import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_TOOL_PATH = REPO_ROOT / "tools/train_character_rvc.py"


def load_training_tool_module():
    spec = importlib.util.spec_from_file_location(
        "train_character_rvc",
        TRAINING_TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the character RVC training tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterRvcTrainingToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_training_tool_module()

    def test_safe_identifier_rejects_paths(self):
        self.assertEqual(
            "atton_rvc_character",
            self.tool.safe_identifier("atton_rvc_character", "Model id"),
        )
        with self.assertRaises(ValueError):
            self.tool.safe_identifier("../outside", "Model id")

    def test_prepare_training_sources_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            first = sources / "first.wav"
            second = sources / "second.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            training_root = root / "training"
            stale_root = training_root / "atton"
            stale_root.mkdir(parents=True)
            (stale_root / "line_0003.wav").write_bytes(b"stale")

            dataset = self.tool.prepare_training_sources(
                {"selected_sources": [str(first), str(second)]},
                training_root,
                "atton",
            )

            self.assertEqual(b"first", (dataset / "line_0001.wav").read_bytes())
            self.assertEqual(b"second", (dataset / "line_0002.wav").read_bytes())
            self.assertFalse((dataset / "line_0003.wav").exists())


if __name__ == "__main__":
    unittest.main()
