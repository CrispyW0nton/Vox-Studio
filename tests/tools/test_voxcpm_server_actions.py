from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient


SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "voxcpm_sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))
SCRIPT_PATH = SIDECAR_ROOT / "vox_voxcpm_server.py"
SPEC = importlib.util.spec_from_file_location("vox_voxcpm_server_actions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VocalActionServerTests(unittest.TestCase):
    def test_render_text_caches_repeated_direct_action_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actions = root / "actions"
            actions.mkdir()
            action_path = actions / "sigh.wav"
            action_path.write_bytes(b"wave")
            reference = root / "reference.wav"
            reference.write_bytes(b"wave")
            profile = {
                "name": "Test Character",
                "vocal_actions": {
                    "sigh": [
                        {
                            "audio": "actions/sigh.wav",
                            "tags": ["restrained"],
                            "mode": "direct",
                        }
                    ]
                },
            }
            audio = np.full(480, 0.1, dtype=np.float32)
            with (
                mock.patch.object(
                    MODULE,
                    "load_profile",
                    return_value=(profile, root, reference),
                ),
                mock.patch.object(MODULE, "resolve_profile_asset", return_value=None),
                mock.patch.object(
                    MODULE,
                    "load_vocal_action_audio",
                    return_value=(audio, 48000),
                ) as loader,
            ):
                response = TestClient(MODULE.app).post(
                    "/render_text",
                    data={
                        "voice_id": "test",
                        "text": "*quiet sigh* *quiet sigh*",
                        "delivery": "natural",
                        "mode": "standard",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-vox-vocal-action-count"], "2")
        self.assertEqual(loader.call_count, 1)

    def test_storytelling_selects_variant_from_original_action_cue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actions = root / "actions"
            actions.mkdir()
            for name in ("laugh_warm.wav", "laugh_pained.wav"):
                (actions / name).write_bytes(b"wave")
            reference = root / "reference.wav"
            reference.write_bytes(b"wave")
            profile = {
                "vocal_actions": {
                    "laugh": [
                        {
                            "audio": "actions/laugh_warm.wav",
                            "tags": ["warm"],
                            "intensity": 0.5,
                        },
                        {
                            "audio": "actions/laugh_pained.wav",
                            "tags": ["pained", "restrained"],
                            "intensity": 0.25,
                        },
                    ]
                }
            }

            sections = MODULE.text_synthesis_sections(
                profile,
                root,
                reference,
                "*quiet pained laugh*",
                "natural",
                MODULE.STORYTELLING_MODE,
            )

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].audio_path.name, "laugh_pained.wav")


if __name__ == "__main__":
    unittest.main()
