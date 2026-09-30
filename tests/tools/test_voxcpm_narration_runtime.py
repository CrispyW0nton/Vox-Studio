import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'third_party/voxcpm_sidecar'))
from vox_runtime import runtime_options, validate_runtime_adapter


class NarrationRuntimeTests(unittest.TestCase):
    def test_existing_trained_voice_keeps_optimized_adapter_runtime(self):
        runtime = runtime_options('default')
        self.assertTrue(runtime.optimize)
        self.assertTrue(runtime.allocate_lora)
        validate_runtime_adapter('default', True)

    def test_narration_does_not_silently_drop_trained_identity(self):
        validate_runtime_adapter('narration', False)
        with self.assertRaisesRegex(ValueError, 'trained LoRA'):
            validate_runtime_adapter('narration', True)

    def test_unknown_runtime_is_rejected_before_native_loading(self):
        for value in ('', 'arbitrary', '../default', 'NARRATION'):
            with self.assertRaises(ValueError):
                validate_runtime_adapter(value, False)


if __name__ == '__main__':
    unittest.main()
