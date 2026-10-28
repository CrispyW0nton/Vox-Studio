import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
import indextts_audition as index
import narration_audition as audition
import voice_studio_lab as lab


class IndexAuditionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = self.root / 'lab/index-tts'
        self.model = self.engine / 'checkpoints'
        files = {'PROVENANCE.json': json.dumps({'models': [{'revision': 'a' * 40}]}),
                 'pyproject.toml': 'installed', 'uv.lock': 'locked', 'LICENSE': 'terms',
                 'indextts/infer_v2_5.py': 'source', 'checkpoints/config.yaml': 'version: 2.5',
                 'checkpoints/gpt.pth': 'gpt', 'checkpoints/codec.pth': 'codec',
                 'checkpoints/s2mel.pth': 'decoder', 'checkpoints/LICENSE': 'model terms'}
        for name, text in files.items():
            path = self.engine / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        ref = self.root / 'original.wav'
        ref.write_bytes(b'original authorized reference')
        self.recipe = dict(backend='indextts25', consentConfirmed=True,
                           licenseAcknowledged=index.LICENSE, batchId='index-test',
                           reference=str(ref), referenceSha256=audition.sha(ref),
                           referenceText='Original speech.', modelPath=str(self.model),
                           modelRevision='a' * 40, outputRoot=str(self.root / 'VoiceAuditionV1'),
                           jobs=[dict(id='alarm', seed=42, text='Get away from him!',
                                      emotionVector=[0, .1, 0, .7, 0, 0, 0, 0])])

    def test_emotion_validation_and_backend_isolation(self):
        audition.validate(self.recipe)
        for vector in ([0] * 7, [0] * 9, [0, 0, 0, .81, 0, 0, 0, 0],
                       [.5, .5, 0, 0, 0, 0, 0, 0], [0, 0, 0, True, 0, 0, 0, 0],
                       [0, 0, 0, float('nan'), 0, 0, 0, 0],
                       [0, 0, 0, float('inf'), 0, 0, 0, 0], None):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                index.settings(dict(emotionVector=vector))
        for backend in ('qwen', 'chatterbox'):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, 'only for IndexTTS'):
                audition.validate(dict(self.recipe, backend=backend))

    def test_license_and_provenance_required(self):
        for fields in ({'licenseAcknowledged': True}, {'modelRevision': 'b' * 40}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                audition.validate(dict(self.recipe, **fields))
        (self.model / 'codec.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'Complete existing'):
            audition.validate(self.recipe)

    def test_pre_clamp_gain_validation_and_backend_isolation(self):
        job = self.recipe['jobs'][0]
        for value in (True, False, None, '-6', float('nan'), float('inf'), -18.001, .001, -(10**400)):
            for validate in (index.settings, lambda j: audition.validate(dict(self.recipe, jobs=[j])),
                             lambda j: audition.preview(dict(self.recipe, jobs=[j]))):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'preClampGainDb'):
                    validate(dict(job, preClampGainDb=value))
        for value in (-18, -6.0, 0):
            audition.validate(dict(self.recipe, jobs=[dict(job, preClampGainDb=value)]))
        for backend in ('qwen', 'chatterbox'):
            for value in (-6, 0):
                other_job = {k: v for k, v in job.items() if k != 'emotionVector'}
                with self.subTest(backend=backend, value=value), self.assertRaisesRegex(ValueError, 'preClampGainDb'):
                    audition.validate(dict(self.recipe, backend=backend,
                                           jobs=[dict(other_job, preClampGainDb=value)]))

    def test_explicit_zero_retains_legacy_without_torch_or_hooks(self):
        import numpy as np

        class Model:
            @property
            def bigvgan(self):
                raise AssertionError('Default path must not inspect or hook the vocoder')

            def infer(self, **kwargs):
                return 22050, np.array([-32767, 0, 32767], dtype=np.int16)

        for value in (0, -0.0):
            with self.subTest(value=value), mock.patch.dict(sys.modules, {'torch': None}):
                audio, _, metadata = index.generate(Model(), self.recipe['reference'],
                                                    dict(self.recipe['jobs'][0], preClampGainDb=value))
            np.testing.assert_array_equal(audio, np.array([-32767, 0, 32767], dtype=np.float32) / 32768)
            self.assertNotIn('preClampGainDb', metadata)

    def test_pre_clamp_gain_invalidates_preview_before_output(self):
        with mock.patch.object(audition.importlib.metadata, 'version', return_value='test'):
            first = audition.preview(self.recipe)
            self.recipe['jobs'][0]['preClampGainDb'] = -6
            second = audition.preview(self.recipe)
            self.assertNotEqual(first['revision'], second['revision'])
            with self.assertRaisesRegex(ValueError, 'Preview revision changed'):
                audition.run(self.recipe, first['revision'])
        self.assertFalse(Path(self.recipe['outputRoot']).exists())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional actual PyTorch CPU fixture unavailable')
    def test_pre_clamp_headroom_preserves_signal_before_native_clipping(self):
        import numpy as np
        import torch

        class Vocoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_post = torch.nn.Identity()
                self.use_tanh_at_final = False

            def forward(self, value):
                # Installed BigVGAN config selects this earlier hard clamp.
                return torch.clamp(self.conv_post(value), -1, 1)

        class Model:
            def __init__(self):
                self.bigvgan = Vocoder()
                self.input = torch.tensor([[[-1.8, -1.2, -.5, 0., .5, 1.2, 1.8]]])

            def infer(self, **kwargs):
                # Raw final conv -> vocoder clamp -> PCM clamp -> int16.
                output = self.bigvgan(self.input)
                return 22050, torch.clamp(32767 * output, -32767, 32767).to(torch.int16).numpy()

        model = Model()
        original = model.input.clone()
        job = self.recipe['jobs'][0]
        legacy, _, old_settings = index.generate(model, self.recipe['reference'], job)
        self.assertEqual(legacy[-1], 32767 / 32768)
        self.assertEqual(legacy[-2], legacy[-1])  # Distinct over-unity peaks were lost.
        self.assertNotIn('preClampGainDb', old_settings)
        with contextlib.redirect_stdout(io.StringIO()):
            clean, _, settings = index.generate(model, self.recipe['reference'], dict(job, preClampGainDb=-6))
        gain = 10 ** (-6 / 20)
        np.testing.assert_allclose(clean, original.numpy().reshape(-1) * gain * (32767 / 32768),
                                   rtol=0, atol=1 / 32768)
        self.assertGreater(clean[-1], clean[-2])
        self.assertLess(max(abs(clean)), 32767 / 32768)
        self.assertAlmostEqual(settings['vocoderPeaks'][0]['rawPeak'], 1.8, places=6)
        self.assertAlmostEqual(settings['vocoderPeaks'][0]['adjustedPeak'], 1.8 * gain, places=6)
        self.assertEqual(settings['preClampGainDb'], -6)
        torch.testing.assert_close(model.input, original, rtol=0, atol=0)
        self.assertEqual(len(model.bigvgan._forward_hooks), 0)
        self.assertEqual(len(model.bigvgan.conv_post._forward_hooks), 0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional actual PyTorch CPU fixture unavailable')
    def test_multiple_vocoder_calls_and_low_precision_gain_math(self):
        import numpy as np
        import torch

        class Vocoder(torch.nn.Module):
            use_tanh_at_final = False

            def __init__(self):
                super().__init__()
                self.conv_post = torch.nn.Identity()

            def forward(self, x):
                return torch.clamp(self.conv_post(x), -1, 1)

        class Model:
            def __init__(self, dtype):
                self.bigvgan = Vocoder()
                self.chunks = [torch.tensor([-.3, .3], dtype=dtype),
                               torch.tensor([-1.8, 1.8], dtype=dtype)]

            def infer(self, **kwargs):
                chunks = [torch.clamp(32767 * self.bigvgan(c), -32767, 32767).to(torch.int16).numpy()
                          for c in self.chunks]
                return 22050, np.concatenate(chunks)

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            model = Model(dtype)
            with self.subTest(dtype=dtype), contextlib.redirect_stdout(io.StringIO()):
                audio, _, metadata = index.generate(model, self.recipe['reference'],
                                                    dict(self.recipe['jobs'][0], preClampGainDb=-6))
            expected = torch.cat(model.chunks).float().numpy() * 10**(-6/20) * (32767/32768)
            np.testing.assert_allclose(audio, expected, rtol=0, atol=1/32768)
            self.assertEqual(len(metadata['vocoderPeaks']), 2)
            self.assertTrue(all(p['vocoderOutputVerified'] for p in metadata['vocoderPeaks']))
            self.assertEqual(metadata['preClampMathDtype'], 'float32')
            self.assertEqual(len(model.bigvgan._forward_hooks), 0)
            self.assertEqual(len(model.bigvgan.conv_post._forward_hooks), 0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional actual PyTorch CPU fixture unavailable')
    def test_caught_guard_post_gain_changes_and_unobserved_pcm_are_rejected(self):
        import numpy as np
        import torch

        class Vocoder(torch.nn.Module):
            use_tanh_at_final = False

            def __init__(self, alter):
                super().__init__()
                self.conv_post = torch.nn.Identity()
                self.alter = alter

            def forward(self, x):
                out = torch.clamp(self.conv_post(x), -1, 1)
                return out * .5 if self.alter else out

        class Model:
            def __init__(self, mode):
                self.bigvgan = Vocoder(mode == 'alter')
                self.mode = mode

            def infer(self, **kwargs):
                first = self.bigvgan(torch.tensor([.5]))
                if self.mode == 'swallow':
                    try:
                        self.bigvgan(torch.tensor([10.]))
                    except RuntimeError:
                        pass  # Mimic a fallback that discards a failed segment.
                pcm = torch.clamp(32767*first, -32767, 32767).to(torch.int16).numpy()
                return 22050, np.concatenate([pcm, np.array([32767], dtype=np.int16)]) if self.mode == 'bypass' else pcm

        errors = dict(alter='modified the scaled waveform', swallow='Headroom check failed',
                      bypass='exceeds observed pre-clamp headroom')
        for mode, message in errors.items():
            model = Model(mode)
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, message):
                index.generate(model, self.recipe['reference'], dict(self.recipe['jobs'][0], preClampGainDb=-6))
            self.assertEqual(len(model.bigvgan._forward_hooks), 0)
            self.assertEqual(len(model.bigvgan.conv_post._forward_hooks), 0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional actual PyTorch CPU fixture unavailable')
    def test_pre_clamp_hook_cleanup_and_unobserved_output_rejection(self):
        import numpy as np
        import torch

        class Vocoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_post = torch.nn.Identity()
                self.use_tanh_at_final = False

            def forward(self, value):
                return torch.clamp(self.conv_post(value), -1, 1)

        class Model:
            def __init__(self, mode):
                self.bigvgan = Vocoder()
                self.mode = mode

            def infer(self, **kwargs):
                if self.mode == 'bypass':
                    return 22050, np.zeros(8, dtype=np.int16)
                value = dict(nonfinite=float('nan'), too_loud=10.).get(self.mode, .5)
                output = (torch.tensor([value]),) if self.mode == 'tuple' else torch.tensor([value])
                if self.mode == 'empty':
                    output = torch.empty(0)
                elif self.mode == 'integer':
                    output = torch.tensor([1], dtype=torch.int32)
                self.bigvgan(output)
                raise RuntimeError('inference failed')

        expected = dict(bypass='not observed', nonfinite='Nonfinite', too_loud='no PCM headroom',
                        tuple='Unexpected vocoder output', empty='Unexpected vocoder output',
                        integer='Unexpected vocoder output', failure='inference failed')
        job = dict(self.recipe['jobs'][0], preClampGainDb=-6)
        for mode, error in expected.items():
            model = Model(mode)
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, error):
                index.generate(model, self.recipe['reference'], job)
            self.assertEqual(len(model.bigvgan._forward_hooks), 0)
            self.assertEqual(len(model.bigvgan.conv_post._forward_hooks), 0)
        with self.assertRaisesRegex(RuntimeError, 'inspected BigVGAN'):
            index.generate(object(), self.recipe['reference'], job)
        model = Model('failure')
        model.bigvgan.use_tanh_at_final = True
        with self.assertRaisesRegex(RuntimeError, 'non-tanh final convolution'):
            index.generate(model, self.recipe['reference'], job)
        self.assertEqual(len(model.bigvgan.conv_post._forward_hooks), 0)

    def test_engine_code_and_emotion_invalidate_preview_before_output(self):
        with mock.patch.object(audition.importlib.metadata, 'version', return_value='test'):
            first = audition.preview(self.recipe)
            (self.engine / 'indextts/infer_v2_5.py').write_text('changed source')
            second = audition.preview(self.recipe)
            self.assertNotEqual(first['revision'], second['revision'])
            with self.assertRaisesRegex(ValueError, 'Preview revision changed'):
                audition.run(self.recipe, first['revision'])
            self.recipe['jobs'][0]['emotionVector'] = [0] * 8
            self.assertNotEqual(second['revision'], audition.preview(self.recipe)['revision'])
        self.assertFalse(Path(self.recipe['outputRoot']).exists())

    def test_native_pcm_forwarding_and_single_use_receipt(self):
        import numpy as np
        model = mock.Mock()
        pcm = np.tile(np.array([-32768, -100, 0, 100, 32767], dtype=np.int16), 8820)
        model.infer.return_value = (22050, pcm)
        torch = mock.MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.max_memory_allocated.return_value = 123
        torch.inference_mode.side_effect = contextlib.nullcontext
        with mock.patch.dict(sys.modules, {'torch': torch}), \
             mock.patch.dict(audition.os.environ), \
             mock.patch.object(index, 'load', return_value=model), \
             mock.patch.object(audition.importlib.metadata, 'version', return_value='test'), \
             contextlib.redirect_stdout(io.StringIO()):
            revision = audition.preview(self.recipe)['revision']
            audition.run(self.recipe, revision)
            with self.assertRaises(FileExistsError):
                audition.run(self.recipe, revision)
        kwargs = model.infer.call_args.kwargs
        self.assertEqual(kwargs['spk_audio_prompt'], self.recipe['reference'])
        self.assertEqual(kwargs['emo_vector'], [0, .1, 0, .7, 0, 0, 0, 0])
        self.assertFalse(kwargs['use_random'])
        self.assertIsNone(kwargs['output_path'])
        receipt = json.loads((Path(self.recipe['outputRoot']) / 'Raw/indextts25/alarm.json').read_text())
        self.assertEqual(receipt['settings']['license'], index.LICENSE)
        self.assertEqual(receipt['referenceSha256'], self.recipe['referenceSha256'])
        self.assertEqual(receipt['sampleRate'], 22050)
        self.assertEqual(receipt['previewRevision'], revision)
        import soundfile as sf
        reopened, _ = sf.read(receipt['path'], dtype='float32')
        np.testing.assert_array_equal(reopened, pcm.astype(np.float32) / 32768.)

    def test_owner_uses_installed_runtime_and_preserves_foreign_lock(self):
        work = self.root / 'job'
        work.mkdir()
        script = work / 'recipe.json'
        script.write_text(json.dumps(self.recipe))
        jobfile = work / 'job.json'
        jobfile.write_text(json.dumps(dict(backend='indextts25', script=str(script))))
        roots = dict(lab=self.root / 'lab', data=self.root / 'data', voices=self.root / 'voices')
        roots['voices'].mkdir()
        lock = roots['voices'] / 'gpu_train.lock'
        lock.write_text('{"owner":"another-session"}')
        preview = mock.Mock(returncode=0, stdout=json.dumps(dict(revision='test-revision')), stderr='')
        with mock.patch.object(lab.subprocess, 'run', return_value=preview) as run, \
             mock.patch.object(lab, 'resource_gpu', return_value='GPU-compute'), \
             mock.patch.object(lab, 'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            lab.operation(jobfile, 'preview', '', roots)
            self.assertEqual(run.call_args.args[0][0], str(self.engine / '.venv/Scripts/python.exe'))
            with self.assertRaises(FileExistsError):
                lab.operation(jobfile, 'render', 'test-revision', roots)
            execute.assert_not_called()
        self.assertEqual(json.loads(lock.read_text())['owner'], 'another-session')


if __name__ == '__main__':
    unittest.main()
