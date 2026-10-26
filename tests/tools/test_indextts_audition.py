import contextlib
import io
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
