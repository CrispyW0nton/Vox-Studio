import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec=importlib.util.spec_from_file_location('audition',Path(__file__).resolve().parents[2]/'tools/narration_audition.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AuditionContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ref=self.root/'ref.wav';self.ref.write_bytes(b'preserved reference')
        self.model_root=self.root/'model';self.model_root.mkdir()
        self.recipe=dict(backend='qwen',batchId='test-batch',consentConfirmed=True,reference=str(self.ref),referenceSha256=a.sha(self.ref),
            referenceText='Original speech.',modelPath=str(self.model_root),modelRevision='a'*40,outputRoot=str(self.root/'VoiceAuditionV1'),
            jobs=[dict(id='intro',seed=42,text='A new explanation.')])
    def test_valid(self): a.validate(self.recipe)
    def test_changed_reference(self):
        self.ref.write_bytes(b'changed')
        with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_consent_required(self):
        self.recipe['consentConfirmed']=False
        with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_no_path_in_take_id(self):
        self.recipe['jobs'][0]['id']='../overwrite'
        with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_no_duplicate_jobs(self):
        self.recipe['jobs']*=2
        with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_bounded_text(self):
        self.recipe['jobs'][0]['text']='word '*201
        with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_explicit_output(self):
        self.recipe['outputRoot']=str(self.root)
        with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_non_streaming_mode_requires_boolean(self):
        for value in ['true', 'false', 0, 1, 0.0, None, [], {}]:
            with self.subTest(value=value):
                self.recipe['nonStreamingMode']=value
                with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_non_streaming_mode_is_qwen_only(self):
        self.recipe['backend']='chatterbox'
        a.validate(self.recipe)
        for value in [True, False]:
            with self.subTest(value=value):
                self.recipe['nonStreamingMode']=value
                with self.assertRaises(ValueError): a.validate(self.recipe)
    def test_non_streaming_mode_changes_preview_revision(self):
        with mock.patch.object(a.importlib.metadata, 'version', return_value='test'):
            default=a.preview(self.recipe)
            self.recipe['nonStreamingMode']=False
            disabled=a.preview(self.recipe)
            self.recipe['nonStreamingMode']=True
            enabled=a.preview(self.recipe)
            self.assertEqual(len({default['revision'], disabled['revision'], enabled['revision']}), 3)
            with self.assertRaisesRegex(ValueError, 'Preview revision changed'):
                a.run(self.recipe, disabled['revision'])
        self.assertFalse(Path(self.recipe['outputRoot']).exists())
    def test_qwen_mode_forwarding_and_receipt_without_model_loading(self):
        import numpy as np
        for label, option, effective in [('default', {}, False), ('enabled', {'nonStreamingMode':True}, True),
                                         ('disabled', {'nonStreamingMode':False}, False)]:
            with self.subTest(mode=label):
                recipe=dict(self.recipe, outputRoot=str(self.root/label/'VoiceAuditionV1'), **option)
                torch=mock.MagicMock()
                torch.cuda.is_available.return_value=True
                torch.cuda.max_memory_allocated.return_value=0
                torch.inference_mode.side_effect=contextlib.nullcontext
                qwen=mock.MagicMock()
                model=qwen.Qwen3TTSModel.from_pretrained.return_value
                prompt=object()
                model.create_voice_clone_prompt.return_value=prompt
                # Synthetic silence exercises the native WAV receipt path, never speech inference.
                model.generate_voice_clone.return_value=([np.zeros(48000, dtype=np.float32)], 24000)
                with mock.patch.dict(sys.modules, {'torch':torch, 'qwen_tts':qwen}), \
                     mock.patch.dict(a.os.environ), \
                     mock.patch.object(a.importlib.metadata, 'version', return_value='test'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    revision=a.preview(recipe)['revision']
                    a.run(recipe, revision)
                model.create_voice_clone_prompt.assert_called_once_with(
                    ref_audio=recipe['reference'], ref_text=recipe['referenceText'], x_vector_only_mode=False)
                model.generate_voice_clone.assert_called_once_with(
                    text=recipe['jobs'][0]['text'], language='English', voice_clone_prompt=prompt,
                    non_streaming_mode=effective, max_new_tokens=2400)
                receipt=json.loads((Path(recipe['outputRoot'])/'Raw/qwen/intro.json').read_text())
                self.assertIs(receipt['settings']['non_streaming_mode'], effective)
                self.assertEqual(receipt['previewRevision'], revision)
                self.assertEqual(receipt['referenceSha256'], recipe['referenceSha256'])

    def test_chatterbox_expression_bounds_and_backend(self):
        job = self.recipe['jobs'][0]
        job['exaggeration'] = 1.0
        with self.assertRaises(ValueError): a.validate(self.recipe)
        self.recipe['backend'] = 'chatterbox'
        a.validate(self.recipe)
        for field, invalid in [('exaggeration', [-.1, 2.1]), ('cfgWeight', [-.1, 1.1]),
                               ('temperature', [0, 1.6])]:
            for value in [*invalid, True, False, '0.8', None, float('nan'), float('inf')]:
                with self.subTest(field=field, value=value):
                    job[field] = value
                    with self.assertRaises(ValueError): a.validate(self.recipe)
            job.pop(field, None)

    def test_chatterbox_expression_invalidates_preview(self):
        self.recipe['backend'] = 'chatterbox'
        with mock.patch.object(a.importlib.metadata, 'version', return_value='test'):
            previous = a.preview(self.recipe)['revision']
            self.recipe['jobs'][0]['exaggeration'] = 1.0
            self.assertNotEqual(previous, a.preview(self.recipe)['revision'])
            with self.assertRaisesRegex(ValueError, 'Preview revision changed'):
                a.run(self.recipe, previous)
        self.assertFalse(Path(self.recipe['outputRoot']).exists())

    def test_chatterbox_expression_forwarding_and_watermark_receipt(self):
        import numpy as np
        self.recipe['backend'] = 'chatterbox'
        self.recipe['jobs'] = [dict(id='default', seed=1, text='A restrained line.'),
                              dict(id='fear', seed=2, text='Get away!', exaggeration=1.1,
                                   cfgWeight=.25, temperature=.7)]
        torch = mock.MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.max_memory_allocated.return_value = 0
        torch.inference_mode.side_effect = contextlib.nullcontext
        chatterbox = mock.MagicMock()
        model = chatterbox.ChatterboxTTS.from_local.return_value
        model.sr = 24000
        model.generate.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(48000, dtype=np.float32)
        with mock.patch.dict(sys.modules, {'torch': torch, 'chatterbox.tts': chatterbox}), \
             mock.patch.dict(a.os.environ), \
             mock.patch.object(a.importlib.metadata, 'version', return_value='test'), \
             contextlib.redirect_stdout(io.StringIO()):
            revision = a.preview(self.recipe)['revision']
            a.run(self.recipe, revision)
        self.assertEqual(model.generate.call_args_list, [
            mock.call('A restrained line.', exaggeration=.35, cfg_weight=.3, temperature=.8),
            mock.call('Get away!', exaggeration=1.1, cfg_weight=.25, temperature=.7)])
        receipt = json.loads((Path(self.recipe['outputRoot'])/'Raw/chatterbox/fear.json').read_text())
        self.assertEqual(receipt['settings'], dict(exaggeration=1.1, cfg_weight=.25,
                                                 temperature=.7, watermark='native Perth retained'))
        self.assertEqual(receipt['previewRevision'], revision)
        self.assertEqual(receipt['referenceSha256'], self.recipe['referenceSha256'])

if __name__=='__main__': unittest.main()
