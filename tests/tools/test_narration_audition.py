import importlib.util
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('audition',Path(__file__).resolve().parents[2]/'tools/narration_audition.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AuditionContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ref=self.root/'ref.wav';self.ref.write_bytes(b'preserved reference')
        self.recipe=dict(backend='qwen',batchId='test-batch',consentConfirmed=True,reference=str(self.ref),referenceSha256=a.sha(self.ref),
            referenceText='Original speech.',modelPath=str(self.root),modelRevision='a'*40,outputRoot=str(self.root/'VoiceAuditionV1'),
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

if __name__=='__main__': unittest.main()
