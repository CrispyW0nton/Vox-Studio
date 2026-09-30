import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave

spec=importlib.util.spec_from_file_location('chunks',Path(__file__).parents[2]/'tools/assemble_narration_chunks.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'take.wav';self.pcm=b'\x40\x00'*4800
        with wave.open(str(self.path),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(48000);w.writeframes(self.pcm)
        self.chunk={'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'text':'A complete thought.','pauseAfterSeconds':0}
    def tearDown(self):self.temp.cleanup()
    def test_preserves_each_speech_sample_and_exact_join(self):
        first=copy.copy(self.chunk);first['pauseAfterSeconds']=.35
        report=module.assemble({'profileId':'test','chunks':[first,self.chunk]},self.root/'out')
        with wave.open(report['output'],'rb') as w:data=w.readframes(w.getnframes())
        self.assertEqual(data,self.pcm+bytes(16800*2)+self.pcm)
        self.assertEqual(report['chunks'][1]['startFrame'],21600)
    def test_stale_source_is_rejected_before_writing(self):
        self.chunk['sha256']='0'*64
        with self.assertRaises(ValueError):module.assemble({'profileId':'test','chunks':[self.chunk]},self.root/'out')
        self.assertFalse((self.root/'out').exists())
    def test_mid_clause_and_nonfinite_pause_are_rejected(self):
        for field,value in [('text','An unfinished clause,'),('pauseAfterSeconds',float('nan'))]:
            chunk=copy.copy(self.chunk);chunk[field]=value
            with self.assertRaises(ValueError):module.assemble({'profileId':'test','chunks':[chunk]},self.root/'out')
    def test_existing_output_is_protected(self):
        out=self.root/'out';out.mkdir()
        with self.assertRaises(FileExistsError):module.assemble({'profileId':'test','chunks':[self.chunk]},out)

if __name__=='__main__':unittest.main()
