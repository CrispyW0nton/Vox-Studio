"""Bounded opt-in paragraph continuity, using the real planner and render chain."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
from fastapi.testclient import TestClient
SIDECAR=Path(__file__).resolve().parents[2]/'third_party/voxcpm_sidecar'
sys.path.insert(0,str(SIDECAR))
spec=importlib.util.spec_from_file_location('vox_short_paragraph_server',SIDECAR/'vox_voxcpm_server.py')
vox=importlib.util.module_from_spec(spec);sys.modules[spec.name]=vox;spec.loader.exec_module(vox)
TEXT="We're starting Nullstar with one simple question: can magic and science feel good together? First, we'll build a small playable room in Godot. We'll test movement, the layout, and one interaction before worrying about detail. Then we'll carry those decisions into Unreal, rebuilding the gameplay there. I'll walk you through each step, including what needs to change along the way."
PROMPT='I want to show a little workflow for making quick and easy animations.'

class ShortParagraphTests(unittest.TestCase):
    def test_continuous_monologue_keeps_long_script_in_one_native_call(self):
        text = TEXT + ' ' + TEXT
        self.assertGreater(len(text), 400)
        model=mock.Mock();model.tts_model.sample_rate=48000
        model.generate.return_value=np.full(4800,.1,dtype=np.float32)
        with mock.patch.object(vox,'load_profile',return_value=(self.profile,self.root,self.reference)),mock.patch.object(vox,'get_tts_model',return_value=model):
            response=TestClient(vox.app).post('/render_text',data={'voice_id':'test','text':text,'delivery':'natural','mode':'standard','segmentation':'continuous-monologue'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(model.generate.call_count,1)
        self.assertEqual(model.generate.call_args.kwargs['text'],text)
        self.assertEqual(model.generate.call_args.kwargs['prompt_text'],PROMPT)
        self.assertEqual(response.headers['x-vox-section-count'],'1')
        self.assertEqual(len(response.content),9600)

    def test_continuous_monologue_is_bounded_before_loading_any_voice(self):
        for text,mode in [('x'*1201,'standard'),(' '.join(['a']*221),'standard'),('First.\nSecond.','standard'),('[calm] Hello.','standard'),('Hello.','storytelling')]:
            with self.subTest(text=text,mode=mode),mock.patch.object(vox,'load_profile') as loader,mock.patch.object(vox,'get_tts_model') as model:
                response=TestClient(vox.app).post('/render_text',data={'voice_id':'test','text':text,'mode':mode,'segmentation':'continuous-monologue'})
                self.assertEqual(response.status_code,400)
                loader.assert_not_called();model.assert_not_called()

    def test_monologue_exact_limits_and_old_paragraph_limit(self):
        for text in ['x'*1200,' '.join(['a']*220)]:
            sections=vox.text_synthesis_sections(self.profile,self.root,self.reference,text,'natural','standard','continuous-monologue')
            self.assertEqual(len(sections),1)
            self.assertEqual(sections[0].text,text)
        with self.assertRaises(ValueError):
            vox.text_synthesis_sections(self.profile,self.root,self.reference,'x'*401,'natural','standard','short-paragraph')

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.reference=self.root/'reference.wav';self.reference.write_bytes(b'fixture')
        self.profile={'name':'Baseline','engine':'VoxCPM2','reference_audio':'reference.wav','cfg_value':2,'inference_timesteps':12,'use_controlled_cloning':False,'use_stable_character_identity':False,'style_anchors':[{'audio':'reference.wav','prompt_text':PROMPT,'features':{}}]}
    def plan(self,**kwargs):
        return vox.text_synthesis_sections(self.profile,self.root,self.reference,TEXT,'natural','standard',**kwargs)
    def test_legacy_sections_and_seeds_unchanged(self):
        sections=self.plan();self.assertEqual(len(sections),2)
        self.assertEqual([s.seed for s in sections],[24493453,1678050119])
        self.assertTrue(sections[0].text.endswith('about detail.'))
        self.assertTrue(sections[1].text.startswith("Then we'll carry"))
    def test_one_section_retains_continuation_options(self):
        sections=self.plan(segmentation='short-paragraph');self.assertEqual(len(sections),1)
        s=sections[0];self.assertEqual(s.text,TEXT);self.assertEqual(s.instruction,'')
        options=vox.generation_options(s.text,s.prompt_path,s.prompt_text,s.reference_path,self.profile,s.instruction)
        self.assertEqual(options,{'text':TEXT,'prompt_wav_path':str(self.reference),'reference_wav_path':str(self.reference),'prompt_text':PROMPT,'cfg_value':2.0,'inference_timesteps':12,'normalize':False})
    def test_render_uses_one_generate_call_without_added_internal_gap(self):
        model=mock.Mock();model.tts_model.sample_rate=48000
        model.generate.return_value=np.full(4800,.1,dtype=np.float32)
        with mock.patch.object(vox,'load_profile',return_value=(self.profile,self.root,self.reference)),mock.patch.object(vox,'get_tts_model',return_value=model):
            response=TestClient(vox.app).post('/render_text',data={'voice_id':'test','text':TEXT,'delivery':'natural','mode':'standard','segmentation':'short-paragraph'})
        self.assertEqual(response.status_code,200);self.assertEqual(model.generate.call_count,1)
        self.assertEqual(response.headers['x-vox-section-count'],'1');self.assertEqual(len(response.content),9600)
        self.assertEqual(model.generate.call_args.kwargs['prompt_text'],PROMPT)
    def test_invalid_short_paragraph_rejected_before_profile_or_model(self):
        for text,mode in [('x'*401,'standard'),(' '.join(['a']*81),'standard'),('First.\nSecond.','standard'),('[warm] Hi','standard'),('*sigh* Hi','standard'),('(whispering) Hi','standard'),('<speak>Hi</speak>','standard'),('Hi','storytelling')]:
            with self.subTest(text=text,mode=mode),mock.patch.object(vox,'load_profile') as loader,mock.patch.object(vox,'get_tts_model') as model:
                response=TestClient(vox.app).post('/render_text',data={'voice_id':'test','text':text,'mode':mode,'segmentation':'short-paragraph'})
                self.assertEqual(response.status_code,400);loader.assert_not_called();model.assert_not_called()

if __name__=='__main__':unittest.main()
