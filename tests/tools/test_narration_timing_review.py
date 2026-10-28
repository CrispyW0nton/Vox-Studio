import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).parents[2]/'tools/review_narration_timing.py'
spec=importlib.util.spec_from_file_location('narration_timing_review',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TimingReviewTests(unittest.TestCase):
    def test_sentence_boundary_and_last_unpunctuated_phrase(self):
        words=[{'word':x} for x in ['One',' thought.',' Why?',' Unfinished']]
        self.assertEqual([len(s) for s in module.sentence_groups(words)],[2,1,1])

    def test_declines_sparse_and_unvoiced_pitch(self):
        t=np.arange(100)*.01
        self.assertIsNone(module.terminal_rise(t,np.full(100,np.nan),np.ones(100),1)['review'])
        self.assertIsNone(module.terminal_rise(t,np.full(100,120),np.zeros(100),1)['review'])

    def test_rising_and_falling_contours(self):
        t=np.arange(100)*.01;p=np.where(t<.7,120.,150.)
        self.assertTrue(module.terminal_rise(t,p,np.ones(100),1)['review'])
        self.assertFalse(module.terminal_rise(t,p[::-1],np.ones(100),1)['review'])

    def test_local_slowdown_is_visible_without_automatic_rejection(self):
        words=[]
        for i in range(16):
            start=i*.3 if i<8 else 2.4+(i-8)*.7
            words.append({'word':' word', 'start':start, 'end':start+.2})
        report=module.pacing_observations(words, step_words=8)
        self.assertGreater(report['windows'][0]['wordsPerMinute'], report['windows'][1]['wordsPerMinute']*2)
        self.assertNotIn('approved', report)

    def test_estimated_gap_is_not_a_waveform_cut_point(self):
        report=module.pacing_observations([
            {'word':'Done.', 'start':0, 'end':.5},
            {'word':' Next', 'start':1.6, 'end':2}])
        self.assertEqual(report['longGapReview'][0]['estimatedSeconds'], 1.1)
        self.assertEqual(report['windows'], [])
        self.assertIn('no automatic audio edits', report['meaning'])

    def test_invalid_or_zero_length_windows_do_not_divide_by_zero(self):
        with self.assertRaises(ValueError):
            module.pacing_observations([], window_words=1)
        report=module.pacing_observations([{'word':' x','start':1,'end':1}]*8)
        self.assertEqual(report['windows'], [])

if __name__=='__main__':unittest.main()
