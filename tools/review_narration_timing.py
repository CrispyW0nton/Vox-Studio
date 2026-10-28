"""Offline review cues for a Vox take; never an automatic acting approval.

Uses a local Whisper model for approximate word timing and pYIN for a
confidence-screened terminal-pitch observation. Does not alter source audio.
Sentence punctuation and alignment can be wrong: review the flagged excerpt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


def sentence_groups(words):
    groups, current = [], []
    for word in words:
        current.append(word)
        if re.search(r'[.!?][\"\u201d\u2019]*$', word['word'].strip()):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def pacing_observations(words, window_words=8, step_words=4):
    """Keep local slowdowns visible; an episode-wide WPM hides them.

    Word times are ASR estimates, not safe audio edit points. A slow window
    or a long gap can be appropriate emphasis and is never a failed take.
    """
    if window_words < 2 or step_words < 1:
        raise ValueError('Use at least two words and a positive step')
    windows = []
    for index in range(0, len(words)-window_words+1, step_words):
        group = words[index:index+window_words]
        start, end = float(group[0]['start']), float(group[-1]['end'])
        duration = end-start
        if duration <= 0:
            continue
        windows.append({'text': ''.join(w['word'] for w in group).strip(),
            'start': start, 'end': end,
            'wordsPerMinute': round(60*len(group)/duration, 2)})
    gaps = []
    for previous, current in zip(words, words[1:]):
        gap = float(current['start'])-float(previous['end'])
        if gap >= .9:
            gaps.append({'after': previous['word'].strip(),
                'before': current['word'].strip(),
                'start': float(previous['end']), 'end': float(current['start']),
                'estimatedSeconds': round(gap, 3)})
    return {'windows': windows, 'longGapReview': gaps,
        'meaning': 'Local timing review cues only; no universal target rate and no automatic audio edits.'}


def terminal_rise(times, pitches, probabilities, end, minimum_semitones=2.0):
    """Compare two terminal windows; abstain on too little voiced evidence."""
    import numpy as np
    times, pitches, probabilities = map(np.asarray, (times, pitches, probabilities))
    valid = np.isfinite(pitches) & (pitches > 0) & (probabilities >= .7)
    earlier = pitches[valid & (times >= end-.65) & (times < end-.30)]
    later = pitches[valid & (times >= end-.30) & (times <= end-.04)]
    if len(earlier) < 6 or len(later) < 6:
        return {'status': 'insufficient-voiced-evidence', 'review': None}
    delta = 12*math.log2(float(np.median(later))/float(np.median(earlier)))
    return {'status': 'measured', 'deltaSemitones': round(delta, 3),
            'review': delta >= minimum_semitones,
            'meaning': 'Possible terminal rise; not proof of unnatural or questioning speech'}


def review(wav, local_model, output):
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import soundfile as sf
    import librosa
    from faster_whisper import WhisperModel
    if output.exists():
        raise FileExistsError(output)
    if not (local_model/'model.bin').is_file():
        raise ValueError('An existing local Whisper model is required; no download is performed')
    audio, sr = sf.read(wav, dtype='float32', always_2d=True)
    if audio.shape[1] != 1 or len(audio) == 0 or not np.all(np.isfinite(audio)):
        raise ValueError('Expected a finite, nonempty mono narration master')
    mono = audio[:, 0]
    model = WhisperModel(str(local_model), device='cpu', compute_type='int8', cpu_threads=6)
    segments, _ = model.transcribe(str(wav), language='en', word_timestamps=True, beam_size=5)
    words = [{'word': w.word, 'start': float(w.start), 'end': float(w.end), 'probability':float(w.probability)}
             for segment in segments for w in segment.words]
    groups = sentence_groups(words)
    sentences = []
    for i, group in enumerate(groups):
        start, end = group[0]['start'], group[-1]['end']
        clip_start = max(start, end-1.0)
        clip = mono[int(clip_start*sr):int(end*sr)]
        samples = librosa.resample(clip, orig_sr=sr, target_sr=16000)
        if len(samples) >= 1024:
            f0, voiced, probability = librosa.pyin(samples, fmin=65, fmax=420,
                sr=16000, frame_length=1024, hop_length=160, fill_na=np.nan)
            pitch = terminal_rise(np.arange(len(f0))*.01+clip_start, f0, probability, end)
        else:
            pitch = {'status':'insufficient-duration', 'review':None}
        text = ''.join(w['word'] for w in group).strip()
        gap = group[0]['start']-groups[i-1][-1]['end'] if i else None
        sentences.append({'text':text, 'start':start, 'end':end,
            'wordCount': len(group),
            'wordsPerMinute': round(60*len(group)/(end-start), 2) if end>start else None,
            'previousGapSeconds':round(gap, 3) if gap is not None else None,
            'shortTransitionReview':bool(gap is not None and gap < .22),
            'terminalPitch':pitch, 'statementRiseReview':not text.endswith('?') and pitch.get('review')})
    report = {'source':str(wav.resolve()), 'sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
        'durationSeconds':len(mono)/sr, 'sampleRate':sr, 'wordsPerMinute':len(words)*60/(len(mono)/sr),
        'clippedSamples':int(np.count_nonzero(abs(mono)>=.999)), 'words':words, 'sentences':sentences,
        'pacing': pacing_observations(words),
        'acceptance':'Review cues only. ASR timing, punctuation, pitch and thresholds do not establish acting, identity or naturalness.',
        'processing':'Read-only local ASR and pitch review; no voice conversion, respeed, flattening or upload.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized=json.dumps(report,indent=2)
    with output.open('x',encoding='utf-8') as f:f.write(serialized)
    print(json.dumps({k:v for k,v in report.items() if k not in ('words',)},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav',type=Path)
    parser.add_argument('--local-model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    review(args.wav,args.local_model,args.output)
