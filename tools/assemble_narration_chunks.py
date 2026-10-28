"""Assemble reviewed Vox thought-sized takes without changing speech samples.

Input JSON: profileId, chunks [{path, sha256, text, pauseAfterSeconds}].
Pauses are editorial choices, not automatic prosody repairs. All source tails
are retained. New output directory only; no render, upload or profile mutation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import wave


def assemble(plan, output):
    if output.exists():
        raise FileExistsError(output)
    if not plan.get('profileId') or not plan.get('chunks'):
        raise ValueError('A profile identity and nonempty chunk list are required')
    pieces, ledger, offset = [], [], 0
    for i, chunk in enumerate(plan['chunks']):
        text=chunk['text'].strip()
        if not text or len(text)>400 or len(text.split())>80:
            raise ValueError('Each thought must fit the Vox short-paragraph contract')
        if not re.search(r'[.!?][\"\u201d\u2019]*$',text):
            raise ValueError('Finish the thought before the edit; do not cut mid-clause')
        pause=float(chunk.get('pauseAfterSeconds',0))
        if not math.isfinite(pause) or not 0<=pause<=2:
            raise ValueError('Pause must be finite and between zero and two seconds')
        if i==len(plan['chunks'])-1 and pause:
            raise ValueError('Final take must not request a following join')
        path=Path(chunk['path'])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=chunk['sha256']:
            raise ValueError('Source take changed; review its hash before assembling')
        with wave.open(str(path),'rb') as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(48000,1,2,'NONE'):
                raise ValueError('Expected a native Vox 48 kHz mono PCM16 master')
            count=w.getnframes();pcm=w.readframes(count)
        if count==0 or len(pcm)!=count*2:
            raise ValueError('Empty or truncated source take')
        gap=round(pause*48000)
        pieces.extend([pcm,bytes(gap*2)])
        ledger.append({'source':str(path.resolve()),'sourceSha256':digest,'text':text,
            'startFrame':offset,'speechFrames':count,'insertedPauseFrames':gap,
            'startSeconds':offset/48000,'durationSeconds':count/48000})
        offset+=count+gap
    output.mkdir(parents=True,exist_ok=False)
    destination=output/'joined.wav'
    with wave.open(str(destination),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(48000);w.writeframes(b''.join(pieces))
    report={'profileId':plan['profileId'],'chunks':ledger,'durationSeconds':offset/48000,
        'output':str(destination.resolve()),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'speechSamplesUnchanged':True,'sourceTailsPreserved':True,'respeed':False,'pitchEditing':False,
        'status':'Listening required: complete endings, coherent starts, breaths, identity and energy across every join. Punctuation is not proof of natural delivery.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(assemble(json.loads(args.plan.read_text(encoding='utf-8-sig')),args.output),indent=2))
