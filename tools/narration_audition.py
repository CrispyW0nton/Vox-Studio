"""Local Vox Studio audition backends, also launched by the native Voice Lab.

Preview binds the recipe, source recording, model files and this worker. Render
requires that exact revision, retains raw float WAVs, and never overwrites a take.
All model loading is offline. This module never uploads audio or trains a model.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

BACKENDS = {"qwen": "qwen-tts", "chatterbox": "chatterbox-tts", "indextts25": "indextts"}

# Omission retains the existing restrained narration recipe. Each phrase can
# opt into more expression without changing its speaker reference or watermark.
CHATTERBOX_CONTROLS = {
    'exaggeration': ('exaggeration', 0.35, 0.0, 2.0),
    'cfgWeight': ('cfg_weight', 0.3, 0.0, 1.0),
    'temperature': ('temperature', 0.8, 0.1, 1.5),
}


def chatterbox_settings(job: dict) -> dict:
    settings = {}
    for key, (native_key, default, low, high) in CHATTERBOX_CONTROLS.items():
        value = job.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be a finite number from {low} to {high}')
        settings[native_key] = float(value)
    return settings

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()

def validate(recipe: dict) -> None:
    if recipe.get('backend') not in BACKENDS or recipe.get('consentConfirmed') is not True:
        raise ValueError('Known local backend and explicit reference consent required')
    if 'nonStreamingMode' in recipe:
        if recipe['backend'] != 'qwen' or type(recipe['nonStreamingMode']) is not bool:
            raise ValueError('nonStreamingMode must be a boolean and is supported only for Qwen')
    if 'cooldownSeconds' in recipe and (type(recipe['cooldownSeconds']) is not int or not 0 <= recipe['cooldownSeconds'] <= 120):
        raise ValueError('cooldownSeconds must be an integer from 0 to 120')  # Oct 10: rest between jobs (gentle 3090)
    if 'xVectorOnlyMode' in recipe and (recipe['backend'] != 'qwen' or type(recipe['xVectorOnlyMode']) is not bool):
        raise ValueError('xVectorOnlyMode must be a boolean and is supported only for Qwen')  # Oct 9: timbre without the reference cadence
    if not re.fullmatch('[a-z][a-z0-9_-]{0,60}', recipe.get('batchId', '')):
        raise ValueError('Safe batch ID required')
    if not Path(recipe['reference']).is_file() or sha(Path(recipe['reference'])) != recipe['referenceSha256']:
        raise ValueError('Reference changed or missing')
    if not recipe.get('referenceText', '').strip():
        raise ValueError('An accurate reference transcript is required')
    if not Path(recipe['modelPath']).is_dir() or not re.fullmatch('[a-f0-9]{40}', recipe['modelRevision']):
        raise ValueError('Pinned local model required')
    if recipe['backend'] == 'indextts25':
        import indextts_audition
        indextts_audition.engine_files(recipe)
    root = Path(recipe['outputRoot'])
    if not root.is_absolute() or root.name != 'VoiceAuditionV1':
        raise ValueError('Output must be the explicitly named VoiceAuditionV1 directory')
    jobs = recipe['jobs']
    if not 1 <= len(jobs) <= 12:
        raise ValueError('One to twelve bounded audition jobs required')
    names = set()
    for job in jobs:
        if 'emotionVector' in job or recipe['backend'] == 'indextts25':
            if recipe['backend'] != 'indextts25':
                raise ValueError('emotionVector is supported only for IndexTTS-2.5')
            import indextts_audition
            indextts_audition.settings(job)
        if any(key in job for key in CHATTERBOX_CONTROLS):
            if recipe['backend'] != 'chatterbox':
                raise ValueError('Expression controls are supported only for Chatterbox')
            chatterbox_settings(job)
        name = job['id']
        if not re.fullmatch('[a-z][a-z0-9_-]{0,45}', name) or name in names:
            raise ValueError('Unique safe take IDs required')
        names.add(name)
        if not isinstance(job['seed'], int) or not 0 <= job['seed'] < 2**31:
            raise ValueError('Bounded integer seed required')
        if not 1 <= len(job['text']) <= 1200 or len(job['text'].split()) > 200:
            raise ValueError('Audition text too long')

def preview(recipe: dict) -> dict:
    validate(recipe)
    files = [dict(path=str(p.relative_to(recipe['modelPath'])), sha256=sha(p))
             for p in sorted(Path(recipe['modelPath']).rglob('*')) if p.is_file()]
    versions = {}
    for name in [BACKENDS[recipe['backend']], 'torch', 'torchaudio', 'transformers', 'numpy', 'soundfile']:
        versions[name] = importlib.metadata.version(name)
    evidence = dict(recipe=recipe, modelFiles=files, versions=versions, workerSha256=sha(Path(__file__)), python=sys.version)
    if recipe['backend'] == 'indextts25':
        import indextts_audition
        evidence['engineFiles'] = [dict(path=str(p), sha256=sha(p))
                                   for p in indextts_audition.engine_files(recipe)]
    return dict(revision=hashlib.sha256(canonical(evidence)).hexdigest(), **evidence)

def run(recipe: dict, expected: str) -> None:
    evidence = preview(recipe)
    if evidence['revision'] != expected:
        raise ValueError('Preview revision changed')
    root = Path(recipe['outputRoot']) / 'Raw' / recipe['backend']
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive claim: a failed or interrupted batch is inspected, never retried silently.
    claim = root / (recipe['batchId'] + '.claim.json')
    with claim.open('x', encoding='utf-8') as f:
        json.dump(evidence, f, indent=2)
    for job in recipe['jobs']:
        if (root / (job['id']+'.wav')).exists():
            raise ValueError('Existing take is protected')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import soundfile as sf
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    if recipe['backend'] == 'qwen':
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(recipe['modelPath'], device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
        prompt = model.create_voice_clone_prompt(ref_audio=recipe['reference'], ref_text=recipe['referenceText'], x_vector_only_mode=recipe.get('xVectorOnlyMode', False))
    elif recipe['backend'] == 'chatterbox':
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_local(recipe['modelPath'], device='cuda')
        model.prepare_conditionals(recipe['reference'], exaggeration=0.35)
    else:
        import indextts_audition
        model = indextts_audition.load(recipe, torch)
    print(json.dumps(dict(stage='loaded', backend=recipe['backend'], seconds=time.time()-started)), flush=True)
    for job in recipe['jobs']:
        if (Path(recipe['outputRoot']) / 'STOP_REQUESTED.txt').exists():
            raise RuntimeError('Stopped by user steering')
        random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed']); torch.cuda.manual_seed_all(job['seed'])
        begin = time.time()
        print(json.dumps(dict(stage='rendering', id=job['id'])), flush=True)
        with torch.inference_mode():
            if recipe['backend'] == 'qwen':
                non_streaming_mode = recipe.get('nonStreamingMode', False)
                waves, sr = model.generate_voice_clone(text=job['text'], language='English', voice_clone_prompt=prompt,
                    non_streaming_mode=non_streaming_mode, max_new_tokens=2400)
                audio = np.asarray(waves[0], dtype=np.float32).reshape(-1)
                settings = dict(dtype='bfloat16', attention='sdpa', x_vector_only_mode=recipe.get('xVectorOnlyMode', False),
                    non_streaming_mode=non_streaming_mode, max_new_tokens=2400)
            elif recipe['backend'] == 'chatterbox':
                settings = chatterbox_settings(job)
                wave = model.generate(job['text'], **settings)
                audio = wave.detach().cpu().numpy().reshape(-1).astype(np.float32)
                sr = model.sr
                settings = dict(**settings, watermark='native Perth retained')
            else:
                audio, sr, settings = indextts_audition.generate(model, recipe['reference'], job)
        if not len(audio) or not np.isfinite(audio).all() or not 1 < len(audio)/sr < 150:
            raise RuntimeError('Invalid generated audio')
        out = root / (job['id']+'.wav')
        with out.open('xb') as f:
            sf.write(f, audio, sr, subtype='FLOAT', format='WAV')
        reopened, reopened_sr = sf.read(out)
        assert reopened_sr == sr and len(reopened) == len(audio)
        if recipe.get('cooldownSeconds'):
            time.sleep(recipe['cooldownSeconds'])
        receipt = dict(backend=recipe['backend'], modelRevision=recipe['modelRevision'], referenceSha256=recipe['referenceSha256'],
            previewRevision=expected, job=job, settings=settings, path=str(out), sha256=sha(out), sampleRate=sr,
            seconds=len(audio)/sr, peak=float(np.max(np.abs(audio))), outOfRangeSamples=int(np.sum(np.abs(audio)>=1)),
            elapsedSeconds=time.time()-begin, peakGpuAllocatedBytes=torch.cuda.max_memory_allocated(),
            processing=('Native int16 PCM losslessly scaled to float WAV' if recipe['backend'] == 'indextts25'
                        else 'Raw native float WAV') + '; no EQ, denoise, pitch or time processing', listeningStatus='unreviewed')
        out.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(dict(stage='rendered',id=job['id'],seconds=receipt['seconds'],elapsed=receipt['elapsedSeconds'])),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('operation',choices=['preview','render'])
    ap.add_argument('recipe',type=Path)
    ap.add_argument('--revision')
    args=ap.parse_args()
    recipe=json.loads(args.recipe.read_text(encoding='utf-8-sig'))
    if args.operation=='preview':
        print(json.dumps(preview(recipe),indent=2))
    else:
        if not args.revision: ap.error('render requires --revision')
        run(recipe,args.revision)

if __name__=='__main__': main()
