"""Fixed offline IndexTTS-2.5 audition adapter; no training or donor character data.

The engine stays in the existing external installation. Voice Lab owns resource
admission, locking and cancellation. The generic audition worker owns claims,
source hashes, seeds, WAVs and per-take receipts.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
from pathlib import Path
import sys

EMOTION_ORDER = ('happy', 'angry', 'sad', 'afraid', 'disgusted',
                 'melancholic', 'surprised', 'calm')
LICENSE = 'bilibili-model-license'


def settings(job):
    vector = job.get('emotionVector')
    if not isinstance(vector, list) or len(vector) != 8:
        raise ValueError('IndexTTS requires an eight-value emotionVector')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= .8
           for v in vector) or sum(vector) > .800001:
        raise ValueError('Emotion values must be finite, nonnegative and sum to at most 0.8')
    return dict(emo_vector=[float(v) for v in vector], emo_alpha=1.0,
                use_random=False, use_emo_text=False, duration_factor=1.0,
                do_sample=True, top_p=.8, top_k=30, temperature=.8,
                num_beams=3, repetition_penalty=10.0, max_mel_tokens=800,
                max_text_tokens_per_segment=120)


def engine_files(recipe):
    if recipe.get('licenseAcknowledged') != LICENSE:
        raise ValueError('IndexTTS requires its bilibili-model-license acknowledgment')
    model = Path(recipe['modelPath']).resolve()
    engine = model.parent
    required = [engine / 'PROVENANCE.json', engine / 'pyproject.toml',
                engine / 'uv.lock', engine / 'LICENSE',
                engine / 'indextts/infer_v2_5.py', model / 'config.yaml',
                model / 'gpt.pth', model / 'codec.pth', model / 's2mel.pth',
                model / 'LICENSE']
    if model.name != 'checkpoints' or any(not p.is_file() for p in required):
        raise ValueError('Complete existing IndexTTS-2.5 checkpoints/installation required')
    provenance = json.loads(required[0].read_text(encoding='utf-8-sig'))
    revisions = {m.get('revision') for m in provenance.get('models', [])}
    if recipe['modelRevision'] not in revisions:
        raise ValueError('IndexTTS model revision disagrees with installation provenance')
    if 'version: 2.5' not in (model / 'config.yaml').read_text(encoding='utf-8'):
        raise ValueError('Only the inspected IndexTTS-2.5 configuration is supported')
    sources = [p for folder in ('indextts', 'backends')
               for p in (engine / folder).rglob('*.py')]
    return sorted(set([Path(__file__), *required, *sources]))


def offline_environment(engine):
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'DO_NOT_TRACK'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HOME'] = str(engine / '.hf_home')
    os.environ['HF_HUB_CACHE'] = str(engine / 'checkpoints/hf_cache')
    os.environ['MODELSCOPE_CACHE'] = str(engine / '.modelscope_cache')

    def network_guard(event, args):
        if event not in ('socket.connect', 'socket.getaddrinfo'):
            return
        address = args[1] if event == 'socket.connect' else args[0]
        host = address[0] if isinstance(address, tuple) else address
        if str(host).lower() == 'localhost':
            return
        try:
            if ipaddress.ip_address(str(host).strip('[]')).is_loopback:
                return
        except ValueError:
            pass
        raise OSError('IndexTTS audition is offline; non-loopback networking refused')

    sys.addaudithook(network_guard)


def load(recipe, torch):
    model_root = Path(recipe['modelPath']).resolve()
    engine = model_root.parent
    offline_environment(engine)
    # The pinned upstream module uses a cwd-relative HF_HUB_CACHE on import.
    os.chdir(engine)
    sys.path.insert(0, str(engine))
    free, total = torch.cuda.mem_get_info()
    cap = 11_000_000_000
    if free < cap + 8_000_000_000:
        raise RuntimeError('IndexTTS needs its 11 GB allocator cap plus 8 GB GPU reserve')
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)
    from indextts.infer_v2_5 import IndexTTS2
    return IndexTTS2(cfg_path=str(model_root / 'config.yaml'), model_dir=str(model_root),
                     use_bf16=True, device='cuda:0', use_cuda_kernel=False,
                     use_deepspeed=False, use_accel=False, use_torch_compile=False,
                     use_qwen_emo=False)


def generate(model, reference, job):
    import numpy as np
    effective = settings(job)
    result = model.infer(spk_audio_prompt=reference, text=job['text'],
                         output_path=None, lang='EN', **effective)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError('IndexTTS returned no native PCM audio')
    rate, pcm = result
    native = np.asarray(pcm)
    if native.dtype != np.int16:
        raise RuntimeError('Unexpected IndexTTS native PCM type; inspect before conversion')
    audio = native.reshape(-1).astype(np.float32) / 32768.0
    return audio, int(rate), dict(**effective, emotion_order=EMOTION_ORDER,
                                 license=LICENSE, dtype='bfloat16',
                                 native_encoding='int16; lossless scaling to float WAV')
