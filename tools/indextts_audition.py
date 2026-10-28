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


def pre_clamp_gain_db(job):
    value = job.get('preClampGainDb', 0)
    if type(value) not in (int, float) or not -18 <= value <= 0:
        raise ValueError('preClampGainDb must be a finite number from -18 to 0')
    return float(value)


def settings(job):
    pre_clamp_gain_db(job)
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


def infer_with_headroom(model, reference, job, effective):
    """Apply constant gain before upstream PCM clipping, on this owned model only."""
    gain_db = pre_clamp_gain_db(job)
    kwargs = dict(spk_audio_prompt=reference, text=job['text'],
                  output_path=None, lang='EN', **effective)
    if gain_db == 0:
        return model.infer(**kwargs), {}

    import torch
    vocoder = getattr(model, 'bigvgan', None)
    final_conv = getattr(vocoder, 'conv_post', None)
    if (not isinstance(vocoder, torch.nn.Module) or
            getattr(vocoder, 'use_tanh_at_final', None) is not False or
            not isinstance(final_conv, torch.nn.Module)):
        raise RuntimeError('Pre-clamp headroom requires the inspected BigVGAN non-tanh final convolution')
    gain = 10 ** (gain_db / 20)
    peaks = []
    verified = []
    errors = []
    expected_output = None

    def attenuate(_module, _args, output):
        nonlocal expected_output
        try:
            if not torch.is_tensor(output) or not output.is_floating_point() or output.numel() == 0:
                raise RuntimeError('Unexpected vocoder output for pre-clamp headroom')
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError('Nonfinite vocoder output')
            # Final waveform gain in fp32 avoids bf16/fp16 rounding before PCM.
            scaled = output.float() * gain
            row = dict(rawPeak=float(output.detach().float().abs().max().item()),
                       adjustedPeak=float(scaled.detach().abs().max().item()))
            peaks.append(row)
            print(json.dumps(dict(stage='pre-clamp-headroom', id=job.get('id'),
                                  gainDb=gain_db, **row)), flush=True)
            if row['adjustedPeak'] >= 1:
                raise RuntimeError('Pre-clamp gain leaves no PCM headroom; preview a quieter job')
            expected_output = scaled.detach().clone()
            return scaled
        except Exception as error:
            errors.append(error)
            raise

    def observe_output(_module, _args, output):
        try:
            if len(peaks) != len(verified) + 1 or expected_output is None:
                raise RuntimeError('Expected exactly one observed final convolution per vocoder call')
            if not torch.is_tensor(output) or not torch.equal(output, expected_output):
                raise RuntimeError('Vocoder modified the scaled waveform after its final convolution')
            peaks[-1]['vocoderOutputVerified'] = True
            verified.append(peaks[-1])
        except Exception as error:
            errors.append(error)
            raise

    # The configured BigVGAN also clamps inside forward(), before returning.
    # Hook its final convolution so gain precedes that first irreversible clamp.
    handles = []
    try:
        handles.append(final_conv.register_forward_hook(attenuate))
        handles.append(vocoder.register_forward_hook(observe_output))
        result = model.infer(**kwargs)
        if errors:
            raise RuntimeError('Headroom check failed during inference') from errors[0]
        if not peaks or len(peaks) != len(verified):
            raise RuntimeError('Vocoder headroom hook was not observed; refuse unqualified PCM')
    finally:
        for handle in reversed(handles):
            handle.remove()
    return result, dict(preClampGainDb=gain_db, preClampGain=gain, vocoderPeaks=peaks,
                        preClampStage='bigvgan.conv_post before vocoder and PCM hard clamps',
                        preClampMathDtype='float32',
                        preClampProcessing='Constant gain; no normalization or limiting')


def generate(model, reference, job):
    import numpy as np
    effective = settings(job)
    result, headroom = infer_with_headroom(model, reference, job, effective)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError('IndexTTS returned no native PCM audio')
    rate, pcm = result
    native = np.asarray(pcm)
    if native.dtype != np.int16:
        raise RuntimeError('Unexpected IndexTTS native PCM type; inspect before conversion')
    if headroom:
        if not native.size:
            raise RuntimeError('No native PCM samples after observed vocoder output')
        bound = min(32766, math.ceil(32767 * max(p['adjustedPeak'] for p in headroom['vocoderPeaks'])))
        if int(np.abs(native.astype(np.int32)).max()) > bound:
            raise RuntimeError('Returned native PCM exceeds observed pre-clamp headroom')
        headroom['nativePCMPeakBound'] = bound
    audio = native.reshape(-1).astype(np.float32) / 32768.0
    return audio, int(rate), dict(**effective, **headroom, emotion_order=EMOTION_ORDER,
                                 license=LICENSE, dtype='bfloat16',
                                 native_encoding='int16; lossless scaling to float WAV')
