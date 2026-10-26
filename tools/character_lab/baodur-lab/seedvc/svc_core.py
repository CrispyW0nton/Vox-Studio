"""Seed-VC V2 conversion core shared by convert.py and check_offline.py (import after common.setup_seedvc()).

Follows VoiceConversionWrapper.convert_voice_with_streaming (seed-vc modules/v2/vc_wrapper.py, commit 51383ef)
step for step - same 30 s DiT window, 16-frame chunk overlap and crossfade, AR chunking - with three changes:
1. length_adjust is honoured in timbre-only mode (upstream applies it only on the AR path);
2. the waveform is returned directly (upstream's generator yields nothing unless it MP3-encodes every chunk);
3. it takes in-memory waves, so several of the character's clips can form one prompt, and caches reference features.
Precision: content tokens, style vector, length regulation, CFM and BigVGAN run in fp32 by default, matching how
train.py computed its features; fp16=True restores upstream's fp16 autocast for HuBERT/ASTRAL and the AR model.
"""
from __future__ import annotations

import numpy as np
import torch

import common as C


def vocode(vocoder, mel):
    """BigVGAN forward without its final hard clamp to [-1, 1] (use_tanh_at_final is false for the 22k model).

    The CFM can render hotter than the source (6-8 dB on the 3-step CPU check), and the clamp would clip those
    peaks before any gain stage; unclamped floats are levelled afterwards by common.finalize().
    """
    x = vocoder.conv_pre(mel)
    for i in range(vocoder.num_upsamples):
        for up in vocoder.ups[i]:
            x = up(x)
        xs = None
        for j in range(vocoder.num_kernels):
            y = vocoder.resblocks[i * vocoder.num_kernels + j](x)
            xs = y if xs is None else xs + y
        x = xs / vocoder.num_kernels
    x = vocoder.conv_post(vocoder.activation_post(x))
    return torch.tanh(x) if getattr(vocoder, "use_tanh_at_final", False) else x


class Reference:
    def __init__(self, key, seconds, mel, content, style, prompt, narrow=None):
        self.key, self.seconds = key, seconds
        self.mel, self.content, self.style, self.prompt, self.narrow = mel, content, style, prompt, narrow


class Converter:
    def __init__(self, wrapper, device, fp16: bool = False, convert_style: bool = False):
        self.w = wrapper
        self.device = device
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.convert_style = convert_style
        self._refs: dict = {}
        needed = ["cfm", "cfm_length_regulator", "content_extractor_wide", "style_encoder", "vocoder"]
        if convert_style:
            needed += ["content_extractor_narrow", "ar", "ar_length_regulator"]
        for name in needed:  # the AR model and narrow quantizer stay on the CPU unless style conversion is on
            getattr(wrapper, name).to(device)
        wrapper.eval()
        if convert_style:
            wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096,
                                    dtype=torch.float16 if self.fp16 else torch.float32, device=device)

    def clear_cache(self) -> None:
        """Call after swapping CFM weights: the prompt condition depends on the fine-tuned length regulator."""
        self._refs.clear()

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16)

    def _tensor(self, wave):
        return torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None].to(self.device)

    @torch.inference_mode()
    def reference(self, key, wave22) -> Reference:
        if key in self._refs:
            return self._refs[key]
        w = self.w
        wave22 = wave22[: C.MODEL_SR * (w.dit_max_context_len - 5)]  # upstream caps the prompt at 25 s
        w16 = self._tensor(C.resample(wave22, C.MODEL_SR, C.FEATURE_SR))
        mel = w.mel_fn(self._tensor(wave22))
        with self._autocast():
            content = w._process_content_features(w16, is_narrow=False)
            narrow = None
            if self.convert_style:
                narrow = w.duration_reduction_func(w._process_content_features(w16, is_narrow=True)[0], 1)[0]
        style = w.compute_style(w16).float()
        prompt, _ = w.cfm_length_regulator(content, ylens=torch.LongTensor([mel.size(2)]).to(self.device))
        ref = Reference(key, len(wave22) / C.MODEL_SR, mel, content, style, prompt, narrow)
        self._refs[key] = ref
        return ref

    @torch.inference_mode()
    def convert(self, source22, ref: Reference, *, diffusion_steps: int = 30, length_adjust: float = 1.0,
                intelligibility_cfg_rate: float = 0.7, similarity_cfg_rate: float = 0.7, top_p: float = 0.9,
                temperature: float = 1.0, repetition_penalty: float = 1.0, seed: int = 0) -> np.ndarray:
        """Return the prompt character's rendering of source22 (22.05 kHz float32) at 22.05 kHz."""
        w, dev = self.w, self.device
        torch.manual_seed(seed)
        src16 = self._tensor(C.resample(source22, C.MODEL_SR, C.FEATURE_SR))
        src_mel_len = w.mel_fn(self._tensor(source22)).size(2)
        with self._autocast():
            src_content = w._process_content_features(src16, is_narrow=False)
        cfg_rates = [float(intelligibility_cfg_rate), float(similarity_cfg_rate)]
        prompt_len = ref.mel.size(2)
        overlap_wave = w.overlap_frame_len * w.hop_size
        state = {"pieces": [], "previous": None, "processed": 0}

        def render(cond_chunk, is_last: bool) -> bool:
            cat = torch.cat([ref.prompt, cond_chunk.to(ref.prompt.dtype)], dim=1)
            total = cat.size(1)
            mel = w.cfm.inference(cat, torch.LongTensor([total]).to(dev), ref.mel, ref.style, diffusion_steps,
                                  inference_cfg_rate=cfg_rates)
            mel = mel[:, :, prompt_len:total]
            wave = vocode(w.vocoder, mel.float()).squeeze().float().cpu().numpy()
            if state["processed"] == 0 and is_last:
                state["pieces"].append(wave)
                return True
            if state["processed"] == 0:
                state["pieces"].append(wave[:-overlap_wave])
            elif is_last:
                state["pieces"].append(w.crossfade(state["previous"], wave, overlap_wave))
                return True
            else:
                state["pieces"].append(w.crossfade(state["previous"], wave[:-overlap_wave], overlap_wave))
            state["previous"] = wave[-overlap_wave:]
            state["processed"] += mel.size(2) - w.overlap_frame_len
            return False

        if not self.convert_style:
            target_len = max(1, int(round(src_mel_len * float(length_adjust))))
            cond, _ = w.cfm_length_regulator(src_content, ylens=torch.LongTensor([target_len]).to(dev))
            max_source_window = C.MODEL_SR // w.hop_size * w.dit_max_context_len - prompt_len
            if max_source_window <= 2 * w.overlap_frame_len:
                raise ValueError("reference too long for the 30 s DiT context")
            while state["processed"] < cond.size(1):
                start = state["processed"]
                is_last = start + max_source_window >= cond.size(1)
                if render(cond[:, start: start + max_source_window], is_last):
                    break
        else:
            with self._autocast():
                src_narrow = w._process_content_features(src16, is_narrow=True)
            src_reduced = w.duration_reduction_func(src_narrow[0], 1)[0]
            max_chunk = w.ar_max_content_len - len(ref.narrow)
            for i in range(0, len(src_reduced), max_chunk):
                is_last = i + max_chunk >= len(src_reduced)
                with self._autocast():
                    ar_cond = w.ar_length_regulator(torch.cat([ref.narrow, src_reduced[i: i + max_chunk]], dim=0)[None])[0]
                    ar_out = w.ar.generate(ar_cond, ref.content, compiled_decode_fn=None, top_p=top_p,
                                           temperature=temperature, repetition_penalty=repetition_penalty)
                mel_len = int(src_mel_len / src_content.size(-1) * ar_out.size(-1) * float(length_adjust))
                chunk_cond, _ = w.cfm_length_regulator(ar_out, ylens=torch.LongTensor([max(1, mel_len)]).to(dev))
                if render(chunk_cond, is_last):
                    break
        return np.concatenate(state["pieces"]).astype(np.float32)
