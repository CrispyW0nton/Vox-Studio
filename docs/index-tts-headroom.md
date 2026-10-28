# IndexTTS output headroom

The inspected IndexTTS-2.5 configuration hard-clamps its final vocoder
convolution, then converts the resulting waveform to bounded int16 PCM.
A loud waveform can lose peaks before the later float WAV exists. Lowering
the finished WAV cannot recover them.

An IndexTTS audition job may set `"preClampGainDb": -6` to apply a constant
gain before that native conversion. The allowed range is -18 through 0 dB;
omitting the field or setting zero retains the existing behavior. Other
backends reject the field. A changed value requires a fresh source-bound
preview and its exact revision before rendering.

The worker attaches a temporary forward hook to the owned BigVGAN module's
final `conv_post` convolution, before its first hard clamp. This option
supports the inspected `use_tanh_at_final: false` path and rejects other
variants. It scales a new output tensor, preserves the original, and removes
both hooks when inference exits or fails. An observe-only hook verifies that
each vocoder call uses one final convolution and returns its scaled waveform
unchanged. Gain arithmetic uses float32 even for lower precision outputs.
The upstream installation,
weights, reference recording, model admission, GPU ownership, license checks
and exclusive output claims remain unchanged.

Each vocoder call records its raw and adjusted sample peaks in the take's
settings and render log. Nonfinite or unsupported output, an unobserved hook,
or an adjusted peak reaching unity stops the job. The worker does not silently
normalize, limit, retry or select a different gain. If more attenuation is
needed, prepare a new job and preview it.

The worker retains a guard failure even if the upstream generator catches the
exception, and checks returned PCM against the observed headroom. A fallback
cannot quietly replace a checked chunk with full-scale, unobserved audio.

Six decibels is an example, not a guarantee for every performance. Check the
actual native waveform and leave headroom for resampling and mixing. This
option preserves sample peaks at one known conversion; it cannot remove
distortion already present in a reference, model or earlier stage. Listen for
identity, phrasing, intelligibility and artifacts before selecting a take.

The focused tests include an actual PyTorch CPU fixture that reproduces the
upstream final-convolution, vocoder-clamp, PCM-clamp and int16 order. It needs an existing PyTorch runtime;
the ordinary CPU test environment may skip that optional fixture. No speech
model is loaded for this test. A release qualification must run the fixture
and a fresh native audition, not infer success from a skipped test.
