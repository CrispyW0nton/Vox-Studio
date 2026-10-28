# ADR 0002: Isolate the Reference Mirror Seed-VC Fallback

- Status: Accepted
- Date: 2026-07-27

## Context

VoxCPM2 can produce high-quality character speech after a phrase or monologue
has been captured, but regeneration does not preserve every moment of the
performer's waveform. Live character acting needs a different path that keeps
the source timing, pauses, emphasis, energy, and pitch movement while changing
speaker identity.

Character-trained RVC checkpoints hold a known character identity more reliably
than a short zero-shot reference. Seed-VC still provides a useful real-time
fallback for characters that do not yet have a trained checkpoint. Its upstream
project is GPL-3.0, uses a large Python/CUDA dependency set, and downloads model
weights separately.

## Decision

Vox Studio prefers the trained RVC model assigned to the selected character.
When that model is unavailable, it uses the official pinned Seed-VC real-time
tiny model through a separate localhost process on port `18910`.

- The C++ app owns capture, monitoring, broadcast routing, recording, and UI.
- The sidecar owns model loading, stateful conversion, silence gating, SOLA
  crossfading, and source-energy matching.
- The app sends 48 kHz mono PCM in 360 ms blocks and drops stale waiting blocks
  if inference ever falls behind.
- Character IDs resolve only inside the local VoxCPM profile directory.
- The official Seed-VC checkout, Python environment, model cache, and character
  references remain outside the repository and application binary.
- VoxCPM2 remains the HQ Phrase, Monologue, Storytelling, and TTS engine.
- Imported RVC remains available for selecting a checkpoint directly.

## Consequences

Reference Mirror has a one-time local engine setup and requires an NVIDIA CUDA
GPU. Its monitoring delay is higher than a conventional DSP voice changer, and
its zero-shot identity match is weaker than a trained character model, but it
preserves the performed signal more directly than phrase regeneration. The
process boundary keeps the upstream license and heavyweight runtime explicit,
allows independent upgrades, and prevents those dependencies from becoming
part of the native app.
