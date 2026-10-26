# Local narration auditions

`tools/narration_audition.py` supplies experimental Qwen3-TTS, Chatterbox and IndexTTS-2.5 audition workers inside the Vox Studio repository. The native [Voice Lab](DESKTOP_VOICE_LAB.md) can import, preview and render these recipes. The production MCP and existing VoxCPM2 profiles remain unchanged.

IndexTTS recipes use `"backend": "indextts25"`, the existing external laboratory
`index-tts/checkpoints` directory and `"licenseAcknowledged": "bilibili-model-license"`.
This model has its own terms; it is not an OSI permissive model. Retain its license
and provenance with derivatives and review distribution terms before release.
No character lab, other project's voice, training or new download is used.
Each job requires `emotionVector`: eight finite nonnegative values, summing to at
most 0.8, ordered happy, angry, sad, afraid, disgusted, melancholic, surprised,
calm. The voice reference remains fixed while these controls change. They pass
directly to the pinned 2.5 API without hidden bias or post-production treatment.
The adapter uses BF16, English, fixed duration factor 1, no random exemplars,
no emotion-text model and an 11 GB GPU allocator cap with 8 GB free reserve.
Native PCM is losslessly scaled into float WAVs. Preview also binds the adapter,
upstream Python sources, configuration, model files and license. The existing
Voice Lab lock, host memory admission, cancellation and exclusive claims apply.
Emotion controls do not establish authentic acting; audition before selecting.

The worker supports `preview recipe.json` and `render recipe.json --revision <preview revision>`. Recipes bind the authorized source WAV hash and exact transcript, model snapshot, seed, bounded text and a dedicated `VoiceAuditionV1` output folder. Preview records dependency versions and hashes all model files. Render verifies the revision, claims the batch exclusively and uses local-only model loading. A failed batch remains for inspection. Render writes untouched float WAVs and provenance receipts; it cannot approve speaker identity or narration quality.

Qwen uses an isolated environment under `%LOCALAPPDATA%/VoxStudio/engines/qwen3-tts-audition`, full audio/transcript conditioning, BF16 and SDPA. Chatterbox uses its existing separate environment and the English TTS checkpoint, retaining native Perth watermarking. It does not use the unrelated voice-conversion benchmark recording.

Chatterbox jobs may set `exaggeration` (0-2), `cfgWeight` (0-1) and `temperature`
(0.1-1.5). Omission preserves the existing settings: 0.35, 0.3 and 0.8.
For example, a short emergency command can audition `"exaggeration": 1.0` while
an adjacent clinical phrase keeps a lower value. These are native synthesis
controls, not post-production pitch or volume effects. They are rejected on Qwen
jobs, checked for finite numeric values, bound into the preview revision, and
recorded per take. Changing one requires a fresh preview. The Voice Lab's
existing recipe import, GPU admission, exclusive claims and native watermark
remain in effect. Increased expression does not establish convincing fear or
voice consistency; listen before selecting any performance.

Qwen recipes may set `"nonStreamingMode": true` to use the installed Qwen Base API's `non_streaming_mode` option. Omission or `false` retains the existing simulated streaming-text conditioning. The value must be a JSON boolean; other types and use with Chatterbox are rejected. The exact recipe remains part of the preview revision, and each Qwen receipt records the effective value in `settings.non_streaming_mode`. Changing this option requires a fresh preview.

This option is supported by the inspected `qwen-tts` 0.1.1 runtime. It changes how the complete text and reference speech codes are arranged in the model context while retaining the same reference audio, transcript and speaker prompt. It is a phrase-continuity experiment: its audible effect on pauses and sentence endings is unknown until compared by listening. The default can already include all text in its initial context, depending on reference and text lengths. This is neither an intonation instruction nor a noise-removal control. Keep the same text, reference, seed and other settings for a controlled comparison, preserve raw takes, and retain all existing memory admission checks; the longer prompt can increase memory use. No new model, profile or production MCP option is introduced.

Current task recipes and qualification evidence are under `Documents/Youtube/CavernEnvironment/VoiceAuditionV1`. The research and orchestration packet is in `ZbrushMCP/knowledge/cavern-voice-audition-20260923`. Successful native generation qualifies only these snapshots and this audition route, not other engine versions or a trained speaker model.

Tests: `python -m unittest discover -s tests/tools -p test_narration_audition.py`.

Upstream sources: https://github.com/QwenLM/Qwen3-TTS (Apache-2.0); https://github.com/resemble-ai/chatterbox (MIT). Preserve upstream package notices and model-card terms when distributing components.
