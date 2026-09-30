# Local narration auditions

`tools/narration_audition.py` adds experimental Qwen3-TTS and Chatterbox audition workers inside the Vox Studio repository. It does not expose a new GUI option or extend the production MCP. Existing VoxCPM2 profiles and runtime remain unchanged.

The worker supports `preview recipe.json` and `render recipe.json --revision <preview revision>`. Recipes bind the authorized source WAV hash and exact transcript, model snapshot, seed, bounded text and a dedicated `VoiceAuditionV1` output folder. Preview records dependency versions and hashes all model files. Render verifies the revision, claims the batch exclusively and uses local-only model loading. A failed batch remains for inspection. Render writes untouched float WAVs and provenance receipts; it cannot approve speaker identity or narration quality.

Qwen uses an isolated environment under `%LOCALAPPDATA%/VoxStudio/engines/qwen3-tts-audition`, full audio/transcript conditioning, BF16 and SDPA. Chatterbox uses its existing separate environment and the English TTS checkpoint, retaining native Perth watermarking. It does not use the unrelated voice-conversion benchmark recording.

Current task recipes and qualification evidence are under `Documents/Youtube/CavernEnvironment/VoiceAuditionV1`. The research and orchestration packet is in `ZbrushMCP/knowledge/cavern-voice-audition-20260923`. Successful native generation qualifies only these snapshots and this audition route, not other engine versions or a trained speaker model.

Tests: `python -m unittest discover -s tests/tools -p test_narration_audition.py`.

Upstream sources: https://github.com/QwenLM/Qwen3-TTS (Apache-2.0); https://github.com/resemble-ai/chatterbox (MIT). Preserve upstream package notices and model-card terms when distributing components.
