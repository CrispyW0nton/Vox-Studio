# Vox Studio RVC Sidecar Bundle

This directory carries the pinned local RVC sidecar metadata and launcher wrapper for
Pass 9a. The runtime payload is w-okada/voice-changer, pinned by `sidecar_manifest.json`.

Do not commit voice model files here. User-imported `.pth` and `.index` files belong under
`%LOCALAPPDATA%/VoxStudio/rvc_models/`, managed by `RvcModelRegistry`.

The installer copies this folder to `%LOCALAPPDATA%/VoxStudio/rvc_sidecar/`. The launcher
expects the offline-bundled runtime payload to be present beside it.
