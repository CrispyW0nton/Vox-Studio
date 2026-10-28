# Performance Mirror Sidecar

This small local HTTP adapter lets Vox Studio use the official
[Seed-VC](https://github.com/Plachtaa/seed-vc) real-time tiny model without
copying its source or model weights into Vox Studio.

Run `setup_performance_mirror.ps1` once to install the pinned upstream revision
and its Python environment under `%LOCALAPPDATA%\VoxStudio\engines\seed-vc`.
Seed-VC is GPL-3.0 software and its model files retain their upstream terms.
Character reference recordings stay under the user's local Vox Studio profile
directory and are never included in the application or repository.
