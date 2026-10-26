# Character-laboratory source recovery

This is a source-only copy of the shared laboratory tools, captured October 26,
2026. Preserve `baodur-lab` and `characters` as siblings when restoring. Despite
the historical folder name, these tools select characters through
`VOICE_LAB_CHARACTER` and private character configuration.

`SOURCE_MANIFEST.json` binds the 63 original Python files by SHA256. Run
`python tests/tools/test_character_lab_snapshot.py` from the repository root to
verify them without importing the tools, loading models or rendering audio.
The training example is a portable configuration example, not an original
character recipe, and is outside that original-source manifest.

No character recordings, transcripts, model weights, private recipes or
historical installation receipts are included. Public upstream engine identities
are in `UPSTREAM_REVISIONS.json`. See [recovery instructions](../../docs/RECOVERY.md).
