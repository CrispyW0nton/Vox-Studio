# Local narration timing review

Run `tools/review_narration_timing.py TAKE.wav --local-model LOCAL_WHISPER_DIRECTORY --output NEW_REPORT.json` with the Vox Python runtime. The model directory must already contain `model.bin`; the tool stays offline and does not download models. Source audio and profiles are read-only, and an existing report is never overwritten.

The report contains approximate word and sentence timing, local eight-word pacing windows, long-gap review cues, clipping counts and confidence-screened terminal-pitch observations. The local windows help locate a slowdown that an average words-per-minute value can conceal. They do not set an ideal speaking speed. Compare a passage with its intended meaning and the host's accepted delivery.

ASR punctuation and boundaries can be wrong. Never cut audio at those timestamps without checking the waveform and listening. Pitch analysis abstains when it lacks voiced evidence; abstention does not mean statement endings passed. A rising contour may also be natural. Every candidate still needs listening review for likeness, clarity, emotion and rhythm.

For Nullstar's current feedback, retain the accepted voice profile and opening. Test a short, direct paragraph before replacing long chapters. Record text changes separately from delivery or engine changes. Do not hide stretched syllables with a faster playback rate, insert a fixed pause after every sentence, or flatten the entire pitch contour. Preserve clean masters and rejected auditions.

This is a standalone review utility, not a live synthesis-engine modification or proof that Vox's prosody is fixed.

## Thought-sized takes and coherent joins

Nullstar's latest requested test is roughly two connected sentences per generation, each ending on a completed thought. About 25–40 words is a starting experiment, not a universal rule or a measured optimum. Keep a joined passage script alongside the individual takes so the next opening follows the previous idea; avoid a succession of unrelated announcements or repeated introductions.

Generate each take through the existing qualified Vox route with the same identity and settings. Review its ending and the next opening together. Use `tools/assemble_narration_chunks.py PLAN.json --output NEW_DIRECTORY` to join native masters using explicit per-boundary pauses. The plan contains `profileId` and `chunks`, each with `path`, `sha256`, `text`, and `pauseAfterSeconds`. Set the last pause to zero. The current short-paragraph character and word limits still apply.

The assembler verifies hashes and PCM format, retains every source speech sample and tail, and inserts only the specified silence. It does not shorten words, pitch-shift, crossfade over speech, or invent breaths. The output timeline records each source offset so picture edits and captions can follow it. A 0.35-second join is an audition choice, not a mandatory pause for every boundary. Listen for artificial gaps or an unfinished intonation; regenerate the affected thought when spacing cannot fix its delivery.
