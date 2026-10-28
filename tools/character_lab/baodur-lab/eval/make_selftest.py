"""CPU self-test for the eval harness, built from real TRAIN clips only (never test clips), plus its checker.

The character is VOICE_LAB_CHARACTER (default baodur); <lab> is its lab folder and the fixture clips come from
<lab>/eval/eval_character.json "selftest" (Bao-Dur's choices are the examples below).

  seed-vc/.venv/Scripts/python.exe make_selftest.py            # writes the fixtures, prints the run command
  python run_eval.py --run selftest_cpu --manifest <realclips manifest> --manifest <altclips manifest> \
                     --script <fixture script> --device cpu [--delivery-map <map>] [--metrics ...]
  seed-vc/.venv/Scripts/python.exe make_selftest.py --check <lab>/eval/selftest_cpu [--structural]

Writes <lab>/eval/_selftest/ (not <lab>/takes, so no engine comparison ever picks it up):
takes/realclips/selftest/manifest.json and takes/altclips/selftest/manifest.json (48 kHz mono PCM16 WAVs
unless noted), delivery/takes_script.json and a fixture delivery/delivery_map.json.
Fixture spec ("selftest" in the eval character file):
  clips           train clips A, B, C (Bao-Dur: gblbaodur056, gblbaodur218, gblbaodur121)
  heldout         the test clip whose line C is filed under (Bao-Dur: gblbaodur131)
  heldout_dup     optional: a test clip whose line train clip B repeats (Bao-Dur: 218 repeats gblbaodur200); a
                  character without such a pair (Jolee) gets no H / H_dupref takes and no checks on them
  leak_reference  the test clip B_leak claims as its reference (default heldout_dup, else heldout)
  wrong_voice     takes in another character's voice, each {"take": "W_<name>", "speaker", and either "id": a train
                  clip the delivery task confirmed as another voice (Bao-Dur: G0-T0 gblbaodur110, Atton
                  gblbaodur008), or "audio": another character's clip (absolute or relative to the Voices folder)
                  with its exact "text"} (Jolee: a Carth line from RvcDatasets/CarthExact)
  references      the reference ids the fixture takes name: warm, troubled, questioning, natural (train clips)
  delivery_refs   label -> train clips of the fixture delivery map (warm, troubled, questioning)
Takes and what they must produce (Bao-Dur's clip numbers in brackets):
  A          A [056] as-is, warm line                     -> best on its line together with A_24k (same clip)
  A_broken   A +12 dB hard-clipped, +0.8 s lead, +2.5 s tail -> hard fail (clipping), silences flagged
  A_hiss     A + white noise at -30 dBFS (seeded)          -> soft hiss flags (and a DNSMOS drop), below A
  A_trimhiss A trimmed to its speech, + hiss low-passed at 9 kHz, -30 dBFS -> steady-noise-floor flag without
             the high-band flag (the VAD-free measure catches hiss on takes that have no silence; the high-band
             expectation is checked only when the character has restored clips, which raise that ceiling)
  A_click    A through GPT-SoVITS's int16 cast at 32 kHz (peak +1.0 wraps to -32768), resampled to 48 kHz and
             peak-rescaled like gsv/infer.py -> hard fail (full-scale one-sample click)
  A_24k      [altclips] A at -6 dB, band-limited via 24 kHz -> identity within the tie margin of A
  B          B [218] as-is, troubled line                  -> passes
  B_22k      [altclips] B written at 22.05 kHz             -> format note (contract is 48 kHz)
  B_leak     B as-is, reference_id = test clip leak_reference [gblbaodur200] -> leak by id, ranks last on its line
  G_v2pro    B as-is with params.model_version v2ProPlus   -> ERes2NetV2 left out of its identity (coupled)
  M_flat     B pitch-flattened (TD-PSOLA to its median F0; rate unchanged) -> monotone prosody flag, below B
  C          C [121] filed under held-out line heldout [131], whose text it does not say -> WER gate fail
  H          B filed under held-out line heldout_dup [200] (B repeats that line: a real second performance)
             -> passes; paired identity vs the real test clip                            (only with heldout_dup)
  H_dupref   B under held-out line heldout_dup, prompt_wav = byte copy of train clip B under another name
             -> leak by audio hash (a train clip repeating the held-out line), ranks below H  (only with heldout_dup)
  W_<name>   another character's line [G0-T0 110, Atton 008] filed as the character's take (its transcript)
             -> identity gate (tier "identity"), WER passes
  P_train    C on a train diagnostic line paired with C itself -> passes; memorisation table
  P_selfref  C on that line, reference_id = C              -> leak (conditioned on the line's own recording)
Test clips are only read as ground truth by the metrics; they are never copied or used as input.
--structural skips the checks that need full-corpus baselines (for runs on a reduced scratch corpus).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalcommon as ec  # noqa: E402
import evalchar  # noqa: E402

ROOT = ec.EVAL_ROOT / "_selftest"
TAKES = ROOT / "takes" / "realclips" / "selftest"
ALT = ROOT / "takes" / "altclips" / "selftest"
SPEC = ec.EVAL_CHARACTER.get("selftest") or {}
CLIPS = dict(SPEC.get("clips") or {})               # roles A, B, C -> train clip ids
HELDOUT = SPEC.get("heldout")                        # C is filed under this test line (text mismatch)
HELDOUT_DUP = SPEC.get("heldout_dup")                # train clip B repeats this test line (optional)
LEAK_REF = SPEC.get("leak_reference") or HELDOUT_DUP or HELDOUT   # B_leak claims this test clip as reference
WRONG = list(SPEC.get("wrong_voice") or [])          # [{take, speaker, id | audio + text}]
REFS = dict(SPEC.get("references") or {})            # warm / troubled / questioning / natural -> reference ids
DELIVERY_REFS = dict(SPEC.get("delivery_refs") or {})
DUP_TAG = (re.findall(r"\d+", HELDOUT_DUP) or [HELDOUT_DUP])[-1] if HELDOUT_DUP else None   # "200"
DUP_LINE = f"selftest_heldout{DUP_TAG}" if HELDOUT_DUP else None
SV_NAMES = ("sv_eres2net", "sv_resemblyzer", "sv_wavlm")


def fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def clip_tag(clip_id: str) -> str:
    """The number in a clip id ('gblbaodur056' -> '056') for check messages."""
    return (re.findall(r"\d+", clip_id) or [clip_id])[-1]


def wrong_line(w: dict) -> str:
    return w.get("line") or f"selftest_wrongvoice_{w['take'].split('_', 1)[-1].lower()}"


def wrong_audio(w: dict) -> Path:
    return evalchar.voices_path(w["audio"])


def require_spec() -> None:
    where = ec.EVAL_CHARACTER_FILE or ec.EVAL_ROOT / evalchar.FILE_NAME
    missing = [k for k in ("clips", "heldout", "references", "delivery_refs") if not SPEC.get(k)]
    if missing or sorted(CLIPS) != ["A", "B", "C"]:
        raise SystemExit(f"{ec.NAME}: no usable self-test spec in {where} (\"selftest\" needs clips A/B/C, heldout, "
                         f"references and delivery_refs; missing {missing or 'clips A/B/C'})")
    for w in WRONG:
        if not str(w.get("take", "")).startswith("W_") or not w.get("speaker") or not (w.get("id") or (w.get("audio")
                                                                                                     and w.get("text"))):
            raise SystemExit(f"{where}: selftest.wrong_voice entry {w} needs take W_<name>, speaker and id or audio+text")


def psola_flatten(x: np.ndarray, rate: int) -> np.ndarray:
    """TD-PSOLA to a constant pitch (the clip's median F0): two-period grains from pyin-spaced source marks are
    overlap-added at the target period; unvoiced stretches keep a 10 ms grid. Duration is unchanged."""
    import librosa
    prosody = ec.EVAL_CHARACTER["prosody"]
    f0, voiced, _ = librosa.pyin(ec.resample(x, rate, 16000), fmin=prosody["fmin_hz"], fmax=prosody["fmax_hz"],
                                 sr=16000, frame_length=1024, hop_length=160, resolution=0.2)
    good = voiced & np.isfinite(f0)
    target = float(np.median(f0[good]))
    hop = int(0.010 * rate)

    def period(n: int):
        i = min(max(int(round(n / hop)), 0), len(f0) - 1)
        return rate / f0[i] if good[i] else None

    marks, periods, n = [], [], 0
    while n < len(x):
        t = period(n)
        marks.append(n)
        periods.append(t)
        n += int(round(t if t else hop))
    marks = np.array(marks)
    pad = int(rate / 40)
    out = np.zeros(len(x) + 2 * pad)
    norm = np.zeros_like(out)
    o = 0
    while o < len(x):
        k = min(int(np.searchsorted(marks, o)), len(marks) - 1)
        if k > 0 and abs(marks[k - 1] - o) < abs(marks[k] - o):
            k -= 1
        t = periods[k]
        half = int(round(t if t else hop))
        a, b = marks[k] - half, marks[k] + half
        seg = np.zeros(2 * half)
        lo, hi = max(a, 0), min(b, len(x))
        seg[lo - a:hi - a] = x[lo:hi]
        window = np.hanning(2 * half)
        s = o - half + pad
        if s >= 0 and s + 2 * half <= len(out):
            out[s:s + 2 * half] += seg * window
            norm[s:s + 2 * half] += window
        o += int(round(rate / target if t else hop))
    y = out[pad:pad + len(x)] / np.maximum(norm[pad:pad + len(x)], 0.5)
    return (y * (np.abs(x).max() / (np.abs(y).max() + 1e-9))).astype(np.float32)


def gsv_int16_wrap(x48: np.ndarray) -> np.ndarray:
    """GPT-SoVITS output path: 32 kHz audio normalised to a peak of exactly 1.0, cast with (x * 32768).astype(int16)
    (TTS_infer_pack/TTS.py; +1.0 wraps to -32768), back to float, soxr VHQ to 48 kHz, peak rescaled to 0.999."""
    x = ec.resample(x48, 48000, 32000).astype(np.float64)
    k = int(np.argmax(np.abs(x)))
    if x[k] < 0:  # polarity so the full-scale sample is positive (inaudible change)
        x = -x
    x = x / x[k]
    pcm = (x * 32768).astype(np.int16)
    y = pcm.astype(np.float32) / 32768.0
    try:
        import soxr
        y = soxr.resample(y, 32000, 48000, quality="VHQ")
    except ImportError:
        y = ec.resample(y, 32000, 48000)
    peak = float(np.abs(y).max())
    return (y * (0.999 / peak) if peak > 0.999 else y).astype(np.float32)


def lowpassed_noise(n: int, rate: int, cutoff: float, db: float, seed: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    noise = np.random.default_rng(seed).standard_normal(n)
    noise = sosfilt(butter(8, cutoff, btype="low", fs=rate, output="sos"), noise)
    noise *= 10 ** (db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return noise.astype(np.float32)


def build() -> None:
    require_spec()
    corpus = ec.corpus_index()

    def row(clip: str) -> dict:
        if clip not in corpus:
            raise SystemExit(f"{clip} is not in {ec.CORPUS}")
        return corpus[clip]

    in_set_wrong = [w["id"] for w in WRONG if w.get("id")]
    for clip in (list(CLIPS.values()) + in_set_wrong + [r for refs in DELIVERY_REFS.values() for r in refs]
                 + list(REFS.values())):
        if row(clip)["split"] != "train":
            raise SystemExit(f"{clip} is not a train clip")
    for clip in [c for c in (HELDOUT, HELDOUT_DUP, LEAK_REF) if c]:
        if row(clip)["split"] != "test":
            raise SystemExit(f"{clip} is not a test clip")
    if HELDOUT_DUP and ec.norm_words(corpus[CLIPS["B"]]["text"]) != ec.norm_words(corpus[HELDOUT_DUP]["text"]):
        raise SystemExit(f"{CLIPS['B']} no longer repeats {HELDOUT_DUP}'s line")
    others, _src = ec.known_other_speakers()
    missing = [c for c in in_set_wrong if c not in others]
    if missing:
        raise SystemExit(f"{missing} are no longer confirmed other speakers; pick other wrong-voice clips")
    absent = [str(wrong_audio(w)) for w in WRONG if not w.get("id") and not wrong_audio(w).is_file()]
    if absent:
        raise SystemExit(f"wrong-voice clips not found: {absent}")
    TAKES.mkdir(parents=True, exist_ok=True)
    ALT.mkdir(parents=True, exist_ok=True)

    def write(name: str, make, rate: int, folder: Path = TAKES) -> str:
        path = folder / f"{name}.wav"
        if not path.exists():  # deterministic content; never rewrite a take another run may be reading
            sf.write(str(path), np.clip(make(), -1.0, 1.0), rate, subtype="PCM_16")
        return path.name

    def take(take_id, engine, line_id, text, delivery, wav, reference_id, note, seed=0, **params):
        return {"take_id": take_id, "engine": engine, "line_id": line_id, "text": text,
                "delivery": delivery, "wav": wav, "reference_id": reference_id, "seed": seed,
                "params": {"note": note, **params}}

    def text_of(clip_id: str) -> str:
        return corpus[clip_id]["text"]

    a, rate = ec.read_mono(corpus[CLIPS["A"]]["audio"], 48000)
    b, _ = ec.read_mono(corpus[CLIPS["B"]]["audio"], 48000)
    c, _ = ec.read_mono(corpus[CLIPS["C"]]["audio"], 48000)
    noise = np.random.default_rng(1234).standard_normal(len(a)).astype(np.float32) * 10 ** (-30 / 20)
    prompt_copy = TAKES / "prompt_copy.wav"  # byte copy of a TRAIN clip under a name without its id
    if HELDOUT_DUP and not prompt_copy.exists():
        shutil.copyfile(corpus[CLIPS["B"]]["audio"], prompt_copy)

    def broken():
        return np.concatenate([np.zeros(int(0.8 * rate), np.float32), np.clip(a * 4.0, -1.0, 1.0),
                               np.zeros(int(2.5 * rate), np.float32)])

    def trimhiss():
        speech = ec.speech_segment(a, rate, pad_s=0.0)
        return speech + lowpassed_noise(len(speech), rate, 9000.0, -30.0, 4321)

    def other(w: dict):
        source = corpus[w["id"]]["audio"] if w.get("id") else str(wrong_audio(w))
        return lambda: ec.read_mono(source, 48000)[0]

    def wrong_take(w: dict) -> dict:
        ident = w.get("id") or wrong_audio(w).stem
        text = text_of(w["id"]) if w.get("id") else w["text"]
        return take(w["take"], "realclips", wrong_line(w), text, "natural", write(w["take"], other(w), rate),
                    REFS["natural"], f"{w['speaker']} line {ident} (another character) filed as a {ec.NAME} take: "
                                     "identity gate")

    warm, troubled, questioning = REFS["warm"], REFS["troubled"], REFS["questioning"]
    takes = [
        take("A", "realclips", "selftest_warm", text_of(CLIPS["A"]), "warm", write("A", lambda: a, rate), warm,
             f"real train clip {CLIPS['A']} copied unchanged"),
        take("A_broken", "realclips", "selftest_warm", text_of(CLIPS["A"]), "warm", write("A_broken", broken, rate),
             warm, f"{CLIPS['A']} +12 dB hard-clipped, 0.8 s lead + 2.5 s tail silence added", seed=1),
        take("A_hiss", "realclips", "selftest_warm", text_of(CLIPS["A"]), "warm", write("A_hiss", lambda: a + noise, rate),
             warm, f"{CLIPS['A']} plus white noise at -30 dBFS (seed 1234)", seed=2),
        take("A_trimhiss", "realclips", "selftest_warm", text_of(CLIPS["A"]), "warm", write("A_trimhiss", trimhiss, rate),
             warm, f"{CLIPS['A']} trimmed to its speech span plus noise low-passed at 9 kHz, -30 dBFS "
                   "(seed 4321)", seed=3),
        take("A_click", "realclips", "selftest_warm", text_of(CLIPS["A"]), "warm",
             write("A_click", lambda: gsv_int16_wrap(a), rate),
             warm, f"{CLIPS['A']} through GPT-SoVITS's int16 cast at 32 kHz (+1.0 wraps to -32768), "
                   "resampled to 48 kHz", seed=4),
        take("B", "realclips", "selftest_troubled", text_of(CLIPS["B"]), "troubled", write("B", lambda: b, rate), troubled,
             f"real train clip {CLIPS['B']} copied unchanged"),
        take("B_leak", "realclips", "selftest_troubled", text_of(CLIPS["B"]), "troubled", write("B_leak", lambda: b, rate),
             LEAK_REF, f"{CLIPS['B']} as-is but claiming test clip {LEAK_REF} as its reference (must be caught)",
             seed=1),
        take("G_v2pro", "realclips", "selftest_troubled", text_of(CLIPS["B"]), "troubled", write("G_v2pro", lambda: b, rate),
             troubled, f"{CLIPS['B']} as-is, labelled as a GPT-SoVITS v2ProPlus take (speaker-model coupling)",
             seed=2, model_version="v2ProPlus"),
        take("M_flat", "realclips", "selftest_troubled", text_of(CLIPS["B"]), "troubled",
             write("M_flat", lambda: psola_flatten(b, rate), rate), troubled,
             f"{CLIPS['B']} pitch-flattened to its median F0 with TD-PSOLA (monotone; duration unchanged)", seed=3),
        take("C", "realclips", "selftest_heldout", text_of(HELDOUT), "questioning", write("C", lambda: c, rate),
             questioning, f"train clip {CLIPS['C']} filed under the held-out line on purpose (text mismatch)"),
    ]
    if HELDOUT_DUP:
        takes += [
            take("H", "realclips", DUP_LINE, text_of(HELDOUT_DUP), "troubled", write("H", lambda: b, rate), troubled,
                 f"train clip {CLIPS['B']}, a second real performance of held-out line {HELDOUT_DUP}"),
            take("H_dupref", "realclips", DUP_LINE, text_of(HELDOUT_DUP), "troubled", write("H_dupref", lambda: b, rate),
                 None, f"{CLIPS['B']} prompted with a renamed byte copy of itself, a train clip repeating the held-out "
                       "line (must be caught by hash)", seed=1, prompt_wav=prompt_copy.name),
        ]
    takes += [wrong_take(w) for w in WRONG]
    takes += [
        take("P_train", "realclips", "selftest_trainpair", text_of(CLIPS["C"]), "questioning",
             write("P_train", lambda: c, rate), questioning,
             f"train clip {CLIPS['C']} on a train diagnostic line paired with itself"),
        take("P_selfref", "realclips", "selftest_trainpair", text_of(CLIPS["C"]), "questioning",
             write("P_selfref", lambda: c, rate), CLIPS["C"],
             f"{CLIPS['C']} conditioned on the diagnostic line's own recording (must be caught)", seed=1),
    ]
    manifest = TAKES / "manifest.json"
    ec.write_json(manifest, {"takes": takes})

    def band_limited():
        return ec.resample(ec.resample(a * 0.5, 48000, 24000), 24000, 48000)

    b22, rate_b = ec.read_mono(corpus[CLIPS["B"]]["audio"], 22050)
    alt = [
        take("A_24k", "altclips", "selftest_warm", text_of(CLIPS["A"]), "warm", write("A_24k", band_limited, 48000, ALT),
             warm, f"{CLIPS['A']} at -6 dB, band-limited through 24 kHz"),
        take("B_22k", "altclips", "selftest_troubled", text_of(CLIPS["B"]), "troubled",
             write("B_22k", lambda: b22, rate_b, ALT),
             troubled, f"{CLIPS['B']} written at 22.05 kHz (breaks the 48 kHz contract on purpose)"),
    ]
    alt_manifest = ALT / "manifest.json"
    ec.write_json(alt_manifest, {"takes": alt})

    script = ROOT / "delivery" / "takes_script.json"
    lines = [
        {"id": "selftest_warm", "text": text_of(CLIPS["A"]), "delivery": "warm", "source": "new"},
        {"id": "selftest_troubled", "text": text_of(CLIPS["B"]), "delivery": "troubled", "source": "new"},
        {"id": "selftest_heldout", "text": text_of(HELDOUT), "delivery": "questioning", "source": "heldout",
         "heldout_id": HELDOUT},
    ]
    if HELDOUT_DUP:
        lines.append({"id": DUP_LINE, "text": text_of(HELDOUT_DUP), "delivery": "troubled", "source": "heldout",
                      "heldout_id": HELDOUT_DUP})
    for w in WRONG:
        lines.append({"id": wrong_line(w), "text": text_of(w["id"]) if w.get("id") else w["text"],
                      "delivery": "natural", "source": "new"})
    lines.append({"id": "selftest_trainpair", "text": text_of(CLIPS["C"]), "delivery": "questioning", "source": "train",
                  "real_id": CLIPS["C"]})
    ec.write_json(script, {"lines": lines})
    delivery = ROOT / "delivery" / "delivery_map.json"
    ec.write_json(delivery, {"deliveries": {label: {
        "references": [{"id": i, "audio": corpus[i]["audio"], "restored_audio": str(ec.RESTORED_DIR / f"{i}.wav"),
                        "text": corpus[i]["text"], "duration": corpus[i]["duration"], "why": "selftest fixture"}
                       for i in ids],
        "voxcpm_instruction": "", "notes": "selftest fixture, not a real delivery design"} for label, ids in DELIVERY_REFS.items()}})
    print(f"manifest {manifest}\nalt_manifest {alt_manifest}\nscript {script}\ndelivery_map {delivery}")


def check(run_dir: Path, structural: bool = False) -> int:
    """Assert the outcomes the fixtures are built to produce. Returns the number of failed checks."""
    require_spec()
    card = ec.read_json(run_dir / "scorecard.json")
    rows = {r["take_id"]: r for r in card["takes"]}
    lines = {l["line_id"]: l for l in card["lines"]}
    tests = ec.test_ids()
    present = set(card.get("metrics_present") or [])
    sv = [m for m in SV_NAMES if m in present]
    variant = card["reference_audio"]
    results: list[tuple[bool, str]] = []
    wrong = [w["take"] for w in WRONG]
    paired_take = "H" if HELDOUT_DUP else "P_train"   # a take with a paired real recording

    def expect(ok, what: str, stat: bool = False) -> None:
        if stat and structural:
            return
        results.append((bool(ok), what))

    def rank(take_id: str) -> int:
        return rows[take_id]["rank_in_line"] or 10 ** 6

    def flags(take_id: str, part: str) -> str:
        return " ".join(rows[take_id]["flags"][part]).lower()

    steps = {s["metric"]: s for s in card.get("steps", [])}
    ran = sorted(m for m in steps if m in present)
    expect(ran and all(steps[m]["exit_code"] == 0 for m in ran), f"metric steps {ran} exited 0")
    expect(not [r["take_id"] for r in card["takes"] if r["unscored"]], "no take unscored")
    # leaks
    expect(rows["B_leak"]["tier"] == 3 and LEAK_REF in " ".join(rows["B_leak"]["violations"]),
           "B_leak: test-clip reference caught (tier leak)")
    expect(rank("B_leak") == lines["selftest_troubled"]["n_scored"], "B_leak ranks last on its line despite identical audio")
    if HELDOUT_DUP:
        expect(rows["H_dupref"]["tier"] == 3 and CLIPS["B"] in " ".join(rows["H_dupref"]["violations"]),
               "H_dupref: renamed near-duplicate prompt caught by hash")
    expect(rows["P_selfref"]["tier"] == 3 and "own real recording" in " ".join(rows["P_selfref"]["violations"]),
           "P_selfref: conditioning on the diagnostic line's own recording caught")
    clean = ["A", "B"] + (["H"] if HELDOUT_DUP else []) + ["G_v2pro", "M_flat", "P_train"] + wrong[:1]
    expect(not [t for t in clean if rows[t]["violations"]], f"no false leak flags on {', '.join(clean)}")
    if HELDOUT_DUP:
        expect(lines[DUP_LINE].get("heldout_near_duplicates"), f"held-out-{DUP_TAG} line lists its near-duplicates")
    if sv:
        if HELDOUT_DUP:
            expect(lines[DUP_LINE]["best"].endswith("/H"), f"H is best on the held-out-{DUP_TAG} line")
        expect(rank("A_hiss") > rank("A") and rank("A_trimhiss") > rank("A"), "hissy copies rank below A")
        a24 = rows["A_24k"]
        expect(a24["similarity"] is not None and rows["A"]["similarity"] is not None
               and abs(min(a24["similarity"], 1) - min(rows["A"]["similarity"], 1)) <= 0.1,
               "A_24k (-6 dB, band-limited) keeps A's identity within the tie margin", stat=True)
    if "artifacts" in present:
        expect(rows["A_broken"]["tier"] == 2 and "clipped" in flags("A_broken", "hard"), "A_broken fails on clipping")
        expect("lead silence" in flags("A_broken", "soft") and "tail silence" in flags("A_broken", "soft"),
               "A_broken's added lead/tail silence is flagged")
        expect(rows["A_click"]["tier"] == 2 and "one-sample click" in flags("A_click", "hard"),
               "A_click: GPT-SoVITS int16 wrap fails as a full-scale click")
        expect(any(k in flags("A_hiss", "soft") for k in ("high-band", "noise floor")), "A_hiss gets a hiss soft flag")
        expect("steady noise floor" in flags("A_trimhiss", "soft"),
               "A_trimhiss: hiss on a trimmed take (no silence) caught by the VAD-free floor")
        if (card["baselines"].get("artifacts") or {}).get("restore_report"):  # the ceiling includes restored clips
            expect("high-band" not in flags("A_trimhiss", "soft"),
                   "A_trimhiss: its 9 kHz low-passed hiss stays under the high-band ceiling (restored clips' p95)",
                   stat=True)
        expect(any("format 22050" in n for n in rows["B_22k"]["flags"]["notes"]), "B_22k gets the format note")
        real_fired = card["baselines"].get("artifacts_flags_on_real_train") or {}
        expect(real_fired.get("clips") and real_fired.get("flagged", 99) <= max(1, real_fired["clips"] // 100),
               f"artifact flags fire on <= 1% of real train clips ({real_fired.get('flagged')} of "
               f"{real_fired.get('clips')})", stat=True)
    if "prosody" in present:
        expect("monotone" in flags("M_flat", "prosody"), "M_flat: pitch-flattened take flagged monotone")
        expect(rank("M_flat") > rank("B"), "M_flat ranks below B")
        expect(not rows["M_flat"].get("tie"), "M_flat's monotone penalty puts it outside the tie margin", stat=True)
        p = card["baselines"].get("prosody") or {}
        expect(p and p.get("heldout_real_flagged", 99) <= 3,
               f"prosody flags on <= 3 of {ec.HIS} {p.get('n_heldout')} held-out clips ({p.get('heldout_real_flagged')})",
               stat=True)
    # identity
    if sv:
        expect(lines["selftest_warm"]["best"].rsplit("/", 1)[-1] in ("A", "A_24k"),
               f"a clean copy of {clip_tag(CLIPS['A'])} (A or A_24k) is best on the warm line")
        for take_id in wrong:
            r = rows[take_id]
            expect(r["identity_gate_value"] is not None and r["identity_gate"] is not None
                   and r["identity_gate_value"] < r["identity_gate"],
                   f"{take_id}: another character's voice is below the identity gate ({fmt(r['identity_gate_value'])} "
                   f"< {fmt(r['identity_gate'])})")
            if "asr" in present:
                expect(r["tier"] == 1, f"{take_id}: demoted by the identity gate alone (WER passes)")
        expect(rows[paired_take]["identity_models"]
               and all(rows[paired_take]["similarity_detail"][m].get("identity_paired") is not None
                       for m in rows[paired_take]["identity_models"]),
               f"{paired_take} has a paired identity for every model")
    if "sv_eres2net" in sv and len(sv) > 1:
        g = rows["G_v2pro"]
        expect("sv_eres2net" not in g["identity_models"] and g["similarity_detail"]["sv_eres2net"].get("coupled"),
               "G_v2pro: ERes2NetV2 left out of a v2ProPlus take's identity (coupled), still reported")
    for m in sv:
        metric = ec.read_json(ec.metric_out_path(run_dir, m))
        b = (metric.get("baselines") or {}).get(variant)
        if b is None:
            expect(False, f"{m}: per-variant baselines present for {variant}")
            continue
        expect(b["test_vs_centroid"].get("n") == len(tests), f"{m}: all {len(tests)} test clips in the real baseline",
               stat=True)
        expect(not (set(b.get("centroid_train_ids") or []) & tests), f"{m}: no test clip in the centroid")
        expect(set(b["known_other_speakers"]["ids_in_train"]) <= {s["id"] for s in b["train_suspects"]},
               f"{m}: confirmed other-speaker clips are out of the centroid")
        cb = card["baselines"].get(m) or {}
        bias = cb.get("bias_check") or {}
        expect((cb.get("identity_model") or {}).get("kind") == "span+expressiveness",
               f"{m}: identity anchor fitted on span and expressiveness", stat=True)
        rho_span, rho_aty = bias.get("spearman_adjusted_vs_span"), bias.get("spearman_adjusted_vs_atypicality")
        spread = bias.get("adjusted_span_spread")
        expect(rho_span is not None and rho_aty is not None and abs(rho_span) < 0.2 and abs(rho_aty) < 0.2
               and spread is not None and spread <= 0.10,
               f"{m}: fitted identity no longer tracks clip length or expressiveness (Spearman {fmt(rho_span)} / "
               f"{fmt(rho_aty)}, span-bin medians within {fmt(spread)}; constant anchor "
               f"{fmt(bias.get('spearman_constant_vs_span'))} / {fmt(bias.get('spearman_constant_vs_atypicality'))}, "
               f"{fmt(bias.get('constant_span_spread'))})", stat=True)
        held = (cb.get("real_identity_heldout") or {}).get("median")
        expect(held is not None and abs(held - 1.0) < 0.15, f"{m}: {ec.HIS} held-out clips sit near identity 1 ({held})",
               stat=True)
    # memorisation / paired lines
    mem = card.get("memorisation") or {}
    expect(any("train" in e["by_split"] and "test" in e["by_split"] for e in mem.values()),
           "memorisation table pairs train and held-out lines")
    expect(rows["P_train"]["tier"] in (0, 1) and rows["P_train"]["paired_split"] == "train",
           "P_train is scored as a train diagnostic take")
    if "asr" in present:
        real = "H" if HELDOUT_DUP else "B"
        expect(rows["C"]["tier"] == 2 and "wer" in flags("C", "hard"), "C fails the WER gate")
        expect(all(r["wer"] is not None for r in card["takes"]), "every take has a WER")
        expect(rows["A"]["wer"] <= 0.1 and rows[real]["wer"] <= 0.1, "real clips transcribe with WER <= 0.1")
    if "prosody" in present:
        expect(all(r["plausibility_penalty"] is not None for r in card["takes"]), "every take has prosody features")
    if "dnsmos" in present:
        da, dh = rows["A"].get("dnsmos") or {}, rows["A_hiss"].get("dnsmos") or {}
        expect(da.get("OVRL") is not None and dh.get("OVRL") is not None and dh["OVRL"] < da["OVRL"] - 0.3,
               f"DNSMOS: hiss lowers OVRL ({dh.get('OVRL')} vs {da.get('OVRL')})")
        expect("dnsmos" in flags("A_hiss", "quality"), "A_hiss gets a DNSMOS quality flag", stat=True)
    if "sv_wavlm" in present:
        expect(all(r["similarity_detail"].get("sv_wavlm", {}).get("identity") is not None
                   for r in card["takes"] if r["tier"] is not None), "every scored take has a WavLM-SV identity")
        expect("wavlm" in (card.get("delivery_signals") or {}), "WavLM layer features tested as a delivery signal")
    expect("prosody" in (card.get("delivery_signals") or {}) or "prosody" not in present,
           "prosody tested as a delivery signal")
    for ok, what in results:
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"{len(results) - failed}/{len(results)} checks passed (metrics: {', '.join(sorted(present))}"
          + ("; structural only" if structural else "") + ")")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="RUN_DIR", help="assert the expected outcomes of a finished selftest run")
    parser.add_argument("--structural", action="store_true",
                        help="skip checks that need full-corpus baselines (reduced scratch corpus)")
    args = parser.parse_args()
    if args.check:
        sys.exit(1 if check(Path(args.check), args.structural) else 0)
    build()


if __name__ == "__main__":
    main()
