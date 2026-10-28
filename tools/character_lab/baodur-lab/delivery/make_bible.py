"""Render <lab>/delivery/performance_bible.md from the computed outputs (CPU, instant; VOICE_LAB_CHARACTER, default
baodur).

Every number and clip id in the bible comes from bible_facts.json (bible_facts.py), delivery_map.json,
label_stats.json, takes_script.json and voxcpm_instruction_variants.json, so the bible cannot drift from the map or
the code. Prose that states engine behaviour was checked against each engine's code (see section 4 for the
source of each claim). What the bible says about the character himself (how his numbers read, his habits, his
humour, what could not be verified) is his: <lab>/delivery/content/bible.json holds those sentences as templates
(str.format fields filled from the facts, listed in render()). validate.py re-renders it and requires identical
text.

Run (any lab python):  python make_bible.py [--map-dir DIR] [--out PATH] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_takes_script import PAIR_SUFFIX, heldout_delivery  # noqa: E402
import labcfg  # noqa: E402

HERE = Path(__file__).resolve().parent
LAB = labcfg.LAB
OUT = labcfg.DELIVERY
VALIDATED_KINDS = ("direction", "question_ending")


def short(clip_id: str | None) -> str:
    return labcfg.short_id(clip_id)


def ids(items) -> str:
    return ", ".join(short(i) for i in items)


def snippet(text: str, limit: int = 56) -> str:
    return text if len(text) <= limit else text[:limit].rstrip(" ,.;-") + "..."


def probe_takes(deliveries: dict, variants: dict, script: dict) -> int:
    """Takes of a default probe_voxcpm.py run over this map and script (its own planner, so the count cannot drift)."""
    from probe_voxcpm import plan

    defaults = SimpleNamespace(labels="", audio="original", seeds="11,23,37,53", cfg="1.5", variants="A,B,P,P2",
                               lora=None)
    return len(plan(defaults, {"deliveries": deliveries}, variants, script)[0])


def render(map_dir: Path) -> str:
    facts = json.loads((map_dir / "bible_facts.json").read_text(encoding="utf-8"))
    deliveries = json.loads((map_dir / "delivery_map.json").read_text(encoding="utf-8"))["deliveries"]
    script = json.loads((map_dir / "takes_script.json").read_text(encoding="utf-8"))
    variants = json.loads((map_dir / "voxcpm_instruction_variants.json").read_text(encoding="utf-8"))["variants"]
    picks_path = map_dir / "instruction_picks.json"
    picks = json.loads(picks_path.read_text(encoding="utf-8")).get("picks", {}) if picks_path.is_file() else {}
    T = {k: v for k, v in labcfg.content("bible").items() if not k.startswith("_")}  # his prose (templates)
    corpus_lines = sum(1 for line in (LAB / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    v, t, o, e = facts["voice"], facts["text"], facts["other_speakers"], facts["endings"]
    labels = facts["labels"]
    validated = sorted(l for l, x in labels.items() if x["kind"] in VALIDATED_KINDS)
    descriptive = sorted(l for l, x in labels.items() if x["kind"] == "descriptive")
    unmeasured = sorted(l for l, x in labels.items() if x["kind"] == "none")
    identity = facts.get("identity", {})
    imp = identity.get("impostors", {})
    july = facts.get("july_adapter", {})
    natural1 = labels["natural"]["rank1"]
    restored = facts["restored_vs_original"]
    rv = restored["abs_diff"]
    near = o["near_duplicate_of_test"]
    n_test = len(facts["script"]["heldout"])
    address = labcfg.address_key()
    q = labels.get("questioning", {}).get("question") or {}
    pz = facts["pauses"]
    owner = {clip: name for name, clips in o.get("confirmed_ids", {}).items() for clip in clips}
    # Fields the bible.json templates may use (str.format: {v[f0_range_st][median]:.1f}, {share_pct}, ...).
    values = {
        "name": labcfg.NAME, "address": labcfg.ADDRESS, "facts": facts, "v": v, "t": t, "o": o, "e": e, "pz": pz,
        "wl": t["words_per_line"], "peak": v["peak_dbfs"], "level": v["level_dbfs"],
        "level_spread": v["level_dbfs"]["p95"] - v["level_dbfs"]["p05"],
        "center_hz": round(facts["register_center_hz"]), "f0_p10_hz": round(v["f0_median_hz"]["p10"]),
        "f0_p90_hz": round(v["f0_median_hz"]["p90"]), "words_per_min": round(v["words_per_min"]["median"]),
        "pause_pct": 100 * v["pause_ratio"]["median"], "no_pause_pct": round(100 * pz["lines_without_pause_share"]),
        "q_yes_no": q.get("questions_median_st", 0), "q_statements": q.get("statements_median_st", 0),
        "energy_abs": abs(v["energy_slope_db"]["median"]),
        "energy_dir": "falls" if v["energy_slope_db"]["median"] < 0 else "rises",
        "statements_dir": "fall" if (e["statement_final_rise_median_st"] or 0) < 0 else "rise",
        "centroid_hz": round(v["centroid_hz"]["median"]), "centroid_p10": round(v["centroid_hz"]["p10"]),
        "centroid_p90": round(v["centroid_hz"]["p90"]),
        "creak": ", ".join(f"{short(c['id'])} ({round(100 * c['low_f0_ratio'])}%)" for c in facts["creak_top"][:6]),
        "address_lines": t.get(f"{address}_lines"), "address_closing": t.get(f"{address}_closing_tag_lines"),
        "address_opening": t.get(f"{address}_opening_lines"),
        "id_glob": labcfg.id_glob(), "confirmed_clips": o["confirmed_clips"], "confirmed_seconds": o["confirmed_seconds"],
        "chars": "; ".join(f"{name} ({o['confirmed'][name]}): {ids(o['confirmed_ids'][name])}" for name in o["confirmed"]),
        "confirmed_seconds_in_tts_pool": o["confirmed_seconds_in_tts_pool"],
        "share_pct": round(100 * (o["share_of_tts_pool"] or 0)), "tts_pool_s": round(o["tts_pool_seconds"]),
        "n_uncertain": len(o["uncertain"]), "uncertain": ids(o["uncertain"]),
        "train_held": ids(july.get("train.jsonl", {}).get("heldout", [])),
        "val_held": ids(july.get("validation.jsonl", {}).get("heldout", [])),
        "train_near": ids(july.get("train.jsonl", {}).get("near_duplicates", [])),
        "vox_dir": f"%LOCALAPPDATA%\\VoxStudio\\voxcpm_training\\{labcfg.VOX_TRAINING.name}",
        "validated": ", ".join(validated) or "none",
        "loo_p05": identity.get("loo_p05"), "loo_median": identity.get("loo_median"), "gate": identity.get("gate"),
        "imp_n": imp.get("n", 0), "imp_median": imp.get("median"), "imp_max": imp.get("max"),
        "imp_max_clip": short(imp.get("max_clip")), "imp_max_clip_duration": imp.get("max_clip_duration") or 0,
        "imp_max_character": owner.get(imp.get("max_clip"), "other-character"), "imp_max_3s": imp.get("max_3s"),
    }

    def fill(key: str, default: str = "") -> str:
        return str(T.get(key, default)).format_map(values)

    out: list[str] = []
    w = out.append

    w(f"# {labcfg.NAME} performance bible")
    w("")
    w(f"How {labcfg.NAME} actually sounds in the game, measured, and how to direct each engine so a clone keeps his "
      "voice and his feelings. This file is generated: `make_bible.py` renders it from `bible_facts.json` (every "
      "figure recomputed from the analysis outputs by `bible_facts.py`), `delivery_map.json` and `label_stats.json`, "
      "and `validate.py` fails if it is out of date. To change it, change the code or the data and rerun "
      "`build_delivery_map.py`, `make_takes_script.py`, `bible_facts.py`, `make_bible.py`, then `validate.py`.")
    w("")
    w(f"- **Voice profile:** {facts['profile_members']} clips ({facts['profile_seconds']} s): speech with exact "
      "TLK/DLG text, ASR WER of 0.15 or less, at least 1.5 s long, not flagged as another speaker.")
    w(f"- **Text habits:** all {facts['habit_lines']} of his exact-text lines ({facts['habit_seconds']} s).")
    w(f"- **Held out:** the {n_test} test clips were never read for any of it. The user's ears are the final judge; "
      "these numbers show where to look.")
    if T.get("audio_decision"):
        w(fill("audio_decision"))
    w("")
    w("| File | Contents |")
    w("|---|---|")
    for name, what in (
            ("`delivery_map.json`", "References (caution-free, ranked) and the VoxCPM2 instruction per delivery"),
            ("`takes_script.json`", f"{facts['script']['lines']} lines to render: one new line per delivery + the "
                                    f"{n_test} held-out lines"),
            ("`takes_script_pairs.json`", "Optional: natural reads of the emotional new lines (paired baselines)"),
            ("`label_stats.json`", "What each delivery measurably does, its kind (validated or not) and flag bounds"),
            ("`delivery_candidates.json`", "Every scored candidate, and the caution clips kept out of the map"),
            ("`voxcpm_instruction_variants.json`", "VoxCPM2 instruction variants A and B"),
            ("`instruction_picks.json`", "Probe picks applied after listening (absent until then)"),
            ("`exclude_other_speakers.json`", "Other characters, uncertain clips, near-duplicates of held-out lines"),
            ("`features*.jsonl/.csv`, `voice_profile*.json`", "Per-clip measurements and his baseline"),
            ("`bible_facts.json`, `validation_report.json`", "Every figure here; the output of `validate.py`")):
        w(f"| {name} | {what} |")
    w("")

    # ---------------------------------------------------------------- 1
    w("## 1. Read first: corpus facts that change the plan")
    w("")
    items = [fill("other_characters") if o["confirmed"] else fill("no_other_characters")]
    if near:
        items.append(f"**Train lines that repeat held-out text** (`near_duplicate_of_test`, {len(near)} clips, "
                     f"{o['near_duplicate_seconds_in_tts_pool']} s of the TTS pool): "
                     + "; ".join(f"{short(k)} ({val})" for k, val in sorted(near.items()))
                     + ". Training on them inflates those held-out scores, so every training manifest must drop them "
                       f"too. About {o['tts_pool_seconds_after']} s of the TTS pool remain after both exclusions."
                     + fill("near_duplicate_note"))
    else:
        items.append("**No train line repeats held-out text** (`near_duplicate_of_test` is empty under the leak rule: "
                     f"text ratio >= 0.6 or 5 words in a row). About {o['tts_pool_seconds_after']} s of the TTS pool "
                     "remain after the speaker exclusions.")
    if july and T.get("vox_adapter"):
        items.append(fill("vox_adapter"))
    if T.get("loudness"):
        items.append(fill("loudness"))
    for number, item in enumerate(items, 1):
        w(f"{number}. {item}")
    if restored.get("clips"):
        w(f"{len(items) + 1}. **No engine's reference conditioning reads the band the restorer adds** (above ~11 kHz), "
          "which is one more reason the user's original-audio decision costs nothing:")
        w("")
        w("| Engine | What it reads of a reference (from its code) |")
        w("|---|---|")
        w("| GPT-SoVITS v2Pro/ProPlus | GPT prompt: HuBERT at 16 kHz. SoVITS style vector: `ref_enc(refer[:, :704])`, "
          "the first 704 bins of a 32 kHz / 2048-point spectrogram = 0-11.0 kHz (`SynthesizerTrn.decode`), plus the "
          "v2Pro speaker vector at 16 kHz. |")
        w("| VoxCPM2 | Prompt and reference audio are encoded at 16 kHz; output is decoded at 48 kHz. |")
        w("| Seed-VC V2 | Source and prompt at 22.05 kHz, CAM++ style vector at 16 kHz; output 22.05 kHz, the same "
          "bandwidth as the game's own clips. |")
        w("")
        w(f"   Measured at 16 kHz over the profile clips, restored and original copies match in pitch and tempo (p95 "
          f"difference {rv['f0_median_hz']['p95_abs_diff']} Hz and {rv['articulation_rate_syl_s']['p95_abs_diff']} "
          f"syl/s) and differ slightly in tilt (alpha p95 {rv['alpha_ratio_db']['p95_abs_diff']:.1f} dB, Hammarberg p95 "
          f"{rv['hammarberg_db']['p95_abs_diff']:.1f} dB).")
    w("")

    # ---------------------------------------------------------------- 2
    w("## 2. His measurable voice")
    w("")
    w("| Dimension | Measurement (median, p10-p90 across his clips) | What it means for a clone |")
    w("|---|---|---|")
    for dimension, measurement, meaning in T.get("voice_table", []):
        w(f"| {dimension} | {measurement.format_map(values)} | {meaning.format_map(values)} |")
    w("")
    vf = facts["vox_features"]
    w(f"**Compared with Vox's own analysis** (its sidecar features for the same clips): pitch range "
      f"{vf['pitch_range_semitones']['median']:.2f} st, variation {vf['pitch_variation_semitones']['median']:.2f} st, "
      f"{vf['characters_per_second']['median']:.1f} characters/s, pause ratio {vf['pause_ratio']['median']:.2f}, "
      f"terminal pitch delta {vf['terminal_pitch_delta']['median']:+.1f} st. Vox's delivery classifier reads his "
      "profile clips as: " + ", ".join(f"{k or 'none'} {n}" for k, n in facts["vox_readings"].items()) + ".")
    w("")
    sof = t["softeners"]
    w(f"**Verbal habits** (from the text of his {facts['habit_lines']} lines):")
    if T.get("address_habit") and labcfg.ADDRESS:
        w(fill("address_habit"))
    w("- **Softeners:** " + ", ".join(f"\"{k}\" ({n})" for k, n in sorted(sof.items(), key=lambda kv: -kv[1]) if n) + ".")
    for key in ("contractions_habit", "hesitations_habit"):
        if T.get(key):
            w(fill(key))
    exclamations = f" ({ids(t['exclamation_lines'])})" if t["exclamation_lines"] else ""
    w(f"- **Exclamations:** {len(t['exclamation_lines'])} \"!\" in {facts['habit_lines']} lines{exclamations}. "
      "Engines read \"!\" as a cue to shout; none of the new lines has one.")
    if T.get("humour_habit"):
        w(fill("humour_habit"))
    w("")

    # ---------------------------------------------------------------- 3
    w("## 3. How each delivery shows up in his clips")
    w("")
    w("For every label the build takes up to 12 evidence-backed clips (my reading of the line in its scene, or a TLK "
      "note, at least 0.6) and asks two questions. On which features do at least 80% of them move the same way (the "
      "signature)? And is the pool further from his medians than random same-size sets of his own lines "
      "(permutation p, 4000 draws)?")
    w("")
    w(f"- **Validated** ({', '.join(validated) or 'none'}): coherent beyond chance. Only these get a direction score, "
      "measured-sounding VoxCPM2 wording, and a rank-1 rule that follows the measurement. With "
      f"{len(validated) + len(descriptive) + len(unmeasured)} labels tested, even these would not all survive a strict "
      "multiple-comparison correction: best supported, not certain.")
    w(f"- **Descriptive** ({', '.join(descriptive) or 'none'}): the clips agree on something, but no more than random "
      "sets of his lines do. Described below, never scored; rank 1 only has to not run against it.")
    w(f"- **No signature** ({', '.join(unmeasured) or 'none'}): nothing measurable is shared. The delivery lives in "
      "the words and the voice: judge it by ear.")
    w("")
    performer = (fill("performer") + " ") if T.get("performer") else ""
    w(f"{performer}Caution clips (speaker-suspect, atypical contour, prosody against the label, a TLK text he does "
      "not say, context-only) are kept out of `delivery_map.json`, so every listed reference is safe to prompt with "
      "and the eval's per-delivery targets come from clean clips only.")
    w("")
    w("| Label | Kind (perm p) | What his clips agree on (median vs his own, n agreeing) | Rank 1 | Rank 2 | Kept out |")
    w("|---|---|---|---|---|---|")
    words_for = {"f0_median_st": "pitch {:+.1f} st", "f0_range_st": "range {:+.1f} st", "final_rise_st": "ending {:+.1f} st",
                 "articulation_rate_syl_s": "{:+.1f} syl/s", "pause_ratio": "pause share {:+.0%}",
                 "alpha_ratio_db": "tilt {:+.1f} dB", "energy_slope_db": "energy slope {:+.1f} dB"}
    for label, x in labels.items():
        if label == "questioning" and x.get("question"):
            agree = (f"yes/no questions end {x['question']['questions_median_st']:+.1f} st "
                     f"({x['question']['agree']}/{x['question']['n']} rise)")
        elif label == "natural":
            agree = "(closeness to his medians)"
        else:
            agree = ", ".join(words_for[k].format(x["deltas"][k]) + f" ({s['agree']}/{s['n']})"
                              for k, s in x["signature"].items() if k in words_for and k in x["deltas"]) or "-"
        kind = x["kind"] if x["perm_p"] is None else f"{x['kind']} ({x['perm_p']:.4g})"
        w(f"| {label} | {kind} | {agree} | {short(x['rank1'])} \"{snippet(x['rank1_text'], 40)}\" "
          f"({x['rank1_duration']:.1f} s) | {short(x['rank2'])} | {ids(x['kept_out']) or '-'} |")
    w("")
    w("Per delivery (the map's own notes, generated from the same result):")
    w("")
    for label, entry in deliveries.items():
        w(f"- **{label.capitalize()}:** {entry['notes']}")
    w("")

    # ---------------------------------------------------------------- 4
    w("## 4. Directing notes per engine")
    w("")
    w("### GPT-SoVITS (`gsv/infer.py`, reference-prompted)")
    w("")
    w("The GPT (AR) model continues the prompt's semantic tokens, so **the prompt clip is the direction**.")
    w("- **Prompt:** the delivery's first valid reference; `infer.py` derives the audio path and the exact transcript "
      "from the corpus (original audio) and re-checks the 3-10 s window, WER and the speaker exclusions itself.")
    w("- **Two prompts per emotional label:** render with `--refs-per-line 2` (seeds rotate through ranks 1-2). Where "
      "rank 2 is the intent alternative (the TLK-noted take), that A/B is the point; decide by ear.")
    w("- **Aux references:** `aux_ref_audio_paths` only feed the SoVITS decoder, which averages the style vector "
      "(`ge`, from 0-11 kHz of each reference plus the v2Pro speaker vector) over main and aux references; the GPT "
      "prosody comes from the main prompt alone. One natural aux (`--aux-refs 1 --aux-delivery natural`) steadies "
      "identity at a 50/50 voice-quality mix. For deliveries whose colour is voice quality (weary, troubled, angry) "
      "try the label's own next clip as aux (`--aux-refs 1`) in a separate run.")
    w("- **Sampling: one setting for every delivery.** `infer.py`'s default preset is upstream's (top_k 15, top_p 1.0, "
      "temperature 1.0, repetition_penalty 1.35). Temperature changes sampling diversity and error risk, not emotion: "
      "sweep it once for all deliveries (`--preset stable` 0.8 vs default 1.0) and let the prompt carry the emotion.")
    w(f"- **Speed and splitting:** `--speed 1.0` always; the default split keeps 1-3 sentence lines whole. If a line "
      f"is split, set `--fragment-interval` near his {pz['longest_pause_in_multi_sentence_lines']['median']:.1f} s "
      "between sentences (default 0.3).")
    w("- **Double direction:** when the new line's words already carry strong emotion (the angry line), also try a "
      "milder prompt (resolute or natural rank 1); double-directed lines tip into melodrama.")
    w("")
    w("### Seed-VC V2 (`seedvc/convert.py`, voice conversion)")
    w("")
    w("- Timing, words and part of the contour come from the **source** take; register **and pitch range** come from "
      "the prompt (measured by the Seed-VC task: three sources shifted -4/0/+4 st all came out at the prompt's pitch).")
    w("- The prompt is the delivery's references accumulated in rank order until it has 6 s (`--ref-min-sec 6`, up "
      "to 3 clips, at most 15 s), so the top references' pitch placement becomes the output's. `--ref-select "
      "prosody` (closest pitch range to the source first) is the A/B.")
    w("- `--convert-style` stays off (default): the source's acting survives. On only as an A/B: it replaces the "
      "source's emotion and timing with the prompt's.")
    w("- Defaults: diffusion steps 30 (40-50 slightly cleaner), similarity and intelligibility cfg 0.7/0.7, "
      "`--length-adjust 1.0` (never to slow him down).")
    w("- Output is 22.05 kHz, the same bandwidth as the game's clips; bandwidth extension is an optional "
      "post-processing experiment.")
    w("")
    w("### VoxCPM2 (`voxcpm/render.py`, instruction-following, optionally with a LoRA)")
    w("")
    w("Two ways to get an emotion; `render.py` renders both by default (`--modes ref_prompt,idref_instr`). Never "
      "combine them: VoxCPM2 joins `prompt_text + text`, so an instruction would land in the middle of the sequence.")
    w(f"- **Controlled cloning (`idref_instr`, Vox's production convention):** text = `\"(\" + voxcpm_instruction + "
      f"\")\" + line`, `reference_wav_path` = one fixed identity clip for every line (natural rank 1, {short(natural1)}), "
      "no prompt. With an adapter, `render.py --identity-clause auto` prefixes \"Keep the trained voice stable.\" (Vox's "
      "identity clause) and capitalises the instruction.")
    w("- **Ultimate cloning (`ref_prompt`):** `prompt_wav_path` + `prompt_text` = the delivery's rank-1 clip and its "
      "exact text plus a trailing space, `reference_wav_path` = the same clip (default) or natural rank 1 "
      "(`--identity-ref natural`). The clip's delivery carries into the line.")
    if T.get("instruction_wording"):
        w(fill("instruction_wording"))
    cfg_note = (f"render.py's default and Vox's {labcfg.NAME} profile" if labcfg.CHARACTER.get("vox_profile_id")
                else "render.py's default")
    w(f"- **Settings:** `--cfg 1.5` ({cfg_note}; the official default is 2.0, worth one sweep), 10 timesteps, "
      "normalize off, denoiser off, at least 3 seeds per line.")
    w("")
    w("### Every engine: selection loop")
    w("")
    w("1. **Render** 3-4 seeds per line, `--refs-per-line 2` for the emotional deliveries. Optionally render "
      f"`takes_script_pairs.json` with the same settings: natural reads of the emotional new lines (ids ending "
      f"`{PAIR_SUFFIX}`), the cleanest A/B by ear and a paired baseline for the scorer.")
    w("2. **Score** with `score_delivery.py --manifest <run>/manifest.json [--baseline-manifest <pairs run>/manifest.json]`:")
    w(f"   - **Direction** only for validated deliveries ({', '.join(validated) or 'none'}; natural = closeness to his "
      "medians): how far the take moved, on the label's signature, from a paired neutral read of the same words (else "
      "from his length-matched medians). 1 = as far as his clips, 0 = no movement, below 0 = the wrong way; `along` "
      "above 2 is flagged as overacting.")
    if identity:
        w(fill("identity") if imp.get("n") else fill("identity_no_impostors"))
    w("   - **Words:** Whisper WER (deterministic decoding) catches wrong words and instructions read aloud.")
    w("3. **Drop** takes flagged sing-song, rushed, dragging, flat (bounds widened to his own clips of that delivery), "
      "overacted, below the identity gate, WER above 0.15, or with the instruction spoken.")
    w("4. **Rerank:** `reference_rerank.json` proposes a reference order from multi-reference renders: a proposal to "
      "audition, never applied automatically.")
    w("5. " + fill("master_step", "**Master and listen** blind against the held-out originals (match their level and "
                                  "the lead/tail silence above)."))
    w("")
    w("One take's features wobble by about one of his standard deviations, so the scorer reports means with standard "
      "errors over seeds and lines, and the probe picks an alternative only on a paired confidence interval.")
    w("")

    # ---------------------------------------------------------------- 5
    w("## 5. The takes script")
    w("")
    w(f"New lines (written for this lab, checked against all {corpus_lines} corpus texts and the held-out lines):")
    w("")
    w("| Line | Delivery | Words | Est. seconds at his pace |")
    w("|---|---|---|---|")
    texts = {l["id"]: l["text"] for l in script["lines"]}
    for line in facts["script"]["new"]:
        w(f"| {line['id']}: \"{texts[line['id']]}\" | {line['delivery']} | {line['words']} | {line['seconds']} |")
    w("")
    w("Held-out lines: each delivery was chosen from the script alone (the words plus the TLK note), never from the "
      "held-out audio.")
    w("")
    w("| Held-out line | Delivery | Why |")
    w("|---|---|---|")
    why = heldout_delivery()
    for clip_id, delivery in sorted(facts["script"]["heldout"].items()):
        w(f"| {short(clip_id)} | {delivery} | {why[clip_id][1]} |")
    w("")

    # ---------------------------------------------------------------- 6
    w("## 6. Reference rules the map guarantees (checked by `validate.py`)")
    w("")
    w(f"- {facts['references_total']} references ({facts['reference_clips_unique']} clips) from {facts['eligible']} "
      f"eligible clips; {facts['excluded']} train/val clips were not eligible (reasons in `delivery_candidates.json`).")
    w("- Train or val only, never test. Speech with exact TLK/DLG text identical to `corpus.jsonl` and ASR WER 0.15 or "
      "less. 3.0-10.0 s, inside GPT-SoVITS's prompt window at 16 kHz; sha256 equals the corpus.")
    w("- Not another character, not uncertain, not a near-duplicate of a held-out line, and no text overlap with one.")
    w("- No caution clip anywhere in the map. Rank 1 is unique across deliveries and meets its label's rank-1 rule; a "
      "clip appears in at most two deliveries.")
    w("- The shipped map, label stats, candidates, instruction variants, exclusions, takes scripts, facts and this "
      "bible are byte-identical to a fresh rebuild from the code.")
    w("")

    # ---------------------------------------------------------------- 7
    w("## 7. Next steps on the GPU (main session, one at a time)")
    w("")
    py = "%LOCALAPPDATA%\\VoxStudio\\engines\\voxcpm2\\.venv\\Scripts\\python.exe"
    env = [f"`VOICE_LAB_CHARACTER={labcfg.SLUG}`"] if not labcfg.IS_DEFAULT else []
    env += ["`PYTHONDONTWRITEBYTECODE=1`", "`HF_HUB_OFFLINE=1`"]
    recipes = f"`{HERE.parent.name}\\delivery`" if labcfg.IS_DEFAULT else "the delivery recipe folder"
    w(f"All with the VoxCPM2 runtime python (`{py}`) from {recipes}, with {', '.join(env)} and "
      f"`NUMBA_CACHE_DIR={LAB.name}\\delivery\\.cache\\numba`.")
    w("")
    w("1. `probe_voxcpm.py --run instr_probe_v1`: base VoxCPM2, 4 seeds x 4 lines x A/B/P/P2 per delivery + a neutral "
      f"read of every line ({probe_takes(deliveries, variants, script)} takes). Refuses to start unless ~15 GB of "
      "Windows commit (VoxCPM2 builds in fp32 on the CPU first) and 9 + 8 GB of VRAM are free; its 9 GB budget "
      "includes ~1 GB of CUDA context, so torch's allocator is capped at 8 GB. `--resume` continues an interrupted "
      "run; `--labels` splits it.")
    w("2. `score_delivery.py --manifest <probe>\\manifest.json --asr cuda --pick-instructions --no-build-centroid`: "
      "Whisper on the GPU only if it leaves 8 GB free (else CPU). Listen to each delivery's takes, edit "
      "`voxcpm_instruction_probe.json` if your ears disagree, then apply on the CPU with `score_delivery.py "
      "--apply-from <probe>\\voxcpm_instruction_probe.json` (stores the exact rendered instruction in "
      "`instruction_picks.json` and rebuilds the map; then rerun `bible_facts.py`, `make_bible.py`, `validate.py`).")
    w("3. After the voxcpm task trains its adapter: repeat 1-2 with `--lora <checkpoint> --run instr_probe_lora_v1`.")
    w(f"4. After each engine's render: `score_delivery.py --manifest {LAB.name}\\takes\\<engine>\\<run>\\manifest.json "
      "--asr cuda --no-build-centroid` (plus `--baseline-manifest` for a pairs run).")
    if picks:
        w("")
        w("Applied probe picks: " + "; ".join(f"{k}: {p_['variant']} ({p_['run']})" for k, p_ in sorted(picks.items())) + ".")
    w("")

    # ---------------------------------------------------------------- 8
    w("## 8. What I could not verify")
    w("")
    for item in T.get("unverified", []):
        w(item.format_map(values))
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Render {labcfg.NAME}'s performance_bible.md from the computed outputs.")
    parser.add_argument("--map-dir", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file differs from a fresh render")
    args = parser.parse_args()
    target = args.out or args.map_dir / "performance_bible.md"
    text = render(args.map_dir)
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        print("performance_bible.md is current" if current == text else "performance_bible.md is OUT OF DATE")
        sys.exit(0 if current == text else 1)
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
