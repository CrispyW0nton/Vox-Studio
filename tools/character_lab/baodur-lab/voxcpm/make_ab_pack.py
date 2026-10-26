"""Blind A/B listening pack for the director (CPU, VoxCPM2 runtime python: numpy + soundfile).

  python make_ab_pack.py --spec pairs_spec.json --out <lab>/listening/ab_saberstory_v001 --key <lab>/listening/ab_saberstory_v001_KEY.json

spec: {"title": str, "seed": int, "pairs": [{"line_id", "text", "delivery", "script_id"?, "comparison",
        "x": {"label", "wav"}, "y": {"label", "wav"}, "original"?: wav}]}
Writes <out>/index.html (play A, B and, for held-out lines, the original game recording; pick A / B / no
preference; answers stay in the browser and export as answers.csv), <out>/sheet.csv (same thing as a sheet),
<out>/audio/pNN_A.wav, pNN_B.wav, pNN_original.wav. Pair order and A/B sides are randomised with --seed; the
mapping is written ONLY to --key (outside the pack folder). Each pair is loudness-matched: both takes are scaled
to the same active-speech RMS (-23 dBFS on frames within 30 dB of the take's peak frame level), so loudness does
not vote. The original recording is copied unchanged apart from the same level match.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

TARGET_DB = -23.0


def active_rms_db(x: np.ndarray, rate: int) -> float:
    hop = max(1, int(rate * 0.01))
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 10 * np.log10(np.maximum((frames ** 2).mean(axis=1), 1e-12))
    act = frames[db > db.max() - 30]
    return float(10 * np.log10(max((act ** 2).mean(), 1e-12)))


def write_matched(src: str, dst: Path) -> dict:
    x, rate = sf.read(src, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    level = active_rms_db(x, rate)
    gain = 10 ** ((TARGET_DB - level) / 20)
    y = x * gain
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 0.97:
        y = y * (0.97 / peak)
    sf.write(str(dst), y, rate, subtype="PCM_16")
    return {"src": src, "level_db": round(level, 2), "gain_db": round(20 * np.log10(gain), 2),
            "peak_limited": peak > 0.97, "seconds": round(len(y) / rate, 2)}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bao-Dur blind A/B</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e2e0da;--acc:#3a5ba0;--sel:#e7eefb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e22;--ink:#ececf0;--mute:#9a9aa3;--line:#33333a;--acc:#8fb0f0;--sel:#24324d}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0 0 4px} .mute{color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
.hd{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap} .n{font-weight:700} .line{font-style:italic}
.row{display:grid;grid-template-columns:90px 1fr;gap:8px;align-items:center;margin:6px 0}
audio{width:100%;height:36px} .orig{opacity:.85}
.pick{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.pick label{border:1px solid var(--line);border-radius:8px;padding:5px 12px;cursor:pointer}
.pick input{margin-right:6px} .pick label:has(input:checked){background:var(--sel);border-color:var(--acc)}
textarea{width:100%;box-sizing:border-box;margin-top:8px;background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px;font:inherit;min-height:34px}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:9px 16px;font:inherit;cursor:pointer}
.bar{position:sticky;bottom:0;background:var(--bg);padding:10px 0;border-top:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
</style></head><body><main>
<h1>Bao-Dur blind A/B: saber story</h1>
<p class="mute">__INTRO__</p>
__CARDS__
<div class="bar"><button id="exp">Export answers.csv</button><span class="mute" id="cnt"></span></div>
</main>
<script>
const KEY="ab_saberstory_v001";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){};count()};
const count=()=>{const n=Object.values(st).filter(v=>v&&v.pick).length;document.getElementById("cnt").textContent=n+" / __N__ answered"};
document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.pair;const s=st[id]||{};
 c.querySelectorAll("input[type=radio]").forEach(r=>{if(s.pick===r.value)r.checked=true;r.addEventListener("change",()=>{st[id]={...(st[id]||{}),pick:r.value};save()})});
 const t=c.querySelector("textarea");t.value=s.note||"";t.addEventListener("input",()=>{st[id]={...(st[id]||{}),note:t.value};save()})});
document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(b!==a)b.pause()})));
document.getElementById("exp").onclick=()=>{const rows=[["pair","line","pick","note"]];
 document.querySelectorAll(".card").forEach(c=>{const s=st[c.dataset.pair]||{};rows.push([c.dataset.pair,c.dataset.line,s.pick||"",(s.note||"").replace(/"/g,"'")])});
 const csv=rows.map(r=>r.map(v=>'"'+v+'"').join(",")).join("\\r\\n");const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="answers.csv";a.click()};
count();
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    args = p.parse_args()
    if args.key.resolve().is_relative_to(args.out.resolve()):
        raise SystemExit("--key must be outside the pack folder")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rng = random.Random(int(spec.get("seed", 20261007)))
    pairs = list(spec["pairs"])
    rng.shuffle(pairs)
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    key_rows, cards, sheet = [], [], []
    for i, pr in enumerate(pairs, 1):
        pid = f"p{i:02d}"
        sides = [pr["x"], pr["y"]]
        if rng.random() < 0.5:
            sides.reverse()
        a, b = sides
        info_a = write_matched(a["wav"], audio / f"{pid}_A.wav")
        info_b = write_matched(b["wav"], audio / f"{pid}_B.wav")
        orig = None
        if pr.get("original"):
            orig = write_matched(pr["original"], audio / f"{pid}_original.wav")
        key_rows.append({"pair": pid, "line_id": pr["line_id"], "comparison": pr["comparison"],
                         "A": a["label"], "B": b["label"], "A_wav": a["wav"], "B_wav": b["wav"],
                         "levels": {"A": info_a, "B": info_b, "original": orig}})
        title = (pr.get("script_id") or pr["line_id"]).replace("heldout_", "held-out ")
        orig_row = (f'<div class="row orig"><span class="mute">Original game</span>'
                    f'<audio controls preload="none" src="audio/{pid}_original.wav"></audio></div>') if orig else ""
        cards.append(
            f'<section class="card" data-pair="{pid}" data-line="{html.escape(pr["line_id"])}">'
            f'<div class="hd"><span class="n">{pid}</span><span class="mute">{html.escape(title)} - '
            f'{html.escape(pr.get("delivery", ""))}</span></div>'
            f'<div class="line">"{html.escape(pr["text"])}"</div>{orig_row}'
            f'<div class="row"><b>A</b><audio controls preload="none" src="audio/{pid}_A.wav"></audio></div>'
            f'<div class="row"><b>B</b><audio controls preload="none" src="audio/{pid}_B.wav"></audio></div>'
            f'<div class="pick">' + "".join(
                f'<label><input type="radio" name="{pid}" value="{v}">{t}</label>'
                for v, t in (("A", "A is better"), ("B", "B is better"), ("same", "No preference"))) +
            '</div><textarea placeholder="Optional note (delivery, pauses, endings, artefacts)"></textarea></section>')
        sheet.append({"pair": pid, "line": pr["line_id"], "text": pr["text"], "pick (A/B/same)": "", "note": ""})
    intro = (f"{len(pairs)} pairs. Each pair is the same line rendered two ways; which one is more Bao-Dur "
             "(his voice, his halting pauses, falling line ends, no rushing)? Held-out lines also have his original "
             "game recording for reference. Levels are matched. Your picks are kept in this browser; press Export "
             "when done (or fill sheet.csv). Generated " + time.strftime("%Y-%m-%d %H:%M") + ".")
    page = PAGE.replace("__INTRO__", html.escape(intro)).replace("__CARDS__", "\n".join(cards)).replace(
        "__N__", str(len(pairs)))
    (args.out / "index.html").write_text(page, encoding="utf-8")
    with (args.out / "sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        w.writerows(sheet)
    (args.out / "README.txt").write_text(
        "Open index.html in a browser (double-click). Play A and B for each pair, pick the better Bao-Dur take,\n"
        "then press 'Export answers.csv' (or fill sheet.csv). The answer key is NOT in this folder.\n",
        encoding="utf-8")
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps({"pack": str(args.out), "title": spec.get("title"), "seed": spec.get("seed"),
                                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": key_rows}, indent=1),
                        encoding="utf-8")
    print(f"{len(pairs)} pairs -> {args.out}; key -> {args.key}")


if __name__ == "__main__":
    main()
