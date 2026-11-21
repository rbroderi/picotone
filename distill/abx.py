"""Blind ABX listening test between two codebook counts of a trained student (default q5 against q6).

    uv run --extra cpu python abx.py --ckpt runs\\w512_tiers\\ckpt_best.pt --eval-set data\\pd.jsonl.eval.json \\
        --out abx_q5_q6 --a 5 --b 6 --serve

For each chosen eval clip it writes the original and the two decodes as WAVs, plus index.html, a page that
plays A, B and X sample-aligned (switching keeps the playback position) and asks whether X is A or B. The page
scores the answers and gives the one-sided binomial p-value; p < 0.05 means you can hear a difference.
--serve starts a local web server for the folder (the page needs http:// to load its audio) and opens it.

Weights: ckpt_best.pt holds one set ("G"); for ckpt_last.pt / ckpt_<step>.pt the smoothed "G_ema" is used when
present, unless --raw. Decoding runs in fp32 on the CPU; 8 clips of 10 s take about a minute.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch

import audio_io
import data
from student import StudentDecoder
from teacher import codes_to_z, encode_codes, load_teacher

SR = audio_io.SR
DEFAULT_CLIPS = ("old_10_piano,old_13_piano,old_08_orchestraLR,old_00_band,"
                 "new_19_jazz,new_25_rock,new_10_game,new_03_electronic")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-set", action="append", required=True, help="eval JSON(s) holding the clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=int, default=5, help="codebook count of A")
    ap.add_argument("--b", type=int, default=6, help="codebook count of B")
    ap.add_argument("--clips", default=DEFAULT_CLIPS, help="comma list of eval clip names")
    ap.add_argument("--trials", type=int, default=2, help="trials per clip (X drawn at random each time)")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--raw", action="store_true", help="use G, not G_ema, from a training checkpoint")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--serve", action="store_true", help="serve the folder on localhost and open the page")
    ap.add_argument("--port", type=int, default=8741)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    G = StudentDecoder(s["width"]).eval()
    use_ema = s.get("G_ema") is not None and not a.raw
    G.load_state_dict(s["G_ema"] if use_ema else s["G"])
    which = s.get("w", "ema" if use_ema else "raw")
    print(f"{a.ckpt}: step {s['step']}, {which} weights, width {s['width']}", flush=True)

    maps = data.parse_path_map(a.path_map)
    want = [c.strip() for c in a.clips.split(",") if c.strip()]
    pool = {}
    for p in a.eval_set:
        for e in json.load(open(p, encoding="utf-8")):
            pool[e["name"]] = e
    missing = [c for c in want if c not in pool]
    if missing:
        raise SystemExit(f"not in the eval set(s): {missing}; available: {sorted(pool)}")

    T = load_teacher()
    clips = []
    for name in want:
        e = pool[name]
        x = audio_io.decode_44k(data.remap_ref(e["ref"], maps), e["offset"], e["dur"])
        x = np.ascontiguousarray(x[:, : x.shape[1] // 512 * 512])
        with torch.no_grad():
            codes = encode_codes(T, torch.from_numpy(x)[:, None])
            outs = {}
            for nq in (a.a, a.b):
                y = G(codes_to_z(T, codes, nq).float()).float()[:, 0].clamp(-1, 1).numpy()
                outs[nq] = y
        sf.write(os.path.join(a.out, f"{name}__orig.wav"), x.T, SR, subtype="PCM_16")
        for nq, y in outs.items():
            sf.write(os.path.join(a.out, f"{name}__q{nq}.wav"), y.T, SR, subtype="PCM_16")
        clips.append({"name": name, "title": e.get("title", ""), "artist": e.get("artist", ""),
                      "licence": e.get("licence", "")})
        print(f"  {name}", flush=True)

    rng = random.Random()
    trials = [{"clip": c["name"], "x_is": rng.choice("AB")} for c in clips for _ in range(a.trials)]
    rng.shuffle(trials)
    cfg = {"a": f"q{a.a}", "b": f"q{a.b}", "clips": clips, "trials": trials,
           "kbps": {f"q{n}": round(n * 10 * SR / 512 * 2 / 1000, 2) for n in (a.a, a.b)},
           "ckpt": f"{os.path.basename(a.ckpt)} step {s['step']} ({which})"}
    html = PAGE.replace("/*CFG*/null", json.dumps(cfg))
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {len(clips)} clips and {len(trials)} trials to {a.out}\\index.html", flush=True)

    if a.serve:
        import functools
        import http.server
        import webbrowser

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=a.out)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
        url = f"http://127.0.0.1:{a.port}/index.html"
        print(f"serving {url} (Ctrl+C to stop)", flush=True)
        webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ABX listening test</title>
<style>
:root { --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#ddddd6; --accent:#2b59c3; --on:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94; --card:#20201e;
  --line:#3a3a36; --accent:#7aa2ff; --on:#0d0d0c; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width:640px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.35rem; margin:0 0 4px; }
.muted { color:var(--muted); font-size:.9rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-top:16px; }
.row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
button { font:inherit; padding:10px 18px; border-radius:8px; border:1px solid var(--line); background:var(--card);
  color:var(--fg); cursor:pointer; min-width:64px; }
button.on { background:var(--accent); color:var(--on); border-color:var(--accent); }
button.pick { border-color:var(--accent); }
button:disabled { opacity:.45; cursor:default; }
table { width:100%; border-collapse:collapse; margin-top:8px; font-size:.95rem; }
td, th { text-align:left; padding:4px 6px; border-bottom:1px solid var(--line); }
kbd { font:inherit; font-size:.8rem; padding:0 5px; border:1px solid var(--line); border-radius:4px; }
</style></head><body><main>
<h1>ABX listening test</h1>
<div class="muted" id="sub"></div>
<div class="card" id="trial"></div>
<div class="card" id="results" hidden></div>
<p class="muted">Keys: <kbd>1</kbd> A, <kbd>2</kbd> B, <kbd>3</kbd> X, <kbd>Space</kbd> stop, <kbd>a</kbd>/<kbd>b</kbd>
answer. Switching keeps the playback position. Each clip loops.</p>
</main>
<script>
const CFG = /*CFG*/null;
const ctx = new AudioContext();
const buffers = {};
let cur = 0, answers = [], src = null, playing = null, startAt = 0, offset = 0;

async function load(name) {
  if (buffers[name]) return buffers[name];
  const r = await fetch(name);
  buffers[name] = await ctx.decodeAudioData(await r.arrayBuffer());
  return buffers[name];
}
function fileOf(trial, which) {
  const t = which === "X" ? trial.x_is : which;
  return `${trial.clip}__${t === "A" ? CFG.a : CFG.b}.wav`;
}
function stop() {
  if (src) { offset = (ctx.currentTime - startAt + offset) % src.buffer.duration; src.stop(); src = null; }
  playing = null; render();
}
async function play(which) {
  const tr = CFG.trials[cur];
  const buf = await load(fileOf(tr, which));
  if (ctx.state === "suspended") await ctx.resume();
  if (src) { offset = (ctx.currentTime - startAt + offset) % src.buffer.duration; src.stop(); }
  src = ctx.createBufferSource(); src.buffer = buf; src.loop = true; src.connect(ctx.destination);
  startAt = ctx.currentTime; src.start(0, offset % buf.duration); playing = which; render();
}
function answer(choice) {
  const tr = CFG.trials[cur];
  answers.push({ clip: tr.clip, x_is: tr.x_is, said: choice, ok: choice === tr.x_is });
  if (src) { src.stop(); src = null; } playing = null; offset = 0; cur++;
  if (cur < CFG.trials.length) preload(cur); render();
}
function preload(i) { const t = CFG.trials[i]; ["A", "B"].forEach(w => load(fileOf(t, w))); }
function binomP(k, n) {  // P(at least k of n correct by guessing)
  let p = 0; for (let i = k; i <= n; i++) { let c = 1; for (let j = 0; j < i; j++) c = c * (n - j) / (j + 1); p += c; }
  return p / Math.pow(2, n);
}
function render() {
  document.getElementById("sub").textContent =
    `A = ${CFG.a} (${CFG.kbps[CFG.a]} kbps), B = ${CFG.b} (${CFG.kbps[CFG.b]} kbps). ${CFG.ckpt}.`;
  const el = document.getElementById("trial");
  if (cur >= CFG.trials.length) { el.hidden = true; return results(); }
  const tr = CFG.trials[cur];
  const btn = (w, k) => `<button class="${playing === w ? "on" : ""}" onclick="play('${w}')">${w}</button>`;
  el.innerHTML = `<div class="muted">Trial ${cur + 1} of ${CFG.trials.length}, clip ${tr.clip}</div>
    <div class="row">${btn("A")}${btn("B")}${btn("X")}<button onclick="stop()">Stop</button></div>
    <div class="row"><button class="pick" onclick="answer('A')">X is A</button>
    <button class="pick" onclick="answer('B')">X is B</button></div>`;
}
function results() {
  const el = document.getElementById("results"); el.hidden = false;
  const n = answers.length, k = answers.filter(a => a.ok).length, p = binomP(k, n);
  const per = {};
  answers.forEach(a => { per[a.clip] = per[a.clip] || [0, 0]; per[a.clip][1]++; if (a.ok) per[a.clip][0]++; });
  const rows = Object.entries(per).map(([c, [ok, all]]) => `<tr><td>${c}</td><td>${ok} / ${all}</td></tr>`).join("");
  const verdict = p < 0.05 ? `You can hear a difference (p = ${p.toFixed(3)}).`
    : `Not distinguishable in this test (p = ${p.toFixed(3)}; guessing gives about half right).`;
  el.innerHTML = `<b>${k} of ${n} correct.</b> ${verdict}<table><tr><th>Clip</th><th>Correct</th></tr>${rows}</table>
    <div class="row"><button onclick="save()">Save results (JSON)</button></div>`;
}
function save() {
  const blob = new Blob([JSON.stringify({ cfg: { a: CFG.a, b: CFG.b, ckpt: CFG.ckpt }, answers }, null, 1)],
                        { type: "application/json" });
  const u = URL.createObjectURL(blob), l = document.createElement("a");
  l.href = u; l.download = `abx_${CFG.a}_${CFG.b}.json`; l.click(); URL.revokeObjectURL(u);
}
document.addEventListener("keydown", e => {
  if (cur >= CFG.trials.length) return;
  if (e.key === "1") play("A"); else if (e.key === "2") play("B"); else if (e.key === "3") play("X");
  else if (e.key === " ") { e.preventDefault(); stop(); }
  else if (e.key === "a") answer("A"); else if (e.key === "b") answer("B");
});
preload(0); render();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
