"""Score picotone music24/12/6 on the held-out eval excerpts, and print them next to stock DAC and the student.

    uv run python compare_picotone.py --eval-set data\\pd.jsonl.eval.json --exe C:\\...\\picotone.exe \\
        --out data\\picotone_eval.json [--run runs\\pilot512]

Each excerpt is cut at its native rate (48 kHz for the picotone test clips), encoded with
`picotone encode --preset <p>`, decoded with `picotone decode`, resampled to 44.1 kHz and scored against the
same 44.1 kHz reference the trainer uses. Bitrate = .pico bytes x 8 / seconds (10 s excerpts carry a little
more header and page-reset cost than whole files). With --run, adds the run's stock-DAC and latest student
rows; DAC bitrates are raw (10 bits per code, 86.13 frames/s, 2 channels), before any entropy coding.
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

import audio_io
import corpus
import metrics

DAC_FPS = 44100 / 512


def dac_kbps(n_q):
    return n_q * 10 * DAC_FPS * 2 / 1000


def set_means(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}):
        out[s] = metrics.mean_scores([r["m"] for r in rows if r["set"] == s])
    out["all"] = metrics.mean_scores([r["m"] for r in rows])
    return out


def score_picotone(a):
    clips = []
    for p in a.eval_set:
        clips += json.load(open(p, encoding="utf-8"))
    if a.max_clips:
        clips = clips[: a.max_clips]
    work = a.work or tempfile.mkdtemp(prefix="pico_cmp_")
    os.makedirs(work, exist_ok=True)
    res, kb = {}, {}
    for preset in a.presets.split(","):
        rows, bits, secs = [], {}, {}
        for i, e in enumerate(clips):
            x, sr = audio_io.decode(e["ref"], e["offset"], e["dur"])
            ref = audio_io.decode_44k(e["ref"], e["offset"], e["dur"])
            T = ref.shape[1] // 512 * 512
            ref = ref[:, :T]
            wav, pico, out = (os.path.join(work, f"{i:03d}{s}") for s in (".wav", f"_{preset}.pico",
                                                                          f"_{preset}.wav"))
            if x.shape[0] == 1:
                x = np.repeat(x, 2, axis=0)
            sf.write(wav, x.T, sr, subtype="FLOAT")
            for cmd in (["encode", wav, pico, "--preset", preset], ["decode", "--float", pico, out]):
                r = subprocess.run([a.exe, "--threads", str(a.threads), *cmd], capture_output=True, text=True,
                                   timeout=600)
                if r.returncode != 0:
                    raise SystemExit(f"picotone {cmd[0]} failed on {e['name']}: {r.stderr[-500:]}")
            y, ysr = sf.read(out, dtype="float32", always_2d=True)
            y = audio_io.resample(y.T, ysr)[: ref.shape[0], :T]
            rows.append({"name": e["name"], "set": e["set"], "m": metrics.score(ref, y)})
            bits[e["set"]] = bits.get(e["set"], 0) + os.path.getsize(pico) * 8
            secs[e["set"]] = secs.get(e["set"], 0) + e["dur"]
        res[preset] = set_means(rows)
        kb[preset] = {s: bits[s] / secs[s] / 1000 for s in bits}
        kb[preset]["all"] = sum(bits.values()) / sum(secs.values()) / 1000
        print(f"picotone {preset}: {kb[preset]['all']:.2f} kbps | " + " | ".join(
            f"{s}: mel {m['logmel']:.3f} mrstft {m['mrstft']:.3f} sisdr {m['sisdr']:.2f}" for s, m in
            res[preset].items()), flush=True)
    res["_kbps"] = kb
    return res


def table(pico, run):
    rows = []
    for p in ("music24", "music12", "music6"):
        if p in pico:
            rows.append((f"picotone {p}", pico["_kbps"][p]["all"], pico[p]))
    if run:
        stock = json.load(open(os.path.join(run, "stock_eval.json")))
        for nq, v in sorted(stock.items(), key=lambda kv: int(kv[0])):
            rows.append((f"stock DAC q{nq}", dac_kbps(int(nq)), v["means"]))
        evp = os.path.join(run, "evals.jsonl")
        if os.path.exists(evp):
            ev = corpus.read_jsonl(evp)
            last = {}
            for e in ev:
                last[e["n_q"]] = e
            for nq, e in sorted(last.items()):
                rows.append((f"student q{nq} @ step {e['step']}", dac_kbps(nq), e["means"]))
    sets = [s for s in ("old", "new", "all") if all(s in r[2] for r in rows)]
    head = f"{'codec':32s} {'kbps':>6s} " + " ".join(f"{s + ' mel':>9s} {s + ' mrstft':>11s} {s + ' sisdr':>10s}"
                                                         for s in sets)
    print(head)
    for name, kbps, m in rows:
        print(f"{name:32s} {kbps:6.2f} " + " ".join(f"{m[s]['logmel']:9.3f} {m[s]['mrstft']:11.3f} "
                                                    f"{m[s]['sisdr']:10.2f}" for s in sets))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", action="append", default=[])
    ap.add_argument("--exe", default=r"C:\Users\richa\workspace_local\encodecx\target\release\picotone.exe")
    ap.add_argument("--presets", default="music24,music12,music6")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True, help="json with picotone scores (reused if it exists)")
    ap.add_argument("--work", default="")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--run", default="", help="training run dir to tabulate against")
    a = ap.parse_args()
    if os.path.exists(a.out):
        pico = json.load(open(a.out))
    else:
        pico = score_picotone(a)
        json.dump(pico, open(a.out, "w"), indent=1)
    table(pico, a.run)


if __name__ == "__main__":
    main()
