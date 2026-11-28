"""Check the sampled left/right vs mid/side choice (src/stereo.rs) against a full comparison, on whole tracks.

    cd train/hq
    uv run --extra cuda python ../../tools/stereo_pick_check.py --manifest data/jamendo.jsonl --manifest data/pd.jsonl \
        --limit 20 --out stereo_pick.jsonl [--path-map OLD=NEW] [--ckpt runs/vocos/ckpt_best.pt]

It runs in train/hq's environment and imports its helpers (audio_io, corpus, data, metrics, teacher).

Per track (the first --max-seconds of it):
  1. the mono check (stereo_scan.py's rule: the loudest-side page below --mono-db). Mono tracks are recorded
     and skipped: they have no layout to choose;
  2. full decision: the whole track encoded, decoded and scored in both layouts (left/right --lr per channel,
     mid/side --ms MID,SIDE codebooks). Score = mean log-mel over L, R and the side (L-R)/2; lower is better;
  3. sampled decision: --excerpts windows of --excerpt-s, one per equal region of the track (the loudest
     window in each region, so away from silence), each encoded and decoded with --context-s of audio on
     both sides and scored on its centre only.
Every excerpt's scores are saved, so the summary can replay the choice from the first k excerpts and the
early exit (stop after 2-3 excerpts when the margin is large) without re-running anything.

Rows are appended to --out as each track finishes; a re-run skips tracks already there. Decoding uses stock
DAC unless --ckpt names a train.py --arch vocos checkpoint (its smoothed weights unless --raw).

Rule for long runs: a 5-minute track takes seconds on a GPU and several minutes on CPU. A track still
sampling after --track-timeout (default 900 s) stops there and is marked "timeout"; a track that fails is
logged with its error and skipped. Run it under an overall time limit too, or check on it every 10-15 min.
"""
import argparse
import json
import os
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train", "hq"))  # noqa: E402

import numpy as np
import torch

import audio_io
import corpus
import data
import metrics
from stereo_scan import page_side_db
from teacher import HOP, codes_to_z, encode_codes, load_teacher

SR = 44100


class Codec:
    def __init__(self, a, dev):
        self.T = load_teacher(dev)
        self.dev = dev
        self.G = None
        if a.ckpt:
            from vocos_student import make_decoder
            s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            self.G = make_decoder(s["arch"]).eval().to(dev)
            use_ema = s.get("G_ema") is not None and not a.raw
            self.G.load_state_dict(s["G_ema"] if use_ema else s["G"])
            self.name = f"decoder {os.path.basename(a.ckpt)} step {s['step']} ({'ema' if use_ema else 'raw'})"
        else:
            self.name = "stock DAC"
        self.chunk_f = int(a.chunk_s * SR / HOP)
        self.enc_ctx_f = int(a.encode_context_s * SR / HOP)
        self.dec_ctx_f = 32  # the Vocos decoder reaches +-29 frames

    @torch.no_grad()
    def encode(self, x):
        """x [T] float32 (T a multiple of HOP) -> codes [9, F], encoded in chunks with context."""
        F = len(x) // HOP
        out = []
        for f0 in range(0, F, self.chunk_f):
            f1 = min(F, f0 + self.chunk_f)
            a0, a1 = max(0, f0 - self.enc_ctx_f), min(F, f1 + self.enc_ctx_f)
            seg = torch.from_numpy(np.ascontiguousarray(x[a0 * HOP:a1 * HOP])).to(self.dev)[None, None]
            c = encode_codes(self.T, seg)[0]
            out.append(c[:, f0 - a0:f0 - a0 + (f1 - f0)])
        return torch.cat(out, 1)

    @torch.no_grad()
    def decode(self, codes, q):
        """codes [9, F] -> audio [F*HOP] float32 numpy, decoded in chunks with context."""
        F = codes.shape[1]
        out = []
        for f0 in range(0, F, self.chunk_f):
            f1 = min(F, f0 + self.chunk_f)
            a0, a1 = max(0, f0 - self.dec_ctx_f), min(F, f1 + self.dec_ctx_f)
            z = codes_to_z(self.T, codes[None, :, a0:a1], q).float()
            y = (self.G(z) if self.G is not None else self.T.decoder(z)).float()[0, 0]
            out.append(y[(f0 - a0) * HOP:(f0 - a0 + f1 - f0) * HOP].cpu().numpy())
        return np.concatenate(out)

    def layouts(self, x, lr_q, ms_q):
        """x [2, T] -> {'lr': y [2, T], 'ms': y [2, T]}, clipped to [-1, 1]."""
        m, s = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
        y_lr = np.stack([self.decode(self.encode(x[0]), lr_q), self.decode(self.encode(x[1]), lr_q)])
        ym, ys = self.decode(self.encode(m), ms_q[0]), self.decode(self.encode(s), ms_q[1])
        return {"lr": np.clip(y_lr, -1, 1), "ms": np.clip(np.stack([ym + ys, ym - ys]), -1, 1)}


def lm(r, y):
    return metrics.score(r[None], y[None])["logmel"]  # nan when r is digital silence


def stereo_score(x, y):
    t = [lm(x[0], y[0]), lm(x[1], y[1]), lm((x[0] - x[1]) / 2, (y[0] - y[1]) / 2)]
    return float(np.nanmean(t)), t


def excerpt_windows(x, n, win, ctx):
    """n windows of `win` samples (frame aligned), the loudest in each of n equal regions, leaving `ctx`
    samples free on both sides where the track allows."""
    T = x.shape[1]
    e = np.mean(x ** 2, 0)
    hop = SR // 2
    edges = np.linspace(0, T, n + 1).astype(int)
    out = []
    for r0, r1 in zip(edges, edges[1:]):
        lo, hi = max(r0, ctx), min(r1, T - ctx) - win
        if hi < lo:
            lo, hi = r0, max(r0, r1 - win)
        best, best_e = lo, -1.0
        for s in range(lo, hi + 1, hop):
            v = float(np.mean(e[s:s + win]))
            if v > best_e:
                best, best_e = s, v
        out.append(best // HOP * HOP)
    return out


def check_track(ref, a, codec):
    t0 = time.perf_counter()
    x = audio_io.decode_44k(ref, 0.0, a.max_seconds or None)
    if x.shape[0] < 2:
        return {"ref": ref, "channels": 1}
    x = np.ascontiguousarray(x[:2, : x.shape[1] // HOP * HOP]).astype(np.float32)
    secs = x.shape[1] / SR
    if secs < a.min_seconds:
        return {"ref": ref, "skipped": f"shorter than {a.min_seconds} s"}
    pages = page_side_db(x, SR)
    worst = float(max(pages)) if pages else -np.inf
    row = {"ref": ref, "channels": 2, "seconds": secs, "worst_side_db": worst}
    if worst < a.mono_db:
        row["mono"] = True
        return row
    ms_q = tuple(int(v) for v in a.ms.split(","))

    full = codec.layouts(x, a.lr, ms_q)
    row["full"] = {k: stereo_score(x, y)[0] for k, y in full.items()}
    del full

    win, ctx = int(a.excerpt_s * SR) // HOP * HOP, int(a.context_s * SR) // HOP * HOP
    ex = []
    for s in excerpt_windows(x, a.excerpts, win, ctx):
        c0, c1 = max(0, s - ctx), min(x.shape[1], s + win + ctx)
        ys = codec.layouts(x[:, c0:c1], a.lr, ms_q)
        ref_c = x[:, s:s + win]
        ex.append({"start_s": s / SR, **{k: stereo_score(ref_c, y[:, s - c0:s - c0 + win])[0] for k, y in ys.items()}})
        if time.perf_counter() - t0 > a.track_timeout:
            row["timeout"] = True
            break
    row["excerpts"] = ex
    row["sampled_s"] = len(ex) * (win + 2 * ctx) / SR
    row["took_s"] = time.perf_counter() - t0
    return row


def pick(scores):
    return "lr" if scores["lr"] <= scores["ms"] else "ms"


def sampled_pick(ex, k):
    use = ex[:k]
    return pick({c: np.nanmean([e[c] for e in use]) for c in ("lr", "ms")})


def early_pick(ex, k0, margin):
    """Stop after k0 excerpts when the layouts differ by more than `margin` (relative); else use them all."""
    for k in range(k0, len(ex) + 1):
        m = {c: np.nanmean([e[c] for e in ex[:k]]) for c in ("lr", "ms")}
        if k == len(ex) or abs(m["lr"] - m["ms"]) / min(m.values()) > margin:
            return pick(m), k
    return pick(m), len(ex)


def summary(rows, a):
    st = [r for r in rows if r.get("excerpts")]
    mono = [r for r in rows if r.get("mono")]
    print(f"\n{len(rows)} tracks: {len(mono)} mono by the rule, {len(st)} compared "
          f"({sum(r['seconds'] for r in st) / 60:.0f} min of audio)")
    if not st:
        return
    wins = {c: sum(1 for r in st if pick(r["full"]) == c) for c in ("lr", "ms")}
    print(f"full decision: left/right {wins['lr']}, mid/side {wins['ms']}")
    gap = [abs(r["full"]["lr"] - r["full"]["ms"]) for r in st]
    print(f"full-score gap between the layouts: median {np.median(gap):.3f}, max {max(gap):.3f}")

    def report(label, picks, used):
        agree = [p == pick(r["full"]) for p, r in zip(picks, st)]
        loss = [r["full"][p] - min(r["full"].values()) for p, r in zip(picks, st)]
        frac = np.mean([u * (a.excerpt_s + 2 * a.context_s) / r["seconds"] for u, r in zip(used, st)])
        print(f"  {label:28s} agree {sum(agree):3d}/{len(st)}  mean loss {np.mean(loss):.4f}  "
              f"max loss {max(loss):.3f}  sampled {100 * frac:4.1f}% of the audio")

    print("sampled decision (loss = full score of the pick minus the better layout's):")
    n = min(len(r["excerpts"]) for r in st)
    for k in sorted({1, 2, 3, 4, 6, n} - {0}):
        if k <= n:
            report(f"first {k} excerpts", [sampled_pick(r["excerpts"], k) for r in st], [k] * len(st))
    for k0 in (2, 3):
        for margin in (0.05, 0.10, 0.20):
            res = [early_pick(r["excerpts"], k0, margin) for r in st]
            report(f"early exit k>={k0}, margin {int(margin * 100)}%", [p for p, _ in res], [u for _, u in res])
    bad = [r for r in st if sampled_pick(r["excerpts"], n) != pick(r["full"])]
    if bad:
        print(f"disagreements with all {n} excerpts (full lr / ms):")
        for r in bad:
            print(f"  {r['full']['lr']:.3f} / {r['full']['ms']:.3f}  {os.path.basename(r['ref'])[:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", default=[], help="jsonl manifest(s); rows with 'ref'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20, help="tracks per manifest (random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=300, help="use the first this many seconds (0 = all)")
    ap.add_argument("--min-seconds", type=float, default=30)
    ap.add_argument("--lr", type=int, default=5, help="codebooks per channel for left/right")
    ap.add_argument("--ms", default="7,3", help="mid,side codebooks for mid/side")
    ap.add_argument("--mono-db", type=float, default=-40.0)
    ap.add_argument("--excerpts", type=int, default=8)
    ap.add_argument("--excerpt-s", type=float, default=3.0)
    ap.add_argument("--context-s", type=float, default=1.0, help="extra audio each side of an excerpt")
    ap.add_argument("--chunk-s", type=float, default=30.0, help="encode/decode chunk length")
    ap.add_argument("--encode-context-s", type=float, default=1.0, help="encoder context each side of a chunk")
    ap.add_argument("--ckpt", default="", help="decode with this --arch vocos checkpoint instead of stock DAC")
    ap.add_argument("--raw", action="store_true", help="the checkpoint's raw weights, not the smoothed ones")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--track-timeout", type=float, default=900)
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--summary-only", action="store_true", help="print the summary of --out and exit")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.out, encoding="utf-8")] if os.path.exists(a.out) else []
    if a.summary_only:
        summary(rows, a)
        return
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    maps = data.parse_path_map(a.path_map)
    rng = np.random.default_rng(a.seed)
    refs = []
    for p in a.manifest:
        rr = [r["ref"] for r in corpus.read_jsonl(p)]
        if a.limit and len(rr) > a.limit:
            rr = [rr[i] for i in rng.choice(len(rr), a.limit, replace=False)]
        refs += [data.remap_ref(r, maps) for r in rr]
    done = {r["ref"] for r in rows}
    todo = [r for r in refs if r not in done]
    codec = Codec(a, dev)
    print(f"{len(refs)} tracks ({len(todo)} to do), {codec.name} on {dev}, left/right {a.lr}+{a.lr} "
          f"against mid/side {a.ms.replace(',', '+')}", flush=True)
    t0 = time.perf_counter()
    for i, ref in enumerate(todo):
        try:
            row = check_track(ref, a, codec)
        except Exception as e:  # noqa: BLE001
            row = {"ref": ref, "error": str(e)[:300]}
        rows.append(row)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        what = ("mono" if row.get("mono") else row.get("error", row.get("skipped", ""))[:60] if "full" not in row
                else f"full lr {row['full']['lr']:.3f} ms {row['full']['ms']:.3f}, "
                     f"{len(row['excerpts'])} excerpts")
        print(f"  {i + 1}/{len(todo)} ({time.perf_counter() - t0:.0f} s) {os.path.basename(ref)[:50]}: {what}",
              flush=True)
    summary(rows, a)


if __name__ == "__main__":
    main()
