"""Measure how 'stereo' each track is, page by page, to set the automatic mono flag for HQ (DAC) encoding.

    uv run --extra cpu python stereo_scan.py --manifest data\\pd.jsonl --out stereo_pd.jsonl [--limit 300]

Per page of 40,960 samples at 44.1 kHz (80 DAC frames, about 0.93 s; the native-rate equivalent elsewhere):
  mid = (L+R)/2, side = (L-R)/2; side_db = 10 log10(E[side^2] / E[mid^2]).
  A page is skipped as silent when both channels are below --silence-dbfs. A side below --side-floor-dbfs
  counts as zero (-inf dB): dither and joint-stereo residue are not stereo content.
Per track it records the worst (highest) page side_db, the 99th percentile and the median. (Fitting one gain
R = g L was tried on 1,030 files on 2026-11-21 and turned no extra file mono, so it was removed.)

The proposed rule: mono when the worst non-silent page is below --threshold-db (default -40). The summary
prints the distribution of worst-page levels, so the threshold can sit in the gap between mono and stereo.
"""
import argparse
import json
import math
import multiprocessing as mp
import os

import numpy as np

import audio_io
import corpus
import data

PAGE_S = 80 * 512 / 44100


def db(x):
    return 10 * math.log10(x) if x > 0 else -math.inf


def scan(job):
    ref, a = job
    try:
        x, sr = audio_io.decode(ref, 0.0, a.max_seconds or None)
    except Exception as e:  # noqa: BLE001
        return {"ref": ref, "error": str(e)[:200]}
    if x.ndim == 1 or x.shape[0] == 1:
        return {"ref": ref, "channels": 1}
    x = x[:2].astype(np.float64)
    n = int(round(PAGE_S * sr))
    sil = 10 ** (a.silence_dbfs / 10)
    floor = 10 ** (a.side_floor_dbfs / 10)
    pages = []
    for i in range(0, x.shape[1] - n // 2, n):
        p = x[:, i:i + n]
        e_l, e_r = float(np.mean(p[0] ** 2)), float(np.mean(p[1] ** 2))
        if max(e_l, e_r) < sil:
            continue
        m, s = (p[0] + p[1]) / 2, (p[0] - p[1]) / 2
        e_m, e_s = float(np.mean(m ** 2)), float(np.mean(s ** 2))
        pages.append(-math.inf if e_s < floor or e_m <= 0 else db(e_s / e_m))
    if not pages:
        return {"ref": ref, "channels": 2, "pages": 0}
    v = np.array(pages)

    def q(arr, k):
        fin = arr[np.isfinite(arr)]
        if len(fin) < len(arr) * (1 - k / 100):  # most pages are -inf
            return -math.inf
        return float(np.percentile(np.where(np.isfinite(arr), arr, -200.0), k))

    return {"ref": ref, "channels": 2, "sr": sr, "pages": len(pages),
            "worst_db": float(v.max()), "p99_db": q(v, 99), "median_db": q(v, 50), "seconds": x.shape[1] / sr}


def fmt(v):
    return "-inf" if v == -math.inf else f"{v:+.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", default=[], help="jsonl manifest(s); rows with 'ref'")
    ap.add_argument("--eval-set", action="append", default=[], help="eval JSON(s) (clip refs; whole files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="at most this many tracks per manifest (random)")
    ap.add_argument("--max-seconds", type=float, default=0, help="scan only the start (0 = whole file)")
    ap.add_argument("--silence-dbfs", type=float, default=-60.0)
    ap.add_argument("--side-floor-dbfs", type=float, default=-70.0)
    ap.add_argument("--threshold-db", type=float, default=-40.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    maps = data.parse_path_map(a.path_map)
    refs = []
    rng = np.random.default_rng(a.seed)
    for p in a.manifest:
        rows = [r["ref"] for r in corpus.read_jsonl(p)]
        if a.limit and len(rows) > a.limit:
            rows = [rows[i] for i in rng.choice(len(rows), a.limit, replace=False)]
        refs += rows
    for p in a.eval_set:
        refs += [e["ref"] for e in json.load(open(p, encoding="utf-8"))]
    refs = sorted({data.remap_ref(r, maps) for r in refs})
    print(f"{len(refs)} tracks, {a.workers} workers", flush=True)
    out = []
    with mp.Pool(a.workers) as pool, open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap_unordered(scan, [(ref, a) for ref in refs], chunksize=2)):
            out.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} / {len(refs)}", flush=True)

    st = [r for r in out if r.get("channels") == 2 and r.get("pages")]
    print(f"\n{len(out)} scanned: {sum(1 for r in out if r.get('channels') == 1)} already mono files, "
          f"{sum(1 for r in out if 'error' in r)} errors, {len(st)} stereo files")
    edges = [-math.inf, -80, -60, -50, -40, -35, -30, -25, -20, -15, -10, -5, 0, math.inf]
    print("worst-page side level (dB below mid) -> tracks")
    for lo, hi in zip(edges, edges[1:]):
        k = sum(1 for r in st if lo <= r["worst_db"] < hi) if lo > -math.inf else \
            sum(1 for r in st if r["worst_db"] == -math.inf or r["worst_db"] < hi)
        print(f"  {fmt(lo):>6} .. {fmt(hi):>6}: {k:5d}")
    mono = [r for r in st if r["worst_db"] < a.threshold_db]
    print(f"\nrule (worst page < {a.threshold_db:.0f} dB): {len(mono)} of {len(st)} stereo files are mono")
    near = sorted(st, key=lambda r: abs(r["worst_db"] - a.threshold_db))[:12]
    print("closest to the threshold (worst / p99 / median dB):")
    for r in near:
        print(f"  {fmt(r['worst_db']):>6} {fmt(r['p99_db']):>6} {fmt(r['median_db']):>6}  {os.path.basename(r['ref'])[:70]}")


if __name__ == "__main__":
    main()
