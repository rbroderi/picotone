"""Builds the cx1 model tables by counting. No gradient descent.

Usage: build_tables.py TAG OUT.ecxm [--budget N] [--fitall] [--no-static] [--samples N]

Pipeline (fit = fit clips except every 8th, dev = every 8th fit clip, test never touched):
  1. `picotone dev stats` on fit     -> bucket histograms of each feature (true code vs all codes)
  2. tables = shrunk log-likelihood ratios of those histograms; bias = log prior;
     static sparse pair tables = log(1 + n(ctx, code) / (m0 * p(code))) from fit pair counts
  3. `picotone dev tune` on dev    -> one weight per expert and codebook (grid search, 11 values)
  4. tables are multiplied by their weights and written out; with --fitall the counts of
     steps 1-2 are redone on fit + dev first (weights kept).
"""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import torch

import lib

sys.path.insert(0, os.path.join(lib.ROOT, "train"))
import export_net as netexport  # noqa: E402

V = 1024
FB = 6
FLOOR = 64
GEO, REC, PAIR = 8 * 34, 29 * 16, 96
M0 = 8.0
SHRINK = 5.0
PICOTONE = lib.PICOTONE
EXPERTS = ["geo", "rec", "pair0", "pair1", "pair2", "stat0", "stat1", "stat2", "hist0", "hist1", "net"]


def weights_path(tag):
    return os.path.join(lib.WEIGHTS, lib.MODEL_NAME[tag] + ".safetensors")


def write_list(name, files):
    os.makedirs(os.path.join(lib.WORK, "lists"), exist_ok=True)
    path = os.path.join(lib.WORK, "lists", name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for item in files:
            f.write(lib.bundle(item["path"]) + "\n")
    return path


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("command failed: " + " ".join(args) + "\n" + r.stderr[-2000:])
    return r.stderr


def llr(counts):
    """Shrunk log2 likelihood ratio per bucket, in logit units (1/64 bit)."""
    hit = np.asarray(counts["true"], dtype=np.float64)
    al = np.asarray(counts["all"], dtype=np.float64)
    ratio = al / max(al.sum(), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.log2((hit + SHRINK * ratio) / (hit.sum() + SHRINK)) - np.log2(ratio)
    t = np.where(al > 0, t, 0.0)
    return np.clip(t, -16, 16) * (1 << FB)


def context_codes(codes, start, k):
    """Mirrors `State::contexts` in features.rs. Returns [3] arrays of context codes."""
    prev = [lib.prev_frame(codes[:, i].copy(), start) for i in range(min(codes.shape[1], 2))]
    if k == 0:
        return [prev[0], prev[1], lib.prev_frame(prev[0], start)]
    if k == 1:
        return [prev[1], codes[:, 0], prev[0]]
    return [lib.prev_frame(codes[:, k].copy(), start), codes[:, k - 1], codes[:, 0]]


def static_entries(ctx, y, prior, cap):
    """Sparse pair table: (ctx, code, value in logit units, count). Singletons dropped."""
    tab = np.bincount(ctx * V + y, minlength=V * V)
    idx = np.nonzero(tab >= 2)[0]
    n = tab[idx].astype(np.float64)
    c, j = idx // V, idx % V
    val = np.log2(1.0 + n / (M0 * prior[j])) * (1 << FB)
    score = n * val
    if len(idx) > cap:
        keep = np.argsort(-score)[:cap]
        c, j, val, score = c[keep], j[keep], val[keep], score[keep]
    return c, j, val, score


HIST_LAGS = 40
HIST_DECAY = 7.0 / 8.0          # must match HISTORY_DECAY in model.rs
HIST_PRESCALE = 0.5             # unit-weight tables carry half the PMI sum


def hist_rank(k):
    return 64 if k < 2 else (32 if k < 8 else 8)


def history_factors(src, y, start, prior, first_lag, rank):
    """Low-rank factors of the decayed co-occurrence PMI between recent `src` codes and `y`.

    Returns the float factors: embedding A [V, r] and output B [V, r], with
    sum over lags of decay^lag * (A[code] . B[j]) approximating the summed PMI in nats.
    """
    since = np.arange(len(y)) - np.maximum.accumulate(np.where(start, np.arange(len(y)), 0))
    n = np.zeros(V * V)
    for i in range(HIST_LAGS):
        lag = first_lag + i
        if lag == 0:
            c, yy = src, y
        else:
            ok = since[lag:] >= lag
            c, yy = src[:-lag][ok], y[lag:][ok]
        n += HIST_DECAY ** i * np.bincount(c * V + yy, minlength=V * V)
    n = n.reshape(V, V)
    rows = n.sum(1, keepdims=True)
    pmi = np.log((n + M0 * prior[None, :]) / (rows + M0)) - np.log(prior)[None, :]
    wt = np.sqrt((rows[:, 0] + 1.0) / (rows.sum() + V))
    u, s, vt = torch.linalg.svd(torch.from_numpy((pmi * wt[:, None]).astype(np.float32)),
                                full_matrices=False)
    root = s[:rank].sqrt()
    a = ((u[:, :rank] * root).numpy() / wt[:, None])
    b = (vt[:rank].T * root).numpy()
    return a, b


def history_ints(a, b, weight):
    """Integer tables of a history expert at the given weight: (emb i16, out i8, shift).

    logit = (out . decayed sum of emb) >> shift, in units of 2^-FB bits.
    """
    unit = (1 << FB) / np.log(2.0) * HIST_PRESCALE * weight
    amax = np.maximum(np.abs(a).max(0), 1e-9)
    bmax = np.maximum(np.abs(b).max(0), 1e-9)
    beta = 127.0 / bmax
    # emb = A * unit * 2^shift / beta must stay within 4095
    room = (4095.0 * beta / (unit * amax)).min()
    shift = int(np.clip(np.floor(np.log2(room)), 0, 20))
    alpha = unit * (2.0 ** shift) / beta
    emb = np.clip(np.rint(a * alpha), -4095, 4095).astype("<i2")
    out = np.clip(np.rint(b * beta), -127, 127).astype(np.int8)
    return emb, out, shift


def build_tables(tag, files, hist, cap, use_static, use_hist=True):
    codes, start = lib.concat(files)
    layers = []
    for k in range(lib.KMAX[tag]):
        y = codes[:, k]
        cnt = np.bincount(y, minlength=V).astype(np.float64)
        prior = (cnt + 0.5) / (cnt.sum() + 0.5 * V)
        layer = {"bias": np.log2(prior) * (1 << FB),
                 "geo": llr(hist[k]["geo"]), "rec": llr(hist[k]["rec"]),
                 "pair": [llr(p) for p in hist[k]["pair"]], "stat": [], "hist": []}
        if use_hist:
            # slot 0: this codebook's own past; slot 1: codebook 0 up to the current frame
            # (for codebook 0: codebook 1's past). Mirrors `Predictor::update` in model.rs.
            layer["hist"].append(history_factors(y, y, start, prior, 1, hist_rank(k)))
            if k == 0:
                layer["hist"].append(history_factors(codes[:, 1], y, start, prior, 1, hist_rank(k)))
            else:
                layer["hist"].append(history_factors(codes[:, 0], y, start, prior, 0, hist_rank(k)))
        for ctx in context_codes(codes, start, k):
            if use_static:
                layer["stat"].append(static_entries(ctx, y, prior, cap))
            else:
                e = np.zeros(0, dtype=np.int64)
                layer["stat"].append((e, e, np.zeros(0), np.zeros(0)))
        layers.append(layer)
    return layers


def fixed_centroids(tag, codebooks):
    """The coder's fixed-point centroids: round(x * 128) half away from zero, clamped to
    +-4095, exactly as `to_fixed` in features.rs (the Rust loader checks this against
    the weights whenever it is given them)."""
    v = lib.load_codebooks(tag)[:codebooks].astype(np.float64) * 128.0
    v = np.sign(v) * np.floor(np.abs(v) + 0.5)
    return np.clip(v, -4095, 4095).astype("<i2")


def serialise(layers, weights=None, budget=None, net=None, version=5, centroids=None):
    """Writes the table file; `weights[k][e]` are sixteenths (None = unit weights).

    Version 5 stores `centroids` (fixed point, [codebook][code][dim]) after the
    exponential table and writes the net in format 2; version 4 is the older layout.
    """
    exp = [int(round((1 << 24) * 2.0 ** (-i / (1 << FB)))) for i in range(1 << FB)]
    out = [b"ECXM", struct.pack("<4I", version, len(layers), FB, FLOOR), struct.pack(f"<{len(exp)}I", *exp)]
    if version == 5:
        assert centroids is not None and centroids.shape == (len(layers), V, 128)
        out.append(centroids.tobytes())
    unit = [[16] * len(EXPERTS) for _ in layers]
    weights = weights or unit
    # global budget for static entries: keep the highest weighted scores
    threshold = 0.0
    if budget is not None:
        scores = np.concatenate([layer["stat"][s][3] * weights[k][5 + s] / 16.0
                                 for k, layer in enumerate(layers) for s in range(3)] + [np.zeros(1)])
        scores = scores[scores > 0]
        if len(scores) > budget:
            threshold = np.sort(scores)[-budget]
    kept = 0
    for k, layer in enumerate(layers):
        w = [x / 16.0 for x in weights[k]]
        out.append(np.rint(layer["bias"]).astype("<i4").tobytes())
        out.append(np.rint(layer["geo"] * w[0]).astype("<i4").tobytes())
        out.append(np.rint(layer["rec"] * w[1]).astype("<i4").tobytes())
        for s in range(3):
            out.append(np.rint(layer["pair"][s] * w[2 + s]).astype("<i4").tobytes())
        for s in range(3):
            c, j, val, score = layer["stat"][s]
            v = np.rint(val * w[5 + s])
            keep = (v != 0) & (score * w[5 + s] >= threshold)
            c, j, v = c[keep], j[keep], np.clip(v[keep], -32767, 32767)
            order = np.lexsort((j, c))
            c, j, v = c[order], j[order], v[order]
            offsets = np.searchsorted(c, np.arange(V + 1)).astype("<u4")
            out += [offsets.tobytes(), j.astype("<u2").tobytes(), v.astype("<i2").tobytes()]
            kept += len(c)
        for s in range(2):
            if s < len(layer["hist"]) and w[8 + s] > 0:
                emb, mat, shift = history_ints(*layer["hist"][s], w[8 + s])
                out += [struct.pack("<2I", emb.shape[1], shift), emb.tobytes(), mat.tobytes()]
            else:
                out.append(struct.pack("<2I", 0, 0))
    if net:
        out.append(struct.pack("<I", 1))
        out.append(netexport.blob(net, None, [w[10] for w in weights], 2 if version == 5 else 1))
    else:
        out.append(struct.pack("<I", 0))
    return b"".join(out), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--budget", type=int, default=100000, help="static entries kept in total")
    ap.add_argument("--cap", type=int, default=40000, help="static entries per table before tuning")
    ap.add_argument("--samples", type=int, default=2000, help="dev frames sampled for tuning")
    ap.add_argument("--geo", type=int, default=4, help="codebooks that get the geo expert")
    ap.add_argument("--fitall", action="store_true")
    ap.add_argument("--net", default=None, help="trained net checkpoint (netfit.py)")
    ap.add_argument("--reset", type=int, default=0, help="reset interval in seconds (0 = never)")
    ap.add_argument("--no-hist", action="store_true")
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--table-version", type=int, default=5, choices=[4, 5],
                    help="5: self-contained table (centroids inside, net format 2); "
                         "4: the older layout, for nets without a position input")
    args = ap.parse_args()
    tag = args.tag
    stem = os.path.splitext(os.path.basename(args.out))[0]
    wpath = weights_path(tag)
    reset = args.reset * lib.FRAME_RATE[tag]
    # The net's bandwidths come from its checkpoint (old checkpoints: 2, 4, 8, 16).
    net = args.net
    version = args.table_version
    centroids = fixed_centroids(tag, lib.KMAX[tag]) if version == 5 else None
    if net and version == 4:
        assert not netexport.checkpoint_period(netexport.load_state(net)), \
            "this net has a position input; it needs --table-version 5"

    def serialise_as(*a, **kw):
        return serialise(*a, **kw, version=version, centroids=centroids)
    fit, dev = lib.load_files(tag, "fit"), lib.load_files(tag, "dev")
    print(tag, "fit clips", len(fit), "dev clips", len(dev), flush=True)

    def stats(name, files):
        path = os.path.join(lib.WORK, f"hist_{stem}_{name}.json")
        run([PICOTONE, "dev", "stats", wpath, path, str(args.geo), str(reset), "@" + write_list(f"{stem}_{name}.txt", files)])
        return json.load(open(path))

    layers = build_tables(tag, fit, stats("fit", fit), args.cap, not args.no_static, not args.no_hist)
    unit_path = os.path.join(lib.WORK, f"{stem}_unit.ecxm")
    blob, kept = serialise_as(layers, net=net)
    open(unit_path, "wb").write(blob)
    print("unit model", len(blob), "bytes,", kept, "static entries", flush=True)
    tuned_path = os.path.join(lib.WORK, f"tuned_{stem}.json")
    run([PICOTONE, "dev", "tune", unit_path, wpath, tuned_path, str(args.samples), str(reset),
         "@" + write_list(f"{stem}_dev.txt", dev)])
    tuned = json.load(open(tuned_path))
    for k, t in enumerate(tuned):
        print(k, dict(zip(EXPERTS, t["weights"])), "dev bits prior %.3f unit %.3f tuned %.3f"
              % (t["bits_prior"], t["bits_unit"], t["bits_tuned"]), flush=True)
    weights = [t["weights"] for t in tuned]
    if args.fitall:
        everything = lib.load_files(tag, "fitall")
        layers = build_tables(tag, everything, stats("fitall", everything), args.cap,
                              not args.no_static, not args.no_hist)
    blob, kept = serialise_as(layers, weights, args.budget, net=net)
    open(args.out, "wb").write(blob)
    print("wrote", args.out, len(blob), "bytes,", kept, "static entries", flush=True)
    for kk in (2, 4, 8, 16, 32):
        if kk <= len(tuned):
            print("K=%d dev bits/code: prior %.3f tuned %.3f" % (
                kk, np.mean([t["bits_prior"] for t in tuned[:kk]]),
                np.mean([t["bits_tuned"] for t in tuned[:kk]])))


if __name__ == "__main__":
    main()
