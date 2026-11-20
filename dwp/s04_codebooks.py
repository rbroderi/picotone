"""Codebook statistics and direct edits for better quality per bit.

usage: python s04_codebooks.py 24khz|48khz

A. usage statistics / near-duplicates / per-stage gain          (inspection only)
B. later codebooks ablated: the stock bitrate ladder             (ablation)
C. decoder-side corrections for a truncated code stream, SAME bitstream:
     C1 [LS] decode tables re-estimated per n_q (conditional means, exact coordinate-wise least squares)
     C2 [LS] linear FIR on the dequantised latent predicting the missing residual (folds into decoder conv 0)
D. rarely-used entries removed (smaller codebooks, more stages at the same bitrate): NEW FORMAT VARIANT for
   fixed-rate packing, although every stream stays decodable by the stock decoder tables.
Lossless entropy coding of the codes is deliberately not touched (other agent).
"""
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from common import OUT, Bench, Log, perceptual, snr_db

name = sys.argv[1]
log = Log(f"s04_{name}")
test = Bench(name)
fit = Bench(name, "fit")
m = test.model
NQ = m.quantizer.num_quantizers
frame_rate = 75 if name == "24khz" else 150
causal = name == "24khz"
E = [l.codebook.embed.clone() for l in m.quantizer.layers]              # stock tables [1024,128]
csize = [l.codebook.cluster_size.clone() for l in m.quantizer.layers]   # EMA usage counts left over from training

zf = fit.emb.permute(0, 2, 1).reshape(-1, 128)        # [N,128] encoder output before quantisation
zt = test.emb.permute(0, 2, 1).reshape(-1, 128)
cf = fit.codes.reshape(NQ, -1)                        # [NQ, N]
ct = test.codes.reshape(NQ, -1)
Bt, _, Tt = test.emb.shape
print(f"{name}: fit frames {len(zf)}, test frames {len(zt)}")


def audio(zhat):
    """Decode a test latent [N,128] with the stock decoder and score against the source audio."""
    lat = zhat.reshape(Bt, Tt, 128).permute(0, 2, 1).contiguous()
    y = test.decode_from_pre(m.decoder.layers[0](lat))
    return perceptual(test.src, test._wave_rows(y), test.sr)


def entropy_bits(counts):
    p = counts[counts > 0].double()
    p = p / p.sum()
    return float(-(p * p.log2()).sum())


# ============================================================== A. statistics
print("\n== A. per-codebook statistics")
print("cb | train-usage entropy (bits, from stored cluster_size) | dead in training (<1e-3 of mean) | fit-set entropy | used on fit | entry RMS norm | nearest-neighbour dist / RMS norm (min, median) | latent SNR after this stage (test)")
res = zt.clone()
all_c = torch.cat([cf, ct], 1)
for k in range(NQ):
    cs = csize[k]
    cnt = torch.bincount(all_c[k], minlength=1024).float()
    nrm = E[k].pow(2).sum(1).sqrt()
    d = torch.cdist(E[k], E[k]) + 1e9 * torch.eye(1024)
    nn = d.min(1).values / nrm.pow(2).mean().sqrt()
    res = res - E[k][ct[k]]
    lat_snr = 10 * torch.log10(zt.pow(2).sum() / res.pow(2).sum())
    if k < 8 or k % 4 == 3:
        print(f"{k:2d} | {entropy_bits(cs):.2f} | {int((cs < 1e-3 * cs.mean()).sum()):3d} | {entropy_bits(cnt):.2f} | {int((cnt > 0).sum()):4d} | {float(nrm.pow(2).mean().sqrt()):.3f} | {float(nn.min()):.3f} {float(nn.median()):.3f} | {float(lat_snr):.2f} dB")
    log(f"A codebook {k}", None, train_entropy_bits=entropy_bits(cs), fit_entropy_bits=entropy_bits(cnt), used=int((cnt > 0).sum()),
        rms_norm=float(nrm.pow(2).mean().sqrt()), nn_min=float(nn.min()), nn_median=float(nn.median()), latent_snr_db=float(lat_snr))

# cross-stage dependence visible to a decoder: how much of stage k's vector is predictable from the stage-1 code?
print("\nmean-vector predictability: energy of E[q_k | c_1] relative to energy of q_k (fit set; 0 = independent)")
line = []
for k in range(1, min(NQ, 8)):
    qk = E[k][cf[k]]
    sums = torch.zeros(1024, 128).index_add_(0, cf[0], qk)
    n = torch.bincount(cf[0], minlength=1024).float().clamp_min(1)
    cm = (sums / n[:, None])[cf[0]]
    line.append(f"k={k}: {float(cm.pow(2).sum() / qk.pow(2).sum()):.3f}")
print("  ".join(line), " (note: values near 1024/frames-per-entry are just estimation noise)")

# ============================================================== B. stock ladder
print("\n== B. stock bitrate ladder (later codebooks removed)")
stock_q = {}
ladder = [k for k in (1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 32) if k <= NQ]
for k in ladder:
    q = sum(E[i][ct[i]] for i in range(k))
    stock_q[k] = q
    r = audio(q)
    r["latent_snr"] = snr_db(zt, q)
    log(f"B stock n_q={k}", r, kbps=k * 10 * frame_rate / 1000, latent_snr=r["latent_snr"])
    stock_q[k] = (q, r)

# ============================================================== C. decoder-side corrections
print("\n== C. decoder-side corrections for truncated streams (bitstream unchanged)")


def refit_tables(k, lam=4.0, sweeps=6):
    """Exact coordinate-wise ridge least squares of  z ~ sum_i T_i[c_i]  (ridge pulls every entry to its stock value)."""
    T = [E[i].clone() for i in range(k)]
    parts = [T[i][cf[i]] for i in range(k)]
    total = sum(parts)
    for _ in range(sweeps):
        for i in range(k):
            r = zf - (total - parts[i])
            sums = torch.zeros(1024, 128).index_add_(0, cf[i], r)
            n = torch.bincount(cf[i], minlength=1024).float()
            T[i] = (sums + lam * E[i]) / (n + lam)[:, None]
            new = T[i][cf[i]]
            total = total - parts[i] + new
            parts[i] = new
    return T


def taps_of(q, B, T_, n_taps):
    """Stack the latent at offsets (causal: t, t-1, ...; else centred) -> [N, 128*n_taps]."""
    lat = q.reshape(B, T_, 128)
    offs = list(range(n_taps)) if causal else [o - n_taps // 2 for o in range(n_taps)]
    cols = []
    for o in offs:
        if o >= 0:
            cols.append(F.pad(lat, (0, 0, o, 0))[:, :T_])
        else:
            cols.append(F.pad(lat, (0, 0, 0, -o))[:, -o:])
    return torch.cat(cols, -1).reshape(-1, 128 * n_taps), offs


def solve_fir(qf, n_taps, ridge=1e-3):
    Bf, Tf = fit.emb.shape[0], fit.emb.shape[2]
    X, offs = taps_of(qf, Bf, Tf, n_taps)
    X = X.double()
    Y = (zf - qf).double()
    mx, my = X.mean(0), Y.mean(0)
    G = (X - mx).T @ (X - mx)
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype)
    W = torch.linalg.solve(G, (X - mx).T @ (Y - my)).T
    return W.float(), (my - mx @ W.T).float(), offs


saved = {}
for k in [k for k in (1, 2, 4, 8, 16) if k < NQ]:
    q0, r0 = stock_q[k]
    base = f"C n_q={k}"
    # C1 tables
    T = refit_tables(k)
    q1 = sum(T[i][ct[i]] for i in range(k))
    r = audio(q1)
    log(f"{base} [LS] C1 re-estimated decode tables", r, latent_snr=snr_db(zt, q1), latent_gain_db=snr_db(zt, q1) - r0["latent_snr"],
        d_snr=r["snr"] - r0["snr"], d_logmel=r["logmel"] - r0["logmel"], d_mrstft=r["mrstft"] - r0["mrstft"])
    # C2 FIR on the stock dequantised latent
    qf0 = sum(E[i][cf[i]] for i in range(k))
    for n_taps in (1, 5):
        W, b_, offs = solve_fir(qf0, n_taps)
        X, _ = taps_of(q0, Bt, Tt, n_taps)
        q2 = q0 + X @ W.T + b_
        r = audio(q2)
        log(f"{base} [LS] C2 residual predicted by {n_taps}-tap linear FIR of the latent", r, latent_snr=snr_db(zt, q2), latent_gain_db=snr_db(zt, q2) - r0["latent_snr"],
            d_snr=r["snr"] - r0["snr"], d_logmel=r["logmel"] - r0["logmel"], d_mrstft=r["mrstft"] - r0["mrstft"])
        if n_taps == 5:
            saved[f"nq{k}.fir.weight"] = W.reshape(128, n_taps, 128).permute(0, 2, 1).contiguous()  # [out, in, tap]; tap j = offset offs[j]
            saved[f"nq{k}.fir.bias"] = b_
    # C1 + C2
    qf1 = sum(T[i][cf[i]] for i in range(k))
    W, b_, offs = solve_fir(qf1, 5)
    X, _ = taps_of(q1, Bt, Tt, 5)
    q3 = q1 + X @ W.T + b_
    r = audio(q3)
    log(f"{base} [LS] C1+C2 tables then 5-tap FIR", r, latent_snr=snr_db(zt, q3), latent_gain_db=snr_db(zt, q3) - r0["latent_snr"],
        d_snr=r["snr"] - r0["snr"], d_logmel=r["logmel"] - r0["logmel"], d_mrstft=r["mrstft"] - r0["mrstft"])
    # half-way shrink: does a milder correction keep the texture better?
    q4 = q0 + 0.5 * (q3 - q0)
    r = audio(q4)
    log(f"{base} [LS] C1+C2 applied at half strength", r, latent_snr=snr_db(zt, q4), latent_gain_db=snr_db(zt, q4) - r0["latent_snr"],
        d_snr=r["snr"] - r0["snr"], d_logmel=r["logmel"] - r0["logmel"], d_mrstft=r["mrstft"] - r0["mrstft"])
    if k <= 8:
        for i in range(k):
            saved[f"nq{k}.table{i}"] = T[i]
save_file(saved, os.path.join(OUT, f"decode_corrections_{name}.safetensors"))

# ============================================================== D. smaller codebooks (rare entries removed)
print("\n== D. rare entries removed: M-entry codebooks, more stages at the same fixed bitrate")


def rvq_restricted(z, k, M):
    """Greedy RVQ using only the M most-used entries (by training cluster_size) of each stock codebook."""
    res = z.clone()
    q = torch.zeros_like(z)
    for i in range(k):
        keep = csize[i].argsort(descending=True)[:M]
        Ei = E[i][keep]
        idx = torch.cdist(res, Ei).argmin(1)
        q = q + Ei[idx]
        res = res - Ei[idx]
    return q


for M, bits, pairs in ((512, 9, [(10, 9), (20, 18)]), (256, 8, [(5, 4), (10, 8), (20, 16)]), (32, 5, [(8, 4), (16, 8)])):
    for k_small, k_stock in pairs:
        if k_small > NQ or k_stock > NQ:
            continue
        q = rvq_restricted(zt, k_small, M)
        r = audio(q)
        if k_stock in stock_q:
            q0, r0 = stock_q[k_stock]
        else:
            q0 = sum(E[i][ct[i]] for i in range(k_stock))
            r0 = audio(q0)
            r0["latent_snr"] = snr_db(zt, q0)
        log(f"D [W] {k_small} codebooks x {M} entries ({bits} bit) vs stock {k_stock} x 1024 at {k_stock * 10 * frame_rate / 1000:g} kbps", r,
            latent_snr=snr_db(zt, q), latent_gain_db=snr_db(zt, q) - r0["latent_snr"],
            d_snr=r["snr"] - r0["snr"], d_logmel=r["logmel"] - r0["logmel"], d_mrstft=r["mrstft"] - r0["mrstft"])
