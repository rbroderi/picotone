"""Interpretation of the encoder- and decoder-side LSTMs: weights, biases, gate statistics on real audio.

Pure inspection (no solving) except the 'linear predictability' section, which uses least squares
as a measuring instrument.

usage: python s01_lstm_interp.py 24khz|48khz
Writes cache/lstm_stats_<name>.pt for later scripts.
"""
import os
import sys

import numpy as np
import torch

from common import CACHE, Bench, get_lstm, lstm_layer

name = sys.argv[1]
b = Bench(name)
torch.manual_seed(0)


def q(x, ps=(0.01, 0.1, 0.5, 0.9, 0.99)):
    x = x.flatten().double()
    return " ".join(f"{float(torch.quantile(x, p)):.3f}" for p in ps)


def pca_dims(x, levels=(0.9, 0.99, 0.999)):
    x = x - x.mean(0)
    s = torch.linalg.svdvals(x.double())
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return [int((e < l).sum()) + 1 for l in levels]


def r2_linear(X, Y, ridge=1e-3):
    """Fraction of variance of Y explained by an affine map of X (fit on even rows, scored on odd rows)."""
    X = torch.cat([X, torch.ones(len(X), 1)], 1).double()
    Y = Y.double()
    A, B = X[::2], Y[::2]
    G = A.T @ A
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype)
    W = torch.linalg.solve(G, A.T @ B)
    P = X[1::2] @ W
    T = Y[1::2]
    res = (T - P).pow(2).sum(0)
    var = (T - T.mean(0)).pow(2).sum(0).clamp_min(1e-12)
    return 1 - res / var, 1 - res.sum() / var.sum()


stats = {}
for side in ("enc", "dec"):
    idx, ps = get_lstm(b.model, side)
    x = (b.enc_pre if side == "enc" else b.dec_pre).permute(2, 0, 1).contiguous()  # [T, B, 512]
    T, B, D = x.shape
    print(f"\n================ {name} {side}oder LSTM: input [T={T}, B={B}, D={D}]")
    print(f"input x: mean {float(x.mean()):.3f} std {float(x.std()):.3f} per-channel std quantiles {q(x.std((0, 1)))}  PCA dims 90/99/99.9% {pca_dims(x.reshape(-1, D)[::3])}")
    inp = x
    recs = []
    for l, p in enumerate(ps):
        H = p["w_hh"].shape[1]
        h, rec = lstm_layer(inp, p["w_ih"], p["w_hh"], p["b"], record=True)
        recs.append((h, rec))
        print(f"\n--- layer {l}")
        # weights
        for gi, g in enumerate("ifgo"):
            wi = p["w_ih"][gi * H:(gi + 1) * H]
            wh = p["w_hh"][gi * H:(gi + 1) * H]
            bb = p["b"][gi * H:(gi + 1) * H]
            diag = wh.diagonal()
            print(f"gate {g}: bias mean {float(bb.mean()):+.3f} std {float(bb.std()):.3f} q[{q(bb)}] | row-norm W_ih {float(wi.norm(dim=1).mean()):.2f} W_hh {float(wh.norm(dim=1).mean()):.2f} | W_hh diag mean {float(diag.mean()):+.3f} (|offdiag| mean {float((wh.abs().sum() - diag.abs().sum()) / (H * H - H)):.4f})")
        # how much of each gate pre-activation comes from the input vs the recurrence
        pre_x = inp @ p["w_ih"].T
        hprev = torch.cat([torch.zeros(1, B, H), h[:-1]])
        pre_h = hprev @ p["w_hh"].T
        for gi, g in enumerate("ifgo"):
            vx = pre_x[..., gi * H:(gi + 1) * H].var((0, 1)).mean()
            vh = pre_h[..., gi * H:(gi + 1) * H].var((0, 1)).mean()
            print(f"gate {g}: pre-activation variance from input {float(vx):.3f}, from recurrence {float(vh):.3f}  (recurrent share {100 * float(vh / (vx + vh)):.0f}%)")
        # gate statistics
        for g in "ifo":
            v = rec[g]
            m_unit = v.mean((0, 1))
            s_unit = v.std((0, 1))
            print(f"gate {g}: mean {float(v.mean()):.3f}  frac>0.9 {100 * float((v > 0.9).float().mean()):.1f}%  frac<0.1 {100 * float((v < 0.1).float().mean()):.1f}%  frac in[0.1,0.9] {100 * float(((v >= 0.1) & (v <= 0.9)).float().mean()):.1f}% | per-unit mean q[{q(m_unit)}] per-unit std q[{q(s_unit)}] | units with std<0.05: {int((s_unit < 0.05).sum())}, <0.1: {int((s_unit < 0.1).sum())}")
        gg = rec["g"]
        print(f"cell input g=tanh: |g|>0.9 {100 * float((gg.abs() > 0.9).float().mean()):.1f}%  |g|<0.5 {100 * float((gg.abs() < 0.5).float().mean()):.1f}%")
        c = rec["c"]
        print(f"cell c: std {float(c.std()):.3f} |c| q[{q(c.abs())}]  |c|>2 {100 * float((c.abs() > 2).float().mean()):.1f}%  |c|<0.5 (tanh ~linear) {100 * float((c.abs() < 0.5).float().mean()):.1f}%")
        fm = rec["f"].mean((0, 1)).clamp(1e-6, 1 - 1e-6)
        tau = -1 / torch.log(fm)
        print(f"memory: per-unit time constant tau=-1/ln(mean f) frames q[{q(tau)}]; units tau>10: {int((tau > 10).sum())}, >50: {int((tau > 50).sum())}, >200: {int((tau > 200).sum())}; tau<1: {int((tau < 1).sum())}")
        # leaky-integrator check: how well is c_t = a*c_{t-1} + b*g_t (constant gates) per unit
        cprev = torch.cat([torch.zeros(1, B, H), c[:-1]])
        const = fm * cprev + rec["i"].mean((0, 1)) * gg
        err = (c - const).pow(2).mean((0, 1)) / c.var((0, 1)).clamp_min(1e-9)
        print(f"constant-gate one-step model c_t=f_mean*c_(t-1)+i_mean*g_t: relative error per unit q[{q(err)}]; units <1% err: {int((err < 0.01).sum())}, <10%: {int((err < 0.1).sum())}")
        hs = h.std((0, 1))
        print(f"hidden h: std overall {float(h.std()):.3f} per-unit std q[{q(hs)}]; units std<0.01: {int((hs < 0.01).sum())}, <0.05: {int((hs < 0.05).sum())}; |h|>0.9 {100 * float((h.abs() > 0.9).float().mean()):.1f}%")
        flat = h.reshape(-1, H)[::3]
        print(f"hidden h PCA dims for 90/99/99.9% variance: {pca_dims(flat)};  cell c PCA: {pca_dims(c.reshape(-1, H)[::3])}")
        # linear predictability of h from the *current* input frame, and from a 5-frame causal window
        X0 = inp.reshape(-1, D)
        r_unit, r_all = r2_linear(X0, h.reshape(-1, H))
        win = torch.cat([torch.cat([torch.zeros(k, B, D), inp[:T - k]]) for k in range(5)], -1).reshape(-1, 5 * D)
        r5_unit, r5_all = r2_linear(win[::2], h.reshape(-1, H)[::2])
        print(f"linear predictability of h (held-out R^2): from x_t {float(r_all):.3f} (units R2>0.9: {int((r_unit > 0.9).sum())}, >0.5: {int((r_unit > 0.5).sum())}); from x_(t-4..t) {float(r5_all):.3f} (units >0.9: {int((r5_unit > 0.9).sum())}, >0.5: {int((r5_unit > 0.5).sum())})")
        stats[f"{side}{l}"] = {
            "h_std": hs, "h_mean": h.mean((0, 1)), "tau": tau, "f_mean": fm, "f_std": rec["f"].std((0, 1)),
            "i_mean": rec["i"].mean((0, 1)), "i_std": rec["i"].std((0, 1)), "o_mean": rec["o"].mean((0, 1)), "o_std": rec["o"].std((0, 1)),
            "r2_x": r_unit, "r2_x5": r5_unit, "c_std": c.std((0, 1)),
        }
        inp = h
    # the block as a whole: y = x + h2
    h2 = recs[-1][0]
    print(f"\n--- whole block y = x + LSTM(x)")
    print(f"energy: ||x||^2 {float(x.pow(2).mean()):.3f}, ||h2||^2 {float(h2.pow(2).mean()):.3f}, ratio h2/x {float(h2.pow(2).mean() / x.pow(2).mean()):.3f}; cos(x,h2) {float((x * h2).sum() / (x.norm() * h2.norm())):+.3f}")
    r_unit, r_all = r2_linear(x.reshape(-1, D), (x + h2).reshape(-1, D))
    print(f"y predicted by affine map of x_t: held-out R^2 {float(r_all):.4f}  (skip alone: R^2 {float(1 - h2.pow(2).sum() / ((x + h2) - (x + h2).mean((0, 1))).pow(2).sum()):.4f})")
    # memory length: restart the LSTM (zero state) at frame t0 and see how fast the output converges to the full-context one
    t0 = T // 3
    inp2 = x[t0:]
    for p in ps:
        inp2 = lstm_layer(inp2, p["w_ih"], p["w_hh"], p["b"])
    ref = h2[t0:]
    line = []
    for k in (0, 1, 2, 5, 10, 20, 50, 100, 200, 400):
        if k + 5 < len(ref):
            seg = slice(k, min(len(ref), k + max(5, k // 4)))
            snr = 10 * torch.log10(ref[seg].pow(2).sum() / (ref[seg] - inp2[seg]).pow(2).sum().clamp_min(1e-20))
            line.append(f"{k}:{float(snr):.1f}dB")
    print("state-reset convergence (SNR of LSTM output vs full-context, by frames since reset): " + "  ".join(line))

torch.save(stats, os.path.join(CACHE, f"lstm_stats_{name}.pt"))
