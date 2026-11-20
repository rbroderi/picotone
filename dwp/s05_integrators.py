"""What do the long-memory LSTM units track?  Pure inspection.

usage: python s05_integrators.py 24khz|48khz
For the units with the largest forget-gate time constants: is the cell a clock (correlates with the frame
index), a loudness integrator (correlates with accumulated input energy), or a latch?
Also classifies every unit by regime.
"""
import sys

import torch

from common import Bench, get_lstm, lstm_layer

name = sys.argv[1]
b = Bench(name)


def corr(a, c):
    a = a - a.mean()
    c = c - c.mean()
    return float((a * c).sum() / (a.norm() * c.norm()).clamp_min(1e-12))


for side in ("enc", "dec"):
    idx, ps = get_lstm(b.model, side)
    x = (b.enc_pre if side == "enc" else b.dec_pre).permute(2, 0, 1).contiguous()
    T, B, D = x.shape
    t_idx = torch.arange(T, dtype=torch.float32)[:, None].expand(T, B)
    energy = x.pow(2).mean(-1).sqrt()
    cum = energy.cumsum(0)
    inp = x
    for l, p in enumerate(ps):
        h, rec = lstm_layer(inp, p["w_ih"], p["w_hh"], p["b"], record=True)
        f, i, o, c, g = rec["f"], rec["i"], rec["o"], rec["c"], rec["g"]
        fm = f.mean((0, 1))
        tau = -1 / torch.log(fm.clamp(1e-6, 1 - 1e-6))
        # regimes
        memoryless = (f < 0.1).float().mean((0, 1)) > 0.95          # forget gate shut: c_t = i*g, a gated feed-forward unit
        integrator = (f > 0.9).float().mean((0, 1)) > 0.95          # forget gate open: accumulates
        sat = (c.abs() > 3).float().mean((0, 1)) > 0.9              # tanh(c) pinned at +-1: output is just +-o
        const_out = h.std((0, 1)) < 0.02
        switching = ~(memoryless | integrator)
        print(f"\n{name} {side} layer {l + 1}: units by regime: memoryless (f<0.1 for >95% of frames) {int(memoryless.sum())}, "
              f"integrators (f>0.9 for >95%) {int(integrator.sum())}, data-dependent forget {int(switching.sum())}; "
              f"cell pinned |c|>3 for >90% of frames {int(sat.sum())}; near-constant output (std<0.02) {int(const_out.sum())}")
        # share of output variance carried by each regime
        hv = h.var((0, 1))
        print(f"   share of hidden-state variance: memoryless {100 * float(hv[memoryless].sum() / hv.sum()):.0f}%, integrators {100 * float(hv[integrator].sum() / hv.sum()):.0f}%, data-dependent {100 * float(hv[switching].sum() / hv.sum()):.0f}%")
        top = tau.argsort(descending=True)[:8]
        print("   longest-memory units: unit | tau frames | mean|c| | corr(c, frame index) | corr(c, accumulated input RMS) | corr(|dc|, input RMS) | frames with |c|>3 | h std | sign changes of c per 1000 frames")
        for u in top.tolist():
            cu = c[:, :, u]
            dc = (cu[1:] - cu[:-1]).abs()
            flips = float(((cu[1:] * cu[:-1]) < 0).float().mean() * 1000)
            print(f"   {u:4d} | {float(tau[u]):8.1f} | {float(cu.abs().mean()):7.2f} | {corr(cu, t_idx):+.2f} | {corr(cu, cum):+.2f} | {corr(dc, energy[1:]):+.2f} | {100 * float((cu.abs() > 3).float().mean()):3.0f}% | {float(h[:, :, u].std()):.3f} | {flips:.1f}")
        inp = h
