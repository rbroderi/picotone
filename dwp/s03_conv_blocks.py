"""Interpretation + direct rewrites of the SEANet conv / residual blocks.

usage: python s03_conv_blocks.py 24khz|48khz enc|dec

A residual block is   out = shortcut_1x1(x) + conv_1x1(ELU(conv_k3(ELU(x))))   with a C/2 bottleneck.
Tags: [W] pure writing (delete / fold mean), [LS] closed-form least squares on recorded activations.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import Bench, Log, raw_weights, run_layers

name, side = sys.argv[1], sys.argv[2]
log = Log(f"s03_{name}_{side}")
test = Bench(name)
fit = Bench(name, "fit")
m = test.model
layers = m.encoder.layers if side == "enc" else m.decoder.layers
causal = name == "24khz"
sr = test.sr
frame_rate = 75 if name == "24khz" else 150

# inputs of the side being edited
FIT_SUB = slice(0, None, 7)
if side == "enc":
    x_test, x_fit = test.x, fit.x[FIT_SUB]
else:
    x_test, x_fit = test.dequant(test.codes), fit.dequant(fit.codes[:, FIT_SUB])


def batched(fn, x, bs=6):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def finish(h, start):
    """Run layers[start:] on the test activations h and score."""
    out = batched(lambda z: run_layers(layers, z, start), h)
    if side == "enc":
        return test.eval_emb(out)
    return test.eval_wave(out)


def lsq(X, Y, ridge=1e-5):
    X, Y = X.double(), Y.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype)
    W = torch.linalg.solve(G, Xc.T @ Yc).T
    return W.float(), (my - mx @ W.T).float()


def cols(a, max_rows=400000):
    """[B,C,T] -> [rows,C] (subsampled)."""
    a = a.permute(0, 2, 1).reshape(-1, a.shape[1])
    step = max(1, len(a) // max_rows)
    return a[::step]


# ---- MAC budget of this side
def conv_w(mod):
    return mod.conv.weight


rates = {}
r = sr if side == "enc" else frame_rate
total_mac = 0.0
mac_of = {}
for i, l in enumerate(layers):
    cls = type(l).__name__
    if cls == "EncodecConv1d":
        s = l.conv.stride[0]
        r = r / s
        mac_of[i] = conv_w(l).numel() * r
    elif cls == "EncodecConvTranspose1d":
        mac_of[i] = conv_w(l).numel() * r
        r = r * l.conv.stride[0]
    elif cls == "EncodecResnetBlock":
        mac_of[i] = {"k3": conv_w(l.block[1]).numel() * r, "k1": conv_w(l.block[3]).numel() * r, "sc": conv_w(l.shortcut).numel() * r}
    elif cls == "EncodecLSTM":
        mac_of[i] = 4 * 2048 * 512 * frame_rate
    rates[i] = r
total_mac = sum(v if not isinstance(v, dict) else sum(v.values()) for v in mac_of.values())
print(f"{name} {side}: total MAC/s {total_mac / 1e6:.0f} M")

# ---- first / last conv: what do the filters look like?
if side == "enc":
    w0 = conv_w(layers[0])  # [32, ch, 7]
    spec = torch.fft.rfft(w0, 256, dim=-1).abs().mean(1)
    peak = spec.argmax(1).float() / 256 * sr
    dc = w0.sum(-1).abs().mean(1) / w0.abs().sum(-1).mean(1)
    print("encoder conv0: 7-tap filters; peak frequency (Hz) of each of the 32 filters, sorted:", [int(v) for v in peak.sort().values])
    print("   |sum(taps)|/sum|taps| (1 = pure low-pass / averaging, 0 = pure band/high-pass):", [round(float(v), 2) for v in dc])
else:
    wl = conv_w(layers[-1])  # [ch, 32, 7]
    spec = torch.fft.rfft(wl, 256, dim=-1).abs()
    peak = spec.argmax(-1).float() / 256 * sr
    print("decoder last conv: peak frequency (Hz) of the 7-tap filter applied to each of the 32 channels:", [int(v) for v in peak.flatten().sort().values])
    print("   tap-norm per input channel:", [round(float(v), 2) for v in wl.norm(dim=-1).flatten()])

res_idx = [i for i, l in enumerate(layers) if type(l).__name__ == "EncodecResnetBlock"]
log("stock", finish(batched(lambda z: run_layers(layers, z, 0, res_idx[0]), x_test), res_idx[0]))

for bi in res_idx:
    blk = layers[bi]
    C = conv_w(blk.shortcut).shape[0]
    Hc = C // 2
    mac = mac_of[bi]
    share = lambda v: 100 * v / total_mac
    hin_t = batched(lambda z: run_layers(layers, z, 0, bi), x_test)
    hin_f = batched(lambda z: run_layers(layers, z, 0, bi), x_fit)

    def branch(z, upto=4):
        for l in list(blk.block)[:upto]:
            z = l(z)
        return z

    sc_f = batched(blk.shortcut, hin_f)
    br_f = batched(branch, hin_f)
    hid_f = batched(lambda z: F.elu(branch(z, 2)), hin_f)        # ELU(conv_k3(ELU(x))): input of the 1x1 output conv
    S = conv_w(blk.shortcut)[:, :, 0]
    sv = torch.linalg.svdvals(S)
    eye = torch.eye(C)
    print(f"\n=== {side} layer {bi}: residual block C={C} bottleneck={Hc} rate={rates[bi]:.0f} Hz  MAC share of side: k3 {share(mac['k3']):.1f}% k1 {share(mac['k1']):.1f}% shortcut {share(mac['sc']):.1f}%")
    print(f"shortcut 1x1: singular values max {float(sv[0]):.2f} median {float(sv[C // 2]):.2f} min {float(sv[-1]):.3f}; diag mean {float(S.diagonal().mean()):+.3f}; ||S-I||/||S|| {float((S - eye).norm() / S.norm()):.2f}; ||S-diag||/||S|| {float((S - torch.diag(S.diagonal())).norm() / S.norm()):.2f}")
    print(f"energy on fit audio: shortcut {float(sc_f.pow(2).mean()):.4f}, branch {float(br_f.pow(2).mean()):.4f} (branch/shortcut {float(br_f.pow(2).mean() / sc_f.pow(2).mean()):.3f}), cos {float((sc_f * br_f).sum() / (sc_f.norm() * br_f.norm())):+.3f}")
    pre_k3 = batched(lambda z: branch(z, 2), hin_f)
    print(f"ELU regime: block input negative {100 * float((hin_f < 0).float().mean()):.0f}% (saturated <-3: {100 * float((hin_f < -3).float().mean()):.0f}%), bottleneck negative {100 * float((pre_k3 < 0).float().mean()):.0f}% (saturated: {100 * float((pre_k3 < -3).float().mean()):.0f}%)")
    hstd = cols(hid_f).std(0)
    print(f"bottleneck channel std: min {float(hstd.min()):.3f} median {float(hstd.median()):.3f} max {float(hstd.max()):.3f}; channels with std < 5% of median: {int((hstd < 0.05 * hstd.median()).sum())}")

    sc_t = batched(blk.shortcut, hin_t)
    full_mac = sum(mac.values())
    tag = f"L{bi} C={C}"
    # 1/2: branch deleted
    log(f"[W] {tag}: branch deleted (shortcut only)", finish(sc_t, bi + 1), side_mac_saved_pct=share(mac["k3"] + mac["k1"] + (mac["sc"] if causal else 0)))
    mean_br = br_f.mean((0, 2))
    log(f"[W] {tag}: branch deleted, its mean folded into bias", finish(sc_t + mean_br[None, :, None], bi + 1), side_mac_saved_pct=share(mac["k3"] + mac["k1"] + (mac["sc"] if causal else 0)))
    # 3: whole block -> solved 1x1
    out_f = sc_f + br_f
    W, b_ = lsq(cols(hin_f), cols(out_f))
    log(f"[LS] {tag}: whole block -> one solved 1x1 conv", finish(F.conv1d(hin_t, W[:, :, None], b_), bi + 1), side_mac_saved_pct=share(full_mac if causal else mac["k3"] + mac["k1"]))
    # 4: whole block -> solved 3-tap linear conv
    def win3(a):
        if causal:
            p = F.pad(a, (2, 0))
        else:
            p = F.pad(a, (1, 1))
        return torch.cat([p[:, :, k:k + a.shape[-1]] for k in range(3)], 1)

    W, b_ = lsq(cols(win3(hin_f)), cols(out_f))
    log(f"[LS] {tag}: whole block -> one solved 3-tap linear conv", finish(F.conv1d(win3(hin_t), W[:, :, None], b_), bi + 1), side_mac_saved_pct=share(full_mac - 3 * mac["sc"]))
    # 5: shortcut -> identity / diagonal
    br_t = batched(branch, hin_t)
    log(f"[W] {tag}: shortcut conv -> identity", finish(hin_t + br_t, bi + 1), side_mac_saved_pct=share(mac["sc"]))
    Wd, bd = lsq(cols(hin_f), cols(sc_f))  # sanity: equals S
    # 6: bottleneck pruning
    k1w = conv_w(blk.block[3])[:, :, 0]                      # [C, Hc]
    k1b = blk.block[3].conv.bias
    imp = cols(hid_f).pow(2).mean(0) * k1w.pow(2).sum(0)     # activation energy x read-out strength
    hid_t = batched(lambda z: F.elu(branch(z, 2)), hin_t)
    is_gn = hasattr(blk.block[3], "norm")
    for keep_frac in (0.75, 0.5, 0.25):
        K = int(Hc * keep_frac)
        keep = imp.argsort(descending=True)[:K].sort().values
        drop = torch.tensor([j for j in range(Hc) if j not in set(keep.tolist())])
        saved = share((mac["k3"] + mac["k1"]) * (1 - keep_frac))
        if not is_gn:
            mean_drop = cols(hid_f)[:, drop].mean(0)
            b2 = k1b + k1w[:, drop] @ mean_drop
            out = F.conv1d(hid_t[:, keep], k1w[:, keep][:, :, None], b2)
            log(f"[W] {tag}: bottleneck pruned to {K}/{Hc} by activation importance, mean folded", finish(sc_t + out, bi + 1), side_mac_saved_pct=saved)
        W, b_ = lsq(cols(hid_f)[:, keep], cols(br_f))
        out = F.conv1d(hid_t[:, keep], W[:, :, None], b_)
        log(f"[LS] {tag}: bottleneck pruned to {K}/{Hc}, 1x1 read-out re-solved", finish(sc_t + out, bi + 1), side_mac_saved_pct=saved)
