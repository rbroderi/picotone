"""Smaller LSTM written by unit selection + layer-wise closed-form reconstruction  [LS].

usage: python s08_lstm_prune_refit.py 24khz|48khz enc|dec [n_q]

Compared with the one-shot pruning of s02, every interface is re-solved against the STOCK network's signals
while being driven by the PRUNED network's own (drifted) states:

  1. keep the K1 / K2 units whose activity is least predictable from the other units (x downstream weight);
  2. layer 1: fold the dropped units into W_hh (least squares), then re-solve W_hh so that
     W_hh' h_pruned(t-1) reproduces the stock recurrent drive  W_hh h_stock(t-1)  (two fixed-point passes);
  3. layer 2 input weights: W_ih' h1_pruned(t) ~ W_ih h1_stock(t);
  4. layer 2 recurrence: as in 2;
  5. output: a dense 512 x K2 read-out  E h2_pruned(t) ~ h2_stock(t)  replaces the identity.

Optionally layer 1 is also made feed-forward ('ff'): its recurrent drive is predicted from the current input
and folded into W_ih, so the layer has no matrix recurrence at all (only the element-wise cell update).

No gradients anywhere; each step is one ridge-regularised normal-equation solve on recorded activations.
Derived weights are saved to out/lstm_<model>_<side>_refit<K>[_ff].safetensors.
"""
import os
import sys

import torch
from safetensors.torch import save_file

from common import OUT, Bench, Log, get_lstm, lstm_layer, run_layers

name, side = sys.argv[1], sys.argv[2]
n_q = int(sys.argv[3]) if len(sys.argv) > 3 else None
BYPASS = os.environ.get("DWP_BYPASS") == "1"
log = Log(f"s08_{name}_{side}" + (f"_nq{n_q}" if n_q else "") + ("_bypass" if BYPASS else ""))
test = Bench(name, n_q=n_q)
fit = Bench(name, "fit", n_q=n_q)
idx, stock = get_lstm(test.model, side)
D = 512
STOCK_MAC = 4 * 4 * D * D
pre = lambda b: (b.enc_pre if side == "enc" else b.dec_pre).permute(2, 0, 1).contiguous()
xt, xf = pre(test), pre(fit)
T, Bf, _ = xf.shape
flat = lambda a: a.reshape(-1, a.shape[-1])
shift = lambda h: torch.cat([torch.zeros(1, *h.shape[1:]), h[:-1]])


def lsq(X, Y, ridge=1e-4, affine=True):
    X, Y = X.double(), Y.double()
    mx = X.mean(0) if affine else torch.zeros(X.shape[1], dtype=X.dtype)
    my = Y.mean(0) if affine else torch.zeros(Y.shape[1], dtype=Y.dtype)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype)
    W = torch.linalg.solve(G, Xc.T @ Yc).T
    return W.float(), (my - mx @ W.T).float()


def cond_var(h):
    c = torch.cov(flat(h).double().T)
    c += 1e-6 * c.diagonal().mean() * torch.eye(len(c), dtype=c.dtype)
    return (1 / torch.linalg.inv(c).diagonal()).float()


h1s = lstm_layer(xf, stock[0]["w_ih"], stock[0]["w_hh"], stock[0]["b"])
h2s = lstm_layer(h1s, stock[1]["w_ih"], stock[1]["w_hh"], stock[1]["b"])
rows = lambda keep: torch.cat([keep + g * D for g in range(4)])
crit1 = cond_var(h1s) * (stock[1]["w_ih"].pow(2).sum(0) + stock[0]["w_hh"].pow(2).sum(0))
crit2 = cond_var(h2s) * (1 + stock[1]["w_hh"].pow(2).sum(0))


def score(label, y, mac, extra=None):
    y = y.permute(1, 2, 0).contiguous()
    if side == "enc":
        r = test.eval_emb(run_layers(test.model.encoder.layers, y, idx + 1))
    else:
        r = test.eval_wave(test.decode_from_pre(y, start=idx + 1))
    for k in ("snr", "logmel", "mrstft"):
        r["d_" + k] = r[k] - test.stock[k]
    log(label, r, lstm_mac=mac / STOCK_MAC, **(extra or {}))
    return r


def refit_recurrence(w_ih, w_hh0, b0, b_base, inp, target_rec, passes=2):
    """Re-solve W_hh so that the pruned layer's own previous state reproduces the stock recurrent drive."""
    w_hh, b_ = w_hh0, b0
    for _ in range(passes):
        h = lstm_layer(inp, w_ih, w_hh, b_)
        w_hh, db = lsq(flat(shift(h)), target_rec)
        b_ = b_base + db
    return w_hh, b_


def build(K1, K2, ff1=False, refit=True):
    a, b = stock
    keep1 = crit1.argsort(descending=True)[:K1].sort().values
    keep2 = crit2.argsort(descending=True)[:K2].sort().values
    drop1 = torch.tensor(sorted(set(range(D)) - set(keep1.tolist())), dtype=torch.long)
    drop2 = torch.tensor(sorted(set(range(D)) - set(keep2.tolist())), dtype=torch.long)
    r1, r2 = rows(keep1), rows(keep2)
    # ---- layer 1
    rec1_target = flat(shift(h1s)) @ a["w_hh"][r1].T                       # stock recurrent drive of the kept rows
    if ff1:
        Q, q = lsq(flat(xf), rec1_target)
        l1 = {"w_ih": a["w_ih"][r1] + Q, "w_hh": torch.zeros(4 * K1, K1), "b": a["b"][r1] + q}
        mac1 = 4 * K1 * D
    else:
        if len(drop1):
            P1, q1 = lsq(flat(h1s)[:, keep1], flat(h1s)[:, drop1], 1e-3)
        else:
            P1, q1 = torch.zeros(0, K1), torch.zeros(0)
        w_hh = a["w_hh"][r1][:, keep1] + a["w_hh"][r1][:, drop1] @ P1
        b_ = a["b"][r1] + a["w_hh"][r1][:, drop1] @ q1
        if refit and len(drop1):
            w_hh, b_ = refit_recurrence(a["w_ih"][r1], w_hh, b_, a["b"][r1], xf, rec1_target)
        l1 = {"w_ih": a["w_ih"][r1], "w_hh": w_hh, "b": b_}
        mac1 = 4 * K1 * D + 4 * K1 * K1
    h1p = lstm_layer(xf, l1["w_ih"], l1["w_hh"], l1["b"])
    # ---- layer 2
    in2_target = flat(h1s) @ b["w_ih"][r2].T
    W2, c2 = lsq(flat(h1p), in2_target)
    rec2_target = flat(shift(h2s)) @ b["w_hh"][r2].T
    if len(drop2):
        P2, q2 = lsq(flat(h2s)[:, keep2], flat(h2s)[:, drop2], 1e-3)
    else:
        P2, q2 = torch.zeros(0, K2), torch.zeros(0)
    w_hh = b["w_hh"][r2][:, keep2] + b["w_hh"][r2][:, drop2] @ P2
    b_ = b["b"][r2] + c2 + b["w_hh"][r2][:, drop2] @ q2
    if refit and len(drop2):
        w_hh, b_ = refit_recurrence(W2, w_hh, b_, b["b"][r2] + c2, h1p, rec2_target)
    l2 = {"w_ih": W2, "w_hh": w_hh, "b": b_}
    h2p = lstm_layer(h1p, W2, w_hh, b_)
    # ---- output read-out
    mac = mac1 + 4 * K2 * K1 + 4 * K2 * K2 + D * K2
    if BYPASS:
        # read-out sees the block input too: branch ~ E h2_pruned + A x.  In the 24 kHz model A (like the skip) folds
        # into the conv feeding the LSTM, so it is free; in the 48 kHz model a GroupNorm sits in between -> 512x512 extra.
        EA, e = lsq(torch.cat([flat(h2p), flat(xf)], 1), flat(h2s))
        E, A = EA[:, :K2].contiguous(), EA[:, K2:].contiguous()
        pred = h2p @ E.T + xf @ A.T + e
        mac += 0 if name == "24khz" else D * D
    else:
        E, e = lsq(flat(h2p), flat(h2s))
        A = torch.zeros(D, D)
        pred = h2p @ E.T + e
    fit_snr = float(10 * torch.log10(h2s.pow(2).sum() / (h2s - pred).pow(2).sum()))
    return [l1, l2], (E, e, A), mac, fit_snr


def run(layers, out, x):
    h = x
    for p in layers:
        h = lstm_layer(h, p["w_ih"], p["w_hh"], p["b"])
    y = h @ out[0].T + out[1]
    if BYPASS:
        y = y + x @ out[2].T
    return y


score("stock", xt + lstm_layer(lstm_layer(xt, stock[0]["w_ih"], stock[0]["w_hh"], stock[0]["b"]), stock[1]["w_ih"], stock[1]["w_hh"], stock[1]["b"]), STOCK_MAC)
configs = [(512, 512, True), (448, 448, False), (384, 384, False), (320, 320, False), (256, 256, False), (192, 192, False), (128, 128, False),
           (384, 384, True), (256, 256, True), (384, 256, False), (256, 384, False)]
if BYPASS:
    configs = [(384, 384, False), (256, 256, False), (192, 192, False), (128, 128, False), (64, 64, False), (256, 256, True), (128, 128, True)]
if os.environ.get("DWP_CONFIGS"):   # e.g. "384,384,0;256,256,0"
    configs = [(int(a), int(b), c == "1") for a, b, c in (s.split(",") for s in os.environ["DWP_CONFIGS"].split(";"))]
for K1, K2, ff1 in configs:
    layers, out, mac, fit_snr = build(K1, K2, ff1)
    lab = f"[LS] refit prune L1={K1} L2={K2}" + (" + layer 1 feed-forward" if ff1 else "") + (" + linear bypass" if BYPASS else "")
    score(lab, xt + run(layers, out, xt), mac, {"branch_fit_snr_db": fit_snr})
    tag = f"refit{K1}_{K2}" + ("_ff" if ff1 else "") + ("_bypass" if BYPASS else "")
    if (K1, K2) in ((384, 384), (256, 256), (128, 128)) and not n_q:
        save_file({"l0.w_ih": layers[0]["w_ih"], "l0.w_hh": layers[0]["w_hh"], "l0.b": layers[0]["b"],
                   "l1.w_ih": layers[1]["w_ih"], "l1.w_hh": layers[1]["w_hh"], "l1.b": layers[1]["b"],
                   "out.weight": out[0], "out.bias": out[1], "bypass.weight": out[2]},
                  os.path.join(OUT, f"lstm_{name}_{side}_{tag}.safetensors"))
