"""Direct rewrites of the LSTM block  y = x + LSTM2(LSTM1(x)).

usage: python s02_lstm_replace.py 24khz|48khz enc|dec [group ...]
groups: abl lin prune lowrank gates ff   (default: all)

Every variant is scored on the held-out test split.  'fit' data (different files) is used only for
means (bias folding) and closed-form least-squares solves.  Tags in the label:
  [W]  pure inspection + writing (zeroing / deleting / constant folding, bias = recorded mean)
  [LS] uses a closed-form least-squares / eigen solve on recorded activations
"""
import os
import sys

import torch
from safetensors.torch import save_file

from common import OUT, Bench, Log, get_lstm, lstm_layer, run_layers

name, side = sys.argv[1], sys.argv[2]
groups = sys.argv[3:] or ["abl", "lin", "prune", "lowrank", "gates", "ff"]
n_q = int(os.environ.get("DWP_NQ", "0")) or None   # decode with only the first n_q codebooks (lower bitrate operating point)
log = Log(f"s02_{name}_{side}" + (f"_nq{n_q}" if n_q else "") + ("" if len(sys.argv) <= 3 or not n_q and name == "48khz" else "_" + "_".join(groups)))
test = Bench(name, n_q=n_q)
fit = Bench(name, "fit", n_q=n_q)
idx, stock = get_lstm(test.model, side)
D = 512
STOCK_MAC = sum(p["w_ih"].numel() + p["w_hh"].numel() for p in stock)


def pre(b):
    return (b.enc_pre if side == "enc" else b.dec_pre).permute(2, 0, 1).contiguous()  # [T,B,D]


xt, xf = pre(test), pre(fit)
T, Bf, _ = xf.shape


def run(layers, x, out=None):
    h = x
    for p in layers:
        h = lstm_layer(h, p["w_ih"], p["w_hh"], p["b"])
    if out is not None:
        h = h @ out[0].T + out[1]
    return h


def score(label, y, mac):
    """y: block output [T,B,D] on the test split."""
    y = y.permute(1, 2, 0).contiguous()
    if side == "enc":
        emb = run_layers(test.model.encoder.layers, y, idx + 1)
        r = test.eval_emb(emb)
    else:
        r = test.eval_wave(test.decode_from_pre(y, start=idx + 1))
    for k in ("snr", "logmel", "mrstft"):
        r["d_" + k] = r[k] - test.stock[k]
    log(label, r, lstm_mac=mac / STOCK_MAC)
    return r


def lsq(X, Y, ridge=1e-4):
    """Affine least squares Y ~ X W^T + b, ridge relative to the mean feature power.  Returns W [out,in], b."""
    X = X.double()
    Y = Y.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype)
    W = torch.linalg.solve(G, Xc.T @ Yc).T
    return W.float(), (my - mx @ W.T).float()


# ---- recorded activations of the stock LSTM on the fit split
h1f, rec1 = lstm_layer(xf, stock[0]["w_ih"], stock[0]["w_hh"], stock[0]["b"], record=True)
h2f, rec2 = lstm_layer(h1f, stock[1]["w_ih"], stock[1]["w_hh"], stock[1]["b"], record=True)
recs = [rec1, rec2]
hf = [h1f, h2f]
flat = lambda a: a.reshape(-1, a.shape[-1])

score("stock (manual LSTM, sanity)", xt + run(stock, xt), STOCK_MAC)

# ============================================================== ablations
if "abl" in groups:
    score("[W] LSTM removed: y = x", xt, 0)
    score("[W] LSTM removed, mean folded: y = x + mean(h2)", xt + flat(h2f).mean(0), 0)
    score("[W] skip removed: y = LSTM(x)", run(stock, xt), STOCK_MAC)
    # second layer only bypassed (y = x + h1) and first layer bypassed (y = x + LSTM2(x)): are the layers interchangeable?
    score("[W] layer 2 deleted: y = x + h1", xt + run(stock[:1], xt), STOCK_MAC / 2)
    score("[W] layer 1 deleted: y = x + LSTM2(x)", xt + run(stock[1:], xt), STOCK_MAC / 2)

# ============================================================== solved linear replacements of the whole block
if "lin" in groups:
    yf = xf + h2f
    causal = name == "24khz"
    for K in (1, 3, 7):
        if causal:
            offs = list(range(K))            # x_t, x_{t-1}, ...
        else:
            offs = [k - K // 2 for k in range(K)]

        def window(x):
            cols = []
            for o in offs:
                if o >= 0:
                    cols.append(torch.cat([torch.zeros(o, *x.shape[1:]), x[: len(x) - o]]))
                else:
                    cols.append(torch.cat([x[-o:], torch.zeros(-o, *x.shape[1:])]))
            return torch.cat(cols, -1)

        W, bias = lsq(flat(window(xf)), flat(yf))
        y = window(xt) @ W.T + bias
        note = "folds into the preceding conv: 0 extra MAC" if (K == 1 and name == "24khz") else ""
        score(f"[LS] whole block -> linear FIR, {K} tap(s) {note}", y, D * D * K)
        if K == 1:
            lin1 = (W, bias)
    # residual form: keep the skip exactly, solve only for the LSTM branch
    W, bias = lsq(flat(xf), flat(h2f))
    score("[LS] LSTM branch -> 1-tap linear (skip kept)", xt + xt @ W.T + bias, D * D)

# ============================================================== unit pruning
if "prune" in groups:
    def cond_var(h):
        """Variance of each unit not explained linearly by all the other units (closed form: 1/diag(inv(cov)))."""
        c = torch.cov(flat(h).double().T)
        c += 1e-6 * c.diagonal().mean() * torch.eye(len(c), dtype=c.dtype)
        return (1 / torch.linalg.inv(c).diagonal()).float()

    sens1 = stock[1]["w_ih"].pow(2).sum(0) + stock[0]["w_hh"].pow(2).sum(0)   # how strongly h1_j is read downstream
    sens2 = 1 + stock[1]["w_hh"].pow(2).sum(0)                                # h2_j goes straight to the output + recurrence
    crit = {
        "std": (flat(h1f).std(0) ** 2 * sens1, flat(h2f).std(0) ** 2 * sens2),
        "cond": (cond_var(h1f) * sens1, cond_var(h2f) * sens2),
    }

    def pruned(keep1, keep2, mode):
        """Returns (layers, out) for an LSTM keeping units keep1 in layer 1 and keep2 in layer 2."""
        H = D
        all_ = torch.arange(H)
        drop1 = all_[~torch.isin(all_, keep1)]
        drop2 = all_[~torch.isin(all_, keep2)]
        rows = lambda keep: torch.cat([keep + g * H for g in range(4)])
        m1, m2 = flat(h1f).mean(0), flat(h2f).mean(0)
        if mode == "mean":
            P1 = torch.zeros(len(drop1), len(keep1)); q1 = m1[drop1]
            P2 = torch.zeros(len(drop2), len(keep2)); q2 = m2[drop2]
        else:
            P1, q1 = lsq(flat(h1f)[:, keep1], flat(h1f)[:, drop1], 1e-3) if len(drop1) else (torch.zeros(0, len(keep1)), torch.zeros(0))
            P2, q2 = lsq(flat(h2f)[:, keep2], flat(h2f)[:, drop2], 1e-3) if len(drop2) else (torch.zeros(0, len(keep2)), torch.zeros(0))
        r1, r2 = rows(keep1), rows(keep2)
        a, b_ = stock
        l1 = {"w_ih": a["w_ih"][r1],
              "w_hh": a["w_hh"][r1][:, keep1] + a["w_hh"][r1][:, drop1] @ P1,
              "b": a["b"][r1] + a["w_hh"][r1][:, drop1] @ q1}
        l2 = {"w_ih": b_["w_ih"][r2][:, keep1] + b_["w_ih"][r2][:, drop1] @ P1,
              "w_hh": b_["w_hh"][r2][:, keep2] + b_["w_hh"][r2][:, drop2] @ P2,
              "b": b_["b"][r2] + b_["w_ih"][r2][:, drop1] @ q1 + b_["w_hh"][r2][:, drop2] @ q2}
        E = torch.zeros(H, len(keep2)); e = torch.zeros(H)
        E[keep2, torch.arange(len(keep2))] = 1
        E[drop2] = P2; e[drop2] = q2
        mac = l1["w_ih"].numel() + l1["w_hh"].numel() + l2["w_ih"].numel() + l2["w_hh"].numel() + (P2.numel() if mode == "ls" else 0)
        return [l1, l2], (E, e), mac

    for mode, cname in (("mean", "std"), ("ls", "cond")):
        tag = "[W]" if mode == "mean" else "[LS]"
        c1, c2 = crit[cname]
        for K in (496, 480, 448, 384, 320, 256):
            keep1 = c1.argsort(descending=True)[:K].sort().values
            keep2 = c2.argsort(descending=True)[:K].sort().values
            for which, (k1, k2) in {"both layers": (keep1, keep2), "layer 1 only": (keep1, torch.arange(D)), "layer 2 only": (torch.arange(D), keep2)}.items():
                if which != "both layers" and K not in (448, 384):
                    continue
                layers, out, mac = pruned(k1, k2, mode)
                r = score(f"{tag} prune units to {K}/512 ({which}), compensation={mode}", xt + run(layers, xt, out), mac)
                if which == "both layers" and mode == "ls" and K in (384, 320, 256) and not n_q:
                    save_file({"l0.w_ih": layers[0]["w_ih"], "l0.w_hh": layers[0]["w_hh"], "l0.b": layers[0]["b"],
                               "l1.w_ih": layers[1]["w_ih"], "l1.w_hh": layers[1]["w_hh"], "l1.b": layers[1]["b"],
                               "out.weight": out[0], "out.bias": out[1]},
                              os.path.join(OUT, f"lstm_{name}_{side}_pruned{K}_ls.safetensors"))

# ============================================================== activation-aware low rank of the recurrent matrices
if "lowrank" in groups:
    for r_ in (448, 384, 256):
        layers = []
        mac = 0
        for l, p in enumerate(stock):
            h = flat(hf[l]).double()
            m = h.mean(0)
            cov = torch.cov(h.T)
            ev, U = torch.linalg.eigh(cov)
            U = U[:, -r_:].float()                       # top-r principal directions of the hidden state
            w = (p["w_hh"] @ U) @ U.T                    # W_hh restricted to the subspace the state actually visits
            b_ = p["b"] + (p["w_hh"] - w) @ m.float()    # the discarded directions contribute their mean
            layers.append({"w_ih": p["w_ih"], "w_hh": w, "b": b_})
            mac += p["w_ih"].numel() + r_ * D + 4 * D * r_
        score(f"[LS] W_hh -> rank {r_} in the PCA basis of recorded h (both layers)", xt + run(layers, xt), mac)
    # plain weight SVD for reference (the earlier agent's baseline, recurrent matrices only)
    layers = []
    for p in stock:
        U, S, V = torch.linalg.svd(p["w_hh"], full_matrices=False)
        layers.append({"w_ih": p["w_ih"], "w_hh": (U[:, :448] * S[:448]) @ V[:448], "b": p["b"]})
    score("[baseline] W_hh -> rank 448 by plain weight SVD (both layers)", xt + run(layers, xt), 2 * (4 * D * D + 448 * D + 4 * D * 448))

# ============================================================== gates that never move -> constants
if "gates" in groups:
    for thr in (0.02, 0.05, 0.1, 0.2):
        layers = []
        removed = 0
        for l, p in enumerate(stock):
            w_ih, w_hh, b_ = p["w_ih"].clone(), p["w_hh"].clone(), p["b"].clone()
            for gi, g in ((0, "i"), (1, "f"), (3, "o")):
                v = flat(recs[l][g])
                const = v.std(0) < thr
                mean = v.mean(0).clamp(1e-4, 1 - 1e-4)
                rows = torch.nonzero(const).flatten() + gi * D
                w_ih[rows] = 0
                w_hh[rows] = 0
                b_[rows] = torch.logit(mean[const])
                removed += int(const.sum())
            layers.append({"w_ih": w_ih, "w_hh": w_hh, "b": b_})
        mac = STOCK_MAC - removed * 2 * D
        score(f"[W] gates with std<{thr} frozen to their mean ({removed}/{6 * D} gate rows deleted)", xt + run(layers, xt), mac)

# ============================================================== recurrence -> feed-forward
if "ff" in groups:
    for l in (0, 1):
        p = stock[l]
        inp = xf if l == 0 else h1f
        h = hf[l]
        hprev = torch.cat([torch.zeros(1, Bf, D), h[:-1]])
        rec_term = flat(hprev) @ p["w_hh"].T                 # what the recurrence adds to the gate pre-activations
        # (a) delete the recurrence, fold its mean into the bias
        la = {"w_ih": p["w_ih"], "w_hh": torch.zeros_like(p["w_hh"]), "b": p["b"] + rec_term.mean(0)}
        layers = list(stock); layers[l] = la
        score(f"[W] layer {l + 1}: W_hh deleted, mean folded into bias (cell state kept)", xt + run(layers, xt), STOCK_MAC - 4 * D * D)
        # (b) predict the recurrent term from the current input and fold it into W_ih
        Q, q = lsq(flat(inp), rec_term)
        lb = {"w_ih": p["w_ih"] + Q, "w_hh": torch.zeros_like(p["w_hh"]), "b": p["b"] + q}
        layers = list(stock); layers[l] = lb
        score(f"[LS] layer {l + 1}: W_hh h(t-1) predicted from x(t), folded into W_ih", xt + run(layers, xt), STOCK_MAC - 4 * D * D)
    # both layers
    layers = []
    for l in (0, 1):
        p = stock[l]
        inp = xf if l == 0 else lstm_layer(xf, layers[0]["w_ih"], layers[0]["w_hh"], layers[0]["b"])
        h = hf[l]
        hprev = torch.cat([torch.zeros(1, Bf, D), h[:-1]])
        Q, q = lsq(flat(inp), flat(hprev) @ p["w_hh"].T)
        layers.append({"w_ih": p["w_ih"] + Q, "w_hh": torch.zeros_like(p["w_hh"]), "b": p["b"] + q})
    score("[LS] both layers: recurrence folded into W_ih (LSTM becomes a gated feed-forward cell)", xt + run(layers, xt), STOCK_MAC - 8 * D * D)
