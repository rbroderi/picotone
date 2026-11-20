"""Gradient-descent comparison for the history experts (the main part of the cx1 model).

Usage: gd_comparison.py TAG KLIMIT [--epochs N] [--max-frames N]
Architecture (identical for all three variants, same ranks as the shipped tables):
    logits = bias + sum over the two history sources of  Out_s . (decayed sum of Emb_s[code])
Variants, scored by float cross-entropy on the test split:
  constructed : Emb/Out = truncated SVD of the decayed co-occurrence PMI, one scalar per
                source picked on dev (no gradient descent)
  finetune    : the same weights, then trained with Adam
  scratch     : random initialisation, trained with Adam
"""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn

import build_tables as build2
import lib

V = 1024
LN2 = float(np.log(2.0))
LAGS = 24
DECAY = 7.0 / 8.0
torch.set_num_threads(6)


def lag_matrix(src, start, idx, first_lag):
    """Codes of `src` at lags first_lag..first_lag+LAGS-1 before frames idx; V marks 'none'."""
    since = np.arange(len(src)) - np.maximum.accumulate(np.where(start, np.arange(len(src)), 0))
    out = np.full((len(idx), LAGS), V, dtype=np.int64)
    for i in range(LAGS):
        lag = first_lag + i
        ok = since[idx] >= lag
        out[ok, i] = src[idx[ok] - lag]
    return torch.from_numpy(out)


class Net(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(V + 1, rank, padding_idx=V) for _ in range(2)])
        self.out = nn.ModuleList([nn.Linear(rank, V, bias=False) for _ in range(2)])
        self.bias = nn.Parameter(torch.zeros(V))
        self.register_buffer("decay", DECAY ** torch.arange(LAGS, dtype=torch.float32))

    def forward(self, h0, h1):
        lg = self.bias
        for emb, out, h in zip(self.emb, self.out, (h0, h1)):
            lg = lg + out((emb(h) * self.decay[None, :, None]).sum(1))
        return lg


def evaluate(net, data):
    h0, h1, y = data
    tot = 0.0
    with torch.no_grad():
        for s in range(0, len(y), 8192):
            tot += float(nn.functional.cross_entropy(net(h0[s:s + 8192], h1[s:s + 8192]),
                                                     y[s:s + 8192], reduction="sum"))
    return tot / len(y) / LN2


def train(net, data, epochs, lr, seed):
    h0, h1, y = data
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = max(epochs * (len(y) // 512), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        for s in range(0, len(y) - 511, 512):
            i = perm[s:s + 512]
            loss = nn.functional.cross_entropy(net(h0[i], h1[i]), y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("klimit", type=int)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200000)
    args = ap.parse_args()
    sets = {s: lib.concat(lib.load_files(args.tag, s)) for s in ("fit", "dev", "test")}
    rng = np.random.default_rng(0)
    pick = {s: np.sort(rng.choice(len(c), min(n, len(c)), replace=False))
            for (s, (c, _)), n in zip(sets.items(), (args.max_frames, 20000, 60000))}
    rows = []
    for k in range(args.klimit):
        rank = build2.hist_rank(k)
        other, first = (1, 1) if k == 0 else (0, 0)
        data = {}
        for s, (codes, start) in sets.items():
            idx = pick[s]
            data[s] = (lag_matrix(codes[:, k], start, idx, 1),
                       lag_matrix(codes[:, other], start, idx, first),
                       torch.from_numpy(codes[idx, k]))
        codes, start = sets["fit"]
        y = codes[:, k]
        cnt = np.bincount(y, minlength=V).astype(np.float64)
        prior = (cnt + 0.5) / (cnt.sum() + 0.5 * V)
        factors = []
        for src, fl in ((y, 1), (codes[:, other], first)):
            a_int, b_int, scale = build2.history_factors(src, y, start, prior, fl, rank)
            # float value of one unit of the integer dot product, in nats (undo the prescale)
            nat = scale / build2.HIST_PRESCALE / 65536.0 * 16.0 * np.log(2.0) / (1 << build2.FB)
            factors.append((torch.from_numpy(a_int.astype(np.float32)),
                            torch.from_numpy(b_int.astype(np.float32)) * nat))
        net = Net(rank)
        best = None
        for w0 in (0.1, 0.2, 0.3, 0.45):
            for w1 in (0.0, 0.1, 0.2, 0.3):
                with torch.no_grad():
                    net.bias.copy_(torch.from_numpy(np.log(prior)).float())
                    for s, w in enumerate((w0, w1)):
                        net.emb[s].weight[:V].copy_(factors[s][0])
                        net.emb[s].weight[V].zero_()
                        net.out[s].weight.copy_(factors[s][1] * w)
                v = evaluate(net, data["dev"])
                if best is None or v < best[0]:
                    best = (v, w0, w1)
        with torch.no_grad():
            for s, w in enumerate(best[1:]):
                net.out[s].weight.copy_(factors[s][1] * w)
        row = {"k": k, "rank": rank, "unigram": float(-np.log2(prior[data["test"][2].numpy()]).mean()),
               "constructed": evaluate(net, data["test"]), "weights": best[1:]}
        train(net, data["fit"], args.epochs, 2e-4, k)
        row["finetune"] = evaluate(net, data["test"])
        torch.manual_seed(k)
        net = Net(rank)
        with torch.no_grad():
            net.bias.copy_(torch.from_numpy(np.log(prior)).float())
        train(net, data["fit"], args.epochs, 2e-3, k)
        row["scratch"] = evaluate(net, data["test"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(lib.WORK, f"train2_{args.tag}.json"), "w"), indent=1)
    for key in ("unigram", "constructed", "finetune", "scratch"):
        print(key, "mean bits/code over", len(rows), "codebooks:",
              round(float(np.mean([r[key] for r in rows])), 4), flush=True)


if __name__ == "__main__":
    main()
