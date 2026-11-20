"""Scores the held-out test files with Meta's EnCodec language model (the `--lm` model).

Usage: meta_lm_score.py TAG SPLIT K [K ...]   (SPLIT: test, test2, or a directory of .ecdc files under the work dir)
Loads Meta's published LM checkpoint, runs it exactly as `encodec/compress.py` does (previous
frame's K codes summed as input, state reset per coded segment), and reports
  ideal : cross-entropy of the float probabilities
  coded : bits with Meta's `build_stable_quantized_cdf` (24-bit ranges, min range 2)
This measures the size their arithmetic coder would produce without porting the coder itself.
The model classes below re-implement encodec 0.1.1's `LMModel` and streaming transformer
(MIT licence); no downloaded code is executed and the checkpoint is loaded with
`weights_only=True`.
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn

import ecdcio
import lib

TAG = sys.argv[1]
SPLIT = sys.argv[2]
KS = [int(a) for a in sys.argv[3:]]
SCRATCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(6)


def create_sin_embedding(positions, dim, max_period=10000):
    half = dim // 2
    adim = torch.arange(half).view(1, 1, -1)
    phase = positions / (max_period ** (adim / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class Layer(nn.TransformerEncoderLayer):
    """Post-norm encoder layer attending to [past state, current] within `past_context`."""

    def forward(self, x, x_past, past_context):
        sa_input = x
        t, h = x.shape[1], x_past.shape[1]
        keys = torch.cat([x_past, x], dim=1)
        delta = torch.arange(h, t + h).view(-1, 1) - torch.arange(t + h).view(1, -1)
        valid = (delta >= 0) & (delta <= past_context)
        att = self.self_attn(x, keys, keys, attn_mask=~valid, need_weights=False)[0]
        x = self.norm1(x + att)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x, sa_input


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=5, past_context=1000):
        super().__init__()
        self.past_context = past_context
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            Layer(dim, num_heads, dim * 4, activation=nn.functional.gelu, batch_first=True,
                  dropout=0.0) for _ in range(num_layers)])

    def forward(self, x, states=None, offset=0):
        t, c = x.shape[1], x.shape[2]
        if states is None:
            states = [torch.zeros_like(x[:, :1]) for _ in self.layers]
        pos = torch.arange(t).view(1, -1, 1) + offset
        x = self.norm_in(x) + create_sin_embedding(pos, c)
        new_state = []
        for st, layer in zip(states, self.layers):
            x, inp = layer(x, st, self.past_context)
            new_state.append(torch.cat([st, inp], dim=1)[:, -self.past_context:, :])
        return x, new_state, offset + t


class LMModel(nn.Module):
    def __init__(self, n_q=32, card=1024, dim=200, **kwargs):
        super().__init__()
        self.card, self.n_q, self.dim = card, n_q, dim
        self.transformer = Transformer(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices, states=None, offset=0):
        _, k, _ = indices.shape
        input_ = sum([self.emb[i](indices[:, i]) for i in range(k)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[i](out) for i in range(k)], dim=1).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset


def score(lm, codes):
    """codes [B, T, K] long. Returns (ideal bits, coded bits) summed, and symbol count."""
    b, t, k = codes.shape
    inp = torch.zeros(b, k, t, dtype=torch.long)
    inp[:, :, 1:] = 1 + codes[:, :-1].permute(0, 2, 1)
    ideal = coded = 0.0
    states, offset = None, 0
    step = 256
    alpha = 2 * 1024 / 2 ** 24
    for s in range(0, t, step):
        with torch.no_grad():
            probas, states, offset = lm(inp[:, :, s:s + step], states, offset)
        tgt = codes[:, s:s + step].permute(0, 2, 1)  # [B, K, T']
        p = probas.gather(1, tgt[:, None])[:, 0].double()  # [B, K, T']
        ideal += float(-torch.log2(p).sum())
        pf = (p / 1e-8).floor() * 1e-8
        rng = (((1 - alpha) * 2 ** 24) * pf).floor() + 2
        coded += float((24 - torch.log2(rng)).sum())
    return ideal, coded, b * t * k


def main():
    frame_rate = lib.FRAME_RATE[TAG]
    lm = LMModel(lib.KMAX[TAG], 1024, num_layers=5, dim=200,
                 past_context=int(3.5 * frame_rate))
    ckpt = os.path.join(SCRATCH, "metalm", "ckpt", "lm24.th" if TAG == "24k" else "lm48.th")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    lm.load_state_dict(state)
    lm.eval()
    nparams = sum(p.numel() for p in lm.parameters())
    print("LM parameters", nparams, flush=True)
    files = lib.load_files(TAG, SPLIT) if SPLIT in ("test", "test2") else [
        {"name": os.path.basename(p), "path": p}
        for p in sorted(__import__("glob").glob(os.path.join(lib.WORK, SPLIT, "*.ecdc")))]
    out = {"params": nparams, "files": {}}
    for kk in KS:
        tot_i = tot_c = tot_n = 0
        for f in files:
            meta, segs = ecdcio.read(f["path"])
            if TAG == "24k":
                batches = [torch.from_numpy(segs[0][1][None, :, :kk].astype(np.int64))]
            else:
                full = [c for _, c in segs if len(c) == 150]
                rest = [c for _, c in segs if len(c) != 150]
                batches = []
                for i in range(0, len(full), 32):
                    batches.append(torch.from_numpy(
                        np.stack(full[i:i + 32])[:, :, :kk].astype(np.int64)))
                batches += [torch.from_numpy(c[None, :, :kk].astype(np.int64)) for c in rest]
            fi = fc = fn = 0
            for bt in batches:
                i, c, n = score(lm, bt)
                fi, fc, fn = fi + i, fc + c, fn + n
            out["files"].setdefault(f["name"], {})[str(kk)] = {"ideal_bits": fi, "coded_bits": fc,
                                                                "symbols": fn}
            tot_i, tot_c, tot_n = tot_i + fi, tot_c + fc, tot_n + fn
            print(kk, f["name"][:50], round(fi / fn, 3), round(fc / fn, 3), flush=True)
        print(f"TOTAL {TAG} K={kk}: ideal {tot_i / tot_n:.4f} coded {tot_c / tot_n:.4f} "
              f"bits/code over {tot_n} codes", flush=True)
        out[f"total_{kk}"] = {"ideal": tot_i / tot_n, "coded": tot_c / tot_n, "symbols": tot_n}
    json.dump(out, open(os.path.join(lib.WORK, f"metalm_{TAG}_{SPLIT.replace(chr(47), chr(95))}.json"), "w"), indent=1)


main()
