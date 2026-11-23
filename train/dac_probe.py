"""M5 probe: how compressible are HQ (DAC 44 kHz) codes? Held-out bits per code under a few context models.

    uv run --extra cpu python dac_probe.py --codes ../work/dac_codes --weights ../weights --out out/dac_probe.json

Input: `<codes>/{fit,test}/<clip>.npy` (or `<clip>.0.npy` and `<clip>.1.npy` for two coded channels), each
`[frames, 9]` u16, plus `<clip>.json` with the layout; `picotone dev codes` writes them. Fit and test are the cx2
phase-1 split, so no clip (or creator) is in both.

Every number is a held-out cross-entropy on the test clips, in bits per code, against 10 bits for the raw
bit-packing. Models:

- unigram: a smoothed frequency table per codebook.
- bigram prev-frame / prev-codebook: absolute-discounting bigrams (as in tools/context_entropy.py) on the same
  codebook one frame back, or the previous codebook in the same frame.
- net: a small MLP that sees the last `--history` frames (all codebooks), the codebooks already coded in this
  frame and, for the second coded channel of a stereo stream, the first channel's codes in this frame. Codes enter
  as a learnt embedding plus DAC's own unit-norm codebook vector. Big, float and slow next to a real cx2 model:
  it's an estimate of what context modelling can reach, not the coder. A `--no-cross` run leaves the other
  channel out.

INTEGRATION.md M5: build the cx2 port only if this shows at least about 5%.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

V = 1024
NQ = 9
PAD = V  # the code id for frames before the start


def load_units(path: Path) -> torch.Tensor:
    """DAC's unit-norm codebook vectors, `[9, 1024, 8]`, from the encoder's safetensors file."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    out = []
    for q in range(NQ):
        meta = header[f"quant{q}.codebook_unit"]
        assert meta["dtype"] == "F32" and meta["shape"] == [V, 8]
        a, b = meta["data_offsets"]
        out.append(np.frombuffer(raw[8 + n + a : 8 + n + b], dtype="<f4").reshape(V, 8))
    return torch.from_numpy(np.stack(out).copy())


def load_split(root: Path, split: str):
    """Streams of one split: a list of clips, each a list of channel arrays [frames, 9] (int64)."""
    clips = []
    for side in sorted((root / split).glob("*.json")):
        meta = json.loads(side.read_text())
        if len(meta["codebooks"]) == 1:
            chans = [np.load(side.with_suffix(".npy"))]
        else:
            chans = [np.load(side.with_suffix(f".{c}.npy")) for c in range(len(meta["codebooks"]))]
        if any(c.shape[1] != NQ for c in chans):
            continue  # the probe wants all 9 codebooks
        clips.append([c.astype(np.int64) for c in chans])
    return clips


def unigram(fit):
    p = np.zeros((NQ, V))
    for q in range(NQ):
        c = np.bincount(fit[:, q], minlength=V).astype(np.float64)
        p[q] = (c + 0.5) / (c.sum() + 0.5 * V)
    return p


def bigram_bits(ctx_fit, y_fit, ctx_test, y_test, p0, disc=0.75):
    """Held-out bits per symbol of an interpolated absolute-discounting bigram p(y | ctx)."""
    tab = np.bincount(ctx_fit * V + y_fit, minlength=V * V).reshape(V, V).astype(np.float64)
    na = tab.sum(axis=1)
    n1 = (tab > 0).sum(axis=1)
    c = tab[ctx_test, y_test]
    n = na[ctx_test]
    lam = np.where(n > 0, disc * n1[ctx_test] / np.maximum(n, 1), 1.0)
    p = np.where(n > 0, np.maximum(c - disc, 0) / np.maximum(n, 1), 0.0) + lam * p0[y_test]
    return float(-np.log2(p).mean())


def table_models(fit_clips, test_clips):
    def stack(clips, shift_frame):
        ys, ctx = [], []
        for clip in clips:
            for ch in clip:
                ys.append(ch)
                prev = np.vstack([np.full((1, NQ), 0), ch[:-1]]) if shift_frame else None
                ctx.append(prev)
        return np.vstack(ys), (np.vstack(ctx) if shift_frame else None)

    fit, fit_prev = stack(fit_clips, True)
    test, test_prev = stack(test_clips, True)
    p = unigram(fit)
    out = {"unigram": [], "bigram_prev_frame": [], "bigram_prev_codebook": []}
    for q in range(NQ):
        out["unigram"].append(float(-np.log2(p[q][test[:, q]]).mean()))
        out["bigram_prev_frame"].append(bigram_bits(fit_prev[:, q], fit[:, q], test_prev[:, q], test[:, q], p[q]))
        if q == 0:
            out["bigram_prev_codebook"].append(out["unigram"][0])
        else:
            out["bigram_prev_codebook"].append(bigram_bits(fit[:, q - 1], fit[:, q], test[:, q - 1], test[:, q], p[q]))
    return out, len(fit), len(test)


class Frames(torch.utils.data.Dataset):
    """One item per (stream, frame): history [H, 9], current [9], other channel's current [9] (or PAD), has_other."""

    def __init__(self, clips, history, cross):
        self.history = history
        self.items = []  # (padded codes, other padded codes or None, frame)
        self.streams = []
        for clip in clips:
            for c, ch in enumerate(clip):
                pad = np.vstack([np.full((history, NQ), PAD), ch])
                other = None
                if cross and c == 1:
                    other = clip[0][: len(ch)]
                self.streams.append((pad, other))
        self.index = [(s, t) for s, (pad, _) in enumerate(self.streams) for t in range(len(pad) - history)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, t = self.index[i]
        pad, other = self.streams[s]
        hist = pad[t : t + self.history]
        cur = pad[t + self.history]
        if other is not None:
            oth, has = other[t], 1.0
        else:
            oth, has = np.full(NQ, PAD), 0.0
        return torch.from_numpy(hist), torch.from_numpy(cur), torch.from_numpy(np.asarray(oth)), torch.tensor(has)


class Net(nn.Module):
    def __init__(self, unit, history, emb=16, width=512):
        super().__init__()
        self.learnt = nn.Embedding(NQ * (V + 1), emb)
        fixed = torch.zeros(NQ * (V + 1), 8)
        for q in range(NQ):
            fixed[q * (V + 1) : q * (V + 1) + V] = unit[q]
        self.register_buffer("fixed", fixed)
        self.register_buffer("offset", torch.arange(NQ) * (V + 1))
        d = emb + 8
        self.trunk = nn.Sequential(nn.Linear(history * NQ * d, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(width + 2 * NQ * d + NQ + 1, width), nn.GELU(), nn.Linear(width, V))
        # mask[k, j] = 1 when codebook j is already coded before codebook k (j < k)
        self.register_buffer("mask", torch.tril(torch.ones(NQ, NQ), diagonal=-1))

    def embed(self, codes):
        ids = codes + self.offset
        return torch.cat([self.learnt(ids), self.fixed[ids]], dim=-1)

    def forward(self, hist, cur, oth, has):
        b = hist.shape[0]
        h = self.trunk(self.embed(hist).reshape(b, -1))
        e_cur = self.embed(cur)  # [b, 9, d]
        e_oth = self.embed(oth).reshape(b, 1, -1).expand(b, NQ, -1)
        masked = (self.mask[None, :, :, None] * e_cur[:, None, :, :]).reshape(b, NQ, -1)  # [b, k, 9*d]
        k1 = torch.eye(NQ, device=hist.device)[None].expand(b, NQ, NQ)
        x = torch.cat([h[:, None].expand(b, NQ, -1), masked, e_oth, k1, has[:, None, None].expand(b, NQ, 1)], dim=-1)
        return self.head(x)  # [b, 9, V] logits for each codebook


def evaluate(net, loader):
    net.eval()
    nll = torch.zeros(NQ, dtype=torch.float64)
    nll_other = torch.zeros(NQ, dtype=torch.float64)
    n = 0
    n_other = 0
    with torch.no_grad():
        for hist, cur, oth, has in loader:
            logits = net(hist, cur, oth, has.float())
            lp = torch.log_softmax(logits, -1).gather(-1, cur[..., None])[..., 0]  # [b, 9]
            nll -= lp.sum(0).double()
            nll_other -= (lp * has[:, None]).sum(0).double()
            n += len(cur)
            n_other += int(has.sum())
    net.train()
    bits = (nll / n / math.log(2)).tolist()
    bits_second = (nll_other / max(n_other, 1) / math.log(2)).tolist() if n_other else None
    return bits, bits_second


def train_net(fit_clips, test_clips, unit, args, cross):
    fit = Frames(fit_clips, args.history, cross)
    test = Frames(test_clips, args.history, cross)
    torch.manual_seed(0)
    net = Net(unit, args.history)
    loader = torch.utils.data.DataLoader(fit, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=2048, num_workers=0)
    steps = args.epochs * len(loader)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    start = time.time()
    step = 0
    for epoch in range(args.epochs):
        for hist, cur, oth, has in loader:
            logits = net(hist, cur, oth, has.float())
            loss = nn.functional.cross_entropy(logits.reshape(-1, V), cur.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"  step {step}/{steps}: {loss.item() / math.log(2):.3f} bits/code, {time.time() - start:.0f} s", flush=True)
            if time.time() - start > args.max_minutes * 60:
                print("  time limit reached; stopping early", flush=True)
                break
        else:
            continue
        break
    bits, second = evaluate(net, test_loader)
    return {"bits": bits, "bits_second_channel": second, "steps": step, "params": sum(p.numel() for p in net.parameters())}


def summary(bits):
    mean = float(np.mean(bits))
    return f"{mean:.3f} bits/code ({100 * (1 - mean / 10):.1f}% under 10), per codebook " + " ".join(f"{b:.2f}" for b in bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-minutes", type=float, default=90, help="per net; stops training early past this")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    fit_clips = load_split(args.codes, "fit")
    test_clips = load_split(args.codes, "test")
    print(f"fit {len(fit_clips)} clips, test {len(test_clips)} clips", flush=True)
    tables, n_fit, n_test = table_models(fit_clips, test_clips)
    result = {"fit_frames": n_fit, "test_frames": n_test, **{k: v for k, v in tables.items()}}
    for k, v in tables.items():
        print(f"{k:22s} {summary(v)}", flush=True)

    if not args.no_net:
        unit = load_units(args.weights / "dac44_encoder.safetensors")
        for cross in (True, False):
            name = "net" if cross else "net_no_cross"
            print(f"training {name} ...", flush=True)
            r = train_net(fit_clips, test_clips, unit, args, cross)
            result[name] = r
            print(f"{name:22s} {summary(r['bits'])}", flush=True)
            if r["bits_second_channel"]:
                print(f"{'  second channel':22s} {summary(r['bits_second_channel'])}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
