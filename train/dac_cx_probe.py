"""M5 stage 0: an integer-friendly cx2 net for HQ (DAC 44 kHz) codes, under the coder's real conditions.

    uv run --extra cpu python dac_cx_probe.py --codes ../work/dac_codes --weights ../weights \
        --name e1 --out out/dac_cx [--history 8] [--cross -1,0,1] [--fit-fraction 1.0]

Differences from dac_probe.py, which measured what context modelling can reach at all:

- **Pages.** cx2 resets its whole context at every Ogg page (80 frames for HQ), so training cuts the
  history at page starts. Evaluation reports pages of 80, 160 and 320 frames and no resets.
- **Tiers.** Files carry 4 to 9 codebooks per channel. Training mixes the codebook count per channel
  (`--tiers`); evaluation reports q9, q6 and q4 (bits per code over the coded codebooks only).
- **A coder-sized net.** Per-codebook embeddings (learnt, plus DAC's fixed 8-dim unit vectors), a trunk
  over `--history` past frames, and one low-rank head per codebook that also sees this frame's codes so
  far and, for the second channel, the first channel's codes at frame offsets `--cross` (within the page;
  the first channel's page is coded before the second's, so offset +1 is legal). ReLU throughout, so it
  quantises like cx2's EnCodec net (`src/cx2/net.rs`).
- **Integer check.** `--quant` emulates the integer net (i16 weights, activations 0..1023, i32-safe
  accumulators, arithmetic right shifts) on part of the test set and reports the float/int gap.
- **Other experts.** `--experts` fits additive tables (recency in the page, in-page pair counts, two
  "geo" similarities) on top of the frozen net, two-fold across test pages, and reports what each adds.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import dac_probe

V = 1024
NQ = 9
PAD = V
LN2 = math.log(2)


# ----------------------------------------------------------------------------------------------- data
class Codes:
    """All streams of a split in one array, each stream preceded by `pad` rows of PAD codes."""

    def __init__(self, clips, pad):
        rows, items = [], []
        at = 0
        for ci, clip in enumerate(clips):
            starts = []
            for ch in clip:
                rows.append(np.full((pad, NQ), PAD))
                at += pad
                starts.append(at)
                rows.append(ch)
                at += len(ch)
            for c, ch in enumerate(clip):
                other = starts[0] if (c == 1) else -1
                olen = len(clip[0]) if c == 1 else 0
                for t in range(len(ch)):
                    items.append((starts[c] + t, t, c, other, olen, ci))
        rows.append(np.full((pad, NQ), PAD))
        self.codes = torch.from_numpy(np.vstack(rows).astype(np.int64))
        it = torch.tensor(items, dtype=torch.long)
        self.row, self.t, self.chan, self.orow0, self.olen, self.clip = it.unbind(1)

    def __len__(self):
        return len(self.row)


def batch(data, idx, history, cross, page, tiers_self, tiers_other):
    """Inputs for items `idx`: hist [B,H,9], cur [B,9], oth [B,W,9], has [B], ncb [B]."""
    row, t = data.row[idx], data.t[idx]
    start = t - t % page if page else torch.zeros_like(t)
    end = start + page if page else torch.full_like(t, 1 << 40)
    d = torch.arange(1, history + 1)
    hist = data.codes[row[:, None] - d[None, :]]  # [B,H,9]
    hist = torch.where(((t[:, None] - d[None, :]) >= start[:, None])[..., None], hist, PAD)
    q = torch.arange(NQ)
    hist = torch.where(q[None, None, :] < tiers_self[:, None, None], hist, PAD)
    cur = data.codes[row]
    orow0, olen = data.orow0[idx], data.olen[idx]
    w = torch.tensor(cross, dtype=torch.long) if cross else torch.zeros(0, dtype=torch.long)
    u = t[:, None] + w[None, :]
    ok = (orow0[:, None] >= 0) & (u >= 0) & (u < olen[:, None]) & (u >= start[:, None]) & (u < end[:, None])
    oth = data.codes[torch.where(ok, orow0[:, None] + u, 0)]
    oth = torch.where(ok[..., None] & (q[None, None, :] < tiers_other[:, None, None]), oth, PAD)
    has = (orow0 >= 0).float()
    return hist, cur, oth, has, tiers_self


# ---------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, unit, history, cross, emb=8, h1=512, h2=256, rank=128):
        super().__init__()
        self.history, self.cross, self.emb_dim = history, len(cross), emb + 8
        self.learnt = nn.Embedding(NQ * (V + 1), emb)
        nn.init.normal_(self.learnt.weight, std=0.3)
        fixed = torch.zeros(NQ * (V + 1), 8)
        for q in range(NQ):
            fixed[q * (V + 1): q * (V + 1) + V] = unit[q]
        self.register_buffer("fixed", fixed)
        self.register_buffer("offset", torch.arange(NQ) * (V + 1))
        d = self.emb_dim
        self.l1 = nn.Linear(history * NQ * d, h1)
        self.l2 = nn.Linear(h1, h2)
        self.n_in = h2 + NQ * d + self.cross * NQ * d + 1
        self.A = nn.Parameter(torch.randn(NQ, self.n_in, rank) / math.sqrt(self.n_in))
        self.c = nn.Parameter(torch.zeros(NQ, rank))
        self.O = nn.Parameter(torch.randn(NQ, rank, V) / math.sqrt(rank) * 0.5)
        self.d = nn.Parameter(torch.zeros(NQ, V))
        self.register_buffer("mask", torch.tril(torch.ones(NQ, NQ), diagonal=-1))
        self.h1, self.h2, self.rank = h1, h2, rank

    def table(self):
        """The embedding table the integer coder would store: [9*(V+1), emb+8], PAD rows zero."""
        e = torch.cat([self.learnt.weight, self.fixed], dim=-1)
        pad_rows = self.offset + V
        e = e.clone()
        e[pad_rows] = 0
        return e

    def embed(self, codes, table=None):
        table = self.table() if table is None else table
        return table[codes + self.offset]

    def head_input(self, hist, cur, oth, has, table=None):
        b = hist.shape[0]
        e_h = self.embed(hist, table).reshape(b, -1)
        e_cur = self.embed(cur, table)
        masked = (self.mask[None, :, :, None] * e_cur[:, None, :, :]).reshape(b, NQ, -1)
        parts = [masked]
        if self.cross:
            parts.append(self.embed(oth, table).reshape(b, 1, -1).expand(b, NQ, -1))
        parts.append(has[:, None, None].expand(b, NQ, 1))
        return e_h, torch.cat(parts, dim=-1)

    def forward(self, hist, cur, oth, has):
        e_h, x = self.head_input(hist, cur, oth, has)
        h = torch.relu(self.l2(torch.relu(self.l1(e_h))))
        x = torch.cat([h[:, None].expand(-1, NQ, -1), x], dim=-1)
        g = torch.relu(torch.einsum("bki,kir->bkr", x, self.A) + self.c)
        return torch.einsum("bkr,krv->bkv", g, self.O) + self.d

    def macs(self, second_channel):
        """Multiply-adds per frame of one channel, skipping inputs that are always zero."""
        d = self.emb_dim
        total = self.history * NQ * d * self.h1 + self.h1 * self.h2
        for k in range(NQ):
            n = self.h2 + k * d + 1 + (self.cross * NQ * d if second_channel else 0)
            total += n * self.rank + self.rank * V
        return total

    def params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------------------------------- training
def sample_tiers(n, tiers, probs, gen):
    choice = torch.multinomial(torch.tensor(probs, dtype=torch.float), n, replacement=True, generator=gen)
    return torch.tensor(tiers)[choice]


def loss_bits(logits, cur, ncb):
    lp = torch.log_softmax(logits, -1).gather(-1, cur[..., None])[..., 0]  # [B,9]
    m = (torch.arange(NQ)[None, :] < ncb[:, None]).float()
    return -(lp * m).sum() / m.sum() / LN2, lp, m


def evaluate(net, data, args, page, tier, max_items=None, by_codebook=True):
    net.eval()
    n = len(data) if max_items is None else min(len(data), max_items)
    nll = torch.zeros(NQ, dtype=torch.float64)
    cnt = torch.zeros(NQ, dtype=torch.float64)
    nll2 = torch.zeros(NQ, dtype=torch.float64)
    cnt2 = torch.zeros(NQ, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, n, 4096):
            idx = torch.arange(s, min(n, s + 4096))
            tiers = torch.full((len(idx),), tier)
            hist, cur, oth, has, ncb = batch(data, idx, args.history, args.cross, page, tiers, tiers)
            _, lp, m = loss_bits(net(hist, cur, oth, has), cur, ncb)
            nll -= (lp * m).sum(0).double()
            cnt += m.sum(0).double()
            sec = has[:, None] * m
            nll2 -= (lp * sec).sum(0).double()
            cnt2 += sec.sum(0).double()
    net.train()
    per = (nll / cnt.clamp(min=1) / LN2)[:tier]
    per2 = (nll2 / cnt2.clamp(min=1) / LN2)[:tier]
    return {"mean": float(nll.sum() / cnt.sum() / LN2), "per_codebook": [round(float(v), 3) for v in per],
            "second_channel_mean": float(nll2.sum() / cnt2.sum().clamp(min=1) / LN2),
            "second_channel": [round(float(v), 3) for v in per2]}


def train(net, data, args):
    gen = torch.Generator().manual_seed(1)
    steps = args.steps
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    pages = [int(p) for p in args.train_pages.split(",")]
    start = time.time()
    run = 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(len(data), (args.batch,), generator=gen)
        page = pages[step % len(pages)]
        ts = sample_tiers(args.batch, args.tiers, args.tier_probs, gen)
        to = sample_tiers(args.batch, args.tiers, args.tier_probs, gen)
        hist, cur, oth, has, ncb = batch(data, idx, args.history, args.cross, page, ts, to)
        loss, _, _ = loss_bits(net(hist, cur, oth, has), cur, ncb)
        opt.zero_grad()
        (loss * LN2).backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 250 == 0:
            print(f"  step {step}/{steps}: {run:.3f} bits/code, {time.time() - start:.0f} s", flush=True)
        if time.time() - start > args.max_minutes * 60:
            print("  time limit reached; stopping early", flush=True)
            break
    return step, time.time() - start


# ---------------------------------------------------------------------------------------- integer net
def pow2_scale(maxval, limit):
    return int(math.floor(math.log2(limit / max(maxval, 1e-12))))


def quant_check(net, data, args, items):
    """Emulates an integer version of the net (float64 holds every integer exactly here)."""
    net.eval()
    idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(2))[:items].sort().values
    tiers = torch.full((len(idx),), NQ)
    hist, cur, oth, has, ncb = batch(data, idx, args.history, args.cross, 80, tiers, tiers)
    with torch.no_grad():
        f_logits = net(hist, cur, oth, has)
        f_bits = loss_bits(f_logits, cur, ncb)[0].item()
        # float activations, for calibrating the activation scales
        e_h, x = net.head_input(hist, cur, oth, has)
        a1 = torch.relu(net.l1(e_h))
        a2 = torch.relu(net.l2(a1))
        xg = torch.cat([a2[:, None].expand(-1, NQ, -1), x], dim=-1)
        g = torch.relu(torch.einsum("bki,kir->bkr", xg, net.A) + net.c)
        D = torch.float64
        lim = 1023
        acc_limit = 2 ** 31 - 1 - 2 ** 28

        def wlim(n_in):
            return min(32767, acc_limit // (n_in * lim))

        tab = net.table().double()
        se = pow2_scale(tab.abs().max().item(), lim)
        tab_i = torch.round(tab * 2 ** se)
        info = {"emb_shift": se}

        def layer(x_i, sx, W, b, act, name, n_in):
            sw = pow2_scale(W.abs().max().item(), wlim(n_in))
            W_i = torch.round(W.double() * 2 ** sw)
            sa = pow2_scale(torch.quantile(act.flatten()[:2_000_000].float(), 0.9999).item(), lim)
            shift = sx + sw - sa
            b_i = torch.round(b.double() * 2 ** (sx + sw))
            info[name] = {"w_shift": sw, "act_shift": sa, "shift": shift, "w_limit": wlim(n_in),
                          "max_bias": float(b_i.abs().max())}
            return W_i, b_i, shift, sa

        e_hi = tab_i[hist + net.offset].reshape(len(idx), -1)
        W1, b1, s1, a1s = layer(None, se, net.l1.weight, net.l1.bias, a1, "l1", net.l1.in_features)
        h1 = torch.clamp(torch.floor((e_hi @ W1.T + b1) / 2 ** s1), 0, lim)
        W2, b2, s2, a2s = layer(None, a1s, net.l2.weight, net.l2.bias, a2, "l2", net.h1)
        h2 = torch.clamp(torch.floor((h1 @ W2.T + b2) / 2 ** s2), 0, lim)
        # head inputs: trunk output in units 2^a2s, embeddings in 2^se; rescale the embeddings to the trunk's
        # units with a shift so one matrix takes both (a2s >= se is checked).
        _, x_i = net.head_input(hist, cur, oth, has, tab_i)
        x_i = x_i.clone()
        x_i[..., -1] = has[:, None].double() * 2 ** se
        # One accumulator per head: the trunk columns' weights are in units 2^sw_h, the embedding columns'
        # in 2^(sw_h + a2s - se), so both products land in units 2^(a2s + sw_h).
        logits_i = torch.zeros(len(idx), NQ, V, dtype=D)
        info["heads"] = []
        wl = wlim(net.n_in)
        for k in range(NQ):
            A = net.A[k].double()
            sw_h = min(pow2_scale(A[: net.h2].abs().max().item(), wl),
                       pow2_scale(A[net.h2:].abs().max().item(), wl) - (a2s - se))
            sw_e = sw_h + a2s - se
            Ah = torch.round(A[: net.h2] * 2 ** sw_h)
            Ae = torch.round(A[net.h2:] * 2 ** sw_e)
            sg = pow2_scale(torch.quantile(g[:, k].flatten().float(), 0.9999).item(), lim)
            sh = a2s + sw_h - sg
            c_i = torch.round(net.c[k].double() * 2 ** (a2s + sw_h))
            acc = h2 @ Ah + x_i[:, k] @ Ae + c_i
            gi = torch.clamp(torch.floor(acc / 2 ** sh), 0, lim)
            O = net.O[k].double()
            so = pow2_scale(O.abs().max().item(), 16383)
            O_i = torch.round(O * 2 ** so)
            d_i = torch.round(net.d[k].double() * 2 ** (sg + so))
            logits_i[:, k] = (gi @ O_i + d_i) / 2 ** (sg + so)
            info["heads"].append({"sw_h": sw_h, "sw_e": sw_e, "act_shift": sg, "shift": sh, "o_shift": so,
                                  "max_acc": float(acc.abs().max())})
        i_bits = loss_bits(logits_i.float(), cur, ncb)[0].item()
    net.train()
    return {"items": items, "float_bits": f_bits, "int_bits": i_bits, "gap": i_bits - f_bits, "scales": info}


# --------------------------------------------------------------------------------------- other experts
REC_EDGES = [1, 2, 3, 5, 9, 17, 33]  # frames since last seen in the page: classes 1..7 by these lower bounds
SIZES = {"rec": 8, "pair": 5, "geo_prev": 17, "geo_resid": 17}


def load_proj(path):
    """8x8 maps between DAC's codebook spaces: e_q = in_proj_q(latent) for a latent made of codebook vectors."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8: 8 + n])

    def get(name):
        meta = header[name]
        a, b = meta["data_offsets"]
        return np.frombuffer(raw[8 + n + a: 8 + n + b], dtype="<f4").reshape(meta["shape"]).astype(np.float64)

    W_in = [get(f"quant{q}.in_proj.weight")[:, :, 0] for q in range(NQ)]  # [8,1024]
    b_in = [get(f"quant{q}.in_proj.bias") for q in range(NQ)]
    W_out = [get(f"quant{q}.out_proj.weight")[:, :, 0] for q in range(NQ)]  # [1024,8]
    b_out = [get(f"quant{q}.out_proj.bias") for q in range(NQ)]
    cb = np.stack([get(f"quant{q}.codebook") for q in range(NQ)])  # [9,V,8]
    M = np.array([[W_in[q] @ W_out[qq] for qq in range(NQ)] for q in range(NQ)])
    # Residual seen by codebook q, predicted from the previous frame: its whole latent minus this frame's
    # codebooks < q; the out_proj biases of codebooks >= q survive.
    base = np.array([b_in[q] + W_in[q] @ sum(b_out[qq] for qq in range(q, NQ)) for q in range(NQ)])
    return M, base, cb


def experts_check(net, test, test_clips, unit, args, page=80):
    """Additive per-class tables on top of the frozen net, fitted two-fold across sampled test pages.

    For a table T over candidate classes c(j), p'(j) = p(j) exp(T[c(j)]) / sum_c exp(T[c]) S_c with
    S_c = sum of p(j) over class c, so only the per-class sums S need storing.
    """
    M, base, cb = load_proj(args.weights / "dac44_encoder.safetensors")
    rng = np.random.default_rng(3)
    cand = [(ci, c, p0) for ci, clip in enumerate(test_clips) for c, ch in enumerate(clip)
            for p0 in range(0, len(ch) - page + 1, page)]
    pick = [cand[i] for i in sorted(rng.choice(len(cand), min(args.expert_pages, len(cand)), replace=False))]
    key = (test.clip * 2 + test.chan) * 10_000_000 + test.t
    order = torch.argsort(key)
    sorted_key = key[order]
    un = unit.numpy().astype(np.float64)
    combos = {"rec": ["rec"], "pair": ["pair"], "geo_prev": ["geo_prev"], "geo_resid": ["geo_resid"],
              "rec+pair+geo_resid": ["rec", "pair", "geo_resid"]}
    S = {name: [] for name in combos}
    target = {name: [] for name in combos}
    lp_y, qs, pid = [], [], []
    net.eval()
    for pi, (ci, c, p0) in enumerate(pick):
        ch = test_clips[ci][c]
        want = (ci * 2 + c) * 10_000_000 + torch.arange(p0, p0 + page)
        items = order[torch.searchsorted(sorted_key, want)]
        assert torch.equal(key[items], want)
        tiers = torch.full((page,), NQ)
        with torch.no_grad():
            hist, cur, oth, has, _ = batch(test, items, args.history, args.cross, page, tiers, tiers)
            prob = torch.softmax(net(hist, cur, oth, has).double(), -1).numpy()  # [page,9,V]
        last = np.full((NQ, V), -1)
        counts = {}
        for i, t in enumerate(range(p0, p0 + page)):
            for q in range(NQ):
                y = int(ch[t, q])
                cls = {}
                seen = last[q] >= 0
                d = t - last[q]
                rec = np.zeros(V, dtype=np.int64)
                for r, edge in enumerate(REC_EDGES):
                    rec[seen & (d >= edge)] = r + 1
                cls["rec"] = rec
                pair = np.zeros(V, dtype=np.int64)
                g1 = np.full(V, 16)
                g2 = np.full(V, 16)
                if t > p0:
                    ctx = (q, int(ch[t - 1, q]))
                    if ctx in counts:
                        pair = np.minimum(counts[ctx], 3) + 1
                    g1 = np.clip(((un[q] @ un[q, ch[t - 1, q]] + 1) * 8).astype(np.int64), 0, 15)
                    e = base[q].copy()
                    for qq in range(NQ):
                        e += M[q, qq] @ cb[qq, ch[t - 1, qq]]
                    for qq in range(q):
                        e -= M[q, qq] @ cb[qq, ch[t, qq]]
                    g2 = np.clip(((un[q] @ (e / (np.linalg.norm(e) + 1e-12)) + 1) * 8).astype(np.int64), 0, 15)
                cls["pair"], cls["geo_prev"], cls["geo_resid"] = pair, g1, g2
                p = prob[i, q]
                for name, parts in combos.items():
                    joint = np.zeros(V, dtype=np.int64)
                    size = 1
                    for part in parts:
                        joint = joint * SIZES[part] + cls[part]
                        size *= SIZES[part]
                    S[name].append(np.bincount(joint, weights=p, minlength=size))
                    target[name].append(int(joint[y]))
                lp_y.append(math.log(p[y]))
                qs.append(q)
                pid.append(pi)
            for q in range(NQ):
                last[q, ch[t, q]] = t
                if t > p0:
                    ctx = (q, int(ch[t - 1, q]))
                    counts.setdefault(ctx, np.zeros(V, dtype=np.int64))[ch[t, q]] += 1
    net.train()
    lp_y = torch.tensor(lp_y, dtype=torch.float64)
    q = torch.tensor(qs)
    fold = torch.tensor(pid) % 2
    base_bits = float(-lp_y.mean() / LN2)
    out = {"codes": len(lp_y), "pages": len(pick), "net_bits": base_bits}
    print(f"  experts: {len(pick)} pages, {len(lp_y)} codes, net alone {base_bits:.4f} bits/code", flush=True)
    for name, parts in combos.items():
        s = torch.tensor(np.stack(S[name]))
        y = torch.tensor(target[name])
        shape = [SIZES[p] for p in parts]
        total = torch.zeros(NQ, dtype=torch.float64)
        cnt = torch.zeros(NQ, dtype=torch.float64)
        for f in (0, 1):
            tabs = [torch.zeros(NQ, n, dtype=torch.float64, requires_grad=True) for n in shape]

            def joint_table(qv):
                t = torch.zeros(len(qv), 1, dtype=torch.float64)
                for tab in tabs:
                    t = (t[:, :, None] + tab[qv][:, None, :]).reshape(len(qv), -1)
                return t  # [N, prod(shape)]

            def nll(mask):
                tq = joint_table(q[mask])
                logz = torch.logsumexp(tq + torch.log(s[mask].clamp(min=1e-300)), -1)
                return -(lp_y[mask] + tq.gather(1, y[mask][:, None])[:, 0] - logz)

            opt = torch.optim.LBFGS(tabs, max_iter=100, line_search_fn="strong_wolfe")
            tr, te = fold != f, fold == f

            def closure():
                opt.zero_grad()
                loss = nll(tr).mean() + 1e-4 * sum((tab ** 2).sum() for tab in tabs)
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                v = nll(te)
                total.index_add_(0, q[te], v)
                cnt.index_add_(0, q[te], torch.ones(int(te.sum()), dtype=torch.float64))
        bits = float(total.sum() / cnt.sum() / LN2)
        per = [round(float(a / b / LN2), 3) for a, b in zip(total, cnt)]
        out[name] = {"bits": bits, "gain": base_bits - bits, "per_codebook": per}
        print(f"  experts {name:20s} {bits:.4f} bits/code (gain {base_bits - bits:+.4f}), per codebook "
              + " ".join(f"{v:.2f}" for v in per), flush=True)
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--cross", default="-1,0,1", help="other-channel frame offsets; 'none' for no cross input")
    ap.add_argument("--emb", type=int, default=8, help="learnt embedding size (8 fixed dims are added)")
    ap.add_argument("--h1", type=int, default=512)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--tiers", default="9,6,5,4")
    ap.add_argument("--tier-probs", default="0.4,0.2,0.2,0.2")
    ap.add_argument("--train-pages", default="80", help="page lengths to train with, cycled per step (0 = none)")
    ap.add_argument("--fit-fraction", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--quant", type=int, default=0, help="test items for the integer check (0: skip)")
    ap.add_argument("--experts", action="store_true")
    ap.add_argument("--expert-pages", type=int, default=60)
    ap.add_argument("--load", action="store_true", help="load <out>/<name>.pt instead of training")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.cross = [] if args.cross == "none" else [int(v) for v in args.cross.split(",")]
    args.tiers = [int(v) for v in args.tiers.split(",")]
    args.tier_probs = [float(v) for v in args.tier_probs.split(",")]

    fit_clips = dac_probe.load_split(args.codes, "fit")
    test_clips = dac_probe.load_split(args.codes, "test")
    if args.fit_fraction < 1:
        keep = np.random.default_rng(0).permutation(len(fit_clips))[: int(round(len(fit_clips) * args.fit_fraction))]
        fit_clips = [fit_clips[i] for i in sorted(keep)]
    fit = Codes(fit_clips, args.history)
    test = Codes(test_clips, args.history)
    print(f"{args.name}: fit {len(fit_clips)} clips / {len(fit)} channel-frames, test {len(test_clips)} / {len(test)}",
          flush=True)
    unit = dac_probe.load_units(args.weights / "dac44_encoder.safetensors")
    net = Net(unit, args.history, args.cross, args.emb, args.h1, args.h2, args.rank)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / f"{args.name}.pt"
    result = {"name": args.name, "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "params": net.params(), "macs_per_frame_first_channel": net.macs(False),
              "macs_per_frame_second_channel": net.macs(True)}
    result["mmac_per_s_stereo"] = (net.macs(False) + net.macs(True)) * 44100 / 512 / 1e6
    print(f"  params {net.params():,}, MAC/frame {net.macs(False):,} / {net.macs(True):,}, "
          f"{result['mmac_per_s_stereo']:.0f} MMAC per stereo second", flush=True)
    if args.load:
        net.load_state_dict(torch.load(ckpt, weights_only=True))
    else:
        steps, secs = train(net, fit, args)
        result.update(steps=steps, train_seconds=secs)
        torch.save(net.state_dict(), ckpt)
    ev = {}
    for page, tier in ((80, 9), (160, 9), (320, 9), (0, 9), (80, 6), (80, 5), (80, 4), (0, 6), (0, 4)):
        r = evaluate(net, test, args, page, tier)
        ev[f"p{page}_q{tier}"] = r
        print(f"  page {page:3d} q{tier}: {r['mean']:.4f} bits/code ({100 * (1 - r['mean'] / 10):.2f}% under 10), "
              f"per codebook {' '.join(f'{v:.2f}' for v in r['per_codebook'])}; second channel "
              f"{r['second_channel_mean']:.4f}", flush=True)
    result["eval"] = ev
    if args.quant:
        r = quant_check(net, test, args, args.quant)
        result["quant"] = r
        print(f"  integer check on {r['items']} frames: float {r['float_bits']:.4f}, int {r['int_bits']:.4f}, "
              f"gap {r['gap']:+.4f} bits/code", flush=True)
    if args.experts:
        result["experts"] = experts_check(net, test, test_clips, unit, args)
    (args.out / f"{args.name}.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out / (args.name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
