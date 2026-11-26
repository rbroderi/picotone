"""The HQ cx2 net: trains it on dumps of HQ (DAC 44 kHz) codes, then exports its integer table.

    cd train
    uv run --extra cpu python hq_net.py train --fit ../work/dac_codes/fit ../work/dac_codes/fit2 \
        --test ../work/dac_codes/test --weights ../weights --out out/hq_net.pt
    uv run --extra cpu python hq_net.py export --ckpt out/hq_net.pt --test ../work/dac_codes/test \
        --weights ../weights --table ../models/cx2_dac44.ecxm --fixture ../tests/data/cx2_dac44_fixture.json

The code dumps come from `tools/dac_codes.py` (or `picotone dev codes`): per clip one `[frames, 9]` u16 `.npy`
per coded channel and a `.json` sidecar with the layout. Nothing here depends on how the encoder made the codes,
only on the codebook vectors (`quantK.codebook_unit` in `dac44_quantizer.safetensors`), so a new encoder over the
same codebooks means: dump its codes, rerun `train` and `export`, and commit the new table.

The net (see `src/cx2/hq.rs` for the integer definition) predicts each code of a channel from that channel's 8
previous frames in the same page and the codebooks already coded in this frame. Every page (80 frames) is coded
on its own, so history never crosses a page start. Channels are independent; the channel's role (left/right,
mid or mono, side) selects a bias of the first layer. Training mixes the codebook count per channel (`--tiers`).

Export quantises with power-of-two scales: embeddings and activations are integers up to 1023, weights are i16
with limits that keep every i32 accumulator in range, and the shifts are stored in the table. It then emulates
the integer net and the coder's frequency table exactly and reports the coded bits per code per tier.
"""
import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

V = 1024
NQ = 9
PAD = V
LN2 = math.log(2)
DIM = 16  # embedding values per code: 8 learnt, then the 8 of DAC's unit codebook vector
LEARNT = DIM - 8
PAGE = 80
ROLES = {"lr": 0, "mid": 1, "side": 2}  # the order of the role biases in the table
LIMIT = 1023  # largest activation and embedding magnitude
BIAS_LIMIT = 1 << 29
FB = 6  # logits are in units of 2^-FB bits
FLOOR = 64
TABLE_MAGIC = b"ECXH"
TABLE_VERSION = 1


def roles_of(layout, channels):
    if layout == "Mono":
        return [ROLES["mid"]]
    if layout == "MidSide":
        return [ROLES["mid"], ROLES["side"]]
    if layout == "LeftRight":
        return [ROLES["lr"], ROLES["lr"]]
    if layout == "Parametric":
        return [ROLES["mid"]]
    raise ValueError(f"unknown layout {layout}")


def load_units(weights: Path) -> torch.Tensor:
    """DAC's unit-norm codebook vectors, `[9, 1024, 8]`."""
    raw = (weights / "dac44_quantizer.safetensors").read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8: 8 + n])
    out = []
    for q in range(NQ):
        meta = header[f"quant{q}.codebook_unit"]
        assert meta["dtype"] == "F32" and meta["shape"] == [V, 8]
        a, b = meta["data_offsets"]
        out.append(np.frombuffer(raw[8 + n + a: 8 + n + b], dtype="<f4").reshape(V, 8))
    return torch.from_numpy(np.stack(out).copy())


def load_dirs(dirs):
    """Channels of every clip in `dirs`: a list of (codes [frames, n] int64, role, n, clip name)."""
    out = []
    for d in dirs:
        for side in sorted(Path(d).glob("*.json")):
            meta = json.loads(side.read_text())
            books = meta["codebooks"]
            paths = [side.with_suffix(".npy")] if len(books) == 1 else \
                [side.with_suffix(f".{c}.npy") for c in range(len(books))]
            for codes, role, n in zip([np.load(p) for p in paths], roles_of(meta["layout"], len(books)), books):
                out.append((codes.astype(np.int64), role, n, side.stem))
    return out


class Codes:
    """All channels in one array, each preceded by `pad` rows of PAD codes; one item per (channel, frame)."""

    def __init__(self, channels, pad):
        rows, row, t, role, books = [], [], [], [], []
        at = 0
        for codes, r, n, _ in channels:
            full = np.full((len(codes), NQ), PAD)
            full[:, :n] = codes[:, :n]
            rows += [np.full((pad, NQ), PAD), full]
            at += pad
            row.append(np.arange(at, at + len(codes)))
            t.append(np.arange(len(codes)))
            role.append(np.full(len(codes), r))
            books.append(np.full(len(codes), n))
            at += len(codes)
        self.codes = torch.from_numpy(np.vstack(rows).astype(np.int64))
        self.row, self.t, self.role, self.books = (torch.from_numpy(np.concatenate(a)) for a in (row, t, role, books))

    def __len__(self):
        return len(self.row)


def batch(data, idx, history, tiers):
    """hist [B,H,9] (most recent first), cur [B,9], role [B], ncb [B]: inputs for a channel of `tiers` codebooks."""
    row, t = data.row[idx], data.t[idx]
    ncb = torch.minimum(tiers, data.books[idx])
    start = t - t % PAGE
    d = torch.arange(1, history + 1)
    hist = data.codes[row[:, None] - d[None, :]]
    q = torch.arange(NQ)
    ok = ((t[:, None] - d[None, :]) >= start[:, None])[..., None] & (q[None, None, :] < ncb[:, None, None])
    hist = torch.where(ok, hist, PAD)
    return hist, data.codes[row], data.role[idx], ncb


class Net(nn.Module):
    def __init__(self, unit, history=8, h1=512, h2=256, rank=128):
        super().__init__()
        self.history, self.h1, self.h2, self.rank = history, h1, h2, rank
        self.learnt = nn.Embedding(NQ * (V + 1), LEARNT)
        nn.init.normal_(self.learnt.weight, std=0.3)
        fixed = torch.zeros(NQ * (V + 1), 8)
        for q in range(NQ):
            fixed[q * (V + 1): q * (V + 1) + V] = unit[q]
        self.register_buffer("fixed", fixed)
        self.register_buffer("offset", torch.arange(NQ) * (V + 1))
        self.l1 = nn.Linear(history * NQ * DIM, h1)
        self.role = nn.Embedding(len(ROLES), h1)
        nn.init.zeros_(self.role.weight)
        self.l2 = nn.Linear(h1, h2)
        self.n_in = h2 + NQ * DIM
        self.A = nn.Parameter(torch.randn(NQ, self.n_in, rank) / math.sqrt(self.n_in))
        self.c = nn.Parameter(torch.zeros(NQ, rank))
        self.O = nn.Parameter(torch.randn(NQ, rank, V) / math.sqrt(rank) * 0.5)
        self.d = nn.Parameter(torch.zeros(NQ, V))
        # mask[k, j] = 1 when codebook j is coded before codebook k
        self.register_buffer("mask", torch.tril(torch.ones(NQ, NQ), diagonal=-1))

    def table(self):
        """The embedding table, [9*(V+1), DIM], with the PAD rows zero."""
        e = torch.cat([self.learnt.weight, self.fixed], dim=-1).clone()
        e[self.offset + V] = 0
        return e

    def parts(self, hist, cur, table=None):
        table = self.table() if table is None else table
        b = hist.shape[0]
        e_h = table[hist + self.offset].reshape(b, -1)
        e_cur = table[cur + self.offset]
        masked = (self.mask[None, :, :, None] * e_cur[:, None, :, :]).reshape(b, NQ, -1)
        return e_h, masked

    def forward(self, hist, cur, role):
        e_h, masked = self.parts(hist, cur)
        h = torch.relu(self.l2(torch.relu(self.l1(e_h) + self.role(role))))
        x = torch.cat([h[:, None].expand(-1, NQ, -1), masked], dim=-1)
        g = torch.relu(torch.einsum("bki,kir->bkr", x, self.A) + self.c)
        return torch.einsum("bkr,krv->bkv", g, self.O) + self.d

    def macs(self, books=NQ):
        total = self.history * NQ * DIM * self.h1 + self.h1 * self.h2
        return total + sum((self.h2 + k * DIM) * self.rank + self.rank * V for k in range(books))


def loss_terms(logits, cur, ncb):
    # Codebooks past a channel's count hold PAD; they are masked out, the clamp only keeps the gather legal.
    lp = torch.log_softmax(logits, -1).gather(-1, cur.clamp(max=V - 1)[..., None])[..., 0]
    m = (torch.arange(NQ)[None, :] < ncb[:, None]).float()
    return lp, m


def evaluate(net, data, history, tier, role=None):
    """Bits per code over the coded codebooks of every test channel cut to `tier` codebooks."""
    net.eval()
    nll = torch.zeros(NQ, dtype=torch.float64)
    cnt = torch.zeros(NQ, dtype=torch.float64)
    sel = torch.arange(len(data)) if role is None else torch.nonzero(data.role == role)[:, 0]
    with torch.no_grad():
        for s in range(0, len(sel), 4096):
            idx = sel[s: s + 4096]
            hist, cur, rl, ncb = batch(data, idx, history, torch.full((len(idx),), tier))
            lp, m = loss_terms(net(hist, cur, rl), cur, ncb)
            nll -= (lp * m).sum(0).double()
            cnt += m.sum(0).double()
    net.train()
    return float(nll.sum() / cnt.sum().clamp(min=1) / LN2), [round(float(v), 3) for v in (nll / cnt.clamp(min=1) / LN2)[:tier]]


def cmd_train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    unit = load_units(args.weights)
    fit = Codes(load_dirs(args.fit), args.history)
    test = Codes(load_dirs(args.test), args.history) if args.test else None
    print(f"fit {len(fit)} channel-frames ({len(fit) / 86.13 / 3600:.2f} channel-hours)"
          + (f", test {len(test)}" if test else ""), flush=True)
    net = Net(unit, args.history, args.h1, args.h2, args.rank)
    tiers = torch.arange(1, NQ + 1)
    probs = torch.tensor([float(v) for v in args.tier_probs.split(",")])
    assert len(probs) == NQ
    gen = torch.Generator().manual_seed(1)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=max(0.05, 3 / args.steps))
    start, run = time.time(), None
    log = open(str(args.out) + ".log", "a", encoding="utf-8")
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(fit), (args.batch,), generator=gen)
        ts = tiers[torch.multinomial(probs, args.batch, replacement=True, generator=gen)]
        hist, cur, role, ncb = batch(fit, idx, args.history, ts)
        lp, m = loss_terms(net(hist, cur, role), cur, ncb)
        loss = -(lp * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        bits = loss.item() / LN2
        run = bits if run is None else 0.98 * run + 0.02 * bits
        if step % 250 == 0:
            print(f"  step {step}/{args.steps}: {run:.3f} bits/code, {time.time() - start:.0f} s", flush=True)
            log.write(json.dumps({"step": step, "train_bits": run, "seconds": time.time() - start}) + "\n")
            log.flush()
        if time.time() - start > args.max_minutes * 60:
            print("  time limit reached; stopping early", flush=True)
            break
    sd = net.state_dict()
    sd["config"] = torch.tensor([args.history, args.h1, args.h2, args.rank])
    torch.save(sd, args.out)
    if test:
        report = {}
        for tier in range(NQ, 0, -1):
            mean, per = evaluate(net, test, args.history, tier)
            report[f"q{tier}"] = {"bits": mean, "per_codebook": per}
            print(f"  q{tier}: {mean:.4f} bits/code, per codebook {' '.join(f'{v:.2f}' for v in per)}", flush=True)
        for name, r in ROLES.items():
            for tier in (9, 3):
                mean, _ = evaluate(net, test, args.history, tier, r)
                report[f"{name}_q{tier}"] = mean
                print(f"  role {name} q{tier}: {mean:.4f} bits/code", flush=True)
        Path(str(args.out) + ".json").write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------------------------- export
def load_net(path, weights):
    sd = torch.load(path, weights_only=True)
    history, h1, h2, rank = (int(v) for v in sd.pop("config"))
    net = Net(load_units(weights), history, h1, h2, rank)
    net.load_state_dict(sd)
    net.eval()
    return net


def hidden_limit(n_in):
    """Largest weight of a layer with `n_in` inputs and a bias: n_in * w * 1023 + 2^29 stays inside an i32."""
    return min(32767, (2 ** 31 - 1 - BIAS_LIMIT) // (n_in * LIMIT))


def out_limit(n_in):
    """Largest weight of an output layer (no bias inside the accumulator)."""
    return min(32767, (2 ** 31 - 1) // (n_in * LIMIT))


def shift_for(maxval, limit):
    """Largest s with maxval * 2^s <= limit."""
    return int(math.floor(math.log2(limit / max(float(maxval), 1e-12))))


def act_shift(act):
    flat = act.flatten().float()
    if len(flat) > 4_000_000:
        flat = flat[torch.randperm(len(flat), generator=torch.Generator().manual_seed(0))[:4_000_000]]
    return shift_for(torch.quantile(flat, 0.9999).item(), LIMIT)


def rnd(x):
    return torch.round(x.double())


def quantise(net, data, items=20000):
    """Integer tables. Scales are powers of two, calibrated on `items` frames of `data`."""
    idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(2))[:items]
    hist, cur, role, ncb = batch(data, idx, net.history, torch.full((len(idx),), NQ))
    with torch.no_grad():
        e_h, masked = net.parts(hist, cur)
        a1 = torch.relu(net.l1(e_h) + net.role(role))
        a2 = torch.relu(net.l2(a1))
        x = torch.cat([a2[:, None].expand(-1, NQ, -1), masked], dim=-1)
        g = torch.relu(torch.einsum("bki,kir->bkr", x, net.A) + net.c)
        tab = net.table().double()
        se = shift_for(tab.abs().max(), LIMIT)
        q = {"emb": rnd(tab * 2 ** se), "emb_shift": se}
        n1 = net.l1.in_features
        sw1 = shift_for(net.l1.weight.abs().max(), hidden_limit(n1))
        s_a1 = act_shift(a1)
        q["w1"] = rnd(net.l1.weight * 2 ** sw1)
        q["b1"] = rnd(net.l1.bias * 2 ** (se + sw1))
        q["role"] = rnd(net.role.weight * 2 ** (se + sw1))
        q["s1"] = se + sw1 - s_a1
        sw2 = shift_for(net.l2.weight.abs().max(), hidden_limit(net.h1))
        s_a2 = act_shift(a2)
        q["w2"] = rnd(net.l2.weight * 2 ** sw2)
        q["b2"] = rnd(net.l2.bias * 2 ** (s_a1 + sw2))
        q["s2"] = s_a1 + sw2 - s_a2
        q["heads"] = []
        for k in range(NQ):
            n_in = net.h2 + k * DIM
            lim = hidden_limit(max(n_in, 1))
            A = net.A[k, : n_in].double()
            # Trunk columns in units 2^sw, embedding columns in 2^(sw + s_a2 - se): both products in 2^(s_a2 + sw).
            sw = shift_for(A[: net.h2].abs().max(), lim)
            if k:
                sw = min(sw, shift_for(A[net.h2:].abs().max(), lim) - (s_a2 - se))
            Ai = torch.cat([rnd(A[: net.h2] * 2 ** sw), rnd(A[net.h2:] * 2 ** (sw + s_a2 - se))])
            s_g = act_shift(g[:, k])
            O = net.O[k].double() * (2 ** FB / LN2)  # logits in 2^-FB bit units
            so = shift_for(O.abs().max() / 2 ** s_g, out_limit(net.rank))
            q["heads"].append({
                "a": Ai, "c": rnd(net.c[k] * 2 ** (s_a2 + sw)), "sa": s_a2 + sw - s_g,
                "o": rnd(O * 2 ** (so - s_g)), "so": so, "d": rnd(net.d[k].double() * 2 ** FB / LN2),
            })
    for k, h in enumerate(q["heads"]):
        assert h["so"] >= 0 and h["sa"] >= 0, (k, h["so"], h["sa"])
        assert h["a"].abs().max() <= hidden_limit(max(net.h2 + k * DIM, 1)) and h["o"].abs().max() <= out_limit(net.rank)
    assert q["s1"] >= 0 and q["s2"] >= 0
    assert (q["b1"].abs()[None] + q["role"].abs()).max() <= BIAS_LIMIT
    return q


def int_logits(q, net, hist, cur, role):
    """The integer net, exactly as `src/cx2/hq.rs` computes it (float64 holds every value exactly)."""
    b = hist.shape[0]
    emb = q["emb"]
    x = emb[hist + net.offset].reshape(b, -1)
    acc = x @ q["w1"].T + q["b1"] + q["role"][role]
    h1 = torch.clamp(torch.floor(acc / 2 ** q["s1"]), 0, LIMIT)
    h2 = torch.clamp(torch.floor((h1 @ q["w2"].T + q["b2"]) / 2 ** q["s2"]), 0, LIMIT)
    e_cur = emb[cur + net.offset]  # [b, 9, DIM]
    out = torch.zeros(b, NQ, V, dtype=torch.float64)
    for k, h in enumerate(q["heads"]):
        u = torch.cat([h2, e_cur[:, :k].reshape(b, -1)], dim=-1)
        g = torch.clamp(torch.floor((u @ h["a"] + h["c"]) / 2 ** h["sa"]), 0, LIMIT)
        out[:, k] = h["d"] + torch.floor((g @ h["o"]) / 2 ** h["so"])
    return out


def exp_table():
    return [int(round((1 << 24) * 2.0 ** (-i / (1 << FB)))) for i in range(1 << FB)]


def coded_bits(logits, y):
    """Bits the range coder spends on `y` under integer `logits` [N, V], with the coder's frequency table."""
    exp = torch.tensor(exp_table(), dtype=torch.float64)
    below = (logits.max(-1, keepdim=True).values - logits).long()
    below = below.clamp(max=32 * (1 << FB) - 1)
    freq = torch.floor(exp[below % (1 << FB)] / 2.0 ** (below // (1 << FB))) + FLOOR
    return -(torch.log2(freq.gather(1, y[:, None])[:, 0] / freq.sum(-1)))


def table_bytes(q, net):
    def i16(t):
        a = t.cpu().numpy()
        assert np.abs(a).max(initial=0) <= 32767
        return a.astype("<i2").tobytes()

    def i32(t):
        a = t.cpu().numpy()
        assert np.abs(a).max(initial=0) <= BIAS_LIMIT
        return a.astype("<i4").tobytes()

    exp = exp_table()
    out = [TABLE_MAGIC, struct.pack("<9I", TABLE_VERSION, NQ, V, DIM, net.history, net.h1, net.h2, net.rank,
                                    len(ROLES)),
           struct.pack("<2I", FB, FLOOR), struct.pack(f"<{len(exp)}I", *exp)]
    emb = q["emb"].reshape(NQ, V + 1, DIM)[:, :V]  # PAD rows are zero and not stored
    assert emb.abs().max() <= LIMIT
    out.append(i16(emb))
    # Matrices are stored column by column (one input at a time), as Matrix::from_columns takes them.
    out += [struct.pack("<I", q["s1"]), i16(q["w1"].T.contiguous()), i32(q["b1"]), i32(q["role"])]
    out += [struct.pack("<I", q["s2"]), i16(q["w2"].T.contiguous()), i32(q["b2"])]
    for h in q["heads"]:
        out += [struct.pack("<2I", h["sa"], h["so"]), i16(h["a"]), i32(h["c"]), i16(h["o"]), i32(h["d"])]
    return b"".join(out)


def fnv1a(data, h=0xCBF29CE484222325):
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def cmd_export(args):
    torch.set_num_threads(args.threads)
    net = load_net(args.ckpt, args.weights)
    test = Codes(load_dirs(args.test), net.history)
    q = quantise(net, test)
    blob = table_bytes(q, net)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_bytes(blob)
    print(f"wrote {args.table}: {len(blob):,} bytes, hash {fnv1a(blob):016x}", flush=True)
    # Integer against float on part of the test set, per tier, in coded bits.
    idx = torch.randperm(len(test), generator=torch.Generator().manual_seed(5))[: args.items]
    report = {}
    with torch.no_grad():
        for tier in range(NQ, 0, -1):
            hist, cur, role, ncb = batch(test, idx, net.history, torch.full((len(idx),), tier))
            lp, m = loss_terms(net(hist, cur, role), cur, ncb)
            fb = float(-(lp * m).sum() / m.sum() / LN2)
            li = int_logits(q, net, hist, cur, role)
            ib = coded_bits(li.reshape(-1, V), cur.reshape(-1)).reshape(len(idx), NQ)
            ib = float((ib * m).sum() / m.sum())
            report[f"q{tier}"] = {"float": fb, "coded": ib}
            print(f"  q{tier}: float {fb:.4f}, integer and coded {ib:.4f} bits/code ({ib - fb:+.4f})", flush=True)
    if args.fixture:
        write_fixture(args, q, net, blob)
    Path(str(args.table) + ".json").write_text(json.dumps(report, indent=1))


def write_fixture(args, q, net, blob):
    """A short stream with the integer logits' hash per code, for the Rust parity test."""
    channels = load_dirs(args.test)
    codes, role, _, name = channels[0]
    frames = 100  # two pages: 80 frames and 20
    cases = []
    for tier, r in ((9, role), (3, ROLES["side"])):
        data = Codes([(codes[:frames], r, tier, name)], net.history)
        hist, cur, rl, ncb = batch(data, torch.arange(frames), net.history, torch.full((frames,), tier))
        li = int_logits(q, net, hist, cur, rl).long()
        hashes = []
        for t in range(frames):
            for k in range(tier):
                hashes.append(f"{fnv1a(li[t, k].numpy().astype('<i4').tobytes()):016x}")
        bits = float(coded_bits(li[:, :tier].reshape(-1, V), cur[:, :tier].reshape(-1)).sum())
        cases.append({"role": int(r), "codebooks": tier, "codes": codes[:frames, :tier].reshape(-1).tolist(),
                      "logit_hashes": hashes, "coded_bits": bits})
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps({"table_hash": f"{fnv1a(blob):016x}", "frames": frames, "page": PAGE,
                                        "cases": cases}))
    print(f"wrote {args.fixture}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--fit", type=Path, nargs="+", required=True)
    t.add_argument("--test", type=Path, nargs="*", default=[])
    t.add_argument("--weights", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--history", type=int, default=8)
    t.add_argument("--h1", type=int, default=512)
    t.add_argument("--h2", type=int, default=256)
    t.add_argument("--rank", type=int, default=128)
    t.add_argument("--tier-probs", default="0.08,0.08,0.08,0.15,0.15,0.15,0,0,0.31",
                   help="probability of each codebook count 1..9 per training channel")
    t.add_argument("--steps", type=int, default=80000)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--threads", type=int, default=6)
    t.add_argument("--max-minutes", type=float, default=240)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--test", type=Path, nargs="+", required=True)
    e.add_argument("--weights", type=Path, required=True)
    e.add_argument("--table", type=Path, required=True)
    e.add_argument("--fixture", type=Path)
    e.add_argument("--items", type=int, default=20000)
    e.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_export(args)


if __name__ == "__main__":
    main()
