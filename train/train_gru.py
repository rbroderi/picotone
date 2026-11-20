"""One-hour long-context test: does memory (a GRU or attention) on top of the 4-frame MLP help?

    uv run --extra cuda train_gru.py --data train_data_48k.npz --steps 20000 --compare

Float only; there is no integer or Rust version of the memory. The MLP net (train_net.py)
sees the previous 4 frames. The model here is that same net (train_net.Net, trunk and
heads, same sizes and weight clamps) plus a memory path that runs over the frames of a
chunk and carries state, so it can use everything since the chunk start:

  gru   u_t = LayerNorm(W x_t) + bandwidth and position embeddings, where x_t is the MLP's
        input (the previous 4 frames' sums); one GRU layer (hidden --hidden)
  attn  the same u_t, 2 causal self-attention layers that see the last --window frames

The memory's output h_t is added to the MLP's first hidden layer through a projection P
that starts at zero:

  h1 = relu8(W1 x_t + b1 + e[K] + p[pos] + P h_t),  then the MLP's h2 and heads

so at step 0 the model is exactly an MLP net and the memory can only add to it. The check
printed at the start scores it with the memory forced off against its own MLP path.

Training: --chunks chunks of --chunk consecutive frames starting on segment starts, drawn
from the whole fit set (so from many clips), with the context cut at each chunk start like
a decoder starting a page (300 frames = a music12 page). The memory runs over every frame
of each chunk; the loss is taken on --loss-frames random frames per chunk (64 x 36 = 2304
frames per step, against the MLP's batch of 2048). The MLP path has the MLP's learning
rate and schedule; the memory path has its own (--mem-lr) and gradient clipping.

--compare also trains the MLP baseline with train_net.py for the same number of steps
(batch 2048), then scores both on the same held-out chunks (all their frames, MLP context
cut at chunk starts too) and prints bits per code at K = 4, 8, 15, 16 and the verdict:
a gain under 0.2 bits per code at K = 15 means the long-context model is not worth building.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

import train_net
from train_net import LN2, PAST, Frames, Net, select_clips

HERE = os.path.dirname(os.path.abspath(__file__))
KILL_GAIN = 0.2


class SeqNet(nn.Module):
    """The MLP net (train_net.Net) plus a recurrent or attention memory added to its h1."""

    def __init__(self, arch, bandwidths, codebooks, hidden=320, h1=384, h2=256, period=150, window=32,
                 rank_low=64, rank_high=32):
        super().__init__()
        self.arch, self.window, self.hidden = arch, window, hidden
        self.mlp = Net(bandwidths, codebooks, h1, h2, rank_low, rank_high, period)
        self.bandwidths = self.mlp.bandwidths
        self.period = period
        self.mem_on = True
        self.inp = nn.Linear(128 * PAST, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.kemb = nn.Embedding(len(bandwidths), hidden)
        nn.init.zeros_(self.kemb.weight)
        if period:
            self.pemb = nn.Embedding(int(self.mlp.pos_map.max()) + 1, hidden)
            nn.init.zeros_(self.pemb.weight)
        if arch == "gru":
            self.core = nn.GRU(hidden, hidden, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden,
                                               dropout=0.0, batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.proj = nn.Linear(hidden, h1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def mlp_parameters(self):
        return list(self.mlp.parameters())

    def mem_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("mlp.")]

    def memory(self, x, kidx, pos):
        """x [B, T, 512], kidx [B], pos [B, T] -> contribution to h1 [B, T, h1]."""
        u = self.norm(self.inp(x)) + self.kemb(kidx)[:, None, :]
        if self.period:
            u = u + self.pemb(self.mlp.pos_map[pos])
        if self.arch == "gru":
            h, _ = self.core(u)
        else:
            t = x.shape[1]
            i = torch.arange(t, device=x.device)
            far = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= self.window)
            h = self.core(u, mask=torch.zeros(t, t, device=x.device).masked_fill(far, float("-inf")))
        return self.proj(h)

    def trunk(self, x, kidx, pos, sel=None):
        """x [B, T, 512], kidx [B], pos [B, T] (or None), sel [B, n] frames wanted (default:
        all) -> h2 [B, n, h2]. The memory runs over all T frames either way."""
        def pick(t):
            return t if sel is None else t.gather(1, sel[..., None].expand(-1, -1, t.shape[-1]))
        xs = pick(x)
        h = self.mlp.l1(xs) + self.mlp.kemb(kidx)[:, None, :]
        if self.period:
            ps = pos if sel is None else pos.gather(1, sel)
            h = h + self.mlp.pemb(self.mlp.pos_map[ps])
        if self.mem_on:
            h = h + pick(self.memory(x, kidx, pos))
        h = torch.clamp(h, 0, 8)
        return torch.clamp(self.mlp.l2(h), 0, 8)

    def head(self, k, h2, s):
        return self.mlp.head(k, h2, s)


def macs_per_frame(arch, hidden, h1=384, h2=256, window=32):
    """Multiply-adds of the trunk per frame (heads are the same for every model). The
    residual models are the MLP trunk plus the memory and its projection into h1."""
    mlp = 128 * PAST * h1 + h1 * h2
    if arch == "mlp":
        return mlp
    if arch == "gru":
        core = 3 * hidden * hidden * 2
    else:
        per_layer = 4 * hidden * hidden + 2 * window * hidden + 2 * 2 * hidden * hidden
        core = 2 * per_layer
    return mlp + 128 * PAST * hidden + core + hidden * h1


def head_macs_per_frame(k_total, h2=256, rank_low=64, rank_high=32):
    return sum(((h2 + 128) * r + r * 1024) for r in
               [rank_low if k < 8 else rank_high for k in range(k_total)])


def chunk_starts(lengths, chunk, period):
    """Every chunk start on a segment start whose chunk fits inside its stream."""
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    step = period or chunk
    return np.concatenate([s + np.arange(0, n - chunk + 1, step) for s, n in zip(starts, lengths)
                           if n >= chunk]).astype(np.int64)


def seq_batch(frames, starts, chunk, k_total, period):
    """Inputs of `len(starts)` chunks; the context is cut at each chunk start."""
    b = len(starts)
    offs = torch.arange(chunk, device=starts.device)
    idx = (starts[:, None] + offs).reshape(-1)
    since = offs.repeat(b)
    x, part, y = frames.batch(idx, since, k_total)
    pos = (frames.since[idx].long() % period).view(b, chunk) if period else None
    return x.view(b, chunk, -1), part.view(b, chunk, k_total, 128), y.view(b, chunk, k_total), pos


def seq_inputs(frames, starts, chunk, k_total):
    """The trunk input x [B, chunk, 512] of every frame of the chunks, context cut at the
    chunk starts; the same integers as `Frames.batch` (and `seq_batch`), but each frame's
    centroid sum is computed once instead of once per lag."""
    b = len(starts)
    rows = torch.clamp(starts[:, None] + torch.arange(-PAST, chunk, device=starts.device), min=0)
    c = frames.codes[rows][:, :, :k_total].long() + frames.offsets[:k_total]   # [B, T+4, k]
    s = frames.centroids[c].sum(2, dtype=torch.int64)                          # [B, T+4, 128]
    t = torch.arange(chunk, device=starts.device)
    past = torch.stack([s[:, PAST - lag:PAST - lag + chunk] * (t >= lag).long()[None, :, None]
                        for lag in range(1, PAST + 1)], 2)                      # [B, T, 4, 128]
    x = torch.clamp(torch.div(past, 8, rounding_mode="floor"), -1023, 1023)
    return x.reshape(b, chunk, PAST * 128).float() / 16.0


def score(model, frames, starts, chunk, k_total, period, device):
    """Mean bits per code over every frame of the given chunks."""
    total, count = 0.0, 0
    kid = model.bandwidths.index(k_total)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(starts), 32):
            st = starts[s:s + 32]
            x, part, y, pos = (t.to(device) if t is not None else None
                               for t in seq_batch(frames, st, chunk, k_total, period))
            kidx = torch.full((len(st),), kid, dtype=torch.long, device=device)
            if isinstance(model, Net):
                h2 = model.trunk(x.reshape(-1, x.shape[-1]), kidx.repeat_interleave(chunk),
                                 pos.reshape(-1) if period else None).view(len(st), chunk, -1)
            else:
                h2 = model.trunk(x, kidx, pos)
            for k in range(k_total):
                logits = model.head(k, h2, part[:, :, k]).float()
                total += float(nn.functional.cross_entropy(logits.reshape(-1, 1024), y[:, :, k].reshape(-1),
                                                           reduction="sum")) / LN2
            count += y.numel()
    model.train()
    return total / count


def score_mem_off(model, *a):
    model.mem_on = False
    try:
        return score(model, *a)
    finally:
        model.mem_on = True


def data_period(data):
    return 150 if str(data["model"]) == "encodec_48khz" else 0


def eval_sets(args, data, device):
    """[(name, frames, chunk starts)] of the held-out chunks, the same for every model."""
    centroids = data["centroids"].astype(np.int32)
    period = data_period(data)
    out = []
    eval_rng = np.random.default_rng(args.seed)
    for name in ("test_old", "test_new"):
        if f"{name}_codes" not in data:
            continue
        frames = Frames(centroids, data[f"{name}_codes"], data[f"{name}_lengths"], device)
        starts = chunk_starts(data[f"{name}_lengths"], args.chunk, period)
        pick = np.sort(eval_rng.choice(len(starts), min(len(starts), max(args.eval_frames // args.chunk, 1)),
                                       replace=False))
        out.append((name, frames, torch.from_numpy(starts[pick]).to(device)))
    return out


def train_seq(args, data, device, tests):
    centroids = data["centroids"].astype(np.int32)
    kmax = centroids.shape[0]
    bandwidths = sorted(int(k) for k in args.bandwidths.split(","))
    ks, kp = train_net.parse_k_mix(args.k_mix, bandwidths)
    period = data_period(data)
    names = json.loads(str(data["files"]))["fit"]
    lengths = data["fit_lengths"]
    keep = np.zeros(len(lengths), dtype=bool)
    keep[select_clips(names, args.fraction)] = True
    fit = Frames(centroids, data["fit_codes"], lengths, device)
    all_starts = chunk_starts(lengths, args.chunk, period)
    clip_of = np.searchsorted(np.cumsum(lengths), all_starts, side="right")
    pool = torch.from_numpy(all_starts[keep[clip_of]]).to(device)
    model = SeqNet(args.arch, bandwidths, kmax, args.hidden, period=period, window=args.window).to(device)
    n_loss = min(args.loss_frames, args.chunk)
    mlp_params, mem_params = model.mlp_parameters(), model.mem_parameters()
    opt = torch.optim.Adam([{"params": mlp_params, "lr": args.lr}, {"params": mem_params, "lr": args.mem_lr}],
                           fused=device.type == "cuda")
    # The MLP path gets train_net.py's schedule (one cycle, 5 % warm-up); the memory the same
    # shape at its own, lower peak.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.mem_lr], total_steps=args.steps,
                                                pct_start=args.warmup)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1)
    print(json.dumps({"arch": args.arch, "hidden": args.hidden, "chunk": args.chunk, "chunks": args.chunks,
                      "loss_frames_per_chunk": n_loss, "loss_frames_per_step": args.chunks * n_loss,
                      "memory_frames_per_step": args.chunks * args.chunk,
                      "fit_chunks": int(len(pool)), "fit_clips": int(keep.sum()), "steps": args.steps,
                      "lr": args.lr, "mem_lr": args.mem_lr, "clip": args.clip,
                      "trunk_macs_per_frame": macs_per_frame(args.arch, args.hidden, window=args.window),
                      "mlp_trunk_macs_per_frame": macs_per_frame("mlp", 0),
                      "head_macs_per_frame_K15": head_macs_per_frame(15)}), flush=True)

    # Sanity check: with the memory forced off the model is its MLP path, exactly.
    if tests:
        name, frames, starts = tests[0]
        kk = 15 if 15 in bandwidths else bandwidths[-1]
        st = starts[:8]
        a = score_mem_off(model, frames, st, args.chunk, kk, period, device)
        b = score(model.mlp, frames, st, args.chunk, kk, period, device)
        c = score(model, frames, st, args.chunk, kk, period, device)
        print(json.dumps({"check": f"memory off vs MLP path, {name} K={kk}, {len(st)} chunks",
                          "memory_off": round(a, 6), "mlp_path": round(b, 6), "memory_on_at_init": round(c, 6),
                          "match": abs(a - b) < 1e-4 and abs(c - b) < 1e-4}), flush=True)

    t0 = time.time()
    t_last, s_last = t0, 0
    acc = torch.zeros((), device=device)
    for step in range(1, args.steps + 1):
        k_total = int(ks[rng.choice(len(ks), p=kp)])
        st = pool[torch.randint(0, len(pool), (args.chunks,), generator=gen, device=device)]
        x = seq_inputs(fit, st, args.chunk, k_total)
        sel = torch.rand(args.chunks, args.chunk, generator=gen, device=device).argsort(1)[:, :n_loss]
        _, part, y = fit.batch((st[:, None] + sel).reshape(-1), sel.reshape(-1), k_total)
        offs = torch.arange(args.chunk, device=device)
        pos = (fit.since[st[:, None] + offs].long() % period) if period else None
        kidx = torch.full((args.chunks,), bandwidths.index(k_total), dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
            h2 = model.trunk(x, kidx, pos, sel).reshape(args.chunks * n_loss, -1)
            loss = sum(nn.functional.cross_entropy(model.head(k, h2, part[:, k]).float(), y[:, k])
                       for k in range(k_total)) / k_total
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mem_params, args.clip)
        opt.step()
        sched.step()
        model.mlp.clamp_()
        acc += loss.detach()
        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            n = step - s_last
            print(json.dumps({"step": step, "minutes": round((now - t0) / 60, 2),
                              "ms_per_step": round(1000 * (now - t_last) / n, 1),
                              "train_bits": round(float(acc) / n / LN2, 4),
                              "memory_proj_norm": round(float(model.proj.weight.norm()), 3)}), flush=True)
            acc.zero_()
            t_last, s_last = now, step
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="gru", choices=["gru", "attn"])
    ap.add_argument("--out", default=None, help="default: out/cmp_<arch>.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-3, help="peak learning rate of the MLP path (train_net.py's)")
    ap.add_argument("--mem-lr", type=float, default=1e-3, help="peak learning rate of the memory path")
    ap.add_argument("--warmup", type=float, default=0.05, help="share of the steps spent warming up")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient norm clip of the memory path")
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--window", type=int, default=32, help="attn: frames each frame can see")
    ap.add_argument("--chunk", type=int, default=300, help="frames per training and test chunk")
    ap.add_argument("--chunks", type=int, default=64, help="chunks per step")
    ap.add_argument("--loss-frames", type=int, default=36, help="frames per chunk the loss is taken on")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--bandwidths", default="2,4,8,15,16")
    ap.add_argument("--k-mix", default=train_net.DEFAULT_K_MIX)
    ap.add_argument("--eval-frames", type=int, default=45000, help="held-out frames per test set")
    ap.add_argument("--log-every", type=int, default=1000, help="steps per log line (train_bits is their mean)")
    ap.add_argument("--compare", action="store_true", help="also train the MLP baseline and print the verdict")
    ap.add_argument("--mlp", default=None, help="existing MLP baseline to compare with "
                    "(default: train out/cmp_mlp.pt with train_net.py for --steps steps)")
    ap.add_argument("--mlp-timeout-hours", type=float, default=3.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    args.out = args.out or os.path.join(HERE, "out", f"cmp_{args.arch}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data = np.load(args.data)

    mlp = None
    if args.compare:
        mlp_path = args.mlp or os.path.join(os.path.dirname(os.path.abspath(args.out)), "cmp_mlp.pt")
        if not args.mlp:
            cmd = [sys.executable, os.path.join(HERE, "train_net.py"), "--data", args.data, "--out", mlp_path,
                   "--steps", str(args.steps), "--batch", "2048", "--eval-every", str(max(args.steps // 4, 1)),
                   "--fraction", str(args.fraction), "--bandwidths", args.bandwidths, "--k-mix", args.k_mix,
                   "--device", args.device, "--seed", str(args.seed)]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            print("MLP baseline:", " ".join(cmd), flush=True)
            t0 = time.time()
            subprocess.run(cmd, check=True, timeout=args.mlp_timeout_hours * 3600)
            print(json.dumps({"mlp_minutes": round((time.time() - t0) / 60, 1)}), flush=True)
        import export_net
        mlp = export_net.load_net(mlp_path, data["centroids"].shape[0]).to(device)

    tests = eval_sets(args, data, device)
    t0 = time.time()
    model = train_seq(args, data, device, tests)
    print(json.dumps({"seq_minutes": round((time.time() - t0) / 60, 1)}), flush=True)

    period = model.period
    ks = [k for k in (4, 8, 15, 16) if k in model.bandwidths]
    rows = {}
    for name, frames, starts in tests:
        for kk in ks:
            rows[(name, kk, args.arch)] = score(model, frames, starts, args.chunk, kk, period, device)
            rows[(name, kk, "off")] = score_mem_off(model, frames, starts, args.chunk, kk, period, device)
            if mlp is not None:
                rows[(name, kk, "mlp")] = score(mlp, frames, starts, args.chunk, kk, period, device)
    print(f"\nheld-out bits per code, {args.steps} steps, chunks of {args.chunk} frames (context cut at chunk starts)")
    print("(mem off = this model with its memory forced to zero, i.e. its own MLP path)")
    print(f"{'set':9} {'K':>3} {args.arch:>8}  mem off" + ("      mlp     gain" if mlp is not None else ""))
    for (name, kk, arch), v in rows.items():
        if arch != args.arch:
            continue
        line = f"{name:9} {kk:>3} {v:8.4f} {rows[(name, kk, 'off')]:8.4f}"
        if mlp is not None:
            m = rows[(name, kk, "mlp")]
            line += f" {m:8.4f} {m - v:+8.4f}"
        print(line)
    hm, mm = macs_per_frame(args.arch, args.hidden, window=args.window), macs_per_frame("mlp", 0)
    heads = head_macs_per_frame(15)
    print(f"\nmultiply-adds per frame: MLP trunk + {args.arch} memory {hm:,} (MLP trunk {mm:,} + memory "
          f"{hm - mm:,}), heads at K=15 {heads:,}; net cost per frame x{(hm + heads) / (mm + heads):.2f} of today's")
    if mlp is not None:
        gains = [rows[(n, 15, "mlp")] - rows[(n, 15, args.arch)] for n in ("test_old", "test_new")
                 if (n, 15, "mlp") in rows]
        gain = float(np.mean(gains)) if gains else float("nan")
        verdict = "WORTH BUILDING" if gain >= KILL_GAIN else "NOT WORTH BUILDING"
        print(f"verdict: {args.arch} gains {gain:+.3f} bits per code at K=15 (mean of the test sets); "
              f"kill rule: under {KILL_GAIN} -> {verdict}")


if __name__ == "__main__":
    main()
