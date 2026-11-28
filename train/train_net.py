"""Trains the optional "net" expert of the cx2 entropy model for the EnCodec tables (gradient descent, PyTorch).

    uv run python train_net.py --data train_data_48k_full.npz --out out/net48v2.pt

Self-contained: needs only the data archive made by collect_data.py (codes and the
fixed-point RVQ centroids), numpy and torch. Runs on CUDA when available, else CPU.

The network predicts each EnCodec code from the previous four frames' latents and the part
of the current frame already coded. Every size and range below is chosen so that the
integer inference in src/cx2/net.rs is exact and overflow-free; do not change the
clamps without changing that file:

  x     = centroid sums of the previous 4 frames over the stream's K codebooks,
          clamp(sum >> 3, +-1023), i.e. units of 1/16                     (4 x 128 values)
  h1    = relu8(W1 x + b1 + e[K] + p[bucket(t mod P)])                     (H1 <= 512)
  h2    = relu8(W2 h1 + b2)                                               (H2 <= 256)
  g_k   = relu8(A_k h2 + B_k s_k + c_k),  s_k = this frame's partial sum  (rank <= 128)
  logit = O_k g_k + d_k                                                   (1024)

relu8 clamps to [0, 8); |W|, |A| < 2, |B| < 0.25, |O| < 1.38 (weights are clamped after
every step). Each batch draws the codebook count K from the model's bandwidths, so one
network serves all of them.

Position. The 48 kHz model codes 1 s segments of 150 frames, whose first ~30 ms and last
~10 ms are damaged, so code statistics depend on where a frame sits in its segment. The net
gets t mod P (P = 150; t = frames since the stream start, which is what the coder sees,
since its resets always fall on segment starts) through a learned first-layer bias per
position bucket: frames 0-11 and 138-149 have one bucket each, the middle shares 9. The
24 kHz model is not segmented and gets P = 0 (no position input).

Defaults are for the full 24.3 h archive at batch 2048 (see README.md for the reasoning):
180,000 steps = 369 M frame samples = 28 passes over 13.1 M frames, peak learning rate
3e-3 (1.5e-3 at batch 512 times sqrt(4)), one-cycle schedule.
"""
import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import torch
from torch import nn

LN2 = math.log(2.0)
PAST = 4
W_MAX, B_MAX, O_MAX = 1.999, 0.2499, 1.38
# Biases are stored as round(value * 2^14) and Rust accepts at most 2^28 per bias.
BIAS_MAX = 2.0 ** 28 / 2 ** 14 * 0.99
DEFAULT_K_MIX = "2:0.10,4:0.15,8:0.15,15:0.35,16:0.25"


def position_buckets(period, head=12, tail=12, middle=9):
    """Bucket of every position 0..period-1 (empty when period is 0)."""
    if period <= 0:
        return np.zeros(0, dtype=np.int64)
    if period <= head + tail + middle:
        return np.arange(period, dtype=np.int64)
    mid = period - head - tail
    return np.concatenate([np.arange(head), head + np.arange(mid) * middle // mid,
                           head + middle + np.arange(tail)]).astype(np.int64)


class Net(nn.Module):
    def __init__(self, bandwidths, codebooks, h1=384, h2=256, rank_low=64, rank_high=32, period=0):
        super().__init__()
        self.bandwidths = list(bandwidths)
        ranks = [rank_low if k < 8 else rank_high for k in range(codebooks)]
        self.l1 = nn.Linear(128 * PAST, h1)
        self.kemb = nn.Embedding(len(self.bandwidths), h1)
        self.register_buffer("bandwidth_list", torch.tensor(self.bandwidths, dtype=torch.long))
        self.period = int(period)
        pos_map = position_buckets(self.period)
        self.register_buffer("pos_map", torch.from_numpy(pos_map))
        if self.period:
            self.pemb = nn.Embedding(int(pos_map.max()) + 1, h1)
            nn.init.zeros_(self.pemb.weight)
        self.l2 = nn.Linear(h1, h2)
        self.a = nn.ModuleList([nn.Linear(h2, r) for r in ranks])
        self.b = nn.ModuleList([nn.Linear(128, r, bias=False) for r in ranks])
        self.o = nn.ModuleList([nn.Linear(r, 1024) for r in ranks])
        nn.init.zeros_(self.kemb.weight)
        for b in self.b:
            nn.init.uniform_(b.weight, -0.02, 0.02)

    def clamp_(self):
        with torch.no_grad():
            for lin in [self.l1, self.l2, *self.a]:
                lin.weight.clamp_(-W_MAX, W_MAX)
            for lin in self.b:
                lin.weight.clamp_(-B_MAX, B_MAX)
            for lin in self.o:
                lin.weight.clamp_(-O_MAX, O_MAX)
            for emb in [self.kemb, *([self.pemb] if self.period else [])]:
                emb.weight.clamp_(-BIAS_MAX, BIAS_MAX)

    def trunk(self, x, kidx, pos=None):
        h = self.l1(x) + self.kemb(kidx)
        if self.period:
            h = h + self.pemb(self.pos_map[pos])
        h = torch.clamp(h, 0, 8)
        return torch.clamp(self.l2(h), 0, 8)

    def head(self, k, h2, s):
        g = torch.clamp(self.a[k](h2) + self.b[k](s), 0, 8)
        return self.o[k](g)


def since_start(lengths):
    """Frames since the start of each stream, for concatenated streams."""
    return np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])


def batch(centroids, codes, since, idx, k_total):
    """NumPy reference of `Frames.batch` (kept to check it): trunk input [B, 512], partial
    sums [B, k_total, 128] (floats, units of 1/16), targets."""
    xs = []
    for lag in range(1, PAST + 1):
        c = codes[np.maximum(idx - lag, 0), :k_total].astype(np.int64)
        z = np.zeros((len(idx), 128), dtype=np.int32)
        for i in range(k_total):
            z += centroids[i][c[:, i]]
        z[since < lag] = 0
        xs.append(np.clip(z >> 3, -1023, 1023))
    cur = codes[idx, :k_total].astype(np.int64)
    part = np.zeros((len(idx), k_total, 128), dtype=np.int32)
    run = np.zeros((len(idx), 128), dtype=np.int32)
    for i in range(k_total):
        part[:, i] = np.clip(run >> 3, -1023, 1023)
        run += centroids[i][cur[:, i]]
    x = torch.from_numpy(np.concatenate(xs, axis=1).astype(np.float32) / 16.0)
    return x, torch.from_numpy(part.astype(np.float32) / 16.0), torch.from_numpy(cur)


class Frames:
    """Codes of concatenated streams and the centroids, as tensors on one device.

    Batches are assembled with tensor gathers on that device, so on a GPU nothing goes
    through the CPU per step. The arithmetic is integer and identical to `batch()`.
    """

    def __init__(self, centroids, codes, lengths, device):
        kmax = centroids.shape[0]
        self.device = device
        self.codes = torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int16)).to(device)
        self.since = torch.from_numpy(since_start(lengths)).to(device)
        self.centroids = torch.from_numpy(
            centroids.reshape(kmax * 1024, 128).astype(np.int32)).to(device)
        self.offsets = torch.arange(kmax, device=device) * 1024
        self.lags = torch.arange(PAST + 1, device=device)
        self.n = len(codes)

    def batch(self, idx, since, k_total):
        """idx, since: long tensors [B] (frame index, frames of past available)."""
        rows = torch.clamp(idx[:, None] - self.lags, min=0)                      # [B, 5]
        c = self.codes[rows][:, :, :k_total].long() + self.offsets[:k_total]   # [B, 5, k]
        z = self.centroids[c]                                                  # [B, 5, k, 128]
        past = z[:, 1:].sum(2, dtype=torch.int64)                              # [B, 4, 128]
        past = past * (since[:, None] >= self.lags[1:]).long()[:, :, None]
        x = torch.clamp(torch.div(past, 8, rounding_mode="floor"), -1023, 1023)
        cur = z[:, 0].to(torch.int64)                                          # [B, k, 128]
        run = torch.cumsum(cur, dim=1) - cur
        part = torch.clamp(torch.div(run, 8, rounding_mode="floor"), -1023, 1023)
        y = self.codes[idx, :k_total].long()
        return x.reshape(len(idx), PAST * 128).float() / 16.0, part.float() / 16.0, y


def evaluate(net, frames, idx, k_total, period):
    """Mean bits per code of each codebook, with all context available."""
    bits = np.zeros(k_total)
    kid = net.bandwidths.index(k_total)
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            part_idx = idx[s:s + 4096]
            since = frames.since[part_idx].long()
            x, part, y = (t.to(device) for t in frames.batch(part_idx, since, k_total))
            kidx = torch.full((len(part_idx),), kid, dtype=torch.long, device=device)
            pos = (since % period).to(device) if period else None
            h2 = net.trunk(x, kidx, pos)
            for k in range(k_total):
                loss = nn.functional.cross_entropy(net.head(k, h2, part[:, k]).float(), y[:, k],
                                                   reduction="sum")
                bits[k] += float(loss) / LN2
    net.train()
    return bits / len(idx)


def select_clips(names, fraction):
    """Indices of a deterministic `fraction` of the clips, chosen by a hash of their names
    (so 0.25 is a subset of 0.5, which is a subset of 1.0, whatever the archive order)."""
    order = sorted(range(len(names)), key=lambda i: (zlib.crc32(names[i].encode()), names[i]))
    keep = max(1, math.ceil(fraction * len(names)))
    return sorted(order[:keep])


def parse_k_mix(text, bandwidths):
    pairs = [item.split(":") for item in text.split(",") if item]
    mix = {int(k): float(p) for k, p in pairs}
    assert set(mix) <= set(bandwidths), f"--k-mix names a bandwidth outside {bandwidths}"
    ks = sorted(mix)
    p = np.array([mix[k] for k in ks])
    return ks, p / p.sum()


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="archive from collect_data.py")
    ap.add_argument("--out", required=True, help="network to write (saved at every evaluation); "
                    "OUT.json is its description, OUT.log the progress, OUT.ckpt the resume state")
    ap.add_argument("--steps", type=int, default=180000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3, help="peak learning rate (one-cycle schedule)")
    ap.add_argument("--resume", default=None, help="OUT.ckpt of an interrupted run: continue it exactly")
    ap.add_argument("--init", default=None, help="network (.pt) to start from (same architecture)")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the fit clips to train on "
                    "(by clip, deterministic; for learning curves)")
    ap.add_argument("--bandwidths", default="2,4,8,15,16", help="codebook counts the net serves")
    ap.add_argument("--k-mix", default=DEFAULT_K_MIX, help="probability of each codebook count per batch")
    ap.add_argument("--period", type=int, default=-1, help="frames per segment for the position input "
                    "(-1: 150 for the 48 kHz model, 0 otherwise; 0 switches it off)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-frames", type=int, default=20000, help="frames per test set and evaluation")
    ap.add_argument("--ckpt-every", type=int, default=5000, help="steps between resume checkpoints")
    ap.add_argument("--h1", type=int, default=384)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--rank-low", type=int, default=64, help="head width of codebooks 0-7 (<= 128)")
    ap.add_argument("--rank-high", type=int, default=32, help="head width of codebooks 8+ (<= 128)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA (off by default)")
    ap.add_argument("--no-tf32", action="store_true", help="full fp32 matmuls on CUDA")
    ap.add_argument("--data-on-cpu", action="store_true", help="keep the codes in host memory "
                    "(for GPUs with under ~2 GB free); batches are then pinned and copied")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: PyTorch default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.h1 <= 512 and args.h2 <= 256 and max(args.rank_low, args.rank_high) <= 128
    assert 0 < args.fraction <= 1
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume:
        # The schedule and the data depend on these; keep them as they were.
        for key in ("steps", "batch", "lr", "fraction", "bandwidths", "k_mix", "period", "h1", "h2",
                    "rank_low", "rank_high", "seed"):
            if getattr(args, key) != resume["args"][key]:
                print(f"resume: using the checkpoint's --{key.replace('_', '-')} {resume['args'][key]}")
                setattr(args, key, resume["args"][key])
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data_device = torch.device("cpu") if args.data_on_cpu else device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=data_device)
    gen.manual_seed(args.seed + 1)

    data = np.load(args.data)
    centroids = data["centroids"].astype(np.int32)          # [K, 1024, 128], fixed point x128
    kmax = centroids.shape[0]
    bandwidths = sorted(int(k) for k in args.bandwidths.split(","))
    assert bandwidths[-1] == kmax and len(bandwidths) <= 8, "bandwidths must end at the archive's codebooks"
    ks, kp = parse_k_mix(args.k_mix, bandwidths)
    model_name = str(data["model"])
    period = args.period if args.period >= 0 else (150 if model_name == "encodec_48khz" else 0)
    args.period = period
    rate = float(data["frame_rate"])

    names = json.loads(str(data["files"]))["fit"]
    lengths = data["fit_lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    chosen = select_clips(names, args.fraction)
    fit = Frames(centroids, data["fit_codes"], lengths, data_device)
    pool = torch.from_numpy(np.concatenate(
        [np.arange(starts[i], starts[i] + lengths[i]) for i in chosen])).to(data_device)
    tests, eval_idx = {}, {}
    eval_rng = np.random.default_rng(args.seed)
    for name in ("test_old", "test_new"):
        if f"{name}_codes" in data:
            tests[name] = Frames(centroids, data[f"{name}_codes"], data[f"{name}_lengths"], data_device)
            n = tests[name].n
            eval_idx[name] = torch.from_numpy(np.sort(eval_rng.choice(n, min(args.eval_frames, n),
                                                                     replace=False))).to(data_device)
    print(json.dumps({"device": str(device), "model": model_name, "fit_clips": len(chosen),
                      "fit_frames": int(len(pool)), "fit_hours": round(len(pool) / rate / 3600, 2),
                      "passes": round(args.steps * args.batch / len(pool), 1), "bandwidths": bandwidths,
                      "k_mix": dict(zip(ks, np.round(kp, 3).tolist())), "period": period,
                      "test_frames": {n: f.n for n, f in tests.items()}}), flush=True)

    net = Net(bandwidths, kmax, args.h1, args.h2, args.rank_low, args.rank_high, period)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    net.to(device)
    fused = device.type == "cuda"
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=fused)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    first = 1
    if resume:
        net.load_state_dict(resume["net"])
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        rng.bit_generator.state = resume["rng"]
        gen.set_state(resume["gen"])
        first = resume["step"] + 1
        print(json.dumps({"resumed_at": resume["step"]}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"bandwidths": bandwidths, "codebooks": kmax, "h1": args.h1, "h2": args.h2,
            "rank_low": args.rank_low, "rank_high": args.rank_high, "steps": args.steps,
            "batch": args.batch, "lr": args.lr, "fraction": args.fraction, "k_mix": args.k_mix,
            "period": period, "position_buckets": position_buckets(period).tolist(),
            "model": model_name, "data": os.path.basename(args.data)}
    log = open(args.out + ".log", "a", encoding="utf-8")
    # Real resets fall on segment starts every 1, 2 or 4 segments (music24/12/6 pages);
    # a quarter of the samples each see one of those, a quarter only the stream start.
    resets = torch.tensor([period, 2 * period, 4 * period, 1 << 30], device=data_device)
    t0 = time.time()
    t_last, s_last = t0, first - 1
    for step in range(first, args.steps + 1):
        k_total = int(ks[rng.choice(len(ks), p=kp)])
        idx = pool[torch.randint(0, len(pool), (args.batch,), generator=gen, device=data_device)]
        since_stream = fit.since[idx].long()
        if period:
            pick = torch.randint(0, 4, (args.batch,), generator=gen, device=data_device)
            since = since_stream % resets[pick]
            pos = since_stream % period
        else:
            # unsegmented model: mimic reset points at random frames
            cut = torch.rand(args.batch, generator=gen, device=data_device) < 0.03
            since = torch.where(cut, torch.randint(0, PAST, (args.batch,), generator=gen,
                                                   device=data_device), since_stream)
            pos = None
        x, part, y = fit.batch(idx, since, k_total)
        kidx = torch.full((args.batch,), bandwidths.index(k_total), dtype=torch.long, device=data_device)
        if data_device != device:
            x, part, y, kidx = (t.pin_memory().to(device, non_blocking=True) for t in (x, part, y, kidx))
            pos = pos.pin_memory().to(device, non_blocking=True) if pos is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
            h2 = net.trunk(x, kidx, pos)
            loss = sum(nn.functional.cross_entropy(net.head(k, h2, part[:, k]).float(), y[:, k])
                       for k in range(k_total)) / k_total
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        net.clamp_()
        if step % args.eval_every == 0 or step == args.steps:
            now = time.time()
            row = {"step": step, "minutes": round((now - t0) / 60, 1),
                   "ms_per_step": round(1000 * (now - t_last) / max(step - s_last, 1), 1),
                   "train_bits": round(float(loss.detach()) / LN2, 3)}
            for name, frames in tests.items():
                for kk in bandwidths:
                    b = evaluate(net, frames, eval_idx[name], kk, period)
                    row[f"{name}_K{kk}"] = round(float(b.mean()), 4)
                    if kk == kmax:
                        row[f"{name}_per_codebook"] = [round(float(v), 2) for v in b]
            line = json.dumps(row)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            atomic_save({k: v.cpu() for k, v in net.state_dict().items()}, args.out)
            with open(args.out + ".json", "w", encoding="utf-8") as f:
                json.dump({**meta, "steps_done": step, "last": row}, f, indent=1)
            t_last, s_last = time.time(), step
        if step % args.ckpt_every == 0 and step < args.steps:
            atomic_save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "rng": rng.bit_generator.state, "gen": gen.get_state(), "step": step,
                         "args": vars(args)}, args.out + ".ckpt")
    log.close()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
