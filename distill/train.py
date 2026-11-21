"""Distil DAC 44 kHz's decoder into a narrow student, with DAC's own GAN recipe against the original audio.

    uv run --extra cuda python train.py --run runs\\pilot512 --manifest data\\pd.jsonl --manifest data\\jamendo.jsonl \\
        --eval-set data\\pd.jsonl.eval.json --eval-set data\\jamendo.jsonl.eval.json --steps 50000

Inputs: a frozen DAC encoder + quantiser encode each crop with `--enc-ctx` samples of context on both sides
(the codes then equal whole-file codes), the student decodes the crop's frames plus `--dec-ctx` frames each side,
and the losses see the centre only. Losses: 15 x multi-scale mel + 2 x feature matching + 1 x adversarial
(least squares as in DAC's code, or --gan hinge), optional output matching against the stock decoder
(--w-teacher-mel). AdamW (lr 1e-4, betas 0.8/0.99, decay 0.999996 per step) for generator and discriminator,
gradient clipping 1e3 / 10, bf16 autocast on CUDA (Snake, residual sums, the output head, STFTs and losses stay fp32; evals run fully in fp32).

Every --eval-every steps: log-mel / MR-STFT / SNR / SI-SDR on the held-out clips, stock DAC's numbers on the
same clips (once, as the target line), the pilot verdict, and listening WAVs in <run>/listen/.
`--resume` continues from <run>/ckpt_last.pt with the identical batch sequence (batches are a function of the
step) and restored optimiser/scheduler/RNG state. `--bench` measures memory and ms/step for batch sizes.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

import corpus
import data
import losses
import metrics
import verdict
from student import STOCK_WIDTH, StudentDecoder, count_cost, n_params, warm_start_slice
from teacher import HOP, codes_to_z, encode_codes, load_teacher, stock_decode, stock_decode_bf16

SR = 44100


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("run")
    g.add_argument("--run", required=True, help="run directory (checkpoints, logs, listen/)")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--steps", type=int, default=50000)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--amp", choices=("auto", "bf16", "off"), default="auto", help="auto = bf16 on CUDA")
    g.add_argument("--bench", default="", help="comma list of batch sizes: measure memory + ms/step, then exit")
    g.add_argument("--eval-only", default="", metavar="CKPT",
                   help="load G from this checkpoint, run one eval (at --eval-n-q) into --run, then exit; "
                        "use a fresh --run so the logs don't mix with the training run's")
    g = ap.add_argument_group("data")
    g.add_argument("--manifest", action="append", default=[])
    g.add_argument("--eval-set", action="append", default=[])
    g.add_argument("--licences", default=corpus.DEFAULT_LICENCES,
                   help=f"classes or presets {tuple(corpus.LICENCE_PRESETS)}; default sharealike = pd,cc0,by,by-sa,fal")
    g.add_argument("--mix", default="auto", help="'auto' (by duration, floored) or e.g. pd=0.25,jamendo=0.75")
    g.add_argument("--mix-floor", type=float, default=0.2, help="auto mix: minimum weight of any source")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--tracks-per-batch", type=int, default=12)
    g.add_argument("--chunk-seconds", type=float, default=12.0)
    g.add_argument("--max-eval-clips", type=int, default=0)
    g.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                   help="rewrite manifest/eval paths made on another machine, e.g. "
                        "'E:\\Shared\\audio=/media/demo/External20TB/ai/audio' (repeatable)")
    g.add_argument("--eval-n-q", default="", help="codebook counts to evaluate, e.g. 5,9 (default: those of --n-q)")
    g = ap.add_argument_group("model")
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--init", choices=("random", "slice"), default="random",
                   help="slice = keep the teacher's highest-energy channels per stage")
    g.add_argument("--n-q", default="5", help="codebooks: one count, or a set for quantiser dropout, e.g. 4,5,6,9")
    g = ap.add_argument_group("batch")
    g.add_argument("--batch", type=int, default=32)
    g.add_argument("--seg-frames", type=int, default=32, help="loss segment in 512-sample frames (32 = 0.37 s)")
    g.add_argument("--enc-ctx", type=int, default=4096, help="encoder context samples each side (mult. of 512)")
    g.add_argument("--dec-ctx", type=int, default=2, help="decoder context frames each side")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--betas", type=float, nargs=2, default=(0.8, 0.99))
    g.add_argument("--gamma", type=float, default=0.999996)
    g.add_argument("--clip-g", type=float, default=1e3)
    g.add_argument("--clip-d", type=float, default=10.0)
    g.add_argument("--gan", choices=("ls", "hinge"), default="ls")
    g.add_argument("--w-mel", type=float, default=15.0)
    g.add_argument("--w-fm", type=float, default=2.0)
    g.add_argument("--w-adv", type=float, default=1.0)
    g.add_argument("--w-teacher-mel", type=float, default=0.0, help="mel loss vs the stock decoder's output")
    g.add_argument("--w-teacher-wave", type=float, default=0.0, help="L1 waveform loss vs the stock output")
    g.add_argument("--teacher-amp", choices=("bf16", "fp32"), default="bf16",
                   help="teacher target precision when bf16 autocast is on: bf16 body + fp32 output head "
                        "(about 0.07 dB log-mel off fp32, faster) or fully fp32")
    g.add_argument("--w-lf", type=float, default=100.0,
                   help="L1 on the 1024-sample moving average (DC and sub-bass below ~40 Hz), which the mel loss "
                        "and the discriminators barely see; without it the output can drift a DC offset")
    g.add_argument("--gan-start", type=int, default=0, help="steps of mel-only training before the GAN")
    g = ap.add_argument_group("schedule")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--eval-every", type=int, default=5000)
    g.add_argument("--ckpt-every", type=int, default=2500)
    g.add_argument("--keep-every", type=int, default=25000, help="also keep ckpt_<step>.pt at multiples")
    g.add_argument("--listen", type=int, default=6, help="eval clips written as WAVs at each eval")
    g.add_argument("--listen-every", type=int, default=0,
                   help="write listening WAVs only at evals on multiples of this (0 = every eval)")
    g.add_argument("--g-ema", type=float, default=0.999,
                   help="decay of a moving average of G's weights, evaluated beside G (0 = off); the "
                        "checkpoints keep both, ckpt_best.pt keeps whichever scored better")
    g.add_argument("--stop-plateau", type=float, default=0.0,
                   help="stop once the best score (mean picotone log-mel over --eval-n-q, the average of the "
                        "last 4 evals) improved by less than this over --plateau-steps (0 = never)")
    g.add_argument("--plateau-steps", type=int, default=20000)
    g.add_argument("--stop-when-pass", action="store_true",
                   help="stop once every --eval-n-q tier passes the full-run rule")
    g.add_argument("--pilot-steps", type=int, default=50000, help="verdict rule: pilot before, full after")
    g.add_argument("--picotone-json", default="", help="compare_picotone.py output for the verdict")
    a = ap.parse_args(argv)
    a.n_q_list = [int(v) for v in a.n_q.split(",")]
    a.path_map = data.parse_path_map(a.path_map)
    a.eval_n_q = sorted({int(v) for v in a.eval_n_q.split(",")} if a.eval_n_q else set(a.n_q_list))
    assert a.enc_ctx % HOP == 0 and a.enc_ctx // HOP >= a.dec_ctx
    return a


class Log:
    def __init__(self, run):
        self.f = open(os.path.join(run, "log.txt"), "a", encoding="utf-8")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def jsonl(run, name, row):
    with open(os.path.join(run, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def amp_on(a):
    return a.amp == "bf16" or (a.amp == "auto" and a.device.startswith("cuda"))


def autocast(a):
    return torch.autocast(device_type="cuda" if a.device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                          enabled=amp_on(a))


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ one training step

def forward_g(a, teacher, G, batch, dev):
    x = batch["x"].to(dev, non_blocking=True)
    n_q = batch["n_q"].to(dev)
    S, ce, cd = a.seg_frames, a.enc_ctx // HOP, a.dec_ctx
    codes = encode_codes(teacher, x)
    z = codes_to_z(teacher, codes[:, :, ce - cd: ce + S + cd], n_q)
    with autocast(a):
        y_full = G(z)
    y_hat = y_full[..., cd * HOP: (cd + S) * HOP]
    y = x[..., a.enc_ctx: a.enc_ctx + S * HOP]
    y_t = None
    if a.w_teacher_mel > 0 or a.w_teacher_wave > 0:
        if a.teacher_amp == "bf16" and amp_on(a):
            y_t = stock_decode_bf16(teacher, z.float())
        else:
            with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
                y_t = stock_decode(teacher, z.float()).float()
        y_t = y_t[..., cd * HOP: (cd + S) * HOP]
    return y_hat, y, y_t


def train_step(a, step, teacher, G, D, mel, opt_g, opt_d, sch_g, sch_d, batch, dev):
    y_hat, y, y_t = forward_g(a, teacher, G, batch, dev)
    out = {}
    gan = step >= a.gan_start
    if gan:
        with autocast(a):
            loss_d = losses.d_loss(D(y_hat.detach()), D(y), a.gan)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        out["gn_d"] = float(torch.nn.utils.clip_grad_norm_(D.parameters(), a.clip_d))
        opt_d.step()
        out["d"] = loss_d.item()
    sch_d.step()
    l_mel = mel(y_hat, y)
    total = a.w_mel * l_mel
    out["mel"] = l_mel.item()
    if a.w_lf > 0:
        l_lf = (torch.nn.functional.avg_pool1d(y_hat.float(), 1024, 256)
                - torch.nn.functional.avg_pool1d(y.float(), 1024, 256)).abs().mean()
        total = total + a.w_lf * l_lf
        out["lf"] = l_lf.item()
    if gan:
        with autocast(a):
            adv, fm = losses.g_losses(D(y_hat), D(y), a.gan)
        total = total + a.w_adv * adv + a.w_fm * fm
        out["adv"], out["fm"] = adv.item(), fm.item()
    if y_t is not None:
        if a.w_teacher_mel > 0:
            t_mel = mel(y_hat, y_t)
            total = total + a.w_teacher_mel * t_mel
            out["t_mel"] = t_mel.item()
        if a.w_teacher_wave > 0:
            t_w = (y_hat - y_t).abs().mean()
            total = total + a.w_teacher_wave * t_w
            out["t_wave"] = t_w.item()
    opt_g.zero_grad(set_to_none=True)
    total.backward()
    out["gn_g"] = float(torch.nn.utils.clip_grad_norm_(G.parameters(), a.clip_g))
    opt_g.step()
    sch_g.step()
    out["total"] = total.item()
    return out


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def decode_clip(a, teacher, G, codes, n_q):
    # Eval decodes in fp32, as deployment will and as picotone's scores were measured: under bf16, stock DAC
    # scores about 0.8 dB worse log-mel (same SI-SDR), which would skew every comparison.
    z = codes_to_z(teacher, codes, n_q).float()
    with torch.autocast(device_type=z.device.type, enabled=False):
        y = G(z) if G is not None else stock_decode(teacher, z)
    return y.float()[:, 0].clamp(-1, 1).cpu().numpy()


def prepare_eval(a, teacher, dev, log):
    clips = data.load_eval(a.eval_set, a.max_eval_clips, log, a.path_map)
    for e in clips:
        x = torch.from_numpy(e["x"])[:, None].to(dev)
        e["codes"] = encode_codes(teacher, x)
    return clips


def set_means(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}):
        out[s] = metrics.mean_scores([r["m"] for r in rows if r["set"] == s])
    out["all"] = metrics.mean_scores([r["m"] for r in rows])
    return out


def write_wav(path, x):
    sf.write(path, np.asarray(x).T, SR, subtype="PCM_16")


def run_eval(a, teacher, G, clips, n_q, tag, listen_dir):
    rows = []
    for i, e in enumerate(clips):
        y = decode_clip(a, teacher, G, e["codes"], n_q)
        rows.append({"name": e["name"], "set": e["set"], "m": metrics.score(e["x"], y)})
        if i < a.listen and listen_dir:
            write_wav(os.path.join(listen_dir, f"{e['name']}__{tag}.wav"), y)
    return set_means(rows), rows


def stock_eval(a, teacher, clips, run, log):
    p = os.path.join(run, "stock_eval.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    todo = [nq for nq in a.eval_n_q if str(nq) not in res]
    if not todo:
        return res
    listen = os.path.join(run, "listen")
    os.makedirs(listen, exist_ok=True)
    if not res:
        for i, e in enumerate(clips[: a.listen]):
            write_wav(os.path.join(listen, f"{e['name']}__0_original.wav"), e["x"])
    for nq in todo:
        t = time.perf_counter()
        means, rows = run_eval(a, teacher, None, clips, nq, f"0_stockDAC_q{nq}", listen)
        res[str(nq)] = {"means": means, "rows": rows}
        log(f"stock DAC at {nq} codebooks ({time.perf_counter() - t:.0f} s): " + fmt_means(means))
    json.dump(res, open(p, "w"), indent=1)
    return res


def fmt_means(means):
    def dc(m):
        return f" dc {m['dc_mfs']:.2f}" if "dc_mfs" in m else ""

    return " | ".join(f"{s}: mel {m['logmel']:.3f} mrstft {m['mrstft']:.3f} snr {m['snr']:.2f} sisdr {m['sisdr']:.2f}"
                      + dc(m) for s, m in means.items())


# ------------------------------------------------------------------ checkpoints

def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    # torch.load(map_location="cuda") moves these to the GPU too; the setters want CPU ByteTensors.
    torch.set_rng_state(s["torch"].cpu())
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


@torch.no_grad()
def update_g_ema(G_ema, G, decay):
    e, p = list(G_ema.parameters()), list(G.parameters())
    torch._foreach_lerp_(e, [q.detach() for q in p], 1.0 - decay)


def save_ckpt(path, step, G, D, opts, scheds, ema, a, G_ema=None):
    tmp = path + ".tmp"
    torch.save({"step": step, "G": G.state_dict(), "D": D.state_dict(),
                "G_ema": G_ema.state_dict() if G_ema is not None else None,
                "opt_g": opts[0].state_dict(), "opt_d": opts[1].state_dict(),
                "sch_g": scheds[0].state_dict(), "sch_d": scheds[1].state_dict(), "ema": ema,
                "rng": rng_state(), "args": vars(a), "width": a.width}, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ bench

def bench(a, teacher, G, D, mel, opts, scheds, ds_factory, dev, log):
    """Per batch size: ms/step (after warm-up) and peak memory (CUDA) or saved-activation bytes (CPU)."""
    for B in [int(v) for v in a.bench.split(",")]:
        ds = ds_factory(B)
        it = iter(data.loader(ds, a.workers))
        batches = [next(it) for _ in range(3)]
        saved = [0]

        def pack(t):
            saved[0] += t.numel() * t.element_size()
            return t

        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                train_step(a, a.gan_start, teacher, G, D, mel, *opts, *scheds, batches[0], dev)
            n = 3 if dev == "cpu" else 10
            t = time.perf_counter()
            for i in range(n):
                train_step(a, a.gan_start, teacher, G, D, mel, *opts, *scheds, batches[i % 3], dev)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) / n * 1000
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else float("nan")
            log(f"bench batch {B}: {ms:.0f} ms/step, peak CUDA {peak:.2f} GiB, saved activations "
                f"{saved[0] / 2 ** 30:.2f} GiB ({saved[0] / B / 2 ** 20:.1f} MiB/item)")
        except torch.OutOfMemoryError:
            log(f"bench batch {B}: out of memory")
            break


# ------------------------------------------------------------------ main

def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.run, exist_ok=True)
    log = Log(a.run)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    dev = a.device
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    log(f"== {' '.join(sys.argv)}")
    teacher = load_teacher(dev)
    G = StudentDecoder(a.width).to(dev)
    if a.init == "slice" and not a.resume:
        warm_start_slice(G, teacher.decoder)
    D = losses.Discriminator().to(dev)
    mel = losses.MultiScaleMel().to(dev)
    # Cost report (MACs per second of audio, one channel), measured.
    fz = torch.zeros(1, 1024, 87, device=dev)
    sc, tc = count_cost(G, fz), count_cost(teacher.decoder, fz)
    sec = 87 * HOP / SR
    log(f"student width {a.width}: {n_params(G):,} params, {sc['mac'] / sec / 1e9:.2f} GMAC/s, "
        f"{sc['snake'] / sec / 1e6:.1f} M Snake/s | stock DAC decoder width {STOCK_WIDTH}: "
        f"{n_params(teacher.decoder):,} params, {tc['mac'] / sec / 1e9:.2f} GMAC/s -> "
        f"{tc['mac'] / sc['mac']:.2f}x fewer MACs | discriminator {n_params(D):,} params")
    dcost = count_cost(D, torch.zeros(1, 1, SR, device=dev))["mac"]
    ecost = count_cost(teacher.encoder, torch.zeros(1, 1, 86 * HOP, device=dev))["mac"] / (86 * HOP / SR)
    seg_s, dec_s = a.seg_frames * HOP / SR, (a.seg_frames + 2 * a.dec_ctx) * HOP / SR
    enc_s = seg_s + 2 * a.enc_ctx / SR
    # Per item and step: encoder forward, student forward+backward (~3x), discriminator ~10 forward-equivalents
    # (D step: fake+real forward and backward = 6; G step: fake forward+backward = 3, real forward = 1).
    per_item = ecost * enc_s + 3 * sc["mac"] / sec * dec_s + 10 * dcost * seg_s
    log(f"discriminator {dcost / 1e9:.2f} GMAC/s, encoder {ecost / 1e9:.2f} GMAC/s -> about {per_item / 1e9:.0f} "
        f"GMAC per batch item per step (encoder {ecost * enc_s / per_item:.0%}, student "
        f"{3 * sc['mac'] / sec * dec_s / per_item:.0%}, discriminator {10 * dcost * seg_s / per_item:.0%})")

    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=tuple(a.betas))
    opt_d = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=tuple(a.betas))
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, a.gamma)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, a.gamma)
    step, ema = 0, {}
    ck = os.path.join(a.run, "ckpt_last.pt")
    if a.resume:
        s = torch.load(ck, map_location=dev, weights_only=False)
        G.load_state_dict(s["G"])
        D.load_state_dict(s["D"])
        opt_g.load_state_dict(s["opt_g"])
        opt_d.load_state_dict(s["opt_d"])
        sch_g.load_state_dict(s["sch_g"])
        sch_d.load_state_dict(s["sch_d"])
        set_rng_state(s["rng"])
        step, ema = s["step"], s["ema"]
        log(f"resumed from {ck} at step {step} (G {state_hash(G)})")
    G_ema = None
    if a.g_ema > 0:
        G_ema = copy.deepcopy(G).eval()
        for p in G_ema.parameters():
            p.requires_grad_(False)
        if a.resume and s.get("G_ema") is not None:
            G_ema.load_state_dict(s["G_ema"])
        elif a.resume:
            log("G_ema: none in the checkpoint, so it starts as a copy of G")

    lic = corpus.parse_licences(a.licences)
    log(f"licences: {sorted(lic)}")
    sources = None if a.eval_only else data.load_sources(a.manifest, lic, a.mix, a.mix_floor, log, a.path_map)
    crop = a.seg_frames * HOP + 2 * a.enc_ctx

    def ds_factory(B, start=0):
        return data.BatchStream(sources, B, crop, a.n_q_list, a.seed, start, a.tracks_per_batch, a.chunk_seconds)

    if a.bench:
        bench(a, teacher, G, D, mel, (opt_g, opt_d), (sch_g, sch_d), ds_factory, dev, log)
        return

    clips = prepare_eval(a, teacher, dev, log) if a.eval_set else []
    stock = stock_eval(a, teacher, clips, a.run, log) if clips else {}
    pico = json.load(open(a.picotone_json)) if a.picotone_json and os.path.exists(a.picotone_json) else None
    past = (corpus.read_jsonl(os.path.join(a.run, "evals.jsonl"))
            if os.path.exists(os.path.join(a.run, "evals.jsonl")) else [])
    # Pilot-verdict trend, per weights variant ("raw" = G, "ema" = G_ema; older rows have no "w" and are raw).
    history = {"raw": [], "ema": []}
    by_step = {}
    for r in past:
        w = r.get("w", "raw")
        if r["n_q"] == a.n_q_list[0]:
            history[w].append((r["step"], r["means"]))
        by_step.setdefault((r["step"], w), {})[r["n_q"]] = r["means"]
    # Best score per eval (best variant), for the plateau rule: only evals that covered every --eval-n-q.
    scores = []
    for st in sorted({k[0] for k in by_step}):
        sc = [sum(verdict.picotone_logmel(m[nq]) for nq in a.eval_n_q) / len(a.eval_n_q)
              for (s2, w), m in by_step.items() if s2 == st and all(nq in m for nq in a.eval_n_q)]
        if sc:
            scores.append((st, min(sc)))
    best_path = os.path.join(a.run, "ckpt_best.pt")
    best = {"score": float("inf")}
    if os.path.exists(best_path):
        b = torch.load(best_path, map_location="cpu", weights_only=False)
        if b.get("eval_n_q") == a.eval_n_q:
            best = {k: b[k] for k in ("score", "step", "w")}
            log(f"ckpt_best.pt: {best['w']} weights at step {best['step']}, score {best['score']:.4f}")
        else:
            log(f"ckpt_best.pt was scored on tiers {b.get('eval_n_q')}, not {a.eval_n_q}: it will be replaced")

    def evaluate(step):
        """-> True when a stop rule fires."""
        listen = os.path.join(a.run, "listen") if not a.listen_every or step % a.listen_every == 0 else None
        variants = [("raw", G)] + ([("ema", G_ema)] if G_ema is not None else [])
        G.eval()
        results = {}
        for w, model in variants:
            per = {}
            for nq in a.eval_n_q:
                t = time.perf_counter()
                means, rows = run_eval(a, teacher, model, clips, nq, f"s{step:07d}_{w}_q{nq}", listen)
                per[nq] = means
                jsonl(a.run, "evals.jsonl", {"step": step, "w": w, "n_q": nq, "means": means, "rows": rows})
                log(f"eval step {step} {w} q{nq} ({time.perf_counter() - t:.0f} s): " + fmt_means(means))
                sm = stock[str(nq)]["means"]
                log("   stock DAC          : " + fmt_means({k: v for k, v in sm.items() if k in means}))
                if nq == a.n_q_list[0] and any(s in means for s in verdict.SETS):
                    history[w].append((step, means))
                    rule = "pilot" if step <= a.pilot_steps else "full"
                    status, text = verdict.verdict(means, sm, history[w], pico, rule, nq)
                    log(f"[{w}] " + text)
                    jsonl(a.run, "verdicts.jsonl", {"step": step, "w": w, "rule": rule, "status": status})
            results[w] = per
        G.train()
        if not any(s in results["raw"][a.eval_n_q[0]] for s in verdict.SETS):
            return False
        # Every tier against the full-run rule, then the best-checkpoint and stop rules.
        summary = {}
        for w, per in results.items():
            oks = []
            for nq in a.eval_n_q:
                ok, line = verdict.tier_line(per[nq], stock[str(nq)]["means"], nq, "full")
                oks.append(ok)
                log(f"  [{w}] full-run rule {line}")
            summary[w] = (sum(verdict.picotone_logmel(per[nq]) for nq in a.eval_n_q) / len(a.eval_n_q), all(oks))
            log(f"  [{w}] score {summary[w][0]:.4f} (mean picotone log-mel over q{a.eval_n_q}), "
                f"{'ALL TIERS PASS' if summary[w][1] else 'not every tier passes yet'}")
        w_best = min(summary, key=lambda k: summary[k][0])
        score = summary[w_best][0]
        if score < best["score"] and not a.eval_only:
            best.update(score=score, step=step, w=w_best)
            model = G if w_best == "raw" else G_ema
            tmp = best_path + ".tmp"
            torch.save({"step": step, "w": w_best, "score": score, "eval_n_q": a.eval_n_q, "width": a.width,
                        "args": vars(a),
                        "G": model.state_dict(), "tiers": {str(nq): results[w_best][nq] for nq in a.eval_n_q}}, tmp)
            os.replace(tmp, best_path)
            log(f"  new best: {w_best} weights, score {score:.4f} -> ckpt_best.pt")
        scores.append((step, score))
        if a.stop_when_pass and summary[w_best][1]:
            log(f"stop: every tier passes the full-run rule ({w_best} weights)")
            return True
        if a.stop_plateau > 0:
            def avg_upto(s_end):
                v = [sc for st, sc in scores if st <= s_end][-4:]
                return sum(v) / len(v) if len(v) == 4 else None
            now, then = avg_upto(step), avg_upto(step - a.plateau_steps)
            if now is not None and then is not None:
                log(f"  plateau check: 4-eval average {then:.4f} at {step - a.plateau_steps} -> {now:.4f} "
                    f"(stop if better by < {a.stop_plateau})")
                if then - now < a.stop_plateau:
                    log("stop: the score has plateaued")
                    return True
        return False

    if a.eval_only:
        if not clips:
            raise SystemExit("--eval-only needs --eval-set")
        s = torch.load(a.eval_only, map_location="cpu", weights_only=False)
        if s.get("width", a.width) != a.width:
            raise SystemExit(f"checkpoint width {s['width']} != --width {a.width}")
        G.load_state_dict(s["G"])
        if G_ema is not None:
            if s.get("G_ema") is not None:
                G_ema.load_state_dict(s["G_ema"])
            else:
                G_ema = None  # an older checkpoint: raw weights only
        log(f"eval-only: {a.eval_only} at step {s['step']} (G {state_hash(G)})")
        evaluate(s["step"])
        return

    it = iter(data.loader(ds_factory(a.batch, step), a.workers))
    G.train()
    D.train()
    t_last, t_data = time.perf_counter(), 0.0
    if step == 0 and clips:
        evaluate(0)
    while step < a.steps:
        t0 = time.perf_counter()
        batch = next(it)
        assert batch["step"] == step, (batch["step"], step)
        t_data += time.perf_counter() - t0
        out = train_step(a, step, teacher, G, D, mel, opt_g, opt_d, sch_g, sch_d, batch, dev)
        if G_ema is not None:
            update_g_ema(G_ema, G, a.g_ema)
        step += 1
        for k, v in out.items():
            if math.isfinite(v):
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(out["total"]):
            log(f"non-finite loss at step {step}: {out}")
        if step % a.log_every == 0 or step == a.steps:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t_last) / a.log_every * 1000
            row = {"step": step, "ms": dt, "data_ms": t_data / a.log_every * 1000, "lr": sch_g.get_last_lr()[0],
                   **{k: round(v, 5) for k, v in ema.items()}}
            jsonl(a.run, "train.jsonl", row)
            log(f"step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in ema.items())
                + f" | {dt:.0f} ms/step (data wait {row['data_ms']:.0f}) lr {row['lr']:.3g}")
            t_last, t_data = time.perf_counter(), 0.0
        if step % a.ckpt_every == 0 or step == a.steps:
            save_ckpt(ck, step, G, D, (opt_g, opt_d), (sch_g, sch_d), ema, a, G_ema)
            if step % a.keep_every == 0:
                save_ckpt(os.path.join(a.run, f"ckpt_{step:07d}.pt"), step, G, D, (opt_g, opt_d), (sch_g, sch_d),
                          ema, a, G_ema)
        if clips and (step % a.eval_every == 0 or step == a.steps):
            if evaluate(step):
                if step % a.ckpt_every != 0:
                    save_ckpt(ck, step, G, D, (opt_g, opt_d), (sch_g, sch_d), ema, a, G_ema)
                break
    log(f"done at step {step}; G {state_hash(G)}")


if __name__ == "__main__":
    main()
