# SPDX-License-Identifier: GPL-3.0-or-later
"""Quality against total bitrate for an --arch sideinfo checkpoint: DAC codes (0.861 kbps per codebook) plus
the side code, per codebook tier, next to the student alone, stock DAC and (optionally) a zero-bit post-filter.

For every tier and every rate setting t on a grid (plus "off"), every eval clip is encoded and decoded, and the
side rate is measured from the entropy model on the rounded side latent (what an arithmetic coder would write,
up to a few bytes per file). Then:

  curve.json   every point: tier, t, side kbps, log-mel / mrstft / SI-SDR / SNR, per set
  curve.md     tables at side rates 0, 0.25, 0.5, 1, 2, 4 kbps (interpolated between grid points), and
               (a) qN + side at the rate of q(N+1) against q(N+1); (b) what each extra kbps buys at the top tier;
               (c) where the bits go: per set, against transients / high-frequency share / loudness per frame,
               and the log-mel gain per band (low < 500 Hz, mid, high > 4 kHz)
  curve.svg    log-mel against total kbps
  listen/      WAVs at --listen-kbps side rates (nearest grid point) for --listen-n-q tiers, with the original,
               the student and stock DAC

    uv run --extra cuda python sideinfo_curve.py --ckpt runs/si_s384/ckpt_last.pt --out runs/si_s384/curve \
        --eval-set data/pd.jsonl.eval.clean.json [--stock-json runs/.../stock_eval.json] [--pf-ckpt ...]

The q9 report (--points, for the two-stage plan: --si-mode ceiling, then constrained fine-tunes):

    uv run --extra cuda python sideinfo_curve.py --out runs/si_q9/report --eval-set data/pd.jsonl.eval.clean.json \
        --points ceiling=runs/si_q9_ceiling/ckpt_last.pt --points eps0.5=runs/si_q9_e005/ckpt_last.pt ...

  Every point is one fixed-rate checkpoint, evaluated at --n-q (default 9) with the side code on; the first
  point is also evaluated with it off (the zero-bit post-filter). Next to stock DAC and the student alone, it
  writes q9.md (log-mel / mrstft / SI-SDR against total kbps = q9's 7.75 + side; the gap to the ceiling; per
  content group and type; where the bits go), q9.json, q9.svg, and listening WAVs for --listen-points.
"""
import argparse
import json
import math
import os
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

import clip_gap
import data
import metrics
import postfilter
import sideinfo
import train
from teacher import load_teacher
from vocos_student import make_decoder

SR = 44100
KBPS_PER_CODEBOOK = SR / 512 * 10 / 1000  # 10-bit codes at 86.13 frames/s
BANDS = (("low <500 Hz", 0, 500), ("mid 0.5-4 kHz", 500, 4000), ("high >4 kHz", 4000, SR))


def mel_centres(n_mels=80, sr=SR):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    pts = 700 * (10 ** (np.linspace(hz2mel(30), hz2mel(sr / 2), n_mels + 2) / 2595) - 1)
    return pts[1:-1]


def band_err(ref, y):
    """Mean |log-mel error| (dB) per mel band, [80], over the clip's non-silent channels (metrics.score's
    definition before the mean over bands)."""
    fb = metrics._mel_fb(SR, 2048)
    acc, n = np.zeros(fb.shape[0]), 0
    n_s = min(ref.shape[-1], y.shape[-1])
    for r, x in zip(torch.from_numpy(np.ascontiguousarray(ref[:, :n_s], dtype=np.float32)),
                    torch.from_numpy(np.ascontiguousarray(y[:, :n_s], dtype=np.float32))):
        if float(r.abs().max()) < 1e-4:
            continue
        mr, my = fb @ metrics._mag(r, 2048), fb @ metrics._mag(x, 2048)
        floor = mr.max() * 1e-4
        acc += (20 * (torch.log10(my.clamp_min(floor)) - torch.log10(mr.clamp_min(floor)))).abs().mean(1).numpy()
        n += 1
    return acc / max(n, 1)


def frame_feats(x):
    """x [C, T] -> per channel and 512-sample frame: spectral flux (transients), share of energy above 6 kHz,
    level (dB). Each [C, F]."""
    X = torch.stft(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), 2048, 512,
                   window=torch.hann_window(2048), center=True, return_complex=True).abs()
    P = X.pow(2)
    lg = torch.log(X + 1e-5)
    flux = torch.relu(lg[..., 1:] - lg[..., :-1]).sum(1)
    flux = torch.cat([torch.zeros_like(flux[..., :1]), flux], -1)
    freqs = torch.linspace(0, SR / 2, X.shape[1])
    hf = P[:, freqs > 6000].sum(1) / P.sum(1).clamp_min(1e-12)
    level = 10 * torch.log10(P.sum(1).clamp_min(1e-12))
    return flux.numpy(), hf.numpy(), level.numpy()


def interp(points, kb, key, set_="all"):
    """Linear interpolation of a metric at side rate kb over the measured (kbps, means) points; None outside."""
    pts = sorted((p["means"][set_]["kbps"], p["means"][set_][key]) for p in points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if kb < xs[0] - 1e-9 or kb > xs[-1] + 1e-9:
        return None
    return float(np.interp(kb, xs, ys))


def svg_plot(series, path, title, x0=0.0):
    """series: [(label, colour, dashed, [(kbps, logmel)])] -> a small line chart."""
    W, H, L, R, T, B = 760, 460, 64, 190, 36, 48
    xs = [p[0] for _, _, _, pts in series for p in pts]
    ys = [p[1] for _, _, _, pts in series for p in pts]
    x1 = max(xs) + 0.05 * (max(xs) - x0)
    tick = next(s for s in (1, 2, 5, 10, 20, 50, 100) if (x1 - x0) / s <= 12)
    y0, y1 = min(ys) - 0.05 * (max(ys) - min(ys) + 1e-6), max(ys) + 0.05 * (max(ys) - min(ys) + 1e-6)

    def px(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(y):
        return T + (y1 - y) / (y1 - y0) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" '
         f'font-size="12"><rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="20" font-size="14">{title}</text>']
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="#ddd"/>'
                 f'<text x="{L - 6}" y="{py(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
    k = math.ceil(x0 / tick) * tick
    while k <= x1:
        o.append(f'<line x1="{px(k):.1f}" x2="{px(k):.1f}" y1="{T}" y2="{H - B}" stroke="#eee"/>'
                 f'<text x="{px(k):.1f}" y="{H - B + 16}" text-anchor="middle">{k}</text>')
        k += tick
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 10}" text-anchor="middle">total kbps per channel '
             f'(codes + side)</text><text x="14" y="{(T + H - B) / 2}" transform="rotate(-90 14 '
             f'{(T + H - B) / 2})" text-anchor="middle">log-mel distance (dB), lower is better</text>')
    for j, (label, col, dashed, pts) in enumerate(series):
        pts = sorted(pts)
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        o.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        for x, y in pts:
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{col}"/>')
        ly = T + 14 + 18 * j
        o.append(f'<line x1="{W - R + 12}" x2="{W - R + 36}" y1="{ly - 4}" y2="{ly - 4}" stroke="{col}" '
                 f'stroke-width="2"{dash}/><text x="{W - R + 42}" y="{ly}">{label}</text>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


# Content groups for the per-content breakdown, from the clip names (clip_gap.kind: old_14_piano -> piano).
# "band" (old public-domain band recordings) sits with orchestral as the other acoustic-ensemble material.
GROUPS = (("speech", {"speech"}), ("piano", {"piano"}), ("drums/rock", {"rock", "hiphop", "pop"}),
          ("electronic", {"electronic", "game", "chiptune"}), ("orchestral/band", {"orchestraLR", "band"}))


def group_of(name):
    k = clip_gap.kind(name)
    return next((g for g, kinds in GROUPS if k in kinds), "other")


def frame_stats(feats, fb_all, down):
    """Where the bits go: Spearman correlations of per-frame bits with flux / HF share / level, and the share
    of bits in the top-10% frames. fb_all: per clip [C, frames] bits."""
    bits, flux, hf, lev = [], [], [], []
    for (fl, h, lv), b in zip(feats, fb_all):
        b = np.repeat(b, down, axis=-1)
        n = min(b.shape[-1], fl.shape[-1])
        bits.append(b[:, :n].ravel())
        flux.append(fl[:, :n].ravel())
        hf.append(h[:, :n].ravel())
        lev.append(lv[:, :n].ravel())
    bits, flux, hf, lev = (np.concatenate(v) for v in (bits, flux, hf, lev))
    keep = lev > lev.max() - 70  # drop near-silent frames
    bits, flux, hf, lev = bits[keep], flux[keep], hf[keep], lev[keep]
    if bits.sum() <= 0:
        return {}
    from scipy.stats import spearmanr

    top = flux >= np.quantile(flux, 0.9)
    hi_hf = hf >= np.quantile(hf, 0.9)
    return {"rho_flux": float(spearmanr(bits, flux)[0]), "rho_hf": float(spearmanr(bits, hf)[0]),
            "rho_level": float(spearmanr(bits, lev)[0]), "share_top10_flux": float(bits[top].sum() / bits.sum()),
            "share_top10_hf": float(bits[hi_hf].sum() / bits.sum())}


def eval_point(ns, teacher, G, clips, nq, t, feats, centres, down):
    """Decode every clip at tier nq and rate setting t (None = off) -> a point dict."""
    G.eval_t = t
    rows, bands, fb_all = [], [], []
    for e in clips:
        G.bits = 0.0
        y = train.decode_clip(ns, teacher, G, e["codes"], nq, e["x"])
        m = metrics.score(e["x"], y)
        m["kbps"] = G.bits / (y.shape[-1] / SR) / y.shape[0] / 1000
        rows.append({"name": e["name"], "set": e["set"], "m": m})
        bands.append(band_err(e["x"], y))
        fb_all.append(G.frame_bits.cpu().numpy())
    be = np.mean(bands, 0)
    band = {name: float(be[(centres >= lo) & (centres < hi)].mean()) for name, lo, hi in BANDS}
    return {"n_q": nq, "t": t, "means": train.set_means(rows), "band": band,
            "frames": frame_stats(feats, fb_all, down) if t is not None else {}, "rows": rows}


def group_means(rows, key):
    """Per content group and per type: the mean of metric key (and the side kbps, if any)."""
    out = {}
    for label, f in (("group", group_of), ("type", lambda n: clip_gap.kind(n))):
        acc = {}
        for r in rows:
            acc.setdefault(f(r["name"]), []).append(r["m"])
        out[label] = {g: {"n": len(ms), key: float(np.mean([m[key] for m in ms])),
                          "kbps": float(np.mean([m.get("kbps", 0.0) for m in ms]))} for g, ms in acc.items()}
    return out


def q9_report(a, dev):
    """--points: the fixed-rate checkpoints of the two-stage plan, at one tier."""
    nq = int(a.n_q.split(",")[0]) if a.n_q else 9
    listen = os.path.join(a.out, "listen")
    pts_in = []
    for p in a.points:
        label, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"--points wants LABEL=CKPT, not {p}")
        pts_in.append((label, path))
    listen_pts = {v for v in a.listen_points.split(",") if v}
    teacher = load_teacher(dev)
    ns = clips = stock = stud = feats = None
    centres = mel_centres()
    points = []
    for i, (label, path) in enumerate(pts_in):
        G, s, w = load_model(path, a.w, dev)
        if not isinstance(G, sideinfo.SideInfoModel) or G.mode != "fixed":
            raise SystemExit(f"{path} is not a --si-mode ceiling/constrained checkpoint")
        arch, args = s["arch"], s.get("args", {})
        if ns is None:
            ns = SimpleNamespace(eval_set=a.eval_set, max_eval_clips=a.max_eval_clips,
                                 path_map=data.parse_path_map(a.path_map), listen=a.listen, eval_n_q=[nq],
                                 pf_student_sha=arch["student_sha"])
            ns.si_codes_enc = codes_enc_of(arch)
            clips = train.prepare_eval(ns, teacher, dev, print)
            for src, name in ((a.stock_json, "stock_eval.json"), (a.student_json, "student_eval.json")):
                if src and os.path.abspath(src) != os.path.abspath(os.path.join(a.out, name)):
                    shutil.copyfile(src, os.path.join(a.out, name))
            stock = train.stock_eval(ns, teacher, clips, a.out, print)
            stud = train.student_alone_eval(ns, teacher, G.student, clips, a.out, print)
            feats = [frame_feats(e["x"]) for e in clips]
        elif arch["student_sha"] != ns.pf_student_sha:
            raise SystemExit(f"{path} was trained on another student than the first point")
        elif arch.get("codes_enc_sha") != (postfilter.file_sha16(ns.si_codes_enc) if ns.si_codes_enc else None):
            raise SystemExit(f"{path} was trained on other codes than the first point")
        meta = {"label": label, "ckpt": path, "step": s["step"], "w": w, "mode": args.get("si_mode"),
                "eps": args.get("si_eps") if args.get("si_mode") == "constrained" else None,
                "lam": (s.get("si") or {}).get("lam"), "d_slow": (s.get("si") or {}).get("d_slow")}
        for t in ([None] if i == 0 else []) + [sideinfo.FIXED_T]:
            t0 = time.perf_counter()
            p = eval_point(ns, teacher, G, clips, nq, t, feats, centres, arch["down"])
            p.update(meta, label=label if t is not None else "side off (zero-bit post-filter)")
            points.append(p)
            al = p["means"]["all"]
            print(f"{p['label']}: side {al['kbps']:.3f} kbps, log-mel {al['logmel']:.3f}, mrstft {al['mrstft']:.3f}, "
                  f"SI-SDR {al['sisdr']:.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
            if (label in listen_pts or t is None) and a.listen:
                tag = "off" if t is None else label
                for e in clips[: a.listen]:
                    G.eval_t = t
                    G.bits = 0.0
                    y = train.decode_clip(ns, teacher, G, e["codes"], nq, e["x"])
                    kb = G.bits / (y.shape[-1] / SR) / y.shape[0] / 1000
                    train.write_wav(os.path.join(listen, f"{e['name']}__side_q{nq}_{tag}"
                                                 + ("" if t is None else f"_{kb:.2f}kbps") + ".wav"), y)
        del G
    json.dump({"n_q": nq, "points": points,
               "student": stud[str(nq)], "stock": stock[str(nq)]}, open(os.path.join(a.out, "q9.json"), "w"),
              indent=1)

    base_kb = nq * KBPS_PER_CODEBOOK
    st_m, sd_m = stud[str(nq)]["means"]["all"], stock[str(nq)]["means"]["all"]
    ceil = next((p for p in points if p["label"] == pts_in[0][0]), None)
    md = [f"# q{nq} + side information\n\n{len(clips)} clips; q{nq} codes are {base_kb:.2f} kbps per channel, side "
          f"rates are measured (the entropy model on the rounded code), kbps per channel. 'vs ceiling' is the "
          f"log-mel gap to the first point ({pts_in[0][0]}). Lower log-mel and mrstft are better, higher SI-SDR.\n",
          "| point | step | side kbps | total kbps | log-mel | vs ceiling | vs student | mrstft | SI-SDR | lambda |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    def line(label, step, kb, m, lam=None):
        vc = f"{m['logmel'] / ceil['means']['all']['logmel'] - 1:+.2%}" if ceil else ""
        return (f"| {label} | {step} | {kb:.3f} | {base_kb + kb:.2f} | {m['logmel']:.3f} | {vc} | "
                f"{m['logmel'] / st_m['logmel'] - 1:+.2%} | {m['mrstft']:.3f} | {m['sisdr']:.2f} | "
                f"{'' if lam is None else f'{lam:.3g}'} |")

    md.append(line(f"stock DAC q{nq}", "", 0.0, sd_m))
    md.append(line(f"student q{nq}", "", 0.0, st_m))
    for p in points:
        md.append(line(p["label"], p["step"], p["means"]["all"]["kbps"], p["means"]["all"],
                       p["lam"] if p["t"] is not None else None))
    on = [p for p in points if p["t"] is not None]
    if ceil and len(on) > 1:
        md.append(f"\n## Rate needed to stay within epsilon of the ceiling\n")
        for p in on[1:]:
            if p["eps"] is None:
                continue
            gap = p["means"]["all"]["logmel"] / ceil["means"]["all"]["logmel"] - 1
            md.append(f"- eps {p['eps']:.1%} (training target): {p['means']['all']['kbps']:.3f} kbps side, "
                      f"{p['means']['all']['kbps'] / max(ceil['means']['all']['kbps'], 1e-9):.0%} of the ceiling's; "
                      f"eval log-mel {gap:+.2%} vs the ceiling ({'within' if gap <= p['eps'] + 1e-12 else 'OUTSIDE'} "
                      f"eps on the eval clips)")

    md.append("\n## Per content group: log-mel (side kbps)\n")
    cols = [("stock", stock[str(nq)]["rows"]), ("student", stud[str(nq)]["rows"])] + [
        (p["label"], p["rows"]) for p in points]
    gm = {lbl: group_means(rows, "logmel") for lbl, rows in cols}
    order = [x[0] for x in GROUPS] + ["other"]
    for level in ("group", "type"):
        keys = sorted({g for v in gm.values() for g in v[level]},
                      key=lambda g: (order.index(g) if level == "group" and g in order else 99, g))
        md.append(f"\n| {level} | clips | " + " | ".join(lbl for lbl, _ in cols) + " |\n|---|---" + "|---" * len(cols)
                  + "|")
        for g in keys:
            cells = []
            for lbl, _ in cols:
                v = gm[lbl][level].get(g)
                cells.append("" if v is None else f"{v['logmel']:.3f}" + (f" ({v['kbps']:.2f})" if v["kbps"] else ""))
            n = next(gm[lbl][level][g]["n"] for lbl, _ in cols if g in gm[lbl][level])
            md.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    md.append("\nGroups: " + "; ".join(f"{g} = {', '.join(sorted(k))}" for g, k in GROUPS) + "; other = the rest.")

    md.append("\n## Where the bits go\n\nPer frame (near-silent frames dropped): Spearman correlation of the "
              "side bits with spectral flux (transients), the energy share above 6 kHz and the level; the share of "
              "all bits in the 10% of frames with the most flux / HF. Then the log-mel error per band (dB).\n")
    md.append("| point | side kbps | rho flux | rho HF | rho level | top-10% flux | top-10% HF | "
              + " | ".join(b[0] for b in BANDS) + " |\n|---|---|---|---|---|---|---|" + "---|" * len(BANDS))
    for p in points:
        f = p["frames"]
        fcells = (f"{f['rho_flux']:+.2f} | {f['rho_hf']:+.2f} | {f['rho_level']:+.2f} | {f['share_top10_flux']:.0%} | "
                  f"{f['share_top10_hf']:.0%}") if f else " | | | | "
        md.append(f"| {p['label']} | {p['means']['all']['kbps']:.3f} | {fcells} | "
                  + " | ".join(f"{p['band'][b[0]]:.2f}" for b in BANDS) + " |")
    open(os.path.join(a.out, "q9.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    series = [(f"stock DAC q{nq}", "#777777", True, [(base_kb, sd_m["logmel"])]),
              (f"student q{nq}", "#000000", False, [(base_kb, st_m["logmel"])]),
              (f"q{nq} + side", "#1f77b4", False,
               sorted((base_kb + p["means"]["all"]["kbps"], p["means"]["all"]["logmel"]) for p in points))]
    svg_plot(series, os.path.join(a.out, "q9.svg"), f"q{nq} + side information ({a.w} weights)",
             x0=base_kb - 0.5)
    print(open(os.path.join(a.out, "q9.md"), encoding="utf-8").read())


def codes_enc_of(arch):
    """patch si_enc: the trained encoder a checkpoint's codes came from ('' = DAC's encoder), checked by hash."""
    p = arch.get("codes_enc", "")
    if not p:
        return ""
    if not os.path.exists(p) or postfilter.file_sha16(p) != arch["codes_enc_sha"]:
        raise SystemExit(f"the codes' encoder {p} (sha256 {arch['codes_enc_sha']}...) is not here")
    print(f"codes from the trained encoder {p} (sha256 {arch['codes_enc_sha']}...)")
    return p


def load_model(path, w, dev):
    s = torch.load(path, map_location="cpu", weights_only=False)
    G = make_decoder(s["arch"])
    sd = s.get("G_ema") if w == "ema" and s.get("G_ema") is not None else s["G"]
    G.load_state_dict(sd)
    return G.to(dev).eval(), s, ("ema" if sd is s.get("G_ema") else "raw")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="", help="an --si-mode vr checkpoint (ckpt_last.pt / ckpt_best.pt)")
    ap.add_argument("--points", action="append", default=[], metavar="LABEL=CKPT",
                    help="the q9 report instead: fixed-rate checkpoints (ceiling first), repeatable")
    ap.add_argument("--listen-points", default="ceiling,eps1,eps2",
                    help="--points labels written as WAVs (plus side off, always)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-set", action="append", required=True)
    ap.add_argument("--path-map", action="append", default=[])
    ap.add_argument("--max-eval-clips", type=int, default=0)
    ap.add_argument("--n-q", default="", help="tiers (default 4,5,6,7,9; with --points the one tier, default 9)")
    ap.add_argument("--w", choices=("raw", "ema"), default="ema", help="weights (ema falls back to raw)")
    ap.add_argument("--t-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--targets", default="0.25,0.5,1,2,4", help="side kbps for the tables")
    ap.add_argument("--stock-json", default="", help="a run's stock_eval.json to reuse (else computed)")
    ap.add_argument("--student-json", default="", help="a run's student_eval.json to reuse (else computed)")
    ap.add_argument("--pf-ckpt", default="", help="optional: a zero-bit --arch postfilter checkpoint")
    ap.add_argument("--listen", type=int, default=4, help="clips written as WAVs")
    ap.add_argument("--listen-n-q", default="4,9")
    ap.add_argument("--listen-kbps", default="0.5,2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    listen = os.path.join(a.out, "listen")
    os.makedirs(listen, exist_ok=True)
    dev = a.device
    if a.points:
        return q9_report(a, dev)
    if not a.ckpt:
        raise SystemExit("give --ckpt (a variable-rate checkpoint) or --points")
    tiers = [int(v) for v in (a.n_q or "4,5,6,7,9").split(",")]
    grid = [float(v) for v in a.t_grid.split(",")]
    targets = [float(v) for v in a.targets.split(",")]
    G, s, w = load_model(a.ckpt, a.w, dev)
    if not isinstance(G, sideinfo.SideInfoModel):
        raise SystemExit(f"{a.ckpt} is not an --arch sideinfo checkpoint")
    arch = s["arch"]
    print(f"{a.ckpt}: step {s['step']}, {w} weights, side ch {arch['ch']} down {arch['down']}", flush=True)
    ns = SimpleNamespace(eval_set=a.eval_set, max_eval_clips=a.max_eval_clips,
                         path_map=data.parse_path_map(a.path_map), listen=a.listen, eval_n_q=tiers,
                         pf_student_sha=arch["student_sha"])
    ns.si_codes_enc = codes_enc_of(arch)
    teacher = load_teacher(dev)
    clips = train.prepare_eval(ns, teacher, dev, print)
    for src, name in ((a.stock_json, "stock_eval.json"), (a.student_json, "student_eval.json")):
        if src and os.path.abspath(src) != os.path.abspath(os.path.join(a.out, name)):
            shutil.copyfile(src, os.path.join(a.out, name))
    stock = train.stock_eval(ns, teacher, clips, a.out, print)
    stud = train.student_alone_eval(ns, teacher, G.student, clips, a.out, print)
    pf = {}
    if a.pf_ckpt:
        P, _, pw = load_model(a.pf_ckpt, a.w, dev)
        for nq in tiers:
            pf[nq] = train.run_eval(ns, teacher, P, clips, nq, f"0_pf0bit_q{nq}", listen)[0]
        del P

    feats = [frame_feats(e["x"]) for e in clips]
    centres = mel_centres()
    points = []  # one per (tier, t)
    for nq in tiers:
        for t in [None] + grid:
            t0 = time.perf_counter()
            G.eval_t = t
            rows, bands, fb_all = [], [], []
            for e in clips:
                G.bits = 0.0
                y = train.decode_clip(ns, teacher, G, e["codes"], nq, e["x"])
                m = metrics.score(e["x"], y)
                m["kbps"] = G.bits / (y.shape[-1] / SR) / y.shape[0] / 1000
                rows.append({"name": e["name"], "set": e["set"], "m": m})
                bands.append(band_err(e["x"], y))
                fb_all.append(G.frame_bits.cpu().numpy())
            means = train.set_means(rows)
            be = np.mean(bands, 0)
            band = {name: float(be[(centres >= lo) & (centres < hi)].mean()) for name, lo, hi in BANDS}
            stats = {}
            if t is not None:
                bits, flux, hf, lev = [], [], [], []
                for (fl, h, lv), b in zip(feats, fb_all):
                    b = np.repeat(b, arch["down"], axis=-1)
                    n = min(b.shape[-1], fl.shape[-1])
                    bits.append(b[:, :n].ravel())
                    flux.append(fl[:, :n].ravel())
                    hf.append(h[:, :n].ravel())
                    lev.append(lv[:, :n].ravel())
                bits, flux, hf, lev = (np.concatenate(v) for v in (bits, flux, hf, lev))
                keep = lev > lev.max() - 70  # drop near-silent frames
                bits, flux, hf, lev = bits[keep], flux[keep], hf[keep], lev[keep]
                if bits.sum() > 0:
                    from scipy.stats import spearmanr

                    top = flux >= np.quantile(flux, 0.9)
                    hi_hf = hf >= np.quantile(hf, 0.9)
                    stats = {"rho_flux": float(spearmanr(bits, flux)[0]), "rho_hf": float(spearmanr(bits, hf)[0]),
                             "rho_level": float(spearmanr(bits, lev)[0]),
                             "share_top10_flux": float(bits[top].sum() / bits.sum()),
                             "share_top10_hf": float(bits[hi_hf].sum() / bits.sum())}
            points.append({"n_q": nq, "t": t, "means": means, "band": band, "frames": stats,
                           "rows": [{"name": r["name"], "set": r["set"], "m": r["m"]} for r in rows]})
            al = means["all"]
            print(f"q{nq} t {'off' if t is None else f'{t:.2f}'}: side {al['kbps']:.3f} kbps, log-mel "
                  f"{al['logmel']:.3f}, mrstft {al['mrstft']:.3f}, SI-SDR {al['sisdr']:.2f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    json.dump({"ckpt": a.ckpt, "step": s["step"], "w": w, "arch": arch, "points": points,
               "student": {k: v["means"] for k, v in stud.items() if k != "student_sha"},
               "stock": {k: v["means"] for k, v in stock.items()}, "pf": {str(k): v for k, v in pf.items()}},
              open(os.path.join(a.out, "curve.json"), "w"), indent=1)

    # Listening WAVs: the grid point nearest each requested side rate.
    for nq in [int(v) for v in a.listen_n_q.split(",") if v and int(v) in tiers]:
        for kb in [float(v) for v in a.listen_kbps.split(",") if v]:
            cand = [p for p in points if p["n_q"] == nq and p["t"] is not None]
            p = min(cand, key=lambda p: abs(p["means"]["all"]["kbps"] - kb))
            G.eval_t = p["t"]
            for e in clips[: a.listen]:
                G.bits = 0.0
                y = train.decode_clip(ns, teacher, G, e["codes"], nq, e["x"])
                kbps = G.bits / (y.shape[-1] / SR) / y.shape[0] / 1000
                train.write_wav(os.path.join(listen, f"{e['name']}__side_q{nq}_{kbps:.2f}kbps.wav"), y)
        G.eval_t = None
        for e in clips[: a.listen]:
            y = train.decode_clip(ns, teacher, G, e["codes"], nq, e["x"])
            train.write_wav(os.path.join(listen, f"{e['name']}__side_q{nq}_off.wav"), y)

    # Tables.
    md = [f"# Side information: quality against rate\n\n{a.ckpt}, step {s['step']}, {w} weights, "
          f"{len(clips)} clips. Codes are {KBPS_PER_CODEBOOK:.3f} kbps per codebook; side rates are measured "
          f"(entropy model on the rounded side latent), kbps per channel, interpolated between grid points.\n"]
    for key, name in (("logmel", "log-mel (dB)"), ("mrstft", "mrstft"), ("sisdr", "SI-SDR (dB)")):
        md.append(f"\n## {name}\n\n| tier | stock DAC | student | " + ("zero-bit PF | " if pf else "")
                  + "side off = zero-bit post-filter | " +" | ".join(f"+{kb:g} kbps" for kb in targets) + " |")
        md.append("|---" * (4 + bool(pf) + len(targets)) + "|")
        for nq in tiers:
            pts = [p for p in points if p["n_q"] == nq]
            off = next(p for p in pts if p["t"] is None)["means"]["all"][key]
            row = [f"q{nq} ({nq * KBPS_PER_CODEBOOK:.2f})", f"{stock[str(nq)]['means']['all'][key]:.3f}",
                   f"{stud[str(nq)]['means']['all'][key]:.3f}"]
            if pf:
                row.append(f"{pf[nq]['all'][key]:.3f}")
            row.append(f"{off:.3f}")
            for kb in targets:
                v = interp(pts, kb, key)  # the off point is the curve's 0 kbps end
                row.append("—" if v is None else f"{v:.3f}")
            md.append("| " + " | ".join(row) + " |")
    md.append("\nMeasured side-rate range per tier (kbps): " + ", ".join(
        f"q{nq} {min(p['means']['all']['kbps'] for p in points if p['n_q'] == nq and p['t'] is not None):.3f}"
        f"–{max(p['means']['all']['kbps'] for p in points if p['n_q'] == nq):.3f}" for nq in tiers))

    md.append("\n## Zero-bit answer: side off (the zero-bit post-filter) against the student alone\n")
    for nq in tiers:
        off = next(p for p in points if p["n_q"] == nq and p["t"] is None)["means"]["all"]
        st = stud[str(nq)]["means"]["all"]
        md.append(f"- q{nq}: log-mel {st['logmel']:.3f} -> {off['logmel']:.3f} ({off['logmel'] / st['logmel'] - 1:+.2%}), "
                  f"SI-SDR {st['sisdr']:.2f} -> {off['sisdr']:.2f} ({off['sisdr'] - st['sisdr']:+.2f} dB)")
    md.append("- bar: at least 1.0% better log-mel at every tier, SI-SDR no worse than -0.1 dB")

    md.append("\n## (a) qN + side at the rate of the next tier\n")
    for lo, hi in zip(tiers, tiers[1:]):
        kb = (hi - lo) * KBPS_PER_CODEBOOK
        pts = [p for p in points if p["n_q"] == lo]
        v = interp(pts, kb, "logmel")
        ref_s, ref_d = stud[str(hi)]["means"]["all"]["logmel"], stock[str(hi)]["means"]["all"]["logmel"]
        if v is None:
            md.append(f"- q{lo} + {kb:.2f} kbps: outside the measured side rates; q{hi} student {ref_s:.3f}, "
                      f"stock {ref_d:.3f}")
        else:
            md.append(f"- q{lo} + {kb:.2f} kbps side: log-mel {v:.3f} vs q{hi} student {ref_s:.3f} "
                      f"({v / ref_s - 1:+.1%}) and stock {ref_d:.3f} ({v / ref_d - 1:+.1%}): "
                      f"{'BEATS' if v < ref_s else 'does not beat'} the student's next tier")

    top = max(tiers)
    md.append(f"\n## (b) Above q{top}: what each extra kbps buys\n")
    pts = [p for p in points if p["n_q"] == top]
    prev_kb, prev = 0.0, next(p for p in points if p["n_q"] == top and p["t"] is None)["means"]["all"]["logmel"]
    for kb in targets:
        v = interp(pts, kb, "logmel")
        if v is None:
            continue
        md.append(f"- {prev_kb:g} -> {kb:g} kbps: log-mel {prev:.3f} -> {v:.3f}, "
                  f"{(v - prev) / (kb - prev_kb):+.3f} dB per kbps ({v / prev - 1:+.1%})")
        prev_kb, prev = kb, v
    md.append(f"- for scale: q{tiers[-2] if len(tiers) > 1 else top} -> q{top} on the student costs "
              f"{(top - (tiers[-2] if len(tiers) > 1 else top)) * KBPS_PER_CODEBOOK:.2f} kbps for "
              f"{stud[str(tiers[-2] if len(tiers) > 1 else top)]['means']['all']['logmel']:.3f} -> "
              f"{stud[str(top)]['means']['all']['logmel']:.3f}")

    md.append("\n## (c) Where the bits go\n")
    md.append("Per frame (all clips and channels, near-silent frames dropped): Spearman correlation of the "
              "frame's side bits with its spectral flux (transients), its energy share above 6 kHz, and its "
              "level; and the share of all bits in the 10% of frames with the most flux / the most HF.\n")
    md.append("| tier | t | side kbps | rho flux | rho HF | rho level | bits in top-10% flux | bits in top-10% HF |"
              "\n|---|---|---|---|---|---|---|---|")
    for p in points:
        f = p["frames"]
        if f:
            md.append(f"| q{p['n_q']} | {p['t']:.2f} | {p['means']['all']['kbps']:.3f} | {f['rho_flux']:+.2f} | "
                      f"{f['rho_hf']:+.2f} | {f['rho_level']:+.2f} | {f['share_top10_flux']:.0%} | "
                      f"{f['share_top10_hf']:.0%} |")
    md.append("\nLog-mel error per band (dB), side off vs the highest-rate point:\n\n| tier | "
              + " | ".join(f"{b[0]} off -> max" for b in BANDS) + " |\n|---" + "|---" * len(BANDS) + "|")
    for nq in tiers:
        pts = [p for p in points if p["n_q"] == nq]
        off = next(p for p in pts if p["t"] is None)
        hi = max((p for p in pts if p["t"] is not None), key=lambda p: p["means"]["all"]["kbps"])
        md.append(f"| q{nq} | " + " | ".join(
            f"{off['band'][b[0]]:.2f} -> {hi['band'][b[0]]:.2f} ({hi['band'][b[0]] / off['band'][b[0]] - 1:+.0%})"
            for b in BANDS) + f" (at {hi['means']['all']['kbps']:.2f} kbps) |")
    sets = sorted(k for k in points[0]["means"] if k != "all")
    if len(sets) > 1:
        md.append("\nSide kbps per eval set at t = 0.5 (q" + str(top) + "): " + ", ".join(
            f"{st} {next(p for p in points if p['n_q'] == top and p['t'] == 0.5)['means'][st]['kbps']:.3f}"
            for st in sets) if any(p["t"] == 0.5 for p in points) else "")
    open(os.path.join(a.out, "curve.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    series = [("stock DAC q4..q9", "#777777", True,
               [(nq * KBPS_PER_CODEBOOK, stock[str(nq)]["means"]["all"]["logmel"]) for nq in tiers]),
              ("student q4..q9", "#000000", False,
               [(nq * KBPS_PER_CODEBOOK, stud[str(nq)]["means"]["all"]["logmel"]) for nq in tiers])]
    for i, nq in enumerate(tiers):
        series.append((f"q{nq} + side", cols[i % len(cols)], False,
                       [(nq * KBPS_PER_CODEBOOK + p["means"]["all"]["kbps"], p["means"]["all"]["logmel"])
                        for p in points if p["n_q"] == nq]))
    if pf:
        series.append(("zero-bit post-filter", "#bcbd22", True,
                       [(nq * KBPS_PER_CODEBOOK, pf[nq]["all"]["logmel"]) for nq in tiers]))
    svg_plot(series, os.path.join(a.out, "curve.svg"), f"Side information, step {s['step']} ({w})")
    print(open(os.path.join(a.out, "curve.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
