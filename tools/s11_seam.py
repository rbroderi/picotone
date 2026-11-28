"""Segment-seam artifact of the 48 kHz model (1 s segments, stride 0.99 s) and fixes, ranked per bit.

usage: uv run --extra parity python s11_seam.py [n_q] [group ...]        (n_q default 16 = 24 kbps; 4 = 6 kbps; groups: diag dec enc fmt)
env  : PICOTONE_PROBE_LISTEN=<dir>  use the two clips <dir>/{orchestral,ambient-electronic}-0-original.wav (first 30 s) instead of
                         the test set, and write every variant's output to work/encodec_probe/out/listen/ as 16-bit WAV.

The codec cuts the audio into 48000-sample segments every 47520 samples, normalises each by its RMS, codes
each independently (reflect padding, LSTM state zero at the segment start) and cross-fades the 480-sample
overlap.  Quality is measured BY SEGMENT PHASE (period 47520 samples for every variant, so numbers compare):
SNR against the source and a short-time log-spectral distance in 10 ms bins; 'start' = first 20 ms after a
stock seam position, 'mid' = median of the middle, 'end' = last 20 ms.

Variant tags:
  [dec]  decoder-only, stock bitstream
  [enc]  encoder-only, stock bitstream layout (codes differ)
  [fmt]  different segmentation -> new container variant; the total bitrate (overlap overhead included) is in 'kbps'
  [W]    a written constant (mean state); nothing else here is solved or trained.
"""
import os
import re
import sys
import wave

import numpy as np
import torch

from encodec_eval import OUT, Log, get_lstm, load_clips, load_model, perceptual, run_layers

args = sys.argv[1:]
N_Q = int(args[0]) if args and args[0].isdigit() else 16
groups = [a for a in args if not a.isdigit()] or ["diag", "dec", "enc", "fmt"]
LISTEN = os.environ.get("PICOTONE_PROBE_LISTEN")
log = Log(f"s11_seam_nq{N_Q}" + ("_listenclips" if LISTEN else "") + ("" if len(groups) == 4 else "_" + "_".join(groups)))
m = load_model("48khz")
if LISTEN:
    xs = []
    for clip in ("orchestral", "ambient-electronic"):
        with wave.open(os.path.join(LISTEN, f"{clip}-0-original.wav")) as w:
            pcm = np.frombuffer(w.readframes(30 * 48000), "<i2").reshape(-1, 2).T
        xs.append(pcm.astype(np.float32) / 32768)
    X = torch.from_numpy(np.stack(xs))
    names = ["orchestral", "ambient-electronic"]
else:
    names, _, x48 = load_clips("test")
    X = torch.from_numpy(x48)                  # [clips, 2, 480000]
C, _, N = X.shape
HOP = 320
SEG = 48000
enc_idx, enc_ps = get_lstm(m, "enc")
dec_idx, dec_ps = get_lstm(m, "dec")
E = m.encoder.layers
Dl = m.decoder.layers
MEAN_STATE = {}


def lstm_run(x, ps, init=None, grab=None):
    """x [B,512,T] -> x + LSTM(x).  init: list of (h,c) per layer.  Also returns the states after `grab` frames."""
    inp = x.permute(2, 0, 1)
    T, B, D = inp.shape
    states = []
    for l, p in enumerate(ps):
        h, c = (torch.zeros(B, D), torch.zeros(B, D)) if init is None else init[l]
        pre = inp @ p["w_ih"].T + p["b"]
        wt = p["w_hh"].T.contiguous()
        hs = torch.empty(T, B, D)
        g_state = None
        for t in range(T):
            z = pre[t] + h @ wt
            i, f, g, o = z[:, :D].sigmoid(), z[:, D:2 * D].sigmoid(), z[:, 2 * D:3 * D].tanh(), z[:, 3 * D:].sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            hs[t] = h
            if grab is not None and t == grab - 1:
                g_state = (h.clone(), c.clone())
        states.append(g_state)
        inp = hs
    return x + inp.permute(1, 2, 0), states


def bat(fn, x, bs=16):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def offsets(stride):
    return list(range(0, N - SEG + 1, stride))


def _init(mode, si, state, side, B):
    init = state if (mode == "warm" and si > 0) else (MEAN_STATE[side] if mode == "mean" else None)
    if init is not None and init[0][0].shape[0] != B:
        init = [(h.expand(B, -1), c.expand(B, -1)) for h, c in init]
    return init


def encode(stride, mode="stock", K=0, nq=None):
    """Per-segment quantised latents and scales.  Segments are processed in order so that state / context can be
    chained from the previous one, as a streaming encoder would do."""
    lat, scales = [], []
    state = None
    grab = stride // HOP                                    # frames of this segment that precede the next segment
    offs = offsets(stride)
    if mode != "warm":
        # nothing is chained through time here, so all segments of all clips go through as one batch
        segs = torch.stack([X[:, :, o:o + SEG] for o in offs])                        # [S,C,2,SEG]
        sc = segs.mean(2, keepdim=True).pow(2).mean(3, keepdim=True).sqrt() + 1e-8   # [S,C,1,1]

        def enc_plain(z):
            pre = run_layers(E, z, 0, enc_idx)
            y, _ = lstm_run(pre, enc_ps, _init(mode, 0, None, "enc", pre.shape[0]))
            return run_layers(E, y, enc_idx + 1)

        if mode == "ctx":
            first = bat(enc_plain, segs[0] / sc[0])
            rest = torch.stack([X[:, :, o - K * HOP:o + SEG] for o in offs[1:]]) / sc[1:]
            emb = torch.cat([first, bat(enc_plain, rest.flatten(0, 1))[:, :, K:]])
        else:
            emb = bat(enc_plain, (segs / sc).flatten(0, 1))
        q = m.quantizer.decode(m.quantizer.encode(emb, None)[: (nq or N_Q)])
        q = q.reshape(len(offs), C, 128, -1)
        return list(q), list(sc)
    for si, off in enumerate(offs):
        seg = X[:, :, off:off + SEG]
        scale = seg.mean(1, keepdim=True).pow(2).mean(2, keepdim=True).sqrt() + 1e-8
        if mode == "ctx" and si > 0:
            inp = X[:, :, off - K * HOP:off + SEG] / scale   # real preceding audio instead of reflect padding + zero state
            y, _ = lstm_run(run_layers(E, inp, 0, enc_idx), enc_ps)
            emb = run_layers(E, y, enc_idx + 1)[:, :, K:]
        else:
            pre = run_layers(E, seg / scale, 0, enc_idx)
            y, state = lstm_run(pre, enc_ps, _init(mode, si, state, "enc", pre.shape[0]), grab)
            emb = run_layers(E, y, enc_idx + 1)
        codes = m.quantizer.encode(emb, None)[: (nq or N_Q)]
        lat.append(m.quantizer.decode(codes))
        scales.append(scale)
    return lat, scales


def decode(lat, scales, stride, mode="stock", K=0, rescale=False):
    out = []
    state = None
    grab = stride // HOP
    if mode != "warm":
        def dec_plain(z):
            pre = Dl[0](z)
            y, _ = lstm_run(pre, dec_ps, _init(mode, 0, None, "dec", pre.shape[0]))
            return run_layers(Dl, y, dec_idx + 1)

        L = torch.stack(lat)                                 # [S,C,128,T]
        S_ = torch.stack(scales)
        if mode == "ctx":
            ctx = L[:-1, :, :, grab - K:grab]
            if rescale:
                ctx = ctx * (S_[:-1] / S_[1:])
            first = bat(dec_plain, L[0])
            rest = bat(dec_plain, torch.cat([ctx, L[1:]], 3).flatten(0, 1))[:, :, K * HOP:]
            w = torch.cat([first[:, :, :SEG], rest[:, :, :SEG]])
        else:
            w = bat(dec_plain, L.flatten(0, 1))[:, :, :SEG]
        w = w.reshape(len(lat), C, 2, SEG) * S_
        return list(w)
    for si, (q, scale) in enumerate(zip(lat, scales)):
        if mode == "ctx" and si > 0:
            ctx = lat[si - 1][:, :, grab - K:grab]
            if rescale:
                ctx = ctx * (scales[si - 1] / scale)
            y, _ = lstm_run(Dl[0](torch.cat([ctx, q], 2)), dec_ps)
            w = run_layers(Dl, y, dec_idx + 1)[:, :, K * HOP:]
        else:
            pre = Dl[0](q)
            y, state = lstm_run(pre, dec_ps, _init(mode, si, state, "dec", pre.shape[0]), grab)
            w = run_layers(Dl, y, dec_idx + 1)
        out.append(w[:, :, :SEG] * scale)
    return out


def overlap_add(frames, stride, fade="meta"):
    total = stride * (len(frames) - 1) + SEG
    out = torch.zeros(C, 2, total, dtype=torch.float64)
    sw = torch.zeros(total, dtype=torch.float64)
    ov = SEG - stride
    if fade == "meta":                                      # Meta's triangle over the whole segment
        t = torch.linspace(0, 1, SEG + 2, dtype=torch.float64)[1:-1]
        w = 0.5 - (t - 0.5).abs()
    else:
        w = torch.ones(SEG, dtype=torch.float64)
        if ov > 0 and fade == "prev":                       # trust the END of the previous segment through the overlap
            w[:ov] = 1e-9
        elif ov > 0 and fade == "skip":                     # skip the first part of the new segment, fade over the last third of the overlap
            n = max(2, ov // 3)
            ramp = torch.linspace(0, 1, n + 2, dtype=torch.float64)[1:-1]
            w[:ov - n] = 1e-9
            w[ov - n:ov] = ramp
            w[SEG - n:] = 1 - ramp
        elif ov > 0 and fade == "mid":
            # 'edge-avoiding' fade: never use the last 10 ms of the previous segment nor (if the overlap allows) the first
            # 30 ms of the new one; cross-fade linearly in between.
            hi = ov - 480 if ov > 960 else ov
            lo = max(0, min(1440, hi - 240))
            ramp = torch.linspace(0, 1, hi - lo + 2, dtype=torch.float64)[1:-1]
            w[:lo] = 1e-9
            w[lo:hi] = ramp
            w[SEG - ov + lo:SEG - ov + hi] = 1 - ramp
            w[SEG - ov + hi:] = 1e-9
    for i, f in enumerate(frames):
        out[:, :, i * stride:i * stride + SEG] += w * f.double()
        sw[i * stride:i * stride + SEG] += w
    return (out / sw).float()


# phase period = the variant's own stride (seam spacing); 47520 when there are no seams


def phase_report(y, PH=47520):
    """SNR and log-spectral distance by position inside the STOCK stride (47520), first period excluded."""
    n_per = y.shape[-1] // PH
    a, b = PH, PH * n_per
    ref = X[:, :, :y.shape[-1]]
    win = 480
    bins = PH // win
    err = (y - ref).pow(2).sum(1)[:, a:b].double()
    sig = ref.pow(2).sum(1)[:, a:b].double()
    # per clip: energy pooled over that clip's seams; then the dB values are averaged over clips, so that a loud
    # clip does not hide the others (clips that are silent in a bin are floored)
    e = err.reshape(C, -1, PH)[:, :, :bins * win].reshape(C, -1, bins, win).mean(3).mean(1)
    s = sig.reshape(C, -1, PH)[:, :, :bins * win].reshape(C, -1, bins, win).mean(3).mean(1)
    snr = (10 * torch.log10(s.clamp_min(1e-12) / e.clamp_min(1e-12))).mean(0)
    wdw = torch.hann_window(512)
    Y = torch.stft(y.reshape(-1, y.shape[-1]), 512, 120, window=wdw, return_complex=True).abs()
    R = torch.stft(ref.reshape(-1, ref.shape[-1]), 512, 120, window=wdw, return_complex=True).abs()
    floor = R.amax(dim=(1, 2), keepdim=True) * 1e-4
    lsd = (20 * (torch.log10(Y.clamp_min(floor)) - torch.log10(R.clamp_min(floor)))).abs().mean(1)
    centre = torch.arange(lsd.shape[1]) * 120
    ok = (centre >= a) & (centre < b)
    ph = (((centre[ok] - a) % PH) // win).clamp_max(bins - 1)
    lsd_ph = torch.zeros(bins).index_add_(0, ph, lsd[:, ok].mean(0)) / torch.bincount(ph, minlength=bins).clamp_min(1)
    mid = slice(10, bins - 10)
    return {"snr_start": float(snr[:2].mean()), "snr_mid": float(snr[mid].median()), "snr_end": float(snr[-2:].mean()),
            "snr_dip": float(snr[mid].median() - min(snr[:2].mean(), snr[-2:].mean())),
            "lsd_start": float(lsd_ph[:2].mean()), "lsd_mid": float(lsd_ph[mid].median()),
            "lsd_bump": float(lsd_ph[:2].mean() - lsd_ph[mid].median()),
            "profile": [round(float(v), 1) for v in snr[:6]] + ["..."] + [round(float(v), 1) for v in snr[-3:]]}


def write_wav(path, x):
    pcm = np.round(np.clip(x.numpy(), -1, 1).T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(pcm.tobytes())


COUNT = [0]


def report(label, frames, stride, fade="meta", kbps=None, y=None):
    if y is None:
        y = overlap_add(frames, stride, fade)
    r = phase_report(y, stride if stride < y.shape[-1] - 480 else 47520)
    ref = X[:, :, :y.shape[-1]]
    p = perceptual(ref.reshape(-1, ref.shape[-1]), y.reshape(-1, y.shape[-1]), 48000)
    row = {**{k: v for k, v in r.items() if k != "profile"}, **p, "kbps": kbps if kbps is not None else N_Q * 1.5 * SEG / stride}
    log(label, row)
    print("      SNR per 10 ms bin, after the seam ... before the next:", r["profile"], flush=True)
    if LISTEN:
        d = os.path.join(OUT, "listen")
        os.makedirs(d, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:70]
        for i, nme in enumerate(names):
            if COUNT[0] == 0:
                write_wav(os.path.join(d, f"{nme}__00_source.wav"), X[i])
            write_wav(os.path.join(d, f"{nme}__{N_Q * 1.5:g}kbps_{COUNT[0] + 1:02d}_{slug}.wav"), y[i])
    COUNT[0] += 1
    return y


STRIDE = 47520

# mean steady-state LSTM states for the 'written initial state' variant, recorded 100 frames into a segment
seg = X[:, :, :SEG]
sc = seg.mean(1, keepdim=True).pow(2).mean(2, keepdim=True).sqrt() + 1e-8
y_, st = lstm_run(run_layers(E, seg / sc, 0, enc_idx), enc_ps, None, 100)
MEAN_STATE["enc"] = [(h.mean(0, keepdim=True), c.median(0, keepdim=True).values) for h, c in st]
q_ = m.quantizer.decode(m.quantizer.encode(run_layers(E, y_, enc_idx + 1), None)[:N_Q])
_, st = lstm_run(Dl[0](q_), dec_ps, None, 100)
MEAN_STATE["dec"] = [(h.mean(0, keepdim=True), c.median(0, keepdim=True).values) for h, c in st]

lat0, sc0 = encode(STRIDE)
stock_frames = decode(lat0, sc0, STRIDE)
report("stock: 1 s segments, overlap 1 %, Meta cross-fade", stock_frames, STRIDE)

if "diag" in groups:
    latE, scE = encode(STRIDE, "ctx", 32)
    report("diag: encoder gets 32 frames of real audio context, decoder stock", decode(latE, scE, STRIDE), STRIDE)
    report("diag: decoder gets 32 latent context frames, encoder stock", decode(lat0, sc0, STRIDE, "ctx", 32), STRIDE)
    report("diag: both get 32 frames of context", decode(latE, scE, STRIDE, "ctx", 32), STRIDE)

if "dec" in groups:
    report("[dec] cross-fade: previous tail used through the overlap", stock_frames, STRIDE, "prev")
    report("[dec][W] decoder LSTM starts from a written mean state", decode(lat0, sc0, STRIDE, "mean"), STRIDE)
    report("[dec] decoder LSTM state carried over from the previous segment", decode(lat0, sc0, STRIDE, "warm"), STRIDE)
    for K in (2, 8, 32, 75):
        report(f"[dec] {K} latent context frames from the previous segment", decode(lat0, sc0, STRIDE, "ctx", K), STRIDE)
    report("[dec] 8 latent context frames, context rescaled by the RMS ratio", decode(lat0, sc0, STRIDE, "ctx", 8, True), STRIDE)
    report("[dec] 8 latent context frames + previous-tail cross-fade", decode(lat0, sc0, STRIDE, "ctx", 8), STRIDE, "prev")

if "enc" in groups:
    latW, scW = encode(STRIDE, "warm")
    report("[enc] encoder LSTM state carried over, decoder stock", decode(latW, scW, STRIDE), STRIDE)
    report("[enc+dec] LSTM state carried over on both sides", decode(latW, scW, STRIDE, "warm"), STRIDE)
    for K in (8, 32):
        latE, scE = encode(STRIDE, "ctx", K)
        report(f"[enc] {K} frames of real audio context in the encoder, decoder stock", decode(latE, scE, STRIDE), STRIDE)
        report(f"[enc+dec] {K} context frames on both sides", decode(latE, scE, STRIDE, "ctx", K), STRIDE)
        if K == 32:
            report("[enc+dec] 32 context frames on both sides + previous-tail cross-fade", decode(latE, scE, STRIDE, "ctx", K), STRIDE, "prev")

if "align" in groups:
    # frame-aligned segmentation (stride a multiple of the 320-sample hop): context from the previous segment lines up exactly
    l0, s0 = encode(48000)
    report("[fmt] stride 48000, hard cut, nothing else changed", decode(l0, s0, 48000), 48000, "prev")
    for K in (4, 32):
        report(f"[fmt] stride 48000, decoder-only: {K} latent context frames", decode(l0, s0, 48000, "ctx", K), 48000, "prev")
        lE, sE = encode(48000, "ctx", K)
        report(f"[fmt] stride 48000, encoder-only: {K} audio context frames", decode(lE, sE, 48000), 48000, "prev")
        report(f"[fmt] stride 48000, {K} context frames on both sides", decode(lE, sE, 48000, "ctx", K), 48000, "prev")
        report(f"[fmt] stride 48000, {K} context frames on both sides, context rescaled", decode(lE, sE, 48000, "ctx", K, True), 48000, "prev")

if "key" in groups:
    # the three candidates worth confirming on the full test set (a subset of the 'ov' sweep below)
    lat, sc_ = encode(45600)
    report("[fmt] overlap 5.0 % (stride 45600), edge-avoiding cross-fade", decode(lat, sc_, 45600), 45600, "mid")
    if N_Q == 16:
        lat, sc_ = encode(45000, nq=15)
        report("[fmt] 15 codebooks, overlap 6.25 % (stride 45000), edge-avoiding cross-fade", decode(lat, sc_, 45000), 45000, "mid", kbps=15 * 1.5 * 48000 / 45000)
    SEG = 96000
    lat, sc_ = encode(SEG - 2400)
    report("[fmt] 2 s segments, 2400-sample overlap, edge-avoiding cross-fade", decode(lat, sc_, SEG - 2400), SEG - 2400, "mid")
    SEG = 48000

if "ov" in groups:
    # overlap length x cross-fade shape x segment length.  'kbps' always includes the overlap overhead.
    eq = {16: [(15, 45000)], 8: [(7, 42000)], 4: [(3, 36000)]}.get(N_Q, [])
    for nq, stride in eq:                                   # same TOTAL bitrate as stock: pay for the overlap with a codebook
        lat, sc_ = encode(stride, nq=nq)
        fr = decode(lat, sc_, stride)
        k = nq * 1.5 * SEG / stride
        report(f"[fmt] {nq} codebooks, overlap {100 * (1 - stride / SEG):.2f} % (stride {stride}), Meta cross-fade", fr, stride, kbps=k)
        report(f"[fmt] {nq} codebooks, overlap {100 * (1 - stride / SEG):.2f} % (stride {stride}), edge-avoiding cross-fade", fr, stride, "mid", kbps=k)
    for stride in (47040, 46560, 45600, 44400):             # 2 %, 3 %, 5 %, 7.5 % overlap, all codebooks kept
        lat, sc_ = encode(stride)
        fr = decode(lat, sc_, stride)
        if stride in (47040, 45600):
            report(f"[fmt] overlap {100 * (1 - stride / SEG):.1f} % (stride {stride}), Meta cross-fade", fr, stride)
        report(f"[fmt] overlap {100 * (1 - stride / SEG):.1f} % (stride {stride}), edge-avoiding cross-fade", fr, stride, "mid")
    SEG_SAVE = SEG
    for seg_len, ov in ((96000, 480), (96000, 2400), (144000, 2400), (192000, 2400)):
        SEG = seg_len
        stride = SEG - ov
        lat, sc_ = encode(stride)
        fr = decode(lat, sc_, stride)
        report(f"[fmt] {SEG // 48000} s segments, {ov}-sample overlap, {'Meta' if ov == 480 else 'edge-avoiding'} cross-fade", fr, stride, "meta" if ov == 480 else "mid")
    SEG = (N // HOP) * HOP
    lat, sc_ = encode(SEG)
    report("[fmt] whole clip as ONE segment (no seams)", decode(lat, sc_, SEG), SEG)
    SEG = SEG_SAVE
    l0, s0 = encode(48000)
    report("[fmt] stride 48000 (no overlap), hard cut", decode(l0, s0, 48000), 48000, "prev")
    lE, sE = encode(48000, "ctx", 32)
    report("[fmt] stride 48000 (no overlap), 32 context frames on both sides, hard cut", decode(lE, sE, 48000, "ctx", 32), 48000, "prev")

if "fmt" in groups:
    # (1) longer overlap with fewer codebooks, so that the TOTAL bitrate equals the stock one
    combos = {16: [(15, 45000), (14, 42000), (12, 36000)], 4: [(3, 36000)], 8: [(7, 42000), (6, 36000)]}.get(N_Q, [])
    for nq, stride in combos:
        lat, sc_ = encode(stride, nq=nq)
        fr = decode(lat, sc_, stride)
        k = nq * 1.5 * SEG / stride
        report(f"[fmt] {nq} codebooks, overlap {100 * (1 - stride / SEG):.2f} % (stride {stride}), Meta cross-fade", fr, stride, kbps=k)
        report(f"[fmt] {nq} codebooks, overlap {100 * (1 - stride / SEG):.2f} % (stride {stride}), skip-start cross-fade", fr, stride, "skip", kbps=k)
    # (2) same n_q, longer overlap: what the extra bits buy
    for stride in (46560, 45600, 43200):
        lat, sc_ = encode(stride)
        fr = decode(lat, sc_, stride)
        if stride == 45600:
            report(f"[fmt] {N_Q} codebooks, overlap 5.0 % (stride 45600), Meta cross-fade", fr, stride)
        report(f"[fmt] {N_Q} codebooks, overlap {100 * (1 - stride / SEG):.1f} % (stride {stride}), skip-start cross-fade", fr, stride, "skip")
    # (3) no overlap at all, context carried on both sides (stateful segmentation): 1 % FEWER bits than stock
    SEG_SAVE = SEG
    latE, scE = encode(48000, "ctx", 32)
    report("[fmt] stride = segment (no overlap), 32 context frames on both sides", decode(latE, scE, 48000, "ctx", 32), 48000, "prev")
    report("[fmt] stride = segment (no overlap), encoder stock, 32 latent context frames in the decoder", decode(*encode(48000), 48000, "ctx", 32), 48000, "prev")
    report("[fmt] stride = segment (no overlap), nothing else changed (hard cut)", decode(*encode(48000), 48000), 48000, "prev")
    latE, scE = encode(47360, "ctx", 32)
    report("[fmt] stride 47360 (frame-aligned, 1.33 % overlap), 32 context frames on both sides, skip-start fade", decode(latE, scE, 47360, "ctx", 32), 47360, "skip")
    latE, scE = encode(48000, "warm")
    report("[fmt] stride = segment (no overlap), LSTM state carried on both sides", decode(latE, scE, 48000, "warm"), 48000, "prev")
    # (4) longer segments: fewer seams for the same bits
    for seg_len in (96000, 192000, (N // HOP) * HOP):
        SEG = seg_len
        stride = SEG - 480 if SEG < N - 480 else SEG
        lat, sc_ = encode(stride)
        fr = decode(lat, sc_, stride)
        what = "whole clip as ONE segment" if stride == SEG else f"{SEG // 48000} s segments, 480-sample overlap"
        report(f"[fmt] {what}, stock codec otherwise", fr, stride)
    SEG = SEG_SAVE
