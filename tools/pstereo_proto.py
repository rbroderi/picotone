"""Parametric stereo prototype for the HQ codec (INTEGRATION.md, M8 item 2).

    uv run python pstereo_proto.py prep          # 10 s excerpts at 44.1 kHz, their mids
    uv run python pstereo_proto.py code          # HQ round trips with picotone.exe (--threads 4)
    uv run python pstereo_proto.py eval          # parametric stereo, metrics, rates
    uv run python pstereo_proto.py listen        # listening WAVs

Parametric stereo here: the file codes the mid (L + R) / 2 as one HQ channel, plus per band and per
parameter frame the intensity difference (IID, dB), the coherence (ICC) and optionally the phase
difference (IPD), quantised on HE-AAC v2's grids. The decoder runs a 1024-point sine-window STFT
(hop 512, one HQ frame) on the decoded mid M and on a decorrelated copy D (a velvet-noise FIR), and
rebuilds per bin

    L = (1 + g) M + h D,     R = (1 - g) M - h D,

with g (complex with IPD, else real) and h >= 0 derived from IID/ICC/IPD so that L and R get the
coded powers and cross-correlation. (L + R) / 2 stays exactly the decoded mid, so no overall phase
(OPD) is needed. D is ducked per band so that its energy never exceeds M's (transients).

Parameter coding resets per page (80 HQ frames): the first parameter frame of a page is coded as
frequency deltas, the others as time or frequency deltas (one flag each), each type with a static
Huffman code. The rates reported are held out: codes fitted on half the clips, scored on the other.
"""
import argparse
import glob
import heapq
import json
import os
import subprocess
import sys
import time
import wave

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 44100
HOP = 512
PAGE = 80
EXE = "E:/encodecx/target/release/picotone.exe"
WORK = "E:/encodecx/work/pstereo"
LISTEN = "X:/audio/distill/pstereo_listen"
SECONDS = 10

CLIPS = [
    "wav/test__band__MusicOfRichardStrauss__05Parade-marschCavallerieNo.2__w0.wav",
    "wav/test__band__TheFourSeasonsWinter__USAFB_Winter__w0.wav",
    "wav/test__electronic__gt577Noisesurfer-RainTearDrops__01-Noisesurfer-Distocusion__w0.wav",
    "wav/test__electronic__gt577Noisesurfer-RainTearDrops__02-Noisesurfer-RainteardropsPart1__w0.wav",
    "wav/test__piano__OpenGoldbergVariations__KimikoIshizaka-J.s.Bach--open-GoldbergVariationsBwv988piano-26Variatio25A2Clav.__w0.wav",
    "wav/test__piano__musopen-chopin__Ballade_no._2_-_Op._38__w0.wav",
    "wav2/test2__hiphop__catrap__3c157bad__c0.wav",
    "wav2/test2__hiphop__nonono_918__4dde0dc0__c0.wav",
    "wav2/test2__jazz__mt021__585fc476__c0.wav",
    "wav2/test2__jazz__hr068__eaa07449__c0.wav",
    "wav2/test2__pop__MZR043__7a7c50bb__c0.wav",
    "wav2/test2__pop__hujan027__bd2fd10d__c0.wav",
    "wav2/test2__rock__GT586__4a3f47a1__c0.wav",
    "wav2/test2__ambient__Seetyca_Suite__71daa3bf__c0.wav",
    "wav2/test2__folk__rrrrrose-azerty-the-wave-my-heart-cannot-take-2021__494a03f8__c0.wav",
]
SHORT = ["band1", "band2", "elec1", "elec2", "piano1", "piano2", "hiphop1", "hiphop2", "jazz1", "jazz2",
         "pop1", "pop2", "rock", "ambient", "folk"]

# Coded conditions: name -> (picotone args, total kbps of codes).
KBPS = 10 * SR / HOP / 1000  # one codebook of one channel: 0.861 kbps
STEREO_CODES = {
    "lr2": (["--stereo", "lr", "--codebooks", "2"], 4 * KBPS),
    "ms3+1": (["--stereo", "ms", "--codebooks", "3", "--side-codebooks", "1"], 4 * KBPS),
    "lr3": (["--stereo", "lr", "--codebooks", "3"], 6 * KBPS),
    "ms4+2": (["--stereo", "ms", "--codebooks", "4", "--side-codebooks", "2"], 6 * KBPS),
}
MID_TIERS = [2, 3, 4, 5]


# ---------------------------------------------------------------- audio io

def read_wav(path):
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        assert sw == 2, f"{path}: 16-bit PCM expected"
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return sr, x.reshape(-1, ch).T.copy()


def write_wav(path, x):
    x = np.atleast_2d(x)
    pcm = np.clip(np.round(x.T * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def resample_48_to_44(x, taps=48):
    """Windowed-sinc polyphase resampler, 48 kHz -> 44.1 kHz (147/160), cutoff 0.95 of 22.05 kHz."""
    up, down = 147, 160
    fc = 0.95 * 0.5 * up / down  # cycles per input sample
    n_out = len(x) * up // down
    pad = np.concatenate([np.zeros(taps), x, np.zeros(taps + down)])
    out = np.zeros(n_out, dtype=np.float64)
    k = np.arange(-taps + 1, taps + 1)
    beta = 8.0
    for p in range(up):
        base, frac = divmod(p * down, up)
        frac /= up
        t = k - frac
        h = 2 * fc * np.sinc(2 * fc * t) * np.kaiser(2 * taps, beta)
        m = np.arange(p, n_out, up)
        if len(m) == 0:
            continue
        starts = base + down * (m - p) // up + taps + k[0]
        win = sliding_window_view(pad, 2 * taps)
        out[m] = win[starts] @ h
    return out.astype(np.float32)


# ---------------------------------------------------------------- STFT (the codec's analysis)

N = 1024
WIN = np.sin(np.pi * (np.arange(N) + 0.5) / N)


def stft(x):
    """Frames t = 0..F of N samples, frame t covering x[(t-1)H, (t+1)H): [F+1, N/2+1]."""
    F = len(x) // HOP
    xp = np.concatenate([np.zeros(HOP), x, np.zeros(HOP + (-len(x)) % HOP)])
    fr = sliding_window_view(xp, N)[::HOP][: F + 1]
    return np.fft.rfft(fr * WIN, axis=-1)


def istft(X, n):
    fr = np.fft.irfft(X, n=N, axis=-1) * WIN
    out = np.zeros((X.shape[0] + 1) * HOP)
    for t in range(X.shape[0]):
        out[t * HOP: t * HOP + N] += fr[t]
    return out[HOP: HOP + n]


# ---------------------------------------------------------------- bands, grids

EDGES_HZ = {
    20: [0, 129, 258, 388, 517, 646, 775, 904, 1033, 1378, 1723, 2067, 2412, 2756, 3101, 3790, 4823, 6201,
         7924, 12058, 22051],
    10: [0, 258, 517, 775, 1033, 1723, 2412, 3101, 4823, 7924, 22051],
}
IID_COARSE = np.array([-25, -18, -14, -10, -7, -4, -2, 0, 2, 4, 7, 10, 14, 18, 25], dtype=float)
IID_FINE = np.array([-50, -45, -40, -35, -30, -25, -22, -19, -16, -13, -10, -8, -6, -4, -2, 0,
                     2, 4, 6, 8, 10, 13, 16, 19, 22, 25, 30, 35, 40, 45, 50], dtype=float)
ICC_GRID = np.array([1, 0.937, 0.84118, 0.60092, 0.36764, 0, -0.589, -1])
# HE-AAC's grid plus levels near 1: at equal powers, ICC 0.937 already puts the side 15 dB under the mid,
# and centred mixes (pop, hip-hop) have sides 20-30 dB down in many bands.
ICC_EXT = np.array([1, 0.997, 0.99, 0.97, 0.937, 0.84118, 0.60092, 0.36764, 0, -0.589, -1])


def icc_levels(cfg):
    return ICC_EXT if cfg.get("iccx") else ICC_GRID
IPD_LEVELS = 8
IPD_BANDS = 11  # IPD only in the bands under about 1.4 kHz (20-band layout); fewer in the 10-band one


def band_bins(nb):
    e = np.round(np.array(EDGES_HZ[nb]) / (SR / N)).astype(int)
    e[-1] = N // 2 + 1
    return e


def param_index(n_stft, F, k):
    """Global parameter-frame index of each STFT frame, and per parameter frame its page and slot."""
    hq = np.minimum(np.arange(n_stft), F - 1)
    page, local = hq // PAGE, (hq % PAGE) // k
    per_page = PAGE // k
    pidx = page * per_page + local
    # Renumber densely (the last page may be short).
    uniq, dense = np.unique(pidx, return_inverse=True)
    return dense, uniq // per_page, uniq % per_page


# ---------------------------------------------------------------- encoder

def analyse(L, R, cfg):
    SL, SRr = stft(L), stft(R)
    F = len(L) // HOP
    e = band_bins(cfg["bands"])
    pidx, pages, slots = param_index(SL.shape[0], F, cfg["k"])
    P = pidx.max() + 1
    nb = cfg["bands"]
    PL = np.zeros((P, nb)); PR = np.zeros((P, nb)); X = np.zeros((P, nb), complex)
    for b in range(nb):
        sl, sr = SL[:, e[b]:e[b + 1]], SRr[:, e[b]:e[b + 1]]
        np.add.at(PL[:, b], pidx, (np.abs(sl) ** 2).sum(1))
        np.add.at(PR[:, b], pidx, (np.abs(sr) ** 2).sum(1))
        np.add.at(X[:, b], pidx, (sl * sr.conj()).sum(1))
    return PL, PR, X, pidx, pages, slots


def quantise(PL, PR, X, cfg):
    grid = IID_FINE if cfg.get("fine") else IID_COARSE
    eps = 1e-12
    silent = (PL + PR) < 1e-7 * N  # about -90 dBFS per bin
    iid = 10 * np.log10((PL + eps) / (PR + eps))
    norm = np.sqrt(PL * PR) + eps
    ipd_on = cfg.get("ipd", False)
    icc = (np.abs(X) if ipd_on else X.real) / norm
    qi = np.abs(iid[..., None] - grid).argmin(-1)
    icc_grid = icc_levels(cfg)
    if cfg.get("iccx"):
        # Nearest in side-to-mid level (dB at equal powers), where the ear and log-mel work.
        f = lambda r: 10 * np.log10((1.0005 - np.clip(r, -1, 1)) / (1.0005 + np.clip(r, -1, 1)))
        qc = np.abs(f(icc)[..., None] - f(icc_grid)).argmin(-1)
    else:
        qc = np.abs(icc[..., None] - icc_grid).argmin(-1)
    qi[silent] = len(grid) // 2
    qc[silent] = 0
    if cfg.get("hyst"):
        # Keep the previous frame's level when it is nearly as close: fewer non-zero time deltas.
        for vals, idx, g, tol in ((iid, qi, grid, 1.5), (icc, qc, icc_grid, 0.1)):
            for p in range(1, idx.shape[0]):
                prev = idx[p - 1]
                keep = np.abs(vals[p] - g[prev]) <= np.abs(vals[p] - g[idx[p]]) + tol
                idx[p] = np.where(keep & ~silent[p], prev, idx[p])
    q = {"iid": qi, "icc": qc}
    if ipd_on:
        nb_ipd = IPD_BANDS if cfg["bands"] == 20 else 6
        ph = np.angle(X[:, :nb_ipd])
        qp = np.round(ph / (2 * np.pi / IPD_LEVELS)).astype(int) % IPD_LEVELS
        qp[silent[:, :nb_ipd] | (qc[:, :nb_ipd] >= 5)] = 0  # no phase where incoherent
        q["ipd"] = qp
    return q


def dequantise(q, cfg):
    grid = IID_FINE if cfg.get("fine") else IID_COARSE
    iid = grid[q["iid"]]
    rho = icc_levels(cfg)[q["icc"]]
    theta = np.zeros_like(iid)
    if "ipd" in q:
        theta[:, : q["ipd"].shape[1]] = q["ipd"] * (2 * np.pi / IPD_LEVELS)
    return iid, rho, theta


def mix_gains(iid, rho, theta):
    """g (complex) and h of L = (1+g)M + hD, R = (1-g)M - hD from IID (dB), ICC and IPD."""
    c2 = 10 ** (iid / 10)
    c = np.sqrt(c2)
    a = c2 + 1
    pm4 = a + 2 * rho * c * np.cos(theta)
    pm4 = np.maximum(pm4, 0.1 * a)  # mid at least 10 dB under the channels: limits the gains to ~16 dB
    g = (c2 - 1 - 2j * rho * c * np.sin(theta)) / pm4
    ps4 = a - 2 * rho * c * np.cos(theta)
    h2 = np.maximum(ps4 / pm4 - np.abs(g) ** 2, 0.0)
    return g, np.sqrt(h2)


# ---------------------------------------------------------------- decoder

def velvet(seed=7, ms=50.0, density=1500.0, decay_db=-10.0, start_ms=3.0):
    """Sparse +-1 FIR (velvet noise) with an exponential decay, normalised to unit energy."""
    rng = np.random.default_rng(seed)
    td = SR / density
    length = int(ms * SR / 1000)
    cells = int(length / td)
    start = int(start_ms * SR / 1000)
    pos = start + np.floor(np.arange(cells) * td + rng.random(cells) * (td - 1)).astype(int)
    sign = rng.choice([-1.0, 1.0], cells)
    gain = 10 ** (decay_db / 20 * (pos - start) / length)
    h = np.zeros(pos.max() + 1)
    h[pos] = sign * gain
    return h / np.sqrt((h ** 2).sum())


def fir(x, h):
    n = len(x) + len(h)
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[: len(x)]


def upmix(M, q, cfg, pidx, return_coh=False):
    e = band_bins(cfg["bands"])
    iid, rho, theta = dequantise(q, cfg)
    g, h = mix_gains(iid, rho, theta)  # [P, bands]
    Sm = stft(M)
    D = fir(M, velvet(**cfg.get("velvet", {})))
    Sd = stft(D)
    widths = np.diff(e)
    G = np.repeat(g[pidx], widths, axis=1)
    Hh = np.repeat(h[pidx], widths, axis=1)
    # Ducker: per STFT frame and band, D's energy at most M's.
    em = np.add.reduceat(np.abs(Sm) ** 2, e[:-1], axis=1)
    ed = np.add.reduceat(np.abs(Sd) ** 2, e[:-1], axis=1)
    duck = np.minimum(1.0, np.sqrt((em + 1e-12) / (ed + 1e-12)))
    if not cfg.get("duck", True):
        duck = np.ones_like(duck)
    Sd = Sd * np.repeat(duck, widths, axis=1)
    SL = (1 + G) * Sm + Hh * Sd
    SRr = (1 - G) * Sm - Hh * Sd
    L, R = istft(SL, len(M)), istft(SRr, len(M))
    if return_coh:
        cross = np.add.reduceat(Sm * Sd.conj(), e[:-1], axis=1)
        ed2 = np.add.reduceat(np.abs(Sd) ** 2, e[:-1], axis=1)
        coh = np.abs(cross).sum(0) / np.sqrt(em.sum(0) * ed2.sum(0) + 1e-12)
        return L, R, coh
    return L, R


# ---------------------------------------------------------------- parameter coding

def delta_streams(q, pages, slots, cfg):
    """Per type, per parameter frame: the frequency-delta and time-delta symbols (None at a page start)."""
    out = {}
    for name, idx in q.items():
        mod = IPD_LEVELS if name == "ipd" else None
        centre = {"iid": len(IID_FINE if cfg.get("fine") else IID_COARSE) // 2, "icc": 0, "ipd": 0}[name]
        rows = []
        for p in range(idx.shape[0]):
            v = idx[p]
            df = np.diff(np.concatenate([[centre], v]))
            dt = None if slots[p] == 0 else v - idx[p - 1]
            if mod:
                df = df % mod
                dt = None if dt is None else dt % mod
            rows.append((df, dt))
        out[name] = rows
    return out


def huffman_lengths(counts):
    heap = [(c, i, (s,)) for i, (s, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    lengths = {s: 0 for s in counts}
    if len(heap) == 1:
        return {s: 1 for s in counts}
    uid = len(heap)
    while len(heap) > 1:
        c1, _, a = heapq.heappop(heap)
        c2, _, b = heapq.heappop(heap)
        for s in a + b:
            lengths[s] += 1
        heapq.heappush(heap, (c1 + c2, uid, a + b))
        uid += 1
    return lengths


def fit_codes(streams_list, cfg):
    """Huffman lengths per (type, df|dt), from frames of the fit clips (add-one over the symbol range)."""
    codes = {}
    for name in streams_list[0]:
        rng = {"iid": len(IID_FINE if cfg.get("fine") else IID_COARSE), "icc": len(icc_levels(cfg)),
               "ipd": IPD_LEVELS}[name]
        for kind in (0, 1):
            cnt = {s: 1 for s in range(-rng + 1, rng)}
            for st in streams_list:
                for row in st[name]:
                    sym = row[kind]
                    if sym is None:
                        continue
                    for s in sym:
                        cnt[int(s)] = cnt.get(int(s), 0) + 1
            codes[(name, kind)] = huffman_lengths(cnt)
    return codes


def cost_bits(streams, codes):
    bits = 0
    for name, rows in streams.items():
        for df, dt in rows:
            cf = sum(codes[(name, 0)][int(s)] for s in df)
            if dt is None:
                bits += cf
            else:
                ct = sum(codes[(name, 1)][int(s)] for s in dt)
                bits += 1 + min(cf, ct)
    return bits


# ---------------------------------------------------------------- metrics

class LogMel:
    """distill/metrics.py's log-mel (src/stereo.rs's port): 80 mels, 2048 FFT, hop 512, 80 dB floor."""

    def __init__(self):
        nfft, mels = 2048, 80
        hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
        mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        pts = mel2hz(np.linspace(hz2mel(30.0), hz2mel(SR / 2), mels + 2))
        f = np.linspace(0, SR / 2, nfft // 2 + 1)
        a, c, b = pts[:-2, None], pts[1:-1, None], pts[2:, None]
        self.fb = np.maximum(0, np.minimum((f - a) / (c - a), (b - f) / (b - c)))
        self.win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
        self.nfft = nfft

    def mel(self, x):
        p = self.nfft // 2
        xp = np.pad(x.astype(np.float64), p, mode="reflect")
        fr = sliding_window_view(xp, self.nfft)[::512]
        return np.abs(np.fft.rfft(fr * self.win, axis=-1)) @ self.fb.T

    def distance(self, r, y):
        mr, my = self.mel(r), self.mel(y)
        fl = mr.max() * 1e-4
        return float(np.mean(np.abs(20 * (np.log10(np.maximum(my, fl)) - np.log10(np.maximum(mr, fl))))))

    def stereo(self, ref, dec):
        s = lambda x: (x[0] - x[1]) / 2
        return {"L": self.distance(ref[0], dec[0]), "R": self.distance(ref[1], dec[1]),
                "side": self.distance(s(ref), s(dec))}


def spatial(ref, dec):
    """ILD and ICC errors on 20 bands, 2048-point Hann STFT, tiles of 4 frames (93 ms).

    Tiles are scored when their reference energy is within 30 dB of the band's loudest 1%.
    Returns mean |ILD error| (dB) and mean ||coherence| error|.
    """
    nfft, hop = 2048, 1024
    w = np.hanning(nfft)
    S = [np.fft.rfft(sliding_window_view(np.pad(c, (0, nfft)), nfft)[::hop] * w, axis=-1) for c in (*ref, *dec)]
    e = np.round(np.array(EDGES_HZ[20]) / (SR / nfft)).astype(int)
    e[-1] = nfft // 2 + 1
    T = S[0].shape[0] // 4 * 4

    def tiles(a, b):
        x = np.add.reduceat(a[:T] * b[:T].conj(), e[:-1], axis=1)
        return x.reshape(T // 4, 4, -1).sum(1)

    out = {}
    pl, pr, x = tiles(S[0], S[0]).real, tiles(S[1], S[1]).real, tiles(S[0], S[1])
    ql, qr, y = tiles(S[2], S[2]).real, tiles(S[3], S[3]).real, tiles(S[2], S[3])
    en = pl + pr
    act = en > np.percentile(en, 99, axis=0) * 1e-3
    eps = 1e-12
    ild = lambda a, b: np.clip(10 * np.log10((a + eps) / (b + eps)), -30, 30)
    out["ild"] = float(np.abs(ild(ql, qr) - ild(pl, pr))[act].mean())
    out["icc"] = float(np.abs(np.abs(y) / np.sqrt(ql * qr + eps) - np.abs(x) / np.sqrt(pl * pr + eps))[act].mean())
    return out


# ---------------------------------------------------------------- steps

def prep(a):
    os.makedirs(WORK, exist_ok=True)
    for rel, short in zip(CLIPS, SHORT):
        out = f"{WORK}/{short}_orig.wav"
        if os.path.exists(out):
            continue
        sr, x = read_wav(f"X:/audio/work/{rel}")
        assert sr == 48000 and x.shape[0] == 2
        mid = x.shape[1] // 2
        seg = x[:, mid - SECONDS * sr // 2: mid + SECONDS * sr // 2]
        y = np.stack([resample_48_to_44(c) for c in seg])
        n = y.shape[1] // HOP * HOP
        y = y[:, :n]
        write_wav(out, y)
        write_wav(f"{WORK}/{short}_mid.wav", (y[0] + y[1]) / 2)
        print(short, n, flush=True)


def run(cmd, timeout=600):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}{chr(10)}{r.stderr}")
    return time.time() - t


def code(a):
    for short in SHORT:
        jobs = [(f"{WORK}/{short}_orig.wav", f"{WORK}/{short}_{name}.wav", ["--channel-format", "stereo", *args])
                for name, (args, _) in STEREO_CODES.items()]
        jobs += [(f"{WORK}/{short}_mid.wav", f"{WORK}/{short}_mid{q}.wav", ["--codebooks", str(q)])
                 for q in MID_TIERS]
        for src, dst, args in jobs:
            if os.path.exists(dst):
                continue
            dt = run([EXE, "roundtrip", "--threads", str(a.threads), *args, src, dst])
            print(f"{os.path.basename(dst)} {dt:.1f} s", flush=True)


# The first run (2026-11-23) also had HE-AAC's 8-level ICC grid, IPD, hysteresis and no ducker; the
# extended ICC grid (x) won clearly, the others did not pay (see INTEGRATION.md, M8).
PS_CONFIGS = {
    "ps20k4x": {"bands": 20, "k": 4, "iccx": True},
    "ps20k8x": {"bands": 20, "k": 8, "iccx": True},
    "ps10k8x": {"bands": 10, "k": 8, "iccx": True},
    "ps20k8": {"bands": 20, "k": 8},
    "ps20k8xipd": {"bands": 20, "k": 8, "iccx": True, "ipd": True},
}


def load(short, name):
    _, x = read_wav(f"{WORK}/{short}_{name}.wav")
    return x.astype(np.float64)


def eval_(a):
    lm = LogMel()
    rows = []
    streams = {c: [] for c in PS_CONFIGS}
    durations = []
    coh_all = []
    for short in SHORT:
        ref = load(short, "orig")
        n = ref.shape[1]
        durations.append(n / SR)
        res = {"clip": short}

        def score(tag, dec):
            dec = np.clip(dec[:, :n], -1, 1)
            m = lm.stereo(ref, dec)
            m.update(spatial(ref, dec))
            m["mean3"] = (m["L"] + m["R"] + m["side"]) / 3
            res[tag] = m

        for name in STEREO_CODES:
            score(name, load(short, name))
        mid_orig = (ref[0] + ref[1]) / 2
        mids = {"orig": mid_orig, **{f"q{q}": load(short, f"mid{q}")[0][:n] for q in MID_TIERS}}
        for q in MID_TIERS:
            score(f"mono_q{q}", np.stack([mids[f"q{q}"]] * 2))
        for cname, cfg in PS_CONFIGS.items():
            PL, PR, X, pidx, pages, slots = analyse(ref[0], ref[1], cfg)
            qd = quantise(PL, PR, X, cfg)
            streams[cname].append(delta_streams(qd, pages, slots, cfg))
            for mname, M in mids.items():
                if cname == "ps20k8x" and mname == "q3":
                    L, R, coh = upmix(M, qd, cfg, pidx, return_coh=True)
                    coh_all.append(coh)
                else:
                    L, R = upmix(M, qd, cfg, pidx)
                score(f"{cname}_{mname}", np.stack([L, R]))
        rows.append(res)
        print(short, " ".join(f"{k}={v['mean3']:.3f}" for k, v in res.items() if k != "clip" and
                                ("_q3" in k or k in ("lr2", "ms3+1"))), flush=True)
    # Held-out parameter rates (2-fold over clips).
    rates = {}
    total_s = sum(durations)
    for cname, cfg in PS_CONFIGS.items():
        st = streams[cname]
        bits = 0
        for fold in (0, 1):
            fit = [s for i, s in enumerate(st) if i % 2 == fold]
            test = [s for i, s in enumerate(st) if i % 2 != fold]
            codes = fit_codes(fit, cfg)
            bits += sum(cost_bits(s, codes) for s in test)
        rates[cname] = bits / total_s / 1000
    keys = [k for k in rows[0] if k != "clip"]
    mean = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in rows[0][k]} for k in keys}
    out = {"rows": rows, "mean": mean, "param_kbps": rates,
           "decorrelator_coherence_by_band": np.mean(coh_all, 0).round(3).tolist()}
    with open(f"{WORK}/results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nparam kbps (held out):", {k: round(v, 3) for k, v in rates.items()})
    print("decorrelator |coh(M, D)| by band:", out["decorrelator_coherence_by_band"])
    print(f"\n{'condition':22s} {'L':>6s} {'R':>6s} {'side':>6s} {'mean3':>6s} {'ILD dB':>7s} {'ICC':>6s}")
    for k in keys:
        m = mean[k]
        print(f"{k:22s} {m['L']:6.3f} {m['R']:6.3f} {m['side']:6.3f} {m['mean3']:6.3f} {m['ild']:7.2f} "
              f"{m['icc']:6.3f}")


def param_rates(configs):
    """Held-out parameter kbps per config (2-fold over the clips), from the original L/R."""
    streams = {c: [] for c in configs}
    total_s = 0.0
    for short in SHORT:
        ref = load(short, "orig")
        total_s += ref.shape[1] / SR
        for cname, cfg in configs.items():
            PL, PR, X, _, pages, slots = analyse(ref[0], ref[1], cfg)
            streams[cname].append(delta_streams(quantise(PL, PR, X, cfg), pages, slots, cfg))
    rates = {}
    for cname, cfg in configs.items():
        st, bits = streams[cname], 0
        for fold in (0, 1):
            codes = fit_codes([s for i, s in enumerate(st) if i % 2 == fold], cfg)
            bits += sum(cost_bits(s, codes) for i, s in enumerate(st) if i % 2 != fold)
        rates[cname] = bits / total_s / 1000
    return rates


def rates_(a):
    configs = {}
    for nb in (10, 20):
        for k in (2, 4, 8):
            configs[f"ps{nb}k{k}"] = {"bands": nb, "k": k}
            configs[f"ps{nb}k{k}hyst"] = {"bands": nb, "k": k, "hyst": True}
            configs[f"ps{nb}k{k}ipdhyst"] = {"bands": nb, "k": k, "ipd": True, "hyst": True}
    for name, kbps in sorted(param_rates(configs).items()):
        print(f"{name:16s} {kbps:.3f} kbps")


def listen(a):
    os.makedirs(LISTEN, exist_ok=True)
    cfg = PS_CONFIGS[a.config]
    for short in a.clips.split(","):
        ref = load(short, "orig")
        n = ref.shape[1]
        write_wav(f"{LISTEN}/{short}_0_orig.wav", ref)
        for name in STEREO_CODES:
            write_wav(f"{LISTEN}/{short}_{name}.wav", load(short, name)[:, :n])
        PL, PR, X, pidx, _, _ = analyse(ref[0], ref[1], cfg)
        qd = quantise(PL, PR, X, cfg)
        for q in (3, 5):
            L, R = upmix(load(short, f"mid{q}")[0][:n], qd, cfg, pidx)
            write_wav(f"{LISTEN}/{short}_ps_mid{q}.wav", np.stack([L, R]))
        L, R = upmix((ref[0] + ref[1]) / 2, qd, cfg, pidx)
        write_wav(f"{LISTEN}/{short}_ps_origmid.wav", np.stack([L, R]))
        print(short, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["prep", "code", "eval", "listen", "rates"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--config", default="ps20k8x")
    p.add_argument("--clips", default="band1,piano1,hiphop1,jazz1,pop2,ambient")
    a = p.parse_args()
    {"prep": prep, "code": code, "eval": eval_, "listen": listen, "rates": rates_}[a.step](a)


if __name__ == "__main__":
    main()
