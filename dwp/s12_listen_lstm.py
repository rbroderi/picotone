"""Listening set for the pruned-LSTM rewrites, decoded with the real segmentation and overlap-add.

usage: python s12_listen_lstm.py

48 kHz model, 24 kbps (16 codebooks), Meta segmentation (48000-sample segments every 47520, per-segment RMS scale,
triangle overlap-add): stock, 448 [W], 384 [LS], 320 [LS] units per LSTM layer, (a) decoder only (stock codes) and
(b) encoder and decoder.  Clips: the two scratchpad `listen` clips (first 30 s) + speech, piano, chiptune (25 s).
24 kHz model, 6 kbps (8 codebooks): stock vs 384 [LS], one speech and one music clip.

The ONLY difference between a clip's files is the LSTM weights.  Pruned LSTMs are rebuilt here exactly as in
s02_lstm_replace.py (unit selection on the fit split; [W] = dropped units replaced by their mean, [LS] = dropped units
expressed as a least-squares function of the kept ones) and saved to out/lstm_<model>_<side>_pruned<K>_<mean|ls>.safetensors.

Writes out/listen_lstm/{48k_24kbps,24k_6kbps}/<clip>__<n>_<variant>.wav, out/listen_lstm/abx/ (blind trials),
out/listen_lstm/abx_answer_key/key.csv and results/s12_listen_lstm.jsonl.
"""
import csv
import os
import random
import wave

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.signal import resample_poly

from common import OUT, WAVDIR, Bench, Log, _read_wav, get_lstm, load_model, lstm_layer, perceptual, run_layers, snr_db, synth_game

LISTEN = os.environ.get("DWP_LISTEN", r"C:\Users\richa\AppData\Local\Temp\claude\8979288677f01101\7927348c-78e9-4ae8-849c-648d18ca0651\scratchpad\listen")
ROOT = os.path.join(OUT, "listen_lstm")
log = Log("s12_listen_lstm")
D = 512
flat = lambda a: a.reshape(-1, a.shape[-1])


# --------------------------------------------------------------------------- building the pruned LSTMs (as in s02)
def lsq(X, Y, ridge=1e-3):
    X, Y = X.double(), Y.double()
    mx, my = X.mean(0), Y.mean(0)
    G = (X - mx).T @ (X - mx)
    G += ridge * G.diagonal().mean() * torch.eye(len(G), dtype=G.dtype)
    W = torch.linalg.solve(G, (X - mx).T @ (Y - my)).T
    return W.float(), (my - mx @ W.T).float()


def cond_var(h):
    c = torch.cov(flat(h).double().T)
    c += 1e-6 * c.diagonal().mean() * torch.eye(len(c), dtype=c.dtype)
    return (1 / torch.linalg.inv(c).diagonal()).float()


def build_pruned(name, side, wanted):
    """wanted: list of (K, mode).  Returns {(K, mode): path}; builds only what is missing."""
    paths = {(K, mode): os.path.join(OUT, f"lstm_{name}_{side}_pruned{K}_{mode}.safetensors") for K, mode in wanted}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    fit = Bench(name, "fit")
    _, stock = get_lstm(fit.model, side)
    xf = (fit.enc_pre if side == "enc" else fit.dec_pre).permute(2, 0, 1).contiguous()
    h1 = lstm_layer(xf, stock[0]["w_ih"], stock[0]["w_hh"], stock[0]["b"])
    h2 = lstm_layer(h1, stock[1]["w_ih"], stock[1]["w_hh"], stock[1]["b"])
    sens1 = stock[1]["w_ih"].pow(2).sum(0) + stock[0]["w_hh"].pow(2).sum(0)
    sens2 = 1 + stock[1]["w_hh"].pow(2).sum(0)
    crit = {"mean": (flat(h1).std(0) ** 2 * sens1, flat(h2).std(0) ** 2 * sens2), "ls": (cond_var(h1) * sens1, cond_var(h2) * sens2)}
    a, b = stock
    rows = lambda keep: torch.cat([keep + g * D for g in range(4)])
    for (K, mode), path in paths.items():
        if os.path.exists(path):
            continue
        keep1 = crit[mode][0].argsort(descending=True)[:K].sort().values
        keep2 = crit[mode][1].argsort(descending=True)[:K].sort().values
        allu = torch.arange(D)
        drop1, drop2 = allu[~torch.isin(allu, keep1)], allu[~torch.isin(allu, keep2)]
        if mode == "mean":
            P1, q1 = torch.zeros(len(drop1), K), flat(h1).mean(0)[drop1]
            P2, q2 = torch.zeros(len(drop2), K), flat(h2).mean(0)[drop2]
        else:
            P1, q1 = lsq(flat(h1)[:, keep1], flat(h1)[:, drop1])
            P2, q2 = lsq(flat(h2)[:, keep2], flat(h2)[:, drop2])
        r1, r2 = rows(keep1), rows(keep2)
        E = torch.zeros(D, K)
        e = torch.zeros(D)
        E[keep2, torch.arange(K)] = 1
        E[drop2] = P2
        e[drop2] = q2
        save_file({"l0.w_ih": a["w_ih"][r1].contiguous(),
                   "l0.w_hh": a["w_hh"][r1][:, keep1] + a["w_hh"][r1][:, drop1] @ P1,
                   "l0.b": a["b"][r1] + a["w_hh"][r1][:, drop1] @ q1,
                   "l1.w_ih": b["w_ih"][r2][:, keep1] + b["w_ih"][r2][:, drop1] @ P1,
                   "l1.w_hh": b["w_hh"][r2][:, keep2] + b["w_hh"][r2][:, drop2] @ P2,
                   "l1.b": b["b"][r2] + b["w_ih"][r2][:, drop1] @ q1 + b["w_hh"][r2][:, drop2] @ q2,
                   "out.weight": E, "out.bias": e}, path)
        print("built", os.path.basename(path), flush=True)
    return paths


def block(x, stock_ps, path=None):
    """LSTM block  x + LSTM(x)  on [B,512,T]; stock weights, or a pruned file."""
    h = x.permute(2, 0, 1).contiguous()
    if path is None:
        inp = h
        for p in stock_ps:
            inp = lstm_layer(inp, p["w_ih"], p["w_hh"], p["b"])
        y = h + inp
    else:
        w = load_file(path)
        inp = h
        for l in ("l0", "l1"):
            inp = lstm_layer(inp, w[f"{l}.w_ih"], w[f"{l}.w_hh"], w[f"{l}.b"])
        y = h + inp @ w["out.weight"].T + w["out.bias"]
    return y.permute(1, 2, 0).contiguous()


def bat(fn, x, bs=16):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def write_wav(path, x, sr):
    x = x if x.dim() == 2 else x[None]
    pcm = np.round(np.clip(x.numpy(), -1, 1).T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def rows_of(x):
    return x if x.dim() == 2 else x[None]


def score(label, src, y, stock_y, stock_metrics):
    r = perceptual(rows_of(src), rows_of(y), SR)
    r["snr_orig"] = snr_db(stock_y, y) if stock_y is not None else float("inf")
    if stock_metrics:
        for k in ("snr", "logmel", "mrstft"):
            r["d_" + k] = r[k] - stock_metrics[k]
    log(label, r)
    return r


# =========================================================================== 48 kHz, 24 kbps
VARIANTS = [(448, "mean", "448W"), (384, "ls", "384LS"), (320, "ls", "320LS")]
p_enc = build_pruned("48khz", "enc", [(K, m_) for K, m_, _ in VARIANTS])
p_dec = build_pruned("48khz", "dec", [(K, m_) for K, m_, _ in VARIANTS])
m = load_model("48khz")
E, Dl = m.encoder.layers, m.decoder.layers
ei, eps_ = get_lstm(m, "enc")
di, dps = get_lstm(m, "dec")
SEG, STRIDE, SR = 48000, 47520, 48000


def wav30(path, seconds):
    with wave.open(path) as w:
        pcm = np.frombuffer(w.readframes(seconds * 48000), "<i2").reshape(-1, 2).T
    return torch.from_numpy(pcm.astype(np.float32) / 32768)


def test_file(group, contains):
    f = [f for f in sorted(os.listdir(WAVDIR)) if f.startswith(f"test__{group}") and contains in f and ".short" not in f][0]
    return torch.from_numpy(_read_wav(os.path.join(WAVDIR, f), 20, 25))


clips48 = {
    "orchestral": wav30(os.path.join(LISTEN, "orchestral-0-original.wav"), 30),
    "ambient-electronic": wav30(os.path.join(LISTEN, "ambient-electronic-0-original.wav"), 30),
    "speech": test_file("speech", "huck"),
    "piano": test_file("piano", "Ballade_no._1"),
    "chiptune": torch.from_numpy(synth_game(101, "chip", seconds=25)),
}


def encode48(x, path=None):
    offs = list(range(0, x.shape[-1], STRIDE))
    segs = [x[:, o:o + SEG] for o in offs]
    full = torch.stack([s for s in segs if s.shape[-1] == SEG])
    out = []
    groups = [full] + [s[None] for s in segs if s.shape[-1] < SEG]          # the short last segment goes alone
    for g in groups:
        sc = g.mean(1, keepdim=True).pow(2).mean(2, keepdim=True).sqrt() + 1e-8
        emb = bat(lambda z: run_layers(E, block(run_layers(E, z, 0, ei), eps_, path), ei + 1), g / sc)
        codes = m.quantizer.encode(emb, None)                                # [16, B, T]
        out.append((codes, sc, g.shape[-1]))
    return out


def decode48(enc, n, path=None, nq=16):
    frames = []
    for codes, sc, length in enc:
        lat = m.quantizer.decode(codes[:nq])
        w = bat(lambda z: run_layers(Dl, block(Dl[0](z), dps, path), di + 1), lat)[:, :, :length] * sc
        frames += list(w)
    total = STRIDE * (len(frames) - 1) + frames[-1].shape[-1]
    t = torch.linspace(0, 1, SEG + 2, dtype=torch.float64)[1:-1]
    wgt = 0.5 - (t - 0.5).abs()
    out = torch.zeros(2, total, dtype=torch.float64)
    sw = torch.zeros(total, dtype=torch.float64)
    for i, f in enumerate(frames):
        L = f.shape[-1]
        out[:, i * STRIDE:i * STRIDE + L] += wgt[:L] * f.double()
        sw[i * STRIDE:i * STRIDE + L] += wgt[:L]
    return (out / sw).float()[:, :n]


abx_items = []   # (clip, variant, stock tensor, variant tensor, sr)
d48 = os.path.join(ROOT, "48k_24kbps")
os.makedirs(d48, exist_ok=True)
for cname, x in clips48.items():
    n = x.shape[-1]
    write_wav(os.path.join(d48, f"{cname}__0_source.wav"), x, SR)
    enc0 = encode48(x)
    y0 = decode48(enc0, n)
    write_wav(os.path.join(d48, f"{cname}__1_stock.wav"), y0, SR)
    s0 = score(f"48k 24kbps {cname}: stock", x, y0, None, None)
    k = 2
    for both in (False, True):
        for K, mode, tag in VARIANTS:
            enc = encode48(x, p_enc[(K, mode)]) if both else enc0
            y = decode48(enc, n, p_dec[(K, mode)])
            vname = ("both" if both else "dec") + tag
            write_wav(os.path.join(d48, f"{cname}__{k}_{vname}.wav"), y, SR)
            score(f"48k 24kbps {cname}: {vname}", x, y, y0, s0)
            abx_items.append((f"48k_24kbps/{cname}", vname, y0, y, SR))
            k += 1
            if K == 384 and cname in ("orchestral", "ambient-electronic"):
                # 12 kbps mode: the same codes truncated to 8 codebooks
                d12 = os.path.join(ROOT, "48k_12kbps")
                os.makedirs(d12, exist_ok=True)
                if not both:
                    write_wav(os.path.join(d12, f"{cname}__0_source.wav"), x, SR)
                    y0_12 = decode48(enc0, n, None, 8)
                    write_wav(os.path.join(d12, f"{cname}__1_stock.wav"), y0_12, SR)
                    s0_12 = score(f"48k 12kbps {cname}: stock", x, y0_12, None, None)
                y12 = decode48(enc, n, p_dec[(K, mode)], 8)
                write_wav(os.path.join(d12, f"{cname}__{3 if both else 2}_{vname}.wav"), y12, SR)
                score(f"48k 12kbps {cname}: {vname}", x, y12, y0_12, s0_12)
                abx_items.append((f"48k_12kbps/{cname}", vname, y0_12, y12, SR))

# =========================================================================== 24 kHz, 6 kbps
SR = 24000
m24 = load_model("24khz")
E24, D24 = m24.encoder.layers, m24.decoder.layers
ei24, eps24 = get_lstm(m24, "enc")
di24, dps24 = get_lstm(m24, "dec")
pe24 = build_pruned("24khz", "enc", [(384, "ls")])[(384, "ls")]
pd24 = build_pruned("24khz", "dec", [(384, "ls")])[(384, "ls")]
to24 = lambda x: torch.from_numpy(resample_poly(x.mean(0).numpy(), 1, 2).astype(np.float32))
clips24 = {"speech": to24(clips48["speech"]), "orchestral": to24(clips48["orchestral"][:, :25 * 48000])}
d24 = os.path.join(ROOT, "24k_6kbps")
os.makedirs(d24, exist_ok=True)


def codec24(x, penc=None, pdec=None, codes=None):
    if codes is None:
        emb = run_layers(E24, block(run_layers(E24, x[None, None], 0, ei24), eps24, penc), ei24 + 1)
        codes = m24.quantizer.encode(emb, None)[:8]
    y = run_layers(D24, block(D24[0](m24.quantizer.decode(codes)), dps24, pdec), di24 + 1)[0, 0, :len(x)]
    return y, codes


for cname, x in clips24.items():
    write_wav(os.path.join(d24, f"{cname}__0_source.wav"), x, SR)
    y0, c0 = codec24(x)
    write_wav(os.path.join(d24, f"{cname}__1_stock.wav"), y0, SR)
    s0 = score(f"24k 6kbps {cname}: stock", x, y0, None, None)
    yd, _ = codec24(x, None, pd24, c0)
    write_wav(os.path.join(d24, f"{cname}__2_dec384LS.wav"), yd, SR)
    score(f"24k 6kbps {cname}: dec384LS", x, yd, y0, s0)
    yb, _ = codec24(x, pe24, pd24)
    write_wav(os.path.join(d24, f"{cname}__3_both384LS.wav"), yb, SR)
    score(f"24k 6kbps {cname}: both384LS", x, yb, y0, s0)
    abx_items += [(f"24k_6kbps/{cname}", "dec384LS", y0, yd, SR), (f"24k_6kbps/{cname}", "both384LS", y0, yb, SR)]

# =========================================================================== blind ABX
# One trial = A, B (stock and one variant, random order) and X (a copy of A or of B).  10 s excerpts (5 s .. 15 s).
rng = random.Random(20261119)
abx = os.path.join(ROOT, "abx")
keydir = os.path.join(ROOT, "abx_answer_key")
os.makedirs(abx, exist_ok=True)
os.makedirs(keydir, exist_ok=True)
order = list(range(len(abx_items)))
rng.shuffle(order)
with open(os.path.join(keydir, "key.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["trial", "clip", "variant", "A_is", "B_is", "X_is"])
    for t, idx in enumerate(order, 1):
        clip, vname, ys, yv, sr = abx_items[idx]
        cut = lambda y: rows_of(y)[:, 5 * sr:15 * sr]
        a_stock = rng.random() < 0.5
        x_is_a = rng.random() < 0.5
        A, B = (ys, yv) if a_stock else (yv, ys)
        write_wav(os.path.join(abx, f"trial{t:02d}_A.wav"), cut(A), sr)
        write_wav(os.path.join(abx, f"trial{t:02d}_B.wav"), cut(B), sr)
        write_wav(os.path.join(abx, f"trial{t:02d}_X.wav"), cut(A if x_is_a else B), sr)
        wr.writerow([f"trial{t:02d}", clip, vname, "stock" if a_stock else vname, vname if a_stock else "stock", "A" if x_is_a else "B"])
print("done:", ROOT)
