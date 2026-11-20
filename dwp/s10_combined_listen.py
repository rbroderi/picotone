"""Re-loads the derived LSTM weights from out/, checks them end to end (encoder AND decoder replaced together),
and writes listening samples.

usage: python s10_combined_listen.py 24khz|48khz [n_q]

Reads   out/lstm_<model>_{enc,dec}_pruned{K}_ls.safetensors   (written by s02_lstm_replace.py)
Writes  out/listen_<model>/<clip>__<variant>.wav             (16-bit PCM)
"""
import os
import sys
import wave

import numpy as np
import torch
from safetensors.torch import load_file

from common import OUT, Bench, Log, lstm_layer, perceptual, run_layers, snr_db

name = sys.argv[1]
n_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
log = Log(f"s10_{name}" + (f"_nq{n_q}" if n_q else ""))
b = Bench(name, n_q=n_q)
m = b.model


def small_lstm(path, x):
    """x [B,512,T] -> x + read-out(LSTM_small(x))."""
    w = load_file(path)
    h = x.permute(2, 0, 1).contiguous()
    inp = h
    for l in ("l0", "l1"):
        inp = lstm_layer(inp, w[f"{l}.w_ih"], w[f"{l}.w_hh"], w[f"{l}.b"])
    y = h + inp @ w["out.weight"].T + w["out.bias"]
    return y.permute(1, 2, 0).contiguous()


def encode(K):
    if K is None:
        return b.codes
    y = small_lstm(os.path.join(OUT, f"lstm_{name}_enc_pruned{K}_ls.safetensors"), b.enc_pre)
    return b.quantize(run_layers(m.encoder.layers, y, b.enc_idx + 1))


def decode(codes, K):
    pre = m.decoder.layers[0](b.dequant(codes[: b.n_q]))
    if K is None:
        return b.decode_from_pre(pre)
    return b.decode_from_pre(small_lstm(os.path.join(OUT, f"lstm_{name}_dec_pruned{K}_ls.safetensors"), pre), start=b.dec_idx + 1)


def write_wav(path, x, sr):
    pcm = np.round(np.clip(x.numpy(), -1, 1).T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


outdir = os.path.join(OUT, f"listen_{name}" + (f"_nq{n_q}" if n_q else ""))
os.makedirs(outdir, exist_ok=True)
rows_per_clip = b.src.shape[0] // len(b.names)
pick = {"speech": 0, "piano": 4, "electronic": 7, "orchestra": 9, "game_chip": 10, "game_sfx": 11}


def save(variant, rows):
    for label, i in pick.items():
        write_wav(os.path.join(outdir, f"{label}__{variant}.wav"), rows[i * rows_per_clip:(i + 1) * rows_per_clip], b.sr)


save("0_source", b.src)
for Ke, Kd in ((None, None), (None, 384), (384, None), (384, 384), (None, 256), (256, 256)):
    if any(K and not os.path.exists(os.path.join(OUT, f"lstm_{name}_{s}_pruned{K}_ls.safetensors")) for K, s in ((Ke, "enc"), (Kd, "dec"))):
        continue
    codes = encode(Ke)
    y = b._wave_rows(decode(codes, Kd))
    r = perceptual(b.src, y, b.sr)
    r["snr_orig"] = snr_db(b._wave_rows(b.y), y)
    r["codes"] = 100 * float((codes == b.codes).float().mean())
    for k in ("snr", "logmel", "mrstft"):
        r["d_" + k] = r[k] - b.stock[k]
    variant = f"enc{Ke or 512}_dec{Kd or 512}"
    log(f"encoder LSTM {Ke or 512} units, decoder LSTM {Kd or 512} units", r)
    save(("1_stock" if Ke is None and Kd is None else "2_" + variant), y)
print("listening samples in", outdir)
