"""How much perturbation do the two quality bars tolerate?  (calibration, no weight edits)

usage: python s07_sensitivity.py 24khz|48khz
White Gaussian noise is added to the encoder-LSTM output and to the decoder-LSTM output at a given SNR, and the
usual metrics are reported.  This converts 'codes >= 99 %' and 'decoder SNR >= 40 dB' into the accuracy a
rewritten LSTM would need.
"""
import sys

import torch

from common import Bench, Log, get_lstm, lstm_layer, run_layers

name = sys.argv[1]
log = Log(f"s07_{name}")
b = Bench(name)
torch.manual_seed(0)
for side in ("enc", "dec"):
    idx, ps = get_lstm(b.model, side)
    x = (b.enc_pre if side == "enc" else b.dec_pre).permute(2, 0, 1).contiguous()
    h = x
    for p in ps:
        h = lstm_layer(h, p["w_ih"], p["w_hh"], p["b"])
    for snr in (20, 30, 40, 50, 60, 70, 80):
        noise = torch.randn_like(h) * h.std() * 10 ** (-snr / 20)
        y = (x + h + noise).permute(1, 2, 0).contiguous()
        if side == "enc":
            r = b.eval_emb(run_layers(b.model.encoder.layers, y, idx + 1), full_chain=snr <= 50)
        else:
            r = b.eval_wave(b.decode_from_pre(y, start=idx + 1))
        for k in ("snr", "logmel", "mrstft"):
            if k in r:
                r["d_" + k] = r[k] - b.stock[k]
        log(f"{side}: noise on the LSTM branch output at {snr} dB (relative to the branch)", r)
