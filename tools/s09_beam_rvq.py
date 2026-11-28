"""Aside (NOT a weight edit): beam-search residual VQ in the encoder.

usage: uv run --extra parity python s09_beam_rvq.py 24khz|48khz

The stock encoder picks each stage's code greedily.  Keeping the B best partial code sequences per frame and
choosing the one with the smallest final residual gives a lower latent error at exactly the same bitrate, and
the stream is decodable by the unmodified decoder.  Included because the solved decoder-side corrections (s04)
turned out not to help, and this is the cheap way to get 'better quality per bit' without touching the format.
Cost: B x the (tiny) quantiser search; the conv/LSTM stack is untouched.
"""
import sys

import torch

from encodec_eval import Bench, Log, perceptual, snr_db

name = sys.argv[1]
log = Log(f"s09_{name}")
test = Bench(name)
m = test.model
E = [l.codebook.embed.clone() for l in m.quantizer.layers]
frame_rate = 75 if name == "24khz" else 150
zt = test.emb.permute(0, 2, 1).reshape(-1, 128)
Bt, _, Tt = test.emb.shape


def audio(q):
    lat = q.reshape(Bt, Tt, 128).permute(0, 2, 1).contiguous()
    y = test.decode_from_pre(m.decoder.layers[0](lat))
    return perceptual(test.src, test._wave_rows(y), test.sr)


def beam_rvq(z, k, beam):
    N = len(z)
    res = z[:, None, :]                       # [N, b, 128] residual of each hypothesis
    for i in range(k):
        b = res.shape[1]
        d = torch.cdist(res.reshape(-1, 128), E[i]).reshape(N, b * 1024)   # error after adding each entry
        top = d.topk(min(beam, b * 1024), largest=False).indices           # [N, beam]
        parent, entry = top // 1024, top % 1024
        res = torch.gather(res, 1, parent[:, :, None].expand(-1, -1, 128)) - E[i][entry]
    best = res.pow(2).sum(-1).argmin(1)
    final = res[torch.arange(N), best]
    return z - final


for k in [k for k in (2, 4, 8, 16, 32) if k <= len(E)]:
    q0 = beam_rvq(zt, k, 1)
    r0 = audio(q0)
    l0 = snr_db(zt, q0)
    log(f"greedy (stock) n_q={k}", r0, kbps=k * 10 * frame_rate / 1000, latent_snr=l0)
    for beam in (4, 16):
        q = beam_rvq(zt, k, beam)
        r = audio(q)
        log(f"beam {beam} n_q={k}", r, kbps=k * 10 * frame_rate / 1000, latent_snr=snr_db(zt, q), latent_gain_db=snr_db(zt, q) - l0,
            d_snr=r["snr"] - r0["snr"], d_logmel=r["logmel"] - r0["logmel"], d_mrstft=r["mrstft"] - r0["mrstft"])
