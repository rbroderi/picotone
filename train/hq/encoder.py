# SPDX-License-Identifier: GPL-3.0-or-later
"""A trained HQ encoder in front of DAC's frozen residual vector quantiser and a frozen decoder.

DAC's encoder costs 30.6 GMAC per second of audio and channel (about 42 s for 90 s of stereo in the Rust CLI). The
codes no longer need to equal DAC's, only to decode well, so a much smaller encoder can feed the same quantiser:

- `STFTEncoder` (default): Vocos-style, everything at the latent frame rate (86.13 Hz). Features: compressed
  complex STFTs (S |S|^(c - 1), real and imaginary parts, c = 0.3) at one or more resolutions, framed so a latent
  frame's features are centred on its 512 samples; sub-hop frames (n_fft < 2048) are stacked into channels. Then
  a k1 (or k3) embedding, LayerNorm, N ConvNeXt blocks (vocos_student.ConvNeXtBlock), a final LayerNorm and a
  Linear to the 1024-d latent. About 1.2 GMAC/s at 512 x 8.
- `SlimDACEncoder`: DAC's own conv encoder layout (k7 stem, 4 strided blocks of Snake residual units, strides
  2/4/8/8), narrower: d_model 16 instead of 64 costs about 1/16 of DAC's MACs.

`FrozenRVQ` is a copy of DAC's quantiser (9 x 1024 codebooks, factorised 8-d projections, l2-normalised nearest
neighbour) with per-item codebook counts, a straight-through estimator and the commitment loss; nothing in it
trains. `EncoderCodec` = encoder + FrozenRVQ + decoder (frozen unless the joint fine-tune unfreezes it).

Training (train.py --arch encoder): phase A (--enc-phase match) matches DAC's pre-quantisation latent z (normalised
MSE + cosine) and the codes (cross-entropy over each codebook's cosine logits, teacher-forced residuals); phase B
(--enc-phase e2e) trains end to end through the quantiser into the decoder against the true audio.

    uv run --extra cpu python encoder.py              # the cost probe
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

from vocos_student import FRAME, LATENT, ConvNeXtBlock, VocosDecoder

N_CB = 9
KBPS_PER_CB = 44100 / FRAME * 10 / 1000  # 10 bits per codebook per frame: 0.861 kbps per channel
DAC_ENC_GMAC = 30.6  # stock DAC encoder, GMAC per second and channel (measured, teacher.encoder_cost)
DAC_ENC_RUST_S = 42.0  # Rust CLI, seconds per 90 s of stereo


# ---------------------------------------------------------------- encoders

class STFTFeatures(nn.Module):
    """x [B, 1, T] (T a multiple of 512) -> [B, C, T / 512]. Per resolution n_fft w: hop min(512, w / 4), zero
    padding (w - hop) / 2 each side and no centring, so STFT frame j is centred on sample hop j + hop / 2 and the
    512 / hop frames of latent frame f lie inside its samples [512 f, 512 f + 512). Compressed complex values
    S |S|^(c - 1) (real, imaginary) per bin; fp32."""

    def __init__(self, ffts=(2048, 512), power=0.3):
        super().__init__()
        self.ffts, self.power = tuple(ffts), power
        self.channels = 0
        for w in self.ffts:
            hop = min(FRAME, w // 4)
            assert FRAME % hop == 0 and (w - hop) % 2 == 0, w
            self.register_buffer(f"win{w}", torch.hann_window(w), persistent=False)
            self.channels += (FRAME // hop) * 2 * (w // 2 + 1)

    def forward(self, x):
        B, T = x.shape[0], x.shape[-1]
        assert T % FRAME == 0, T
        F_ = T // FRAME
        out = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.reshape(B, T).float()
            for w in self.ffts:
                hop = min(FRAME, w // 4)
                k = FRAME // hop
                p = (w - hop) // 2
                s = torch.stft(F.pad(x, (p, p)), w, hop, window=getattr(self, f"win{w}"), center=False,
                               return_complex=True)  # [B, w/2 + 1, T / hop]
                s = s * (s.real.square() + s.imag.square() + 1e-8).pow((self.power - 1) / 2)
                f = torch.cat([s.real, s.imag], 1)  # [B, 2 (w/2 + 1), k F]
                C = f.shape[1]
                out.append(f.reshape(B, C, F_, k).permute(0, 3, 1, 2).reshape(B, k * C, F_))
        return torch.cat(out, 1)

    def fft_mac_per_frame(self):
        """Analytic FFT cost per latent frame (2.5 n log2 n flops per real FFT, counted as half that in MACs)."""
        return sum((FRAME // min(FRAME, w // 4)) * 2.5 * w * math.log2(w) / 2 for w in self.ffts)


class STFTEncoder(nn.Module):
    def __init__(self, dim=512, blocks=8, mlp=None, ffts=(2048, 512), power=0.3, embed_k=1, latent=LATENT):
        super().__init__()
        mlp = mlp or 3 * dim
        self.feats = STFTFeatures(ffts, power)
        self.embed = nn.Conv1d(self.feats.channels, dim, embed_k, padding=embed_k // 2)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, mlp, 1.0 / blocks) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.out = nn.Linear(dim, latent)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # [B, 1, T] -> z [B, 1024, T / 512]
        h = self.embed(self.feats(x))
        h = self.norm(h.transpose(1, 2)).transpose(1, 2).float()
        for b in self.blocks:
            h = b(h)
        return self.out(self.final_norm(h.transpose(1, 2))).transpose(1, 2).float()

    def extra_mac_per_frame(self):
        return self.feats.fft_mac_per_frame()


class SlimDACEncoder(nn.Module):
    """DAC's encoder layout at another width (DAC 44 kHz: d_model 64). Weight norm as in DAC, via
    parametrizations (deep-copyable for the EMA)."""

    def __init__(self, d_model=16, latent=LATENT):
        super().__init__()
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        import dac.model.dac as dm

        self.net = dm.Encoder(d_model, [2, 4, 8, 8], latent)
        for m in self.net.modules():  # DAC's WNConv1d uses the old weight_norm: swap to parametrizations
            if hasattr(m, "weight_g"):
                torch.nn.utils.remove_weight_norm(m)
                torch.nn.utils.parametrizations.weight_norm(m)

    def forward(self, x):
        return self.net(x).float()

    def extra_mac_per_frame(self):
        return 0.0


def build_encoder(cfg):
    if cfg["enc"] == "stft":
        return STFTEncoder(cfg["dim"], cfg["blocks"], cfg["mlp"], tuple(cfg["ffts"]), cfg["power"], cfg["embed_k"])
    if cfg["enc"] == "conv":
        return SlimDACEncoder(cfg["conv_width"])
    raise ValueError(cfg["enc"])


# ---------------------------------------------------------------- the frozen quantiser

class FrozenRVQ(nn.Module):
    """DAC's ResidualVectorQuantize, frozen: per codebook i, z_e = in_proj(r_i) (8-d), the nearest codebook entry by
    cosine (both l2-normalised), z_q = out_proj(entry) and r_{i+1} = r_i - z_q, as in DAC. Gradients reach the
    encoder by the straight-through estimator (z_e + (entry - z_e).detach()); the commitment loss is DAC's
    mse(z_e, entry). fp32 throughout (nearest-neighbour ties are precision-sensitive)."""

    def __init__(self, quantizer, soft_tau=0.0):
        super().__init__()
        import copy

        self.q = copy.deepcopy(quantizer).float()
        for p in self.q.parameters():
            p.requires_grad_(False)
        # > 0: the soft-gradient estimator (forward = the hard code exactly; backward = the gradient of the
        # softmax(cos / soft_tau)-weighted mean of the codebook entries), in place of plain straight-through.
        self.soft_tau = soft_tau

    def nearest(self, i, z_e):  # z_e [B, 8, F] -> indices [B, F]
        cb = F.normalize(self.q.quantizers[i].codebook.weight.float(), dim=1)
        return torch.einsum("bdf,nd->bfn", F.normalize(z_e, dim=1), cb).argmax(-1)

    def entry(self, i, idx):  # [B, F] -> [B, 8, F] raw codebook vectors
        return self.q.quantizers[i].codebook.weight.float()[idx].transpose(1, 2)

    def soft_entry(self, i, z_e):  # [B, 8, F] -> [B, 8, F]: softmax(cos / soft_tau)-weighted mean of the raw entries
        w = self.q.quantizers[i].codebook.weight.float()
        logits = torch.einsum("bdf,nd->bfn", F.normalize(z_e, dim=1), F.normalize(w, dim=1)) / self.soft_tau
        return torch.einsum("bfn,nd->bdf", logits.softmax(-1), w)

    def forward(self, z, n_q, st=True):
        """z [B, 1024, F]; n_q int or LongTensor [B] -> (z_q [B, 1024, F] summing each item's first n_q codebooks,
        codes [B, max n_q, F], commitment per item [B] (mean over that item's codebooks))."""
        B = z.shape[0]
        if isinstance(n_q, int):
            n_q = torch.full((B,), n_q, device=z.device)
        with torch.autocast(device_type=z.device.type, enabled=False):
            r = z.float()
            zq, commit, codes = 0.0, 0.0, []
            for i in range(int(n_q.max())):
                qz = self.q.quantizers[i]
                z_e = qz.in_proj(r)
                idx = self.nearest(i, z_e.detach())
                c = self.entry(i, idx)
                on = (n_q > i).to(z_e.dtype)
                commit = commit + (z_e - c).square().mean([1, 2]) * on
                if st and self.soft_tau > 0:
                    s = self.soft_entry(i, z_e)
                    zi = qz.out_proj(c + (s - s.detach()))  # s - s.detach() is exactly 0: the forward is c
                else:
                    zi = qz.out_proj(z_e + (c - z_e).detach() if st else c)
                zq = zq + zi * on.view(B, 1, 1)
                r = r - zi
                codes.append(idx)
            return zq, torch.stack(codes, 1), commit / n_q.clamp_min(1).to(commit.dtype)

    def match_losses(self, z, codes_t, tau=0.05):
        """Phase A code loss: per codebook, cross-entropy of the cosine logits (/ tau) of in_proj(r_i) against the
        teacher's code, with teacher-forced residuals r_{i+1} = r_i - out_proj(teacher entry). -> (mean CE,
        agreement per codebook [9] of the argmax, which is the code this z would get given the teacher's earlier
        codes)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            r = z.float()
            ce, agree = 0.0, []
            n = codes_t.shape[1]
            for i in range(n):
                qz = self.q.quantizers[i]
                z_e = qz.in_proj(r)
                cb = F.normalize(qz.codebook.weight.float(), dim=1)
                logits = torch.einsum("bdf,nd->bfn", F.normalize(z_e, dim=1), cb) / tau
                ce = ce + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes_t[:, i].reshape(-1))
                agree.append((logits.argmax(-1) == codes_t[:, i]).float().mean())
                r = r - qz.out_proj(self.entry(i, codes_t[:, i]))
            return ce / n, torch.stack(agree)


def z_match_loss(z, z_t):
    """Normalised MSE (by the teacher latent's mean square) + mean (1 - cosine) per frame."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        z, z_t = z.float(), z_t.float()
        nmse = (z - z_t).square().mean() / z_t.square().mean().clamp_min(1e-8)
        cos = F.cosine_similarity(z, z_t, dim=1).mean()
        return nmse + (1 - cos), nmse, cos


def agreement(codes, codes_t, n_q):
    """Free-running code agreement with DAC's codes: {'agree1': codebook 1, 'agreeq': mean over the first n_q}."""
    n = min(n_q, codes.shape[1], codes_t.shape[1])
    T = min(codes.shape[-1], codes_t.shape[-1])
    eq = (codes[:, :n, :T] == codes_t[:, :n, :T]).float()
    return {"agree1": float(eq[:, 0].mean()), "agreeq": float(eq.mean())}


# ---------------------------------------------------------------- the frozen decoder

def load_decoder(path, use_ema=False, cfg=None):
    """A frozen Vocos decoder for the encoder's codes: a train.py checkpoint of --arch vocos ('G', or 'G_ema' with
    use_ema) or an export .pt (a plain state dict with its .json beside it, as export.py / export_dec_ft.py write).
    The file is only read. When the file is not on this machine, cfg ({'dec': arch cfg, 'dec_width': w}) builds an
    untrained one, whose weights then come from the encoder checkpoint. -> (decoder, its arch cfg, width)."""
    import json

    def frozen(m):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def build(arch):
        if not arch or arch.get("arch") != "vocos":
            raise SystemExit(f"--enc-dec: {path} is not a Vocos decoder ({(arch or {}).get('arch', 'no arch')})")
        return VocosDecoder(arch["dim"], arch["blocks"], arch["mlp"], arch["hop"])

    if not os.path.exists(path):
        if cfg is None:
            raise SystemExit(f"--enc-dec {path} not found")
        print(f"encoder: decoder {path} not here; its weights must come from the checkpoint")
        return frozen(build(cfg["dec"])), cfg["dec"], cfg["dec_width"]
    s = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(s, dict) and "G" in s:
        arch = s.get("arch")
        m = build(arch)
        key = "G_ema" if use_ema and s.get("G_ema") is not None else "G"
        if use_ema and key == "G":
            print(f"encoder: {path} has no G_ema; using G")
        m.load_state_dict(s[key])
        return frozen(m), arch, m.width
    js = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(js):
        raise SystemExit(f"--enc-dec {path}: an export .pt needs its .json beside it")
    arch = json.load(open(js))["arch"]
    m = build(arch)
    m.load_state_dict(s)
    return frozen(m), arch, m.width


N_TIERS = 13  # patch tiercond: codebook counts 0..12 (0 unused), so q10-q12 can reuse the table


def add_tier_cond(dec):
    """Patch tiercond: a learned per-codebook-count embedding (N_TIERS x 1024), zero-initialised, attached to the
    decoder as dec.tier_emb; tier_cond_decode adds it to the decoder's input (the codebook sum z_q). Zero at init, so
    the decoder's output is bit-identical to the decoder without it until the embedding trains."""
    dec.tier_emb = nn.Embedding(N_TIERS, LATENT)
    nn.init.zeros_(dec.tier_emb.weight)
    return dec


def tier_cond_decode(dec, z, n_q):
    """dec(z), or dec(z + tier_emb[n_q]) when the decoder carries a tier embedding (patch tiercond). n_q: int or
    LongTensor [B]."""
    if getattr(dec, "tier_emb", None) is None:
        return dec(z)
    if isinstance(n_q, int):
        n_q = torch.full((z.shape[0],), n_q, device=z.device)
    return dec(z + dec.tier_emb(n_q.to(z.device)).to(z.dtype)[..., None])


class DACEncoder(nn.Module):
    """Patch armA_ctrl: DAC's own encoder (a frozen fp32 copy) in the encoder slot, so a decoder-only fine-tune on
    DAC-encoder codes runs the exact --arch encoder code path (losses, BWE, n_q mix) of the arm 4 fine-tunes."""

    def __init__(self, dac_encoder):
        super().__init__()
        import copy

        self.e = copy.deepcopy(dac_encoder).float()
        for p in self.e.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.e(x.float())


class ExtCodebook(nn.Module):
    """One extra codebook in DAC's style (patch q12_ext): in_proj 1024 -> 8 (k1 conv), 1024 raw 8-d entries, nearest
    by cosine (both l2-normalised), z_q = out_proj(entry) 8 -> 1024. Plain convs (no weight norm), so a PCA init
    can be written straight into the weights."""

    def __init__(self, dim=1024, cd=8, n=1024):
        super().__init__()
        self.in_proj = nn.Conv1d(dim, cd, 1)
        self.out_proj = nn.Conv1d(cd, dim, 1)
        self.codebook = nn.Embedding(n, cd)


class ExtRVQ(FrozenRVQ):
    """Patch q12_ext: FrozenRVQ (DAC's 9 codebooks, frozen) + n_ext trainable ExtCodebooks after them, on the
    residual after q9 (detached: the encoder must be frozen). Per-item n_q up to 9 + n_ext. The extra codebooks'
    DAC-style losses (commitment mse(z_e, sg(c)) x w_commit + codebook mse(sg(z_e), c) x w_codebook, per item, mean
    over the item's active extra codebooks) are left in self.last_ext_loss [B]. Straight-through as DAC."""

    def __init__(self, quantizer, n_ext, soft_tau=0.0, w_commit=0.25, w_codebook=1.0, grad="st"):
        super().__init__(quantizer, soft_tau)
        self.grad = grad  # patch q12c: 'out' = value c, gradient to the entries and (straight-through) in_proj
        self.n_base = len(self.q.quantizers)
        self.ext = nn.ModuleList(ExtCodebook() for _ in range(n_ext))
        self.w_commit, self.w_codebook = w_commit, w_codebook
        self.last_ext_loss = None

    def forward(self, z, n_q, st=True):
        B = z.shape[0]
        if isinstance(n_q, int):
            n_q = torch.full((B,), n_q, device=z.device)
        nb = self.n_base
        assert int(n_q.max()) <= nb + len(self.ext), (int(n_q.max()), nb, len(self.ext))
        zq, codes, commit = super().forward(z, n_q.clamp(max=nb), st)
        self.last_ext_loss = torch.zeros(B, device=z.device)
        n_ext = int(n_q.max()) - nb
        if n_ext <= 0:
            return zq, codes, commit
        with torch.autocast(device_type=z.device.type, enabled=False):
            r = (z.float() - zq).detach()  # every item quantised through all 9 here (n_q.max() > 9)
            extra, lsum = [], 0.0
            for j in range(n_ext):
                cb = self.ext[j]
                ze = cb.in_proj(r)
                w = cb.codebook.weight.float()
                idx = torch.einsum("bdf,nd->bfn", F.normalize(ze.detach(), dim=1), F.normalize(w.detach(), dim=1)
                                   ).argmax(-1)
                c = w[idx].transpose(1, 2)
                on = (n_q > nb + j).float()
                lsum = lsum + on * (self.w_commit * (ze - c.detach()).square().mean([1, 2])
                                    + self.w_codebook * (ze.detach() - c).square().mean([1, 2]))
                if st and self.grad == "out":  # patch q12c: output losses reach the entries too
                    zi = cb.out_proj(c + (ze - ze.detach()))
                else:
                    zi = cb.out_proj(ze + (c - ze).detach() if st else c)
                zq = zq + zi * on.view(B, 1, 1)
                r = r - zi.detach()
                extra.append(idx)
            self.last_ext_loss = lsum / (n_q - nb).clamp(min=1).float()
        return zq, torch.cat([codes, torch.stack(extra, 1)], 1), commit


class EncoderCodec(nn.Module):
    """audio -> trained encoder -> FrozenRVQ (per-item n_q) -> decoder. Only the encoder trains, unless
    train_dec (the joint fine-tune). The quantiser's and decoder's weights ride along in the state dict, so a
    checkpoint is self-contained."""

    is_encoder = True

    def __init__(self, enc, rvq, dec, train_dec=False):
        super().__init__()
        self.enc, self.rvq, self.dec = enc, rvq, dec
        self.width = getattr(dec, "width", 512)
        for p in dec.parameters():
            p.requires_grad_(train_dec)
        self.last_codes = None

    def forward(self, x, n_q):
        zq, codes, commit = self.rvq(self.enc(x), n_q)
        self.last_codes = codes
        if getattr(self, "pf", None) is not None:  # patch pf_lowtier: the post-filter on the frozen decoder
            return self.pf(self.dec, zq, n_q)
        return tier_cond_decode(self.dec, zq, n_q)


# ---------------------------------------------------------------- band-limited losses (SBR-style BWE)

def bwe_cutoffs(spec, max_q, n_q):
    """spec: '0' (off), one cutoff in Hz for every tier <= max_q, or a comma list for q1, q2, ... (the last value
    repeats up to max_q). n_q LongTensor [B] -> cutoff per item [B] (0 = full band)."""
    vals = [float(v) for v in str(spec).split(",") if v != ""]
    if not vals or all(v <= 0 for v in vals):
        return torch.zeros(n_q.shape[0], device=n_q.device)
    per = [vals[min(q - 1, len(vals) - 1)] if q <= max_q else 0.0 for q in range(1, 10)]
    return torch.tensor(per, device=n_q.device)[(n_q - 1).clamp(0, 8)]


def lowpass(x, cut, sr=44100, trans=500.0):
    """x [B, 1, T], cut [B] Hz -> x with a raised-cosine FFT low-pass per item (transition `trans` Hz above cut)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        T = x.shape[-1]
        X = torch.fft.rfft(x.float(), dim=-1)
        f = torch.fft.rfftfreq(T, 1 / sr).to(x.device)  # [T/2 + 1]
        u = ((f[None] - cut[:, None]) / trans).clamp(0, 1)  # [B, T/2 + 1]
        g = 0.5 * (1 + torch.cos(math.pi * u))
        return torch.fft.irfft(X * g[:, None], T, dim=-1)


def lowband_logmel(ref, y, cutoff, sr=44100):
    """metrics.score's log-mel (dB, 80 mels, n_fft 2048), over the mel bands whose peak is below `cutoff` Hz,
    mean over channels. ref, y [C, T] numpy."""
    import numpy as np

    import metrics

    ref = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    n = min(ref.shape[-1], y.shape[-1])
    fb = metrics._mel_fb(sr, 2048)
    keep = fb.argmax(1).float() * sr / 2048 < cutoff
    out = []
    for r, x in zip(ref[..., :n], y[..., :n]):
        if float(r.abs().max()) < 1e-4:
            continue
        mr, my = fb @ metrics._mag(r, 2048), fb @ metrics._mag(x, 2048)
        floor = mr.max() * 1e-4
        d = 20 * (torch.log10(my.clamp_min(floor)) - torch.log10(mr.clamp_min(floor)))
        out.append(float(d[keep].abs().mean()))
    return float(np.mean(out)) if out else float("nan")


# ---------------------------------------------------------------- cost probe

CONFIGS = {  # name: cfg
    "STFT S 384x6": {"enc": "stft", "dim": 384, "blocks": 6, "mlp": 1152, "ffts": [2048, 512], "power": 0.3,
                     "embed_k": 1},
    "STFT M 512x8": {"enc": "stft", "dim": 512, "blocks": 8, "mlp": 1536, "ffts": [2048, 512], "power": 0.3,
                     "embed_k": 1},
    "STFT M 512x8 1-res": {"enc": "stft", "dim": 512, "blocks": 8, "mlp": 1536, "ffts": [2048], "power": 0.3,
                           "embed_k": 1},
    "STFT L 768x8": {"enc": "stft", "dim": 768, "blocks": 8, "mlp": 2304, "ffts": [2048, 512], "power": 0.3,
                     "embed_k": 1},
    "conv d16": {"enc": "conv", "conv_width": 16},
    "conv d24": {"enc": "conv", "conv_width": 24},
    "conv d32": {"enc": "conv", "conv_width": 32},
}


def cost(model, seconds=4.0):
    """-> (params, GMAC per second of audio and channel, Snake evaluations per second in millions)."""
    from nnutil import count_cost, n_params

    T = int(44100 * seconds) // FRAME * FRAME
    c = count_cost(model, torch.zeros(1, 1, T))
    extra = getattr(model, "extra_mac_per_frame", lambda: 0.0)() * (T // FRAME)
    sec = T / 44100
    return n_params(model), (c["mac"] + extra) / sec / 1e9, c["snake"] / sec / 1e6


def time_1t(model, seconds, reps=3):
    x = torch.randn(1, 1, int(44100 * seconds) // FRAME * FRAME) * 0.1
    with torch.inference_mode():
        model(x[..., :FRAME * 16])
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            model(x)
            best = min(best, time.perf_counter() - t)
    return best / seconds


def rust_estimate(gmac):
    """Seconds for 90 s of stereo, scaling the Rust CLI's DAC-encoder time by MACs (DAC's Snake convs run at about
    131 GMAC/s there; a GEMM-shaped ConvNeXt should do at least as well, so this is an upper estimate)."""
    return DAC_ENC_RUST_S * gmac / DAC_ENC_GMAC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-dac", action="store_true", help="skip the stock DAC encoder (no weights here)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = []
    if not a.no_dac:
        from teacher import load_teacher

        t = load_teacher("cpu")
        rows.append(("stock DAC encoder", t.encoder))
    for name, cfg in CONFIGS.items():
        rows.append((name, build_encoder(cfg).eval()))
    print(f"{'encoder':22s} {'params':>11s} {'GMAC/s':>7s} {'M Snake/s':>9s} {'CPU ' + str(a.threads) + 't x RT':>10s} "
          f"{'Rust est. s/90 s stereo':>24s}")
    for name, m in rows:
        p, g, sn = cost(m)
        rt = time_1t(m, a.seconds)
        print(f"{name:22s} {p:11,d} {g:7.2f} {sn:9.1f} {rt:10.4f} {rust_estimate(g):24.1f}")


if __name__ == "__main__":
    main()
