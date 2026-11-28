# SPDX-License-Identifier: GPL-3.0-or-later
"""Low-tier post-filter on the frozen HQ decoder, arm A: in the decoder's own frame domain, before the inverse STFT
(POSTFILTER_LOWTIER_RUN.md, 2026-11-27; patch pf_lowtier).

    frozen decoder (vocos_student.VocosDecoder, no grad): z [B, 1024, T] -> trunk -> final LayerNorm -> h [B, T, 512]
                                                          -> head -> S [B, 1025, T] (the spectrum its iSTFT takes)
    post-filter, items at <= max_q codebooks only:
        features: S power-law compressed (|c| = |S|^0.3), real and imaginary parts [B, 2050, T]  -> k1 conv to dim
                + h through a k1 conv (512 -> dim)
                + a learned embedding of the codebook count (q1..max_q), zero-initialised
        -> LayerNorm -> N ConvNeXt blocks (the decoder's own, k7 depthwise) -> LayerNorm
        -> head Linear dim -> 4 x 1025: complex mask m (tanh per part) and complex residual r, scaled by
           0.1 x the local spectral level (|S| smoothed over +-8 bins): S' = S (1 + m) + r
        -> the decoder's own iSTFT and overlap-add.
    items above max_q: S' = S exactly (the frozen decoder's output, bit for bit).

The head is zero-initialised, so at step 0 S' = S for every item. The decoder's trunk and head are run here by the
same operations, in the same order, as VocosDecoder.forward, so the bypass is the frozen decoder's own output.

Receptive field (latent frames per side, for an exact chunked decode): the decoder's trunk 3 x (1 + blocks) = 27,
plus the post-filter's 3 x its blocks (18 for 6), plus the iSTFT overlap (2): 47, against the decoder's 29 alone.
"""
import torch
import torch.nn.functional as F
from torch import nn

from postfilter import COMP, RES_SCALE, STFT
from vocos_student import FRAME, ConvNeXtBlock


def dec_parts(dec, z):
    """VocosDecoder.forward up to the iSTFT: z -> (h [B, T, dim] after the final LayerNorm, S complex [B, bins, T]).
    The same operations in the same order as VocosDecoder.forward, so dec.istft(S) == dec(z)[:, 0]."""
    x = dec.embed(z)
    x = dec.norm(x.transpose(1, 2)).transpose(1, 2).float()
    for b in dec.blocks:
        x = b(x)
    h = dec.final_norm(x.transpose(1, 2))
    with torch.autocast(device_type=z.device.type, enabled=False):
        o = dec.head(h.float())
        B, T, _ = o.shape
        o = o.reshape(B, T * dec.k, dec.n_fft + 2).transpose(1, 2)
        logmag, p = o.chunk(2, dim=1)
        mag = logmag.exp().clamp(max=1e2)
        S = torch.complex(mag * torch.cos(p), mag * torch.sin(p))
    return h, S


class LowTierPF(nn.Module):
    """The post-filter alone (the frozen decoder is passed in, never owned: its weights stay under G.dec)."""

    def __init__(self, dec_dim=512, dim=384, blocks=6, mlp=None, n_fft=2048, max_q=5, mode="complex", mag_bound=1.0,
                 n_bands=48, sr=44100):
        super().__init__()
        mlp = mlp or 3 * dim
        self.bins = n_fft // 2 + 1
        self.max_q = max_q
        assert mode in ("complex", "mag", "band"), mode
        # mode "mag" (owner 2026-11-27, trial E; default off): the magnitude only, |S'| = |S| exp(b tanh(m)), the
        # phase of S kept exactly; head 1 x bins, no complex residual
        self.mode, self.mag_bound = mode, mag_bound
        # mode "band" (owner 2026-11-27, trial F; default off): as "mag", but one log-gain per band (n_bands
        # triangular bands, centres equally spaced on the mel scale from 0 Hz to Nyquist), interpolated to the bins by
        # the overlapping triangles (a partition of unity: a constant band gain is that gain in every bin)
        self.n_bands = n_bands
        self.cfg = {"dim": dim, "blocks": blocks, "mlp": mlp, "max_q": max_q,
                    **({"mode": mode, "mag_bound": mag_bound} if mode != "complex" else {}),
                    **({"n_bands": n_bands} if mode == "band" else {})}
        if mode == "band":
            self.register_buffer("band_w", band_matrix(n_fft, n_bands, sr), persistent=False)  # [bins, n_bands]
        self.last_g = None  # mode "band": the band log-gains [B_on, n_bands, T] of the last forward (--w-gain-tv)
        self.embed = nn.Conv1d(2 * self.bins, dim, 1)
        self.h_embed = nn.Conv1d(dec_dim, dim, 1)
        self.tier_emb = nn.Embedding(max_q + 1, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, mlp, 1.0 / blocks) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, {"complex": 4 * self.bins, "mag": self.bins, "band": n_bands}[mode])
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.tier_emb.weight)
        nn.init.zeros_(self.head.weight)  # identity at step 0
        nn.init.zeros_(self.head.bias)
        self.last = None  # (S', on [B] bool) of the last forward, for the frame-consistency loss

    def correct(self, S, h, nq):
        """S complex [B, bins, T], h [B, T, dec_dim], nq LongTensor [B] (all <= max_q) -> S'."""
        dev = S.device.type
        with torch.autocast(device_type=dev, enabled=False):
            mag = S.abs()
            c = S * (mag + 1e-5).pow(COMP - 1)
            feat = torch.cat([c.real, c.imag], dim=1)
        x = self.embed(feat) + self.h_embed(h.transpose(1, 2)) + self.tier_emb(nq)[..., None]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2).float()
        for b in self.blocks:
            x = b(x)
        hh = self.final_norm(x.transpose(1, 2))
        with torch.autocast(device_type=dev, enabled=False):
            o = self.head(hh.float()).transpose(1, 2)  # [B, 4 bins, T] (mode "mag": [B, bins, T])
            if self.mode == "mag":  # a real gain per bin: S x 1.0 is S exactly at step 0
                return S * torch.exp(self.mag_bound * torch.tanh(o))
            if self.mode == "band":  # band log-gains -> bins; exp(W 0) = 1 exactly at step 0
                g = self.mag_bound * torch.tanh(o)
                self.last_g = g
                return S * torch.exp(torch.einsum("kn,bnt->bkt", self.band_w, g))
            mr, mi, rr, ri = o.chunk(4, dim=1)
            env = F.avg_pool1d(mag.transpose(1, 2), 17, 1, 8, count_include_pad=False).transpose(1, 2)
            return S * torch.complex(1 + torch.tanh(mr), torch.tanh(mi)) + torch.complex(rr, ri) * (RES_SCALE * env)

    def forward(self, dec, z, n_q):
        """dec: the frozen VocosDecoder; z [B, 1024, T]; n_q int or LongTensor [B] -> audio [B, 1, 512 T]."""
        B = z.shape[0]
        if isinstance(n_q, int):
            n_q = torch.full((B,), n_q, device=z.device)
        n_q = n_q.to(z.device)
        on = n_q <= self.max_q
        with torch.no_grad():  # the decoder is frozen: no graph through it
            h, S = dec_parts(dec, z)
        if not bool(on.any()):
            self.last = (S, on)
            with torch.autocast(device_type=z.device.type, enabled=False):
                return dec.istft(S)[:, None].float()
        idx = torch.nonzero(on)[:, 0]
        Sp = self.correct(S[idx], h[idx], n_q[idx])
        if bool(on.all()):
            out = Sp
        else:
            out = S.clone()
            out[idx] = Sp
        self.last = (out, on)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return dec.istft(out)[:, None].float()

    def receptive_field(self, dec):
        ola = -(-(dec.n_fft // 2) // FRAME)
        return 3 * (1 + len(dec.blocks)) + 3 * len(self.blocks) + ola


def band_matrix(n_fft=2048, n_bands=48, sr=44100):
    """[bins, n_bands] triangular interpolation weights: band centres equally spaced in mel (HTK) from 0 Hz to
    Nyquist; each bin gets the two neighbouring triangles' linear weights (rows sum to 1)."""
    mel = lambda f: 2595.0 * torch.log10(1.0 + f / 700.0)
    imel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    c = imel(torch.linspace(0.0, float(mel(torch.tensor(sr / 2.0))), n_bands, dtype=torch.float64))
    f = torch.arange(n_fft // 2 + 1, dtype=torch.float64) * sr / n_fft
    w = torch.zeros(len(f), n_bands, dtype=torch.float64)
    j = torch.searchsorted(c, f, right=True).clamp(1, n_bands - 1)  # c[j-1] <= f < c[j]
    t = ((f - c[j - 1]) / (c[j] - c[j - 1])).clamp(0.0, 1.0)
    w[torch.arange(len(f)), j - 1] = 1.0 - t
    w[torch.arange(len(f)), j] += t
    return w.float()


def gain_tv(g, frames=None):
    """Temporal smoothness of band log-gains g [B, n_bands, T]: mean |g_t - g_{t-1}| (over frames, if given)."""
    if frames is not None:
        g = g[..., frames]
    return (g[..., 1:] - g[..., :-1]).abs().mean()


def consistency(S, stft, istft, frames=None):
    """Frame consistency: ||STFT(iSTFT(S)) - S||_1 / ||S||_1 over complex spectra, per item -> [B]. stft / istft:
    the decoder's own parameters ("same" padding: postfilter.STFT(n_fft, hop), dec.istft). frames: a slice of the
    frames to score (the loss segment, so the crop's edge frames don't count)."""
    with torch.autocast(device_type=S.device.type, enabled=False):
        S2 = stft(istft(S))
        if frames is not None:
            S, S2 = S[..., frames], S2[..., frames]
        return (S2 - S).abs().sum((1, 2)) / S.abs().sum((1, 2)).clamp(min=1e-8)


def short_loss(y_hat, y, ffts=(128, 256, 512)):
    """Short-window spectral loss: log-magnitude L1 at n_fft 128, 256, 512 (hop n_fft / 4, Hann), mean over sizes.
    y_hat, y [B, 1, L] -> scalar."""
    with torch.autocast(device_type=y.device.type, enabled=False):
        tot = 0.0
        for n in ffts:
            w = torch.hann_window(n, device=y.device)
            a = torch.stft(y_hat[:, 0].float(), n, n // 4, window=w, return_complex=True).abs()
            b = torch.stft(y[:, 0].float(), n, n // 4, window=w, return_complex=True).abs()
            tot = tot + (torch.log(a + 1e-5) - torch.log(b + 1e-5)).abs().mean()
        return tot / len(ffts)
