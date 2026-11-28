# SPDX-License-Identifier: GPL-3.0-or-later
"""A Vocos-style decoder for DAC's quantised latent: every layer runs at the latent frame rate (86.13 Hz), and
an inverse STFT makes the samples.

The design follows Vocos (Siuzdak 2023, "Vocos: Closing the gap between time-domain and Fourier-based neural
vocoders", MIT, Copyright (c) 2023 Charactr Inc.); this is a fresh implementation of that design:

- embed: k7 conv, 1024 -> dim, then LayerNorm;
- N ConvNeXt blocks: depthwise k7 conv, LayerNorm, Linear dim -> mlp, GELU, Linear mlp -> dim, layer scale
  (initialised to 1/N), residual add;
- final LayerNorm, then the iSTFT head: Linear dim -> k (n_fft + 2), split into log-magnitude and phase angle per
  bin; magnitude = exp(.) clipped at 100, spectrum = mag (cos p + i sin p); inverse STFT with Hann window,
  n_fft = 4 hop, and "same" padding, so T latent frames give exactly 512 T samples.

hop 512 (default): one STFT frame per latent frame, n_fft 2048 (46 ms window, 21.5 Hz bins).
hop 256: the head predicts two STFT frames per latent frame (a pixel-shuffle of the head's output), n_fft 1024
(23 ms, 43 Hz bins). The backbone stays at 86 Hz either way, so the cost is the same; it is an A/B option for
transients (pre-echo) against low-frequency resolution.

Alignment: with "same" padding, STFT frame t's window is centred on sample hop t + hop/2, which for hop 512 is
the centre of DAC frame t's samples [512 t, 512 t + 512): the same frame-to-sample alignment as DAC's decoder.

Normalisation is per frame (LayerNorm over channels), never over time, so the decoder is a pure function of a
finite window of frames (see `receptive_field`) and a chunked decode with that much context per side is exact.

Mixed precision: under bf16 autocast the convs and Linears run in bf16, LayerNorm in fp32, the residual
stream in fp32, and the head (Linear, exp, cos/sin, inverse FFT, overlap-add) entirely in fp32.
"""
import torch
import torch.nn.functional as F
from torch import nn

LATENT = 1024
FRAME = 512  # samples per DAC latent frame at 44.1 kHz


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp, layer_scale):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, mlp)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(mlp, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale)))

    def forward(self, x):  # x [B, dim, T], the fp32 residual stream
        h = self.dwconv(x).transpose(1, 2)
        h = self.pw2(self.act(self.pw1(self.norm(h))))
        return x.float() + (self.gamma.float() * h.float()).transpose(1, 2)


class ISTFT(nn.Module):
    """Inverse STFT with "same" padding: T frames -> T * hop samples (torch.istft only pads like center=True)."""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.pad = (n_fft - hop) // 2
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, spec):  # complex [B, n_fft/2 + 1, T] -> [B, T * hop]
        n, hop, pad = self.n_fft, self.hop, self.pad
        T = spec.shape[-1]
        frames = torch.fft.irfft(spec, n, dim=1) * self.window[None, :, None]  # [B, n_fft, T]
        L = (T - 1) * hop + n
        y = F.fold(frames, (1, L), (1, n), stride=(1, hop))[:, 0, 0, pad:L - pad]
        wsq = self.window.square()[None, :, None].expand(1, n, T)
        env = F.fold(wsq, (1, L), (1, n), stride=(1, hop))[0, 0, 0, pad:L - pad]
        return y / env


class VocosDecoder(nn.Module):
    """Input z_q [B, 1024, F] -> audio [B, 1, 512 F]."""

    def __init__(self, dim=512, blocks=8, mlp=None, hop=512, latent=LATENT):
        super().__init__()
        assert FRAME % hop == 0, hop
        mlp = mlp or 3 * dim
        self.cfg = {"arch": "vocos", "dim": dim, "blocks": blocks, "mlp": mlp, "hop": hop}
        self.width = dim
        self.k = FRAME // hop  # STFT frames per latent frame
        self.n_fft = 4 * hop
        self.embed = nn.Conv1d(latent, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, mlp, 1.0 / blocks) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, self.k * (self.n_fft + 2))
        self.istft = ISTFT(self.n_fft, hop)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        x = self.embed(z)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2).float()
        for b in self.blocks:
            x = b(x)
        h = self.final_norm(x.transpose(1, 2))  # [B, T, dim]
        with torch.autocast(device_type=z.device.type, enabled=False):
            o = self.head(h.float())  # [B, T, k (n_fft + 2)]
            B, T, _ = o.shape
            o = o.reshape(B, T * self.k, self.n_fft + 2).transpose(1, 2)  # [B, n_fft + 2, k T]
            logmag, p = o.chunk(2, dim=1)
            mag = logmag.exp().clamp(max=1e2)
            spec = torch.complex(mag * torch.cos(p), mag * torch.sin(p))
            return self.istft(spec)[:, None].float()

    def receptive_field(self):
        """Latent frames of context needed per side for an exact chunked decode: 3 per k7 layer (embed + blocks),
        plus the frames whose iSTFT windows reach into a frame's samples."""
        ola = -(-(self.n_fft // 2) // FRAME)  # a window reaches n_fft/2 - hop/2 samples beyond its own hop
        return 3 * (1 + len(self.blocks)) + ola


def make_decoder(cfg):
    """cfg arch 'vocos' -> VocosDecoder; arch 'postfilter' -> postfilter.PostFiltered (the frozen decoder from
    cfg['student'] + the post-filter); arch 'sideinfo' -> sideinfo's side codec on the frozen decoder."""
    if not cfg:
        raise SystemExit("no decoder config: the DAC-layout student is gone; only Vocos decoders are built")
    if cfg["arch"] == "postfilter":
        import postfilter

        return postfilter.build(cfg)
    if cfg["arch"] == "sideinfo":
        import sideinfo

        return sideinfo.build(cfg)
    return VocosDecoder(cfg["dim"], cfg["blocks"], cfg["mlp"], cfg["hop"])


def istft_cost_per_frame(n_fft, k=1):
    """Analytic per-latent-frame cost of the head after its Linear: real inverse FFT (about 2.5 n log2 n flops,
    counted as half that in MACs), window + overlap-add (2 n), and exp/cos/sin of n/2 + 1 bins (3 per bin)."""
    import math

    fft_mac = 2.5 * n_fft * math.log2(n_fft) / 2
    return {"mac": k * (fft_mac + 2 * n_fft), "transcendental": k * 3 * (n_fft // 2 + 1)}
