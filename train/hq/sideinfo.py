# SPDX-License-Identifier: GPL-3.0-or-later
"""Side information on top of the trained student: a learned, variable-rate residual code, sent next to the DAC
codes, that tells the post-filter (postfilter.py) what the student got wrong.

    student (frozen, fp32, no_grad): z -> y
    decoder-side features: STFT(y) compressed (postfilter.PostFilter.spec), z, and the rate setting c = (t, off)
    encoder (offline, sees the true audio x):  [spec(x), spec(y)] + z + c -> ConvNeXt -> s [B, ch, T / down]
    entropy model (decoder side, no s):        spec(y) + z + c -> small ConvNeXt -> mu, sigma per element of s
    quantiser: residual r = s - mu; training: r + U(-1/2, 1/2) for the rate, round(r) straight-through for the
               decoder (the "mixed" quantiser); eval: round(r).  s_hat = mu + r_hat.
    rate: bits = -log2 P(r_hat | mu, sigma), P = the Gaussian(0, sigma) mass of [r_hat - 1/2, r_hat + 1/2]. This
          is the coded size up to the arithmetic coder's overhead (a few bytes per file), and it is conditional
          on everything the decoder already has, so only what the student lost costs bits.
    decoder: the post-filter's trunk input += Conv1d(ch -> dim)(s_hat, repeated to frame rate) + MLP(c).

Variable rate in one model: per training item, t ~ U(0, 1) sets lambda = lam_min (lam_max / lam_min)^t, the
loss is the perceptual stack + lambda x (side kbps), and t is an input of all three parts. With probability
p_off an item is "off": s_hat = 0, zero bits, t = 1. The off point is the zero-bit post-filter, so one run gives
the whole curve from 0 bits up. The rate setting would travel in the file header (one byte).

Fixed-rate mode (cfg mode "fixed", train.py --si-mode ceiling / constrained): t is a constant (FIXED_T) for
every item that is on, so the rate is set by the training objective, not by t: stage 1 ("ceiling") trains with
a tiny fixed lambda for the best sound, stage 2 ("constrained") fine-tunes from it with lambda adapted by dual
ascent to the least rate that keeps the distortion within epsilon of stage 1's (train.py). In this mode the
entropy model also gets a fit term on the detached code (self.last['kbps_em']): its only job is to price the
code, and under a tiny lambda it would otherwise barely learn (the decoder never sees mu: s_hat = s + noise).

The head of the post-filter is zero-initialised, so at step 0 the output is the student's y at every rate.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

import postfilter
from vocos_student import FRAME, LATENT, ConvNeXtBlock

SIGMA_MIN = 0.11  # lower bound of the entropy model's scale (Balle et al.)
# Gain units (as in variable-rate image codecs): the encoder's output is multiplied by a per-channel gain that
# depends on the rate setting, g(t) = exp((1 - t) log G0 + t log G1), learned, initialised G0 = 16, G1 = 0.5;
# the quantiser rounds at that scale and the decoder sees s_hat / g(t). So the code's resolution, and its rate,
# falls with t from the start. Without them (one fixed gain), CPU checks collapsed to zero bits at every t:
# the shared encoder first solves the lambda-averaged problem (mean lambda ~2.9 per kbps), where sending
# nothing is optimal, and a collapsed code has no gradient left to learn t from.
G0, G1 = 16.0, 0.5
P_MIN = 1e-9  # lower bound of a symbol's probability (about 30 bits)
FIXED_T = 0.0  # the rate setting of every "on" item in fixed-rate mode (gain units at G0)


class Cond(nn.Module):
    """The rate setting c = (t, off) [B, 2] -> [B, dim, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, dim))

    def forward(self, c):
        return self.net(c)[:, :, None]


class Stack(nn.Module):
    """k1 embed (+ z embed + condition) -> LayerNorm -> ConvNeXt blocks -> LayerNorm -> Linear to out_ch."""

    def __init__(self, in_ch, dim, blocks, out_ch, use_z):
        super().__init__()
        self.embed = nn.Conv1d(in_ch, dim, 1)
        self.z_embed = nn.Conv1d(LATENT, dim, 1) if use_z else None
        self.cond = Cond(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, 3 * dim, 1.0 / blocks) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.out = nn.Linear(dim, out_ch)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, z, c):
        x = self.embed(feat) + self.cond(c)
        if self.z_embed is not None:
            x = x + self.z_embed(z)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2).float()
        for b in self.blocks:
            x = b(x)
        return self.out(self.final_norm(x.transpose(1, 2))).transpose(1, 2).float()


def likelihood(r, sigma):
    """P(r) for integer (eval) or noisy (training) r under the discretised Gaussian(0, sigma); computed on the
    lower tail, which is accurate far out. (Reporting only: the rate uses bits(), which keeps its gradient.)"""
    v = r.abs()
    return (torch.special.ndtr((0.5 - v) / sigma) - torch.special.ndtr((-0.5 - v) / sigma)).clamp_min(P_MIN)


def bits(r, sigma):
    """-log2 P(r) in log space: log P = log ndtr(a) + log(1 - exp(log ndtr(b) - log ndtr(a))), a = (1/2 - |r|) /
    sigma, b = (-1/2 - |r|) / sigma. A clamped probability (P >= 1e-9) has zero gradient far in the tail, and
    there the rate stopped responding to lambda (a CPU check: 7.8 kbps at every lambda)."""
    v = r.abs()
    la = torch.special.log_ndtr((0.5 - v) / sigma)
    lb = torch.special.log_ndtr((-0.5 - v) / sigma)
    return -(la + torch.log1p(-torch.exp((lb - la).clamp_max(-1e-6)))) / 0.6931471805599453


class SideCodec(nn.Module):
    def __init__(self, pf, ch=8, down=1, enc_dim=384, enc_blocks=6, em_dim=192, em_blocks=2):
        super().__init__()
        assert pf.k == 1, "side information needs the post-filter at hop 512"
        self.pf, self.ch, self.down = pf, ch, down
        # Training randomness (rate settings, off items, quantisation noise) comes from this generator, which
        # train.py re-seeds from (seed, step) every step: resume-exact, untouched by other users of the global
        # RNG (a DataLoader draws from it when its iterator starts, which differs after a resume). None = global.
        self.gen = None
        self.quant = "noise"  # what the decoder sees in training: "noise" (r + U) or "ste" (round, straight-through)
        use_z = pf.use_z
        self.enc = Stack(4 * pf.bins, enc_dim, enc_blocks, ch, use_z)
        self.em = Stack(2 * pf.bins, em_dim, em_blocks, 2 * ch, use_z)
        dim = pf.embed.out_channels
        self.log_gain = nn.Parameter(torch.log(torch.tensor([[G0] * ch, [G1] * ch])))  # [2, ch]: t = 0, t = 1
        self.side_embed = nn.Conv1d(ch, dim, 1)
        self.dec_cond = Cond(dim)
        for m in (self.side_embed, *self.dec_cond.modules()):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def _pool(self, v):  # frame rate -> side rate
        return F.avg_pool1d(v, self.down, self.down, ceil_mode=True) if self.down > 1 else v

    def _up(self, v, T):
        return v.repeat_interleave(self.down, dim=-1)[..., :T] if self.down > 1 else v

    def forward(self, y, z, c, x=None, em_fit=False):
        """y, x [B, 1, L]; z [B, 1024, T]; c [B, 2] = (t, off). -> (audio [B, 1, L], bits [B, ch, T / down]).
        x None: the off point (nothing is sent; the entropy model still runs, as a decoder would).
        em_fit (training): also set self.em_bits, the bits of the detached code (gradient to the entropy model
        only)."""
        self.em_bits = None
        _, _, fy = self.pf.spec(y)
        T = fy.shape[-1]
        zz = z if self.pf.use_z else None
        p = self._pool(self.em(fy, zz, c))  # the stacks run under the caller's autocast and return fp32
        tt = c[:, :1, None].float()  # [B, 1, 1]
        gain = torch.exp((1 - tt) * self.log_gain[0][None, :, None] + tt * self.log_gain[1][None, :, None])
        s = None
        if x is not None:
            _, _, fx = self.pf.spec(x)
            s = gain * self._pool(self.enc(torch.cat([fx, fy], 1), zz, c))
        with torch.autocast(device_type=y.device.type, enabled=False):
            mu, sc = p.chunk(2, dim=1)
            sigma = F.softplus(sc) + SIGMA_MIN
            if s is None:
                s_hat = torch.zeros_like(mu)
                nbits = torch.zeros_like(mu)
            else:
                r = s - mu
                r_hard = torch.round(r)
                if self.training:
                    r_rate = r + torch.rand(r.shape, generator=self.gen, device=r.device) - 0.5
                    r_dec = r_rate if self.quant == "noise" else r + (r_hard - r).detach()
                else:
                    r_rate = r_dec = r_hard
                on = (1 - c[:, 1]).view(-1, 1, 1)
                s_hat = (mu + r_dec) * on
                nbits = bits(r_rate, sigma) * on
                if em_fit and self.training:
                    self.em_bits = bits(r_rate.detach() + mu.detach() - mu, sigma) * on
            self.s_dec = s_hat / gain  # what the decoder reads (kept for tests and analysis)
            extra = self.side_embed(self._up(self.s_dec, T)) + self.dec_cond(c.float())
        return self.pf(y, zz, extra=extra), nbits


class SideInfoModel(nn.Module):
    """z (+ the true audio x when encoding) -> student (frozen) -> side codec. Training samples the rate setting
    per item; eval uses self.eval_t (a float in [0, 1], or None = off). After each forward, self.last holds
    {'kbps': [B] side kbps per channel, 't': [B], 'off': [B]}; in eval mode self.bits accumulates bits and
    self.frame_bits keeps the last forward's bits per side frame [B, T / down]."""

    needs_target = True

    def __init__(self, student, codec, lam=(0.02, 20.0), p_off=0.15, eval_t=0.5, mode="vr"):
        super().__init__()
        self.student, self.codec = student, codec
        self.pf = codec.pf
        self.width = getattr(student, "width", 512)
        self.lam, self.p_off, self.eval_t = tuple(lam), p_off, eval_t
        self.mode = mode  # "vr": t sampled per item; "fixed": t = FIXED_T (eval_t only switches on / off)
        self.em_fit = False  # train.py sets it: the entropy-model fit term (self.last["kbps_em"])
        self.last, self.bits, self.frame_bits = {}, 0.0, None

    def lam_of(self, t):
        lo, hi = self.lam
        return lo * (hi / lo) ** t

    def forward(self, z, x=None):
        B, dev = z.shape[0], z.device
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            y = self.student(z.float()).float()
        if x is None:  # nothing to encode: the off point (also count_cost's probe)
            t, off = torch.ones(B, device=dev), torch.ones(B, device=dev)
        elif self.training:
            g = self.codec.gen
            t = torch.rand(B, generator=g, device=dev)
            off = (torch.rand(B, generator=g, device=dev) < self.p_off).float()
            if self.mode == "fixed":
                t = torch.full_like(t, FIXED_T)
            t = torch.where(off > 0, torch.ones_like(t), t)
        elif self.eval_t is None:
            t, off = torch.ones(B, device=dev), torch.ones(B, device=dev)
        else:
            tv = FIXED_T if self.mode == "fixed" else float(self.eval_t)
            t, off = torch.full((B,), tv, device=dev), torch.zeros(B, device=dev)
        out, bits = self.codec(y, z, torch.stack([t, off], 1), None if x is None else x.float(), self.em_fit)
        sec = y.shape[-1] / 44100
        self.last = {"kbps": bits.sum((1, 2)) / sec / 1000, "t": t, "off": off}
        if self.codec.em_bits is not None:
            self.last["kbps_em"] = self.codec.em_bits.sum((1, 2)) / sec / 1000
        if not self.training:
            self.bits += float(bits.sum())
            self.frame_bits = bits.sum(1).detach()
        return out


def build(cfg):
    """cfg: the postfilter keys (dim, blocks, mlp, hop, use_z, student, student_sha) + ch, down, enc_dim,
    enc_blocks, em_dim, em_blocks, lam, p_off, mode ("vr", the default, or "fixed"). The eval rate setting is
    not part of the architecture: set model.eval_t (default 0.5; None = off; in fixed mode any value = on)."""
    pfm = postfilter.build({**cfg, "arch": "postfilter"})
    codec = SideCodec(pfm.pf, cfg["ch"], cfg["down"], cfg["enc_dim"], cfg["enc_blocks"], cfg["em_dim"],
                      cfg["em_blocks"])
    return SideInfoModel(pfm.student, codec, cfg.get("lam", (0.02, 20.0)), cfg["p_off"], cfg.get("eval_t", 0.5),
                         cfg.get("mode", "vr"))


def parse_t(v):
    """'off' -> None, else a float in [0, 1]."""
    return None if str(v).lower() == "off" else float(v)


def decoder_cost(model, frames=87):
    """MACs per second of the decode side (post-filter + entropy model + side embed; not the encoder, not the
    student), measured with forward hooks."""
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Linear):
            macs[0] += out.numel() // out.shape[-1] * m.in_features * m.out_features
        else:
            macs[0] += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    mods = [m for part in (model.codec.pf, model.codec.em, model.codec.side_embed, model.codec.dec_cond)
            for m in part.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    hs = [m.register_forward_hook(hook) for m in mods]
    with torch.inference_mode():
        model(torch.zeros(1, LATENT, frames))
    for h in hs:
        h.remove()
    return macs[0] / (frames * FRAME / 44100)
