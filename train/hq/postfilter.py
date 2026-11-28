# SPDX-License-Identifier: GPL-3.0-or-later
"""A zero-bit post-filter on the trained student: a small STFT-domain corrector that sees only what the decoder
already has (the student's waveform y, and optionally the quantised latent z) and is trained against the
original audio with the student's own loss stack. No side information is sent.

    student (frozen, fp32, no_grad): z [B, 1024, F] -> y [B, 1, 512 F]
    post-filter: STFT(y) (Hann, n_fft = 4 hop, "same" padding: frames line up with the latent frames)
                 -> features: power-law compressed complex spectrum, real and imaginary parts (|c| = |S|^0.3)
                 -> k1 conv to dim (+ k1 conv of z when --pf-z) -> LayerNorm -> N ConvNeXt blocks (from
                    vocos_student.py) at the STFT frame rate -> LayerNorm
                 -> head Linear dim -> 4 x bins: a complex mask m (tanh per part) and a complex residual r
                    (scaled by 0.1 x the local spectral level, |S| smoothed over +-8 bins)
                 -> S' = S (1 + m) + r -> inverse STFT (vocos_student.ISTFT)

The head is zero-initialised, so at step 0 m = r = 0, S' = S and the output is the student's y up to float
rounding (the analysis/synthesis pair is an exact identity). Training can only start from the student.

Everything is per frame or a finite stack of k7 depthwise convs, so the post-filter's receptive field is
3 frames per block plus the STFT overlap, and a chunked decode with that much context per side is exact.

The "student" here is the frozen Vocos decoder (vocos_student.py). pf_lowtier.py reuses STFT, COMP and RES_SCALE.
"""
import hashlib
import os

import torch
import torch.nn.functional as F
from torch import nn

from vocos_student import FRAME, LATENT, ISTFT, ConvNeXtBlock

COMP = 0.3  # power-law compression of the input spectrum
RES_SCALE = 0.1  # the additive residual, relative to the local spectral level


class STFT(nn.Module):
    """Same-padded STFT matching vocos_student.ISTFT: L samples -> L / hop frames, and ISTFT(STFT(y)) = y."""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop, self.pad = n_fft, hop, (n_fft - hop) // 2
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, y):  # [B, L] -> complex [B, n_fft/2 + 1, L / hop]
        x = F.pad(y, (self.pad, self.pad))
        frames = x.unfold(-1, self.n_fft, self.hop) * self.window
        return torch.fft.rfft(frames, dim=-1).transpose(1, 2)


class PostFilter(nn.Module):
    def __init__(self, dim=512, blocks=8, mlp=None, hop=512, use_z=False):
        super().__init__()
        assert FRAME % hop == 0, hop
        mlp = mlp or 3 * dim
        self.k = FRAME // hop
        self.n_fft = 4 * hop
        self.bins = self.n_fft // 2 + 1
        self.use_z = use_z
        self.stft = STFT(self.n_fft, hop)
        self.istft = ISTFT(self.n_fft, hop)
        self.embed = nn.Conv1d(2 * self.bins, dim, 1)
        self.z_embed = nn.Conv1d(LATENT, dim, 1) if use_z else None
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, mlp, 1.0 / blocks) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, 4 * self.bins)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.head.weight)  # identity at step 0
        nn.init.zeros_(self.head.bias)

    def spec(self, y):
        """y [B, 1, L] -> (S complex [B, bins, T], |S|, features [B, 2 bins, T]: compressed real and imaginary)."""
        with torch.autocast(device_type=y.device.type, enabled=False):
            S = self.stft(y[:, 0].float())
            mag = S.abs()
            c = S * (mag + 1e-5).pow(COMP - 1)
            return S, mag, torch.cat([c.real, c.imag], dim=1)

    def forward(self, y, z=None, extra=None):
        """y [B, 1, L], z [B, 1024, L / 512] -> [B, 1, L]. extra [B, dim, T]: more conditioning added to the
        trunk's input (sideinfo.py uses it for the side latent and the rate setting)."""
        dev = y.device.type
        S, mag, feat = self.spec(y)
        x = self.embed(feat)
        if self.z_embed is not None:
            zz = self.z_embed(z)
            x = x + (zz.repeat_interleave(self.k, dim=-1) if self.k > 1 else zz)
        if extra is not None:
            x = x + extra
        x = self.norm(x.transpose(1, 2)).transpose(1, 2).float()
        for b in self.blocks:
            x = b(x)
        h = self.final_norm(x.transpose(1, 2))
        with torch.autocast(device_type=dev, enabled=False):
            o = self.head(h.float()).transpose(1, 2)  # [B, 4 bins, T]
            mr, mi, rr, ri = o.chunk(4, dim=1)
            # The residual is scaled by the local spectral level (|S| smoothed over +-8 bins) x RES_SCALE, so it
            # adds content at the level of the neighbouring bins. Scaled by the frame's RMS instead, Adam's first
            # steps (sized by lr, not by the gradient) put noise at that level into near-silent bins: in the CPU
            # smoke, 10 steps made log-mel 52% worse.
            env = F.avg_pool1d(mag.transpose(1, 2), 17, 1, 8, count_include_pad=False).transpose(1, 2)
            out = (S * torch.complex(1 + torch.tanh(mr), torch.tanh(mi))
                   + torch.complex(rr, ri) * (RES_SCALE * env))
            return self.istft(out)[:, None].float()

    def receptive_field(self):
        """Latent frames of context per side for an exact chunked decode (the student's own +-9 comes on top)."""
        stft_frames = 3 * len(self.blocks) + 2 * (-(-(self.n_fft // 2) // self.stft.hop))
        return -(-stft_frames // self.k)


# ---------------------------------------------------------------- the frozen student + post-filter

def file_sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_student(path):
    """The frozen Vocos decoder the post-filter runs on, from a train.py checkpoint of --arch vocos (raw weights G).
    The file is only read."""
    from vocos_student import make_decoder

    s = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(s, dict) and "G" in s and (s.get("arch") or {}).get("arch") == "vocos"):
        raise SystemExit(f"{path}: the post-filter needs a train.py checkpoint of a Vocos decoder (--arch vocos)")
    m = make_decoder(s["arch"])
    m.load_state_dict(s["G"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class PostFiltered(nn.Module):
    """z -> student (frozen, fp32, no_grad) -> post-filter. Only the post-filter's parameters train; the
    student's weights ride along in the state dict, so a checkpoint is self-contained."""

    def __init__(self, student, pf):
        super().__init__()
        self.student, self.pf = student, pf
        self.width = getattr(student, "width", 512)

    def forward(self, z):
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            y = self.student(z.float()).float()
        return self.pf(y, z if self.pf.use_z else None)


def pf_from_cfg(cfg):
    return PostFilter(cfg["dim"], cfg["blocks"], cfg["mlp"], cfg["hop"], cfg["use_z"])


def build(cfg):
    """cfg: {'arch': 'postfilter', dim, blocks, mlp, hop, use_z, student (path), student_sha, student_arch}. When
    the decoder file isn't on this machine (export elsewhere), an untrained one is built from student_arch: the
    checkpoint's state dict holds its weights and replaces it on load."""
    if os.path.exists(cfg["student"]):
        st = load_student(cfg["student"])
    else:
        from vocos_student import make_decoder

        print(f"post-filter: {cfg['student']} not here; the decoder's weights must come from a checkpoint")
        st = make_decoder(cfg["student_arch"]).eval()
        for p in st.parameters():
            p.requires_grad_(False)
    return PostFiltered(st, pf_from_cfg(cfg))
