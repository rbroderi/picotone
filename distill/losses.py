"""DAC's training losses in plain torch (no audiotools), after descript-audio-codec (MIT, Copyright (c)
2023-present, Descript):

- multi-scale log-mel L1: windows 32..2048, hop window/4, 5..320 mels, magnitude clamped at 1e-5, log10;
- discriminator = multi-period (2, 3, 5, 7, 11) + multi-band complex-STFT (windows 2048/1024/512, 5 bands);
- adversarial loss: least squares, as the released DAC code does (`--gan hinge` for the paper's hinge form);
- L1 feature matching over every discriminator layer but the last.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

SR = 44100
MEL_WINDOWS = (32, 64, 128, 256, 512, 1024, 2048)
MEL_BINS = (5, 10, 20, 40, 80, 160, 320)
BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def _mel_fb(n_fft, n_mels, device):
    import torchaudio

    return torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 0.0, SR / 2, n_mels, SR, norm="slaney",
                                                 mel_scale="slaney").to(device)


class MultiScaleMel(nn.Module):
    def __init__(self, windows=MEL_WINDOWS, mels=MEL_BINS, eps=1e-5):
        super().__init__()
        self.cfg = list(zip(windows, mels))
        self.eps = eps
        for w, m in self.cfg:
            self.register_buffer(f"fb{w}", _mel_fb(w, m, "cpu"), persistent=False)
            self.register_buffer(f"win{w}", torch.hann_window(w), persistent=False)

    def mels(self, x, w):
        s = torch.stft(x.reshape(-1, x.shape[-1]).float(), w, w // 4, window=getattr(self, f"win{w}"),
                       return_complex=True, center=True)
        return torch.matmul(s.abs().transpose(1, 2), getattr(self, f"fb{w}"))

    def forward(self, y, x):
        with torch.autocast(device_type=y.device.type, enabled=False):
            loss = 0.0
            for w, _ in self.cfg:
                a, b = self.mels(y, w), self.mels(x, w)
                loss = loss + F.l1_loss(a.clamp_min(self.eps).log10(), b.clamp_min(self.eps).log10())
        return loss


def _c2(*a, act=True, **k):
    c = weight_norm(nn.Conv2d(*a, **k))
    return nn.Sequential(c, nn.LeakyReLU(0.1)) if act else c


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            _c2(1, 32, (5, 1), (3, 1), padding=(2, 0)), _c2(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            _c2(128, 512, (5, 1), (3, 1), padding=(2, 0)), _c2(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            _c2(1024, 1024, (5, 1), 1, padding=(2, 0))])
        self.post = _c2(1024, 1, (3, 1), padding=(1, 0), act=False)

    def forward(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.reshape(x.shape[0], 1, -1, self.period)
        fmap = []
        for c in self.convs:
            x = c(x)
            fmap.append(x)
        fmap.append(self.post(x))
        return fmap


class MRD(nn.Module):
    def __init__(self, window, bands=BANDS, ch=32):
        super().__init__()
        self.window, self.hop = window, window // 4
        n = window // 2 + 1
        self.bands = [(int(a * n), int(b * n)) for a, b in bands]
        self.register_buffer("win", torch.hann_window(window), persistent=False)

        def stack():
            return nn.ModuleList([
                _c2(2, ch, (3, 9), (1, 1), padding=(1, 4)), _c2(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                _c2(ch, ch, (3, 9), (1, 2), padding=(1, 4)), _c2(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                _c2(ch, ch, (3, 3), (1, 1), padding=(1, 1))])

        self.stacks = nn.ModuleList([stack() for _ in self.bands])
        self.post = _c2(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = torch.stft(x[:, 0].float(), self.window, self.hop, window=self.win, return_complex=True)
            s = torch.view_as_real(s).permute(0, 3, 2, 1)  # [B, 2, frames, freq]
        fmap, outs = [], []
        for (a, b), st in zip(self.bands, self.stacks):
            h = s[..., a:b]
            for c in st:
                h = c(h)
                fmap.append(h)
            outs.append(h)
        fmap.append(self.post(torch.cat(outs, dim=-1)))
        return fmap


class Discriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), ffts=(2048, 1024, 512)):
        super().__init__()
        self.discs = nn.ModuleList([MPD(p) for p in periods] + [MRD(f) for f in ffts])

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = 0.8 * x / (x.abs().amax(dim=-1, keepdim=True) + 1e-9)
        return [d(x) for d in self.discs]


def d_loss(d_fake, d_real, kind="ls"):
    loss = 0.0
    for f, r in zip(d_fake, d_real):
        f, r = f[-1].float(), r[-1].float()
        if kind == "hinge":
            loss = loss + F.relu(1 + f).mean() + F.relu(1 - r).mean()
        else:
            loss = loss + f.pow(2).mean() + (1 - r).pow(2).mean()
    return loss


def g_losses(d_fake, d_real, kind="ls"):
    adv, fm = 0.0, 0.0
    for f in d_fake:
        o = f[-1].float()
        adv = adv + ((-o).mean() if kind == "hinge" else (1 - o).pow(2).mean())
    for f, r in zip(d_fake, d_real):
        for a, b in zip(f[:-1], r[:-1]):
            fm = fm + F.l1_loss(a.float(), b.detach().float())
    return adv, fm
