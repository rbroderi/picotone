"""Held-out metrics, the same definitions as the earlier picotone/DAC comparison (codecs/metrics.py):
log-mel distance in dB (80 mels, 2048 FFT, floor 80 dB under the clip peak), MR-STFT (spectral convergence
+ log magnitude at 512/1024/2048), SNR and SI-SDR. Inputs are numpy [rows, T] at 44.1 kHz; rows are mono
channels and the result is the mean over rows."""
import numpy as np
import torch

SR = 44100
_MEL = {}


def _mel_fb(sr, n_fft, n_mels=80):
    key = (sr, n_fft, n_mels)
    if key not in _MEL:
        def hz2mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel2hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        pts = mel2hz(np.linspace(hz2mel(30), hz2mel(sr / 2), n_mels + 2))
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        fb = np.zeros((n_mels, len(freqs)))
        for i in range(n_mels):
            lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
            fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        _MEL[key] = torch.from_numpy(fb).float()
    return _MEL[key]


def _mag(x, n_fft):
    return torch.stft(x, n_fft, n_fft // 4, window=torch.hann_window(n_fft), return_complex=True).abs()


def snr_db(r, y):
    r, y = r.double(), y.double()
    return float(10 * torch.log10(r.pow(2).sum() / (r - y).pow(2).sum().clamp_min(1e-30)))


def si_sdr_db(r, y):
    r, y = r.double() - r.double().mean(), y.double() - y.double().mean()
    a = (r * y).sum() / r.pow(2).sum().clamp_min(1e-30)
    t = a * r
    return float(10 * torch.log10(t.pow(2).sum() / (y - t).pow(2).sum().clamp_min(1e-30)))


def score(ref, x, sr=SR):
    ref = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32))
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    n = min(ref.shape[-1], x.shape[-1])
    ref, x = ref[..., :n], x[..., :n]
    fb = _mel_fb(sr, 2048)
    out = {"logmel": [], "mrstft": [], "snr": [], "sisdr": [], "dc_mfs": []}
    for r, y in zip(ref, x):
        if float(r.abs().max()) < 1e-4:  # digital silence: nothing to score
            continue
        out["snr"].append(snr_db(r, y))
        # DC error in thousandths of full scale: SI-SDR removes the means, SNR doesn't, log-mel barely sees it
        out["dc_mfs"].append(1000 * abs(float(y.double().mean() - r.double().mean())))
        out["sisdr"].append(si_sdr_db(r, y))
        mr, my = fb @ _mag(r, 2048), fb @ _mag(y, 2048)
        floor = mr.max() * 1e-4
        out["logmel"].append(float((20 * (torch.log10(my.clamp_min(floor)) - torch.log10(mr.clamp_min(floor))))
                                   .abs().mean()))
        tot = 0.0
        for nf in (512, 1024, 2048):
            a, b = _mag(r, nf), _mag(y, nf)
            sc = torch.linalg.norm(a - b) / torch.linalg.norm(a).clamp_min(1e-9)
            lm = (torch.log(a.clamp_min(1e-5)) - torch.log(b.clamp_min(1e-5))).abs().mean()
            tot += float(sc + lm) / 3
        out["mrstft"].append(tot)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def mean_scores(rows):
    """rows: list of dicts from score() -> mean of each key."""
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
