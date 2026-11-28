"""Shared helpers of the EnCodec probes s09_beam_rvq.py and s11_seam.py (run them from tools/ with
`uv run --extra parity python s11_seam.py ...`).

Nothing here trains anything. It loads the stock EnCodec weights read-only (../weights/encodec_{24,48}khz.*),
builds a fixed evaluation set and provides the metrics:

* perceptual: SNR / log-mel distance / multi-resolution STFT loss against the SOURCE audio

Inputs and outputs:
  PICOTONE_PROBE_WAVDIR   (required) folder of 48 kHz stereo 16-bit WAVs named <split>__<group>__<name>.wav, split test or
                          fit, groups speech, piano, band, electronic, orchestraLR
  PICOTONE_PROBE_THREADS  torch CPU threads (default 4)
  work/encodec_probe/  cache/ (clips and stock activations), out/ (WAVs), results/ (<name>.jsonl rows)
"""
import json
import math
import os
import wave

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly
from transformers import EncodecConfig, EncodecModel

torch.set_grad_enabled(False)
torch.set_flush_denormal(True)   # silent segments otherwise push the LSTM cell states into (very slow) denormals
torch.set_num_threads(int(os.environ.get("PICOTONE_PROBE_THREADS", "4")))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
WORK = os.path.join(ROOT, "work", "encodec_probe")
OUT = os.path.join(WORK, "out")
CACHE = os.path.join(WORK, "cache")
RESULTS = os.path.join(WORK, "results")
CLIP_SECONDS = 10
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def wavdir():
    d = os.environ.get("PICOTONE_PROBE_WAVDIR")
    if not d or not os.path.isdir(d):
        raise SystemExit("set PICOTONE_PROBE_WAVDIR to the folder of test/fit WAVs (see encodec_eval.py)")
    return d


# --------------------------------------------------------------------------- model
def load_model(name):
    """name: '24khz' or '48khz'.  Returns a fresh HF EncodecModel with the stock weights."""
    cfg = json.load(open(os.path.join(WEIGHTS, f"encodec_{name}.json")))
    cfg = {k: v for k, v in cfg.items() if k not in ("architectures", "torch_dtype", "transformers_version", "_name_or_path", "model_type")}
    model = EncodecModel(EncodecConfig(**cfg))
    sd = load_file(os.path.join(WEIGHTS, f"encodec_{name}.safetensors"))
    # HF >= 4.4x uses parametrizations for weight norm; map the legacy names.
    keys = set(model.state_dict().keys())
    new = {}
    for k, v in sd.items():
        if k not in keys:
            k2 = k.replace("weight_g", "parametrizations.weight.original0").replace("weight_v", "parametrizations.weight.original1")
            assert k2 in keys, k
            k = k2
        new[k] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    assert not unexpected and not [m for m in missing if "kernel" not in m and "stride" not in m and "padding" not in m], (missing, unexpected)
    return model.eval()


# --------------------------------------------------------------------------- audio
def _read_wav(path, offset_s, dur_s):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getframerate() == 48000 and w.getnchannels() == 2
        n = w.getnframes()
        start = int(offset_s * 48000)
        want = int(dur_s * 48000)
        if start + want > n:
            start = max(0, n - want)
        w.setpos(start)
        pcm = np.frombuffer(w.readframes(want), "<i2").reshape(-1, 2).T
    x = pcm.astype(np.float32) / 32768.0
    if x.shape[1] < want:
        x = np.pad(x, ((0, 0), (0, want - x.shape[1])))
    return x


def synth_game(seed, kind, seconds=CLIP_SECONDS, rate=48000):
    """Synthetic game audio (chiptune loop / sound effects), stereo 48 kHz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    out = np.zeros((2, n))
    if kind == "chip":
        bpm = 140 + 10 * (seed % 3)
        step = 60.0 / bpm / 4
        scale = np.array([0, 3, 5, 7, 10, 12, 15, 17])
        root = 220.0 * 2 ** ((seed % 5) / 12)
        nsteps = int(seconds / step)
        pat = rng.integers(0, len(scale), 16)
        bass = rng.integers(0, 4, 4)
        for s in range(nsteps):
            a, b = int(s * step * rate), min(n, int((s + 1) * step * rate))
            tt = t[a:b] - t[a]
            f = root * 2 ** (scale[pat[s % 16]] / 12)
            env = np.exp(-tt * 12)
            lead = 0.22 * np.sign(np.sin(2 * np.pi * f * tt)) * env
            fb = root / 2 * 2 ** (scale[bass[(s // 4) % 4]] / 12)
            tri = 0.3 * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * fb * tt))
            drum = np.zeros_like(tt)
            if s % 4 == 0:
                drum = 0.4 * np.sin(2 * np.pi * (120 * np.exp(-tt * 30) + 40) * tt) * np.exp(-tt * 18)
            elif s % 4 == 2:
                drum = 0.25 * rng.standard_normal(len(tt)) * np.exp(-tt * 35)
            pan = 0.5 + 0.3 * np.sin(s * 0.7)
            out[0, a:b] += lead * (1 - pan) * 2 + tri + drum
            out[1, a:b] += lead * pan * 2 + tri + drum
    else:  # sound effects
        pos = 0.2
        while pos < seconds - 1.0:
            k = rng.integers(0, 5)
            d = rng.uniform(0.15, 0.9)
            a, b = int(pos * rate), int(min(seconds, pos + d) * rate)
            tt = t[a:b] - t[a]
            if k == 0:  # laser sweep
                f0, f1 = rng.uniform(1500, 4000), rng.uniform(150, 500)
                ph = 2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * d))
                s = 0.4 * np.sign(np.sin(ph)) * np.exp(-tt * 4)
            elif k == 1:  # explosion: decaying low-passed noise
                nz = rng.standard_normal(len(tt))
                s = 0.6 * np.convolve(nz, np.ones(24) / 24, "same") * 4 * np.exp(-tt * 5)
            elif k == 2:  # coin: two blips
                f = rng.uniform(900, 1400)
                s = 0.35 * np.sin(2 * np.pi * np.where(tt < 0.08, f, f * 1.5) * tt) * np.exp(-tt * 9)
            elif k == 3:  # jump: rising square
                ph = 2 * np.pi * (200 * tt + 900 * tt ** 2)
                s = 0.3 * np.sign(np.sin(ph)) * np.exp(-tt * 6)
            else:  # hit: click + thud
                s = 0.5 * np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 25) + 0.3 * rng.standard_normal(len(tt)) * np.exp(-tt * 80)
            pan = rng.uniform(0.2, 0.8)
            out[0, a:b] += s * (1 - pan) * 2
            out[1, a:b] += s * pan * 2
            pos += d + rng.uniform(0.05, 0.5)
    return np.clip(out, -1, 1).astype(np.float32)


def clip_list(split):
    """Deterministic list of (name, group, offset_s).  'test' = evaluation, 'fit' = data used for closed-form solves."""
    files = sorted(f for f in os.listdir(wavdir()) if f.startswith(split + "__") and f.endswith(".wav") and ".short" not in f)
    groups = {}
    for f in files:
        groups.setdefault(f.split("__")[1], []).append(f)
    want = {"test": {"speech": 3, "piano": 2, "band": 2, "electronic": 2, "orchestraLR": 1},
            "fit": {"speech": 10, "piano": 8, "band": 6, "electronic": 8, "orchestraLR": 4}}[split]
    out = []
    for g, k in want.items():
        fs = groups[g]
        idx = np.linspace(0, len(fs) - 1, min(k, len(fs))).round().astype(int)
        for j, i in enumerate(idx):
            out.append((fs[i], g, 20 + 7 * j))
        # second window from the same files for the fit split (more data, different passage)
        if split == "fit":
            for j, i in enumerate(idx):
                out.append((fs[i], g, 45 + 3 * j))
    return out


def load_clips(split):
    """Returns (names, groups, x) with x float32 [n_clips, 2, 48000*CLIP_SECONDS]."""
    path = os.path.join(CACHE, f"clips_{split}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return list(z["names"]), list(z["groups"]), z["x"]
    names, groups, xs = [], [], []
    for f, g, off in clip_list(split):
        xs.append(_read_wav(os.path.join(wavdir(), f), off, CLIP_SECONDS))
        names.append(f"{f[:60]}@{off}")
        groups.append(g)
    seeds = {"test": [(101, "chip"), (102, "sfx")], "fit": [(1, "chip"), (2, "sfx"), (3, "chip"), (4, "sfx"), (5, "chip"), (6, "sfx")]}[split]
    for seed, kind in seeds:
        xs.append(synth_game(seed, kind))
        names.append(f"synthetic_{kind}_{seed}")
        groups.append("game_" + kind)
    x = np.stack(xs)
    np.savez(path, names=np.array(names, dtype=object), groups=np.array(groups, dtype=object), x=x)
    return names, groups, x


def model_input(name, x48):
    """x48 [n,2,N] @48k stereo -> model input tensor and per-row scale (or None).

    24khz: mono 24 kHz, one row per clip.
    48khz: stereo, cut into non-overlapping 1 s chunks (the real codec overlaps chunks by 1 %; the
    experiments compare per-chunk outputs so the overlap-add is irrelevant), each divided by its RMS.
    """
    if name == "24khz":
        mono = x48.mean(axis=1)
        y = resample_poly(mono, 1, 2, axis=-1).astype(np.float32)
        return torch.from_numpy(y)[:, None, :], None
    n, c, N = x48.shape
    chunks = torch.from_numpy(x48).reshape(n, c, N // 48000, 48000).permute(0, 2, 1, 3).reshape(-1, c, 48000)
    scale = chunks.mean(dim=1, keepdim=True).pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-8
    return chunks / scale, scale


# --------------------------------------------------------------------------- metrics
def snr_db(ref, x):
    ref = ref.double().flatten()
    x = x.double().flatten()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-30)))


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


def _stft_mag(x, n_fft):
    w = torch.hann_window(n_fft)
    return torch.stft(x, n_fft, n_fft // 4, window=w, return_complex=True).abs()


def perceptual(ref, x, sr):
    """ref, x: [rows, T] waveforms (rows = clips x channels).  Returns dict of metrics averaged over rows.

    logmel : mean |20 log10(mel_x) - 20 log10(mel_ref)| in dB (80 mels, floor -80 dB below the clip peak)
    mrstft : mean over FFT sizes 512/1024/2048 of spectral convergence + mean |log mag difference| (natural log)
    snr    : waveform SNR in dB
    """
    n_fft = 1024 if sr <= 24000 else 2048
    fb = _mel_fb(sr, n_fft)
    out = {"snr": [], "logmel": [], "mrstft": []}
    for r, y in zip(ref, x):
        out["snr"].append(snr_db(r, y))
        mr = fb @ _stft_mag(r, n_fft)
        my = fb @ _stft_mag(y, n_fft)
        floor = mr.max() * 1e-4
        d = 20 * (torch.log10(my.clamp_min(floor)) - torch.log10(mr.clamp_min(floor)))
        out["logmel"].append(float(d.abs().mean()))
        tot = 0.0
        for nf in (512, 1024, 2048):
            a, b = _stft_mag(r, nf), _stft_mag(y, nf)
            sc = torch.linalg.norm(a - b) / torch.linalg.norm(a).clamp_min(1e-9)
            lm = (torch.log(a.clamp_min(1e-5)) - torch.log(b.clamp_min(1e-5))).abs().mean()
            tot += float(sc + lm) / 3
        out["mrstft"].append(tot)
    return {k: float(np.mean(v)) for k, v in out.items()}


# --------------------------------------------------------------------------- LSTM
def get_lstm(model, side):
    """Returns (index of the LSTM layer, list of per-layer dicts with w_ih, w_hh, b)."""
    layers = model.encoder.layers if side == "enc" else model.decoder.layers
    idx = [i for i, l in enumerate(layers) if hasattr(l, "lstm")][0]
    lstm = layers[idx].lstm
    ps = []
    for l in range(lstm.num_layers):
        ps.append({
            "w_ih": getattr(lstm, f"weight_ih_l{l}").clone(),
            "w_hh": getattr(lstm, f"weight_hh_l{l}").clone(),
            "b": (getattr(lstm, f"bias_ih_l{l}") + getattr(lstm, f"bias_hh_l{l}")).clone(),
        })
    return idx, ps


def run_layers(layers, x, start=0, end=None):
    for layer in list(layers)[start:end]:
        x = layer(x)
    return x


# --------------------------------------------------------------------------- bench
class Bench:
    """Stock-model references on one split, plus evaluation of modified encoders / decoders.

    For speed the conv stack before each LSTM is cached:
      enc_pre : encoder activations entering the encoder LSTM  [B, 512, T]
      dec_pre : decoder activations entering the decoder LSTM  [B, 512, T]
    """

    def __init__(self, name, split="test", n_q=None):
        self.name = name
        self.sr = 24000 if name == "24khz" else 48000
        self.model = load_model(name)
        self.names, self.groups, x48 = load_clips(split)
        self.x, self.scale = model_input(name, x48)
        m = self.model
        self.enc_idx, _ = get_lstm(m, "enc")
        self.dec_idx, _ = get_lstm(m, "dec")
        self.n_q = n_q or m.quantizer.num_quantizers
        path = os.path.join(CACHE, f"bench_{name}_{split}.pt")
        if os.path.exists(path):
            z = torch.load(path)
            self.enc_pre, self.emb, self.codes, self.dec_pre, self.y = z["enc_pre"], z["emb"], z["codes"], z["dec_pre"], z["y"]
        else:
            self.enc_pre = self._batched(lambda x: run_layers(m.encoder.layers, x, 0, self.enc_idx), self.x)
            self.emb = self._batched(lambda x: run_layers(m.encoder.layers, x, self.enc_idx), self.enc_pre)
            self.codes = self.quantize(self.emb)  # [n_q_total, B, T]
            self.dec_pre = m.decoder.layers[0](self.dequant(self.codes))
            # the fit split only feeds closed-form solves on activations; its decoded audio is never needed
            self.y = self.decode_from_pre(self.dec_pre) if split == "test" else self.x[:, :, :0]
            torch.save({"enc_pre": self.enc_pre, "emb": self.emb, "codes": self.codes, "dec_pre": self.dec_pre, "y": self.y}, path)
        if n_q:
            self.dec_pre = m.decoder.layers[0](self.dequant(self.codes[: self.n_q]))
            if split == "test":
                self.y = self.decode_from_pre(self.dec_pre)
        self.src = self._wave_rows(self.x)
        self.stock = perceptual(self.src, self._wave_rows(self.y), self.sr) if split == "test" else None

    @staticmethod
    def _batched(fn, x, bs=8):
        return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])

    def _wave_rows(self, y):
        """Waveforms as rows of [clips x channels, samples].  For the 48 kHz model the per-chunk RMS scale is
        undone and the 1 s chunks of a clip are joined, so every metric is per 10 s clip in the original level."""
        y = y[..., : self.x.shape[-1]]
        if self.scale is None:
            return y.reshape(-1, y.shape[-1])
        y = y * self.scale
        n = len(self.names)
        c = y.shape[1]
        return y.reshape(n, -1, c, y.shape[-1]).permute(0, 2, 1, 3).reshape(n * c, -1)

    def quantize(self, emb, model=None):
        return (model or self.model).quantizer.encode(emb, None)

    def dequant(self, codes, model=None):
        return (model or self.model).quantizer.decode(codes)

    def decode_from_pre(self, dec_pre, model=None, start=None):
        m = model or self.model
        return self._batched(lambda x: run_layers(m.decoder.layers, x, self.dec_idx if start is None else start), dec_pre)

    # ---- evaluation
    def eval_emb(self, emb, full_chain=True):
        """Score a modified encoder output (pre-quantiser embedding)."""
        codes = self.quantize(emb)
        same = (codes == self.codes)
        r = {"codes": 100 * float(same.float().mean()), "codes_q8": 100 * float(same[:8].float().mean()),
             "codes_q1": 100 * float(same[:1].float().mean()),
             "emb_snr": snr_db(self.emb, emb)}
        if full_chain:
            y = self.decode_from_pre(self.model.decoder.layers[0](self.dequant(codes[: self.n_q])))
            r["snr_orig"] = snr_db(self._wave_rows(self.y), self._wave_rows(y))
            r.update(perceptual(self.src, self._wave_rows(y), self.sr))
        return r

    def eval_wave(self, y):
        """Score a modified decoder output (decoded from the stock codes)."""
        r = {"snr_orig": snr_db(self._wave_rows(self.y), self._wave_rows(y))}
        r.update(perceptual(self.src, self._wave_rows(y), self.sr))
        return r


def fmt(r):
    keys = ["codes", "codes_q8", "codes_q1", "emb_snr", "snr_orig", "snr", "logmel", "mrstft"]
    return " ".join(f"{k}={r[k]:.3f}" for k in keys if k in r)


class Log:
    """Writes result rows to work/encodec_probe/results/<name>.jsonl and prints them."""

    def __init__(self, name):
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, name + ".jsonl")
        open(self.path, "w").close()

    def __call__(self, label, r=None, **extra):
        row = {"label": label, **(r or {}), **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        ex = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in extra.items())
        print(f"{label:58s} {fmt(r or {})} {ex}", flush=True)
