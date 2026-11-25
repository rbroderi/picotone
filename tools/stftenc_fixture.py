# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity fixture for the Rust STFT encoder (`src/stftenc.rs`), from distill/encoder.py's STFTEncoder at random
init. Run it in distill/'s environment (CPU):

    cd distill
    uv run --extra cpu python ../tools/stftenc_fixture.py --out ../tests/data/stftenc_tiny

1. Builds STFTEncoder(dim, blocks, mlp, ffts (2048, 512), power 0.3, embed k1, latent) and re-initialises it at
   realistic scales (`realistic_init`; the stock init, std 0.02 and zero biases, would test little).
2. Writes it through tools/export_stftenc.py: <out>.safetensors and <out>.json.
3. Synthesises --frames frames of test audio (tones, vibrato, noise, a click, digital silence, a very quiet
   passage, a fade to zero) and writes it as <out>.pcm (little-endian int16, mono 44.1 kHz).
4. Encodes it (int16 / 32768, fp32) and writes z as <out>.z.f32 (little-endian f32, [frames, latent]
   time-major), and a float64 run of the same model (export_stftenc.forward64) as <out>.z64.f32, the precision
   floor of any fp32 implementation; prints the SNR of one against the other.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from export_stftenc import forward64, write  # noqa: E402  (puts distill/ on the path)
from encoder import STFTEncoder  # noqa: E402

SR = 44100


@torch.no_grad()
def realistic_init(enc, seed):
    g = torch.Generator().manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=g)

    for name, p in enc.named_parameters():
        if name.endswith("gamma"):
            p.copy_(0.1 + 0.4 * torch.rand(p.shape, generator=g))
        elif "norm" in name:
            p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.2 * randn(p.shape))
        elif name.endswith("bias"):
            p.copy_(0.05 * randn(p.shape))
        else:
            p.copy_(randn(p.shape) / p[0].numel() ** 0.5)


def synth(frames, seed):
    n = frames * 512
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    f0 = 220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    for h in range(1, 12):
        x += 0.25 / h * np.sin(h * phase + h)
    x += 0.05 * np.sin(2 * np.pi * 3150 * t)
    burst = (t > 0.15) & (t < 0.25)
    x[burst] += 0.3 * rng.standard_normal(burst.sum())
    x[int(0.3 * SR)] += 0.9
    silent = (t > 0.36) & (t < 0.40)
    x[silent] = 0.0
    quiet = (t >= 0.40) & (t < 0.46)
    x[quiet] = 3e-4 * np.sin(2 * np.pi * 1000 * t[quiet]) + 1e-4 * rng.standard_normal(quiet.sum())
    tail = t >= 0.46
    x[tail] *= np.linspace(1, 0, tail.sum())
    x[-300:] = 0.0
    return np.clip(np.round(x * 16384), -32768, 32767).astype("<i2")


def snr(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return 10 * np.log10((want ** 2).sum() / max(((got - want) ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--mlp", type=int, default=24)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    enc = STFTEncoder(a.dim, a.blocks, a.mlp, (2048, 512), 0.3, 1, a.latent)
    realistic_init(enc, a.seed)
    enc.eval()
    write(enc, a.out, {"step": 0, "weights": "random init (tools/stftenc_fixture.py)",
                       "licences": "none (random-init test fixture)"})
    pcm = synth(a.frames, a.seed)
    pcm.tofile(a.out + ".pcm")
    x = torch.from_numpy(pcm.astype(np.float32) / 32768)[None, None]
    with torch.no_grad():
        z = enc(x)[0].T.contiguous().numpy().astype("<f4")  # [frames, latent]
        z64 = forward64(enc, x)[0].T.numpy()
    z.tofile(a.out + ".z.f32")
    z64.astype("<f4").tofile(a.out + ".z64.f32")
    print(f"{a.frames} frames -> z {z.shape}, rms {np.sqrt((z.astype(np.float64) ** 2).mean()):.4f}; "
          f"fp32 vs fp64 SNR {snr(z, z64):.1f} dB")


if __name__ == "__main__":
    main()
