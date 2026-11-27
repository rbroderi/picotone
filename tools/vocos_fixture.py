# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity fixtures for the Rust Vocos decoder (`src/vocos.rs`), from distill/vocos_student.py at random init.
Run it in distill/'s environment (CPU):

    cd distill
    uv run --extra cpu python ../tools/vocos_fixture.py --dac export/weights_44khz_8kbps_0.0.1.pth \
        --ext X:/audio/distill/weights/q16_codebooks.pt --codes <codes.npy> --out ../tests/data/vocos_tiny --dim 32 --blocks 8 --mlp 96 --frames 80

1. Builds VocosDecoder(dim, blocks, mlp, hop 512) and re-initialises it at realistic scales (`realistic_init`):
   the stock init (std 0.02, zero biases) gives log-magnitudes near 0 and phases near 0, which would test
   little of the head. Here log-magnitudes spread over about -9..5 (some bins reach the clamp at 100) and
   phases over several radians.
2. Saves it as a train.py checkpoint and runs distill/export.py on it, the path a trained run takes.
3. Converts that export with tools/export_vocos.py: <out>.safetensors and <out>.json.
4. Decodes the first --frames frames of --codes (`[frames, 16]` u16, e.g. from `picotone dev codes`) at each
   --tiers count through the original path (DAC quantiser z, then the unfolded decoder, fp32), and writes
   <out>.codes.npy and <out>.q<n>.f32 (raw little-endian f32 samples, 512 per frame).
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DISTILL = os.path.abspath(os.path.join(HERE, "..", "distill"))
sys.path.insert(0, DISTILL)
sys.path.insert(0, HERE)

from export_vocos import convert, z_of  # noqa: E402
from vocos_student import VocosDecoder  # noqa: E402


@torch.no_grad()
def realistic_init(G, seed):
    g = torch.Generator().manual_seed(seed)

    def randn(shape):
        return torch.randn(shape, generator=g)

    for name, p in G.named_parameters():
        if name.endswith("gamma"):
            p.copy_(0.1 + 0.4 * torch.rand(p.shape, generator=g))
        elif "norm" in name:
            p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.2 * randn(p.shape))
        elif name.endswith("bias"):
            p.copy_(0.05 * randn(p.shape))
        else:
            p.copy_(randn(p.shape) / p[0].numel() ** 0.5)
    n = G.n_fft // 2 + 1
    G.head.weight[:n] *= 2.0  # log-magnitude: std about 2 around -3
    G.head.bias[:n] = -3.0 + 0.3 * randn(n)
    G.head.weight[n:] *= 3.0  # phase: std about 3 rad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dac", required=True, help="DAC 44 kHz weights (weights_44khz_8kbps_0.0.1.pth)")
    ap.add_argument("--ext", required=True, help="codebooks 10-16 (q16_codebooks.pt)")
    ap.add_argument("--codes", required=True, help="codes [frames, 16] (.npy, u16)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--mlp", type=int, default=1536)
    ap.add_argument("--frames", type=int, default=0, help="frames of --codes to use (0: all)")
    ap.add_argument("--tiers", default="16,9,5,1")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    G = VocosDecoder(a.dim, a.blocks, a.mlp, 512)
    realistic_init(G, a.seed)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "ckpt.pt")
        torch.save({"step": 0, "G": G.state_dict(), "width": 512, "arch": G.cfg,
                    "args": {"n_q": a.tiers, "licences": "none (random-init test fixture)"}}, ckpt)
        # The export a trained run gets, through distill/export.py itself.
        subprocess.run([sys.executable, os.path.join(DISTILL, "export.py"), "--ckpt", ckpt,
                        "--out", os.path.join(tmp, "vocos")], check=True, cwd=DISTILL)
        G, (books, proj, bias) = convert(os.path.join(tmp, "vocos.pt"), a.dac, a.ext, a.out)
    codes = np.load(a.codes).astype(np.int64)
    assert codes.ndim == 2 and codes.shape[1] == 16, codes.shape
    if a.frames:
        codes = codes[: a.frames]
    np.save(a.out + ".codes.npy", codes.astype("<u2"))
    c = torch.from_numpy(codes.T.copy())
    with torch.no_grad():
        for n_q in (int(t) for t in a.tiers.split(",")):
            y = G(z_of(books, proj, bias, c, n_q)[None])[0, 0].numpy().astype("<f4")
            y.tofile(f"{a.out}.q{n_q}.f32")
            print(f"q{n_q}: {codes.shape[0]} frames -> {y.size} samples, peak {np.abs(y).max():.4f}, "
                  f"rms {np.sqrt((y.astype(np.float64) ** 2).mean()):.4f}")


if __name__ == "__main__":
    main()
