# SPDX-License-Identifier: GPL-3.0-or-later
"""Test fixture for the Rust low-tier post-filter (`src/vocos.rs`): the tiny Vocos fixture plus a random
post-filter, at scales where the gain spreads over most of its range (e^-1 .. e^1).

    uv run --no-project --with numpy --with safetensors python tools/postfilter_fixture.py \
        --vocos tests/data/vocos_tiny --out tests/data/vocos_tiny_pf

Reads <vocos>.safetensors and writes <out>.safetensors with `pf.*` tensors in tools/export_postfilter.py's
layout (dim 16, 4 blocks, mlp 48), so the decoder's receptive field grows by 12 frames.
"""
import argparse

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

DIM, BLOCKS, MLP, BINS, KERNEL = 16, 4, 48, 1025, 7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocos", required=True, help="the decoder fixture, path without extension")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--seed", type=int, default=4321)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    with safe_open(a.vocos + ".safetensors", "np") as f:
        meta = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}
    decoder_dim = t["embed.bias"].shape[0]

    def lin(out, inp, gain=1.0):
        return (gain * rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)

    def vec(n, mean=0.0, std=0.05):
        return (mean + std * rng.standard_normal(n)).astype(np.float32)

    pf = {
        "pf.embed.weight": lin(DIM, 2 * BINS, 4.0), "pf.embed.bias": vec(DIM),
        "pf.h_embed.weight": lin(DIM, decoder_dim), "pf.h_embed.bias": vec(DIM),
        "pf.tier_emb.weight": vec(6 * DIM, std=0.5).reshape(6, DIM),
        "pf.norm.weight": vec(DIM, 1.0, 0.2), "pf.norm.bias": vec(DIM, std=0.2),
        "pf.final_norm.weight": vec(DIM, 1.0, 0.2), "pf.final_norm.bias": vec(DIM, std=0.2),
        "pf.head.weight": lin(BINS, DIM, 1.5), "pf.head.bias": vec(BINS, std=0.3),
    }
    for i in range(BLOCKS):
        p = f"pf.block{i}."
        pf |= {
            p + "dwconv.weight": lin(DIM, KERNEL), p + "dwconv.bias": vec(DIM),
            p + "norm.weight": vec(DIM, 1.0, 0.2), p + "norm.bias": vec(DIM, std=0.2),
            p + "pw1.weight": lin(MLP, DIM), p + "pw1.bias": vec(MLP),
            p + "pw2.weight": lin(DIM, MLP), p + "pw2.bias": vec(DIM),
            p + "gamma": (0.1 + 0.4 * rng.random(DIM)).astype(np.float32),
        }
    save_file(t | pf, a.out + ".safetensors", metadata=meta)
    print(f"wrote {a.out}.safetensors (+{sum(v.size for v in pf.values())} post-filter params)")


if __name__ == "__main__":
    main()
