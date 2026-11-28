"""Export q16's extra codebooks (10-16) as a plain tensor dict in DAC's naming (owner request 2026-11-26; only reads
checkpoints; refuses to overwrite). Keys quantizer.quantizers.{9..15}.{in_proj.weight, in_proj.bias, codebook.weight,
out_proj.weight, out_proj.bias}, fp32, the same shapes as DAC's quantizer.quantizers.{0..8} with weight norm removed
(the extra codebooks are plain convs, so there is no weight norm to fold).
  .venv/bin/python export_q16_codebooks.py --ckpt runs/q16/ckpt_0030000.pt \
      --out weights/q16_codebooks.pt
"""
import argparse
import os

import torch

PARTS = ("in_proj.weight", "in_proj.bias", "codebook.weight", "out_proj.weight", "out_proj.bias")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights", default="G_ema")
    p.add_argument("--n-base", type=int, default=9)
    p.add_argument("--out", required=True)
    c = p.parse_args()
    assert not os.path.exists(c.out), f"{c.out} exists; not overwriting"
    s = torch.load(c.ckpt, map_location="cpu", weights_only=False)
    g = s[c.weights]
    n_ext = s["args"]["enc_ext_n"]
    ext = {k for k in g if k.startswith("rvq.ext.")}
    assert len(ext) == 5 * n_ext, sorted(ext)
    out = {}
    for j in range(n_ext):
        for part in PARTS:
            out[f"quantizer.quantizers.{c.n_base + j}.{part}"] = g[f"rvq.ext.{j}.{part}"].float().contiguous().clone()
    shapes = {"in_proj.weight": (8, 1024, 1), "in_proj.bias": (8,), "codebook.weight": (1024, 8),
              "out_proj.weight": (1024, 8, 1), "out_proj.bias": (1024,)}
    for k, v in out.items():
        assert tuple(v.shape) == shapes[k.split(".", 3)[3]] and v.dtype == torch.float32, (k, v.shape, v.dtype)
    torch.save(out, c.out)
    back = torch.load(c.out, map_location="cpu", weights_only=True)
    assert set(back) == set(out) and all(torch.equal(back[k], out[k]) for k in out)
    for j in range(n_ext):
        assert all(torch.equal(back[f"quantizer.quantizers.{c.n_base + j}.{part}"], g[f"rvq.ext.{j}.{part}"])
                   for part in PARTS)
    print(f"{c.out}: {len(out)} tensors (codebooks {c.n_base}..{c.n_base + n_ext - 1}, 0-indexed) from {c.ckpt} "
          f"step {s['step']} {c.weights}; equal to the checkpoint's rvq.ext.* tensor for tensor")


if __name__ == "__main__":
    main()
