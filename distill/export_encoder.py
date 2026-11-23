"""Export DAC 44 kHz's encoder and quantiser for picotone's Rust encoder (milestone M6 in INTEGRATION.md).

    uv run --extra cpu python export_encoder.py --dac export/weights_44khz_8kbps_0.0.1.pth \
        --out export/rust/dac44_encoder

The HQ codec's codes are DAC's own codes: the student only replaces DAC's decoder, so the encoder and the
quantiser are DAC's, unchanged (MIT, Copyright (c) 2023-present, Descript). Writes <out>.safetensors and
<out>.json. Weight norm is removed with PyTorch's own remove_weight_norm, so the plain weights are exactly
what DAC computes.

Tensor names (all float32; Snake alphas as [C]; widths C = 64, 128, 256, 512 for blocks b = 0..3):
  input.weight / .bias     [64, 1, 7] / [64]           k7, padding 3
  block{b}.res{r}.snake1.alpha / .conv1.weight [C, C, 7] (dilation 1/3/9, padding 3d) / .conv1.bias
  block{b}.res{r}.snake2.alpha / .conv2.weight [C, C, 1] / .conv2.bias
  block{b}.snake.alpha / block{b}.down.weight [2C, C, 2s] / .down.bias   stride s = 2, 4, 8, 8, padding s/2
  output.snake.alpha / output.conv.weight [1024, 1024, 3] / .bias         padding 1
  quant{i}.in_proj.weight [8, 1024, 1] / .bias [8]                        i = 0..8
  quant{i}.codebook [1024, 8]          the raw entries (the residual subtracts out_proj of these)
  quant{i}.codebook_unit [1024, 8]     F.normalize of the entries (the nearest-entry search uses these)
  quant{i}.out_proj.weight [1024, 8, 1] / .bias [1024]
"""
import argparse
import json
import os
import warnings

import torch
import torch.nn.functional as F

from export_rust import N_CB, sha, write_safetensors

STRIDES = (2, 4, 8, 8)


def plain(module):
    """Removes weight norm from every conv in module, in place."""
    for m in module.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)


def residual(out, prefix, unit):
    b = unit.block
    out[prefix + "snake1.alpha"] = b[0].alpha.flatten()
    out[prefix + "conv1.weight"], out[prefix + "conv1.bias"] = b[1].weight, b[1].bias
    out[prefix + "snake2.alpha"] = b[2].alpha.flatten()
    out[prefix + "conv2.weight"], out[prefix + "conv2.bias"] = b[3].weight, b[3].bias


def tensors(model):
    enc = model.encoder.block
    out = {"input.weight": enc[0].weight, "input.bias": enc[0].bias}
    for b, stride in enumerate(STRIDES):
        blk = enc[b + 1].block
        for r in range(3):
            residual(out, f"block{b}.res{r}.", blk[r])
        out[f"block{b}.snake.alpha"] = blk[3].alpha.flatten()
        assert blk[4].stride[0] == stride and blk[4].padding[0] == (stride + 1) // 2
        out[f"block{b}.down.weight"], out[f"block{b}.down.bias"] = blk[4].weight, blk[4].bias
    out["output.snake.alpha"] = enc[5].alpha.flatten()
    out["output.conv.weight"], out["output.conv.bias"] = enc[6].weight, enc[6].bias
    for i, q in enumerate(model.quantizer.quantizers):
        p = f"quant{i}."
        out[p + "in_proj.weight"], out[p + "in_proj.bias"] = q.in_proj.weight, q.in_proj.bias
        out[p + "codebook"] = q.codebook.weight
        out[p + "codebook_unit"] = F.normalize(q.codebook.weight)
        out[p + "out_proj.weight"], out[p + "out_proj.bias"] = q.out_proj.weight, q.out_proj.bias
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dac", required=True, help="DAC 44 kHz weights (weights_44khz_8kbps_0.0.1.pth)")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import dac

    model = dac.DAC.load(a.dac).eval()
    plain(model)
    assert len(model.quantizer.quantizers) == N_CB
    with torch.no_grad():
        t = tensors(model)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_safetensors(a.out + ".safetensors", t, {"format": "picotone-dac44-encoder"})
    n_params = sum(v.numel() for v in t.values())
    cfg = {
        "model": "dac44-encoder", "sample_rate": 44100, "hop": 512, "strides": list(STRIDES),
        "encoder_dim": 64, "latent_dim": 1024, "codebooks": N_CB, "codebook_size": 1024, "codebook_dim": 8,
        "params": n_params, "licence": "MIT (descript-audio-codec, Copyright (c) 2023-present, Descript)",
        "source_sha256": {"dac": sha(a.dac)},
    }
    json.dump(cfg, open(a.out + ".json", "w"), indent=1)
    size = os.path.getsize(a.out + ".safetensors")
    print(f"wrote {a.out}.safetensors ({n_params / 1e6:.2f}M params, {size / 1e6:.1f} MB), "
          f"sha256 {sha(a.out + '.safetensors')[:16]}")


if __name__ == "__main__":
    main()
