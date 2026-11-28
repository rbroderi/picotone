# SPDX-License-Identifier: GPL-3.0-or-later
"""Add the low-tier post-filter to picotone's HQ decoder weights (`src/vocos.rs`, tensors `pf.*`).

    uv run --no-project --with numpy --with safetensors python tools/export_postfilter.py \
        --pf train/hq/pf_lowtier/export/pf_g7500.safetensors --vocos weights/dac44_vocos

Reads <pf>.safetensors (the post-filter's state dict, `LowTierPF`) and <pf>.json (its export config), and
rewrites <vocos>.safetensors and <vocos>.json in place: any `pf.*` tensors already there are replaced. The
post-filter's tensors are copied unchanged (float32), renamed and with the k1/k7 conv weights squeezed:

  pf.embed.weight / .bias      [dim, 2 bins] / [dim]       features: Re, then Im, of S (|S| + 1e-5)^(0.3 - 1)
  pf.h_embed.weight / .bias    [dim, decoder dim] / [dim]  the decoder's final_norm output
  pf.tier_emb.weight           [6, dim]                    added per tier (rows 1-5)
  pf.norm.weight / .bias       [dim]                       LayerNorm (eps 1e-6)
  pf.block{i}.*                as the decoder's blocks     ConvNeXt, depthwise k7
  pf.final_norm.weight / .bias [dim]
  pf.head.weight / .bias       [bins, dim] / [bins]        gain = exp(tanh(o)) per bin, phase kept
"""
import argparse
import hashlib
import json

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def renamed(name):
    if name.startswith("blocks."):
        _, i, rest = name.split(".", 2)
        name = f"block{i}.{rest}"
    return "pf." + name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pf", required=True, help="the post-filter's .safetensors (its .json beside it)")
    ap.add_argument("--vocos", required=True, help="the decoder's weights, path without extension")
    a = ap.parse_args()
    pf_stem = a.pf.removesuffix(".safetensors")
    pf_cfg = json.load(open(pf_stem + ".json"))
    if pf_cfg["mask"]["gain_bound"] != 1.0 or pf_cfg["features"]["compression"] != 0.3:
        raise SystemExit("src/vocos.rs is built for gain bound 1.0 and compression 0.3")
    if pf_cfg["tiers"]["max_q"] != 5:
        raise SystemExit("src/vocos.rs runs the post-filter at tiers 1-5")

    with safe_open(a.vocos + ".safetensors", "np") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("pf.")}
    with safe_open(a.pf, "np") as f:
        for k in f.keys():
            t = f.get_tensor(k).astype(np.float32)
            if k.endswith("dwconv.weight"):
                t = t[:, 0, :]
            elif k in ("embed.weight", "h_embed.weight"):
                t = t[:, :, 0]
            tensors[renamed(k)] = np.ascontiguousarray(t)
    save_file(tensors, a.vocos + ".safetensors", metadata=meta)

    cfg = json.load(open(a.vocos + ".json"))
    pf_params = sum(int(t.size) for k, t in tensors.items() if k.startswith("pf."))
    blocks = sum(1 for k in tensors if k.startswith("pf.block") and k.endswith(".gamma"))
    cfg["params"] = sum(int(t.size) for t in tensors.values())
    cfg["receptive_frames"] = 3 * (1 + cfg["arch"]["blocks"]) + 3 * blocks + 2
    cfg["post_filter"] = {
        "tiers": "1-5 (the frame's codebook count); bypassed above",
        "dim": int(tensors["pf.embed.bias"].shape[0]),
        "blocks": blocks,
        "mlp": int(tensors["pf.block0.pw1.bias"].shape[0]),
        "params": pf_params,
        "step": pf_cfg.get("step"),
        "weights": pf_cfg.get("weights"),
        "source_sha256": {"checkpoint": pf_cfg.get("source_sha256"), "export": sha(a.pf)},
    }
    json.dump(cfg, open(a.vocos + ".json", "w"), indent=1)
    print(f"wrote {a.vocos}.safetensors (+{pf_params / 1e6:.2f}M post-filter params, {blocks} blocks), "
          f"receptive field {cfg['receptive_frames']} frames, sha256 {sha(a.vocos + '.safetensors')[:16]}")


if __name__ == "__main__":
    main()
