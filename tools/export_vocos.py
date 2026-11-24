# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert a Vocos-style HQ decoder exported by distill/export.py into picotone's Rust format
(`src/vocos.rs`, decoder name `dac44_vocos`). Run it in distill/'s environment:

    cd distill
    uv run --extra cpu python ../tools/export_vocos.py --vocos X:/audio/distill/dac44_vocos_armA.pt \
        --dac export/weights_44khz_8kbps_0.0.1.pth --out ../weights/dac44_vocos

Reads <vocos>.pt (export.py's state dict) and <vocos>.json (its config; `arch` must be a vocos config), and
writes <out>.safetensors and <out>.json. It copies <vocos>.NOTICE.txt and <vocos>.credits.jsonl to
<out>.NOTICE.txt and <out>.credits.jsonl when they exist.

As in distill/export_rust.py, the 9 DAC codebook projections are folded into the embed conv
(k7, 1024 -> dim becomes k7, 81 -> dim: per codebook, its 8 embedding values and a "present" 1), in float64,
rounded to float32 once. Everything else is copied unchanged (float32):

  codebooks                  [9, 1024, 8]
  embed.weight / .bias       [dim, 81, 7] / [dim]
  norm.weight / .bias        [dim]                    LayerNorm after the embed (eps 1e-6)
  block{i}.dwconv.weight     [dim, 7]                 depthwise k7 conv (the [dim, 1, 7] weight, squeezed)
  block{i}.dwconv.bias       [dim]
  block{i}.norm.weight/.bias [dim]
  block{i}.pw1.weight/.bias  [mlp, dim] / [mlp]       Linear, then exact (erf) GELU
  block{i}.pw2.weight/.bias  [dim, mlp] / [dim]
  block{i}.gamma             [dim]                    layer scale
  final_norm.weight/.bias    [dim]
  head.weight / .bias        [k (n_fft + 2), dim] / [k (n_fft + 2)]
                             per STFT frame: n_fft/2 + 1 log-magnitudes, then n_fft/2 + 1 phases

Check: random codes at several tiers, decoded through the original path (DAC quantiser z, then the unfolded
VocosDecoder) and the folded one; the largest difference relative to the peak must be under 1e-5.
"""
import argparse
import copy
import json
import os
import shutil
import sys

import torch
from torch import nn

DISTILL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distill")
sys.path.insert(0, os.path.abspath(DISTILL))

from export_rust import CB_SIZE, CB_DIM, IN_PER_CB, N_CB, fold_input, folded_input, load_quantiser, sha, write_safetensors  # noqa: E402
from vocos_student import FRAME, VocosDecoder  # noqa: E402


def build(cfg_arch, sd):
    """The decoder export.py wrote, as a module."""
    assert cfg_arch.get("arch") == "vocos", f"not a vocos decoder: {cfg_arch}"
    G = VocosDecoder(cfg_arch["dim"], cfg_arch["blocks"], cfg_arch["mlp"], cfg_arch["hop"])
    G.load_state_dict(sd)
    return G.eval()


def z_of(books, proj, bias, codes, n_q):
    """DAC's quantised latent for codes [9, F] at tier n_q: [1024, F]."""
    return sum(proj[i] @ books[i][codes[i]].T + bias[i][:, None] for i in range(n_q))


def folded_module(G, fw, fb):
    """A copy of G whose embed reads the folded 81-value input."""
    H = copy.deepcopy(G)
    H.embed = nn.Conv1d(N_CB * IN_PER_CB, fw.shape[0], 7, padding=3)
    with torch.no_grad():
        H.embed.weight.copy_(fw)
        H.embed.bias.copy_(fb)
    return H.eval()


@torch.no_grad()
def check(G, books, proj, bias, fw, fb, frames=60, seed=0):
    """Largest |original - folded| relative to the original's peak, per tier."""
    H = folded_module(G, fw, fb)
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, CB_SIZE, (N_CB, frames), generator=g)
    worst = {}
    for n_q in (1, 4, 5, 9):
        y0 = G(z_of(books, proj, bias, codes, n_q)[None])
        y1 = H(folded_input(books, codes, n_q)[None])
        worst[n_q] = float((y0 - y1).abs().max() / y0.abs().max().clamp(min=1e-12))
    return worst


def tensors_of(sd, fw, fb, books, blocks):
    out = {"codebooks": books, "embed.weight": fw, "embed.bias": fb,
           "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"]}
    for i in range(blocks):
        p, o = f"blocks.{i}.", f"block{i}."
        out[o + "dwconv.weight"] = sd[p + "dwconv.weight"][:, 0, :]
        out[o + "dwconv.bias"] = sd[p + "dwconv.bias"]
        for name in ("norm", "pw1", "pw2"):
            out[o + name + ".weight"] = sd[p + name + ".weight"]
            out[o + name + ".bias"] = sd[p + name + ".bias"]
        out[o + "gamma"] = sd[p + "gamma"]
    for name in ("final_norm", "head"):
        out[name + ".weight"] = sd[name + ".weight"]
        out[name + ".bias"] = sd[name + ".bias"]
    return out


def convert(vocos, dac, out):
    stem = os.path.splitext(vocos)[0]
    cfg_in = json.load(open(stem + ".json"))
    arch = cfg_in["arch"]
    sd = torch.load(vocos, map_location="cpu", weights_only=True)
    G = build(arch, sd)
    books, proj, bias = load_quantiser(dac)
    fw, fb = fold_input(sd["embed.weight"], sd["embed.bias"], proj, bias)
    worst = check(G, books, proj, bias, fw, fb)
    print("fold check, max |orig - folded| / peak per tier:", {k: f"{v:.2e}" for k, v in worst.items()})
    if max(worst.values()) > 1e-5:
        raise SystemExit("fold check failed")
    tensors = tensors_of(sd, fw, fb, books, arch["blocks"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_safetensors(out + ".safetensors", tensors, {"format": "picotone-dac44-vocos"})
    n_params = sum(t.numel() for t in tensors.values())
    cfg = {
        "model": "dac44-vocos",
        "arch": {"arch": "vocos", "dim": arch["dim"], "blocks": arch["blocks"], "mlp": arch["mlp"],
                 "hop": arch["hop"], "n_fft": G.n_fft},
        "sample_rate": 44100, "frame": FRAME,
        "codebooks": N_CB, "codebook_size": CB_SIZE, "codebook_dim": CB_DIM, "input_per_codebook": IN_PER_CB,
        "receptive_frames": G.receptive_field(), "params": n_params,
        "step": cfg_in.get("step"), "licences": cfg_in.get("licences"), "teacher": cfg_in.get("teacher"),
        "source_sha256": {"vocos": sha(vocos), "dac": sha(dac)},
        "fold_max_rel_dev": max(worst.values()),
    }
    json.dump(cfg, open(out + ".json", "w"), indent=1)
    for ext in (".NOTICE.txt", ".credits.jsonl"):
        if os.path.exists(stem + ext):
            shutil.copyfile(stem + ext, out + ext)
    size = os.path.getsize(out + ".safetensors")
    print(f"wrote {out}.safetensors ({n_params / 1e6:.2f}M params, {size / 1e6:.1f} MB) and {out}.json, "
          f"sha256 {sha(out + '.safetensors')[:16]}")
    return G, (books, proj, bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocos", required=True, help="export.py's .pt of a --arch vocos run (its .json beside it)")
    ap.add_argument("--dac", required=True, help="DAC 44 kHz weights (weights_44khz_8kbps_0.0.1.pth)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    convert(a.vocos, a.dac, a.out)


if __name__ == "__main__":
    main()
