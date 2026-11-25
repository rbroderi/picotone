# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert a trained HQ encoder (distill/encoder.py STFTEncoder, a train.py --arch encoder checkpoint) into
picotone's Rust format (`src/stftenc.rs`, encoder name `dac44_stftenc`). Run it in distill/'s environment:

    cd distill
    uv run --extra cpu python ../tools/export_stftenc.py --ckpt X:/audio/distill/runs/encB_arm4/ckpt_end_20000.pt \
        --out ../weights/dac44_stftenc

The encoder is taken from the checkpoint's EMA weights (`G_ema`, keys `enc.*`; `--no-ema` for `G`) and its
`arch` config; the quantiser (`rvq.*`) and decoder (`dec.*`) are left out. The checkpoint is only read.
Writes <out>.safetensors and <out>.json (config, source sha256, step, parameter count), all float32:

  window{i}                  [n_fft_i]                torch.hann_window(n_fft_i), the feature resolutions in order
  power                      [1]                      the compression power c (S |S|^(c - 1))
  embed.weight / .bias       [dim, channels] / [dim]  the k1 embed conv (the [dim, channels, 1] weight, squeezed)
  norm.weight / .bias        [dim]                    LayerNorm after the embed (eps 1e-6)
  block{i}.dwconv.weight     [dim, 7]                 depthwise k7 conv (the [dim, 1, 7] weight, squeezed)
  block{i}.dwconv.bias       [dim]
  block{i}.norm.weight/.bias [dim]
  block{i}.pw1.weight/.bias  [mlp, dim] / [mlp]       Linear, then exact (erf) GELU
  block{i}.pw2.weight/.bias  [dim, mlp] / [dim]
  block{i}.gamma             [dim]                    layer scale
  final_norm.weight/.bias    [dim]
  out.weight / .bias         [latent, dim] / [latent]

distill/ is found next to this file, or at $PICOTONE_DISTILL.
"""
import argparse
import json
import os
import sys

import torch

DISTILL = os.environ.get("PICOTONE_DISTILL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                             "distill")
sys.path.insert(0, os.path.abspath(DISTILL))

from encoder import STFTEncoder  # noqa: E402
from export_rust import sha, write_safetensors  # noqa: E402
from vocos_student import FRAME  # noqa: E402


def build(arch):
    assert arch.get("enc") == "stft", f"not an STFT encoder: {arch}"
    assert arch.get("embed_k", 1) == 1, "only the k1 embed is supported"
    return STFTEncoder(arch["dim"], arch["blocks"], arch["mlp"], tuple(arch["ffts"]), arch["power"],
                       arch["embed_k"])


def load_encoder(ckpt, use_ema=True):
    """-> (STFTEncoder in eval mode, checkpoint info)."""
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = s["arch"]
    key = "G_ema" if use_ema else "G"
    if s.get(key) is None:
        raise SystemExit(f"{ckpt} has no {key}")
    sd = {k[4:]: v for k, v in s[key].items() if k.startswith("enc.")}
    enc = build(arch)
    enc.load_state_dict(sd)
    info = {"step": s.get("step"), "weights": key, "licences": (s.get("args") or {}).get("licences"),
            "decoder": {"arch": arch.get("dec"), "sha": arch.get("dec_sha"), "path": arch.get("dec_path")},
            "source_sha256": {"checkpoint": sha(ckpt)}}
    return enc.eval(), info


@torch.no_grad()
def forward64(enc, x):
    """STFTEncoder.forward in float64 throughout (its forward casts back to float32 in places): x [B, 1, T] ->
    z [B, latent, T / 512]. The precision floor any float32 implementation is measured against."""
    import torch.nn.functional as F

    B, T = x.shape[0], x.shape[-1]
    Fr = T // FRAME
    x = x.reshape(B, T).double()
    out = []
    for w in enc.feats.ffts:
        hop = min(FRAME, w // 4)
        k, p = FRAME // hop, (w - hop) // 2
        s = torch.stft(F.pad(x, (p, p)), w, hop, window=torch.hann_window(w).double(), center=False,
                       return_complex=True)
        s = s * (s.real.square() + s.imag.square() + 1e-8).pow((enc.feats.power - 1) / 2)
        f = torch.cat([s.real, s.imag], 1)
        C = f.shape[1]
        out.append(f.reshape(B, C, Fr, k).permute(0, 3, 1, 2).reshape(B, k * C, Fr))
    sd = {k: v.double() for k, v in enc.state_dict().items()}

    def ln(h, name):  # h [B, F, C]
        return F.layer_norm(h, h.shape[-1:], sd[name + ".weight"], sd[name + ".bias"], 1e-6)

    h = F.conv1d(torch.cat(out, 1), sd["embed.weight"], sd["embed.bias"])
    h = ln(h.transpose(1, 2), "norm").transpose(1, 2)
    for i in range(len(enc.blocks)):
        p = f"blocks.{i}."
        d = F.conv1d(h, sd[p + "dwconv.weight"], sd[p + "dwconv.bias"], padding=3, groups=h.shape[1])
        d = F.gelu(F.linear(ln(d.transpose(1, 2), p + "norm"), sd[p + "pw1.weight"], sd[p + "pw1.bias"]))
        d = F.linear(d, sd[p + "pw2.weight"], sd[p + "pw2.bias"])
        h = h + (sd[p + "gamma"] * d).transpose(1, 2)
    return F.linear(ln(h.transpose(1, 2), "final_norm"), sd["out.weight"], sd["out.bias"]).transpose(1, 2)


def tensors_of(enc):
    sd = enc.state_dict()
    out = {}
    for i, w in enumerate(enc.feats.ffts):
        out[f"window{i}"] = torch.hann_window(w)
    out["power"] = torch.tensor([enc.feats.power], dtype=torch.float32)
    out["embed.weight"] = sd["embed.weight"][:, :, 0]
    out["embed.bias"] = sd["embed.bias"]
    out["norm.weight"], out["norm.bias"] = sd["norm.weight"], sd["norm.bias"]
    for i in range(len(enc.blocks)):
        p, o = f"blocks.{i}.", f"block{i}."
        out[o + "dwconv.weight"] = sd[p + "dwconv.weight"][:, 0, :]
        out[o + "dwconv.bias"] = sd[p + "dwconv.bias"]
        for name in ("norm", "pw1", "pw2"):
            out[o + name + ".weight"] = sd[p + name + ".weight"]
            out[o + name + ".bias"] = sd[p + name + ".bias"]
        out[o + "gamma"] = sd[p + "gamma"]
    for name in ("final_norm", "out"):
        out[name + ".weight"] = sd[name + ".weight"]
        out[name + ".bias"] = sd[name + ".bias"]
    return out


def write(enc, out, info):
    tensors = tensors_of(enc)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_safetensors(out + ".safetensors", tensors, {"format": "picotone-dac44-stftenc"})
    n_params = sum(t.numel() for k, t in tensors.items() if not k.startswith(("window", "power")))
    ffts = list(enc.feats.ffts)
    blocks = len(enc.blocks)
    cfg = {
        "model": "dac44-stftenc",
        "arch": {"enc": "stft", "dim": enc.embed.out_channels, "blocks": blocks,
                 "mlp": enc.blocks[0].pw1.out_features, "ffts": ffts, "power": enc.feats.power, "embed_k": 1,
                 "latent": enc.out.out_features},
        "sample_rate": 44100, "frame": FRAME, "channels": enc.feats.channels,
        "receptive_frames": 3 * blocks,
        "reach_samples": max((w - min(FRAME, w // 4)) // 2 for w in ffts),
        "params": n_params, **info,
    }
    json.dump(cfg, open(out + ".json", "w"), indent=1)
    size = os.path.getsize(out + ".safetensors")
    print(f"wrote {out}.safetensors ({n_params / 1e6:.3f}M params, {size / 1e6:.2f} MB) and {out}.json, "
          f"sha256 {sha(out + '.safetensors')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="train.py checkpoint of an --arch encoder --enc-arch stft run")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--no-ema", action="store_true", help="export G rather than G_ema")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    enc, info = load_encoder(a.ckpt, not a.no_ema)
    write(enc, a.out, info)


if __name__ == "__main__":
    main()
