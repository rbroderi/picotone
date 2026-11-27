# SPDX-License-Identifier: GPL-3.0-or-later
"""Export DAC 44 kHz's residual vector quantiser (MIT, Copyright (c) 2023-present, Descript) for picotone's
HQ encoder (`src/dac.rs`, `DacQuantizer`): the trained STFT encoder (`dac44_stftenc`) makes the latent, DAC's
own 9 codebooks code it, and 7 more codebooks, trained with the decoder (the q16 run), code what is left. Run it in
distill/'s environment:

    cd distill
    uv run --extra cpu python ../tools/export_quantizer.py --dac export/weights_44khz_8kbps_0.0.1.pth \
        --ext X:/audio/distill/weights/q16_codebooks.pt --out ../weights/dac44_quantizer

Stages 0..8 are DAC's, as distill/export_encoder.py exports them (weight norm removed with PyTorch's
remove_weight_norm), without DAC's conv encoder. Stages 9..15 are `--ext`'s `quantizer.quantizers.{9..15}.*`
tensors (trained without weight norm, so copied as they are). Writes <out>.safetensors and <out>.json, all float32,
i = 0..15:

  quant{i}.in_proj.weight [8, 1024, 1] / .bias [8]
  quant{i}.codebook [1024, 8]          the raw entries (the residual subtracts out_proj of these)
  quant{i}.codebook_unit [1024, 8]     F.normalize of the entries (the nearest-entry search uses these)
  quant{i}.out_proj.weight [1024, 8, 1] / .bias [1024]

Stage 9 reads the residual recomputed as latent - (sum of stages 0..8's projections), as the training code does,
not the running residual (see `src/dac.rs`).

`--check <file>.safetensors` also compares every tensor, bit for bit, with the same name in another export
(such as an older dac44_encoder.safetensors), so the codebook hash the .pico header binds stays the same.

distill/ is found next to this file, or at $PICOTONE_DISTILL. The DAC weights are only read.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

DISTILL = os.environ.get("PICOTONE_DISTILL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                             "distill")
sys.path.insert(0, os.path.abspath(DISTILL))

from export_encoder import plain, tensors  # noqa: E402
from export_rust import N_CB, sha, write_safetensors  # noqa: E402

EXT = 7  # codebooks 10-16


def read_safetensors(path):
    """name -> float32 array of a safetensors file (float32 tensors only)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, e in header.items():
        if name == "__metadata__":
            continue
        assert e["dtype"] == "F32", f"{name} is {e['dtype']}"
        lo, hi = e["data_offsets"]
        out[name] = np.frombuffer(data[lo:hi], dtype="<f4").reshape(e["shape"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dac", required=True, help="DAC 44 kHz weights (weights_44khz_8kbps_0.0.1.pth)")
    ap.add_argument("--ext", required=True, help="codebooks 10-16 (q16_codebooks.pt)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--check", help="an export to compare the tensors with, bit for bit")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import dac

    model = dac.DAC.load(a.dac).eval()
    plain(model)
    assert len(model.quantizer.quantizers) == N_CB
    with torch.no_grad():
        t = {k: v for k, v in tensors(model).items() if k.startswith("quant")}
    assert len(t) == 6 * N_CB, sorted(t)
    ext = torch.load(a.ext, map_location="cpu", weights_only=True)
    assert len(ext) == 5 * EXT, sorted(ext)
    for j in range(N_CB, N_CB + EXT):
        q, p = f"quantizer.quantizers.{j}.", f"quant{j}."
        book = ext[q + "codebook.weight"].float()
        t[p + "in_proj.weight"], t[p + "in_proj.bias"] = ext[q + "in_proj.weight"], ext[q + "in_proj.bias"]
        t[p + "codebook"] = book
        t[p + "codebook_unit"] = torch.nn.functional.normalize(book)
        t[p + "out_proj.weight"], t[p + "out_proj.bias"] = ext[q + "out_proj.weight"], ext[q + "out_proj.bias"]
    assert len(t) == 6 * (N_CB + EXT)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_safetensors(a.out + ".safetensors", t, {"format": "picotone-dac44-quantizer"})
    n_params = sum(v.numel() for v in t.values())
    cfg = {
        "model": "dac44-quantizer", "sample_rate": 44100, "hop": 512, "latent_dim": 1024, "codebooks": N_CB + EXT,
        "codebook_size": 1024, "codebook_dim": 8, "params": n_params,
        "licence": "stages 0-8: MIT (descript-audio-codec, Copyright (c) 2023-present, Descript); "
                   "stages 9-15: GPL-3.0-or-later (picotone)",
        "source_sha256": {"dac": sha(a.dac), "ext": sha(a.ext)},
    }
    with open(a.out + ".json", "w") as f:
        json.dump(cfg, f, indent=1)
    size = os.path.getsize(a.out + ".safetensors")
    print(f"wrote {a.out}.safetensors ({n_params / 1e6:.2f}M params, {size / 1e6:.2f} MB), "
          f"sha256 {sha(a.out + '.safetensors')[:16]}")
    if a.check:
        ours, theirs = read_safetensors(a.out + ".safetensors"), read_safetensors(a.check)
        shared = [name for name in ours if name in theirs]
        for name in shared:
            v = ours[name]
            assert v.shape == theirs[name].shape and v.tobytes() == theirs[name].tobytes(), f"{name} differs"
        print(f"the {len(shared)} tensors {a.check} also has equal its, bit for bit "
              f"({len(ours) - len(shared)} are new)")


if __name__ == "__main__":
    main()
