# SPDX-License-Identifier: GPL-3.0-or-later
"""Export DAC 44 kHz's residual vector quantiser (MIT, Copyright (c) 2023-present, Descript) for picotone's
HQ encoder (`src/dac.rs`, `DacQuantizer`): the trained STFT encoder (`dac44_stftenc`) makes the latent, and
DAC's own quantiser turns it into DAC's codes. Run it in distill/'s environment:

    cd distill
    uv run --extra cpu python ../tools/export_quantizer.py --dac export/weights_44khz_8kbps_0.0.1.pth \
        --out ../weights/dac44_quantizer

The tensors are those of distill/export_encoder.py (weight norm removed with PyTorch's remove_weight_norm),
without DAC's conv encoder. Writes <out>.safetensors and <out>.json, all float32, i = 0..8:

  quant{i}.in_proj.weight [8, 1024, 1] / .bias [8]
  quant{i}.codebook [1024, 8]          the raw entries (the residual subtracts out_proj of these)
  quant{i}.codebook_unit [1024, 8]     F.normalize of the entries (the nearest-entry search uses these)
  quant{i}.out_proj.weight [1024, 8, 1] / .bias [1024]

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
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_safetensors(a.out + ".safetensors", t, {"format": "picotone-dac44-quantizer"})
    n_params = sum(v.numel() for v in t.values())
    cfg = {
        "model": "dac44-quantizer", "sample_rate": 44100, "hop": 512, "latent_dim": 1024, "codebooks": N_CB,
        "codebook_size": 1024, "codebook_dim": 8, "params": n_params,
        "licence": "MIT (descript-audio-codec, Copyright (c) 2023-present, Descript)",
        "source_sha256": {"dac": sha(a.dac)},
    }
    with open(a.out + ".json", "w") as f:
        json.dump(cfg, f, indent=1)
    size = os.path.getsize(a.out + ".safetensors")
    print(f"wrote {a.out}.safetensors ({n_params / 1e6:.2f}M params, {size / 1e6:.2f} MB), "
          f"sha256 {sha(a.out + '.safetensors')[:16]}")
    if a.check:
        ours, theirs = read_safetensors(a.out + ".safetensors"), read_safetensors(a.check)
        for name, v in ours.items():
            assert name in theirs, f"{a.check} has no {name}"
            assert v.shape == theirs[name].shape and v.tobytes() == theirs[name].tobytes(), f"{name} differs"
        print(f"all {len(ours)} tensors equal {a.check}'s, bit for bit")


if __name__ == "__main__":
    main()
