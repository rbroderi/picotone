"""Helpers shared by the exports to picotone's Rust format (export_encoder.py, ../../tools/export_*.py): a plain
safetensors writer, a file hash, and DAC 44 kHz's quantiser tables read from its weights file."""
import hashlib
import json
import struct

import numpy as np
import torch

N_CB, CB_SIZE, CB_DIM = 9, 1024, 8
IN_PER_CB = CB_DIM + 1  # a codebook's 8 embedding values plus its "present" input, in a folded input conv


def write_safetensors(path, tensors, metadata=None):
    """Plain safetensors writer (float32 only), so the export needs no extra package."""
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def wn(g, v):
    """DAC's weight norm (dim 0): g * v / ||v|| over the other dims."""
    return g * v / v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))


def load_quantiser(path):
    """DAC's 9 codebooks and their output projections: -> (books [9, 1024, 8], proj [9, 1024, 8], bias [9, 1024])."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    q = "quantizer.quantizers.{}."
    books = torch.stack([sd[q.format(i) + "codebook.weight"] for i in range(N_CB)])
    proj = torch.stack([wn(sd[q.format(i) + "out_proj.weight_g"], sd[q.format(i) + "out_proj.weight_v"])[:, :, 0]
                        for i in range(N_CB)])
    bias = torch.stack([sd[q.format(i) + "out_proj.bias"] for i in range(N_CB)])
    return books, proj, bias


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
