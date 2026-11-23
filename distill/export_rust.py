"""Export the student decoder for picotone's Rust decoder (milestone M0 in INTEGRATION.md).

    uv run --extra cpu python export_rust.py --student export/dac44_student_w512_172500.pt \
        --dac export/weights_44khz_8kbps_0.0.1.pth --out ../weights/dac44_student

Writes <out>.safetensors and <out>.json, and copies the .NOTICE.txt and .credits.jsonl that export.py wrote
next to --student (renamed to <out>.*).

The 9 codebook projections are folded into the input conv:
- z = sum over the first q codebooks of out_proj_i(codebook_i[code]), and out_proj_i is a 1x1 conv (8 -> 1024)
  with a bias.
- The input conv (k7, 1024 -> width) therefore equals a k7 conv over 81 inputs per frame: for each codebook,
  its 8 embedding values plus a 1 meaning "this codebook is present".
- A codebook above the tier, and the zero padding at the file edges, give 9 zeros, so the folded form is
  exact there too. The fold is done in float64 and rounded to float32 once.

Tensor names (all float32, weight norm already folded by export.py; Snake alphas as [C]):
  codebooks              [9, 1024, 8]        DAC's frozen codebook tables
  input.weight / .bias   [W, 81, 7] / [W]    the folded input conv
  block{b}.snake.alpha                       b = 0..3
  block{b}.up.weight     [Cin, Cout, 2s]     ConvTranspose1d, stride s, padding ceil(s/2)
  block{b}.res{r}.snake1.alpha / .conv1.weight [C, C, 7] (dilation 1/3/9) / .conv1.bias
  block{b}.res{r}.snake2.alpha / .conv2.weight [C, C, 1] / .conv2.bias
  output.snake.alpha / output.conv.weight [1, C, 7] / output.conv.bias, then tanh

Check: random codes at every tier, decoded through the original path (DAC quantiser + the unfolded student)
and the folded one; the largest difference is printed and must be under 1e-4.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct

import numpy as np
import torch
import torch.nn.functional as F

from student import RATES, StudentDecoder

N_CB, CB_SIZE, CB_DIM = 9, 1024, 8
IN_PER_CB = CB_DIM + 1


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
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    q = "quantizer.quantizers.{}."
    books = torch.stack([sd[q.format(i) + "codebook.weight"] for i in range(N_CB)])          # [9, 1024, 8]
    proj = torch.stack([wn(sd[q.format(i) + "out_proj.weight_g"], sd[q.format(i) + "out_proj.weight_v"])[:, :, 0]
                        for i in range(N_CB)])                                                 # [9, 1024, 8]
    bias = torch.stack([sd[q.format(i) + "out_proj.bias"] for i in range(N_CB)])               # [9, 1024]
    return books, proj, bias


def fold_input(w, b, proj, bias):
    """w [W, 1024, 7] -> [W, 81, 7]: per codebook, 8 embedding columns then its 'present' column."""
    w64, cols = w.double(), []
    for i in range(N_CB):
        cols.append(torch.einsum("ock,cj->ojk", w64, proj[i].double()))                # [W, 8, 7]
        cols.append(torch.einsum("ock,c->ok", w64, bias[i].double()).unsqueeze(1))    # [W, 1, 7]
    return torch.cat(cols, dim=1).float(), b.clone()


def folded_input(books, codes, n_q):
    """codes [9, F] -> the folded conv's input [81, F]."""
    rows = []
    for i in range(N_CB):
        if i < n_q:
            rows += [books[i][codes[i]].T, torch.ones(1, codes.shape[1])]
        else:
            rows += [torch.zeros(CB_DIM, codes.shape[1]), torch.zeros(1, codes.shape[1])]
    return torch.cat(rows, dim=0)


def rename(sd, folded_w, folded_b, books):
    """The student's nn.Sequential names -> the plain names above."""
    out = {"codebooks": books, "input.weight": folded_w, "input.bias": folded_b}
    for blk in range(len(RATES)):
        p = f"model.{blk + 1}.block."
        out[f"block{blk}.snake.alpha"] = sd[p + "0.alpha"].flatten()
        out[f"block{blk}.up.weight"] = sd[p + "1.weight"]
        out[f"block{blk}.up.bias"] = sd[p + "1.bias"]
        for r in range(3):
            q = f"{p}{r + 2}.block."
            o = f"block{blk}.res{r}."
            out[o + "snake1.alpha"] = sd[q + "0.alpha"].flatten()
            out[o + "conv1.weight"], out[o + "conv1.bias"] = sd[q + "1.weight"], sd[q + "1.bias"]
            out[o + "snake2.alpha"] = sd[q + "2.alpha"].flatten()
            out[o + "conv2.weight"], out[o + "conv2.bias"] = sd[q + "3.weight"], sd[q + "3.bias"]
    n = len(RATES) + 1
    out["output.snake.alpha"] = sd[f"model.{n}.alpha"].flatten()
    out["output.conv.weight"], out["output.conv.bias"] = sd[f"model.{n + 1}.weight"], sd[f"model.{n + 1}.bias"]
    assert len(out) == 3 + 4 * (3 + 3 * 6) + 3, len(out)
    return out


@torch.no_grad()
def check(G, books, proj, bias, fw, fb, frames=60, seed=0):
    """Largest |difference| between the original and folded decodes, per tier."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, CB_SIZE, (N_CB, frames), generator=g)
    worst = {}
    for n_q in (1, 4, 5, 6, 7, 9):
        z = sum(proj[i] @ books[i][codes[i]].T + bias[i][:, None] for i in range(n_q))
        y0 = G(z[None])
        h = F.conv1d(folded_input(books, codes, n_q)[None], fw, fb, padding=3)
        y1 = G.model[1:-3](h)
        y1 = G.model[-3:](y1.float())
        worst[n_q] = float((y0 - y1).abs().max())
    return worst


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="export.py's .pt (weight norm folded)")
    ap.add_argument("--dac", required=True, help="DAC 44 kHz weights (weights_44khz_8kbps_0.0.1.pth)")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()

    stem = os.path.splitext(a.student)[0]
    cfg_in = json.load(open(stem + ".json"))
    sd = torch.load(a.student, map_location="cpu", weights_only=True)
    width = cfg_in["width"]
    G = StudentDecoder(width)
    # export.py saved plain weights; build the module without weight norm to load them
    from student import fold_weight_norm
    fold_weight_norm(G)
    G.load_state_dict(sd)
    G.eval()

    books, proj, bias = load_quantiser(a.dac)
    fw, fb = fold_input(sd["model.0.weight"], sd["model.0.bias"], proj, bias)
    worst = check(G, books, proj, bias, fw, fb)
    print("fold check, max |orig - folded| per tier:", {k: f"{v:.2e}" for k, v in worst.items()})
    if max(worst.values()) > 1e-4:
        raise SystemExit("fold check failed")

    tensors = rename(sd, fw, fb, books)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_safetensors(a.out + ".safetensors", tensors, {"format": "picotone-dac44-student"})
    n_params = sum(t.numel() for t in tensors.values())
    cfg = {
        "model": "dac44-student",
        "width": width, "rates": list(RATES), "sample_rate": 44100, "hop": int(np.prod(RATES)),
        "codebooks": N_CB, "codebook_size": CB_SIZE, "codebook_dim": CB_DIM, "input_per_codebook": IN_PER_CB,
        "receptive_frames": 9, "params": n_params,
        "step": cfg_in["step"], "licences": cfg_in["licences"], "teacher": cfg_in["teacher"],
        "source_sha256": {"student": sha(a.student), "dac": sha(a.dac)},
        "fold_max_abs_dev": max(worst.values()),
    }
    json.dump(cfg, open(a.out + ".json", "w"), indent=1)
    for ext in (".NOTICE.txt", ".credits.jsonl"):
        if os.path.exists(stem + ext):
            shutil.copyfile(stem + ext, a.out + ext)
    size = os.path.getsize(a.out + ".safetensors")
    print(f"wrote {a.out}.safetensors ({n_params / 1e6:.2f}M params, {size / 1e6:.1f} MB) and {a.out}.json, "
          f"sha256 {sha(a.out + '.safetensors')[:16]}")


if __name__ == "__main__":
    main()
