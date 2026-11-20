"""Post-training int8 variants of the net expert: net-only bits per code on the test streams.

Usage: int8_eval.py TRAIN_DIR CKPT DATA.npz [FRAMES]
Emulates, in exact integers, a forward pass with int8 weights (one scale per output row, as an
integer multiplier and a shift) and narrow activations, and compares it with the current i16
format and with the float net. Variants: activation bits (7 or 8, unsigned, range [0, 8)),
input bits for the centroid sums (signed 7/8-bit, or the current 11-bit), and which layers use
int8 weights.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import export_net  # noqa: E402
import train_net  # noqa: E402

LN2 = math.log(2.0)
CKPT, DATA = sys.argv[2], sys.argv[3]
FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 3000


def layer(w, bias, in_scale, out_scale, bits=8, per_row=True):
    """Quantises float weights w [out, in]: int weights, int bias, multiplier, shift.

    out_int = ((acc >> PRE) * mult) >> POST with acc = W8 a + bias; chosen so that
    out_int ~ out_scale * (w a_float + bias_float) and everything fits an i32.
    """
    top = 2 ** (bits - 1) - 1
    peak = np.abs(w).max(axis=1, keepdims=True) if per_row else np.full((w.shape[0], 1), np.abs(w).max())
    peak = np.maximum(peak, 1e-9)
    s = top / peak                                        # weight scale per row
    w8 = np.clip(np.rint(w * s), -top, top).astype(np.int64)
    b = np.rint(bias * s[:, 0] * in_scale).astype(np.int64)
    ratio = out_scale / (s[:, 0] * in_scale)              # acc -> output units
    mult = np.clip(np.rint(ratio * 2 ** 20), 1, 32767).astype(np.int64)
    return w8, b, mult


PRE, POST = 8, 12


def apply(lay, a, hi, extra=None):
    w8, b, mult = lay
    acc = a @ w8.T + b
    if extra is not None:
        acc = acc + extra
    assert np.abs(acc).max() < 2 ** 31
    out = ((acc >> PRE) * mult) >> POST
    assert np.abs((acc >> PRE) * mult).max() < 2 ** 31
    return out if hi is None else np.clip(out, 0, hi)


def build(sd, act_bits, in_bits, out_bits=8, per_row=True, wbits=8):
    a_scale = 2 ** act_bits / 8.0                         # activations cover [0, 8)
    x_scale = 2 ** (in_bits - 1) / 64.0                   # inputs cover [-64, 64)
    q = {"act_hi": 2 ** act_bits - 1, "x_scale": x_scale, "in_hi": 2 ** (in_bits - 1) - 1}
    q["l1"] = layer(sd["l1.weight"], sd["l1.bias"], x_scale, a_scale, wbits, per_row)
    s1 = (2 ** (wbits - 1) - 1) / np.maximum(np.abs(sd["l1.weight"]).max(axis=1), 1e-9)
    if not per_row:
        s1 = np.full_like(s1, (2 ** (wbits - 1) - 1) / np.abs(sd["l1.weight"]).max())
    q["kemb"] = np.rint(sd["kemb.weight"] * s1 * x_scale).astype(np.int64)
    q["l2"] = layer(sd["l2.weight"], sd["l2.bias"], a_scale, a_scale, wbits, per_row)
    q["heads"] = []
    k = 0
    while f"a.{k}.weight" in sd:
        # A_k (inputs: activations) and B_k (inputs: centroid sums) share one accumulator, so
        # they share the row scale; B's inputs are rescaled into activation units first.
        wa = sd[f"a.{k}.weight"]
        wb = sd[f"b.{k}.weight"] * (a_scale / x_scale)
        both = layer(np.concatenate([wa, wb], axis=1), sd[f"a.{k}.bias"], a_scale, a_scale, wbits, per_row)
        o = layer(sd[f"o.{k}.weight"] / LN2, sd[f"o.{k}.bias"] / LN2, a_scale, 64.0, out_bits, per_row)
        q["heads"].append((both, wa.shape[1], o))
        k += 1
    return q


def squash(q, v16):
    """Centroid sums in units of 1/16 -> the variant's input integers."""
    return np.clip(np.floor(v16 * q["x_scale"] / 16.0), -q["in_hi"] - 1, q["in_hi"]).astype(np.int64)


def logits(q, x16, part16, kidx, k):
    h1 = apply(q["l1"], squash(q, x16), q["act_hi"], extra=q["kemb"][kidx])
    h2 = apply(q["l2"], h1, q["act_hi"])
    both, _, o = q["heads"][k]
    g = apply(both, np.concatenate([h2, squash(q, part16)], axis=1), q["act_hi"])
    return apply(o, g, None)


def main():
    data = np.load(DATA)
    centroids = data["centroids"].astype(np.int32)
    bandwidths = [int(k) for k in data["bandwidths"]]
    sd = {k: v.numpy().astype(np.float64) for k, v in torch.load(CKPT, map_location="cpu", weights_only=True).items()}
    q16 = export_net.quantise(CKPT)
    variants = {
        "act7 in7": build(sd, 7, 7), "act7 in8": build(sd, 7, 8), "act8 in8": build(sd, 8, 8),
        "act7 in8 one scale per matrix": build(sd, 7, 8, per_row=False),
        "act7 in8, out layer 16-bit": build(sd, 7, 8, out_bits=16),
        "act7 in8, all weights 16-bit": build(sd, 7, 8, out_bits=16, wbits=16),
        "act10 in11, weights 8-bit": build(sd, 10, 11),
    }
    for name in ("test_old", "test_new"):
        codes = data[f"{name}_codes"]
        since = train_net.since_start(data[f"{name}_lengths"])
        idx = np.sort(np.random.default_rng(0).choice(len(codes), min(FRAMES, len(codes)), replace=False))
        for kk in (16, 8, 4):
            x, part, y = train_net.batch(centroids, codes, since[idx], idx, kk)
            x16 = np.rint(x.numpy() * 16).astype(np.int64)
            kidx = bandwidths.index(kk)
            rows = {"i16 (current)": []}
            rows.update({v: [] for v in variants})
            for k in range(kk):
                p16 = np.rint(part[:, k].numpy() * 16).astype(np.int64)
                cur = export_net.int_logits(q16, x16, p16, kidx, k) / 64.0 * LN2
                rows["i16 (current)"].append(float(torch.nn.functional.cross_entropy(torch.from_numpy(cur), y[:, k])) / LN2)
                for v, q in variants.items():
                    li = logits(q, x16, p16, kidx, k) / 64.0 * LN2
                    rows[v].append(float(torch.nn.functional.cross_entropy(torch.from_numpy(li), y[:, k])) / LN2)
            base = float(np.mean(rows["i16 (current)"]))
            for v, r in rows.items():
                m = float(np.mean(r))
                print(f"{name} K={kk:2d} {v:34s} {m:.4f} bits  {100 * (m - base) / base:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
