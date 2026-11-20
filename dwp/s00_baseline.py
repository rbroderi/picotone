"""Baseline: stock quality on the test set at several bitrates, and a MAC (multiply-accumulate) budget per layer.

usage: python s00_baseline.py 24khz|48khz
"""
import sys
import time

import torch

from common import Bench, Log, perceptual, raw_weights

name = sys.argv[1]
log = Log(f"s00_baseline_{name}")
t0 = time.process_time()
b = Bench(name)
print("bench build cpu s", time.process_time() - t0, "clips", len(b.names), "x", tuple(b.x.shape), "codes", tuple(b.codes.shape))
for g, n in zip(b.groups, b.names):
    print("  ", g, n)

frame_rate = 75 if name == "24khz" else 150
for nq in sorted({1, 2, 4, 8, 16, b.codes.shape[0]}):
    if nq > b.codes.shape[0]:
        continue
    y = b.decode_from_pre(b.model.decoder.layers[0](b.dequant(b.codes[:nq])))
    r = perceptual(b.src, b._wave_rows(y), b.sr)
    log(f"stock n_q={nq}", r, kbps=nq * 10 * frame_rate / 1000)

# per-group stock metrics at full bitrate
rows_per_clip = b.src.shape[0] // len(b.names)
yrows = b._wave_rows(b.y)
for g in sorted(set(b.groups)):
    idx = [i * rows_per_clip + k for i, gg in enumerate(b.groups) if gg == g for k in range(rows_per_clip)]
    log(f"stock full, group {g}", perceptual(b.src[idx], yrows[idx], b.sr))

# ---- MAC budget (multiply-accumulates per second of audio, per channel-batch of one stream)
w = raw_weights(name)
sr = b.sr
rates = {}
total = {"enc": 0.0, "dec": 0.0}
rows = []
for side in ("encoder", "decoder"):
    ratios = [2, 4, 5, 8] if side == "encoder" else [8, 5, 4, 2]
    rate = sr if side == "encoder" else frame_rate
    keys = sorted({k.rsplit(".", 1)[0] for k in w if k.startswith(side)}, key=lambda s: [int(p) if p.isdigit() else p for p in s.split(".")])
    seen = set()
    for k in w:
        if not k.startswith(side):
            continue
        parts = k.split(".")
        li = int(parts[2])
        if "lstm" in k:
            if "weight" in k:
                mac = w[k].numel() * frame_rate
                rows.append((side, li, k.split("lstm.")[1], tuple(w[k].shape), mac))
            continue
        if not (k.endswith("weight_v") or k.endswith("conv.weight")):
            continue
        rows.append((side, li, ".".join(parts[3:-1]), tuple(w[k].shape), None))

# conv rates: walk the layer list in order
def conv_rate(side, li, sub, shape):
    if side == "encoder":
        # layers: 0 conv@sr; then for each ratio: resblock, elu, down conv (output rate /ratio)
        order = {0: sr}
        r = sr
        j = 1
        for ratio in [2, 4, 5, 8]:
            order[j] = r          # resblock at current rate
            r = r / ratio
            order[j + 2] = r      # down conv, counted at output rate
            j += 3
        order[15] = frame_rate
        return order[li]
    order = {0: frame_rate}
    r = frame_rate
    j = 3
    for ratio in [8, 5, 4, 2]:
        order[j] = r              # transposed conv: each input frame touches all taps -> count at input rate
        r = r * ratio
        order[j + 1] = r          # resblock at upsampled rate
        j += 3
    order[15] = sr
    return order[li]


tot = {"encoder": 0.0, "decoder": 0.0}
lstm = {"encoder": 0.0, "decoder": 0.0}
out = []
for side, li, sub, shape, mac in rows:
    if mac is None:
        n = 1
        for s in shape:
            n *= s
        mac = n * conv_rate(side, li, sub, shape)
    else:
        lstm[side] += mac
    tot[side] += mac
    out.append((side, li, sub, shape, mac))
for side, li, sub, shape, mac in sorted(out, key=lambda r: (r[0], r[1], r[2])):
    log(f"MAC {side}.{li}.{sub} {shape}", None, mmac_per_s=mac / 1e6, share=100 * mac / tot[side])
for side in tot:
    log(f"MAC total {side}", None, mmac_per_s=tot[side] / 1e6, lstm_share=100 * lstm[side] / tot[side])
