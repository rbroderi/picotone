"""Quantises a trained net to the integer format of entropy/src/net.rs and checks it.

    python export_net.py --ckpt out/net48v2.pt --data train_data_48k_full.npz

Prints the float and the emulated-integer cross-entropy per codebook on the test streams of
the archive; the two must agree to about 0.01 bit. `blob()` is what tools/build_tables.py
embeds in a `.ecxm` table (weights are i16, biases i32, all shifts fixed, so the Rust side
reproduces `int_logits()` exactly).

Two net formats exist. Format 1 (table version 4) is the original one. Format 2 (table
version 5) adds the position input after the bandwidth biases: u32 period, u32 bucket
count, one u8 bucket per position, then one h1-long i32 bias per bucket. A format 1
checkpoint (no position input) can be written in format 2 with period 0.
"""
import argparse
import math
import struct

import numpy as np
import torch

LN2 = math.log(2.0)
OLD_BANDWIDTHS = [2, 4, 8, 16]   # what checkpoints without a bandwidth list were trained for
BIAS_LIMIT = 1 << 28             # net.rs rejects larger biases


def _i16(x, limit):
    return np.clip(np.rint(x), -limit, limit).astype("<i2")


def _bias(x, scale):
    out = np.rint(x * scale).astype("<i4")
    assert np.abs(out.astype(np.int64)).max(initial=0) <= BIAS_LIMIT, "net bias is out of range"
    return out


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_bandwidths(sd):
    if "bandwidth_list" in sd:
        return [int(k) for k in sd["bandwidth_list"]]
    return list(OLD_BANDWIDTHS)


def checkpoint_period(sd):
    return int(len(sd["pos_map"])) if "pemb.weight" in sd else 0


def quantise(path, weights=None):
    """Integer tables. `weights[k]` (sixteenths) scales codebook k's output layer."""
    sd = {k: v.numpy().astype(np.float64) for k, v in load_state(path).items()}
    heads = len([k for k in sd if k.startswith("a.") and k.endswith(".weight")])
    weights = weights or [16] * heads
    q = {"h1": sd["l1.weight"].shape[0], "h2": sd["l2.weight"].shape[0],
         "kemb": _bias(sd["kemb.weight"], 2 ** 14),
         "w1": _i16(sd["l1.weight"] * 1024, 2047).T.copy(),          # [input][output]
         "b1": _bias(sd["l1.bias"], 2 ** 14),
         "w2": _i16(sd["l2.weight"] * 1024, 2047).T.copy(),
         "b2": _bias(sd["l2.bias"], 2 ** 17), "heads": [],
         "period": 0, "pos_map": np.zeros(0, dtype=np.int64), "pemb": np.zeros((0, 0), dtype="<i4")}
    if "pemb.weight" in sd:
        q["period"] = len(sd["pos_map"])
        q["pos_map"] = sd["pos_map"].astype(np.int64)
        q["pemb"] = _bias(sd["pemb.weight"], 2 ** 14)
    for k in range(heads):
        w = weights[k] / 16.0
        q["heads"].append({
            "a": _i16(sd[f"a.{k}.weight"] * 1024, 2047).T.copy(),      # [h2][rank]
            "b": _i16(sd[f"b.{k}.weight"] * 8192, 2047).T.copy(),      # [128][rank]
            "c": _bias(sd[f"a.{k}.bias"], 2 ** 17),
            "o": _i16(sd[f"o.{k}.weight"] / LN2 * 1024 * w, 16383).T.copy(),  # [rank][1024]
            "d": np.rint(sd[f"o.{k}.bias"] / LN2 * 64 * w).astype("<i4"),
        })
    return q


def blob(path, bandwidths=None, weights=None, net_format=None):
    """Bytes of the net section of a model table.

    `bandwidths` defaults to the checkpoint's own list; `net_format` to 2 when the
    checkpoint has a position input, else 1. Table version 4 needs format 1, version 5
    format 2.
    """
    sd = load_state(path)
    bandwidths = bandwidths or checkpoint_bandwidths(sd)
    q = quantise(path, weights)
    net_format = net_format or (2 if q["period"] else 1)
    assert q["kemb"].shape[0] == len(bandwidths), "checkpoint was trained for other bandwidths"
    assert net_format == 2 or not q["period"], "a net with a position input needs format 2 (table v5)"
    out = [struct.pack("<3I", q["h1"], q["h2"], len(bandwidths)),
           struct.pack(f"<{len(bandwidths)}I", *bandwidths), q["kemb"].tobytes()]
    if net_format == 2:
        buckets = q["pemb"].shape[0]
        assert q["period"] <= 4096 and buckets <= 256
        out += [struct.pack("<2I", q["period"], buckets), q["pos_map"].astype(np.uint8).tobytes(),
                q["pemb"].tobytes()]
    out += [q["w1"].tobytes(), q["b1"].tobytes(), q["w2"].tobytes(), q["b2"].tobytes()]
    for h in q["heads"]:
        out += [struct.pack("<I", h["a"].shape[1]), h["a"].tobytes(), h["b"].tobytes(),
                h["c"].tobytes(), h["o"].tobytes(), h["d"].tobytes()]
    return b"".join(out)


def int_logits(q, x, part, kidx, k, pos=None):
    """Integer forward pass, mirroring net.rs. x [B, 512] and part [B, 128] are integers
    in units of 1/16; pos [B] is the frame's position (t mod period); returns logits
    [B, 1024] in units of 1/64 bit."""
    x = x.astype(np.int64)
    acc = q["b1"].astype(np.int64) + q["kemb"][kidx].astype(np.int64) + x @ q["w1"].astype(np.int64)
    if q["period"]:
        acc = acc + q["pemb"][q["pos_map"][pos]].astype(np.int64)
    h1 = np.clip(acc >> 7, 0, 1023)
    h2 = np.clip((q["b2"].astype(np.int64) + h1 @ q["w2"].astype(np.int64)) >> 10, 0, 1023)
    h = q["heads"][k]
    g = np.clip((h["c"].astype(np.int64) + h2 @ h["a"].astype(np.int64)
                 + part.astype(np.int64) @ h["b"].astype(np.int64)) >> 10, 0, 1023)
    return h["d"].astype(np.int64) + ((g @ h["o"].astype(np.int64)) >> 11)


def load_net(path, kmax):
    import train_net

    sd = load_state(path)
    net = train_net.Net(checkpoint_bandwidths(sd), kmax, sd["l1.weight"].shape[0], sd["l2.weight"].shape[0],
                        sd["a.0.weight"].shape[0], sd[f"a.{kmax - 1}.weight"].shape[0], checkpoint_period(sd))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # Old checkpoints lack the two buffers, which then keep their (identical) defaults.
    assert not unexpected and set(missing) <= {"bandwidth_list", "pos_map"}, (missing, unexpected)
    net.eval()
    return net


def main():
    import train_net

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=4000)
    args = ap.parse_args()
    data = np.load(args.data)
    centroids = data["centroids"].astype(np.int32)
    kmax = centroids.shape[0]
    net = load_net(args.ckpt, kmax)
    q = quantise(args.ckpt)
    period = q["period"]
    print("bandwidths", net.bandwidths, "period", period, "position buckets", q["pemb"].shape[0])
    worst = 0.0
    for name in ("test_old", "test_new"):
        if f"{name}_codes" not in data:
            continue
        codes = data[f"{name}_codes"]
        since = train_net.since_start(data[f"{name}_lengths"])
        idx = np.sort(np.random.default_rng(0).choice(len(codes), min(args.frames, len(codes)), replace=False))
        frames = train_net.Frames(centroids, codes, data[f"{name}_lengths"], torch.device("cpu"))
        for k_total in sorted({kmax, *(k for k in (15, 8, 4) if k in net.bandwidths)}, reverse=True):
            x, part, y = train_net.batch(centroids, codes, since[idx], idx, k_total)
            # The tensor batching used for training must give exactly the same inputs.
            tx, tpart, ty = frames.batch(torch.from_numpy(idx), torch.from_numpy(since[idx]).long(), k_total)
            assert torch.equal(tx, x) and torch.equal(tpart, part) and torch.equal(ty, y), "batching differs"
            xi = np.rint(x.numpy() * 16).astype(np.int64)
            kidx = net.bandwidths.index(k_total)
            pos = since[idx] % period if period else None
            fb, ib = [], []
            with torch.no_grad():
                h2 = net.trunk(x, torch.full((len(idx),), kidx, dtype=torch.long),
                               torch.from_numpy(pos).long() if period else None)
                for k in range(k_total):
                    fb.append(float(torch.nn.functional.cross_entropy(net.head(k, h2, part[:, k]), y[:, k])) / LN2)
                    li = int_logits(q, xi, np.rint(part[:, k].numpy() * 16).astype(np.int64), kidx, k, pos)
                    li = li / 64.0 * LN2
                    ib.append(float(torch.nn.functional.cross_entropy(torch.from_numpy(li), y[:, k])) / LN2)
            tag = f"{name} K={k_total}"
            print(tag, "float bits:", [round(v, 2) for v in fb], "mean", round(float(np.mean(fb)), 4))
            print(tag, "int   bits:", [round(v, 2) for v in ib], "mean", round(float(np.mean(ib)), 4))
            gap = abs(float(np.mean(fb)) - float(np.mean(ib)))
            worst = max(worst, gap)
            print(tag, "float/int gap", round(gap, 4), "OK" if gap < 0.02 else "TOO LARGE", flush=True)
    print("worst float/int gap", round(worst, 4), "OK" if worst < 0.02 else "TOO LARGE")
    if worst >= 0.02:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
