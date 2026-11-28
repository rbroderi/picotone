"""Shared helpers: load code grids, EnCodec codebooks, splits."""
import glob
import json
import os
import struct

import numpy as np

import ecdcio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("ENTROPY_WORK", os.path.join(ROOT, "work"))
WEIGHTS = os.environ.get("PICOTONE_WEIGHTS") or os.path.join(ROOT, "weights")
# The picotone executable; its `dev` subcommands are the table builder's Rust steps.
PICOTONE = os.environ.get("PICOTONE_EXE", os.path.join(ROOT, "target", "release", "picotone.exe"))
# ENTROPY_K48 lets the 48 kHz tools work on files with another codebook count (music24: 15).
KMAX = {"24k": 32, "48k": int(os.environ.get("ENTROPY_K48", "16"))}
MODEL_NAME = {"24k": "encodec_24khz", "48k": "encodec_48khz"}
FRAME_RATE = {"24k": 75, "48k": 150}
DATA = os.environ.get("ENTROPY_DATA", "all")


def load_codebooks(tag):
    """Returns float32 array [K, 1024, 128] of RVQ centroids."""
    path = os.path.join(WEIGHTS, MODEL_NAME[tag] + ".safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = []
        for k in range(KMAX[tag]):
            info = header[f"quantizer.layers.{k}.codebook.embed"]
            assert info["dtype"] == "F32" and info["shape"] == [1024, 128]
            a, b = info["data_offsets"]
            f.seek(base + a)
            out.append(np.frombuffer(f.read(b - a), dtype="<f4").reshape(1024, 128))
    return np.stack(out)


def load_files(tag, split):
    """Returns list of dicts {name, group, codes[T, K]} for a split: fit / dev / test.

    `dev` is every 8th fit clip (by sorted name); `fit` excludes those; `fitall` is both.
    """
    if split == "test2":
        paths = sorted(glob.glob(os.path.join(WORK, "ecdc2", f"test2__*.{tag}.ecdc")))
    elif split == "test":
        paths = sorted(glob.glob(os.path.join(WORK, "ecdc", f"test__*.{tag}.ecdc")))
    else:
        paths = sorted(glob.glob(os.path.join(WORK, "ecdc", f"fit__*.{tag}.ecdc")))
        if DATA == "all":
            # phase 2 clips; ENTROPY_DATA=old restricts fitting to the phase 1 set
            paths += sorted(glob.glob(os.path.join(WORK, "ecdc2", f"fit2__*.{tag}.ecdc")))
    out = []
    for i, p in enumerate(paths):
        if split == "dev" and i % 8 != 3:
            continue
        if split == "fit" and i % 8 == 3:
            continue
        cache = p + ".npy"
        codes = None
        if os.path.exists(cache):
            try:
                codes = np.load(cache)
            except Exception:  # noqa: BLE001
                codes = None
        if codes is None:
            _, segs = ecdcio.read(p)
            codes = np.concatenate([c for _, c in segs], axis=0)
            tmp = cache + f".{os.getpid()}.tmp.npy"
            np.save(tmp, codes)
            try:
                os.replace(tmp, cache)
            except OSError:
                pass   # another process cached it first
        base = os.path.basename(p)
        out.append({"name": base, "group": base.split("__")[1], "codes": codes, "path": p})
    return out


def bundle(path):
    """Code bundle of a legacy flat `.ecdc` file, for `picotone dev`: returns `<path>.npy`.

    Writes, once, the codes as `<path>.npy` (`[frames, codebooks]`, uint16) and the layout as
    `<path>.json` (model, codebooks, length, stride, cross-fade, trims, loop flag, scales).
    The fit sets predate the `.pico` format, so they stay in Meta's container on disk.
    """
    npy, side = path + ".npy", path + ".json"
    if os.path.exists(npy) and os.path.exists(side):
        return npy
    meta, segs = ecdcio.read(path)
    codes = np.concatenate([c for _, c in segs], axis=0).astype(np.uint16)
    tmp = npy + f".{os.getpid()}.tmp.npy"
    np.save(tmp, codes)
    os.replace(tmp, npy)
    stride = meta.get("ss", 47520) if meta["m"] == "encodec_48khz" else None
    scales = [struct.unpack(">f", s)[0] for s, _ in segs if s is not None]
    layout = {"model": meta["m"], "codebooks": meta["nc"], "audio_length": meta["al"],
              "stride": stride, "crossfade": meta.get("xf", 0), "pre_skip": meta.get("ps", 0),
              "end_trim": meta.get("tr", 0), "looped": bool(meta.get("loop", False)),
              "scales": scales or None, "segment_frames": [len(c) for _, c in segs]}
    tmp = side + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(layout, f)
    os.replace(tmp, side)
    return npy


def concat(files):
    """Concatenates files; returns codes[N, K] int64 and a bool mask of file-start frames."""
    codes = np.concatenate([f["codes"] for f in files], axis=0).astype(np.int64)
    start = np.zeros(len(codes), dtype=bool)
    pos = 0
    for f in files:
        start[pos] = True
        pos += len(f["codes"])
    return codes, start


def prev_frame(col, start, fill=0):
    """Shifts a per-frame array by one frame, filling file starts."""
    out = np.empty_like(col)
    out[1:] = col[:-1]
    out[0] = fill
    out[start] = fill
    return out
