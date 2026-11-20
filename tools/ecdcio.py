"""Read/write flat ECDC files (lm=false) as numpy code grids."""
import json
import math
import struct

import numpy as np

MODELS = {  # name: (normalize, segment, stride, frame_rate, rate)
    "encodec_24khz": (False, None, None, 75, 24000),
    "encodec_48khz": (True, 48000, 47520, 150, 48000),
}


def seg_frames(meta):
    normalize, seg, stride, fr, rate = MODELS[meta["m"]]
    al = meta["al"]
    if seg and meta.get("ss"):
        stride = meta["ss"]   # configurable segment stride (dwp seam fix)
    seg, stride = seg or al, stride or al
    out = []
    for off in range(0, al, stride):
        length = min(al - off, seg)
        out.append(math.ceil(length * fr / rate))
    return out


def read(path):
    """Returns (meta, list of (scale_bytes_or_None, codes[T, K] uint16))."""
    data = open(path, "rb").read()
    n = struct.unpack(">I", data[5:9])[0]
    meta = json.loads(data[9:9 + n])
    pos = 9 + n
    normalize = MODELS[meta["m"]][0]
    nc = meta["nc"]
    segs = []
    for t in seg_frames(meta):
        scale = None
        if normalize:
            scale = data[pos:pos + 4]
            pos += 4
        nbytes = (t * nc * 10 + 7) // 8
        raw = np.frombuffer(data[pos:pos + nbytes], dtype=np.uint8)
        pos += nbytes
        bits = np.unpackbits(raw, bitorder="little")[: t * nc * 10].reshape(t * nc, 10)
        codes = (bits.astype(np.uint16) << np.arange(10, dtype=np.uint16)).sum(axis=1)
        segs.append((scale, codes.astype(np.uint16).reshape(t, nc)))
    assert pos == len(data), (pos, len(data))
    return meta, segs


def write(path, meta, segs):
    """Writes a flat ECDC file; meta key order follows the Rust writer (m, al, nc, lm)."""
    fields = {"m": meta["m"], "al": meta["al"], "nc": meta["nc"], "lm": False}
    for key in ("ss", "xf", "ps", "tr", "loop"):
        if key in meta:
            fields[key] = meta[key]
    m = json.dumps(fields, separators=(",", ":")).encode()
    out = bytearray(b"ECDC\x00" + struct.pack(">I", len(m)) + m)
    for scale, codes in segs:
        if scale is not None:
            out += scale
        flat = codes.reshape(-1).astype(np.uint16)
        bits = ((flat[:, None] >> np.arange(10, dtype=np.uint16)) & 1).astype(np.uint8)
        out += np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    open(path, "wb").write(bytes(out))


def truncate(src, dst, nc):
    meta, segs = read(src)
    meta = dict(meta, nc=nc)
    write(dst, meta, [(s, c[:, :nc]) for s, c in segs])
