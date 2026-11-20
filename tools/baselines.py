"""Makes per-bandwidth test .ecdc files and runs general-purpose compressors on them.

Usage: baselines.py TAG SPLIT K [K ...]   (SPLIT: test -> testsets/, test2 -> testsets2/)
Writes work/testsets/TAG_K<k>/<clip>.ecdc (codebook-prefix truncations of the 24 kbps
encodes, byte-identical to encoding at the lower bandwidth) and work/baselines_TAG.json.
Each compressor is applied to (a) the packed .ecdc file and (b) the codes as u16 LE.
"""
import bz2
import json
import lzma
import os
import sys
import zlib

import numpy as np

import ecdcio
import lib

try:
    from compression import zstd
except ImportError:
    zstd = None

TAG = sys.argv[1]
SPLIT = sys.argv[2]
SETDIR = {"test": "testsets", "test2": "testsets2"}[SPLIT]
KS = [int(a) for a in sys.argv[3:]]

COMPRESSORS = {
    "deflate-9": lambda b: zlib.compress(b, 9),
    "bzip2-9": lambda b: bz2.compress(b, 9),
    "xz-9e": lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME),
}
if zstd:
    COMPRESSORS["zstd-22"] = lambda b: zstd.compress(b, level=22)

out = {}
for kk in KS:
    d = os.path.join(lib.WORK, SETDIR, f"{TAG}_K{kk}")
    os.makedirs(d, exist_ok=True)
    rows = {}
    for i, f in enumerate(lib.load_files(TAG, SPLIT)):
        short = f["name"].split("__", 2)[2].replace(f".{TAG}.ecdc", "")
        short = "".join(c for c in short if c.isalnum() or c in "_-")
        name = f"{i:02d}_{f['group']}_{short[-40:]}.ecdc"
        dst = os.path.join(d, name)
        ecdcio.truncate(f["path"], dst, kk)
        data = open(dst, "rb").read()
        meta, segs = ecdcio.read(dst)
        codes = np.concatenate([c for _, c in segs], axis=0)
        u16 = codes.astype("<u2").tobytes()
        row = {"group": f["group"], "flat_bytes": len(data), "codes": int(codes.size)}
        for cname, fn in COMPRESSORS.items():
            row[cname + "/packed"] = len(fn(data))
            row[cname + "/u16"] = len(fn(u16))
        rows[name] = row
    out[str(kk)] = rows
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values())) if k != "group"}
    print(TAG, "K", kk, "files", len(rows), "flat", tot["flat_bytes"], flush=True)
    for cname in COMPRESSORS:
        for var in ("packed", "u16"):
            v = tot[f"{cname}/{var}"]
            print(f"  {cname + '/' + var:<18} {v:>10} bytes  {8 * v / tot['codes']:.3f} bits/code "
                  f"{100 * (1 - v / tot['flat_bytes']):+.1f}% saved", flush=True)
json.dump(out, open(os.path.join(lib.WORK, f"baselines_{TAG}_{SPLIT}.json"), "w"), indent=1)
