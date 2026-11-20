"""Test sets from files encoded with the shipping presets, plus generic-compressor baselines.

Usage: preset_sets.py PRESET K
Copies work/preset/<PRESET>/{old,new}/*.ecdc (already K codebooks, stride 45000, beam 8) to
work/ps_<PRESET>_{old,new}/48k_K<K>/ under short names and prints the best general-purpose
compressor result per set (deflate-9, bzip2-9, xz-9e on the packed file and on u16 codes).
"""
import bz2
import glob
import json
import lzma
import os
import shutil
import sys
import zlib

import numpy as np

import ecdcio
import lib

PRESET, K = sys.argv[1], int(sys.argv[2])
out = {}
for label in ("old", "new"):
    files = sorted(glob.glob(os.path.join(lib.WORK, "preset", PRESET, label, "*.ecdc")))
    d = os.path.join(lib.WORK, f"ps_{PRESET}_{label}", f"48k_K{K}")
    os.makedirs(d, exist_ok=True)
    flat = codes_n = 0
    best = {}
    for i, f in enumerate(files):
        meta, segs = ecdcio.read(f)
        assert meta["nc"] == K, (f, meta)
        group = os.path.basename(f).split("__")[1]
        shutil.copyfile(f, os.path.join(d, f"{i:02d}_{group}_{os.path.basename(f)[-44:]}"))
        data = open(f, "rb").read()
        u16 = np.concatenate([c for _, c in segs]).astype("<u2").tobytes()
        flat += len(data)
        codes_n += len(u16) // 2
        for name, fn in (("deflate", lambda b: zlib.compress(b, 9)), ("bzip2", lambda b: bz2.compress(b, 9)),
                         ("xz", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))):
            for var, payload in (("packed", data), ("u16", u16)):
                best[f"{name}/{var}"] = best.get(f"{name}/{var}", 0) + len(fn(payload))
    name, size = min(best.items(), key=lambda kv: kv[1])
    out[label] = {"files": len(files), "flat": flat, "codes": codes_n, "best_generic": name,
                  "best_generic_bytes": size, "header": {k: v for k, v in meta.items() if k != "al"}}
    print(f"{PRESET} {label}: {len(files)} files, flat {flat} bytes ({8 * flat / codes_n:.3f} bits/code), "
          f"best generic {name} {100 * (1 - size / flat):.1f}% saved; header {out[label]['header']}", flush=True)
json.dump(out, open(os.path.join(lib.WORK, f"preset_{PRESET}_baselines.json"), "w"), indent=1)
