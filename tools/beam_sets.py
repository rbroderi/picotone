"""Greedy (beam 1) fit and test sets at 15, 8 and 4 codebooks from the 15-codebook encodes.

Usage: beam_sets.py SCRATCH
Reads workG/ecdc (fit clips) and work/preset_b1/music24/{old,new} (test clips), all coded
with `--preset music24 --beam 1`; a greedy encode with fewer codebooks is a prefix of it
(checked against work/preset_b1/prefixcheck). Writes workG<K>/ecdc and
work/ps_b1_<preset>_{old,new}/48k_K<K>/ with the names preset_sets.py uses.
"""
import glob
import os
import sys

import ecdcio

S = sys.argv[1]
PRESETS = {15: "music24", 8: "music12", 4: "music6"}
for check in glob.glob(os.path.join(S, "work", "preset_b1", "prefixcheck", "*.ecdc")):
    full = os.path.join(S, "work", "preset_b1", "music24", "old", os.path.basename(check))
    tmp = check + ".cut"
    ecdcio.truncate(full, tmp, 8)
    same = open(tmp, "rb").read() == open(check, "rb").read()
    os.remove(tmp)
    print("prefix property holds:", same)
    assert same
for k, preset in PRESETS.items():
    dst = os.path.join(S, f"workG{k}", "ecdc")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(S, "workG", "ecdc", "*.ecdc"))):
        ecdcio.truncate(f, os.path.join(dst, os.path.basename(f)), k)
        n += 1
    for label in ("old", "new"):
        d = os.path.join(S, "work", f"ps_b1_{preset}_{label}", f"48k_K{k}")
        os.makedirs(d, exist_ok=True)
        files = sorted(glob.glob(os.path.join(S, "work", "preset_b1", "music24", label, "*.ecdc")))
        for i, f in enumerate(files):
            group = os.path.basename(f).split("__")[1]
            ecdcio.truncate(f, os.path.join(d, f"{i:02d}_{group}_{os.path.basename(f)[-44:]}"), k)
        print(preset, label, len(files), "test files;", n, "fit clips")
