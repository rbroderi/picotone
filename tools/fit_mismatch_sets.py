"""Prepares the fit-mismatch comparison: the same fit clips as plain and as music24 encodes.

Usage: fit_mismatch_sets.py SCRATCH
workB/ecdc holds music24 encodes (15 codebooks, stride 45000, beam 8) of the clips listed in
work/abfit.txt. This writes workA/ecdc with the plain `-b 24` encodes (stride 47520, greedy)
of the same clips cut to 15 codebooks, and moves clips missing on either side out of the way.
"""
import os
import sys

import ecdcio

S = sys.argv[1]
a_dir, b_dir = os.path.join(S, "workA", "ecdc"), os.path.join(S, "workB", "ecdc")
extra = os.path.join(S, "workB", "extra")
os.makedirs(a_dir, exist_ok=True)
os.makedirs(extra, exist_ok=True)
kept = 0
for line in open(os.path.join(S, "work", "abfit.txt"), encoding="utf-8"):
    wav, out, _ = line.rstrip("\n").split("\t")
    name = os.path.basename(wav)[:-4]
    plain = os.path.join(S, "work", "ecdc2", name + ".48k.ecdc")
    b = os.path.join(b_dir, os.path.basename(out))
    if os.path.exists(plain) and os.path.exists(b):
        dst = os.path.join(a_dir, os.path.basename(out))
        if not os.path.exists(dst):
            ecdcio.truncate(plain, dst, 15)
        kept += 1
    elif os.path.exists(b):
        os.replace(b, os.path.join(extra, os.path.basename(b)))
print(kept, "clips on both sides")
