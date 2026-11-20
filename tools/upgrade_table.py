"""Rewrites a version 4 table as an equivalent self-contained version 5 table.

Usage: upgrade_table.py TAG IN_v4.ecxm OUT_v5.ecxm

Every probability stays the same: the experts, weights and net are copied unchanged. The
fixed-point centroids (computed from the EnCodec weights, as the v4 loader does) go in after
the exponential table, and a net, if any, moves to format 2 with its position input off
(period 0, no buckets). Streams coded with the new table have the same payload bytes as with the old
one; only the table hash they carry differs. Never overwrites IN.
"""
import os
import struct
import sys

import numpy as np

import lib
from build_tables import fixed_centroids

CARD, GEO, REC, PAIR = 1024, 8 * 34, 29 * 16, 8 * 12


class Cursor:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        out = self.data[self.pos:self.pos + n]
        assert len(out) == n, "truncated table"
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]


def skip_layer(c):
    c.take(4 * (CARD + GEO + REC + 3 * PAIR))
    for _ in range(3):  # sparse static tables
        offsets = np.frombuffer(c.take(4 * (CARD + 1)), dtype="<u4")
        c.take(4 * int(offsets[-1]))  # u16 codes + i16 values
    for _ in range(2):  # history experts
        rank = c.u32()
        c.u32()
        c.take(2 * CARD * rank + CARD * rank)


def main():
    tag, src, dst = sys.argv[1:4]
    assert os.path.abspath(src) != os.path.abspath(dst), "never rewrite a table in place"
    assert not os.path.exists(dst), f"{dst} exists; pick a new name"
    data = open(src, "rb").read()
    c = Cursor(data)
    assert c.take(4) == b"ECXM"
    version, layers, fb, _floor = struct.unpack("<4I", c.take(16))
    assert version == 4, f"{src} is version {version}, not 4"
    c.take(4 << fb)
    head = c.pos
    for _ in range(layers):
        skip_layer(c)
    has_net = c.u32()
    out = [b"ECXM", struct.pack("<I", 5), data[8:head],
           fixed_centroids(tag, layers).tobytes(), data[head:c.pos]]
    if has_net:
        h1, _h2, count = struct.unpack("<3I", data[c.pos:c.pos + 12])
        cut = c.pos + 12 + 4 * count + 4 * count * h1  # dims, bandwidths, bandwidth biases
        out += [data[c.pos:cut], struct.pack("<2I", 0, 0), data[cut:]]
    else:
        assert c.pos == len(data), "trailing bytes after the experts"
    blob = b"".join(out)
    with open(dst, "wb") as f:
        f.write(blob)
    print("wrote", dst, len(blob), "bytes from", src, len(data), "bytes", flush=True)


if __name__ == "__main__":
    main()
