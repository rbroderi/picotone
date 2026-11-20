"""Single-thread microbenchmark of the LSTM's two matrix products at hidden size 512 vs smaller.

usage: python s13_matvec_bench.py

* recurrent step  : y = W_hh @ h           (4H x H) @ (H)          -- sequential, once per frame per layer
* input projection: P = X @ W_ih^T         (T x D_in) @ (D_in x 4H) -- batched over the T frames of a segment
Wall time of the fastest of many repeats, 1 BLAS thread (the machine is shared; the minimum filters out contention).  numpy float32.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time

import numpy as np

rng = np.random.default_rng(0)
T = 150


def best(fn, reps=25):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter_ns()
        fn()
        b = min(b, time.perf_counter_ns() - t)
    return b / 1e9


def recurrent(H, n=3000):
    W = rng.standard_normal((4 * H, H), dtype=np.float32)
    h = rng.standard_normal(H, dtype=np.float32)

    def run():
        for _ in range(n):
            W @ h
    return best(run) / n


def recurrent_rows(H, n=300):
    """Row-by-row dot products, the access pattern of encodec-rs's LstmLayer::step (one unit's four gate rows at a time)."""
    W = rng.standard_normal((4 * H, H), dtype=np.float32)
    h = rng.standard_normal(H, dtype=np.float32)
    Wb = W.reshape(H, 4, H)

    def run():
        for _ in range(n):
            for u in range(0, H, 16):          # 16 units per call keeps Python overhead out of the measurement
                Wb[u:u + 16].reshape(-1, H) @ h
    return best(run, 7) / n


def projection(d_in, H, n=20):
    W = rng.standard_normal((4 * H, d_in), dtype=np.float32)
    X = rng.standard_normal((T, d_in), dtype=np.float32)

    def run():
        for _ in range(n):
            X @ W.T
    return best(run) / n / T


base = None
print("H    | W_hh@h us/frame (ratio) | row-blocked us (ratio) | layer-1 proj 512->4H us/frame | layer-2 proj H->4H us/frame | whole 2-layer LSTM us/frame (ratio) | MAC ratio")
for H in (512, 448, 384, 320):
    r = recurrent(H) * 1e6
    rr = recurrent_rows(H) * 1e6
    p1 = projection(512, H) * 1e6
    p2 = projection(H, H) * 1e6
    out = 0.0 if H == 512 else projection(H, 512 - H) / 4 * 1e6   # dense read-out rows for the dropped units ((512-H) x H)
    total = 2 * r + p1 + p2 + out
    mac = (4 * H * 512 + 3 * 4 * H * H + (512 - H) * H) / (4 * 2048 * 512)
    if base is None:
        base = (r, rr, total)
    print(f"{H:4d} | {r:7.1f} ({r / base[0]:.3f}) | {rr:7.1f} ({rr / base[1]:.3f}) | {p1:6.1f} | {p2:6.1f} | {total:7.1f} ({total / base[2]:.3f}) | {mac:.3f}")
