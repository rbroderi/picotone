"""Single-thread CPU-time model of an LSTM block of reduced size (no model weights involved).

Measures process CPU time per frame of a 2-layer LSTM step loop for hidden sizes H1/H2 (input and output 512),
with the input projection done as one batched matrix product (as encodec-rs does) and the recurrence done
per step.  The machine is shared, so the minimum over repeats is reported.  The speed ratios, not the
absolute numbers, are what matter.
"""
import time

import numpy as np
import torch

torch.set_num_threads(1)
torch.manual_seed(0)
T = 600


def bench(h1, h2, extra_out=0, reps=5):
    D = 512
    x = torch.randn(T, D)
    w = [(torch.randn(4 * h1, D) * 0.05, torch.randn(4 * h1, h1) * 0.05), (torch.randn(4 * h2, h1) * 0.05, torch.randn(4 * h2, h2) * 0.05)]
    P = torch.randn(D, h2) * 0.05 if extra_out else None
    best = 1e9
    for _ in range(reps):
        t0 = time.process_time()
        inp = x
        for w_ih, w_hh in w:
            H = w_hh.shape[1]
            pre = inp @ w_ih.T
            h = torch.zeros(H)
            c = torch.zeros(H)
            out = torch.empty(T, H)
            for t in range(T):
                z = pre[t] + w_hh @ h
                i, f, g, o = z[:H].sigmoid(), z[H:2 * H].sigmoid(), z[2 * H:3 * H].tanh(), z[3 * H:].sigmoid()
                c = f * c + i * g
                h = o * c.tanh()
                out[t] = h
            inp = out
        if P is not None:
            inp = inp @ P.T
        best = min(best, time.process_time() - t0)
    return best / T * 1e6


base = bench(512, 512)
print(f"H1=H2=512 (stock): {base:.0f} us CPU per frame (torch, 1 thread)")
for h in (496, 480, 448, 416, 384, 320, 256):
    mac = (4 * h * 512 + 4 * h * h + 4 * h * h + 4 * h * h + (512 - h) * h) / (4 * 2048 * 512)
    t = bench(h, h, extra_out=1)
    r = t / base
    # the previous agent measured the LSTM at ~72 % of single-thread CPU time of the 24 kHz model
    print(f"H1=H2={h}: MAC ratio {mac:.3f}, measured LSTM time ratio {r:.3f} -> whole-codec speedup if LSTM is 72% of time: x{1 / (0.28 + 0.72 * r):.2f} (MAC model: x{1 / (0.28 + 0.72 * mac):.2f})")
