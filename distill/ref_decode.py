"""Parity check: the Rust HQ decoder against the PyTorch student (milestone M0/M1 in INTEGRATION.md).

    uv run --extra cpu python ref_decode.py --student export/dac44_student_w512_172500.pt \
        --dac export/weights_44khz_8kbps_0.0.1.pth --rust-weights export/rust/dac44_student.safetensors \
        --picotone ../target/release/picotone.exe --out ../work/dac_parity

Two code sets:
- **random:** 700 frames (8.1 s) of uniform random codes. Odd audio, but every code path runs.
- **music:** a synthetic 8 s signal (chords, a bass line, noise bursts and clicks), encoded by DAC's own
  encoder and quantiser.

For each set and each tier in --tiers, the PyTorch reference decodes the whole file in one go through the
unfolded path (DAC's quantiser, then the student). `picotone dev dac-decode` decodes the same codes in its
default chunks, and also in 64-frame chunks to stress the chunk joins. The check passes when every max |diff|
is under --tol (default 1e-4).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

from export_rust import CB_SIZE, N_CB, load_quantiser
from student import StudentDecoder, fold_weight_norm

SR, HOP = 44100, 512


def synth(seconds=8.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = np.zeros_like(t)
    for i, root in enumerate([220.0, 261.6, 196.0, 174.6]):
        seg = (t >= i * 2) & (t < (i + 1) * 2)
        for ratio in (1, 1.25, 1.5, 2):
            y[seg] += 0.08 * np.sin(2 * np.pi * root * ratio * t[seg]) * np.exp(-((t[seg] - i * 2) % 0.5) * 3)
        y[seg] += 0.15 * np.sin(2 * np.pi * root / 4 * t[seg])
    for k in range(16):
        s = int((k * 0.5 + 0.25) * SR)
        y[s:s + 2000] += rng.normal(0, 0.2, 2000) * np.exp(-np.arange(2000) / 300)
        y[s] += 0.6
    y = y / np.abs(y).max() * 0.7
    return y.astype(np.float32)


@torch.no_grad()
def dac_codes(dac_path, audio):
    import warnings
    warnings.filterwarnings("ignore")
    import dac
    model = dac.DAC.load(dac_path).eval()
    x = torch.from_numpy(audio)[None, None]
    x = x[..., : x.shape[-1] // HOP * HOP]
    z = model.encoder(x)
    _, codes, *_ = model.quantizer(z, n_quantizers=N_CB)
    return np.ascontiguousarray(codes[0].T.numpy().astype(np.uint16))   # [F, 9], row-major for the .npy


@torch.no_grad()
def reference(G, books, proj, bias, codes, n_q):
    c = torch.from_numpy(codes.astype(np.int64)).T       # [9, F]
    z = sum(proj[i] @ books[i][c[i]].T + bias[i][:, None] for i in range(n_q))
    return G(z[None])[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--dac", required=True)
    ap.add_argument("--rust-weights", required=True)
    ap.add_argument("--picotone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="1,4,9")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    G = StudentDecoder(512)
    fold_weight_norm(G)
    G.load_state_dict(torch.load(a.student, map_location="cpu", weights_only=True))
    G.eval()
    books, proj, bias = load_quantiser(a.dac)

    rng = np.random.default_rng(0)
    sets = {"random": rng.integers(0, CB_SIZE, (700, N_CB)).astype(np.uint16),
            "music": dac_codes(a.dac, synth())}
    worst = 0.0
    for name, codes in sets.items():
        cpath = os.path.join(a.out, f"{name}_codes.npy")
        np.save(cpath, codes)
        for n_q in (int(t) for t in a.tiers.split(",")):
            ref = reference(G, books, proj, bias, codes, n_q)
            np.save(os.path.join(a.out, f"{name}_q{n_q}_ref.npy"), ref)
            for chunk in (None, 64):
                raw = os.path.join(a.out, f"{name}_q{n_q}_{chunk or 'default'}.f32")
                cmd = [a.picotone, "dev", "dac-decode", a.rust_weights, cpath, raw, "--codebooks", str(n_q)]
                if chunk:
                    cmd += ["--chunk", str(chunk)]
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)
                got = np.fromfile(raw, dtype="<f4")
                if got.shape != ref.shape:
                    raise SystemExit(f"{name} q{n_q} chunk {chunk}: {got.shape} samples, expected {ref.shape}")
                d = float(np.abs(got - ref).max())
                snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((got - ref) ** 2), 1e-30))
                worst = max(worst, d)
                print(f"{name:6s} q{n_q} chunk {str(chunk or 'default'):7s}: max |diff| {d:.2e}, SNR {snr:6.1f} dB, "
                      f"rms {np.sqrt(np.mean(ref ** 2)):.3f}")
    print(f"worst max |diff| {worst:.2e} (tolerance {a.tol:.0e}):", "PASS" if worst <= a.tol else "FAIL")
    sys.exit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()
