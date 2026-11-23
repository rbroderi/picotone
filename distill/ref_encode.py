"""Parity check: the Rust HQ encoder against DAC's own PyTorch encoder and quantiser (milestone M6).

    uv run --extra cpu python ref_encode.py --dac export/weights_44khz_8kbps_0.0.1.pth \
        --rust-weights export/rust/dac44_encoder.safetensors \
        --picotone ../target/release/picotone.exe --out ../work/dac_enc_parity

Two signals, neither a whole number of frames long, so the end padding is exercised too:
- **music:** ref_decode.py's synthetic 8 s signal (chords, a bass line, noise bursts and clicks).
- **noise:** 6 s of pink-ish noise with a slow level sweep.

PyTorch pads the end to a whole frame (DAC's preprocess), then runs model.encoder and model.quantizer.
`picotone dev dac-encode` encodes the same samples in its default chunks and in 16-frame chunks. Reported:
the latent's largest |diff| relative to its largest value, and the share of codes that agree per codebook.

Codes are nearest-entry choices, so a latent that differs by rounding can flip a near tie, and a flip changes
the residual that later codebooks see. The check passes when the latent's relative error is under --tol and
at least --agree of the first codebook's codes match (the other codebooks are reported).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

from ref_decode import HOP, SR, synth


def noise(seconds=6.0, seed=2):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    w = rng.normal(0, 1, n)
    y = np.cumsum(w) * 0.02
    y = y - np.convolve(y, np.ones(400) / 400, mode="same") + 0.3 * w
    y *= 0.1 + 0.9 * np.linspace(0, 1, n) ** 2
    return (y / np.abs(y).max() * 0.6).astype(np.float32)


@torch.no_grad()
def reference(model, audio):
    x = model.preprocess(torch.from_numpy(audio)[None, None], SR)
    z = model.encoder(x)
    _, codes, *_ = model.quantizer(z, n_quantizers=9)
    return z[0].T.numpy(), codes[0].T.numpy().astype(np.int64)   # [F, 1024], [F, 9]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dac", required=True)
    ap.add_argument("--rust-weights", required=True)
    ap.add_argument("--picotone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--agree", type=float, default=0.995)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore")
    import dac
    model = dac.DAC.load(a.dac).eval()

    ok = True
    for name, audio in {"music": synth()[: 8 * SR - 123], "noise": noise()}.items():
        ref_z, ref_codes = reference(model, audio)
        raw = os.path.join(a.out, f"{name}.f32")
        audio.astype("<f4").tofile(raw)
        for chunk in (None, 16):
            tag = str(chunk or "default")
            cpath = os.path.join(a.out, f"{name}_{tag}_codes.npy")
            zpath = os.path.join(a.out, f"{name}_{tag}_latent.f32")
            cmd = [a.picotone, "dev", "dac-encode", a.rust_weights, raw, cpath, "--latent", zpath]
            if chunk:
                cmd += ["--chunk", str(chunk)]
            r = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=900)
            got_z = np.fromfile(zpath, dtype="<f4").reshape(-1, 1024)
            got_codes = np.load(cpath).astype(np.int64)
            if got_z.shape != ref_z.shape or got_codes.shape != ref_codes.shape:
                raise SystemExit(f"{name} chunk {tag}: shapes {got_z.shape} {got_codes.shape}, "
                                 f"expected {ref_z.shape} {ref_codes.shape}")
            rel = float(np.abs(got_z - ref_z).max() / np.abs(ref_z).max())
            agree = (got_codes == ref_codes).mean(axis=0)
            ok &= rel <= a.tol and agree[0] >= a.agree
            print(f"{name} chunk {tag:7s}: latent rel max |diff| {rel:.2e}; codes agree per codebook "
                  + " ".join(f"{v:.3f}" for v in agree) + f"   [{r.stderr.strip().splitlines()[-1]}]")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
