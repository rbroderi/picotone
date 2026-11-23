"""End-to-end check of `picotone encode`/`decode` (HQ) against the PyTorch pipeline, per stereo layout.

    uv run --extra cpu python ref_stereo.py --input clip.wav --student export/dac44_student_w512_172500.pt \
        --dac export/weights_44khz_8kbps_0.0.1.pth --picotone ../target/release/picotone.exe \
        --weights ../weights --out ../work/hq_stereo_parity

The input is resampled to 44.1 kHz here first, so picotone codes the same samples (it resamples only when the
rate differs). For each layout (left/right 9+9, mid/side 9+9 and 7+3) the reference codes each signal with DAC's
encoder and quantiser, decodes it with the PyTorch student, and rebuilds L and R; picotone encodes the WAV
with the same flags and decodes it. Printed: the largest |diff| between the two decodes, and the log-mel of
each against the input. They should match to rounding, apart from rare near-tie code flips.
"""
import argparse
import os
import subprocess
from math import gcd

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

import metrics
from export_rust import load_quantiser
from ref_decode import SR, reference
from student import StudentDecoder, fold_weight_norm


def read(path):
    sr, x = wavfile.read(path)
    x = (x.astype(np.float32) / 32768) if x.dtype == np.int16 else x.astype(np.float32)
    x = x.T if x.ndim == 2 else x[None]
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)
    return x


@torch.no_grad()
def codes_of(model, signal):
    x = model.preprocess(torch.from_numpy(signal)[None, None], SR)
    _, codes, *_ = model.quantizer(model.encoder(x), n_quantizers=9)
    return codes[0].T.numpy().astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--input", "--student", "--dac", "--picotone", "--weights", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore")
    import dac
    model = dac.DAC.load(a.dac).eval()
    G = StudentDecoder(512)
    fold_weight_norm(G)
    G.load_state_dict(torch.load(a.student, map_location="cpu", weights_only=True))
    G.eval()
    books, proj, bias = load_quantiser(a.dac)

    x = read(a.input)[:2]
    n = x.shape[1]
    src = os.path.join(a.out, "input44k.wav")
    wavfile.write(src, SR, x.T)
    for name, flags, books_n in (("lr99", ["--stereo", "lr"], (9, 9)),
                                 ("ms99", ["--stereo", "ms"], (9, 9)),
                                 ("ms73", ["--stereo", "ms", "--codebooks", "7", "--side-codebooks", "3"], (7, 3))):
        ms = name.startswith("ms")
        sig = [(x[0] + x[1]) / 2, (x[0] - x[1]) / 2] if ms else [x[0], x[1]]
        dec = [reference(G, books, proj, bias, codes_of(model, s), q)[:n] for s, q in zip(sig, books_n)]
        ref = np.stack([dec[0] + dec[1], dec[0] - dec[1]]) if ms else np.stack(dec)
        pico, wav = os.path.join(a.out, name + ".pico"), os.path.join(a.out, name + ".wav")
        env = dict(os.environ, RUST_BACKTRACE="0")
        subprocess.run([a.picotone, "--weights", a.weights, "encode", src, pico, *flags], check=True,
                       capture_output=True, timeout=900, env=env)
        subprocess.run([a.picotone, "--weights", a.weights, "decode", pico, wav, "--float"], check=True,
                       capture_output=True, timeout=900, env=env)
        got = read(wav)
        d = float(np.abs(got - ref).max())
        lm = lambda y: np.mean([metrics.score(x[c:c + 1], y[c:c + 1])["logmel"] for c in range(2)])  # noqa: E731
        print(f"{name}: max |picotone - reference| {d:.2e}; log-mel picotone {lm(got):.3f}, reference {lm(ref):.3f}")


if __name__ == "__main__":
    main()
