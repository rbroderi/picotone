# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-weights parity of the Rust STFT encoder (`src/stftenc.rs`) with PyTorch, in latent SNR and in DAC codes.
Run in distill/'s environment, in three steps:

    cd distill
    uv run --extra cpu python ../tools/stftenc_parity.py prepare --ckpt X:/audio/distill/runs/encB_arm4/ckpt_end_20000.pt \
        --out ../work/stftenc X:/audio/distill/listen_encoder/arm4_20k/*original*.wav
    cd .. && cargo test --release --test hq_stftenc -- --ignored real_weights_parity_files
    cd distill && uv run --extra cpu python ../tools/stftenc_parity.py compare --out ../work/stftenc \
        --dac export/weights_44khz_8kbps_0.0.1.pth

prepare: per WAV (16-bit PCM, mono-downmixed, the first --seconds, cut to a whole number of 512-sample frames),
writes <out>/<clip>.in.f32 (samples), <clip>.torch.f32 (PyTorch fp32 z, [frames, 1024] time-major) and
<clip>.torch64.f32 (the float64 model, export_stftenc.forward64). The Rust test writes <clip>.rust.f32.

compare: SNRs, and the codes of each latent through DAC's quantiser (encoder.FrozenRVQ, fp32) at --n-q
codebooks: per codebook, the fraction of frames whose code equals PyTorch fp32's. For each frame's first
disagreeing codebook, the cosine gap between PyTorch's best and second-best entries there is reported (a
near-tie when tiny). PyTorch fp32 against float64 gives the floor any fp32 implementation can reach.
"""
import argparse
import glob
import os
import sys
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from export_stftenc import forward64, load_encoder  # noqa: E402  (puts distill/ on the path)

FRAME = 512


def read_wav(path, seconds):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, f"{path}: not 16-bit PCM"
        sr, ch = w.getframerate(), w.getnchannels()
        assert sr == 44100, f"{path}: {sr} Hz"
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, ch).astype(np.float32) / 32768
    x = x.mean(1).astype(np.float32)[: int(seconds * sr)]
    return x[: len(x) // FRAME * FRAME]


def snr(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return 10 * np.log10((want ** 2).sum() / max(((got - want) ** 2).sum(), 1e-300))


def prepare(a):
    torch.set_num_threads(a.threads)
    enc, _ = load_encoder(a.ckpt)
    os.makedirs(a.out, exist_ok=True)
    for path in a.wavs:
        name = os.path.splitext(os.path.basename(path))[0]
        x = read_wav(path, a.seconds)
        stem = os.path.join(a.out, name)
        x.astype("<f4").tofile(stem + ".in.f32")
        xt = torch.from_numpy(x)[None, None]
        with torch.no_grad():
            z = enc(xt)[0].T.contiguous().numpy()
            z64 = forward64(enc, xt)[0].T.contiguous().numpy()
        z.astype("<f4").tofile(stem + ".torch.f32")
        z64.astype("<f4").tofile(stem + ".torch64.f32")
        print(f"{name}: {len(x) / 44100:.1f} s, {z.shape[0]} frames; PyTorch fp32 vs fp64 SNR {snr(z, z64):.1f} dB")


def load_rvq(path):
    """DAC 44 kHz's quantiser from its weights file, weight norm removed as in teacher.load_teacher."""
    import warnings

    warnings.filterwarnings("ignore")
    import dac
    from torch.nn.utils import parametrize

    from encoder import FrozenRVQ

    model = dac.DAC.load(path)
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
        elif parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return FrozenRVQ(model.quantizer).eval()


@torch.no_grad()
def codes_of(rvq, z, n_q):
    """z [frames, 1024] -> codes [n_q, frames]."""
    _, codes, _ = rvq(torch.from_numpy(np.ascontiguousarray(z.T))[None].float(), n_q)
    return codes[0].numpy()


@torch.no_grad()
def gaps(rvq, z, codes, n_q):
    """Per codebook and frame, the cosine gap between the best and second-best entries for z's own residuals."""
    r = torch.from_numpy(np.ascontiguousarray(z.T))[None].float()
    out = []
    for i in range(n_q):
        qz = rvq.q.quantizers[i]
        z_e = qz.in_proj(r)
        cb = torch.nn.functional.normalize(qz.codebook.weight.float(), dim=1)
        cos = torch.einsum("bdf,nd->bfn", torch.nn.functional.normalize(z_e, dim=1), cb)[0]
        top = cos.topk(2, -1).values
        out.append((top[:, 0] - top[:, 1]).numpy())
        idx = torch.from_numpy(codes[i])[None]
        r = r - qz.out_proj(rvq.entry(i, idx))
    return np.stack(out)


def compare(a):
    torch.set_num_threads(a.threads)
    rvq = load_rvq(a.dac)
    total = {"rust": [], "torch64": []}
    for stem in sorted(glob.glob(os.path.join(a.out, "*.in.f32"))):
        stem = stem[: -len(".in.f32")]
        name = os.path.basename(stem)
        z = {k: np.fromfile(f"{stem}.{k}.f32", "<f4").reshape(-1, 1024) for k in ("torch", "torch64", "rust")}
        ref = codes_of(rvq, z["torch"], a.n_q)
        g = gaps(rvq, z["torch"], ref, a.n_q)
        print(f"{name}: {ref.shape[1]} frames; Rust vs PyTorch fp32 {snr(z['rust'], z['torch']):.1f} dB, "
              f"Rust vs fp64 {snr(z['rust'], z['torch64']):.1f} dB, PyTorch fp32 vs fp64 "
              f"{snr(z['torch'], z['torch64']):.1f} dB")
        for k in ("rust", "torch64"):
            c = codes_of(rvq, z[k], a.n_q)
            eq = c == ref
            total[k].append(eq)
            first = [(int(np.argmin(eq[:, f])), f) for f in range(eq.shape[1]) if not eq[:, f].all()]
            print(f"  {k:8s} vs PyTorch fp32 codes, per codebook: "
                  + " ".join(f"{v * 100:.2f}" for v in eq.mean(1)) + " %")
            if first:
                print(f"  {k:8s} first flips (codebook, frame, PyTorch's top-2 cosine gap): "
                      + ", ".join(f"({i}, {f}, {g[i, f]:.1e})" for i, f in first[:12])
                      + (" ..." if len(first) > 12 else ""))
    for k, eqs in total.items():
        eq = np.concatenate(eqs, 1)
        print(f"all clips, {k} vs PyTorch fp32: {eq.shape[1]} frames, identical codes per codebook "
              + " ".join(f"{v * 100:.2f}" for v in eq.mean(1)) + f" %; all {a.n_q} codebooks identical in "
              f"{eq.all(0).mean() * 100:.2f}% of frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("wavs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--dac", required=True, help="DAC 44 kHz weights (weights_44khz_8kbps_0.0.1.pth)")
    c.add_argument("--n-q", type=int, default=9)
    c.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else compare(a)


if __name__ == "__main__":
    main()
