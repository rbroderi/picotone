"""Internal check: picotone's EnCodec encoder and decoder against Hugging Face's EncodecModel.

Usage: uv run --project tools --extra parity python tools/parity.py PICOTONE_EXE WEIGHTS_DIR WORK_DIR

For each model it writes a test signal, encodes it with `picotone encode --beam 1` (the
reference's greedy residual quantiser), dumps the codes with `picotone dev codes`, and runs
the reference on the same segments:

- codes: the reference's codes must equal picotone's (100%). picotone quantises each
  segment's scale to the 12-bit log grid of the .pico format before normalising, so the
  reference divides by that same dumped scale; the line also prints how far the grid scale
  is from the reference's float scale (at most 2^(1/256) - 1 = 0.27%).
- decoder: the reference decodes picotone's codes (triangle cross-fade, EnCodec's own);
  picotone's float output must match it closely.
"""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import wave

import numpy as np
import torch
from transformers import EncodecModel

exe, weights, work = sys.argv[1:4]
os.makedirs(work, exist_ok=True)
torch.set_grad_enabled(False)


def signal(rate, seconds, channels, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    out = []
    for c in range(channels):
        x = 0.3 * np.sin(2 * np.pi * (220 + 110 * c) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
        x += 0.15 * np.sin(2 * np.pi * (300 + 900 * t) * t)  # chirp
        x += 0.05 * rng.standard_normal(t.shape) * (np.sin(2 * np.pi * 3 * t) > 0)
        out.append(x)
    return np.clip(np.stack(out), -1, 1)


def write_wav16(path, x, rate):
    pcm = np.round(x.T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return pcm.T.astype(np.float32) / 32768.0


def read_wav_float(path):
    data = open(path, "rb").read()
    pos, fmt = 12, None
    while pos < len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            return np.frombuffer(body, "<f4").reshape(-1, fmt[1]).T
        pos += 8 + size + (size & 1)
    raise SystemExit(f"{path} has no data chunk")


def read_codes(pico, npy):
    """Codes of a .pico file as a list of ([codebooks, frames] codes, scale or None)."""
    subprocess.run([exe, "--weights", weights, "dev", "codes", pico, npy], check=True)
    codes = np.load(npy)
    side = json.load(open(os.path.splitext(npy)[0] + ".json"))
    out, start = [], 0
    scales = side.get("scales") or [None] * len(side["segment_frames"])
    for frames, scale in zip(side["segment_frames"], scales):
        out.append((codes[start:start + frames].T.astype(np.int64), scale))
        start += frames
    return side, out


def overlap_add(frames, stride):
    total = stride * (len(frames) - 1) + frames[-1].shape[-1]
    fl = frames[0].shape[-1]
    t = torch.linspace(0, 1, fl + 2, dtype=torch.float64)[1:-1]
    w = 0.5 - (t - 0.5).abs()
    out = torch.zeros(frames[0].shape[0], total, dtype=torch.float64)
    sw = torch.zeros(total, dtype=torch.float64)
    off = 0
    for f in frames:
        n = f.shape[-1]
        out[:, off:off + n] += w[:n] * f.double()
        sw[off:off + n] += w[:n]
        off += stride
    return out / sw


def snr(ref, x):
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-30))


failed = False
for name, rate, ch, seconds, bw in [("encodec_24khz", 24000, 1, 4.3, 6.0),
                                    ("encodec_48khz", 48000, 2, 3.7, 12.0)]:
    mdir = os.path.join(work, name)
    os.makedirs(mdir, exist_ok=True)
    for src, dst in [(f"{name}.json", "config.json"), (f"{name}.safetensors", "model.safetensors")]:
        if not os.path.exists(os.path.join(mdir, dst)):
            shutil.copy(os.path.join(weights, src), os.path.join(mdir, dst))
    model = EncodecModel.from_pretrained(mdir).eval()
    cfg = model.config
    normalize = cfg.normalize
    seg = int(cfg.chunk_length_s * rate) if cfg.chunk_length_s else None

    wav_in = os.path.join(work, f"{name}_in.wav")
    x = write_wav16(wav_in, signal(rate, seconds, ch, 1), rate)
    pico = os.path.join(work, f"{name}.pico")
    wav_out = os.path.join(work, f"{name}_out.wav")
    subprocess.run([exe, "--weights", weights, "encode", wav_in, pico, "-m", "mono" if ch == 1 else "stereo",
                    "-b", str(bw), "--beam", "1", "--crossfade", "triangle"], check=True)
    subprocess.run([exe, "--weights", weights, "decode", pico, wav_out, "--float"], check=True)
    side, ours = read_codes(pico, os.path.join(work, f"{name}.npy"))
    stride = side.get("stride")

    # Reference encode on the same segments, divided by picotone's (grid) scale.
    xt = torch.from_numpy(x)[None]
    length = xt.shape[-1]
    seg_l, stride_l = seg or length, stride or length
    agree = total = 0
    scale_err = 0.0
    for i, off in enumerate(range(0, length, stride_l)):
        chunk = xt[..., off:off + seg_l]
        oc, osc = ours[i]
        if normalize:
            mono = chunk.mean(dim=1, keepdim=True)
            exact = (mono.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-8).item()
            scale_err = max(scale_err, abs(osc - exact) / exact)
            chunk = chunk / torch.tensor(osc, dtype=torch.float32)
        emb = model.encoder(chunk)
        codes = model.quantizer.encode(emb, bw)[:, 0]  # [nq, T]
        assert oc.shape == tuple(codes.shape), (oc.shape, codes.shape)
        agree += int((oc == codes.numpy()).sum())
        total += oc.size
    assert len(ours) == i + 1, (len(ours), i + 1)
    print(f"{name}: codes agree {agree}/{total} ({100 * agree / total:.3f}%), "
          f"grid scale vs float scale at most {scale_err:.2e}")
    failed |= agree != total

    # Reference decode of picotone's codes.
    dec = []
    for oc, osc in ours:
        out = model.decoder(model.quantizer.decode(torch.from_numpy(oc)[:, None].long()))[0]
        if osc is not None:
            out = out * osc
        dec.append(out)
    ref = (dec[0].double() if len(dec) == 1 else overlap_add(dec, stride_l))[:, :length].numpy()
    got = read_wav_float(wav_out)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    clipped = np.clip(ref, -0.99, 0.99)
    print(f"{name}: decoder SNR vs reference {snr(clipped, got):.1f} dB; codec SNR vs input {snr(x, got):.1f} dB")
sys.exit(1 if failed else 0)
