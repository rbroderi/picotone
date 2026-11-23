"""Collects everything train_net.py needs into one archive.

    python collect_data.py --tag 48k --weights <dir with encodec_48khz.safetensors> \
        --fit "<work>/ecdc/fit__*.48k.ecdc" "<work>/ecdc2/fit2__*.48k.ecdc" "<dir>/fit3__*.pico" \
        --test-old "<work>/ecdc/test__*.48k.ecdc" --test-new "<work>/ecdc2/test2__*.48k.ecdc" \
        --out train_data_48k.npz

Inputs are encoded at the model's highest bandwidth (16 codebooks for the 48 kHz model, 32
for the 24 kHz one); lower bandwidths are prefixes of the same codes. Each input is one of:

- a `.pico` file (`picotone encode in.wav out.pico --model encodec-48k -b 24 --beam 1`); its codes are
  read with `picotone dev codes` (the binary is `$PICOTONE_EXE`, else
  `../target/release/picotone[.exe]`, else `picotone` on the PATH);
- a code bundle `.npy` (`[frames, codebooks]`, as `picotone dev codes` writes it);
- a legacy flat `.ecdc` file, the format the existing fit archive (work/ecdc, work/ecdc2)
  was encoded in before the picotone merge; it is read directly.

In every case the codes are the segments' frames concatenated in order, so the archive is
the same whichever container carried them. The archive holds the codes as int16, the stream
lengths, and EnCodec's RVQ centroids in the fixed point the entropy coder uses
(round(x * 128), clamped to +-4095). No audio.
"""
import argparse
import glob
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np

MODELS = {  # tag: (name, normalise, segment, stride, frame rate, sample rate, codebooks, bandwidths)
    "24k": ("encodec_24khz", False, None, None, 75, 24000, 32, (2, 4, 8, 16, 32)),
    "48k": ("encodec_48khz", True, 48000, 47520, 150, 48000, 16, (2, 4, 8, 16)),
}


def picotone_exe():
    env = os.environ.get("PICOTONE_EXE")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("picotone.exe", "picotone"):
        path = os.path.join(here, "..", "target", "release", name)
        if os.path.isfile(path):
            return path
    found = shutil.which("picotone")
    if not found:
        raise SystemExit("picotone not found: build it (cargo build --release) or set PICOTONE_EXE")
    return found


def read_pico(path, tag):
    """Codes [frames, codebooks] of a .pico file, through `picotone dev codes`."""
    with tempfile.TemporaryDirectory() as tmp:
        npy = os.path.join(tmp, "codes.npy")
        r = subprocess.run([picotone_exe(), "dev", "codes", path, npy], capture_output=True, text=True,
                           timeout=600)
        if r.returncode != 0:
            raise ValueError(r.stderr.strip()[-500:])
        with open(os.path.join(tmp, "codes.json"), encoding="utf-8") as f:
            model = json.load(f)["model"]
        if model != MODELS[tag][0]:
            raise ValueError(f"coded with {model}, not {MODELS[tag][0]}")
        return np.load(npy).astype(np.int16)


def read_codes(path, tag):
    """Codes [frames, codebooks] of a .pico file, a code bundle .npy or a flat ECDC file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pico":
        return read_pico(path, tag)
    if ext == ".npy":
        return np.load(path).astype(np.int16)
    return read_ecdc(path, tag)


def read_ecdc(path, tag):
    """Codes [frames, codebooks] of a legacy flat ECDC file (segments concatenated)."""
    _, normalise, seg, stride, rate, sample_rate, _, _ = MODELS[tag]
    data = open(path, "rb").read()
    n = struct.unpack(">I", data[5:9])[0]
    meta = json.loads(data[9:9 + n])
    pos = 9 + n
    al, nc = meta["al"], meta["nc"]
    seg, stride = seg or al, meta.get("ss") or stride or al
    out = []
    for off in range(0, al, stride):
        frames = math.ceil(min(al - off, seg) * rate / sample_rate)
        if normalise:
            pos += 4
        nbytes = (frames * nc * 10 + 7) // 8
        raw = np.frombuffer(data[pos:pos + nbytes], dtype=np.uint8)
        pos += nbytes
        bits = np.unpackbits(raw, bitorder="little")[: frames * nc * 10].reshape(frames * nc, 10)
        codes = (bits.astype(np.uint16) << np.arange(10, dtype=np.uint16)).sum(axis=1)
        out.append(codes.astype(np.int16).reshape(frames, nc))
    if pos != len(data):
        raise ValueError(f"{path}: unexpected length")
    return np.concatenate(out, axis=0)


def centroids(weights_dir, tag):
    name, codebooks = MODELS[tag][0], MODELS[tag][6]
    with open(f"{weights_dir}/{name}.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        out = []
        for k in range(codebooks):
            a, b = header[f"quantizer.layers.{k}.codebook.embed"]["data_offsets"]
            f.seek(8 + n + a)
            out.append(np.frombuffer(f.read(b - a), dtype="<f4").reshape(1024, 128))
    return np.clip(np.rint(np.stack(out).astype(np.float64) * 128.0), -4095, 4095).astype(np.int16)


def load(patterns, tag):
    codes, lengths, names = [], [], []
    kmax = MODELS[tag][6]
    for pattern in patterns or []:
        for path in sorted(glob.glob(pattern)):
            try:
                c = read_codes(path, tag)
            except Exception as e:  # noqa: BLE001
                print("skip", path, e)
                continue
            if c.shape[1] != kmax or len(c) < 8:
                continue
            codes.append(c)
            lengths.append(len(c))
            names.append(path.replace("\\", "/").split("/")[-1])
    if not codes:
        return None
    return np.concatenate(codes), np.array(lengths, dtype=np.int64), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test-old", nargs="*")
    ap.add_argument("--test-new", nargs="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    name, _, _, _, rate, _, _, bandwidths = MODELS[args.tag]
    arrays = {"centroids": centroids(args.weights, args.tag), "bandwidths": np.array(bandwidths),
              "frame_rate": np.array(rate), "model": np.array(name)}
    listing = {}
    for key, patterns in (("fit", args.fit), ("test_old", args.test_old), ("test_new", args.test_new)):
        got = load(patterns, args.tag)
        if got is None:
            continue
        arrays[f"{key}_codes"], arrays[f"{key}_lengths"], listing[key] = got
        print(key, len(got[2]), "files,", round(len(got[0]) / rate / 3600, 2), "hours")
    arrays["files"] = np.array(json.dumps(listing))
    np.savez_compressed(args.out, **arrays)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
