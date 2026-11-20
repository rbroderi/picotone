"""Builds 48 kHz test files with a custom segment stride (the dwp seam fix).

Usage: stride_sets.py WORK EXE WEIGHTS SPLIT STRIDE WORKER NWORKERS [MAXCLIPS]
The released encoder only knows the stock stride (47520). Segments are coded independently,
so each wanted segment (48000 samples starting every STRIDE samples) is placed alone in
every second stock segment of a scratch WAV, encoded at 24 kbps, and its scale and codes
are then collected. Output: WORK/stride<STRIDE>/<split>/<clip>.ecdc with all 16 codebooks and
"ss" / "xf" in the header; lower codebook counts are prefixes of it.
"""
import os
import subprocess
import sys
import wave

import numpy as np

import ecdcio
import lib

WORK, EXE, WEIGHTS, SPLIT = sys.argv[1:5]
STRIDE, WORKER, NWORKERS = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
MAXCLIPS = int(sys.argv[8]) if len(sys.argv) > 8 else 10 ** 9
SEG, STOCK = 48000, 47520
out_dir = os.path.join(WORK, f"stride{STRIDE}", SPLIT)
tmp_dir = os.path.join(WORK, f"stride{STRIDE}", "tmp")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
files = lib.load_files("48k", SPLIT)
if len(files) > MAXCLIPS:
    files = files[::max(len(files) // MAXCLIPS, 1)][:MAXCLIPS]
for index, f in enumerate(files):
    if index % NWORKERS != WORKER:
        continue
    name = f["name"].replace(".48k.ecdc", "")
    dst = os.path.join(out_dir, name + ".ecdc")
    if os.path.exists(dst):
        continue
    wav = os.path.join(WORK, "wav2" if SPLIT == "test2" else "wav", name + ".wav")
    with wave.open(wav, "rb") as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 2 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)
    al = len(pcm)
    offsets = list(range(0, al, STRIDE))
    lengths = [min(SEG, al - o) for o in offsets]
    total = 2 * (len(offsets) - 1) * STOCK + lengths[-1]
    scratch = np.zeros((total, 2), dtype="<i2")
    for j, (o, n) in enumerate(zip(offsets, lengths)):
        scratch[2 * j * STOCK:2 * j * STOCK + n] = pcm[o:o + n]
    tmp_wav = os.path.join(tmp_dir, f"w{WORKER}.wav")
    tmp_ecdc = os.path.join(tmp_dir, f"w{WORKER}.ecdc")
    with wave.open(tmp_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(scratch.tobytes())
    r = subprocess.run([EXE, "--weights", WEIGHTS, "--threads", "2", "encode", tmp_wav, tmp_ecdc,
                        "-m", "stereo", "-b", "24"], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED", name, r.stderr[-200:], flush=True)
        continue
    _, segs = ecdcio.read(tmp_ecdc)
    picked = [segs[2 * j] for j in range(len(offsets))]
    meta = {"m": "encodec_48khz", "al": al, "nc": 16, "ss": STRIDE, "xf": 1}
    assert [len(c) for _, c in picked] == ecdcio.seg_frames(meta), name
    ecdcio.write(dst + ".tmp", meta, picked)
    os.replace(dst + ".tmp", dst)
    print("ok", name[:60], len(offsets), "segments", flush=True)
print("worker done", WORKER, flush=True)
