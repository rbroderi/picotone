"""Times Meta's LM entropy-coding path end to end on CPU.

Usage: meta_lm_speed.py TAG K THREADS SECONDS
Runs the path of encodec 0.1.1 `compress.py` (re-implemented, no downloaded code executed):
one streaming LM step per latent frame, `build_stable_quantized_cdf` per codebook, and a
port of their bit-wise arithmetic coder; state resets per coded segment. Decodes again and
checks the codes match. Only the entropy stage is timed, not the EnCodec encoder/decoder.
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch

import ecdcio
import lib

TAG, K, THREADS, SECONDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
torch.set_num_threads(THREADS)
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("metalm_defs", os.path.join(here, "metalm_defs.py"))
defs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(defs)
BITS = 24


def quantized_cdf(pdf):
    pdf = (pdf / 1e-8).floor() * 1e-8
    total = 2 ** BITS
    alpha = 2 * len(pdf) / total
    ranges = (((1 - alpha) * total) * pdf).floor().long() + 2
    return torch.cumsum(ranges, dim=-1)


class Coder:
    def __init__(self):
        self.low = self.high = 0
        self.max_bit = -1
        self.bits = []

    def _flush(self):
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            if b1 != self.high >> self.max_bit:
                break
            self.low -= b1 << self.max_bit
            self.high -= b1 << self.max_bit
            self.max_bit -= 1
            self.bits.append(b1)

    def push(self, symbol, cdf):
        while self.high - self.low + 1 < 2 ** BITS:
            self.low *= 2
            self.high = self.high * 2 + 1
            self.max_bit += 1
        delta = self.high - self.low + 1
        lo = 0 if symbol == 0 else cdf[symbol - 1].item()
        hi = cdf[symbol].item() - 1
        self.high = self.low + int(math.floor(hi * (delta / 2 ** BITS)))
        self.low = self.low + int(math.ceil(lo * (delta / 2 ** BITS)))
        self._flush()

    def finish(self):
        while self.max_bit >= 0:
            self.bits.append((self.low >> self.max_bit) & 1)
            self.max_bit -= 1
        return self.bits


class Decoder:
    def __init__(self, bits):
        self.low = self.high = self.current = 0
        self.max_bit = -1
        self.bits, self.pos = bits, 0

    def pull(self, cdf):
        while self.high - self.low + 1 < 2 ** BITS:
            bit = self.bits[self.pos] if self.pos < len(self.bits) else 0
            self.pos += 1
            self.low *= 2
            self.high = self.high * 2 + 1
            self.current = self.current * 2 + bit
            self.max_bit += 1
        delta = self.high - self.low + 1
        a, b = 0, len(cdf) - 1
        while True:
            mid = (a + b) // 2
            lo = cdf[mid - 1].item() if mid > 0 else 0
            hi = cdf[mid].item() - 1
            low = int(math.ceil(lo * (delta / 2 ** BITS))) + self.low
            high = int(math.floor(hi * (delta / 2 ** BITS))) + self.low
            if self.current < low:
                b = mid - 1
            elif self.current > high:
                a = mid + 1
            else:
                break
        self.low, self.high = low, high
        while self.max_bit >= 0:
            b1 = self.low >> self.max_bit
            if b1 != self.high >> self.max_bit:
                break
            self.low -= b1 << self.max_bit
            self.high -= b1 << self.max_bit
            self.current -= b1 << self.max_bit
            self.max_bit -= 1
        return mid


def encode(lm, codes):
    coder = Coder()
    states, offset = None, 0
    inp = torch.zeros(1, K, 1, dtype=torch.long)
    for t in range(len(codes)):
        with torch.no_grad():
            probas, states, offset = lm(inp, states, offset)
        for k in range(K):
            coder.push(int(codes[t, k]), quantized_cdf(probas[0, :, k, 0]))
        inp = 1 + torch.from_numpy(codes[t].astype(np.int64)).view(1, K, 1)
    return coder.finish()


def decode(lm, bits, frames):
    dec = Decoder(bits)
    states, offset = None, 0
    inp = torch.zeros(1, K, 1, dtype=torch.long)
    out = np.zeros((frames, K), dtype=np.int64)
    for t in range(frames):
        with torch.no_grad():
            probas, states, offset = lm(inp, states, offset)
        for k in range(K):
            out[t, k] = dec.pull(quantized_cdf(probas[0, :, k, 0]))
        inp = 1 + torch.from_numpy(out[t]).view(1, K, 1)
    return out


def main():
    rate = lib.FRAME_RATE[TAG]
    lm = defs.LMModel(lib.KMAX[TAG], 1024, num_layers=5, dim=200, past_context=int(3.5 * rate))
    ckpt = os.path.join(os.path.dirname(here), "metalm", "ckpt",
                        "lm24.th" if TAG == "24k" else "lm48.th")
    lm.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    lm.eval()
    files = lib.load_files(TAG, "test")
    enc_s = dec_s = audio = nbits = ncodes = 0.0
    for f in (files[13], files[18]):
        _, segs = ecdcio.read(f["path"])
        frames_left = int(SECONDS * rate)
        for _, c in segs:
            c = c[:frames_left, :K]
            if len(c) == 0:
                break
            frames_left -= len(c)
            t0 = time.perf_counter()
            bits = encode(lm, c)
            t1 = time.perf_counter()
            back = decode(lm, bits, len(c))
            t2 = time.perf_counter()
            assert (back == c).all(), "round trip failed"
            enc_s += t1 - t0
            dec_s += t2 - t1
            audio += len(c) / rate
            nbits += len(bits)
            ncodes += c.size
    print(f"MetaLM {TAG} K={K} threads={THREADS}: {audio:.1f}s audio, {nbits / ncodes:.3f} bits/code, "
          f"encode {enc_s:.1f}s = {audio / enc_s:.2f}x real time, "
          f"decode {dec_s:.1f}s = {audio / dec_s:.2f}x real time; round trip ok", flush=True)


main()
