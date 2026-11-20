"""Builds and caches the 'fit' split activations (used only as data for closed-form solves).  usage: build_fit.py 24khz|48khz"""
import sys
import time

from common import Bench

t = time.time()
b = Bench(sys.argv[1], "fit")
print(sys.argv[1], "fit split:", len(b.names), "clips", tuple(b.enc_pre.shape), "wall s", round(time.time() - t))
