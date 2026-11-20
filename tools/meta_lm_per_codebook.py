"""Per-codebook bits of Meta's LM on the test set, optionally with history ablations.

Usage: meta_lm_per_codebook.py TAG K [context_frames]
context_frames limits how many previous frames the LM may see (by resetting its state),
to show how much of its gain needs long context.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

import ecdcio
import lib

TAG, K = sys.argv[1], int(sys.argv[2])
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("metalm_defs", os.path.join(here, "metalm_defs.py"))
defs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(defs)
torch.set_num_threads(4)
rate = lib.FRAME_RATE[TAG]
lm = defs.LMModel(lib.KMAX[TAG], 1024, num_layers=5, dim=200, past_context=int(3.5 * rate))
ckpt = os.path.join(os.path.dirname(here), "metalm", "ckpt", "lm24.th" if TAG == "24k" else "lm48.th")
lm.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
lm.eval()
bits = np.zeros(K)
count = 0
by_pos = np.zeros(150)
pos_n = np.zeros(150)
for f in lib.load_files(TAG, "test"):
    _, segs = ecdcio.read(f["path"])
    chunks = [c[:, :K] for _, c in segs]
    if TAG == "24k":
        c = chunks[0][:3000]
        chunks = [c[i:i + 300] for i in range(0, len(c) - 299, 300)]
    else:
        chunks = [c for c in chunks if len(c) == 150][:40]
    if CTX:
        chunks = [c[i:i + CTX + 1] for c in chunks for i in range(0, len(c) - CTX, CTX + 1)]
    x = torch.from_numpy(np.stack(chunks).astype(np.int64))  # [B, T, K]
    b, t, _ = x.shape
    inp = torch.zeros(b, K, t, dtype=torch.long)
    inp[:, :, 1:] = 1 + x[:, :-1].permute(0, 2, 1)
    with torch.no_grad():
        probas, _, _ = lm(inp)
    p = probas.gather(1, x.permute(0, 2, 1)[:, None])[:, 0]  # [B, K, T]
    lb = -torch.log2(p.double())
    if CTX:
        lb = lb[:, :, -1:]  # only the frame that sees exactly CTX previous frames
    else:
        by_pos[:t] += lb.mean(1).sum(0).numpy()[:150]
        pos_n[:t] += b
    bits += lb.sum((0, 2)).numpy()
    count += lb.shape[0] * lb.shape[2]
print(f"{TAG} K={K} ctx={CTX or 'full'}: mean {bits.sum() / count / K:.3f}  per codebook "
      + " ".join(f"{v:.2f}" for v in bits / count), flush=True)
if not CTX:
    pn = np.maximum(pos_n, 1)
    print("  by frame position in segment (mean over codebooks): "
          + " ".join(f"{i}:{by_pos[i] / pn[i]:.2f}" for i in (0, 1, 2, 3, 5, 8, 12, 20, 40, 80, 149)))
