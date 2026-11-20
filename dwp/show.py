"""Prints a results/*.jsonl file as a compact table.  usage: python show.py results/<file>.jsonl [--md]"""
import json
import sys

md = "--md" in sys.argv
cols = ["lstm_mac", "side_mac_saved_pct", "kbps", "codes", "codes_q8", "emb_snr", "snr_orig", "snr", "logmel", "mrstft", "d_snr", "d_logmel", "d_mrstft", "latent_snr", "latent_gain_db"]
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if not r["label"].startswith(("MAC", "A codebook"))]
use = [c for c in cols if any(c in r for r in rows)]
w = max(len(r["label"]) for r in rows)
if md:
    print("| variant | " + " | ".join(use) + " |")
    print("|---|" + "---|" * len(use))
for r in rows:
    vals = [(f"{r[c]:.2f}" if abs(r[c]) < 1e6 else "inf") if c in r and isinstance(r[c], (int, float)) else "" for c in use]
    if md:
        print(f"| {r['label']} | " + " | ".join(vals) + " |")
    else:
        print(r["label"].ljust(w), " ".join(f"{c}={v}" for c, v in zip(use, vals) if v))
