"""Prints a results/s11_seam_*.jsonl file.  usage: python show_seam.py results/<file>.jsonl [--md]"""
import json
import sys

md = "--md" in sys.argv
cols = ["kbps", "snr_start", "snr_mid", "snr_end", "snr_dip", "lsd_start", "lsd_mid", "lsd_bump", "snr", "logmel", "mrstft"]
rows = [json.loads(l) for l in open(sys.argv[1])]
if md:
    print("| variant | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
for r in rows:
    if md:
        print(f"| {r['label']} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |")
    else:
        print(r["label"][:84].ljust(84), " ".join(f"{c}={r[c]:.2f}" for c in cols))
