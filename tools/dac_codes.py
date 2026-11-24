"""Dumps HQ (DAC 44 kHz) codes of a set of clips, for fitting the HQ cx2 tables.

    python tools/dac_codes.py --list X:/audio/work/clips2.json --split fit2 --wav X:/audio/work/wav2 \
        --out work/dac_codes/fit2 [--threads 3] [--limit N]

Each clip is encoded with `picotone encode --codebooks 9` (the shipping encoder, with its stereo choice) and its
codes written with `picotone dev codes`: `<name>.npy` (or `<name>.0.npy` and `<name>.1.npy`) plus `<name>.json`.
Clips already done are skipped, so the script can be stopped and restarted. Codes depend only on the encoder, so
a new encoder means a fresh dump with this script, then a refit (see train/README.md).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import lib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="clips JSON: a list of {name, split}")
    ap.add_argument("--split", required=True)
    ap.add_argument("--wav", required=True, help="directory holding <name>.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use this many clips, spread evenly over the split (0: all)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per clip before it counts as hung")
    args = ap.parse_args()
    with open(args.list, encoding="utf-8") as f:
        clips = [c["name"] for c in json.load(f) if c["split"] == args.split]
    if args.limit and args.limit < len(clips):
        # Every k-th clip, so the subset spreads over the list's groups and sources.
        clips = [clips[i * len(clips) // args.limit] for i in range(args.limit)]
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    done = 0
    for i, name in enumerate(clips):
        npy = os.path.join(args.out, name + ".npy")
        side = os.path.join(args.out, name + ".json")
        if os.path.exists(side):
            continue
        pico = os.path.join(args.out, name + ".pico")
        wav = os.path.join(args.wav, name + ".wav")
        try:
            subprocess.run([lib.PICOTONE, "encode", "--codebooks", "9", "--threads", str(args.threads), wav, pico],
                           check=True, capture_output=True, timeout=args.timeout)
            subprocess.run([lib.PICOTONE, "dev", "codes", "--threads", str(args.threads), pico, npy],
                           check=True, capture_output=True, timeout=args.timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err = getattr(e, "stderr", b"") or b""
            print(f"{name}: failed: {e} {err.decode(errors='replace')[-300:]}", file=sys.stderr, flush=True)
            continue
        done += 1
        rate = (time.time() - start) / done
        print(f"[{i + 1}/{len(clips)}] {name} ({rate:.0f} s per clip)", flush=True)


if __name__ == "__main__":
    main()
