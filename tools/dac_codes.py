"""Dumps HQ (DAC 44 kHz) codes of a set of clips, for fitting the HQ cx2 tables.

    python tools/dac_codes.py --list X:/audio/work/clips2.json --split fit2 --wav X:/audio/work/wav2 \
        --out work/dac_codes/fit2 [--threads 3] [--jobs 4]

Each clip is encoded with `picotone encode --codebooks 9` (the shipping encoder, with its stereo choice) and its
codes written with `picotone dev codes`: `<name>.npy` (or `<name>.0.npy` and `<name>.1.npy`) plus `<name>.json`.
Clips already done are skipped, so the script can be stopped and restarted. Codes depend only on the encoder, so
a new encoder means a fresh dump with this script, then a refit (see train/README.md).

The clips in `EXCLUDE` are never dumped: they are in the fit splits but are not fit material.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import lib

# Clips of the fit splits that the HQ table is not fitted on, with the reason. The DTRASH clips are the ones
# X:/audio/distill/data/pd.jsonl flags `drop:dtrash-distorted` (clipped, distorted masters); the GT585 clips are
# digital silence (-180 dB RMS) for their whole length.
EXCLUDE = {
    "fit2__electronic__DTRASH31-Exist-TheTensionAndTheDarkness__8b50896b__c0": "dtrash-distorted",
    "fit2__electronic__DTRASH31-Exist-TheTensionAndTheDarkness__e3032727__c0": "dtrash-distorted",
    "fit2__jazz__Dtrash145-BabylonDisco-ToFindAndDestroyAtlantis__3cd818e4__c0": "dtrash-distorted",
    "fit2__jazz__Dtrash145-BabylonDisco-ToFindAndDestroyAtlantis__e13163c4__c0": "dtrash-distorted",
    "fit2__rock__DTRASH54-RefuzerVsSchizoid-WhoToldYouItIsDemocracy__407dc295__c0": "dtrash-distorted",
    "fit2__rock__DTRASH54-RefuzerVsSchizoid-WhoToldYouItIsDemocracy__49d2640f__c0": "dtrash-distorted",
    "fit2__ambient__GT585__e4cacef5__c0": "digital silence",
    "fit__electronic__GT585__01.Relative-DeadbyImplication__w0": "digital silence",
}


def dump(name, args):
    """Encodes and dumps one clip; returns False if it failed."""
    npy = os.path.join(args.out, name + ".npy")
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
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="clips JSON: a list of {name, split}")
    ap.add_argument("--split", required=True)
    ap.add_argument("--wav", required=True, help="directory holding <name>.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3, help="threads per picotone run")
    ap.add_argument("--jobs", type=int, default=1, help="clips encoded at once")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per clip before it counts as hung")
    args = ap.parse_args()
    with open(args.list, encoding="utf-8") as f:
        clips = [c["name"] for c in json.load(f) if c["split"] == args.split]
    for name in clips:
        if name in EXCLUDE:
            print(f"{name}: excluded ({EXCLUDE[name]})", flush=True)
    clips = [c for c in clips if c not in EXCLUDE]
    os.makedirs(args.out, exist_ok=True)
    todo = [c for c in clips if not os.path.exists(os.path.join(args.out, c + ".json"))]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} already dumped, {len(todo)} to do", flush=True)
    start = time.time()
    done = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for name, ok in zip(todo, pool.map(lambda c: dump(c, args), todo)):
            done += 1
            rate = (time.time() - start) / done
            if ok:
                print(f"[{done}/{len(todo)}] {name} ({rate:.0f} s per clip)", flush=True)


if __name__ == "__main__":
    main()
