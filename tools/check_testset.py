"""Runs `picotone dev check` (pack, unpack, seek, byte-compare) on the per-bandwidth test sets.

Usage: check_testset.py TAG MODEL.ecxm PAGE_LENGTH|auto TESTSET_DIR K [K ...]
(TESTSET_DIR: testsets, testsets2 or ps_<preset>_<old|new>). The test clips are legacy flat
`.ecdc` files; their codes go to `picotone` as `.npy` bundles (`lib.bundle`).
Writes work/eval_<model stem>.json and prints totals. Every size is the real coded file
size including the container header; a file that fails the round trip aborts the run.
"""
import glob
import json
import os
import subprocess
import sys

import lib

TAG, MODEL = sys.argv[1], sys.argv[2]
PAGES, SETDIR = sys.argv[3], sys.argv[4]
KS = [int(a) for a in sys.argv[5:]]
JOBS = os.environ.get("CHECK_JOBS", "8")
stem = os.path.splitext(os.path.basename(MODEL))[0]
out = {"model_bytes": os.path.getsize(MODEL)}
for kk in KS:
    files = sorted(glob.glob(os.path.join(lib.WORK, SETDIR, f"{TAG}_K{kk}", "*.ecdc")))
    bundles = [lib.bundle(f) for f in files]
    r = subprocess.run([lib.PICOTONE, "dev", "check", MODEL, PAGES, *bundles, "--jobs", JOBS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("check failed:\n" + r.stderr[-3000:])
    rows = {}
    for line in r.stderr.strip().splitlines():
        name, flat, coded, codes, tp, tu, payload = line.split("\t")
        rows[name] = {"flat": int(flat), "coded": int(coded), "codes": int(codes),
                      "payload": int(payload), "pack_s": float(tp), "unpack_s": float(tu), "group": name.split("_")[1]}
    assert len(rows) == len(files)
    out[str(kk)] = rows
    tot = {key: sum(v[key] for v in rows.values()) for key in ("flat", "coded", "codes", "pack_s",
                                                               "unpack_s")}
    audio_s = tot["codes"] / kk / lib.FRAME_RATE[TAG]
    print(f"{TAG} K={kk}: {len(rows)} files lossless; flat {tot['flat']} coded {tot['coded']} "
          f"-> {8 * tot['coded'] / tot['codes']:.3f} bits/code, "
          f"{100 * (1 - tot['coded'] / tot['flat']):.1f}% saved; "
          f"pack {audio_s / tot['pack_s']:.0f}x unpack {audio_s / tot['unpack_s']:.0f}x real time",
          flush=True)
    groups = sorted({v["group"] for v in rows.values()})
    print("   " + "  ".join(
        f"{g} {100 * (1 - sum(v['coded'] for v in rows.values() if v['group'] == g) / sum(v['flat'] for v in rows.values() if v['group'] == g)):.1f}%"
        for g in groups), flush=True)
json.dump(out, open(os.path.join(lib.WORK, f"eval_{stem}_{SETDIR}_r{PAGES}.json"), "w"), indent=1)
