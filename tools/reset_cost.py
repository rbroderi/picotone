"""Size cost of shorter pages (predictor resets). Usage: reset_cost.py TAG MODEL.ecxm TESTSET_DIR K [K ...]

Runs `picotone dev check` (pack, unpack, seek and baseline-path checks) for several page
lengths and prints total coded bytes and the cost relative to a single page. Sizes are whole
`.pico` files. A page is a whole number of segments at 48 kHz (about 1 s each) and a number
of seconds of frames at 24 kHz.
"""
import glob
import os
import subprocess
import sys

import lib

TAG, MODEL, SETDIR = sys.argv[1], sys.argv[2], sys.argv[3]
with open(MODEL, "rb") as f:
    VERSION = int.from_bytes(f.read(8)[4:], "little")
CODEC = ["--codec", os.path.join(lib.WEIGHTS, lib.MODEL_NAME[TAG] + ".safetensors")] if VERSION == 4 else []
JOBS = os.environ.get("CHECK_JOBS", "8")
UNIT = 1 if TAG == "48k" else lib.FRAME_RATE[TAG]
for kk in [int(a) for a in sys.argv[4:]]:
    files = sorted(glob.glob(os.path.join(lib.WORK, SETDIR, f"{TAG}_K{kk}", "*.ecdc")))
    bundles = [lib.bundle(f) for f in files]
    base = None
    for seconds in (0, 10, 5, 4, 3, 2, 1):
        r = subprocess.run([lib.PICOTONE, "dev", "check", MODEL, str(seconds * UNIT), *bundles, *CODEC,
                            "--jobs", JOBS], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            raise SystemExit("check failed:\n" + r.stderr[-2000:])
        rows = [line.split("\t") for line in r.stderr.strip().splitlines()]
        flat = sum(int(x[1]) for x in rows)
        coded = sum(int(x[2]) for x in rows)
        payload = sum(int(x[6]) for x in rows)
        base = base or coded
        print(f"{TAG} K={kk} page {seconds or 'whole':>5} s: {len(rows)} files ok, coded {coded} "
              f"({100 * (1 - coded / flat):.2f}% saved), cost vs one page {100 * (coded / base - 1):+.2f}%, "
              f"Ogg and header overhead {100 * (coded - payload) / coded:.2f}%",
              flush=True)
