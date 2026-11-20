"""Learning curve: does more data still help? Trains on 25%, 50% and 100% of the fit clips
for the same number of steps and prints held-out bits per code.

    uv run learning_curve.py --data train_data_48k_full.npz

Each run is train_net.py with --fraction (clips chosen by a hash of their names, so each
share contains the smaller ones) and writes out/lc_f<share>.pt plus its .log. Same steps,
batch, schedule and evaluation frames for all three, so the numbers are directly
comparable. 40,000 steps at batch 2048 = 82 M samples: 25 passes over the 25% share (about
what the full run makes over the whole set) and 6 over the whole set.

If bits per code still fall clearly from 50% to 100% (more than ~0.02 at K=15), more data
is still helping and the full run is worth it; if 50% and 100% tie, the net is
capacity-bound rather than data-bound.

Every run has a time limit (--hours-per-run); a run that hits it is stopped and reported.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fractions", default="0.25,0.5,1.0")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--hours-per-run", type=float, default=1.0, help="time limit of each run")
    ap.add_argument("--extra", default="", help="more train_net.py arguments, e.g. \"--threads 8\"")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for share in args.fractions.split(","):
        out = os.path.join(args.out_dir, f"lc_f{share.replace('.', '')}.pt")
        for stale in (out + ".log", out + ".ckpt"):
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [sys.executable, os.path.join(HERE, "train_net.py"), "--data", args.data, "--out", out,
               "--fraction", share, "--steps", str(args.steps), "--batch", str(args.batch),
               "--eval-every", str(args.eval_every), *args.extra.split()]
        print("==>", " ".join(cmd), flush=True)
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, timeout=args.hours_per_run * 3600)
        except subprocess.TimeoutExpired:
            print(f"!! fraction {share} hit the {args.hours_per_run} h limit; treat as hung", flush=True)
            continue
        with open(out + ".log", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        results[share] = {**rows[-1], "run_minutes": round((time.time() - t0) / 60, 1)}
    keys = [k for k in ("test_old_K15", "test_new_K15", "test_old_K8", "test_new_K8", "test_old_K4",
                        "test_new_K4") if any(k in r for r in results.values())]
    print(f"\nheld-out bits per code after {args.steps} steps at batch {args.batch}")
    print(f"{'fraction':>8} {'minutes':>8} " + " ".join(f"{k:>13}" for k in keys))
    for share, row in results.items():
        print(f"{share:>8} {row['run_minutes']:>8} " + " ".join(f"{row.get(k, float('nan')):>13.4f}" for k in keys))
    if "0.5" in results and "1.0" in results:
        gain = [results["0.5"][k] - results["1.0"][k] for k in ("test_old_K15", "test_new_K15")]
        print(f"\n50% -> 100% gain at K=15: old {gain[0]:+.4f}, new {gain[1]:+.4f} bits per code; "
              + ("more data still helps" if max(gain) > 0.02 else "little gain from more data"))


if __name__ == "__main__":
    main()
