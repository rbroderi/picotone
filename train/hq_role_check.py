"""Is a mid channel coded worse than a left/right one because of the net, or because of the clips?

Evaluates the test channels of each role twice: with their own role bias, and with the left/right bias.
If the own role wins, the role input helps, and the remaining difference between roles is the clips'.

    uv run --extra cpu python hq_role_check.py --ckpt out/hq_net.pt --test ../work/dac_codes/test --weights ../weights
"""
import argparse
import json
from pathlib import Path

import torch

import hq_net as h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--test", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    net = h.load_net(args.ckpt, args.weights)
    channels = h.load_dirs(args.test)
    report = {}
    for name, role in h.ROLES.items():
        mine = [c for c in channels if c[1] == role]
        if not mine:
            continue
        clips = len({c[3] for c in mine})
        for tier in (9, 3):
            row = {}
            for label, as_role in (("own", role), ("as_lr", h.ROLES["lr"]), ("as_mid", h.ROLES["mid"])):
                data = h.Codes([(c[0], as_role, c[2], c[3]) for c in mine], net.history)
                row[label] = round(h.evaluate(net, data, net.history, tier)[0], 4)
            report[f"{name}_q{tier}"] = row
            print(f"{name:4s} ({len(mine)} channels, {clips} clips) q{tier}: {row}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
