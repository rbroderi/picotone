"""Smoke test: a tiny training run on 2-3 clips that checks losses fall, checkpoint/resume is bit-exact,
evaluation + WAV writing work, and the MAC/parameter report prints.

    uv run python smoke.py --manifest data\\pd.jsonl --out runs\\smoke [--device cpu] [--steps 300]

Each child run has a hard timeout (--timeout, seconds).
"""
import argparse
import json
import os
import re
import subprocess
import sys

import corpus

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, timeout, log):
    print(">>", " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode != 0:
        print(open(log, encoding="utf-8").read()[-3000:])
        raise SystemExit(f"failed: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="a prepared manifest (pd.jsonl) to take clips from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--resume-steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seg-frames", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--skip-long", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = corpus.read_jsonl(a.manifest)
    train = [r for r in rows if r["split"] == "train" and r["keep"] and r["licence"] in ("cc0", "pd")]
    train = sorted(train, key=lambda r: r["id"])[:3]
    for r in train:
        r["seconds"] = min(r["seconds"], 30.0)  # tiny corpus: the first 30 s of each
    man = os.path.join(a.out, "smoke.jsonl")
    if os.path.exists(man):
        os.remove(man)
    corpus.append_jsonl(man, train)
    ev_src = json.load(open(a.manifest + ".eval.json", encoding="utf-8"))
    ev = [dict(e, dur=3.0) for e in (ev_src[:1] + ev_src[-1:])]  # one old, one new clip, 3 s each
    evp = os.path.join(a.out, "smoke_eval.json")
    json.dump(ev, open(evp, "w"), indent=1)
    print("smoke clips:", [r["ref"] for r in train], flush=True)
    py = sys.executable
    common = ["--manifest", man, "--eval-set", evp, "--device", a.device, "--threads", str(a.threads),
              "--batch", str(a.batch), "--seg-frames", str(a.seg_frames), "--workers", "2", "--tracks-per-batch",
              "2", "--chunk-seconds", "6", "--listen", "2", "--log-every", "10"]
    log = os.path.join(a.out, "smoke_log.txt")
    run([py, "report_model.py", "--time-seconds", "1", "--json", os.path.join(a.out, "cost.json")], a.timeout, log)
    run([py, "train.py", "--run", os.path.join(a.out, "bench"), "--bench", "2,4"] + common, a.timeout, log)
    # Bit-exact resume: A straight to N, B to N/2 then --resume to N.
    n = a.resume_steps
    ra, rb = os.path.join(a.out, "resA"), os.path.join(a.out, "resB")
    for d in (ra, rb):
        for f in ("ckpt_last.pt", "log.txt", "evals.jsonl", "train.jsonl", "stock_eval.json", "verdicts.jsonl"):
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
    knobs = ["--ckpt-every", str(n // 2), "--eval-every", str(n), "--keep-every", str(10 ** 9)]
    run([py, "train.py", "--run", ra, "--steps", str(n)] + knobs + common, a.timeout, log)
    run([py, "train.py", "--run", rb, "--steps", str(n // 2)] + knobs + common, a.timeout, log)
    run([py, "train.py", "--run", rb, "--steps", str(n), "--resume"] + knobs + common, a.timeout, log)

    def final_hash(d):
        m = re.findall(r"done at step (\d+); G (\w+)", open(os.path.join(d, "log.txt"), encoding="utf-8").read())
        return m[-1]

    ha, hb = final_hash(ra), final_hash(rb)
    print(f"resume check: straight {ha}, resumed {hb} -> {'BIT-EXACT' if ha == hb else 'DIFFERENT'}", flush=True)
    res = {"resume_bit_exact": ha == hb, "hash_straight": ha, "hash_resumed": hb}
    if not a.skip_long:
        rl = os.path.join(a.out, "long")
        run([py, "train.py", "--run", rl, "--steps", str(a.steps), "--eval-every", str(a.steps // 2),
             "--ckpt-every", str(a.steps // 2), "--keep-every", str(10 ** 9)] + common, a.timeout, log)
        tr = corpus.read_jsonl(os.path.join(rl, "train.jsonl"))
        first, last = tr[0], tr[-1]
        res["loss_first"] = {k: first[k] for k in ("mel", "fm", "adv", "d") if k in first}
        res["loss_last"] = {k: last[k] for k in ("mel", "fm", "adv", "d") if k in last}
        res["ms_per_step"] = [r["ms"] for r in tr[1:]]
        ev = corpus.read_jsonl(os.path.join(rl, "evals.jsonl"))
        res["eval_logmel"] = [(e["step"], e["means"]["all"]["logmel"]) for e in ev]
        res["listen_wavs"] = sorted(os.listdir(os.path.join(rl, "listen")))
        print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.out, "smoke_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
