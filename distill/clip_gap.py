"""Where does the student lose to stock DAC? Per-clip gaps from a run's saved evals (no GPU, no audio).

    uv run python clip_gap.py --run runs/w512_tiers_clean_eval [--n-q 9] [--step 107500] [--w ema]

Reads <run>/evals.jsonl (the student's per-clip scores) and <run>/stock_eval.json (stock DAC's, same clips).
For each tier it prints the clips sorted by the log-mel gap (student - stock, and the ratio), the gap per
content type (from the clip names, e.g. old_14_piano -> piano), and how concentrated the gap is: what share
of the total excess comes from the worst 10% and 25% of clips. If a few content types carry most of it, more
training on that material may help; if it is spread evenly, the gap is the decoder's capacity.
"""
import argparse
import collections
import json
import os


def kind(name):
    parts = name.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-q", default="", help="tiers to show, e.g. 5,9 (default: every tier evaluated)")
    ap.add_argument("--step", type=int, default=0, help="eval step (default: the last one)")
    ap.add_argument("--w", default="", help="weights variant, raw or ema (default: ema if present)")
    ap.add_argument("--top", type=int, default=12, help="worst clips listed per tier")
    ap.add_argument("--json", default="", help="also write the per-clip table here")
    a = ap.parse_args()

    evals = [json.loads(line) for line in open(os.path.join(a.run, "evals.jsonl"), encoding="utf-8")]
    stock = json.load(open(os.path.join(a.run, "stock_eval.json"), encoding="utf-8"))
    step = a.step or max(e["step"] for e in evals)
    at = [e for e in evals if e["step"] == step]
    ws = {e.get("w", "raw") for e in at}
    w = a.w or ("ema" if "ema" in ws else sorted(ws)[0])
    tiers = sorted({e["n_q"] for e in at if e.get("w", "raw") == w})
    if a.n_q:
        tiers = [t for t in tiers if t in {int(v) for v in a.n_q.split(",")}]
    print(f"{a.run}: step {step}, weights {w} (available: {sorted(ws)}), tiers {tiers}")

    table = {}
    for nq in tiers:
        row = next(e for e in at if e["n_q"] == nq and e.get("w", "raw") == w)
        st = {r["name"]: r["m"] for r in stock[str(nq)]["rows"]}
        clips = []
        for r in row["rows"]:
            s = st.get(r["name"])
            if s is None:
                continue
            g = r["m"]["logmel"] - s["logmel"]
            clips.append({"name": r["name"], "kind": kind(r["name"]), "student": r["m"]["logmel"],
                          "stock": s["logmel"], "gap": g, "ratio": r["m"]["logmel"] / s["logmel"],
                          "sisdr_gap": r["m"]["sisdr"] - s["sisdr"]})
        clips.sort(key=lambda c: -c["gap"])
        table[nq] = clips
        n = len(clips)
        tot = sum(max(0.0, c["gap"]) for c in clips)
        mean_gap = sum(c["gap"] for c in clips) / n
        print(f"\n=== q{nq}: {n} clips, mean gap {mean_gap:+.3f} log-mel "
              f"({100 * (sum(c['student'] for c in clips) / sum(c['stock'] for c in clips) - 1):+.1f}% vs stock)")
        if tot > 0:
            for frac in (0.10, 0.25):
                k = max(1, round(n * frac))
                share = sum(max(0.0, c["gap"]) for c in clips[:k]) / tot
                print(f"  worst {int(frac * 100)}% of clips ({k}) carry {100 * share:.0f}% of the excess")
        print(f"  {sum(1 for c in clips if c['gap'] < 0)} clips where the student beats stock")
        print(f"  worst {a.top}:   gap    ratio   student  stock   SI-SDR gap")
        for c in clips[: a.top]:
            print(f"    {c['name']:24s} {c['gap']:+.3f}  {c['ratio']:.3f}  {c['student']:.3f}  {c['stock']:.3f}  "
                  f"{c['sisdr_gap']:+.2f}")
        by = collections.defaultdict(list)
        for c in clips:
            by[c["kind"]].append(c)
        print("  by content type (mean gap, ratio, clips):")
        for k, v in sorted(by.items(), key=lambda kv: -sum(c["gap"] for c in kv[1]) / len(kv[1])):
            print(f"    {k:16s} {sum(c['gap'] for c in v) / len(v):+.3f}  "
                  f"{sum(c['student'] for c in v) / sum(c['stock'] for c in v):.3f}  {len(v)}")
    if a.json:
        json.dump({"run": a.run, "step": step, "w": w, "tiers": table}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
