"""Step 1: empirical entropies of EnCodec codes under simple contexts.

Usage: context_entropy.py TAG   (24k or 48k). Writes work/analysis1_TAG.json.

"plug-in" numbers are entropies of the empirical distribution of the same data they are
measured on (optimistic for conditional contexts, because the table memorises the data).
"xent" numbers are held-out cross-entropies: tables counted on the fit set, smoothed, and
scored on the test set. Those are what a static table coder would really achieve.
"""
import json
import os
import sys

import numpy as np

import lib

TAG = sys.argv[1]
V = 1024


def plugin_entropy(counts):
    """Conditional entropy in bits of rows of a count table (rows = contexts)."""
    n = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(counts > 0, counts * np.log2(counts / row), 0.0)
    return float(-t.sum() / n)


def unigram(y):
    c = np.bincount(y, minlength=V).astype(np.float64)
    return (c + 0.5) / (c.sum() + 0.5 * V), c


def bigram_probs(ctx_fit, y_fit, ctx_eval, y_eval, p0, disc=0.75):
    """Interpolated absolute-discounting bigram p(y | ctx) evaluated per sample."""
    tab = np.bincount(ctx_fit * V + y_fit, minlength=V * V).reshape(V, V).astype(np.float64)
    na = tab.sum(axis=1)
    n1 = (tab > 0).sum(axis=1)
    c = tab[ctx_eval, y_eval]
    n = na[ctx_eval]
    lam = np.where(n > 0, disc * n1[ctx_eval] / np.maximum(n, 1), 1.0)
    p = np.maximum(c - disc, 0) / np.maximum(n, 1) + lam * p0[y_eval]
    return p, tab


def trigram_probs(a_fit, b_fit, y_fit, a_ev, b_ev, y_ev, pa, pb, disc=0.75):
    """p(y | a, b) with absolute discounting, backing off to the mean of p(y|a), p(y|b)."""
    ctx = a_fit * V + b_fit
    key = ctx * V + y_fit
    uk, cnt = np.unique(key, return_counts=True)
    uc, ccnt = np.unique(ctx, return_counts=True)
    n1 = np.bincount(np.searchsorted(uc, uk // V), minlength=len(uc))
    ctx_e = a_ev * V + b_ev
    key_e = ctx_e * V + y_ev
    i = np.searchsorted(uk, key_e)
    i = np.minimum(i, len(uk) - 1)
    c = np.where(uk[i] == key_e, cnt[i], 0).astype(np.float64)
    j = np.searchsorted(uc, ctx_e)
    j = np.minimum(j, len(uc) - 1)
    seen = uc[j] == ctx_e
    n = np.where(seen, ccnt[j], 0).astype(np.float64)
    lam = np.where(seen, disc * n1[j] / np.maximum(n, 1), 1.0)
    back = 0.5 * (pa + pb)
    p = np.maximum(c - disc, 0) / np.maximum(n, 1) + lam * back
    # plug-in conditional entropy of the fit set under the (a, b) context
    ctx_of_key = np.searchsorted(uc, uk // V)
    h = -(cnt * np.log2(cnt / ccnt[ctx_of_key])).sum() / len(y_fit)
    return p, float(h), len(uc)


def bits(p):
    return float(-np.log2(p).mean())


def main():
    fit, fs = lib.concat(lib.load_files(TAG, "fitall"))
    test_files = lib.load_files(TAG, "test")
    test, ts = lib.concat(test_files)
    kmax = lib.KMAX[TAG]
    print(TAG, "fit frames", len(fit), "test frames", len(test), flush=True)
    rows = []
    for k in range(kmax):
        yf, yt = fit[:, k], test[:, k]
        p0, c0 = unigram(yf)
        r = {"k": k}
        r["h0_fit_plugin"] = plugin_entropy(c0[None, :])
        r["h0_test_plugin"] = plugin_entropy(np.bincount(yt, minlength=V)[None, :].astype(float))
        # mean per-file plug-in order-0 entropy (what an ideal per-file static table gives)
        tot, acc = 0, 0.0
        for f in test_files:
            y = f["codes"][:, k].astype(np.int64)
            acc += plugin_entropy(np.bincount(y, minlength=V)[None, :].astype(float)) * len(y)
            tot += len(y)
        r["h0_test_perfile_plugin"] = acc / tot
        r["x0"] = bits(p0[yt])
        af, at = lib.prev_frame(yf, fs), lib.prev_frame(yt, ts)
        pa, tab = bigram_probs(af, yf, at, yt, p0)
        r["ha_fit_plugin"] = plugin_entropy(tab)
        r["xa"] = bits(pa)
        if k > 0:
            bf, bt = fit[:, k - 1], test[:, k - 1]
            pb, tab = bigram_probs(bf, yf, bt, yt, p0)
            r["hb_fit_plugin"] = plugin_entropy(tab)
            r["xb"] = bits(pb)
            paf, _ = bigram_probs(af, yf, af, yf, p0)  # unused for scoring; kept simple
            pab, hab, nctx = trigram_probs(af, bf, yf, at, bt, yt, pa, pb)
            r["hab_fit_plugin"] = hab
            r["hab_fit_contexts"] = nctx
            r["xab"] = bits(pab)
            r["xmix"] = bits(0.5 * (pa + pb))
        else:
            r["xb"] = r["x0"]
            r["xab"] = r["xa"]
            r["xmix"] = r["xa"]
            r["hb_fit_plugin"] = r["h0_fit_plugin"]
            r["hab_fit_plugin"] = r["ha_fit_plugin"]
        rows.append(r)
        print(json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}),
              flush=True)
    out = {"tag": TAG, "fit_frames": len(fit), "test_frames": len(test), "rows": rows}
    json.dump(out, open(os.path.join(lib.WORK, f"analysis1_{TAG}.json"), "w"), indent=1)
    keys = ["h0_fit_plugin", "h0_test_plugin", "h0_test_perfile_plugin", "x0", "ha_fit_plugin",
            "xa", "hb_fit_plugin", "xb", "hab_fit_plugin", "xab", "xmix"]
    for kk in (2, 4, 8, 16, 32):
        if kk <= kmax:
            print("K=%d " % kk + " ".join(
                "%s=%.3f" % (key, np.mean([r[key] for r in rows[:kk]])) for key in keys))


main()
