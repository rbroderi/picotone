"""Markdown tables for the listening set (results/s12_listen_lstm.jsonl -> results/s12_listen_lstm.md)."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "s12_listen_lstm.jsonl"))]
out = ["# Listening set metrics (exactly the files in out/listen_lstm/)", "",
       "`vs stock` = SNR of the file against the stock decode of the same clip.  `d` columns = change against stock, measured",
       "against the source (+ log-mel / + MR-STFT / - SNR are worse).", "",
       "| set | clip | variant | SNR vs stock dB | SNR vs source | log-mel | MR-STFT | d SNR | d log-mel | d MR-STFT |", "|---|---|---|---|---|---|---|---|---|---|"]
agg = defaultdict(list)
for r in rows:
    head, variant = r["label"].split(": ")
    model, kbps, clip = head.split(" ", 2)
    f = lambda k: f"{r[k]:+.2f}" if k in r else ""
    vs = "" if variant == "stock" else f"{r['snr_orig']:.1f}"
    out.append(f"| {model} {kbps} | {clip} | {variant} | {vs} | {r['snr']:.2f} | {r['logmel']:.2f} | {r['mrstft']:.2f} | {f('d_snr')} | {f('d_logmel')} | {f('d_mrstft')} |")
    if variant != "stock":
        agg[(model + " " + kbps, variant)].append(r)
out += ["", "## Mean over clips", "", "| set | variant | clips | SNR vs stock dB | d SNR | d log-mel | d MR-STFT | worst clip d log-mel |", "|---|---|---|---|---|---|---|---|"]
for (s, v), rs in agg.items():
    m = lambda k: sum(r[k] for r in rs) / len(rs)
    out.append(f"| {s} | {v} | {len(rs)} | {m('snr_orig'):.1f} | {m('d_snr'):+.2f} | {m('d_logmel'):+.2f} | {m('d_mrstft'):+.2f} | {max(r['d_logmel'] for r in rs):+.2f} |")
open(os.path.join(HERE, "results", "s12_listen_lstm.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out[out.index("## Mean over clips"):]))
