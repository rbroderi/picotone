"""Export a trained student: weight norm folded, generator only, fp32, plus a config and the training
attribution list (for the licence policy chosen).

    uv run --extra cuda python export.py --ckpt runs\\full512\\ckpt_last.pt --out export\\student_w512_q5
        [--manifest data\\pd.jsonl --manifest data\\jamendo.jsonl]

Writes <out>.pt (state dict of plain Conv1d/ConvTranspose1d weights + Snake alphas), <out>.json (width, rates,
codebooks, step, max deviation of the folded model from the parametrized one) and, with --manifest,
<out>.credits.jsonl (artist/title/licence/licence_url/url of every training track the run's --licences
admitted, BY-SA included) and <out>.NOTICE.txt (DAC MIT notice, and the ShareAlike note when BY-SA/FAL audio
was used). Ship all four files together.
"""
import argparse
import json
import os

import torch

import corpus
from student import RATES, StudentDecoder, fold_weight_norm


NOTICE = """\
This decoder is a distilled student of DAC (descript-audio-codec, 44 kHz weights), MIT licence,
Copyright (c) 2023-present, Descript. It is a derivative of DAC's MIT weights; the MIT notice applies.

It was trained on third-party audio (tracks by licence: {counts}). Every training track is credited, with its
licence URL, in {credits}.
"""
SA_NOTICE = """
Some training audio is licensed CC BY-SA (or the Free Art Licence). Whether trained weights are an adaptation of
their training data is legally unsettled; these weights may be a ShareAlike derivative of that audio. Ship this
notice and the credits file with the weights.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", action="append", default=[])
    a = ap.parse_args()
    s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = s["args"]
    G = StudentDecoder(s["width"])
    G.load_state_dict(s["G"])
    G.eval()
    z = torch.randn(1, 1024, 40)
    with torch.no_grad():
        y0 = G(z)
        fold_weight_norm(G)
        y1 = G(z)
    dev = float((y0 - y1).abs().max())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(G.state_dict(), a.out + ".pt")
    cfg = {"width": s["width"], "rates": list(RATES), "latent": 1024, "sample_rate": 44100, "hop": 512,
           "n_q": args["n_q"], "step": s["step"], "fold_max_abs_dev": dev, "licences": args["licences"],
           "teacher": "descript-audio-codec 1.0.0, 44khz 8kbps weights (MIT, Copyright (c) 2023-present, Descript)"}
    json.dump(cfg, open(a.out + ".json", "w"), indent=1)
    if a.manifest:
        lic = corpus.parse_licences(args["licences"])
        n, per = 0, {}
        with open(a.out + ".credits.jsonl", "w", encoding="utf-8") as f:
            for p in a.manifest:
                for r in corpus.read_jsonl(p):
                    if r.get("split") == "train" and r.get("keep") and r.get("licence") in lic:
                        c = {k: r.get(k, "") for k in ("source", "artist", "title", "album", "licence",
                                                       "licence_url", "url")}
                        # every credited track carries a licence URL (the generic version when the file named none)
                        c["licence_url"] = c["licence_url"] or corpus.LICENCE_URLS.get(c["licence"], "")
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                        per[c["licence"]] = per.get(c["licence"], 0) + 1
                        n += 1
        with open(a.out + ".NOTICE.txt", "w", encoding="utf-8") as f:
            f.write(NOTICE.format(credits=os.path.basename(a.out) + ".credits.jsonl",
                                  counts=", ".join(f"{k} {v}" for k, v in sorted(per.items()))))
            if per.get("by-sa") or per.get("fal"):
                f.write(SA_NOTICE)
        print(f"{n} training tracks credited ({per}) -> {a.out}.credits.jsonl, notice -> {a.out}.NOTICE.txt")
    print(f"exported step {s['step']} width {s['width']} -> {a.out}.pt (fold deviation {dev:.2e})")


if __name__ == "__main__":
    main()
