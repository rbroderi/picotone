"""Export a decoder fine-tune (an --arch encoder checkpoint trained with --enc-train-dec) in the same format as
weights/dac44_vocos.pt (arm A's export): the decoder's state dict only (keys without 'dec.'), fp32, plus
<out>.json (arm A's json with step/source updated), <out>.credits.jsonl, <out>.NOTICE.txt, <out>.PROVENANCE.txt.
Owner request 2026-11-26 (control 5 = HQ decoder). Only reads checkpoints; refuses to overwrite.
  uv run --extra cuda python export_dec_ft.py --ckpt runs/lowtier_ctrl5/ckpt_0005000.pt --ema \
      --ref weights/dac44_vocos --out weights/dac44_vocos_ctrl5
Checks: (1) the same extraction applied to arm A's checkpoint (raw G) reproduces <ref>.pt tensor for tensor (with
--ref-ema: <ref-ckpt>'s G_ema decoder part, for chaining from an export of a fine-tune, e.g. q16 from control 5);
(2) the export loads strictly into vocos_student.make_decoder(json arch) and gives the same output as the
checkpoint's decoder as the evals built it (encoder.load_decoder + the checkpoint's dec.* weights), random latents.
"""
import argparse
import hashlib
import json
import os

import torch

import corpus
import encoder as enc_mod
from vocos_student import make_decoder


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dec_sd(s, key):
    sd = {k[4:]: v.float().clone() for k, v in s[key].items() if k.startswith("dec.")}
    assert not any(k.startswith("tier_emb") for k in sd), "tier-conditioned decoder: not the arm A layout"
    return sd


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ref", required=True, help="arm A's export, without .pt")
    p.add_argument("--ref-ckpt", default="runs/vocos_ft_cplx/ckpt_last.pt")
    p.add_argument("--out", required=True)
    # patch q16_export (defaults = the control 5 export's behaviour and text)
    p.add_argument("--ref-ema", action="store_true", help="check 1 on <ref-ckpt>'s G_ema dec.* (a fine-tune export)")
    p.add_argument("--desc", default="control 5's decoder (HQ decoder, replaces arm A; owner decision 2026-11-26)")
    p.add_argument("--from-desc", default="arm A, = dac44_vocos.pt")
    p.add_argument("--credits-desc", default="arm A's credits")
    p.add_argument("--recipe", default="control 3 (50/50 DAC-encoder / arm 4 20k EMA codes, restored Adam moments, lr "
                   "3e-6 after 500-step warm-up,\nteacher terms on DAC-code items, q1-q3 with 8 kHz BWE on arm-4-code "
                   "items) + --bwe-hf-weight 0.3; 5000 steps.")
    c = p.parse_args()
    for ext in (".pt", ".json", ".credits.jsonl", ".NOTICE.txt", ".PROVENANCE.txt"):
        assert not os.path.exists(c.out + ext), f"{c.out + ext} exists; not overwriting"
    ref = torch.load(c.ref + ".pt", map_location="cpu", weights_only=True)
    rj = json.load(open(c.ref + ".json"))
    # (1) the procedure on arm A reproduces its export
    sa = torch.load(c.ref_ckpt, map_location="cpu", weights_only=False)
    ra = dec_sd(sa, "G_ema") if c.ref_ema else {k: v.float() for k, v in sa["G"].items()}
    assert set(ra) == set(ref) and all(torch.equal(ra[k], ref[k]) for k in ref), "reference check failed"
    print(f"check 1: {c.ref_ckpt} {'G_ema dec.*' if c.ref_ema else 'G'} == {c.ref}.pt ({len(ref)} tensors, exact)")
    s = torch.load(c.ckpt, map_location="cpu", weights_only=False)
    w = "G_ema" if c.ema else "G"
    sd = dec_sd(s, w)
    assert set(sd) == set(ref) and all(sd[k].shape == ref[k].shape and sd[k].dtype == ref[k].dtype for k in ref)
    nd = sum(not torch.equal(sd[k], ref[k]) for k in ref)
    print(f"{c.ckpt} step {s['step']} ({w}): {len(sd)} tensors, same keys/shapes/dtypes as {c.ref}.pt; {nd} differ")
    # (2) loads into make_decoder and matches the eval-path decoder
    m = make_decoder(rj["arch"]).eval()
    m.load_state_dict(sd, strict=True)
    a = s["args"]
    dec, _, _ = enc_mod.load_decoder(a["enc_dec"], a.get("enc_dec_ema", False))
    dec.load_state_dict(sd, strict=True)
    dec.eval()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2, 1024, 60, generator=g)
    dmax = float((m(z) - dec(z)).abs().max())
    print(f"check 2: make_decoder(export) vs eval-path decoder, random latents: max abs diff {dmax:.3g}")
    assert dmax == 0.0
    torch.save(sd, c.out + ".pt")
    back = torch.load(c.out + ".pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(back[k], sd[k]) for k in sd)
    cfg = dict(rj, n_q=a["n_q"], step=s["step"], fold_max_abs_dev=0.0, licences=a["licences"])
    json.dump(cfg, open(c.out + ".json", "w"), indent=1)
    # credits: arm A's (the decoder this was fine-tuned from) + the fine-tune's own manifests (train, keep=true)
    seen, rows, per = set(), [], {}
    lic = corpus.parse_licences(a["licences"])
    src = [json.loads(l) for l in open(c.ref + ".credits.jsonl", encoding="utf-8")]
    for mf in a["manifest"]:
        for r in corpus.read_jsonl(mf):
            if r.get("split") == "train" and r.get("keep") and r.get("licence") in lic:
                cr = {k: r.get(k, "") for k in ("source", "artist", "title", "album", "licence", "licence_url", "url")}
                cr["licence_url"] = cr["licence_url"] or corpus.LICENCE_URLS.get(cr["licence"], "")
                src.append(cr)
    for cr in src:
        k = json.dumps(cr, sort_keys=True, ensure_ascii=False)
        if k not in seen:
            seen.add(k)
            rows.append(cr)
            per[cr["licence"]] = per.get(cr["licence"], 0) + 1
    base = os.path.basename(c.out)
    with open(c.out + ".credits.jsonl", "w", encoding="utf-8") as f:
        for cr in rows:
            f.write(json.dumps(cr, ensure_ascii=False) + "\n")
    import export
    with open(c.out + ".NOTICE.txt", "w", encoding="utf-8") as f:
        f.write(export.NOTICE.format(credits=base + ".credits.jsonl",
                                     counts=", ".join(f"{k} {v}" for k, v in sorted(per.items()))))
        if per.get("by-sa") or per.get("fal"):
            f.write(export.SA_NOTICE)
    with open(c.out + ".PROVENANCE.txt", "w", encoding="utf-8") as f:
        f.write(f"""{base}: {c.desc}. Vocos, dim 512, 8 blocks,
hop 512, same keys and layout as dac44_vocos.pt. Exported 2026-11-26 with export_dec_ft.py: the checkpoint's {w}
(EMA; the weights that were evaluated and rendered), decoder part (dec.*), fp32, no weight norm to fold.
Checkpoint: {c.ckpt}, step {s['step']}, sha256 {sha(c.ckpt)}
Fine-tuned from: {c.ref_ckpt} ({c.from_desc}), sha256 {sha(c.ref_ckpt)}
Recipe: {c.recipe}
Credits: {c.credits_desc} plus the fine-tune's manifests (train rows, keep=true), deduplicated.
Load: vocos_student.make_decoder(json["arch"]).load_state_dict(torch.load("{base}.pt")).
Verified: {c.ref_ckpt}'s {'G_ema' if c.ref_ema else 'G'} reproduces {os.path.basename(c.ref)}.pt exactly by the same extraction; the export gives identical output to
the checkpoint's decoder as the evals built it (random latents, max abs diff {dmax:.1g}).
""")
    print(f"credits: {len(rows)} tracks {per}")
    for ext in (".pt", ".json", ".credits.jsonl", ".NOTICE.txt", ".PROVENANCE.txt"):
        print(f"{sha(c.out + ext)}  {base + ext}")


if __name__ == "__main__":
    main()
