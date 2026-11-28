"""Patch q12_ext, step 1 (user request 2026-11-25): build codebooks 10..9+n in DAC's style on arm 4 20k EMA's
residual after q9, over training crops (the encoder run's manifests and mix; data.load_sources keeps only
split=train, keep=true rows, so the DTRASH rows are out). Per new codebook, sequentially:
  r = the residual (1024-d) at the crops' central frames (full --enc-ctx context each side, fp32);
  in_proj = the top-8 PCA directions of r (weight V^T, bias -V^T mean): the in-space projection p = V^T (r - mean);
  entries = spherical k-means of p (assign by cosine, as the lookup does; centroid = mean of the raw p in the cluster,
            empty clusters re-seeded from random points), 1024 entries;
  out_proj = V, bias mean (so out_proj(entry) maps back); r -= out_proj(entry).
Writes {n_ext, ext (state dict for encoder.ExtRVQ.ext), meta, stats}; stats also on the 48 picotone eval clips.
Only reads checkpoints.
  uv run --extra cuda python build_ext_cb.py --out runs/q12_ext/ext_cb_init.pt
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import corpus
import data
import encoder as enc_mod
import postfilter
import train
import verdict
from teacher import HOP, load_teacher


def kmeans_sph(p, k, iters, gen):
    n = p.shape[0]
    pn = F.normalize(p, dim=1)
    c = p[torch.randperm(n, generator=gen, device="cpu")[:k].to(p.device)].clone()
    for it in range(iters):
        idx = torch.cat([(pn[i:i + 65536] @ F.normalize(c, dim=1).T).argmax(1) for i in range(0, n, 65536)])
        cnt = torch.bincount(idx, minlength=k).float()
        s = torch.zeros_like(c).index_add_(0, idx, p)
        dead = cnt == 0
        c = torch.where(dead[:, None], c, s / cnt.clamp(min=1)[:, None])
        if dead.any():
            c[dead] = p[torch.randint(0, n, (int(dead.sum()),), generator=gen).to(p.device)]
    idx = torch.cat([(pn[i:i + 65536] @ F.normalize(c, dim=1).T).argmax(1) for i in range(0, n, 65536)])
    return c, idx, int((torch.bincount(idx, minlength=k) == 0).sum())


def apply_cb(cb, r):  # r [N, 1024] -> (p, entry, zi, idx)
    W, b = cb["in_proj.weight"][:, :, 0], cb["in_proj.bias"]
    p = r @ W.T + b
    w = cb["codebook.weight"]
    idx = (F.normalize(p, dim=1) @ F.normalize(w, dim=1).T).argmax(1)
    e = w[idx]
    zi = e @ cb["out_proj.weight"][:, :, 0].T + cb["out_proj.bias"]
    return p, e, zi, idx


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, help="the STFT encoder checkpoint (train.py --arch encoder), e.g. runs/encB_arm4/ckpt_0020000.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ext", type=int, default=3)
    ap.add_argument("--frames", type=int, default=400_000)
    ap.add_argument("--seg", type=int, default=128, help="central frames per crop")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base-ckpt", default="",
                    help="patch q16: start from this checkpoint's trained extra codebooks (rvq.ext.*, --base-weights), "
                         "applied first; the --n-ext new ones are built on the residual after them; '' = off")
    ap.add_argument("--base-weights", default="G_ema")
    c = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    teacher = load_teacher(dev)
    s = torch.load(c.enc, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**s["args"])
    enc, rvq = train.code_encoder(c.enc, teacher, dev)
    lic = corpus.parse_licences(a.licences)
    src = data.load_sources(a.manifest, lic, a.mix, a.mix_floor, print, a.path_map, data.eval_artists(a.eval_set))
    ce = a.enc_ctx // HOP
    crop = c.seg * HOP + 2 * a.enc_ctx
    ds = data.BatchStream(src, c.batch, crop, [9], c.seed, 0, a.tracks_per_batch, a.chunk_seconds)
    dl = DataLoader(ds, batch_size=None, num_workers=8, prefetch_factor=4)
    rs, zs = [], 0.0
    nf = 0
    for b in dl:
        x = b["x"].to(dev)
        z = enc(x.float())
        zq = rvq(z, 9, st=False)[0]
        r = (z - zq)[..., ce: ce + c.seg]
        zs = zs + float(z[..., ce: ce + c.seg].pow(2).sum())
        rs.append(r.permute(0, 2, 1).reshape(-1, r.shape[1]))
        nf += rs[-1].shape[0]
        if nf >= c.frames:
            break
    R = torch.cat(rs)[: c.frames]
    print(f"{R.shape[0]} residual frames from {len(rs)} batches of {c.batch} crops x {c.seg} frames "
          f"({time.time() - t0:.0f} s); ||r9||^2/||z||^2 {float(R.pow(2).sum()) / zs:.3f}", flush=True)
    gen = torch.Generator().manual_seed(c.seed)
    ext, stats = {}, []
    nb = 0
    if c.base_ckpt:  # patch q16
        sb = torch.load(c.base_ckpt, map_location="cpu", weights_only=False)[c.base_weights]
        bext = {k[len("rvq.ext."):]: v.float() for k, v in sb.items() if k.startswith("rvq.ext.")}
        nb = len({k.split(".")[0] for k in bext})
        for j in range(nb):
            cb = {k[len(f"{j}."):]: v.to(dev) for k, v in bext.items() if k.startswith(f"{j}.")}
            p2, e, zi, idx2 = apply_cb(cb, R)
            st = {"codebook": 10 + j, "base": True,
                  "in_space_err": float((p2 - e).pow(2).sum() / p2.pow(2).sum()),
                  "residual_energy_ratio": float((R - zi).pow(2).sum() / R.pow(2).sum()),
                  "used_entries": int(torch.unique(idx2).numel())}
            stats.append(st)
            print(f"codebook {10 + j} (from {c.base_ckpt} {c.base_weights}): in-space err {st['in_space_err']:.4f}; "
                  f"||r - out_proj(c)||^2/||r||^2 {st['residual_energy_ratio']:.4f}; entries used "
                  f"{st['used_entries']}/1024", flush=True)
            for k, v in cb.items():
                ext[f"{j}.{k}"] = v.cpu()
            R = R - zi
    for j in range(nb, nb + c.n_ext):
        mu = R.mean(0)
        X = R - mu
        C = (X.T @ X) / X.shape[0]
        ev, V = torch.linalg.eigh(C.double())
        V8 = V[:, -8:].flip(1).float()  # [1024, 8], top first
        pca_share = float(ev[-8:].sum() / ev.sum())
        p = X @ V8
        cent, idx, dead = kmeans_sph(p, 1024, c.iters, gen)
        cb = {"in_proj.weight": V8.T[:, :, None].contiguous(), "in_proj.bias": -(V8.T @ mu),
              "codebook.weight": cent, "out_proj.weight": V8[:, :, None].contiguous(), "out_proj.bias": mu.clone()}
        p2, e, zi, idx2 = apply_cb(cb, R)
        st = {"codebook": 10 + j, "pca_top8_var_share": pca_share,
              "in_space_err": float((p2 - e).pow(2).sum() / p2.pow(2).sum()),
              "in_space_cos": float(F.cosine_similarity(p2, e, dim=1).mean()),
              "residual_energy_ratio": float((R - zi).pow(2).sum() / R.pow(2).sum()),
              "dead_entries": dead, "used_entries": int(torch.unique(idx2).numel())}
        stats.append(st)
        print(f"codebook {10 + j}: top-8 PCA share of residual variance {pca_share:.3f}; in-space err "
              f"{st['in_space_err']:.4f} (cos {st['in_space_cos']:.3f}); ||r - out_proj(c)||^2/||r||^2 "
              f"{st['residual_energy_ratio']:.4f}; entries used {st['used_entries']}/1024", flush=True)
        for k, v in cb.items():
            ext[f"{j}.{k}"] = v.cpu()
        R = R - zi
    c.n_ext = nb + c.n_ext  # patch q16: all extra codebooks from here on
    # check: the state dict loads into ExtRVQ, and the same numbers on the 48 eval clips through ExtRVQ itself
    m = enc_mod.ExtRVQ(teacher.quantizer, c.n_ext).to(dev).eval()
    m.ext.load_state_dict(ext)
    clips = [e for e in train.prepare_eval(a, teacher, dev, print) if e["set"] in verdict.SETS]
    per = {s_: [] for s_ in verdict.SETS}
    for e in clips:
        L = e["codes"].shape[-1] * HOP
        xt = torch.from_numpy(np.ascontiguousarray(e["x"][:, :L], dtype=np.float32))[:, None].to(dev)
        xt = torch.nn.functional.pad(xt, (0, L - xt.shape[-1]))
        z = enc(xt)
        zq = {n: m(z, n, st=False)[0] for n in range(9, 10 + c.n_ext)}
        top = zq[9 + c.n_ext]
        r = (z - zq[9]).permute(0, 2, 1).reshape(-1, 1024)
        row = []
        for j in range(c.n_ext):
            cb = {k[len(f"{j}."):]: v.to(dev) for k, v in ext.items() if k.startswith(f"{j}.")}
            p2, e2, zi, _ = apply_cb(cb, r)
            row += [float((p2 - e2).pow(2).sum() / p2.pow(2).sum()), float((r - zi).pow(2).sum() / r.pow(2).sum())]
            r = r - zi
        row += [float((top - zq[n]).pow(2).sum() / top.pow(2).sum()) for n in range(9, 9 + c.n_ext)]
        per[e["set"]].append(row)
    pic = np.mean([np.mean(per[s_], 0) for s_ in verdict.SETS], 0)
    ev48 = {}
    for j in range(c.n_ext):
        ev48[str(10 + j)] = {"in_space_err": pic[2 * j], "residual_energy_ratio": pic[2 * j + 1],
                             f"settle_vs_q{9 + c.n_ext}": pic[2 * c.n_ext + j] if j < c.n_ext else None}
        print(f"48 clips, codebook {10 + j}: in-space err {pic[2 * j]:.4f}; ||r - out_proj(c)||^2/||r||^2 "
              f"{pic[2 * j + 1]:.4f}; ||z_q{9 + c.n_ext} - z_q{9 + j}||^2/||z_q{9 + c.n_ext}||^2 "
              f"{pic[2 * c.n_ext + j]:.4f}")
    meta = {"enc": c.enc, "enc_sha": postfilter.file_sha16(c.enc), "frames": int(R.shape[0]), "seg": c.seg,
            "batch": c.batch, "seed": c.seed, "iters": c.iters, "manifests": a.manifest, "mix": a.mix,
            **({"base_ckpt": c.base_ckpt, "base_sha": postfilter.file_sha16(c.base_ckpt), "base_weights": c.base_weights,
                "n_base": nb} if c.base_ckpt else {})}
    torch.save({"n_ext": c.n_ext, "ext": ext, "meta": meta, "stats": stats, "eval48": ev48}, c.out)
    json.dump({"meta": meta, "stats": stats, "eval48": ev48}, open(c.out.replace(".pt", ".json"), "w"), indent=1)
    print(f"wrote {c.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
