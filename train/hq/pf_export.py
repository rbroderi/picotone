"""Export the chosen low-tier post-filter (pf_lowtier.LowTierPF, a train.py --pf-lowtier checkpoint; shipped: Run G
7.5k, EMA) for the Rust port, CPU only; only reads checkpoints; refuses to overwrite. Writes into --out:
  pf_g7500.safetensors   the post-filter's EMA weights (LowTierPF state_dict names; G's are these with "pf."), f32
  pf_g7500.json          architecture + inference config
  pf_ref/<case>/         test vectors (training code, G_ema, CPU fp32): codes, z, the post-filter's inputs (S, h, tier),
                         the frozen decoder's output and the post-filter output (float32 WAV, unclipped), and for the
                         first case one activation per layer
  uv run --extra cpu python pf_export.py --ckpt runs/pf_run_g/ckpt_0007500.pt --dec weights/dac44_vocos_q16.pt \
      --enc runs/encB_arm4/ckpt_0020000.pt --out pf_lowtier/export [--bolero-codes pf_lowtier/bolero_codes]
Then ../../tools/export_postfilter.py adds the post-filter to picotone's decoder weights.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

import encoder as enc_mod
import pf_lowtier
import train
import verdict
from teacher import codes_to_z, load_teacher

SECONDS = 4.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(sd, aq, teacher, dev):
    """The checkpoint's codec (encoder, codebooks, frozen decoder) with its post-filter, weights sd."""
    dec, _, _ = enc_mod.load_decoder(aq.enc_dec, aq.enc_dec_ema)
    rvq = enc_mod.ExtRVQ(teacher.quantizer, aq.enc_ext_n)
    G = enc_mod.EncoderCodec(enc_mod.build_encoder(train.arch_cfg(aq)), rvq, dec, True)
    G.pf = pf_lowtier.LowTierPF(dec.width, aq.pflt_dim, aq.pflt_blocks, None, dec.n_fft, aq.pflt_max_q,
                                getattr(aq, "pflt_mode", "complex"), getattr(aq, "pflt_mag_bound", 1.0),
                                getattr(aq, "pflt_bands", 48))
    G.load_state_dict(sd)
    return G.to(dev).eval()


def clip_input(e, dev):
    L = e["codes"].shape[-1] * train.HOP
    xt = torch.from_numpy(np.ascontiguousarray(e["x"][:, :L], dtype=np.float32))[:, None].to(dev)
    return torch.nn.functional.pad(xt, (0, L - xt.shape[-1]))


def wav(path, y):
    sf.write(path, np.ascontiguousarray(np.asarray(y, np.float32).T), 44100, subtype="FLOAT")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="the --pf-lowtier checkpoint (its G_ema is exported)")
    p.add_argument("--dec", required=True, help="the frozen decoder's export .pt (dac44_vocos_q16.pt)")
    p.add_argument("--enc", required=True, help="the encoder checkpoint whose args build the eval clips")
    p.add_argument("--bolero-codes", default="", help="folder with full_q16.0.npy / .1.npy (the Rust encoder's codes "
                   "of pf_lowtier/ravel_bolero.flac); without it only the eval-clip cases are written")
    p.add_argument("--out", required=True)
    c = p.parse_args()
    assert not os.path.exists(os.path.join(c.out, "pf_g7500.safetensors")), "exists; not overwriting"
    os.makedirs(os.path.join(c.out, "pf_ref"), exist_ok=True)
    torch.set_num_threads(8)
    dev = "cpu"
    teacher = load_teacher(dev)
    CKPT = c.ckpt
    sq = torch.load(CKPT, map_location="cpu", weights_only=False)
    aq = argparse.Namespace(**sq["args"])
    G = build(sq["G_ema"], aq, teacher, dev)
    pf, dec = G.pf, G.dec
    # 1. weights
    sd = {k: v.detach().float().contiguous().clone() for k, v in pf.state_dict().items()}
    assert set(sd) == {k[3:] for k in sq["G_ema"] if k.startswith("pf.")}
    assert all(torch.equal(sd[k], sq["G_ema"]["pf." + k]) for k in sd)
    st_path = os.path.join(c.out, "pf_g7500.safetensors")
    save_file(sd, st_path, metadata={"source": f"{CKPT} G_ema step {sq['step']}", "module": "pf_lowtier.LowTierPF"})
    # 2. config
    s16 = torch.load(c.dec, map_location="cpu", weights_only=False)
    dec_same = all(torch.equal(sq["G_ema"]["dec." + k], v) for k, v in s16.items())
    layers = [{"name": k, "shape": list(v.shape), "dtype": "f32"} for k, v in sd.items()]
    cfg = {
        "name": "pf_g7500", "source_checkpoint": CKPT, "source_sha256": sha(CKPT), "step": sq["step"], "weights": "G_ema",
        "owner_choice": "Run G 7.5k, EMA, then the grid-gated Janssen 3 dB declicker (declick_gated.py --th 3 "
                        "--ratio 1.5 --span 2), then picotone's look-ahead limiter",
        "frozen_decoder": {"file": os.path.basename(c.dec), "sha256": sha(c.dec),
                           "bit_identical_to_checkpoint_decoder": dec_same, "dim": dec.width, "blocks": len(dec.blocks),
                           "hop": 512, "n_fft": dec.n_fft, "frames_per_latent": dec.k},
        "tiers": {"post_filter_on": [1, 2, 3, 4, 5], "max_q": pf.max_q,
                  "bypass": "n_q > 5 (q6, q9, q16, ...): the post-filter is not run; S' = S, so the output is the frozen "
                            "decoder's, bit for bit",
                  "tier_source": "the coded channel's codebook count (picotone Parametric q2 codes the mid at 3 codebooks: "
                                 "run the post-filter at tier 3)"},
        "inputs": {
            "S": "the frozen decoder's head spectrum before its iSTFT, complex [1025 bins, T frames] (one frame per latent "
                 "frame): o = head(final_norm(trunk(z))) -> [T, 2050], reshape to [2050, T], logmag = o[:1025], "
                 "phase = o[1025:], mag = min(exp(logmag), 100), S = mag (cos phase + i sin phase), all f32",
            "h": "the decoder's final_norm output (the head's input), [T, 512] f32",
            "n_q": "codebook count of the frame's channel, 1..5",
        },
        "features": {"compression": pf_lowtier.COMP, "eps": 1e-5,
                     "formula": "c = S * (|S| + 1e-5)^(COMP - 1); feat = concat(Re c [1025], Im c [1025]) -> [2050, T] "
                                "(real parts first)"},
        "network": [
            "x = embed(feat) + h_embed(h^T) + tier_emb.weight[n_q][:, None]   (embed: Conv1d 2050->192 k1; h_embed: "
            "Conv1d 512->192 k1; tier_emb: Embedding 6x192, rows 1..5 used, row 0 unused)",
            "x = LayerNorm_192(x over channels, eps 1e-6) (norm)",
            "4 x ConvNeXt block i: y = dwconv_i(x) (depthwise Conv1d k7, padding 3, zero padded in time); "
            "y = LayerNorm(eps 1e-6) over channels; y = pw2(GELU_erf(pw1(y))) (Linear 192->576, exact erf GELU, "
            "Linear 576->192); x = x + gamma_i * y",
            "hh = LayerNorm_192(x) (final_norm, eps 1e-6)",
            "o = head(hh) (Linear 192 -> 1025): one value per bin per frame",
        ],
        "mask": {"mode": "magnitude only (phase of S kept exactly)", "gain_bound": pf.mag_bound,
                 "formula": "gain = exp(b * tanh(o)), b = 1.0 (so gain in [e^-1, e^1]); S' = S * gain (a real gain on "
                            "both Re and Im: |S'| = |S| gain, arg S' = arg S); no additive residual",
                 "step0_identity": "head weights and bias were zero-initialised: gain = 1 exactly"},
        "output": "y = the frozen decoder's iSTFT of S' (Hann window 2048, hop 512, 'same' padding: irfft per frame x "
                  "window, overlap-add, trim (n_fft - hop)/2 = 768 each end, divide by the overlap-added squared window); "
                  "T frames -> 512 T samples",
        "receptive_field_latent_frames": {
            "post_filter_alone": 3 * len(pf.blocks),
            "decoder_alone_samples": dec.receptive_field(),
            "decoder_plus_post_filter_samples": pf.receptive_field(dec),
            "note": "per side. Post-filter: 4 k7 depthwise convs (embed/h_embed/head are k1) = +-12 frames on S'. "
                    "S and h depend on +-27 frames of z (decoder embed k7 + 8 blocks), so S' needs +-39; the iSTFT "
                    "overlap adds 2: an exact chunked decode needs 41 latent frames (41 x 512 = 20992 samples, 476 ms) "
                    "of context per side, vs 29 for the decoder alone. Look-ahead for streaming: 41 frames (+ the "
                    "limiter's 220 samples)."},
        "params": int(sum(v.numel() for v in sd.values())),
        "layers": layers,
        "numerics": "f32 throughout; the training code runs the features, head, gain and iSTFT outside autocast; the "
                    "reference vectors are CPU fp32 (PyTorch); no TF32 involved",
    }
    json.dump(cfg, open(os.path.join(c.out, "pf_g7500.json"), "w"), indent=1)
    # 3. test vectors
    se = torch.load(c.enc, map_location="cpu", weights_only=False)
    clips = {e["name"]: e for e in train.prepare_eval(argparse.Namespace(**se["args"]), teacher, dev, print)
             if e["set"] in verdict.SETS and e["name"] == "new_26_rock"}
    T = int(SECONDS * 44100) // 512  # 344 frames, 176128 samples
    full = ([np.load(os.path.join(c.bolero_codes, f"full_q16.{ch}.npy")).astype(np.int64) for ch in (0, 1)]
            if c.bolero_codes else None)
    f0 = round(5.0 * 44100 / 512)
    xt = clip_input(clips["new_26_rock"], dev)
    _, rock_codes, _ = G.rvq(G.enc(xt), 16, st=False)  # the eval path's codes [2, 16, F] (channels as batch)
    cases = []
    for n in (3, 5):
        if full is not None:
            cases.append((f"bolero_005_065_q{n}", n, [torch.from_numpy(full[ch][f0:f0 + T, :n]).T for ch in (0, 1)],
                          "Bolero, Rust encoder q16 codes (--bolero-codes full_q16), frames 431..774 (5.0 s on), "
                          "first n codebooks"))
        cases.append((f"new_26_rock_q{n}", n, [rock_codes[ch, :n, :T] for ch in (0, 1)],
                      "new_26_rock, the eval path's codes (the q16 checkpoint's encoder), frames 0..343"))
    summary = {}
    for i, (name, n, codes, desc) in enumerate(cases):
        d = os.path.join(c.out, "pf_ref", name)
        os.makedirs(d, exist_ok=True)
        ys_fr, ys_pf, per = [], [], {}
        for ch in (0, 1):
            cd = codes[ch].to(torch.int64)
            z = codes_to_z(teacher, cd[None], n).float()
            h, S = pf_lowtier.dec_parts(dec, z)
            nq = torch.full((1,), n)
            y_fr = dec.istft(S)[0]
            y_ref = enc_mod.tier_cond_decode(dec, z, n)[0, 0]
            assert torch.equal(y_fr, y_ref), "dec.istft(dec_parts) != decoder forward"
            y_pf = pf(dec, z, n)[0, 0]
            Sp = pf.correct(S, h, nq)
            assert torch.equal(dec.istft(Sp)[0], y_pf)
            np.save(os.path.join(d, f"codes.ch{ch}.npy"), cd.numpy().astype(np.int32))
            np.save(os.path.join(d, f"z.ch{ch}.npy"), z[0].numpy())
            np.save(os.path.join(d, f"S_re.ch{ch}.npy"), S[0].real.contiguous().numpy())
            np.save(os.path.join(d, f"S_im.ch{ch}.npy"), S[0].imag.contiguous().numpy())
            np.save(os.path.join(d, f"h.ch{ch}.npy"), h[0].numpy())
            np.save(os.path.join(d, f"gain.ch{ch}.npy"), (Sp[0].abs() / S[0].abs().clamp(min=1e-30)).numpy())
            ys_fr.append(y_fr.numpy())
            ys_pf.append(y_pf.numpy())
            if i == 0 and ch == 0:  # one activation per layer, first case, left channel
                mag = S.abs()
                cc = S * (mag + 1e-5).pow(pf_lowtier.COMP - 1)
                feat = torch.cat([cc.real, cc.imag], 1)
                a = {"00_feat": feat, "01_embed": pf.embed(feat), "02_h_embed": pf.h_embed(h.transpose(1, 2)),
                     "03_tier_emb": pf.tier_emb(nq)[..., None]}
                x = a["01_embed"] + a["02_h_embed"] + a["03_tier_emb"]
                a["04_sum"] = x
                x = pf.norm(x.transpose(1, 2)).transpose(1, 2)
                a["05_norm"] = x
                for b_i, b in enumerate(pf.blocks):
                    x = b(x)
                    a[f"{6 + b_i:02d}_block{b_i}"] = x
                hh = pf.final_norm(x.transpose(1, 2))
                a["10_final_norm"] = hh.transpose(1, 2)
                o = pf.head(hh).transpose(1, 2)
                a["11_head"] = o
                a["12_gain"] = torch.exp(pf.mag_bound * torch.tanh(o))
                assert torch.equal(S * a["12_gain"], Sp)
                os.makedirs(os.path.join(d, "act"), exist_ok=True)
                for k, v in a.items():
                    np.save(os.path.join(d, "act", f"{k}.npy"), v[0].contiguous().numpy())  # [C, T]
                per = {k: list(v[0].shape) for k, v in a.items()}
        wav(os.path.join(d, "frozen.wav"), np.stack(ys_fr))
        wav(os.path.join(d, "pf.wav"), np.stack(ys_pf))
        meta = {"case": name, "n_q": n, "frames": T, "samples": 512 * T, "channels": 2, "source": desc,
                "files": {"codes.chC.npy": f"int32 [{n}, {T}]", "z.chC.npy": f"f32 [1024, {T}] (codes_to_z: DAC quantizer "
                          "out_proj sums)", "S_re/S_im.chC.npy": f"f32 [1025, {T}], the post-filter input spectrum",
                          "h.chC.npy": f"f32 [{T}, 512], the post-filter's second input", "gain.chC.npy": f"f32 [1025, {T}]",
                          "frozen.wav": "float32 stereo, the frozen decoder (= iSTFT(S)), unclipped",
                          "pf.wav": "float32 stereo, the post-filter output (iSTFT(S')), before limiter or declick"},
                "frozen_peak": float(np.abs(np.stack(ys_fr)).max()), "pf_peak": float(np.abs(np.stack(ys_pf)).max())}
        if per:
            meta["activations"] = {"dir": "act/", "channel": 0, "shapes": per,
                                   "note": "[channels, T] per layer; 03_tier_emb is [192, 1] (broadcast over T)"}
        json.dump(meta, open(os.path.join(d, "case.json"), "w"), indent=1)
        summary[name] = meta
        print(f"{name}: written ({T} frames, peaks frozen {meta['frozen_peak']:.3f} pf {meta['pf_peak']:.3f})", flush=True)
    print(f"{st_path}: {len(sd)} tensors, {cfg['params']:,} params; decoder == dac44_vocos_q16.pt: {dec_same}")


if __name__ == "__main__":
    main()
