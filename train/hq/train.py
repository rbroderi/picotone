"""Train picotone's HQ networks on DAC 44 kHz's codes, with DAC's own GAN recipe against the original audio:
the Vocos decoder (--arch vocos), the STFT encoder and extra codebooks in front of a frozen or fine-tuned decoder
(--arch encoder), the low-tier post-filter on the frozen decoder (--arch encoder --pf-lowtier), and the
post-filter / side-info research arms (--arch postfilter, sideinfo).

    uv run --extra cuda python train.py --run runs/vocos --arch vocos --manifest data/pd.jsonl \\
        --manifest data/jamendo.jsonl --eval-set data/pd.jsonl.eval.json --eval-set data/jamendo.jsonl.eval.json

Inputs: a frozen DAC encoder + quantiser encode each crop with `--enc-ctx` samples of context on both sides
(the codes then equal whole-file codes), the decoder decodes the crop's frames plus `--dec-ctx` frames each side,
and the losses see the centre only. Losses: 15 x multi-scale mel + 2 x feature matching + 1 x adversarial
(least squares as in DAC's code, or --gan hinge), optional output matching against the stock decoder
(--w-teacher-mel). AdamW (lr 1e-4, betas 0.8/0.99, decay 0.999996 per step) for generator and discriminator,
gradient clipping 1e3 / 10, bf16 autocast on CUDA (residual sums, the output head, STFTs and losses stay fp32; evals
run fully in fp32).

Every --eval-every steps: log-mel / MR-STFT / SNR / SI-SDR on the held-out clips, stock DAC's numbers on the
same clips (once, as the target line), the pilot verdict, and listening WAVs in <run>/listen/.
`--resume` continues from <run>/ckpt_last.pt with the identical batch sequence (batches are a function of the
step) and restored optimiser/scheduler/RNG state. `--bench` measures memory and ms/step for batch sizes.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

import clip_gap
import corpus
import data
import encoder as enc_mod
import losses
import metrics
import pf_lowtier
import postfilter
import sideinfo
import verdict
from nnutil import count_cost, n_params
from vocos_student import make_decoder
from teacher import HOP, N_CB, codes_to_z, encode_codes, load_teacher, stock_decode, stock_decode_bf16

SR = 44100


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("run")
    g.add_argument("--run", required=True, help="run directory (checkpoints, logs, listen/)")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--steps", type=int, default=50000)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--amp", choices=("auto", "bf16", "off"), default="auto", help="auto = bf16 on CUDA")
    g.add_argument("--bench", default="", help="comma list of batch sizes: measure memory + ms/step, then exit")
    g.add_argument("--eval-only", default="", metavar="CKPT",
                   help="load G from this checkpoint, run one eval (at --eval-n-q) into --run, then exit; "
                        "use a fresh --run so the logs don't mix with the training run's")
    g = ap.add_argument_group("data")
    g.add_argument("--manifest", action="append", default=[])
    g.add_argument("--eval-set", action="append", default=[])
    g.add_argument("--licences", default=corpus.DEFAULT_LICENCES,
                   help=f"classes or presets {tuple(corpus.LICENCE_PRESETS)}; default sharealike = pd,cc0,by,by-sa,fal")
    g.add_argument("--mix", default="auto", help="'auto' (by duration, floored) or e.g. pd=0.25,jamendo=0.75")
    g.add_argument("--mix-floor", type=float, default=0.2, help="auto mix: minimum weight of any source")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--tracks-per-batch", type=int, default=12)
    g.add_argument("--chunk-seconds", type=float, default=12.0)
    g.add_argument("--max-eval-clips", type=int, default=0)
    g.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                   help="rewrite manifest/eval paths made on another machine, e.g. "
                        "'E:\\Shared\\audio=/media/demo/External20TB/ai/audio' (repeatable)")
    g.add_argument("--eval-n-q", default="", help="codebook counts to evaluate, e.g. 5,9 (default: those of --n-q)")
    g.add_argument("--eval-kinds", default="",
                   help="also log the means over clips of these content types, e.g. speech (clip names like "
                        "old_18_speech), as extra sets 'kind:speech'; they don't count toward the score")
    g = ap.add_argument_group("model")
    g.add_argument("--arch", choices=("vocos", "postfilter", "sideinfo", "encoder"), default="vocos",
                   help="vocos = ConvNeXt blocks at the latent frame rate + an iSTFT head (vocos_student.py; --vocos-* flags); postfilter = "
                        "the frozen Vocos --pf-student followed by a trained STFT-domain corrector (postfilter.py); "
                        "sideinfo = that post-filter plus a learned variable-rate side code (sideinfo.py; --pf-* "
                        "and --si-* flags); encoder = a trained encoder in front of DAC's frozen quantiser and the "
                        "frozen --enc-dec decoder (encoder.py; --enc-* and --bwe-* flags)")
    g.add_argument("--enc-arch", choices=("stft", "conv"), default="stft",
                   help="--arch encoder: stft = Vocos-style STFT-input ConvNeXt at the frame rate; conv = DAC's "
                        "conv encoder layout at --enc-conv-width")
    g.add_argument("--enc-dim", type=int, default=512)
    g.add_argument("--enc-blocks", type=int, default=8)
    g.add_argument("--enc-mlp", type=int, default=0, help="0 = 3 x --enc-dim")
    g.add_argument("--enc-ffts", default="2048,512", help="STFT input resolutions (n_fft; hop min(512, n_fft/4))")
    g.add_argument("--enc-power", type=float, default=0.3, help="STFT input compression |S|^c")
    g.add_argument("--enc-embed-k", type=int, choices=(1, 3), default=1)
    g.add_argument("--enc-conv-width", type=int, default=16, help="--enc-arch conv: d_model (DAC's is 64)")
    g.add_argument("--enc-dec", default="", metavar="CKPT",
                   help="--arch encoder: the decoder (a train.py checkpoint of --arch dac or vocos, or an export "
                        ".pt), frozen unless --enc-train-dec; the file is only read")
    g.add_argument("--enc-dec-ema", action="store_true", help="take the decoder's G_ema weights (default G)")
    g.add_argument("--enc-phase", choices=("match", "e2e"), default="e2e",
                   help="match (phase A) = fit DAC's latent z and codes; e2e (phase B) = through the quantiser "
                        "and decoder against the true audio")
    g.add_argument("--enc-w-z", type=float, default=None,
                   help="weight of the z-matching loss (normalised MSE + 1 - cosine; default 1 in match, 0 in e2e)")
    g.add_argument("--enc-w-code", type=float, default=0.1, help="match: weight of the code cross-entropy")
    g.add_argument("--enc-tau", type=float, default=0.05, help="match: temperature of the cosine code logits")
    g.add_argument("--enc-w-commit", type=float, default=0.25, help="e2e: DAC's commitment loss weight")
    g.add_argument("--enc-soft-tau", type=float, default=0.0,
                   help="e2e: > 0 = soft-gradient quantiser (forward exactly the hard codes; backward through the "
                        "softmax(cos / tau)-weighted codebook mean) in place of plain straight-through; 0 = off")
    g.add_argument("--enc-w-cplx", type=float, default=0.0,
                   help="e2e: weight of the multi-resolution compressed complex STFT loss (TeacherComplexSTFT, "
                        "--cplx-*) against the true audio, phase included; only items above --bwe-max-q (BWE "
                        "items have no full-band reconstruction target); 0 = off")
    g.add_argument("--enc-train-dec", action="store_true", help="e2e: the joint fine-tune (decoder trains too)")
    g.add_argument("--enc-dec-lr-mult", type=float, default=0.3, help="--enc-train-dec: decoder lr / --lr")
    g.add_argument("--enc-freeze-enc", action="store_true",
                   help="--enc-train-dec: freeze the encoder (no gradients, not in the optimiser) and train only "
                        "the decoder, at --lr; needs --enc-w-z 0 --enc-w-commit 0 (constants then); off = as before")
    g.add_argument("--enc-ext-cb", default="",
                   help="patch q12_ext: extra DAC-style codebooks after q9 (encoder.ExtRVQ), initialised from this "
                        "file (build_ext_cb.py); they train with the decoder; needs --enc-freeze-enc; --n-q may go "
                        "up to 9 + their count; '' = off (as before)")
    g.add_argument("--enc-freeze-dec", action="store_true",
                   help="patch q12c: with --enc-ext-cb, freeze the decoder too (no gradients, not in the optimiser): "
                        "only the extra codebooks train; --init-opt-from then restores only opt_d; off = as before")
    g.add_argument("--enc-ext-lr", type=float, default=0.0,
                   help="patch q12c: the extra codebooks' lr (0 = --lr, as before); D keeps --lr")
    g.add_argument("--enc-ext-grad", choices=("st", "out"), default="st",
                   help="patch q12c: st = DAC's straight-through (as before); out = the code's value with gradients "
                        "from the output losses to the entries, out_proj and (straight-through) in_proj")
    # patch pf_lowtier (POSTFILTER_LOWTIER_RUN.md): a post-filter on the frozen decoder's head spectrum at <= max-q
    g.add_argument("--pf-lowtier", action="store_true",
                   help="patch pf_lowtier: freeze the decoder and codebooks 10-16 too; train only a post-filter (G.pf) "
                        "that corrects the decoder's frame spectra before its iSTFT, at <= --pflt-max-q codebooks")
    g.add_argument("--pflt-dim", type=int, default=384)
    g.add_argument("--pflt-blocks", type=int, default=6)
    g.add_argument("--pflt-max-q", type=int, default=5)
    g.add_argument("--pflt-mode", choices=("complex", "mag", "band"), default="complex",
                   help="patch pf_lowtier: complex = complex mask + residual (as before); mag = magnitude only, "
                        "|S'| = |S| exp(b tanh(m)), the phase of S kept exactly (owner 2026-11-27, trial E)")
    g.add_argument("--pflt-mag-bound", type=float, default=1.0, help="b for --pflt-mode mag / band")
    g.add_argument("--pflt-bands", type=int, default=48,
                   help="--pflt-mode band (owner 2026-11-27, trial F): one log-gain per band, n triangular bands on the "
                        "mel scale interpolated to the bins; phase of S kept")
    g.add_argument("--w-gain-tv", type=float, default=0.0,
                   help="patch pf_lowtier (--pflt-mode band): temporal smoothness of the band log-gains, mean "
                        "|g_t - g_t-1| over the loss segment (0 = off)")
    g.add_argument("--fresh-d", action="store_true",
                   help="--init-from: do not load the checkpoint's D (it stays freshly initialised; owner "
                        "2026-11-27, trial G)")
    g.add_argument("--gan-g-start", type=int, default=-1,
                   help="step from which G gets the adversarial and feature-matching terms (-1 = --gan-start, as "
                        "before); D trains from --gan-start (owner 2026-11-27, trial G)")
    g.add_argument("--w-consist", type=float, default=0.0,
                   help="patch pf_lowtier: frame-consistency loss ||STFT(iSTFT(S')) - S'||_1 / ||S'||_1 (0 = off)")
    g.add_argument("--w-short", type=float, default=0.0,
                   help="patch pf_lowtier: log-magnitude L1 at n_fft 128/256/512 (0 = off); band-limited like the mel "
                        "loss on the BWE items")
    g.add_argument("--enc-dac-enc", action="store_true",
                   help="patch armA_ctrl: DAC's own encoder (frozen, fp32) in the encoder slot, for a decoder-only "
                        "control on DAC-encoder codes; needs --enc-freeze-enc; off = as before")
    g.add_argument("--enc-mix-dac", type=float, default=0.0,
                   help="patch ctrl3: this share of each batch's items (chosen per step, seeded) take DAC's encoder "
                        "(a frozen fp32 copy, kept outside G) instead of the trained one, with tiers drawn from "
                        "--n-q-dac; the --w-teacher-* terms apply to those items only, weighted by their share; needs "
                        "--enc-freeze-enc; 0 = off (as before)")
    g.add_argument("--n-q-dac", default="", help="--enc-mix-dac: the DAC-code items' tier list (like --n-q)")
    g.add_argument("--dec-tier-cond", action="store_true",
                   help="patch tiercond: a learned embedding per codebook count (1..12), zero-initialised, added to "
                        "the decoder's input (the codebook sum) in every decode; --arch encoder only; off = as before")
    g.add_argument("--enc-ext-w-commit", type=float, default=0.25,
                   help="--enc-ext-cb: the extra codebooks' commitment weight (DAC 0.25)")
    g.add_argument("--enc-ext-w-codebook", type=float, default=1.0,
                   help="--enc-ext-cb: the extra codebooks' codebook-loss weight (DAC 1.0)")
    g.add_argument("--bwe-cutoff", default="0",
                   help="SBR-style bandwidth extension at the low tiers: reconstruction losses (mel) only below this "
                        "many Hz for items at <= --bwe-max-q codebooks; one value, or a comma list for q1, q2, ...; "
                        "0 = off. Adversarial and feature matching stay full band")
    g.add_argument("--bwe-max-q", type=int, default=3)
    g.add_argument("--bwe-hf-weight", type=float, default=0.0,
                   help="weight of the full-band mel term kept for BWE items (0 = drop it)")
    g.add_argument("--bwe-eval-cutoff", type=float, default=8000.0,
                   help="--arch encoder evals: also log the log-mel below this many Hz (logmel_lb)")
    g.add_argument("--vocos-dim", type=int, default=512)
    g.add_argument("--vocos-blocks", type=int, default=8)
    g.add_argument("--vocos-mlp", type=int, default=0, help="ConvNeXt MLP width (0 = 3 x --vocos-dim)")
    g.add_argument("--vocos-hop", type=int, choices=(512, 256), default=512,
                   help="iSTFT hop: 512 = n_fft 2048, one STFT frame per latent frame; 256 = n_fft 1024, two")
    g.add_argument("--pf-student", default="",
                   help="--arch postfilter/sideinfo: the frozen decoder, a train.py checkpoint of --arch vocos "
                        "(only read)")
    g.add_argument("--si-codes-enc", default="", metavar="CKPT",
                   help="--arch postfilter/sideinfo: take the codes from this trained encoder (an --arch encoder "
                        "checkpoint: its G_ema enc.* + DAC's quantiser, fp32) instead of DAC's encoder, in training "
                        "and evals; stock DAC lines keep DAC's own codes. Off (default) = DAC's encoder, as before")
    g.add_argument("--pf-dim", type=int, default=None, help="default 384; 512 with --si-mode ceiling/constrained")
    g.add_argument("--pf-blocks", type=int, default=None, help="default 6; 8 with --si-mode ceiling/constrained")
    g.add_argument("--pf-mlp", type=int, default=0, help="0 = 3 x --pf-dim")
    g.add_argument("--pf-hop", type=int, choices=(512, 256), default=512)
    g.add_argument("--pf-z", action="store_true", help="also feed the quantised latent z (at frame rate)")
    g.add_argument("--si-mode", choices=("vr", "ceiling", "constrained"), default="vr",
                   help="--arch sideinfo: vr = one variable-rate model, t sampled per item, lambda = --si-lam "
                        "(the reference); ceiling = stage 1, one fixed rate setting and a tiny fixed lambda "
                        "(--si-lam1): the best sound q9 + side can reach; constrained = stage 2, fine-tuned from a "
                        "ceiling checkpoint (--si-init) with lambda adapted by dual ascent to the least rate whose "
                        "distortion stays within --si-eps of stage 1's")
    g.add_argument("--si-nq", type=int, default=None,
                   help="train and evaluate at this one codebook count only (overrides --n-q and --eval-n-q); "
                        "default 9 with --si-mode ceiling/constrained, off with vr")
    g.add_argument("--si-ch", type=int, default=None, help="side-latent channels (default 8; 16 in ceiling/constrained)")
    g.add_argument("--si-down", type=int, choices=(1, 2), default=1,
                   help="side-latent rate: 1 = every frame (86 Hz), 2 = every other frame")
    g.add_argument("--si-enc-dim", type=int, default=None, help="default 384; 512 in ceiling/constrained")
    g.add_argument("--si-enc-blocks", type=int, default=None, help="default 6; 8 in ceiling/constrained")
    g.add_argument("--si-em-dim", type=int, default=192, help="entropy model width (decoder side)")
    g.add_argument("--si-em-blocks", type=int, default=2)
    g.add_argument("--si-lam", type=float, nargs=2, default=(0.02, 20.0), metavar=("MIN", "MAX"),
                   help="vr: rate weight per side kbps, sampled log-uniformly per item")
    g.add_argument("--si-lam1", type=float, default=0.002,
                   help="ceiling: the fixed rate weight per side kbps (10x below vr's smallest), there only to "
                        "keep the code finite; constrained: lambda's start value when the --si-init checkpoint has "
                        "none")
    g.add_argument("--si-em-fit", type=float, default=None,
                   help="weight per kbps of the entropy model's fit term on the detached code (default 1 in "
                        "ceiling/constrained, 0 in vr)")
    g.add_argument("--si-p-off", type=float, default=None,
                   help="share of items with the side code off (0 bits); default 0.15 vr, 0.1 ceiling (so the "
                        "zero-bit post-filter point is trained), 0 constrained")
    g.add_argument("--si-eval-t", default="0.5", help="rate setting t in [0, 1] for the training evals, or 'off' "
                                                      "(ceiling/constrained: any value = on)")
    g.add_argument("--si-rate-warmup", type=int, default=None,
                   help="the rate term is 0 for this many steps, then ramps linearly to full over as many again "
                        "(default 2000; 0 with constrained). With the rate on from step 0, the code collapses to "
                        "zero bits before the (zero-initialised) decoder can use it: Adam moves the encoder at full "
                        "lr even under a small rate weight")
    g.add_argument("--si-init", default="", metavar="CKPT",
                   help="constrained: start G from this ceiling checkpoint's raw weights (only read), and take D* "
                        "(its slow average of the training distortion) and lambda from it; ignored with --resume")
    g.add_argument("--si-lam-start", type=float, default=0.0,
                   help="constrained: lambda's start value (0 = the --si-init checkpoint's); with --si-dual-lr 0 "
                        "it stays there, which is the fixed-lambda fallback")
    g.add_argument("--si-eps", type=float, default=0.01,
                   help="constrained: the distortion target is D* x (1 + eps)")
    g.add_argument("--si-dstar", type=float, default=0.0, help="constrained: D* by hand (0 = from --si-init)")
    g.add_argument("--si-dual-lr", type=float, default=0.5,
                   help="constrained: log lambda += this x (target - D) / target per step, D = the fast average "
                        "of the training distortion; the step is clipped to +-0.01 and lambda to [1e-4, 100]")
    g.add_argument("--init-d", default="", metavar="CKPT",
                   help="start the discriminator from this checkpoint's D (only read), e.g. the decoder's own "
                        "ckpt_last; ignored with --resume")
    g.add_argument("--n-q", default="5", help="codebooks: one count, or a set for quantiser dropout, e.g. 4,5,6,9")
    g = ap.add_argument_group("batch")
    g.add_argument("--batch", type=int, default=32)
    g.add_argument("--seg-frames", type=int, default=32, help="loss segment in 512-sample frames (32 = 0.37 s)")
    g.add_argument("--enc-ctx", type=int, default=4096, help="encoder context samples each side (mult. of 512)")
    g.add_argument("--dec-ctx", type=int, default=2, help="decoder context frames each side")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--betas", type=float, nargs=2, default=(0.8, 0.99))
    g.add_argument("--gamma", type=float, default=0.999996)
    g.add_argument("--lr-schedule", choices=("exp", "linear", "warmconst", "warmlin", "warmcos"), default="exp",
                   help="exp = x --gamma per step; linear = from --lr down to 0 at --steps (for fine-tunes); "
                        "warmconst (patch ctrl2) = linear from 0 to --lr over --lr-warmup steps, then constant")
    g.add_argument("--lr-warmup", type=int, default=0, help="--lr-schedule warmconst/warmlin: warm-up steps")
    g.add_argument("--lr-d", type=float, default=0.0, help="patch q16: D's lr (0 = --lr, as before)")
    g.add_argument("--lr-schedule-d", default="", help="patch q16: D's schedule ('' = --lr-schedule, as before)")
    g.add_argument("--lr-warmup-d", type=int, default=-1, help="patch q16: D's warm-up (-1 = --lr-warmup)")
    g.add_argument("--stop-at", type=int, default=0,
                   help="patch q16: exit cleanly after the checkpoint at this step (schedule still spans --steps; "
                        "continue with --resume); 0 = off")
    g.add_argument("--init-opt-from", default="", metavar="CKPT",
                   help="patch ctrl2: copy the Adam moments of CKPT's opt_g (onto the decoder's parameters, in order) "
                        "and opt_d (onto D's) into the fresh optimisers; shapes checked; lr/betas stay this run's; "
                        "'' = off (fresh optimisers, as before)")
    g.add_argument("--init-from", default="", metavar="CKPT",
                   help="fine-tune: start G, G_ema (if present) and D from this checkpoint (only read), with a "
                        "fresh optimiser, schedule and step 0 in a new --run; ignored with --resume")
    g.add_argument("--init-from-ema", action="store_true",
                   help="--init-from: load the checkpoint's G_ema (not G) into G; G_ema starts from it too")
    g.add_argument("--clip-g", type=float, default=1e3)
    g.add_argument("--clip-d", type=float, default=10.0)
    g.add_argument("--gan", choices=("ls", "hinge"), default="ls")
    g.add_argument("--w-mel", type=float, default=15.0)
    g.add_argument("--w-fm", type=float, default=2.0)
    g.add_argument("--w-adv", type=float, default=1.0)
    g.add_argument("--w-teacher-mel", type=float, default=0.0, help="mel loss vs the stock decoder's output")
    g.add_argument("--w-teacher-wave", type=float, default=0.0, help="L1 waveform loss vs the stock output")
    g.add_argument("--w-teacher-cplx", type=float, default=0.0,
                   help="multi-resolution compressed complex STFT loss vs the stock decoder's output "
                        "(TeacherComplexSTFT): matches the teacher's phase as well as its magnitude")
    g.add_argument("--cplx-windows", default="256,512,1024,2048",
                   help="its STFT windows (hop = window / 4); 2048 is the hop-512 iSTFT head's own")
    g.add_argument("--cplx-power", type=float, default=0.3, help="its magnitude compression |S|^c")
    g.add_argument("--cplx-alpha", type=float, default=0.3,
                   help="its complex share: alpha x complex MSE + (1 - alpha) x compressed-magnitude MSE")
    g.add_argument("--teacher-amp", choices=("bf16", "fp32"), default="bf16",
                   help="teacher target precision when bf16 autocast is on: bf16 body + fp32 output head "
                        "(about 0.07 dB log-mel off fp32, faster) or fully fp32")
    g.add_argument("--w-lf", type=float, default=100.0,
                   help="L1 on the 1024-sample moving average (DC and sub-bass below ~40 Hz), which the mel loss "
                        "and the discriminators barely see; without it the output can drift a DC offset")
    g.add_argument("--gan-start", type=int, default=0, help="steps of mel-only training before the GAN")
    g = ap.add_argument_group("schedule")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--eval-every", type=int, default=5000)
    g.add_argument("--ckpt-every", type=int, default=2500)
    g.add_argument("--keep-every", type=int, default=25000, help="also keep ckpt_<step>.pt at multiples")
    g.add_argument("--listen", type=int, default=6, help="eval clips written as WAVs at each eval")
    g.add_argument("--listen-names", default="",
                   help="write listening WAVs for these eval clips (comma list of names) instead of the first "
                        "--listen")
    g.add_argument("--listen-every", type=int, default=0,
                   help="write listening WAVs only at evals on multiples of this (0 = every eval)")
    g.add_argument("--g-ema", type=float, default=0.999,
                   help="decay of a moving average of G's weights, evaluated beside G (0 = off); the "
                        "checkpoints keep both, ckpt_best.pt keeps whichever scored better")
    g.add_argument("--stop-plateau", type=float, default=0.0,
                   help="stop once the best score (mean picotone log-mel over --eval-n-q, the average of the "
                        "last 4 evals) improved by less than this over --plateau-steps (0 = never)")
    g.add_argument("--plateau-steps", type=int, default=20000)
    g.add_argument("--stop-when-pass", action="store_true",
                   help="stop once every --eval-n-q tier passes the full-run rule")
    g.add_argument("--pilot-steps", type=int, default=50000, help="verdict rule: pilot before, full after")
    g.add_argument("--picotone-json", default="", help="compare_picotone.py output for the verdict")
    a = ap.parse_args(argv)
    fixed = a.arch == "sideinfo" and a.si_mode != "vr"

    def dflt(name, vr, fx):
        if getattr(a, name) is None:
            setattr(a, name, fx if fixed else vr)

    dflt("pf_dim", 384, 512)
    dflt("pf_blocks", 6, 8)
    dflt("si_ch", 8, 16)
    dflt("si_enc_dim", 384, 512)
    dflt("si_enc_blocks", 6, 8)
    dflt("si_em_fit", 0.0, 1.0)
    dflt("si_p_off", 0.15, 0.0 if a.si_mode == "constrained" else 0.1)
    dflt("si_rate_warmup", 2000, 0 if a.si_mode == "constrained" else 2000)
    dflt("si_nq", 0, 9)
    if a.arch == "sideinfo" and a.si_nq:
        a.n_q = a.eval_n_q = str(a.si_nq)
    if a.arch == "sideinfo" and a.si_mode == "constrained" and not (a.si_init or a.resume or a.eval_only
                                                                     or a.bench):
        ap.error("--si-mode constrained needs --si-init CKPT (a --si-mode ceiling checkpoint)")
    if a.init_from_ema and not a.init_from:
        ap.error("--init-from-ema needs --init-from")
    if a.enc_freeze_enc and not (a.arch == "encoder" and a.enc_train_dec and a.enc_phase == "e2e"):
        ap.error("--enc-freeze-enc needs --arch encoder --enc-phase e2e --enc-train-dec")
    if a.enc_freeze_enc and (a.enc_w_z or 0) != 0 or a.enc_freeze_enc and a.enc_w_commit != 0:
        ap.error("--enc-freeze-enc needs --enc-w-z 0 --enc-w-commit 0 (the encoder does not train)")
    if a.init_from and os.path.abspath(os.path.dirname(a.init_from)) == os.path.abspath(a.run):
        ap.error("--init-from wants a new --run folder, not the source checkpoint's")
    a.n_q_list = [int(v) for v in a.n_q.split(",")]
    a.path_map = data.parse_path_map(a.path_map)
    a.eval_n_q = sorted({int(v) for v in a.eval_n_q.split(",")} if a.eval_n_q else set(a.n_q_list))
    assert a.enc_ctx % HOP == 0 and a.enc_ctx // HOP >= a.dec_ctx
    if a.arch in ("postfilter", "sideinfo"):
        if not os.path.exists(a.pf_student):
            ap.error(f"--pf-student {a.pf_student!r} not found")
        a.pf_student_sha = postfilter.file_sha16(a.pf_student)
        ps = torch.load(a.pf_student, map_location="cpu", weights_only=False)
        a.pf_student_arch = ps.get("arch") if isinstance(ps, dict) and "G" in ps else None
        del ps
        if (a.pf_student_arch or {}).get("arch") != "vocos":
            ap.error(f"--pf-student {a.pf_student}: not a train.py checkpoint of --arch vocos")
    if a.si_codes_enc:
        if a.arch not in ("postfilter", "sideinfo"):
            ap.error("--si-codes-enc needs --arch postfilter or sideinfo")
        if not os.path.exists(a.si_codes_enc):
            ap.error(f"--si-codes-enc {a.si_codes_enc} not found")
        a.si_codes_enc_sha = postfilter.file_sha16(a.si_codes_enc)
        ectx = torch.load(a.si_codes_enc, map_location="cpu", weights_only=False)["args"]["enc_ctx"]
        if a.enc_ctx < ectx:
            ap.error(f"--si-codes-enc was trained with --enc-ctx {ectx}: pass --enc-ctx {ectx} (or more)")
    if a.arch == "sideinfo" and a.pf_hop != 512:
        ap.error("--arch sideinfo needs --pf-hop 512")
    if a.enc_w_z is None:
        a.enc_w_z = 1.0 if a.enc_phase == "match" else 0.0
    if a.arch == "encoder":
        if not os.path.exists(a.enc_dec):
            ap.error(f"--arch encoder needs --enc-dec CKPT (a decoder); {a.enc_dec!r} not found")
        if (a.w_teacher_mel > 0 or a.w_teacher_wave > 0 or a.w_teacher_cplx > 0) and not (a.enc_dac_enc
                                                                                         or a.enc_mix_dac > 0):
            ap.error("--arch encoder trains against the true audio: the --w-teacher-* terms are not supported "
                     "(except with --enc-dac-enc, patch ctrl2: DAC codes, so DAC's decoder output is a valid target)")
        if a.w_teacher_wave > 0:
            ap.error("--w-teacher-wave is not supported with --arch encoder")
        if a.enc_train_dec and a.enc_phase != "e2e":
            ap.error("--enc-train-dec needs --enc-phase e2e")
        if a.enc_soft_tau < 0 or (a.enc_soft_tau > 0 and a.enc_phase != "e2e"):
            ap.error("--enc-soft-tau must be >= 0, and > 0 only with --enc-phase e2e")
        if a.enc_w_cplx < 0 or (a.enc_w_cplx > 0 and a.enc_phase != "e2e"):
            ap.error("--enc-w-cplx must be >= 0, and > 0 only with --enc-phase e2e")
        a.enc_dec_sha = a.pf_student_sha = postfilter.file_sha16(a.enc_dec)
    a.enc_ext_n = 0
    if a.dec_tier_cond and a.arch != "encoder":
        ap.error("--dec-tier-cond needs --arch encoder")
    if a.enc_mix_dac:  # patch ctrl3
        if not (a.arch == "encoder" and a.enc_freeze_enc and not a.enc_dac_enc and 0 < a.enc_mix_dac < 1):
            ap.error("--enc-mix-dac needs --arch encoder --enc-freeze-enc, not --enc-dac-enc, and a share in (0, 1)")
        if not a.n_q_dac:
            ap.error("--enc-mix-dac needs --n-q-dac")
        a.n_q_dac_list = [int(v) for v in a.n_q_dac.split(",")]
        if max(a.n_q_dac_list) > N_CB:
            ap.error(f"--n-q-dac above {N_CB}")
    if (a.enc_freeze_dec or a.enc_ext_lr or a.enc_ext_grad != "st") and not a.enc_ext_cb:
        ap.error("--enc-freeze-dec / --enc-ext-lr / --enc-ext-grad need --enc-ext-cb")
    if a.enc_dac_enc and not (a.arch == "encoder" and a.enc_freeze_enc and not a.init_from):
        ap.error("--enc-dac-enc needs --arch encoder --enc-freeze-enc and no --init-from (the decoder comes from "
                 "--enc-dec)")
    if a.enc_ext_cb:  # patch q12_ext
        if not (a.arch == "encoder" and a.enc_freeze_enc):
            ap.error("--enc-ext-cb needs --arch encoder --enc-freeze-enc")
        if not os.path.exists(a.enc_ext_cb):
            ap.error(f"--enc-ext-cb {a.enc_ext_cb} not found")
        a.enc_ext_sha = postfilter.file_sha16(a.enc_ext_cb)
        a.enc_ext_n = int(torch.load(a.enc_ext_cb, map_location="cpu", weights_only=False)["n_ext"])
        if a.enc_freeze_dec and a.enc_mix_dac:
            ap.error("--enc-freeze-dec: no --enc-mix-dac (only the arm-4-code items above q9 train the codebooks)")
        if max(a.n_q_list) > N_CB + a.enc_ext_n:
            ap.error(f"--n-q goes to {max(a.n_q_list)}, but there are only {N_CB} + {a.enc_ext_n} codebooks")
    if a.pf_lowtier and not (a.arch == "encoder" and a.enc_phase == "e2e" and a.enc_freeze_enc and not a.enc_mix_dac
                             and not a.enc_freeze_dec and max(a.n_q_list) <= a.pflt_max_q):
        ap.error("--pf-lowtier: needs --arch encoder --enc-phase e2e --enc-freeze-enc, no --enc-mix-dac / --enc-freeze-dec, "
                 "and --n-q tiers <= --pflt-max-q")
    if a.w_gain_tv and not (a.pf_lowtier and a.pflt_mode == "band"):
        ap.error("--w-gain-tv needs --pf-lowtier --pflt-mode band")
    if a.fresh_d and not a.init_from:
        ap.error("--fresh-d needs --init-from")
    if (a.w_consist or a.w_short) and not a.pf_lowtier:
        ap.error("--w-consist / --w-short: only with --pf-lowtier")
    if a.lr_schedule == "warmcos" and not a.pf_lowtier:
        ap.error("--lr-schedule warmcos: only with --pf-lowtier")
    if max(a.eval_n_q) > N_CB:
        ap.error(f"--eval-n-q above {N_CB} is not supported in training evals (stock DAC has {N_CB} codebooks)")
    if a.arch != "encoder" and max(a.n_q_list) > N_CB:
        ap.error(f"--n-q above {N_CB} needs --enc-ext-cb")
    return a


def arch_cfg(a):
    """The model's architecture, as stored in checkpoints."""
    if a.arch == "encoder":  # the decoder's own config is filled in by main() when it loads --enc-dec
        enc = {"enc": "dac"} if getattr(a, "enc_dac_enc", False) else ({
                "enc": "stft", "dim": a.enc_dim, "blocks": a.enc_blocks, "mlp": a.enc_mlp or 3 * a.enc_dim,
                "ffts": [int(v) for v in a.enc_ffts.split(",")], "power": a.enc_power, "embed_k": a.enc_embed_k}
               if a.enc_arch == "stft" else {"enc": "conv", "conv_width": a.enc_conv_width})
        return {"arch": "encoder", **enc, "dec": getattr(a, "enc_dec_arch", None),
                "dec_width": getattr(a, "enc_dec_width", None), "dec_sha": a.enc_dec_sha,
                "dec_ema": a.enc_dec_ema, "dec_path": a.enc_dec,
                **({"ext_cb": a.enc_ext_n, "ext_sha": a.enc_ext_sha} if getattr(a, "enc_ext_n", 0) else {}),
                **({"tier_cond": True} if getattr(a, "dec_tier_cond", False) else {}),
                **({"pf_lowtier": {"dim": a.pflt_dim, "blocks": a.pflt_blocks, "max_q": a.pflt_max_q,
                                   **({"mode": a.pflt_mode, "mag_bound": a.pflt_mag_bound}
                                      if getattr(a, "pflt_mode", "complex") != "complex" else {}),
                                   **({"n_bands": a.pflt_bands} if getattr(a, "pflt_mode", "") == "band" else {})}}
                   if getattr(a, "pf_lowtier", False) else {})}
    if a.arch in ("postfilter", "sideinfo"):
        pf = {"arch": "postfilter", "dim": a.pf_dim, "blocks": a.pf_blocks, "mlp": a.pf_mlp or 3 * a.pf_dim,
              "hop": a.pf_hop, "use_z": a.pf_z, "student": a.pf_student, "student_sha": a.pf_student_sha}
        pf["student_arch"] = a.pf_student_arch
        if getattr(a, "si_codes_enc", ""):  # patch si_enc: the codes' encoder (absent = DAC's, as before)
            pf["codes_enc"], pf["codes_enc_sha"] = a.si_codes_enc, a.si_codes_enc_sha
        if a.arch == "postfilter":
            return pf
        si = {**pf, "arch": "sideinfo", "ch": a.si_ch, "down": a.si_down, "enc_dim": a.si_enc_dim,
              "enc_blocks": a.si_enc_blocks, "em_dim": a.si_em_dim, "em_blocks": a.si_em_blocks}
        if a.si_mode == "vr":
            return {**si, "lam": list(a.si_lam), "p_off": a.si_p_off}
        # Stage 1 and stage 2 share one architecture (the objective is not part of it), so --si-init and
        # sideinfo_curve.py load either; the off share is training-only here (eval sets on / off itself).
        return {**si, "mode": "fixed", "nq": a.si_nq, "p_off": 0.0}
    return {"arch": "vocos", "dim": a.vocos_dim, "blocks": a.vocos_blocks, "mlp": a.vocos_mlp or 3 * a.vocos_dim,
            "hop": a.vocos_hop}


def check_arch(a, s, what):
    """Stop when checkpoint s was trained with another decoder than the command line asks for. A post-filter's
    student is compared by its file hash, not its path (the path differs between machines)."""
    def strip(c):
        return {k: v for k, v in c.items() if k not in ("student", "dec_path", "codes_enc", "ext_cb", "ext_sha", "tier_cond", "pf_lowtier")} if c else c

    have, want = strip(s.get("arch")), strip(arch_cfg(a))
    if have != want:
        raise SystemExit(f"{what} holds {have}, but the command line asks for {want}: pass the same --arch/--vocos-* "
                         f"flags")


class Log:
    def __init__(self, run):
        self.f = open(os.path.join(run, "log.txt"), "a", encoding="utf-8")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def jsonl(run, name, row):
    with open(os.path.join(run, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def amp_on(a):
    return a.amp == "bf16" or (a.amp == "auto" and a.device.startswith("cuda"))


def autocast(a):
    return torch.autocast(device_type="cuda" if a.device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                          enabled=amp_on(a))


def load_g(a, G, sd):
    """load_state_dict; with --enc-ext-cb (patch q12_ext) a source without the extra codebooks keeps their
    initialisation, and with --dec-tier-cond (patch tiercond) one without the tier embedding keeps its zero init (only
    rvq.ext.* / dec.tier_emb.* may be missing; nothing may be unexpected)."""
    may = tuple(p for p, on in (("rvq.ext.", getattr(a, "enc_ext_n", 0)), ("dec.tier_emb.", getattr(a, "dec_tier_cond",
                                                                                                     False)),
                                ("pf.", getattr(a, "pf_lowtier", False))) if on)
    if not may or all(any(k.startswith(p) for k in sd) for p in may):
        return G.load_state_dict(sd)
    r = G.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys and all(k.startswith(may) for k in r.missing_keys), r
    return r


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ one training step

PF_STFT = [None]  # patch pf_lowtier: the decoder's STFT ("same" padding), for --w-consist
CPLX_LOSS = [None]  # main() puts the TeacherComplexSTFT here when --w-teacher-cplx > 0 (not in args: it's saved)


class TeacherComplexSTFT(torch.nn.Module):
    """Multi-resolution compressed complex STFT loss (Braun & Tashev 2021, as used for speech enhancement):
    per resolution, S_c = S |S|^(c - 1) (magnitude compressed to |S|^c, phase kept), and

        L = alpha * mean |S_c(y) - S_c(t)|^2  +  (1 - alpha) * mean (|S_c(y)| - |S_c(t)|)^2,

    averaged over the resolutions. Compression evens out loud and quiet bins (with c = 0.3 a 60 dB level range
    becomes 18 dB), so neither loud frames nor the low band dominate, and a phase error costs in proportion to
    the compressed level of its bin. Against the teacher (stock DAC on the same codes), the phase target is
    reachable: it is a function of the codes, unlike the original's. (|S|^2 + 1e-8) keeps the gradient finite at
    |S| = 0. Computed in fp32."""

    def __init__(self, windows=(256, 512, 1024, 2048), power=0.3, alpha=0.3):
        super().__init__()
        self.windows, self.power, self.alpha = tuple(windows), power, alpha
        for w in self.windows:
            self.register_buffer(f"win{w}", torch.hann_window(w), persistent=False)

    def comp(self, x, w):
        s = torch.stft(x.reshape(-1, x.shape[-1]).float(), w, w // 4, window=getattr(self, f"win{w}"),
                       return_complex=True)
        p2 = s.real.square() + s.imag.square() + 1e-8
        return s * p2.pow((self.power - 1) / 2), p2.pow(self.power / 2)

    def forward(self, y, t):
        with torch.autocast(device_type=y.device.type, enabled=False):
            tot = 0.0
            for w in self.windows:
                sy, my = self.comp(y, w)
                st, mt = self.comp(t, w)
                cplx = (sy - st).abs().square().mean()
                mag = (my - mt).square().mean()
                tot = tot + self.alpha * cplx + (1 - self.alpha) * mag
            return tot / len(self.windows)


DAC_ENC = [None]  # patch ctrl3: the frozen DAC encoder for --enc-mix-dac items (outside G: not in checkpoints)
CODE_ENC = {}  # patch si_enc: (path, device) -> (encoder, FrozenRVQ), loaded once


def code_encoder(path, teacher, dev):
    k = (os.path.abspath(path), str(dev))
    if k not in CODE_ENC:
        s = torch.load(path, map_location="cpu", weights_only=False)
        w = "G_ema" if s.get("G_ema") is not None else "G"
        enc = enc_mod.build_encoder(s["arch"])
        enc.load_state_dict({n[4:]: v for n, v in s[w].items() if n.startswith("enc.")})
        enc = enc.float().to(dev).eval()
        rvq = enc_mod.FrozenRVQ(teacher.quantizer).to(dev).eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        CODE_ENC[k] = (enc, rvq)
    return CODE_ENC[k]


@torch.no_grad()
def get_codes(a, teacher, x):
    """x [B, 1, T] (T a multiple of 512) -> codes [B, 9, T/512]: DAC's encoder, or --si-codes-enc's (fp32)."""
    path = getattr(a, "si_codes_enc", "")
    if not path:
        return encode_codes(teacher, x)
    enc, rvq = code_encoder(path, teacher, x.device)
    with torch.autocast(device_type=x.device.type, enabled=False):
        codes = rvq(enc(x.float()), N_CB, st=False)[1]
    assert codes.shape[-1] == x.shape[-1] // HOP, (codes.shape, x.shape)
    return codes


def forward_g(a, teacher, G, batch, dev):
    x = batch["x"].to(dev, non_blocking=True)
    n_q = batch["n_q"].to(dev)
    S, ce, cd = a.seg_frames, a.enc_ctx // HOP, a.dec_ctx
    codes = get_codes(a, teacher, x)
    z = codes_to_z(teacher, codes[:, :, ce - cd: ce + S + cd], n_q)
    with autocast(a):
        if a.arch == "sideinfo":  # the side encoder sees the true audio of the decoded window
            y_full = G(z, x[..., a.enc_ctx - cd * HOP: a.enc_ctx + (S + cd) * HOP])
        else:
            y_full = G(z)
    y_hat = y_full[..., cd * HOP: (cd + S) * HOP]
    y = x[..., a.enc_ctx: a.enc_ctx + S * HOP]
    y_t = None
    if a.w_teacher_mel > 0 or a.w_teacher_wave > 0 or a.w_teacher_cplx > 0:
        if a.teacher_amp == "bf16" and amp_on(a):
            y_t = stock_decode_bf16(teacher, z.float())
        else:
            with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
                y_t = stock_decode(teacher, z.float()).float()
        y_t = y_t[..., cd * HOP: (cd + S) * HOP]
    return y_hat, y, y_t


SI_D_FAST, SI_D_SLOW = 0.99, 0.999  # decays of the training-distortion averages (~100 and ~1000 steps)
SI_LAM_RANGE, SI_DUAL_CLIP = (1e-4, 100.0), 0.01


def si_state_init(a, lam):
    return {"lam": float(lam), "d_last": float("nan"), "d_fast": None, "d_slow": None, "dstar": None}


@torch.no_grad()
def si_dual_update(a, st, l_mel, y_hat, y, on, mel):
    """--si-mode ceiling / constrained: track the training distortion D (the multi-scale mel loss, over the
    items that are on: off items are worse and their share varies per batch), and in constrained mode move
    lambda by dual ascent (GECO-style): log lambda rises while D is under target, falls while it is over."""
    if float(on.sum()) == 0:
        return
    d = float(l_mel) if float(on.min()) > 0 else float(mel(y_hat[on > 0].detach(), y[on > 0]))
    st["d_last"] = d
    st["d_fast"] = d if st["d_fast"] is None else SI_D_FAST * st["d_fast"] + (1 - SI_D_FAST) * d
    st["d_slow"] = d if st["d_slow"] is None else SI_D_SLOW * st["d_slow"] + (1 - SI_D_SLOW) * d
    if a.si_mode == "constrained" and st["dstar"]:
        target = st["dstar"] * (1 + a.si_eps)
        dl = max(-SI_DUAL_CLIP, min(SI_DUAL_CLIP, a.si_dual_lr * (target - st["d_fast"]) / target))
        st["lam"] = max(SI_LAM_RANGE[0], min(SI_LAM_RANGE[1], st["lam"] * math.exp(dl)))


def enc_mel(a, mel, y_hat, y, n_q):
    """--arch encoder: the multi-scale mel loss, band-limited (SBR-style) for items at <= --bwe-max-q codebooks:
    both signals low-passed at the item's cutoff, plus --bwe-hf-weight x the full-band term. -> (loss, the
    low-band term or None)."""
    cut = enc_mod.bwe_cutoffs(a.bwe_cutoff, a.bwe_max_q, n_q)
    lo = cut > 0
    if not bool(lo.any()):
        return mel(y_hat, y), None
    B, n_lo = y.shape[0], int(lo.sum())
    full = ~lo
    tot = (B - n_lo) / B * mel(y_hat[full], y[full]) if n_lo < B else 0.0
    lb = mel(enc_mod.lowpass(y_hat[lo], cut[lo]), enc_mod.lowpass(y[lo], cut[lo]))
    tot = tot + n_lo / B * lb
    if a.bwe_hf_weight > 0:
        tot = tot + n_lo / B * a.bwe_hf_weight * mel(y_hat[lo], y[lo])
    return tot, lb


def enc_train_step(a, step, teacher, G, D, mel, opt_g, opt_d, sch_g, sch_d, batch, dev):
    """--arch encoder. The encoder reads the whole crop (--enc-ctx context each side); the frames the decoder
    decodes (the segment plus --dec-ctx each side) are the ones quantised and, in phase A, matched."""
    x = batch["x"].to(dev, non_blocking=True)
    n_q = batch["n_q"].to(dev)
    S, ce, cd = a.seg_frames, a.enc_ctx // HOP, a.dec_ctx
    out = {}
    with autocast(a):
        z = G.enc(x)
    dmask = None
    if a.enc_mix_dac:  # patch ctrl3: some items get DAC's encoder and tiers from --n-q-dac (seeded by step)
        B = x.shape[0]
        gen = torch.Generator().manual_seed(a.seed * 1_000_003 + step)
        dmask = torch.zeros(B, dtype=torch.bool)
        dmask[torch.randperm(B, generator=gen)[:round(a.enc_mix_dac * B)]] = True
        nqd = torch.tensor(a.n_q_dac_list)[torch.randint(len(a.n_q_dac_list), (B,), generator=gen)]
        n_q = torch.where(dmask, nqd, n_q.cpu()).to(dev)
        dmask = dmask.to(dev)
        z = z.float().clone()
        with torch.no_grad():
            z[dmask] = DAC_ENC[0](x[dmask])
    zc = z[..., ce - cd: ce + S + cd]
    total = 0.0
    if a.enc_phase == "match" or a.enc_w_z > 0:
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            z_t = teacher.encoder(x.float())[..., ce - cd: ce + S + cd]
        lz, nmse, cos = enc_mod.z_match_loss(zc, z_t)
        total = total + a.enc_w_z * lz
        out["z_nmse"], out["z_cos"] = nmse.item(), cos.item()
    if a.enc_phase == "match":
        with torch.no_grad():
            _, codes_t, _ = G.rvq(z_t, N_CB, st=False)
        l_ce, agree = G.rvq.match_losses(zc, codes_t, a.enc_tau)
        total = total + a.enc_w_code * l_ce
        out["code_ce"], out["agree1"], out["agree9"] = l_ce.item(), agree[0].item(), agree.mean().item()
        sch_d.step()
    else:
        zq, _, commit = G.rvq(zc, n_q)
        with autocast(a):
            y_full = (G.pf(G.dec, zq, n_q) if getattr(G, "pf", None) is not None
                      else enc_mod.tier_cond_decode(G.dec, zq, n_q))
        y_hat = y_full[..., cd * HOP: (cd + S) * HOP]
        y = x[..., a.enc_ctx: a.enc_ctx + S * HOP]
        gan = step >= a.gan_start
        if gan:
            with autocast(a):
                loss_d = losses.d_loss(D(y_hat.detach()), D(y), a.gan)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            out["gn_d"] = float(torch.nn.utils.clip_grad_norm_(D.parameters(), a.clip_d))
            opt_d.step()
            out["d"] = loss_d.item()
        sch_d.step()
        l_mel, l_lb = enc_mel(a, mel, y_hat, y, n_q)
        total = total + a.w_mel * l_mel
        out["mel"] = l_mel.item()
        if l_lb is not None:
            out["mel_bwe"] = l_lb.item()
        if a.w_consist > 0:  # patch pf_lowtier: frame consistency of the corrected spectra, over the loss segment
            Sp, on_ = G.pf.last
            l_cons = pf_lowtier.consistency(Sp[on_], PF_STFT[0], G.dec.istft, slice(cd, cd + S)).mean()
            total = total + a.w_consist * l_cons
            out["consist"] = l_cons.item()
        if a.w_gain_tv > 0:  # patch pf_lowtier (band mode): temporal smoothness of the band log-gains
            l_tv = pf_lowtier.gain_tv(G.pf.last_g, slice(cd, cd + S))
            total = total + a.w_gain_tv * l_tv
            out["gain_tv"] = l_tv.item()
        if a.w_short > 0:  # patch pf_lowtier: short windows; the BWE items low-passed as in enc_mel
            cut = enc_mod.bwe_cutoffs(a.bwe_cutoff, a.bwe_max_q, n_q)
            lo = cut > 0
            yh_s, y_s = y_hat, y
            if bool(lo.any()):
                yh_s, y_s = y_hat.clone(), y.clone()
                yh_s[lo], y_s[lo] = enc_mod.lowpass(y_hat[lo], cut[lo]), enc_mod.lowpass(y[lo], cut[lo])
            l_short = pf_lowtier.short_loss(yh_s, y_s)
            total = total + a.w_short * l_short
            out["short"] = l_short.item()
        if a.enc_w_cplx > 0:
            full = enc_mod.bwe_cutoffs(a.bwe_cutoff, a.bwe_max_q, n_q) <= 0  # the full-band items
            if bool(full.any()):
                l_c = CPLX_LOSS[0](y_hat[full], y[full])
                total = total + a.enc_w_cplx * float(full.float().mean()) * l_c
                out["cplx"] = l_c.item()
        if a.w_lf > 0:
            l_lf = (torch.nn.functional.avg_pool1d(y_hat.float(), 1024, 256)
                    - torch.nn.functional.avg_pool1d(y.float(), 1024, 256)).abs().mean()
            total = total + a.w_lf * l_lf
            out["lf"] = l_lf.item()
        if (a.w_teacher_mel > 0 or a.w_teacher_cplx > 0) and (dmask is None or bool(dmask.any())):
            # patch ctrl2 (--enc-dac-enc): DAC's decoder on the same z_q; patch ctrl3 (--enc-mix-dac): on the DAC-code
            # items only, each term weighted by their share of the batch
            tw = 1.0 if dmask is None else float(dmask.float().mean())
            zt = (zq if dmask is None else zq[dmask]).detach().float()
            yh_t = y_hat if dmask is None else y_hat[dmask]
            if a.teacher_amp == "bf16" and amp_on(a):
                y_t = stock_decode_bf16(teacher, zt)
            else:
                with torch.no_grad(), torch.autocast(device_type=zt.device.type, enabled=False):
                    y_t = stock_decode(teacher, zt).float()
            y_t = y_t[..., cd * HOP: (cd + S) * HOP]
            if a.w_teacher_mel > 0:
                t_mel = mel(yh_t, y_t)
                total = total + tw * a.w_teacher_mel * t_mel
                out["t_mel"] = t_mel.item()
            if a.w_teacher_cplx > 0:
                t_c = CPLX_LOSS[0](yh_t, y_t)
                total = total + tw * a.w_teacher_cplx * t_c
                out["t_cplx"] = t_c.item()
        if gan and step >= (a.gan_g_start if a.gan_g_start >= 0 else a.gan_start):  # --gan-g-start (default: as before)
            with autocast(a):
                adv, fm = losses.g_losses(D(y_hat), D(y), a.gan)
            total = total + a.w_adv * adv + a.w_fm * fm
            out["adv"], out["fm"] = adv.item(), fm.item()
        total = total + a.enc_w_commit * commit.mean()
        out["commit"] = commit.mean().item()
        if a.enc_ext_n:  # patch q12_ext: the extra codebooks' DAC-style losses
            total = total + G.rvq.last_ext_loss.mean()
            out["vq_ext"] = G.rvq.last_ext_loss.mean().item()
    opt_g.zero_grad(set_to_none=True)
    total.backward()
    out["gn_g"] = float(torch.nn.utils.clip_grad_norm_([p for p in G.parameters() if p.requires_grad], a.clip_g))
    opt_g.step()
    sch_g.step()
    out["total"] = total.item()
    return out


def train_step(a, step, teacher, G, D, mel, opt_g, opt_d, sch_g, sch_d, batch, dev):
    if a.arch == "encoder":
        return enc_train_step(a, step, teacher, G, D, mel, opt_g, opt_d, sch_g, sch_d, batch, dev)
    if a.arch == "sideinfo":  # its sampling: a generator seeded from (seed, step), so resume is exact
        G.codec.gen = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + step)
    y_hat, y, y_t = forward_g(a, teacher, G, batch, dev)
    out = {}
    gan = step >= a.gan_start
    if gan:
        with autocast(a):
            loss_d = losses.d_loss(D(y_hat.detach()), D(y), a.gan)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        out["gn_d"] = float(torch.nn.utils.clip_grad_norm_(D.parameters(), a.clip_d))
        opt_d.step()
        out["d"] = loss_d.item()
    sch_d.step()
    l_mel = mel(y_hat, y)
    total = a.w_mel * l_mel
    out["mel"] = l_mel.item()
    if a.w_lf > 0:
        l_lf = (torch.nn.functional.avg_pool1d(y_hat.float(), 1024, 256)
                - torch.nn.functional.avg_pool1d(y.float(), 1024, 256)).abs().mean()
        total = total + a.w_lf * l_lf
        out["lf"] = l_lf.item()
    if gan:
        with autocast(a):
            adv, fm = losses.g_losses(D(y_hat), D(y), a.gan)
        total = total + a.w_adv * adv + a.w_fm * fm
        out["adv"], out["fm"] = adv.item(), fm.item()
    if y_t is not None:
        if a.w_teacher_mel > 0:
            t_mel = mel(y_hat, y_t)
            total = total + a.w_teacher_mel * t_mel
            out["t_mel"] = t_mel.item()
        if a.w_teacher_wave > 0:
            t_w = (y_hat - y_t).abs().mean()
            total = total + a.w_teacher_wave * t_w
            out["t_wave"] = t_w.item()
        if a.w_teacher_cplx > 0:
            t_c = CPLX_LOSS[0](y_hat, y_t)
            total = total + a.w_teacher_cplx * t_c
            out["t_cplx"] = t_c.item()
    if a.arch == "sideinfo":
        # lambda x side kbps per item (rate over the whole decoded window; off items have none).
        last = G.last
        kbps = last["kbps"]
        wu = a.si_rate_warmup
        warm = min(1.0, max(0.0, step / wu - 1.0)) if wu > 0 else 1.0
        on = 1 - last["off"]
        out["kbps"] = float((kbps * on).sum() / on.sum().clamp_min(1))
        if a.si_mode == "vr":
            total = total + warm * (G.lam_of(last["t"]) * kbps).mean()
        else:
            st = G.si_state
            total = total + warm * st["lam"] * kbps.mean()
            out["lam"] = st["lam"]
            if "kbps_em" in last:
                total = total + a.si_em_fit * last["kbps_em"].mean()
            si_dual_update(a, st, l_mel, y_hat, y, on, mel)
            out["d_on"] = st["d_last"]
    opt_g.zero_grad(set_to_none=True)
    total.backward()
    out["gn_g"] = float(torch.nn.utils.clip_grad_norm_(G.parameters(), a.clip_g))
    opt_g.step()
    sch_g.step()
    out["total"] = total.item()
    return out


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def decode_clip(a, teacher, G, codes, n_q, x=None):
    # Eval decodes in fp32, as deployment will and as picotone's scores were measured: under bf16, stock DAC
    # scores about 0.8 dB worse log-mel (same SI-SDR), which would skew every comparison.
    if getattr(G, "is_encoder", False):  # --arch encoder: its own codes from the clip x [C, T]; G.last_codes
        L = codes.shape[-1] * HOP
        xt = torch.from_numpy(np.ascontiguousarray(x[:, :L], dtype=np.float32))[:, None].to(codes.device)
        with torch.autocast(device_type=codes.device.type, enabled=False):
            y = G(torch.nn.functional.pad(xt, (0, L - xt.shape[-1])), n_q)
        return y.float()[:, 0].clamp(-1, 1).cpu().numpy()
    z = codes_to_z(teacher, codes, n_q).float()
    with torch.autocast(device_type=z.device.type, enabled=False):
        if getattr(G, "needs_target", False):  # --arch sideinfo: its encoder reads the clip, x [C, T]
            L = z.shape[-1] * HOP
            xt = torch.from_numpy(np.ascontiguousarray(x[:, :L], dtype=np.float32))[:, None].to(z.device)
            y = G(z, torch.nn.functional.pad(xt, (0, L - xt.shape[-1])))
        else:
            y = (enc_mod.tier_cond_decode(G, z, n_q) if getattr(G, "tier_emb", None) is not None else G(z)) \
                if G is not None else stock_decode(teacher, z)
    return y.float()[:, 0].clamp(-1, 1).cpu().numpy()


def prepare_eval(a, teacher, dev, log):
    clips = data.load_eval(a.eval_set, a.max_eval_clips, log, a.path_map)
    for e in clips:
        x = torch.from_numpy(e["x"])[:, None].to(dev)
        e["codes"] = encode_codes(teacher, x)
        if getattr(a, "si_codes_enc", ""):  # patch si_enc: the trained encoder's codes; stock DAC keeps DAC's
            e["codes_dac"] = e["codes"]
            L = e["codes"].shape[-1] * HOP
            xt = torch.nn.functional.pad(x[..., :L].float(), (0, L - min(L, x.shape[-1])))
            e["codes"] = get_codes(a, teacher, xt)
    return clips


def set_means(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}):
        out[s] = metrics.mean_scores([r["m"] for r in rows if r["set"] == s])
    out["all"] = metrics.mean_scores([r["m"] for r in rows])
    return out


def write_wav(path, x):
    sf.write(path, np.asarray(x).T, SR, subtype="PCM_16")


def listen_pick(a, i, e):
    """Eval clip i gets listening WAVs: the first --listen clips, or those named in --listen-names."""
    names = [v for v in getattr(a, "listen_names", "").split(",") if v]
    return e["name"] in names if names else i < a.listen


def run_eval(a, teacher, G, clips, n_q, tag, listen_dir):
    rows = []
    side = getattr(G, "needs_target", False)
    is_enc = getattr(G, "is_encoder", False)
    for i, e in enumerate(clips):
        if side:
            G.bits = 0.0
        codes = e["codes_dac"] if G is None and "codes_dac" in e else e["codes"]  # stock DAC: DAC's own codes
        y = decode_clip(a, teacher, G, codes, n_q, e["x"] if side or is_enc else None)
        m = metrics.score(e["x"], y)
        if side:  # side-information rate, kbps per channel
            m["kbps"] = G.bits / (y.shape[-1] / SR) / y.shape[0] / 1000
        if a.arch == "encoder":  # every line of an encoder run (stock, DAC encoder + decoder, ours) gets these
            m["logmel_lb"] = enc_mod.lowband_logmel(e["x"], y, a.bwe_eval_cutoff)
            m.update(enc_mod.agreement(G.last_codes, e["codes"], n_q) if is_enc
                     else {"agree1": 1.0, "agreeq": 1.0})
        rows.append({"name": e["name"], "set": e["set"], "m": m})
        if listen_dir and listen_pick(a, i, e):
            write_wav(os.path.join(listen_dir, f"{e['name']}__{tag}.wav"), y)
    means = set_means(rows)
    for k in [v for v in getattr(a, "eval_kinds", "").split(",") if v]:  # extra sets, not in the score
        rs = [r["m"] for r in rows if clip_gap.kind(r["name"]) == k]
        if rs:
            means[f"kind:{k}"] = metrics.mean_scores(rs)
    return means, rows


def stock_eval(a, teacher, clips, run, log):
    p = os.path.join(run, "stock_eval.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    todo = [nq for nq in a.eval_n_q if str(nq) not in res]
    if not todo:
        return res
    listen = os.path.join(run, "listen")
    os.makedirs(listen, exist_ok=True)
    if not res:
        for i, e in [(i, e) for i, e in enumerate(clips) if listen_pick(a, i, e)]:
            write_wav(os.path.join(listen, f"{e['name']}__0_original.wav"), e["x"])
    for nq in todo:
        t = time.perf_counter()
        means, rows = run_eval(a, teacher, None, clips, nq, f"0_stockDAC_q{nq}", listen)
        res[str(nq)] = {"means": means, "rows": rows}
        log(f"stock DAC at {nq} codebooks ({time.perf_counter() - t:.0f} s): " + fmt_means(means))
    json.dump(res, open(p, "w"), indent=1)
    return res


def student_alone_eval(a, teacher, student, clips, run, log):
    """--arch postfilter: the frozen student's own scores on the eval clips (once per student file), the line
    every post-filter eval is compared with. Listening WAVs are tagged 0_student_q<n>."""
    p = os.path.join(run, "student_eval.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    if res.get("student_sha") != a.pf_student_sha:
        res = {"student_sha": a.pf_student_sha}
    todo = [nq for nq in a.eval_n_q if str(nq) not in res]
    listen = os.path.join(run, "listen")
    os.makedirs(listen, exist_ok=True)
    for nq in todo:
        t = time.perf_counter()
        means, rows = run_eval(a, teacher, student, clips, nq, f"0_student_q{nq}", listen)
        res[str(nq)] = {"means": means, "rows": rows}
        log(f"student alone at {nq} codebooks ({time.perf_counter() - t:.0f} s): " + fmt_means(means))
    if todo:
        json.dump(res, open(p, "w"), indent=1)
    return res


def pf_vs_student_line(means, st_means, nq, what="post-filter vs student"):
    """One line: post-filter vs the student alone (or, --arch encoder, the trained encoder vs DAC's encoder into
    the same decoder), on the picotone sets (old + new) as the verdict scores them, plus each other eval set's
    log-mel."""
    lm, lm0 = verdict.picotone_logmel(means), verdict.picotone_logmel(st_means)
    si, si0 = verdict._mean(means, "sisdr"), verdict._mean(st_means, "sisdr")
    other = [f"{s} {st_means[s]['logmel']:.3f}->{means[s]['logmel']:.3f}" for s in sorted(means)
             if s not in verdict.SETS and s != "all" and s in st_means]
    return (f"  {what} q{nq}: log-mel {lm0:.3f} -> {lm:.3f} ({lm / lm0 - 1:+.2%}), "
            f"SI-SDR {si0:.2f} -> {si:.2f} ({si - si0:+.2f} dB)" + (f" | {', '.join(other)}" if other else ""))


def fmt_means(means):
    def dc(m):
        return f" dc {m['dc_mfs']:.2f}" if "dc_mfs" in m else ""

    def kb(m):
        return f" side {m['kbps']:.3f} kbps" if "kbps" in m else ""

    def en(m):
        return (f" lowband mel {m['logmel_lb']:.3f} agree c1 {m['agree1']:.3f} all {m['agreeq']:.3f}"
                if "logmel_lb" in m else "")

    return " | ".join(f"{s}: mel {m['logmel']:.3f} mrstft {m['mrstft']:.3f} snr {m['snr']:.2f} sisdr {m['sisdr']:.2f}"
                      + dc(m) + kb(m) + en(m) for s, m in means.items())


# ------------------------------------------------------------------ checkpoints

def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    # torch.load(map_location="cuda") moves these to the GPU too; the setters want CPU ByteTensors.
    torch.set_rng_state(s["torch"].cpu())
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


@torch.no_grad()
def update_g_ema(G_ema, G, decay):
    e, p = list(G_ema.parameters()), list(G.parameters())
    torch._foreach_lerp_(e, [q.detach() for q in p], 1.0 - decay)


def save_ckpt(path, step, G, D, opts, scheds, ema, a, G_ema=None):
    tmp = path + ".tmp"
    torch.save({"step": step, "G": G.state_dict(), "D": D.state_dict(),
                "G_ema": G_ema.state_dict() if G_ema is not None else None,
                "opt_g": opts[0].state_dict(), "opt_d": opts[1].state_dict(),
                "sch_g": scheds[0].state_dict(), "sch_d": scheds[1].state_dict(), "ema": ema,
                "rng": rng_state(), "args": vars(a), "width": G.width, "arch": arch_cfg(a),
                "si": copy.deepcopy(getattr(G, "si_state", None))}, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ bench

def bench(a, teacher, G, D, mel, opts, scheds, ds_factory, dev, log):
    """Per batch size: ms/step (after warm-up) and peak memory (CUDA) or saved-activation bytes (CPU)."""
    for B in [int(v) for v in a.bench.split(",")]:
        ds = ds_factory(B)
        it = iter(data.loader(ds, a.workers))
        batches = [next(it) for _ in range(3)]
        saved = [0]

        def pack(t):
            saved[0] += t.numel() * t.element_size()
            return t

        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                train_step(a, a.gan_start, teacher, G, D, mel, *opts, *scheds, batches[0], dev)
            n = 3 if dev == "cpu" else 10
            t = time.perf_counter()
            for i in range(n):
                train_step(a, a.gan_start, teacher, G, D, mel, *opts, *scheds, batches[i % 3], dev)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) / n * 1000
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else float("nan")
            log(f"bench batch {B}: {ms:.0f} ms/step, peak CUDA {peak:.2f} GiB, saved activations "
                f"{saved[0] / 2 ** 30:.2f} GiB ({saved[0] / B / 2 ** 20:.1f} MiB/item)")
        except torch.OutOfMemoryError:
            log(f"bench batch {B}: out of memory")
            break


# ------------------------------------------------------------------ main

def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.run, exist_ok=True)
    log = Log(a.run)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    dev = a.device
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    log(f"== {' '.join(sys.argv)}")
    teacher = load_teacher(dev)
    if a.arch == "encoder":
        dec, a.enc_dec_arch, a.enc_dec_width = enc_mod.load_decoder(a.enc_dec, a.enc_dec_ema)
        if a.enc_ext_cb:  # patch q12_ext: DAC's 9 frozen codebooks + the extra ones, initialised from the file
            rvq = enc_mod.ExtRVQ(teacher.quantizer, a.enc_ext_n, a.enc_soft_tau, a.enc_ext_w_commit,
                                 a.enc_ext_w_codebook, a.enc_ext_grad)
            rvq.ext.load_state_dict(torch.load(a.enc_ext_cb, map_location="cpu", weights_only=False)["ext"])
        else:
            rvq = enc_mod.FrozenRVQ(teacher.quantizer, a.enc_soft_tau)
        enc = enc_mod.DACEncoder(teacher.encoder) if a.enc_dac_enc else enc_mod.build_encoder(arch_cfg(a))
        if a.dec_tier_cond:  # patch tiercond: before EncoderCodec, so the embedding trains with the decoder
            enc_mod.add_tier_cond(dec)
        G = enc_mod.EncoderCodec(enc, rvq, dec, a.enc_train_dec).to(dev)
        if a.enc_mix_dac:
            DAC_ENC[0] = enc_mod.DACEncoder(teacher.encoder).to(dev).eval()
            log(f"--enc-mix-dac {a.enc_mix_dac:g}: DAC-encoder items with tiers {a.n_q_dac}; DAC encoder copy hash "
                f"{state_hash(DAC_ENC[0])}; teacher terms on those items only")
        if a.enc_freeze_enc:
            for p in G.enc.parameters():
                p.requires_grad_(False)
        if a.enc_freeze_dec:  # patch q12c
            for p in G.dec.parameters():
                p.requires_grad_(False)
        if a.pf_lowtier:  # patch pf_lowtier: decoder and every codebook frozen; only G.pf trains
            for p in list(G.dec.parameters()) + list(G.rvq.parameters()):
                p.requires_grad_(False)
            G.pf = pf_lowtier.LowTierPF(G.dec.width, a.pflt_dim, a.pflt_blocks, None, G.dec.n_fft, a.pflt_max_q,
                                        a.pflt_mode, a.pflt_mag_bound, a.pflt_bands).to(dev)
            PF_STFT[0] = postfilter.STFT(G.dec.n_fft, G.dec.n_fft // 4).to(dev)
            log(f"--pf-lowtier: post-filter {a.pflt_dim} x {a.pflt_blocks}, q1-q{a.pflt_max_q}, mode {a.pflt_mode}"
                f"{f' (bound {a.pflt_mag_bound:g})' if a.pflt_mode != 'complex' else ''}"
                f"{f', {a.pflt_bands} bands, w_gain_tv {a.w_gain_tv:g}' if a.pflt_mode == 'band' else ''}, "
                f"{sum(p.numel() for p in G.pf.parameters()):,} params; receptive field +-"
                f"{G.pf.receptive_field(G.dec)} latent frames (decoder alone +-{G.dec.receptive_field()}); "
                f"w_consist {a.w_consist:g}, w_short {a.w_short:g}")
    else:
        G = make_decoder(arch_cfg(a)).to(dev)
    g_name = f"model {arch_cfg(a)}"
    D = losses.Discriminator().to(dev)
    if a.init_d and not a.resume:
        D.load_state_dict(torch.load(a.init_d, map_location=dev, weights_only=False)["D"])
        log(f"discriminator initialised from {a.init_d}")
    mel = losses.MultiScaleMel().to(dev)
    if a.w_teacher_cplx > 0 or a.enc_w_cplx > 0:
        CPLX_LOSS[0] = TeacherComplexSTFT([int(v) for v in a.cplx_windows.split(",")], a.cplx_power,
                                          a.cplx_alpha).to(dev)
    if a.w_teacher_cplx > 0:
        log(f"teacher complex STFT loss x {a.w_teacher_cplx:g}: windows {a.cplx_windows}, |S|^{a.cplx_power:g}, "
            f"complex share {a.cplx_alpha:g}")
    if a.enc_w_cplx > 0:
        log(f"encoder complex STFT loss x {a.enc_w_cplx:g} vs the true audio (items above q{a.bwe_max_q} only): "
            f"windows {a.cplx_windows}, |S|^{a.cplx_power:g}, complex share {a.cplx_alpha:g}")
    # Cost report (MACs per second of audio, one channel), measured.
    fz = torch.zeros(1, 1024, 87, device=dev)
    sc, tc = count_cost(G.dec if a.arch == "encoder" else G, fz), count_cost(teacher.decoder, fz)
    sec = 87 * HOP / SR
    if a.arch == "encoder":
        g_name = f"decoder {a.enc_dec} ({a.enc_dec_arch['arch']}, {'trains' if a.enc_train_dec else 'frozen'})"
        ep, eg, es = (n_params(G.enc), enc_mod.DAC_ENC_GMAC, 0.0) if a.enc_dac_enc else enc_mod.cost(G.enc.cpu())
        G.enc.to(dev)
        log(f"encoder {arch_cfg(a)['enc']}: {ep:,} params, {eg:.2f} GMAC/s, {es:.1f} M Snake/s (stock DAC encoder "
            f"{enc_mod.DAC_ENC_GMAC} GMAC/s -> {enc_mod.DAC_ENC_GMAC / eg:.1f}x fewer MACs; Rust estimate "
            f"{enc_mod.rust_estimate(eg):.1f} s per 90 s of stereo vs {enc_mod.DAC_ENC_RUST_S:.0f} s); phase "
            f"{a.enc_phase}; quantiser frozen; BWE cutoff {a.bwe_cutoff} Hz at q <= {a.bwe_max_q}")
    log(f"{g_name}: {n_params(G):,} params, {sc['mac'] / sec / 1e9:.2f} GMAC/s, "
        f"{sc['snake'] / sec / 1e6:.1f} M Snake/s | stock DAC decoder: "
        f"{n_params(teacher.decoder):,} params, {tc['mac'] / sec / 1e9:.2f} GMAC/s -> "
        f"{tc['mac'] / sc['mac']:.2f}x fewer MACs | discriminator {n_params(D):,} params")
    if a.arch in ("postfilter", "sideinfo"):
        pc = sc["mac"] - count_cost(G.student, fz)["mac"]
        what = "post-filter" if a.arch == "postfilter" else "decode side (post-filter + entropy model)"
        trained = n_params(G.pf) if a.arch == "postfilter" else n_params(G.codec) - n_params(G.codec.enc)
        log(f"{what} alone: {trained:,} params, {pc / sec / 1e9:.2f} GMAC/s "
            f"({pc / (sc['mac'] - pc):.1%} on top of the student {a.pf_student}, sha256 {a.pf_student_sha}...)")
        if a.si_codes_enc:
            log(f"codes from the trained encoder {a.si_codes_enc} (sha256 {a.si_codes_enc_sha}..., G_ema enc.*, fp32) "
                f"+ DAC's quantiser, not DAC's encoder (patch si_enc); stock DAC lines keep DAC's codes; decoder "
                f"{a.pf_student_arch['arch']}")
        if a.arch == "sideinfo":
            G.eval_t = sideinfo.parse_t(a.si_eval_t)
            G.p_off = a.si_p_off
            if a.si_mode == "vr":
                log(f"side encoder (offline): {n_params(G.codec.enc):,} params; lambda {a.si_lam[0]:g}.."
                    f"{a.si_lam[1]:g} per side kbps, off share {a.si_p_off}; training evals at t = {a.si_eval_t}")
            else:
                G.em_fit = a.si_em_fit > 0
                G.si_state = si_state_init(a, a.si_lam1)
                log(f"side encoder (offline): {n_params(G.codec.enc):,} params; mode {a.si_mode} at q{a.si_nq}, "
                    f"{a.si_ch} side channels at {86.13 / a.si_down:.0f} Hz, off share {a.si_p_off}, entropy-model "
                    f"fit {a.si_em_fit:g}, rate warm-up {a.si_rate_warmup}; " + (
                        f"lambda fixed at {a.si_lam1:g} per side kbps" if a.si_mode == "ceiling" else
                        f"lambda by dual ascent to D <= D* x {1 + a.si_eps:g} (rate {a.si_dual_lr:g})")
                    + (f"; training evals {'off' if G.eval_t is None else 'on'}"))
    dcost = count_cost(D, torch.zeros(1, 1, SR, device=dev))["mac"]
    ecost = count_cost(teacher.encoder, torch.zeros(1, 1, 86 * HOP, device=dev))["mac"] / (86 * HOP / SR)
    seg_s, dec_s = a.seg_frames * HOP / SR, (a.seg_frames + 2 * a.dec_ctx) * HOP / SR
    enc_s = seg_s + 2 * a.enc_ctx / SR
    # Per item and step: encoder forward, student forward+backward (~3x), discriminator ~10 forward-equivalents
    # (D step: fake+real forward and backward = 6; G step: fake forward+backward = 3, real forward = 1).
    per_item = ecost * enc_s + 3 * sc["mac"] / sec * dec_s + 10 * dcost * seg_s
    log(f"discriminator {dcost / 1e9:.2f} GMAC/s, encoder {ecost / 1e9:.2f} GMAC/s -> about {per_item / 1e9:.0f} "
        f"GMAC per batch item per step (encoder {ecost * enc_s / per_item:.0%}, student "
        f"{3 * sc['mac'] / sec * dec_s / per_item:.0%}, discriminator {10 * dcost * seg_s / per_item:.0%})")

    # Trainable parameters only (all of them except a post-filter's frozen student; the same list for dac/vocos).
    if a.arch == "encoder" and a.pf_lowtier:  # patch pf_lowtier: only the post-filter trains
        opt_g = torch.optim.AdamW(list(G.pf.parameters()), lr=a.lr, betas=tuple(a.betas))
    elif a.arch == "encoder" and a.enc_freeze_dec:  # patch q12c: only the extra codebooks train
        opt_g = torch.optim.AdamW(list(G.rvq.ext.parameters()), lr=a.enc_ext_lr or a.lr, betas=tuple(a.betas))
    elif a.arch == "encoder" and a.enc_freeze_enc:  # decoder-only fine-tune: the encoder is not in the optimiser
        if a.enc_ext_lr:  # patch q12c: the extra codebooks in their own group at --enc-ext-lr
            opt_g = torch.optim.AdamW([{"params": list(G.dec.parameters())},
                                       {"params": list(G.rvq.ext.parameters()), "lr": a.enc_ext_lr}],
                                      lr=a.lr, betas=tuple(a.betas))
        else:
            opt_g = torch.optim.AdamW(list(G.dec.parameters()) + (list(G.rvq.ext.parameters()) if a.enc_ext_n else []),
                                      lr=a.lr, betas=tuple(a.betas))
    elif a.arch == "encoder" and a.enc_train_dec:  # the joint fine-tune: the decoder at --enc-dec-lr-mult x lr
        opt_g = torch.optim.AdamW([{"params": list(G.enc.parameters())},
                                   {"params": list(G.dec.parameters()), "lr": a.lr * a.enc_dec_lr_mult}],
                                  lr=a.lr, betas=tuple(a.betas))
    else:
        opt_g = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=a.lr, betas=tuple(a.betas))
    opt_d = torch.optim.AdamW(D.parameters(), lr=a.lr_d or a.lr, betas=tuple(a.betas))
    if a.lr_schedule_d or a.lr_warmup_d >= 0 or a.lr_d:  # patch q16: G and D on their own schedules
        def lam(kind, wu_):
            wu_ = max(1, wu_)
            if kind == "warmconst":
                return lambda s: min(1.0, s / wu_)
            if kind == "warmlin":
                return lambda s: s / wu_ if s < wu_ else max(0.0, (a.steps - s) / max(1, a.steps - wu_))
            if kind == "linear":
                return lambda s: max(0.0, 1.0 - s / a.steps)
            if kind == "warmcos":  # patch pf_lowtier: linear warm-up, then cosine to 0 at --steps
                return lambda s: s / wu_ if s < wu_ else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - wu_) /
                                                                                         max(1, a.steps - wu_))))
            raise SystemExit(f"--lr-schedule(-d) {kind}: not supported with --lr-d/--lr-schedule-d/--lr-warmup-d")
        wd = a.lr_warmup if a.lr_warmup_d < 0 else a.lr_warmup_d
        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lam(a.lr_schedule, a.lr_warmup))
        sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lam(a.lr_schedule_d or a.lr_schedule, wd))
    elif a.lr_schedule == "warmconst":  # patch ctrl2: 0 -> --lr over --lr-warmup steps, then constant
        wu = max(1, a.lr_warmup)
        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: min(1.0, s / wu))
        sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: min(1.0, s / wu))
    elif a.lr_schedule == "warmlin":  # patch q16: 0 -> --lr over --lr-warmup, then linear to 0 at --steps
        wu = max(1, a.lr_warmup)
        f_ = lambda s: s / wu if s < wu else max(0.0, (a.steps - s) / max(1, a.steps - wu))
        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, f_)
        sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, f_)
    elif a.lr_schedule == "warmcos":  # patch pf_lowtier (only with --pf-lowtier)
        wu = max(1, a.lr_warmup)
        f_ = lambda s: s / wu if s < wu else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - wu) / max(1, a.steps - wu))))
        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, f_)
        sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, f_)
    elif a.lr_schedule == "linear":  # from --lr at step 0 to 0 at --steps
        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: max(0.0, 1.0 - s / a.steps))
        sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: max(0.0, 1.0 - s / a.steps))
    else:
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, a.gamma)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, a.gamma)
    step, ema = 0, {}
    ck = os.path.join(a.run, "ckpt_last.pt")
    if a.resume:
        s = torch.load(ck, map_location=dev, weights_only=False)
        check_arch(a, s, ck)
        G.load_state_dict(s["G"])
        D.load_state_dict(s["D"])
        opt_g.load_state_dict(s["opt_g"])
        opt_d.load_state_dict(s["opt_d"])
        sch_g.load_state_dict(s["sch_g"])
        sch_d.load_state_dict(s["sch_d"])
        set_rng_state(s["rng"])
        step, ema = s["step"], s["ema"]
        if s.get("si") is not None:
            G.si_state = s["si"]
        log(f"resumed from {ck} at step {step} (G {state_hash(G)})"
            + (f", side-info state {G.si_state}" if s.get("si") is not None else ""))
    elif a.arch == "sideinfo" and a.si_init and not (a.eval_only or a.bench):
        s0 = torch.load(a.si_init, map_location=dev, weights_only=False)
        check_arch(a, s0, a.si_init)
        G.load_state_dict(s0["G"])
        si0 = s0.get("si") or {}
        st = G.si_state
        if a.si_mode == "constrained":
            st["lam"] = a.si_lam_start or si0.get("lam", st["lam"])
        st["d_fast"], st["d_slow"] = si0.get("d_fast"), si0.get("d_slow")
        st["dstar"] = a.si_dstar or si0.get("d_slow")
        if a.si_mode == "constrained" and not st["dstar"]:
            raise SystemExit(f"{a.si_init} has no training-distortion average: pass --si-dstar")
        log(f"G initialised from {a.si_init} (step {s0['step']}, raw weights, G {state_hash(G)}); D* "
            f"{st['dstar']:.5f} -> target {st['dstar'] * (1 + a.si_eps):.5f} (eps {a.si_eps:g}), lambda starts at "
            f"{st['lam']:g}")
        del s0
    init_ema = None
    if a.init_from and not (a.resume or a.eval_only or a.bench):
        src_sha = postfilter.file_sha16(a.init_from)
        s0 = torch.load(a.init_from, map_location=dev, weights_only=False)
        check_arch(a, s0, a.init_from)
        if a.init_from_ema and s0.get("G_ema") is None:
            raise SystemExit(f"--init-from-ema: {a.init_from} has no G_ema")
        load_g(a, G, s0["G_ema"] if a.init_from_ema else s0["G"])
        if not a.init_d and not a.fresh_d:  # --init-d (loaded above) wins; --fresh-d: D stays fresh
            D.load_state_dict(s0["D"])
        elif a.fresh_d:
            log(f"--fresh-d: D freshly initialised (not loaded from {a.init_from}), D {state_hash(D)}")
        init_ema = s0.get("G_ema")
        log(f"fine-tune: G, {'' if a.init_d or a.fresh_d else 'D, '}{'G_ema ' if init_ema is not None else ''}from {a.init_from} "
            f"(step {s0['step']}, "
            f"sha256 {src_sha}..., G {state_hash(G)}); fresh optimiser and {a.lr_schedule} schedule from lr "
            f"{a.lr:g}, step 0" + (" (G loaded from its G_ema)" if a.init_from_ema else ""))
        del s0
    G_ema = None
    if a.g_ema > 0:
        G_ema = copy.deepcopy(G).eval()
        for p in G_ema.parameters():
            p.requires_grad_(False)
        if a.resume and s.get("G_ema") is not None:
            G_ema.load_state_dict(s["G_ema"])
        elif a.resume:
            log("G_ema: none in the checkpoint, so it starts as a copy of G")
        elif init_ema is not None:
            load_g(a, G_ema, init_ema)
    if a.enc_freeze_dec:
        log(f"decoder frozen (--enc-freeze-dec): step {step} decoder hash G {state_hash(G.dec)}"
            + (f", G_ema {state_hash(G_ema.dec)}" if G_ema is not None else "")
            + f"; extra codebooks lr {a.enc_ext_lr or a.lr:g}, grad {a.enc_ext_grad}; D lr {a.lr:g}")
    if a.enc_freeze_enc:
        log(f"encoder frozen (--enc-freeze-enc): step {step} encoder state_dict hash G {state_hash(G.enc)}"
            + (f", G_ema {state_hash(G_ema.enc)}" if G_ema is not None else "")
            + f"; decoder G {state_hash(G.dec)}; trainable {sum(p.numel() for p in G.parameters() if p.requires_grad):,}"
            + (f"; codebooks 1-9 hash G {state_hash(G.rvq.q)}" + (f", G_ema {state_hash(G_ema.rvq.q)}" if G_ema
               is not None else "") + f"; extra codebooks {a.enc_ext_n} from {a.enc_ext_cb} (sha256 {a.enc_ext_sha}"
               f"...), G {state_hash(G.rvq.ext)}" if a.enc_ext_n else ""))

    if a.init_opt_from and not a.resume:  # patch ctrl2
        so = torch.load(a.init_opt_from, map_location="cpu", weights_only=False)
        gp = ([p_ for n_, p_ in G.dec.named_parameters() if not n_.startswith("tier_emb.")] if a.arch == "encoder"
              else [p for p in G.parameters() if p.requires_grad])
        msg = []
        for name, opt, params in ((() if a.enc_freeze_dec else (("opt_g", opt_g, gp),))
                                  + (("opt_d", opt_d, list(D.parameters())),)):
            sd = so[name]
            ids = [i for g_ in sd["param_groups"] for i in g_["params"]]
            if len(ids) != len(params):
                raise SystemExit(f"--init-opt-from: {name} has {len(ids)} parameters, this run {len(params)}")
            steps = set()
            for p_, i in zip(params, ids):
                st = sd["state"][i]
                if st["exp_avg"].shape != p_.shape:
                    raise SystemExit(f"--init-opt-from: {name} shape {tuple(st['exp_avg'].shape)} vs {tuple(p_.shape)}")
                opt.state[p_] = {k: (v.to(p_.device) if torch.is_tensor(v) and k != "step" else v)
                                 for k, v in st.items()}
                steps.add(float(st["step"]))
            msg.append(f"{name} {len(params)} tensors (Adam step {sorted(steps)})")
        log(f"optimiser state from {a.init_opt_from}: " + ", ".join(msg) + f"; lr/betas from this run (lr {a.lr:g}, "
            f"{a.lr_schedule}" + (f", warm-up {a.lr_warmup}" if a.lr_schedule == "warmconst" else "") + ")")
        del so
    lic = corpus.parse_licences(a.licences)
    log(f"licences: {sorted(lic)}")
    sources = None if a.eval_only else data.load_sources(a.manifest, lic, a.mix, a.mix_floor, log, a.path_map,
                                                          data.eval_artists(a.eval_set))
    crop = a.seg_frames * HOP + 2 * a.enc_ctx

    def ds_factory(B, start=0):
        return data.BatchStream(sources, B, crop, a.n_q_list, a.seed, start, a.tracks_per_batch, a.chunk_seconds)

    if a.bench:
        bench(a, teacher, G, D, mel, (opt_g, opt_d), (sch_g, sch_d), ds_factory, dev, log)
        return

    clips = prepare_eval(a, teacher, dev, log) if a.eval_set else []
    stock = stock_eval(a, teacher, clips, a.run, log) if clips else {}
    stud = (student_alone_eval(a, teacher, G.dec if a.arch == "encoder" else G.student, clips, a.run, log)
            if clips and a.arch in ("postfilter", "sideinfo", "encoder") else {})
    vs = ("encoder vs DAC encoder, same decoder", "DAC enc + decoder  ") if a.arch == "encoder" else (
        "post-filter vs student", "student alone      ")
    pico = json.load(open(a.picotone_json)) if a.picotone_json and os.path.exists(a.picotone_json) else None
    past = (corpus.read_jsonl(os.path.join(a.run, "evals.jsonl"))
            if os.path.exists(os.path.join(a.run, "evals.jsonl")) else [])
    # Pilot-verdict trend, per weights variant ("raw" = G, "ema" = G_ema; older rows have no "w" and are raw).
    history = {"raw": [], "ema": []}
    by_step = {}
    for r in past:
        w = r.get("w", "raw")
        if r["n_q"] == a.n_q_list[0]:
            history[w].append((r["step"], r["means"]))
        by_step.setdefault((r["step"], w), {})[r["n_q"]] = r["means"]
    # Best score per eval (best variant), for the plateau rule: only evals that covered every --eval-n-q.
    scores = []
    for st in sorted({k[0] for k in by_step}):
        sc = [sum(verdict.picotone_logmel(m[nq]) for nq in a.eval_n_q) / len(a.eval_n_q)
              for (s2, w), m in by_step.items() if s2 == st and all(nq in m for nq in a.eval_n_q)]
        if sc:
            scores.append((st, min(sc)))
    best_path = os.path.join(a.run, "ckpt_best.pt")
    best = {"score": float("inf")}
    if os.path.exists(best_path):
        b = torch.load(best_path, map_location="cpu", weights_only=False)
        if b.get("eval_n_q") == a.eval_n_q:
            best = {k: b[k] for k in ("score", "step", "w")}
            log(f"ckpt_best.pt: {best['w']} weights at step {best['step']}, score {best['score']:.4f}")
        else:
            log(f"ckpt_best.pt was scored on tiers {b.get('eval_n_q')}, not {a.eval_n_q}: it will be replaced")

    def evaluate(step):
        """-> True when a stop rule fires."""
        listen = os.path.join(a.run, "listen") if not a.listen_every or step % a.listen_every == 0 else None
        variants = [("raw", G)] + ([("ema", G_ema)] if G_ema is not None else [])
        G.eval()
        results = {}
        for w, model in variants:
            per = {}
            for nq in a.eval_n_q:
                t = time.perf_counter()
                means, rows = run_eval(a, teacher, model, clips, nq, f"s{step:07d}_{w}_q{nq}", listen)
                per[nq] = means
                jsonl(a.run, "evals.jsonl", {"step": step, "w": w, "n_q": nq, "means": means, "rows": rows})
                kb = f" [{2 * nq * enc_mod.KBPS_PER_CB:.2f} kbps L/R]" if a.arch == "encoder" else ""
                log(f"eval step {step} {w} q{nq}{kb} ({time.perf_counter() - t:.0f} s): " + fmt_means(means))
                sm = stock[str(nq)]["means"]
                log("   stock DAC          : " + fmt_means({k: v for k, v in sm.items() if k in means}))
                if stud:
                    log(f"   {vs[1]}: " + fmt_means(stud[str(nq)]["means"]))
                if nq == a.n_q_list[0] and any(s in means for s in verdict.SETS):
                    history[w].append((step, means))
                    rule = "pilot" if step <= a.pilot_steps else "full"
                    status, text = verdict.verdict(means, sm, history[w], pico, rule, nq)
                    log(f"[{w}] " + text)
                    jsonl(a.run, "verdicts.jsonl", {"step": step, "w": w, "rule": rule, "status": status})
            results[w] = per
        G.train()
        if not any(s in results["raw"][a.eval_n_q[0]] for s in verdict.SETS):
            return False
        # Every tier against the full-run rule, then the best-checkpoint and stop rules.
        summary = {}
        for w, per in results.items():
            oks = []
            for nq in a.eval_n_q:
                ok, line = verdict.tier_line(per[nq], stock[str(nq)]["means"], nq, "full")
                oks.append(ok)
                log(f"  [{w}] full-run rule {line}")
                if stud:
                    log(f"  [{w}]" + pf_vs_student_line(per[nq], stud[str(nq)]["means"], nq, vs[0]))
            summary[w] = (sum(verdict.picotone_logmel(per[nq]) for nq in a.eval_n_q) / len(a.eval_n_q), all(oks))
            log(f"  [{w}] score {summary[w][0]:.4f} (mean picotone log-mel over q{a.eval_n_q}), "
                f"{'ALL TIERS PASS' if summary[w][1] else 'not every tier passes yet'}")
        w_best = min(summary, key=lambda k: summary[k][0])
        score = summary[w_best][0]
        if score < best["score"] and not a.eval_only:
            best.update(score=score, step=step, w=w_best)
            model = G if w_best == "raw" else G_ema
            tmp = best_path + ".tmp"
            torch.save({"step": step, "w": w_best, "score": score, "eval_n_q": a.eval_n_q, "width": model.width,
                        "arch": arch_cfg(a), "args": vars(a),
                        "G": model.state_dict(), "tiers": {str(nq): results[w_best][nq] for nq in a.eval_n_q}}, tmp)
            os.replace(tmp, best_path)
            log(f"  new best: {w_best} weights, score {score:.4f} -> ckpt_best.pt")
        scores.append((step, score))
        if a.stop_when_pass and summary[w_best][1]:
            log(f"stop: every tier passes the full-run rule ({w_best} weights)")
            return True
        if a.stop_plateau > 0:
            def avg_upto(s_end):
                v = [sc for st, sc in scores if st <= s_end][-4:]
                return sum(v) / len(v) if len(v) == 4 else None
            now, then = avg_upto(step), avg_upto(step - a.plateau_steps)
            if now is not None and then is not None:
                log(f"  plateau check: 4-eval average {then:.4f} at {step - a.plateau_steps} -> {now:.4f} "
                    f"(stop if better by < {a.stop_plateau})")
                if then - now < a.stop_plateau:
                    log("stop: the score has plateaued")
                    return True
        return False

    if a.eval_only:
        if not clips:
            raise SystemExit("--eval-only needs --eval-set")
        s = torch.load(a.eval_only, map_location="cpu", weights_only=False)
        check_arch(a, s, a.eval_only)
        G.load_state_dict(s["G"])
        if G_ema is not None:
            if s.get("G_ema") is not None:
                G_ema.load_state_dict(s["G_ema"])
            else:
                G_ema = None  # an older checkpoint: raw weights only
        log(f"eval-only: {a.eval_only} at step {s['step']} (G {state_hash(G)})")
        evaluate(s["step"])
        return

    it = iter(data.loader(ds_factory(a.batch, step), a.workers))
    G.train()
    D.train()
    t_last, t_data = time.perf_counter(), 0.0
    if step == 0 and clips:
        evaluate(0)
    while step < a.steps:
        t0 = time.perf_counter()
        batch = next(it)
        assert batch["step"] == step, (batch["step"], step)
        t_data += time.perf_counter() - t0
        out = train_step(a, step, teacher, G, D, mel, opt_g, opt_d, sch_g, sch_d, batch, dev)
        if G_ema is not None:
            update_g_ema(G_ema, G, a.g_ema)
        step += 1
        for k, v in out.items():
            if math.isfinite(v):
                ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(out["total"]):
            log(f"non-finite loss at step {step}: {out}")
        if step % a.log_every == 0 or step == a.steps:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t_last) / a.log_every * 1000
            row = {"step": step, "ms": dt, "data_ms": t_data / a.log_every * 1000, "lr": sch_g.get_last_lr()[0],
                   **{k: round(v, 5) for k, v in ema.items()}}
            jsonl(a.run, "train.jsonl", row)
            log(f"step {step}: " + " ".join(f"{k} {v:.4g}" for k, v in ema.items())
                + f" | {dt:.0f} ms/step (data wait {row['data_ms']:.0f}) lr {row['lr']:.3g}")
            t_last, t_data = time.perf_counter(), 0.0
        if step % a.ckpt_every == 0 or step == a.steps:
            save_ckpt(ck, step, G, D, (opt_g, opt_d), (sch_g, sch_d), ema, a, G_ema)
            if a.enc_freeze_enc:
                log(f"encoder frozen: step {step} encoder hash G {state_hash(G.enc)}"
                    + (f", G_ema {state_hash(G_ema.enc)}" if G_ema is not None else "")
                    + (f"; codebooks 1-9 hash G {state_hash(G.rvq.q)}" + (
                        f", G_ema {state_hash(G_ema.rvq.q)}" if G_ema is not None else "") if a.enc_ext_n else "")
                    + (f"; DAC encoder copy {state_hash(DAC_ENC[0])}" if a.enc_mix_dac else "")
                    + (f"; decoder G {state_hash(G.dec)}" + (f", G_ema {state_hash(G_ema.dec)}" if G_ema is not None
                                                             else "") if a.enc_freeze_dec or a.pf_lowtier else "")
                    + (f"; extra codebooks G {state_hash(G.rvq.ext)}" + (f", G_ema {state_hash(G_ema.rvq.ext)}"
                       if G_ema is not None else "") + f"; post-filter G {state_hash(G.pf)}" if a.pf_lowtier else ""))
            if step % a.keep_every == 0:
                save_ckpt(os.path.join(a.run, f"ckpt_{step:07d}.pt"), step, G, D, (opt_g, opt_d), (sch_g, sch_d),
                          ema, a, G_ema)
        if clips and (step % a.eval_every == 0 or step == a.steps):
            if evaluate(step):
                if step % a.ckpt_every != 0:
                    save_ckpt(ck, step, G, D, (opt_g, opt_d), (sch_g, sch_d), ema, a, G_ema)
                break
        if a.stop_at and step >= a.stop_at:  # patch q16
            if step % a.ckpt_every != 0:
                save_ckpt(ck, step, G, D, (opt_g, opt_d), (sch_g, sch_d), ema, a, G_ema)
            log(f"--stop-at {a.stop_at}: stopping at step {step} (resume with --resume)")
            break
    si = getattr(G, "si_state", None)
    log(f"done at step {step}; G {state_hash(G)}" + (
        f"; side-info lambda {si['lam']!r} D fast {si['d_fast']!r} slow {si['d_slow']!r}" if si else ""))


if __name__ == "__main__":
    main()
