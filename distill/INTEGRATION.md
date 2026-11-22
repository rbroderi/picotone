# HQ mode: integrating the DAC student into picotone

Status: plan, nothing built (2026-11-21). The full run (`runs/w512_tiers`, `--n-q 4,5,6,7,9`) is still
training; this plan starts from its `ckpt_best.pt`.

## Why

At 75k steps the student, with smoothed weights, scores these on the 53 picotone clips (log-mel, lower is
better; DAC kbps are raw stereo, before any entropy coding):

| | kbps | log-mel |
|---|---|---|
| EnCodec music6 | 4.8 | 3.46 |
| Student q4 | 6.9 | 3.11 |
| EnCodec music12 | 10.0 | 3.05 |
| Student q5 | 8.6 | 2.90 |
| Student q6 | 10.3 | 2.74 |
| Student q9 | 15.5 | 2.38 |
| EnCodec music24 | 19.9 | 2.80 |

By ear (the user, 75k): student and stock DAC are almost impossible to tell apart, and q4 against q9 is very
subtle. The cost is decode speed: the student is projected at about 1.9× real time in single-thread wasm
(stereo), against about 5.5× for EnCodec 48 kHz. So picotone keeps two modes:

- **fast**: today's EnCodec presets (music6/12/24, mono);
- **HQ**: the DAC student at tiers q4–q9.

## What exists, and what doesn't

From a survey of the repo (file references are as of commit f027e32):

- **Nothing outside `distill/` knows DAC or Snake.** Everything assumes EnCodec: `HOP = 320`
  (`src/pico.rs:51`), 128-dim centroids in cx2 (`src/cx2/features.rs:23-27`), and a model byte that implies
  sample rate, channels and segmenting (`src/pico.rs:111-149`).
- **Reusable as is:**
  - the hand-written kernels in `src/kernels.rs` (packed-panel GEMM over `pulp` SIMD: x86-v3, wasm simd128,
    optional relaxed SIMD);
  - `Conv1d` with dilation (`ConvGeometry`) and the polyphase `ConvTranspose1d` (`src/nn.rs:1102`, `:1212`);
  - the lazy safetensors loader (`src/weights.rs`);
  - Ogg framing, the TLV mechanism, granules, loops and trimming (`src/pico.rs`, `src/layout.rs`);
  - the wasm builds and worker selection (`wasm/build.ps1`, `wasm/ts/worker.ts:108-175`);
  - the WebGPU conv kernels, which also run polyphase ConvT (`web/src/shaders.ts`).
- **Free format space:** model byte values 2+, TLV types 4, 6 and 8+. Codebook count is a u8 (up to 64), so 9
  fits.

## The student decoder, in numbers

- Width 512, 8.47M parameters, 7.91 GMAC per second of audio per channel. There is no LSTM and no GroupNorm.
- Ops: k7 conv → 4 × [Snake, ConvT stride 8/8/4/2, 3 × residual unit (Snake, dilated k7 conv at 1/3/9, Snake,
  k1 conv, add)] → Snake, k7 conv, tanh.
- Input: the sum of q codebook vectors (1024 entries of 8 dims each, projected to 1024 channels by a 1×1
  conv), at 44,100 / 512 = 86.13 frames per second. DAC is mono: stereo is two streams.
- **Receptive field: ±9 frames** (measured: one changed latent frame moves the output 8.98 frames before and
  8.91 after, about ±105 ms). So any chunk can be decoded exactly by adding 9 frames of context on each side and
  discarding the edges. That gives output identical to a whole-file decode, up to float rounding.
  - **The EnCodec segment seam doesn't exist here.** There are no segments, scales or cross-fades. The
    once-per-second "beat" from EnCodec's 1 s segments can't happen.

## Milestones

Rough sizes are for one person working with Claude, and they include tests.

### M0: export and reference fixtures (1 day)

- `export.py` gains a safetensors output (weight norm folded, plain names) next to the `.pt`. It also exports the
  9 codebooks (8-dim embeddings plus their 1×1 output projections) from the DAC quantiser.
- A small `ref_decode.py` writes fixtures for the parity tests: codes, then z, then the output after each
  stage, for 2–3 clips at q4 and q9.
- Licences: the exported weights carry the DAC MIT notice plus the training-data notice that `export.py`
  already writes (`.NOTICE.txt`, `.credits.jsonl`, the BY-SA note). Copy them next to the weights in every
  build, as the EnCodec exporters do with `LICENSES/EnCodec-MIT.txt`.

### M1: the decoder in Rust (1–1.5 weeks)

- A `ModelKind::Dac44` (or a separate `dac` module) that loads the safetensors and builds the network from the
  existing `Conv1d` / `ConvTranspose1d`.
- **Snake** is the one new op: `x + sin²(αx)/α`, per channel. Use a degree-3 polynomial with range reduction,
  fused into the conv's input path like ELU is today (`src/nn.rs:602-880`, the `Lazy`/`Input` mechanism). The
  distill study measured it at +0.005 dB log-mel against exact sin.
- Chunked decode with the 9-frame context; streaming decode is the same with a 9-frame look-ahead (about
  105 ms of latency).
- Tests: parity against the M0 fixtures, stage by stage (tolerance about 1e-4), and chunked against
  whole-file decode.
- Bench single-thread native first: that predicts wasm.

### M2: a `.pico` scheme for DAC codes, and an end-to-end CLI (3–5 days)

- **Format** (FORMAT.md and `src/pico.rs`, plus the two TypeScript header parsers in `wasm/ts/pico-header.ts`
  and `web/src/pico.ts`):
  - model byte 2 = `dac44_student`, with the weight file's hash in the header, as tables are hashed today;
  - a per-model hop (512) and rate (44,100), replacing the global `HOP`;
  - channels coded as independent mono streams, interleaved per page;
  - pages as frame ranges. 44,100 / 512 isn't a whole number, so pick a fixed frame count per page, for
    example 80 frames (0.929 s). Granules count samples, as now;
  - a raw code scheme (10 bits per code, bit-packed) so nothing waits on cx2; see M5.
- **Encoding, first version:** an offline `encode_dac.py` in `distill/` that runs DAC's stock encoder and
  quantiser (resampling to 44.1 kHz) and writes codes. `picotone pack` turns them into a `.pico`.
- **The CLI:** `picotone decode` reads model 2 files. Encode gets an HQ flag (for example `--hq q5`) that calls
  the Python tool until M6 lands.
- **Loops** (gaim's music loops): decode with the latent wrapped around, 9 frames each side, so the loop point
  has no seam. This reuses the loop flag and `wrap_loop`.

### M3: wasm and gaim (2–4 days)

- The same crate, through the existing builds (`pkg-st`, `pkg-st-fma`, `pkg-mt`). Export the decoder weights
  through `wasm/tools/export-models.mjs` (add DAC to its `TABLES` map).
- Measure the real-time factor single-threaded, at stereo. The projection is about 1.9×. That's enough for
  plain-HTTP LAN playback, but with less headroom than EnCodec's 5.5×.
- **Decision point:** if single-thread is under about 1.5× on the table device, use the threaded build (gaim's
  HTTPS setup allows it), or consider the 384-wide student (projected 3.3×).
- gaim integration: a new model in the worker, plus `wasm/INTEGRATION-gaim.md` updated with the files to copy.
  Edits to gaim itself happen only when asked.

### M4: WebGPU (1–1.5 weeks, can wait)

- WGSL kernels: dilated conv and ConvT (the existing ones mostly cover them), Snake fused into the conv input,
  f16 weights through `web/tools/export-weights.mjs`.
- Simpler than the EnCodec port: no LSTM and no GroupNorm.
- Worth doing only if M3 shows wasm is too tight on the devices that matter.

### M5: cx2 for DAC codes (1–2 weeks, measure first, maybe skip)

- DAC codes are near-incompressible on their own: the distill study measured 9.41 bits per code zeroth-order
  for codebook 0 and about 9.94–9.99 for the rest, of 10. Context modelling might win 3–8%.
- First a probe in the style of `tools/context_entropy.py`, on DAC codes from the training corpus, 1–2 days.
  Build the cx2 port only if it shows at least about 5%.
- The port: `DIM` per table (8, or the projected 1024), a period-free net (there are no segments), new
  tables, and the int parity tests. The cx2 path must stay bit-exact: no relaxed SIMD.

### M6: the DAC encoder in Rust (about 1 week, before release)

- So that `picotone` stays one command. The encoder is about 22M parameters, 30.6 GMAC/s per channel. It uses
  the same conv and Snake kernels plus DAC's factorised, l2-normalised residual VQ.
- Speed doesn't matter much, because assets are encoded offline.
- Parity: codes must match PyTorch's exactly on the fixtures. Near-ties in the VQ argmin may differ by float
  rounding, so the test should allow rare single-code differences and check the audio instead.

**In total:** about 3–4 weeks to an HQ mode that plays in gaim (M0–M3 plus M6). Add 1–1.5 weeks for WebGPU,
and 1–2 weeks for cx2 if the probe justifies it.

## Decisions for you

1. **Which tiers ship, and the HQ default.** From your listening, I'd ship q4, q5, q6 and q9 and make q5 or
   q6 the default. Settle it with one ABX round on the hardest clips (solo piano, orchestral, sharp
   transients).
2. **Stereo (decided 2026-11-21).** Measured with stock DAC as a proxy on 12 picotone clips, and by blind
   listening; see "Stereo measurements" below. In M2:
   - build the automatic mono flag (`stereo_scan.py`'s rule);
   - carry the stereo layout (left/right or mid/side) as one header bit per file, chosen by a sampled
     comparison rather than coding the whole file twice:
     1. the mono check first (`stereo_scan.py`'s rule; nearly free). A mono file has no layout to choose;
     2. 6–8 excerpts of about 3 s, spread over the file and away from silence, each with enough
        surrounding audio for the encoder and decoder to warm up. Each is encoded, decoded and scored both
        ways: log-mel on L, R and the side, averaged;
     3. one full encode in the winning layout.

     Optionally, stop sampling early when the first 2–3 excerpts already show a large margin. Costs, against
     a plain encode: about 1.1–1.4× for 5–15% of the file sampled, against about 2.5× for a full
     two-layout comparison (the decode for scoring is a fifth to a sixth of the encode). A wrong pick costs
     little: it happens mostly where the layouts score alike, and the ABX heard no difference even where
     they don't. Check it on whole tracks before building: decide about 40 songs both ways, sampled and
     full, and count agreement and the score lost on disagreement. Mid/side is the default when the
     encoder doesn't compare (`--stereo ms`);
   - no per-page side codebook counts.
3. **WebGPU now or later.** I suggest later, after the wasm numbers from M3.
4. **Raw codes first, cx2 after a probe.** I suggest raw first.
5. **Python encoder until M6.** It's fine for gaim's offline asset encoding, but it means `uv` and torch on the
   encoding machine until then.
6. **Licence notice for the weights.** The student was trained partly on CC BY-SA music, under the
   `sharealike` policy you chose. `export.py` already writes the credits and the ShareAlike note. Whether
   ShareAlike extends to trained weights is legally unsettled; carrying the notice with the weights is the
   cautious default. picotone is GPL-3.0-or-later either way.

## Stereo measurements (2026-11-21)

Stock DAC decoder as a proxy for the student, on 12 picotone test clips (10 s each); mean log-mel, lower is
better.

| | 8.6 kbps (10 codebooks/frame) | 10.3 kbps (12) |
|---|---|---|
| Left/right | 2.972 | 2.805 |
| Mid/side, best fixed split | 2.842 (7+3) | 2.662 (9+3) |
| Mid/side, side count per page (energy rule / best possible choice) | 2.837 / 2.803 | 2.657 / 2.643 |
| Per file, best of left/right and mid/side, plus the mono flag | 2.746 | 2.603 |

- **The mono flag** wins wherever it applies: mid alone at q9 (7.75 kbps) scores 2.33 / 2.41 on the two mono
  clips, against 2.63 / 2.70 for the best mid/side.
- **Per-page side counts barely help** on 10 s excerpts, which have a steady side level. Re-test on whole
  tracks before building them.
- **Caution: log-mel per channel barely sees stereo width.** It scores collapsing wide clips to mono as an
  *improvement* (hip-hop 2.620 against 2.677). So the mid/side gains are an upper bound until a score with a
  stereo-width term (log-mel on the side signal) confirms them.
- **The side-signal score splits by file.** Log-mel of the decoded side (L−R)/2 against the original's, on
  the 10 stereo clips, at 8.6 kbps:

  | Clip | Left/right 5+5: L/R, side | Mid/side 7+3: L/R, side | Better by L, R, side |
  |---|---|---|---|
  | Band | 3.33, 3.47 | 3.37, 4.08 | left/right |
  | Piano (two clips) | 2.61, 2.80 / 2.79, 3.21 | 2.72, 3.33 / 2.85, 3.62 | left/right |
  | Electronic, ambient, game, rock | 2.44–3.06, 2.93–4.26 | 2.39–2.89, 2.97–3.69 | mid/side (game a tie) |
  | Hip-hop, jazz, pop | 2.89–3.49, **10.0–12.3** | 2.68–3.48, 3.3–5.2 | mid/side, by far |

  Coding each channel on its own adds errors that don't cancel in L−R. Where the original side is quiet
  in some bands (centred vocals and bass, typical of hip-hop, jazz and pop), left/right fills them with
  noise. Wide acoustic recordings (band, piano) favour left/right instead. Side level alone doesn't predict
  the winner (hip-hop's side is −2.7 dB, as wide as the pianos), hence the encode-both-and-compare rule.
  Mono collapse scores 24–63 on the side, so the metric does see width. Log-mel over the side exaggerates
  errors in bands the mid masks, and the ABX above (which had both kinds of clip) heard no difference; so
  the stakes are small either way. Fixed splits over all 12 clips: L/R, side, mean of the three:
  left/right 2.97, 5.78, 3.75; mid/side 6+4 2.88, 3.40, 3.02; 7+3 2.84, 3.72, 3.08. At 10.3 kbps:
  left/right 2.81, 5.62, 3.59; mid/side 7+5 2.70, 3.16, 2.83; 9+3 2.66, 3.72, 2.94.
- **By ear, left/right and mid/side sound the same.** Blind ABX (`abx.py`'s page), left/right 5+5 against
  mid/side 7+3 at 8.6 kbps, on the five widest clips (band, two piano, pop, hip-hop), 3 trials each: 5 of 15
  correct (p = 0.94), no clip above 2 of 3. Mid/side doesn't audibly narrow the image, nor sound better.

**Automatic mono flag** (`stereo_scan.py`): side level relative to mid on every non-silent page of 0.93 s,
with silent pages (both channels below −60 dBFS) skipped and side below −70 dBFS counted as zero. A file is
mono when its loudest-side page is below −40 dB. Scanned over 1,030 files (500 public-domain, 500 Jamendo, the
picotone test clips):

| Worst-page side level | Files |
|---|---|
| Exactly mono (L = R) | 81 |
| −60 to −40 dB | 9 |
| −40 to −25 dB | 15 |
| Above −25 dB (real stereo) | 918 |

- The distribution splits cleanly: 90% clear stereo and 8% bit-exact mono. The 2% in between are mostly mono
  recordings with a faint stereo effect, which carry little energy below −40 dB.
- At −40 dB the rule marks 3.7% of Jamendo and 13% of the public-domain set as mono. The latter includes many
  sound effects, which matters for gaim's SFX.
- Fitting one gain between the channels (R = g·L) added no files, so that feature isn't needed.
- An override, `--stereo auto|mono|lr|ms`, covers what the rule gets wrong.
