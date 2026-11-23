# HQ mode: integrating the DAC student into picotone

Status (2026-11-23): M0, M1, M2 and M6 are built; `picotone encode` defaults to HQ. The shipped decoder is the
45 h speech fine-tune at step 172,500 (`weights/dac44_student.*`). Next: M3 (wasm and gaim), then M4.

- **M1:** `src/dac.rs`, `DacDecoder`; parity with PyTorch (`ref_decode.py`) passes.
- **M6 (done before M2's Python tool was needed):** `DacEncoder` in `src/dac.rs`, exported by
  `export_encoder.py`; `ref_encode.py` checks the latent (relative error under 1e-4) and codebook 0 (over
  99.5% agreement), whole and in 16-frame chunks. About 4x real time per mono channel on this laptop, so about
  2x for stereo. No `encode_dac.py` or `picotone pack` was needed.
- **M2:** the format is in FORMAT.md ("HQ streams"), `src/pico/dac.rs`. The CLI follows the design below
  (`src/main.rs`, `src/hq.rs`, `src/stereo.rs`). `ref_stereo.py` checks `picotone encode` + `decode`
  against the PyTorch pipeline for left/right 9+9, mid/side 9+9 and 7+3: they agree to under 1e-6.
  **One change from the plan:** the header hashes DAC's quantiser codebooks, not the decoder's weight file.
  The codes depend only on the encoder and quantiser, so any decoder of DAC's codes (a later student, or DAC's
  own) plays existing files, and retraining the student needs no re-encode.
- Not done yet in M2: the two TypeScript header parsers, and the 9-frame wrapped-latent loop decode (loops
  use EnCodec's `wrap_loop` padding today, which works but costs 200 ms of extra coded audio per loop).

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
subtle. Two of the clips in that listening test were Open Goldberg clips, which leaked into training (see
`README.md` §2), so a re-listen on the Chopin piano clips is due.

**Final, clean (step 107,500, smoothed weights, stopped by the plateau rule).** Scored on the 49 picotone
clips left after removing the four leaked Goldberg clips; the percentages are against stock DAC on the same
clips:

| Tier | Log-mel | vs stock, all 49 | vs stock, without the 5 speech clips |
|---|---|---|---|
| q4 | 3.112 | +2.3% | +0.9% |
| q5 | 2.899 | +2.7% | +1.2% |
| q6 | 2.739 | +3.0% | +1.4% |
| q7 | 2.598 | +3.4% | +1.8% |
| q9 | 2.371 | +4.6% | +2.9% |

(Log-mel is the set-weighted mean, as the verdict uses. The last column is a plain mean over clips, from
`clip_gap.py`.)

- **Speech is the largest gap at every tier**: 13–17% behind stock, with SI-SDR 2–3 dB worse, on the five
  LibriVox clips. The training data has no speech.
- **Without speech, every tier is within the 3% target.**
- **At q9 the music gap is spread out:** the worst 25% of clips carry 62% of it. Piano, sfx, rock and
  electronic are 5–8% behind.
- **Orchestral is 7–9% better than stock** at every tier.

**Fine-tune with extra data (`runs/w512_q9_extra`, 2026-11-22).**
- **Setup:** resumed from step 107,500. New sources: `speech` (VCTK, 20 h) and `hits` (rendered drums and piano,
  10 h), made by `prep_extra.py`. Mix `jamendo=0.6,pd=0.15,speech=0.1,hits=0.15`. Tiers `--n-q 4,5,6,7,9,9,9`.
- **Run:** the plateau rule stopped it at step 137,500. The new data first made the score 0.011 worse, and most of
  the run won that back.

Same 49 clips, smoothed weights, % vs stock (plain mean over clips, from `clip_gap.py`):

| Tier | All, base → fine-tune | Without speech | Speech |
|---|---|---|---|
| q4 | +2.1 → +2.3 | +0.9 → +1.2 | +12.8 → +12.5 |
| q5 | +2.4 → +2.6 | +1.2 → +1.5 | +13.4 → +12.5 |
| q6 | +2.8 → +2.9 | +1.4 → +1.6 | +14.7 → +13.6 |
| q7 | +3.2 → +3.1 | +1.8 → +1.9 | +15.3 → +14.0 |
| q9 | +4.3 → +3.8 | +2.9 → +2.6 | +16.7 → +14.5 |

- **Overall:** the score is unchanged, 2.7440 → 2.7446.
- **q9 improved:**
  - Piano went from +7.0% to +4.4%.
  - Speech went from +16.7% to +14.5%.
  - SI-SDR now passes, 9.51 against stock's 9.41.
- **q4–q6 got slightly worse.**
- **Orchestral lost most of its lead:** at q9 it went from −7.0% to −1.6%. Two clips are behind it: old_09
  (+0.22 log-mel) and old_08 (+0.06). old_04_band is +0.14.
  - The likely cause is the lower `pd` weight: 0.15, against about 0.2 in the baseline's auto mix. That isn't
    tested.
- **Speech barely moved.** VCTK is clean studio speech, while the eval speech is LibriVox (home mics, MP3
  sources). So more speech of the same kind may help less than speech that matches the eval.

**Second fine-tune: LibriVox speech (`runs/w512_q9_lv`, 2026-11-22). Every tier passes.**

Setup:
- Resumed from step 137,500 with 15 h of LibriVox 64 kbps excerpts added (`prep_extra.py --stages librivox`).
  The eval books, authors and named readers are left out.
- Mix `jamendo=0.5,pd=0.2,librivox=0.1,speech=0.05,hits=0.15`.
- `--stop-when-pass` stopped it at step 152,500. The checkpoint is protected as `ckpt_best_152500_smoothed.pt`
  (sha256 7960fcf7…).

| Tier | Log-mel / limit | vs stock | SI-SDR / limit |
|---|---|---|---|
| q4 | 3.098 / 3.132 | +1.9% | 4.86 / 4.58 |
| q5 | 2.882 / 2.909 | +2.1% | 6.02 / 5.75 |
| q6 | 2.717 / 2.739 | +2.2% | 6.97 / 6.73 |
| q7 | 2.573 / 2.589 | +2.4% | 7.87 / 7.64 |
| q9 | 2.334 / 2.335 | +3.0% | 9.58 / 9.42 |

The score is 2.7209, against 2.7440 for the baseline.

- **Speech, eval clips:** speech is still the largest gap, but about half the size. At q9 the ratio to stock went
  from 1.167 to 1.082, and at q4 from 1.128 to 1.080.
- **Speech, held-out VCTK speakers:** the gap fell from 23–32% to 8–9%, and SI-SDR rose by about 1.6 dB at every
  tier. This is only 2 clips and 20 s, so it doesn't prove much, but these are unseen speakers on a different
  recording chain. That suggests the gain isn't only "LibriVox MP3 sound".
- **Music, q4–q6:** jazz and pop now beat stock.
- **Orchestral** recovered only a little, 0.984 → 0.976 at q9.
- **Left at q9, besides speech:** new_06_electronic (+13%, almost the whole electronic gap; to be listened to
  before any data change), new_15_game, new_25_rock and old_14_piano.

**Third fine-tune: more speech (`runs/w512_q9_lv45`, 2026-11-23). This is the shipping candidate.**

Setup:
- Resumed from step 152,500. LibriVox went from 15 h to 45 h (`data/lv45/librivox.jsonl`, which includes the
  old 150 excerpts).
- Mix `jamendo=0.45,pd=0.2,librivox=0.15,speech=0.05,hits=0.15`, run to step 172,500 with no stop rules.
- Two held-out speech sets were logged but don't count toward the score: 2 VCTK clips, and 4 LibriVox clips by
  authors with no training rows.
- The last eval was the best. It is protected as `ckpt_best_172500_smoothed.pt` (sha256 f9bb9294…), and
  `ckpt_last_172500.pt` (5414c699…) is kept as well.

| Tier | Log-mel / limit | vs stock | SI-SDR / limit |
|---|---|---|---|
| q4 | 3.093 / 3.131 | +1.7% | 4.89 / 4.58 |
| q5 | 2.878 / 2.909 | +1.9% | 6.04 / 5.75 |
| q6 | 2.712 / 2.739 | +2.0% | 7.00 / 6.73 |
| q7 | 2.567 / 2.589 | +2.2% | 7.91 / 7.64 |
| q9 | 2.328 / 2.335 | +2.7% | 9.62 / 9.41 |

The score is 2.7157, against 2.7209 at 152.5k. q9 now passes with a margin of 0.007, up from 0.001.

The adoption criteria, set before the run, all pass:
1. **Every tier passes,** and the score is better.
2. **Held-out speech is better,** at q4 2.731 → 2.685 and at q9 1.992 → 1.942. That's −0.06 on the 4 LibriVox
   clips and −0.018 on the 2 VCTK clips. VCTK improved steadily and levelled off after about 167.5k.
3. **No music type is more than 1 point worse at q9.** The worst is band at +0.24. Ambient (−0.49), orchestral
   (−0.24) and rock (−0.21) improved. Speech on the eval clips went from 8.3% to 6.5% behind stock.

The scores were still falling at the end, so more training would likely gain a little more.

The cost is decode speed: the student is projected at about 1.9× real time in single-thread wasm
(stereo), against about 5.5× for EnCodec 48 kHz. A review (2026-11-21, see "Decode speed" below) thinks
1.9× is optimistic: picotone's own EnCodec numbers imply 0.9–1.3×. It needs a measurement. So picotone keeps
two modes:

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
- **Fold the codebook projections into the input conv.** z is a sum of 9 projected 8-dim vectors, so the
  k7 1024→512 input conv can take the 72 concatenated embedding values directly (k7 72→512, biases folded).
  That's 3.7% fewer MACs and 3.4M fewer of the 8.47M parameters, about 40% less to download, and the output
  is the same up to float rounding. Details to get right:
  - **Biases must switch off with their codebook.** Each 1×1 projection has a bias. Merged into the conv
    bias, it would wrongly count at tiers below q9, where codebooks q+1..9 are absent, and on the 3 zero-padded
    frames at each end of a file, where the original adds nothing. Instead, give each codebook a ninth input
    value, 1 when the codebook is present and 0 otherwise, and fold its bias into that column. The input is
    then 9 × (8 + 1) = 81 values per frame (k7 81→512), and zero padding and tier masking are exact.
  - **Fold in float64** and round to float32 once.
  - **Fixtures:** the folded decoder never forms z, so the stage-by-stage fixtures check the input conv's
    output instead. `ref_decode.py` keeps the unfolded form and asserts the two agree at every tier and at
    the file edges.
  - The folded weights hold DAC's frozen codebooks, under the DAC MIT notice already carried. Nothing else
    needs z: the encoder uses DAC's own quantiser, and cx2 (M5) can read the 8-dim embeddings from the
    codebook table.
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
- **Before M1, a half-day wasm benchmark** of the conv kernels at the student's ten layer shapes, with
  random weights, on the table device. It replaces the 1.9× projection with a number and gives the
  per-stage time split.
- Chunked decode with the 9-frame context; streaming decode is the same with a 9-frame look-ahead (about
  105 ms of latency). **Don't re-decode the full 9 frames for every page:** at 80-frame pages that wastes
  22%. Only the input conv and stage 1 need most of the context (about 3 and 5 frames); stages 2–4, which are
  68% of the MACs, need under one frame. **Both fixes are planned** (decided 2026-11-21):
  1. **Trimmed context, stage by stage** (about 5% waste). Each stage computes only the span the next stage
     needs. Every page stays independently decodable, for seeks, parallel workers and threads, and the
     first page after a seek.
  2. **Carried state** (no waste). Each layer keeps the tail of its input from the previous page, so playback
     in order and loop decode-ahead compute nothing twice. It keeps the 9-frame look-ahead (about 105 ms).
     After a seek it starts from a page decoded with fix 1.

  Tests: whole-file decode, fix 1 per page and fix 2 in sequence must agree up to float rounding, including
  across seeks and loop wrap-around.
- **Load weights without extra copies.** Today `Weights::from_bytes` converts every tensor to f32 and
  `PackedMat::pack` packs it again, so loading peaks at 2–3 copies of the weights. Wasm memory never shrinks,
  so that peak stays reserved. Stream tensor by tensor into the packed form instead. That takes a decoder
  instance from about 60–100 MB to about 25–40 MB (with the folded input conv), which matters for the
  per-channel workers in M3.
- **Dilated convs** (48% of the MACs) go through an im2col copy today (`src/nn.rs:525-536`). Each tile
  allocates and zeroes a fresh buffer, then copies 7 taps into it. EnCodec has no dilated convs, so the student
  is this path's first heavy user. The copy is about 1/cout of the work: 0.2% at width 512, about 3% at
  stage 4's 32.
  1. **Reuse a per-thread buffer** instead of `vec![0.0; rows * kc]` per tile, as `TILE_INPUT` does. About
     ten lines, no risk.
  2. **Only if the profiler's `im2col` scope shows more than about 3%** of the student's decode: read the 7
     taps in place. Do it by giving `Block::rows` (`src/kernels.rs:357`) one A offset per tap segment, with
     the running sums kept in registers across taps. Keep each weight row in the same place in the summing
     order, so results stay bit-identical. 1–2 days. Not one product per tap: that re-reads and re-writes
     the output block for every tap, about 7 passes instead of 2 in the 32-channel layers, which cancels the
     saving where the copy matters.

  The kernel is shared with the EnCodec decoder and LSTM: every existing parity test must still pass,
  bit-identical.
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
- **The CLI (decided 2026-11-23):**
  - `picotone encode` defaults to the DAC student (HQ), provided its decode runs faster than real time on
    the user's laptop. It calls the Python tool until M6 lands.
  - `--fast` selects EnCodec instead.
  - `--model <name>` names the exact model and overrides both. Names: `dac44-student`, `encodec-48k`,
    `encodec-24k`. Any future model joins the same list.
  - `--channel-format auto|mono|stereo` sets the channel count, separately from the model. `auto` follows
    the input.
    - DAC student: mono is one downmixed stream; stereo is two streams.
    - With `--fast`: mono gives `encodec-24k`, stereo gives `encodec-48k`.
    - A model that can't do the requested format is an error, for example `--model encodec-48k
      --channel-format mono`.
  - The stereo layout stays its own flag, `--stereo ms|lr|pick`. The default is to pick from sampled excerpts;
    see decision 2.
  - `picotone decode` needs no model flag, because the file's header names the model.
- **Loops** (gaim's music loops): decode with the latent wrapped around, 9 frames each side, so the loop point
  has no seam. This reuses the loop flag and `wrap_loop`.

### M3: wasm and gaim (2–4 days)

- The same crate, through the existing builds (`pkg-st`, `pkg-st-fma`, `pkg-mt`). Export the decoder weights
  through `wasm/tools/export-models.mjs` (add DAC to its `TABLES` map).
- Measure the real-time factor single-threaded, at stereo. The projection is about 1.9× (possibly
  0.9–1.3×, see the benchmark in M1).
- **Parallel decode, by page type** (decided 2026-11-21):
  - **HTTPS pages** (gaim serves the LAN table over HTTPS with a self-signed certificate): the threaded
    build. The threads share one copy of the weights, so memory barely grows.
  - **Plain-HTTP pages:** one Web Worker per channel, each with its own wasm instance. Workers don't need
    cross-origin isolation; only shared memory does. That's about 2× for stereo on any device with 2 or more
    cores, for about 25–40 MB more memory (after the M1 loader fix; about 60–100 MB without it). The ±9-frame
    context makes chunks independent, so more workers can split one channel further.
- **Decode music loops ahead,** at scene load, into an AudioBuffer (`decodeToAudioBuffer`), so loops depend
  on load time rather than on real-time speed. That costs about 21 MB per stereo minute (f32).
- **Decision point:** if the table device is still under about 1.5× after parallel decode, try the model
  options under "Decode speed" below, starting with bottleneck residual units.
- gaim integration: a new model in the worker, plus `wasm/INTEGRATION-gaim.md` updated with the files to copy.
  Edits to gaim itself happen only when asked.

### M4: WebGPU (1–1.5 weeks, straight after M3)

The fastest option: about 6.8× real time projected for stereo, against about 1.9× in wasm. It needs a secure
context (`web/src/gpu.ts:23`), which gaim's HTTPS LAN setup provides. Wasm stays as the fallback for
devices without WebGPU and for plain-HTTP pages.

- WGSL kernels: dilated conv and ConvT (the existing ones mostly cover them), Snake fused into the conv input,
  f16 weights through `web/tools/export-weights.mjs`.
- Simpler than the EnCodec port: no LSTM and no GroupNorm.
- With WebGPU or threads giving the headroom, q9 can keep the full 512-wide decoder, with no quality traded
  for speed.

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

**In total:** about 3–4 weeks to an HQ mode that plays in gaim (M0–M3 plus M6), plus 1–1.5 weeks for WebGPU
(M4, planned straight after M3), and 1–2 weeks for cx2 if the probe justifies it.

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
     they don't. Mid/side is the default when the encoder doesn't compare (`--stereo ms`).

     **Checked on whole tracks (2026-11-21, `stereo_pick_check.py`, the student at step 107,500, 40 tracks of
     up to 5 min: 20 Jamendo, 20 public domain).** 4 tracks were mono by the rule. Of the 36 compared, the full
     comparison picked mid/side for 29 and left/right for 7. The gap between the layouts had a median of
     0.18 log-mel and a maximum of 3.9.

     | Sampled pick | Agrees with full | Mean / max loss | Audio sampled |
     |---|---|---|---|
     | first 1 excerpt | 34 / 36 | 0.008 / 0.25 | 1.7% |
     | first 2 excerpts | 36 / 36 | 0 / 0 | 3.3% |
     | first 4 excerpts | 35 / 36 | 0 / 0.001 | 6.7% |
     | all 8 excerpts | 34 / 36 | 0.010 / 0.34 | 13.3% |
     | early exit after ≥ 2, margin 5% | 35 / 36 | 0 / 0.001 | 4.5% |

     More excerpts did slightly worse. On one jig, the later loud windows favour left/right while the whole
     track favours mid/side (2.73 against 2.39). The other miss differs by 0.002. The "first k" rows use the
     first k of 8 regions, which covers only the start of the track. So the plan is **4 excerpts spread over
     the whole file, with an early exit after 2 when the margin is above 5%**: about 5% of the audio, about
     1.1× a plain encode, and a mean loss near zero;
   - no per-page side codebook counts.
3. **WebGPU (decided 2026-11-21):** straight after M3. The LAN table runs over HTTPS, so WebGPU works there.
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

## Decode speed (review, 2026-11-21)

These figures come from counting operations and reading the code; nothing was timed.

**Where the MACs go** (7.91 GMAC per second of audio per channel):

| Part | Rate, channels | Share |
|---|---|---|
| Input k7 conv | 86 Hz, 1024→512 | 4.0% |
| Four ConvT upsamplers | | 13.7% |
| Stage 1 residual units | 689 Hz, 256 | 13.7% |
| Stage 2 | 5.5 kHz, 128 | 27.4% |
| Stage 3 | 22 kHz, 64 | 27.4% |
| Stage 4 | 44.1 kHz, 32 | 13.7% |
| Output k7 conv | | 0.1% |

- Residual units are 82% of the cost: their k7 convs 72%, and the dilated ones (3, 9) 48%.
- Snake is about 3–5% of the time.

**Exact options (same output, no training)** are in M0, M1 and M3:
- the folded input conv;
- per-layer context instead of a full 9-frame re-decode per page;
- a loader that doesn't copy;
- parallel decode;
- decoding loops ahead.

**Model options**, each needing a fine-tune. q9 is already 0.05 over its limit, so none is free. The
quality costs are guesses:
- bottleneck residual units (k7 C→C/2, Snake, k1 C/2→C): about 1.7×, keeping the full-width residual stream;
- uniform 384 wide: 1.75×, quality measured in the distill study;
- trimming only the high-rate stages:
  - drop dilation 9 at stages 3–4: 1.16×;
  - k3 instead of k7 at stages 3–4: 1.26×;
  - widths 96/48 at stages 2–3: 1.38×;
- a sub-band or iSTFT head in place of stages 3–4: up to 1.9×, but a full retrain and the highest risk.

A q9-specialised fine-tune of the 512 student gives no speed, but may recover the 0.05 and leave room for one
of these.

**Not worth it now:**
- f16 weights: download size only, since wasm has no f16 arithmetic;
- int8: weeks of quantisation-aware training, with a noise-floor risk;
- unstructured pruning: no gain for dense SIMD.
