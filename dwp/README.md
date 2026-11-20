# dwp: direct weight programming on EnCodec

Question: can an LLM read EnCodec's weights and recorded activations, work out what the parts compute, and then
**write** smaller / cheaper / better weights directly, with no gradient descent?

Everything here is Python prototyping against the stock weights in `../weights/` (never modified). Nothing in
`src/`, `entropy/`, `web/` or `wasm/` was touched. No training loop exists anywhere in this folder. Results are
tagged:

* **[W]**: pure inspection and writing (delete, zero, constant-fold, bias := recorded mean)
* **[LS]**: a closed-form least-squares / eigen solve on recorded activations
* **[fmt]**: changes the segmentation (a new container variant), no weight change
* **[dec] / [enc]**: decoder-only / encoder-only change that keeps the stock bitstream layout

All numbers are measured, on held-out audio. Nobody listened to anything: "perceptual" below means metrics
against the source audio, and the WAVs under `out/listen*/` are there so a human can check.

## 1. The once-per-second seam of the 48 kHz model (gaim's target: 48 kHz, 24 kbps and 6 kbps)

`s11_seam.py`; rows in `results/s11_seam_*.jsonl`; WAVs in `out/listen/`.

**Short answer: it cannot be fixed in the decoder. It is fixed by a longer segment overlap (a stride change in
encoder and decoder; a new container variant). At 24 kbps that costs nothing (15 codebooks + 6.25 % overlap =
24.00 kbps total, against stock 24.24) and needs no other change; at 6 kbps it costs 2 to 4 % more bits, and at the
cheap end (3 %) also a cross-fade that avoids the damaged segment edges.**

### What is wrong

The model codes 1 s segments independently, 0.99 s apart, and cross-fades the 10 ms overlap. Coding the
segments with a hard cut instead of a cross-fade shows what each segment's edges look like (two `listen` clips,
6 kbps, SNR against the source per 10 ms bin): **3.9, 7.0, 8.3, 8.7, 9.0, 9.1 dB after the cut ... 9.3, 8.6, 5.9 dB
before the next one**. So the first ~30 ms and the last ~10 ms of every segment are damaged (reflect padding
in the non-causal convolutions of encoder and decoder, LSTMs starting from zero), while the stock overlap is
10 ms and cross-fades the two worst pieces into each other. There is no spectral bump at the seam (the
short-time log-spectral distance is flat); it is a loss of waveform match, i.e. a texture change, as the
coordinator heard.

Seam depth = mid-segment SNR minus the SNR of the worse 20 ms edge, per clip, averaged over clips:

| | stock dip |
|---|---|
| two `listen` clips (orchestral, ambient), 6 kbps | 2.76 dB (start 6.87, mid 9.64, end 8.66) |
| 12-clip test set, 6 kbps | 0.77 dB (5.41 / 6.19 / 5.79) |
| 12-clip test set, 24 kbps | 1.44 dB (8.61 / 10.05 / 9.71) |

### Decoder-only attempts (stock bitstream): none works

| variant | `listen` 6 kbps dip | test 6 kbps dip | test 24 kbps dip | note |
|---|---|---|---|---|
| stock | 2.76 | 0.77 | 1.44 | |
| cross-fade: previous tail through the whole overlap | 3.33 | 1.64 | 2.76 | adds a click (log-spectral bump +1.5 to +3 dB) |
| [W] decoder LSTM starts from a written mean state | 2.96 | 1.08 | 1.82 | |
| decoder LSTM state carried over from the previous segment | 2.63 | 1.06 | 1.80 | |
| 2 / 8 / 32 latent context frames from the previous segment | 3.05 / 3.02 / 2.99 | 1.18 / 1.14 / 1.11 | 2.37 / 2.29 / 2.21 | |
| 75 context frames; 8 frames rescaled by the RMS ratio | 2.95; 3.01 | | | |
| (encoder-only) 32 frames of real audio context | 2.83 | 0.98 | 2.22 | codes differ, layout unchanged |
| (both sides) 32 context frames | 2.59 | 0.90 | 2.13 | |

Why: at the stock stride the previous segment's frame grid is offset by half a frame (47520 = 148.5 frames),
each segment is normalised by its own RMS and by GroupNorm statistics of its own activations, and the decoder
is generative. Two independently rendered segments do not line up sample-for-sample even when both saw the
right context (frame-aligned test, stride 48000, 4 context frames on both sides, hard cut: start SNR 7.1 dB
against 9.7 mid-segment, and a +2.1 dB log-spectral click; `results/s11_seam_nq4_listenclips_align.jsonl`). A cross-fade over undamaged audio is needed whatever else is done, and 10 ms of
overlap contains no undamaged audio.

### Format changes, ranked by seam removed per bit

"kbps" is the total, overlap overhead included. *Edge-avoiding cross-fade* = with an overlap of `ov` samples,
never use the last 480 samples of the old segment nor (if the overlap allows) the first 1440 of the new one, and
ramp linearly in between (exact rule below).

`listen` clips, **6 kbps** (stock = 6.06 kbps):

| variant | kbps | dip dB | start / mid / end SNR | overall SNR | log-mel | MR-STFT |
|---|---|---|---|---|---|---|
| stock (1 % overlap, Meta cross-fade) | 6.06 | 2.76 | 6.87 / 9.64 / 8.66 | 9.48 | 4.20 | 1.43 |
| 2 % overlap, Meta cross-fade | 6.12 | 1.46 | 8.26 / 9.72 / 9.59 | 9.60 | 4.20 | 1.43 |
| 2 % overlap, edge-avoiding | 6.12 | 1.94 | 7.78 / 9.72 / 9.59 | 9.59 | 4.19 | 1.43 |
| **3 % overlap, edge-avoiding** | **6.19** | **0.49** | 9.09 / 9.59 / 9.32 | 9.53 | 4.22 | 1.43 |
| 5 % overlap, Meta cross-fade | 6.32 | 0.17 | 9.83 / 9.70 / 9.53 | 9.63 | 4.26 | 1.43 |
| **5 % overlap, edge-avoiding** | **6.32** | **0.17** | 9.61 / 9.70 / 9.53 | 9.61 | 4.25 | 1.43 |
| 7.5 % overlap, edge-avoiding | 6.49 | -0.03 | 9.73 / 9.69 / 9.71 | 9.65 | 4.20 | 1.43 |
| 3 codebooks + 25 % overlap (same bits as stock) | 6.00 | 0.28 | 8.66 / 8.74 / 8.46 | 8.76 | 4.46 | 1.47 |
| 2 s segments, stock 480-sample overlap | 6.03 | 2.42, half as often | 6.99 / 9.41 / 9.22 | 9.38 | 4.23 | 1.44 |
| 2 s segments, 2400-sample overlap, edge-avoiding | 6.15 | -0.32 | 9.81 / 9.35 / 9.67 | 9.42 | 4.23 | 1.44 |
| 3 s / 4 s segments, 2400-sample overlap | 6.10 / 6.08 | 0.62 / 0.29 | mid 9.23 / 9.29 | 9.31 / 9.39 | 4.29 / 4.35 | 1.44 |
| stride = segment, hard cut | 6.00 | 4.24 | 5.43 / 9.67 / 7.26 | 9.35 | 4.39 | 1.46 |

`listen` clips, **24 kbps** (stock = 24.24 kbps):

| variant | kbps | dip dB | start / mid / end SNR | overall SNR | log-mel | MR-STFT |
|---|---|---|---|---|---|---|
| stock (1 % overlap, Meta cross-fade) | 24.24 | 2.71 | 10.33 / 13.05 / 12.47 | 12.91 | 3.57 | 1.32 |
| **15 codebooks, 6.25 % overlap, edge-avoiding** | **24.00** | **-0.33** | 13.13 / 12.80 / 13.21 | 12.87 | 3.57 | 1.32 |
| 15 codebooks, 6.25 % overlap, Meta cross-fade | 24.00 | -0.41 | 13.36 / 12.80 / 13.21 | 12.89 | 3.58 | 1.33 |
| 2 % overlap, Meta cross-fade | 24.49 | 1.59 | 11.50 / 13.09 / 12.87 | 13.00 | 3.56 | 1.32 |
| 2 % overlap, edge-avoiding | 24.49 | 2.11 | 10.98 / 13.09 / 12.87 | 12.98 | 3.56 | 1.32 |
| 3 % overlap, edge-avoiding | 24.74 | 0.47 | 12.54 / 13.00 / 13.06 | 12.95 | 3.54 | 1.32 |
| 5 % overlap, Meta cross-fade | 25.26 | 0.21 | 13.32 / 13.08 / 12.88 | 13.05 | 3.57 | 1.32 |
| 5 % overlap, edge-avoiding | 25.26 | 0.21 | 13.00 / 13.08 / 12.88 | 13.03 | 3.57 | 1.32 |

With 5 % or more of overlap the stock triangle cross-fade does as well as the edge-avoiding one (the triangle
gives the damaged edges almost no weight once the overlap is long); the edge-avoiding rule matters at 3 %.

12-clip test set (speech, piano, band, electronic, orchestral, game audio):

| variant | kbps | dip dB | start / mid / end SNR | overall SNR | log-mel | MR-STFT |
|---|---|---|---|---|---|---|
| stock, 4 codebooks | 6.06 | 0.77 | 5.41 / 6.19 / 5.79 | 6.20 | 4.09 | 1.74 |
| 5 % overlap, edge-avoiding, 4 codebooks | 6.32 | -0.03 | 6.34 / 6.31 / 6.48 | 6.25 | 4.07 | 1.73 |
| 2 s segments, 2400 overlap, 4 codebooks | 6.15 | 0.63 | 6.61 / 6.09 / 5.46 | 6.22 | 4.14 | 1.80 |
| stock, 16 codebooks | 24.24 | 1.44 | 8.61 / 10.05 / 9.71 | 10.01 | 3.40 | 1.59 |
| 5 % overlap, edge-avoiding, 16 codebooks | 25.26 | 0.17 | 9.90 / 10.06 / 10.31 | 10.06 | 3.38 | 1.58 |
| **15 codebooks, 6.25 % overlap, edge-avoiding** | **24.00** | **0.17** | 10.01 / 10.12 / 9.95 | 9.92 | 3.40 | 1.59 |
| 2 s segments, 2400 overlap, 16 codebooks | 24.62 | 0.81 | 10.67 / 9.93 / 9.12 | 10.05 | 3.49 | 1.66 |

Ranking (seam removed per bit):

1. **24 kbps: 15 codebooks + 6.25 % overlap (stride 45000).** 1 % *fewer* bits than stock, seam 1.44 -> 0.17 dB
   on the test set and 2.71 -> none on the `listen` clips, overall log-mel and MR-STFT unchanged, overall SNR
   -0.04 to -0.09 dB (the price of the 16th codebook). At this overlap the stock triangle cross-fade is as good as the
   edge-avoiding one, so **the only change needed is the stride**. This is the recommendation for gaim's 24 kbps.
2. **6 kbps: 4 codebooks + 3 % overlap (stride 46560) + edge-avoiding cross-fade**: +2.1 % bits, seam 2.76 -> 0.49 dB.
   Or 5 % overlap (stride 45600): +4.3 % bits, seam -> 0.17 dB (test set: 0.77 -> -0.03). For scale, a whole extra
   codebook (+25 % bits) buys 0.15 dB of log-mel at this bitrate, so the overlap is by far the best use of 2 to 4 % here.
   Paying for it with a codebook (3 codebooks + 25 %) is a bad trade: -0.7 dB SNR, +0.26 dB log-mel overall.
3. 2 % overlap with the stock cross-fade: +1 % bits, halves the dip. The cheapest partial fix.
4. Longer segments: fewer seams for free, but the model was trained on 1 s: mid-segment quality drops
   (2 s: mid-segment SNR -0.1 to -0.3 dB, and on the test set log-mel +0.05, MR-STFT +0.06; 4 s: log-mel +0.15 on
   the `listen` clips). Not recommended.
5. Carrying LSTM state or context across segments, on either or both sides: no measurable seam reduction.

Not measured: these fixes combined with beam-search quantisation (section 5), which would recover roughly the
SNR that 15 instead of 16 codebooks costs. Nobody has listened; `out/listen/` has the 6 kbps before/after WAVs of
both `listen` clips (`..._6kbps_01_stock...` vs `..._6kbps_06_...overlap-3-0...` and `..._6kbps_08_...overlap-5-0...edge-avoiding...`)
and the 24 kbps ones (`..._24kbps_01_stock...` vs `..._24kbps_02_fmt-15-codebooks-overlap-6-25...`). First 30 s of each clip.

### The format variant, precisely

Container: Meta's ECDC layout unchanged except two optional JSON header fields.

* `"ss"`: segment stride in samples (integer, `1 <= ss <= 48000`). Absent = 47520 (stock). Recommended: 45000 at
  24 kbps with `"nc": 15`; 46560 or 45600 at 6 kbps with `"nc": 4`.
* `"xf"`: cross-fade, `0` or absent = Meta's triangle, `1` = edge-avoiding.

Segments are still 48000 samples (150 frames) starting at `0, ss, 2*ss, ...` while the start is `< al`; the last may be
short, exactly as now. Per segment: f32 BE scale, then codes, packed as now. Encoder: only the stride changes.
Bitrate = `nc * 1500 * 48000 / ss` bit/s plus 32 bits of scale per segment.

Decoder overlap-add for `xf = 1`, with `ov = 48000 - ss`, for the `ov` samples where an old segment's tail
(index `t = 0 .. ov-1` counted from the new segment's start) meets a new segment's head:

```
hi = ov - 480 if ov > 960 else ov
lo = max(0, min(1440, hi - 240))
t <  lo        : out = old[t]
lo <= t < hi   : a = (t - lo + 1) / (hi - lo + 1);  out = (1 - a) * old[t] + a * new[t]
t >= hi        : out = new[t]
```

(If the new segment is the short final one and ends before `hi`, use the same weights for the samples it has.)
Everything outside the overlap is the single segment that covers it. For `ov = 3000` (stride 45000) that is: old
segment up to 30 ms into the overlap, a 22.5 ms linear ramp, new segment from 52.5 ms, and the old segment's last
10 ms are never played.

Rust change (not made): `Config::segment_stride()` takes the header's `ss`; `OverlapAdd::new(stride)` gets the
weight rule above selected by `xf` (it then keeps a 3000-sample tail instead of 480); `ecdc::Header` gains the two
optional serde fields; the encoder CLI gains `--stride`. Decode cost rises by the overlap (5.6 % more segments at
stride 45000) and falls by the dropped codebook (negligible). The web / wasm decoders need the same two changes.

## 2. Method

**Audio.** Public-domain / CC0 clips already fetched by the entropy agent (LibriVox speech, Musopen and Open
Goldberg piano, US military band recordings, netlabel electronic music, Brahms orchestral) plus synthetic game
audio written by `common.synth_game` (chiptune loops, laser / explosion / coin / jump / hit effects).

* test split: 12 clips x 10 s (3 speech, 2 piano, 2 band, 2 electronic, 1 orchestral, 1 chiptune, 1 SFX), from `test__*` files
* fit split: 78 clips x 10 s from different files (`fit__*`) + other synthetic seeds; used **only** as data for means and least-squares solves

**Models.** HF `EncodecModel` built from `../weights/encodec_{24,48}khz.{json,safetensors}`.
24 kHz: mono, causal, 75 frames/s, 32 codebooks (24 kbps). 48 kHz: stereo, 1 s segments, 150 frames/s, 16 codebooks (24 kbps).
In sections 3 to 6 the 48 kHz model is evaluated on non-overlapping 1 s chunks (per-chunk comparisons, overlap-add irrelevant);
section 1 uses the real overlapped segmentation.

**Two quality bars.**

* *Strict*: codes >= 99 % identical and decoder output >= 40 dB SNR versus the stock model.
* *Perceptual*: versus the **source** audio, `SNR`, `log-mel` distance (80 mels, mean |dB| difference) and a
  multi-resolution STFT loss (FFT 512/1024/2048, spectral convergence + log-magnitude L1). A variant *passes* if,
  relative to stock at the same bitrate, **d log-mel <= +0.05 dB, d MR-STFT <= +0.01 and d SNR >= -0.25 dB**.
  That is about a fifth of what halving the bitrate costs (48 kHz model 24 -> 12 kbps: +0.27 dB, +0.06, -1.8 dB;
  24 kHz model: +0.23, +0.045, -1.0). Run-to-run noise of these metrics is about +-0.01 (measured by adding white
  noise, `s07`).

**Cost.** The machine was saturated throughout, so costs are multiply-accumulate counts (MAC), not wall time.
Per second of audio each side of either model spends 21 % of its MACs in the LSTM (4 x 2048 x 512 weights per
frame) and 79 % in convolutions (`results/s00_*`). The earlier profile says the LSTM is nevertheless ~72 % of
the 24 kHz CPU time: the recurrent half of it is a sequential 4 MB matrix-vector product per layer per frame that
cannot be batched over time and does not fit in L2.

## 3. What the parts compute (interpretation)

Sources: `s01_lstm_interp.py`, `s05_integrators.py`, `s03_conv_blocks.py`, `s04_codebooks.py`; raw output in `results/`.

### LSTM block, `y = x + LSTM2(LSTM1(x))`

* **The skip carries the signal, the LSTM adds a small correction that the next layer amplifies.** In the
  24 kHz encoder the LSTM branch has 0.8 % of the energy of `x`; decoder 22 %; 48 kHz encoder 26 %, decoder 65 %.
  Yet deleting the branch is catastrophic everywhere (table in section 4), and deleting the *skip* is worse.
* **A large part of the branch is a constant.** Replacing the whole LSTM by its recorded mean output vector
  recovers most of the damage of deleting it (24 kHz encoder embedding SNR 10 -> 24 dB; a solved linear map of the
  current frame reaches 27.5 dB). The remainder is genuinely nonlinear and history-dependent: a held-out affine map
  from the current input frame explains only 24 to 62 % of the hidden-state variance.
* **Unit regimes** (fraction of the 512 units, per layer, 24 kHz encoder L1 / L2, decoder L1 / L2):
  * *memoryless gated feed-forward units* (forget gate < 0.1 for > 95 % of frames, so `c_t = i*g`): 134 / 43 / 27 / 17
  * *integrators* (forget gate > 0.9 for > 95 % of frames): 15 / 32 / 10 / 25. The longest have time constants of
    500 to 3000 frames and cell values of 100 to 360, far into tanh saturation, so their output is just `+-o_t`.
    Several are **clocks** (cell correlates 0.9 to 0.98 with the frame index), others **loudness accumulators**
    (correlation 0.7 to 0.85 with accumulated input RMS). They are why a time-chunked LSTM needs hundreds of
    warm-up frames (state-reset convergence: 27 dB after 400 frames in the encoder).
  * the rest (~85 %) have a *data-dependent* forget gate and carry 75 to 93 % of the hidden-state variance.
* **Gates are mostly driven by the input, not the recurrence**: in the 24 kHz encoder layer 1 only 5 to 10 % of
  the gate pre-activation variance comes from `W_hh h`; 20 to 55 % in the other layers.
* **Nothing is low-dimensional.** PCA of the recorded hidden state needs about 310 to 440 of 512 dimensions for 99 %
  of the variance (cell state: a handful, because the integrators dominate it). The cell input `g` is saturated
  (|g| > 0.9) 42 to 79 % of the time, gates sit in their soft range (0.1 to 0.9) 20 to 63 % of the time: this is not a
  near-linear system. `W_hh` has a strong negative diagonal on the `g` gate (-0.7 to -1.0): each unit inhibits
  its own next input.
* 48 kHz model: same picture, but each 1 s segment starts from zero state, so the clocks encode *position in
  the segment* (time constants up to 1000 frames against 150-frame segments).

### Convolutions (24 kHz encoder, first two residual blocks; `results/s03_*`)

* A residual block is `shortcut_1x1(x) + conv_1x1(ELU(conv_k3(ELU(x))))` with a C/2 bottleneck. The 1x1
  "shortcut" is **not** an identity in disguise: replacing it by the identity destroys the output (1 % codes).
* The branch is not removable: deleting it (even with its mean folded in, or with the whole block re-solved as
  one 1x1 conv) costs +0.5 to +5 dB log-mel. A solved 3-tap *linear* conv gets close in the first block
  (+0.06 dB) but costs exactly as many MACs as the block it replaces.
* The bottleneck has a little slack: 25 % of its channels can go if the 1x1 read-out is re-solved [LS]
  (d log-mel 0.00 to +0.01), but that is 0.8 to 1.7 % of the encoder's MACs per block. Without the solve [W] the
  same pruning costs +0.13 to +0.19 dB.

### Codebooks (`results/s04_*`)

* Usage is far from uniform but not degenerate: 7.7 to 8.3 bits of entropy per 10-bit code in the stored
  training `cluster_size`, 8.4 to 9.1 bits on this audio. Dead entries (training usage < 0.1 % of the mean)
  grow with depth in the 24 kHz model: about 10 in codebook 0, 149 in codebook 11, 234 in codebook 15; on the test
  audio codebook 31 uses only 507 of 1024 entries. (That is headroom for the entropy coder, which is the other
  agent's job.)
* Later codebooks hold near-duplicates: median nearest-neighbour distance falls from 0.7 of the RMS entry norm
  (codebook 1) to 0.15 (codebook 23).
* Each stage adds only 0.5 to 1 dB of latent SNR.

## 4. Replacements and ablations: tables

`d` columns = difference to stock at the same bitrate, measured against the source (so + log-mel / + MR-STFT /
- SNR are worse). "LSTM MAC" = MACs of the LSTM block relative to stock. Every row, including the ones not shown
here, is in `results/tables.md` (run `python make_tables.py`).

### 4.1 How tight are the bars? (`s07_sensitivity.py`)

White noise added to the LSTM branch output, 24 kHz / 48 kHz:

| noise level vs branch | encoder: codes identical | decoder: SNR vs stock | perceptual change |
|---|---|---|---|
| -20 dB | 34 % / 58 % | 29 / 27 dB | none (<= 0.03) |
| -30 dB | 58 % / 82 % | 39 / 37 dB | none |
| -40 dB | 81 % / 93 % | 49 / 47 dB | none |
| -60 dB | 97.7 % / 99.4 % | 69 / 67 dB | none |
| -70 dB | 99.2 % / 99.8 % | 79 / 77 dB | none |

So the strict bar demands that a rewritten encoder LSTM be accurate to **60 to 70 dB** and a rewritten decoder LSTM
to **~32 dB**, while errors a thousand times larger are inaudible to every metric. No approximate rewrite below gets
within 20 dB of the strict bar; it is only reachable by exact transformations (which `encodec-rs` already does).

### 4.2 LSTM block (`s02_lstm_replace.py`, `s08_lstm_prune_refit.py`), 24 kbps

| what was written | tag | LSTM MAC | 24k enc: codes % / d SNR / d log-mel / d MR-STFT | 24k dec: SNR vs stock / d SNR / d log-mel / d MR-STFT | 48k dec: SNR vs stock / d SNR / d log-mel / d MR-STFT |
|---|---|---|---|---|---|
| LSTM deleted (`y = x`) | W | 0 | 1.3 / -5.78 / +4.16 / +0.68 | 6.7 / -7.54 / +4.14 / +1.10 | 8.8 / -2.93 / +2.63 / +0.31 |
| LSTM := its mean output | W | 0 | 7.3 / -1.10 / +1.21 / +0.25 | 9.1 / -1.94 / +1.60 / +0.40 | 10.8 / -1.30 / +0.97 / +0.15 |
| skip deleted | W | 1 | 0.1 / -9.24 / +16.4 / +2.56 | 0.0 / -9.06 / +15.4 / +1.90 | -2.3 / -11.8 / +13.7 / +2.14 |
| layer 2 deleted | W | 0.5 | 1.1 / -6.00 / +4.61 / +0.72 | 3.4 / -10.4 / +6.55 / +1.69 | 4.7 / -5.98 / +4.28 / +0.51 |
| layer 1 deleted | W | 0.5 | 1.5 / -5.93 / +4.09 / +0.76 | 3.7 / -9.10 / +3.71 / +1.03 | 4.7 / -5.90 / +3.87 / +0.56 |
| whole block := solved 1-tap linear map (24k: folds into the previous conv, zero cost) | LS | 0.06 (0) | 10.4 / -0.66 / +0.46 / +0.08 | 11.1 / -0.75 / +1.24 / +0.18 | 11.9 / -0.70 / +0.60 / +0.09 |
| whole block := solved 7-tap linear FIR | LS | 0.44 | 10.9 / -0.56 / +0.39 / +0.06 | 11.3 / -0.56 / +1.04 / +0.14 | 12.2 / -0.57 / +0.49 / +0.07 |
| 16 units removed per layer, mean folded | W | 0.95 | 66.7 / +0.01 / +0.01 / 0.00 | 28.8 / -0.01 / +0.01 / 0.00 | 34.9 / 0.00 / 0.00 / 0.00 |
| 448 units kept, mean folded | W | 0.79 | 30.1 / -0.02 / +0.02 / +0.01 | 16.1 / -0.01 / +0.03 / +0.01 | 16.2 / -0.09 / 0.00 / +0.01 |
| 384 units kept, mean folded | W | 0.61 | 19.7 / -0.13 / +0.04 / +0.01 | 14.2 / -0.20 / +0.01 / +0.03 | 13.3 / -0.41 / +0.15 / +0.03 |
| 256 units kept, mean folded | W | 0.31 | 11.8 / -0.42 / +0.22 / +0.09 | 11.0 / -0.96 / +0.34 / +0.19 | 11.8 / -1.12 / +0.43 / +0.07 |
| 448 units kept, dropped units solved from kept ones | LS | 0.80 | 35.8 / 0.00 / 0.00 / 0.00 | 20.5 / -0.01 / +0.01 / 0.00 | 16.3 / -0.04 / 0.00 / 0.00 |
| **384 units kept, solved** | LS | **0.62** | 25.1 / -0.02 / +0.03 / +0.01 | 16.9 / -0.04 / +0.01 / +0.01 | 15.1 / -0.16 / +0.02 / +0.01 |
| 320 units kept, solved | LS | 0.46 | 19.8 / -0.07 / +0.07 / +0.02 | 14.5 / -0.21 / +0.06 / +0.03 | 13.8 / -0.39 / +0.08 / +0.02 |
| 256 units kept, solved | LS | 0.33 | 16.4 / -0.16 / +0.13 / +0.02 | 12.9 / -0.54 / +0.12 / +0.05 | 13.6 / -0.66 / +0.23 / +0.04 |
| same + every interface re-solved against stock signals (`s08`), 384 | LS | 0.66 | 25.2 / -0.03 / +0.05 / +0.02 | 15.1 / -0.02 / +0.10 / 0.00 | not run |
| same (`s08`), 256 | LS | 0.34 | not run | 13.2 / -0.41 / +0.28 / +0.03 | not run |
| `W_hh` := rank 448 in the PCA basis of the recorded state | LS | 1.05 | 47.6 / 0.00 / 0.00 / 0.00 | 26.9 / -0.01 / 0.00 / 0.00 | not run |
| `W_hh` := rank 256 in that basis | LS | 0.81 | 27.5 / -0.03 / 0.00 / 0.00 | 20.5 / -0.07 / +0.02 / +0.01 | not run |
| `W_hh` := rank 448 by plain weight SVD (earlier baseline) | - | 1.05 | 46.2 / 0.00 / -0.01 / 0.00 | 25.4 / +0.01 / +0.02 / 0.00 | not run |
| gates with std < 0.05 frozen to their mean (202 / 125 of 3072 rows) | W | 0.95 / 0.97 | 25.2 / -0.15 / +0.08 / +0.01 | 18.3 / -0.08 / +0.14 / +0.03 | not run |
| gates with std < 0.1 frozen (546 / 453 rows) | W | 0.87 / 0.89 | 17.5 / -0.23 / +0.19 / +0.03 | 12.7 / -0.39 / +0.26 / +0.08 | not run |
| layer 1 recurrence deleted, mean folded | W | 0.75 | 13.1 / -0.17 / +0.59 / +0.08 | 10.4 / -0.60 / +0.37 / +0.09 | 12.8 / -0.35 / +0.15 / +0.03 |
| layer 1 recurrence predicted from `x_t`, folded into `W_ih` | LS | 0.75 | 19.4 / -0.04 / +0.06 / +0.03 | 12.9 / -0.09 / +0.12 / +0.02 | 13.5 / -0.24 / +0.05 / +0.01 |
| both layers made feed-forward that way | LS | 0.50 | 13.2 / -0.39 / +0.17 / +0.05 | 12.2 / -0.17 / +0.25 / +0.04 | 12.4 / -0.45 / +0.20 / +0.03 |

48 kHz **encoder** (same script, stopped after the 384 rows to free the machine): LSTM deleted 6.3 % codes / -2.43 /
+1.69 / +0.14; mean only 16 % / -0.29 / +0.47 / +0.01; 1-tap linear 18 % / -0.29 / +0.29 / +0.01; 384 kept [W]
28 % / -0.02 / +0.07 / -0.01; **384 kept [LS] 36 % / -0.01 / +0.02 / -0.01**.

At 6 kbps (24 kHz encoder, 8 codebooks) the linear replacement still costs +0.38 dB log-mel, as much as dropping to
about 3.9 kbps, so "the LSTM matters less at low bitrate" is false.

Reading the table:

* **Strict bar: nothing passes**, not even removing 16 of 512 units (67 % codes, 29 dB). Consistent with 4.1.
* **Perceptual bar: 384 of 512 units per layer [LS] passes on all four LSTMs**; 320 fails narrowly; 256 clearly.
  With only the mean fold [W], 448 is the safe point (384 fails on the 48 kHz decoder).
* The unit-selection rule that worked: keep the units whose activity is *least predictable from the other units*
  (`1 / diag(inverse covariance)`, closed form) times how strongly they are read downstream. The dropped units
  are then expressed as a linear function of the kept ones and that function is folded into `W_hh` and the next
  layer's `W_ih`.
* **More solving was worse**: re-solving every interface against the stock signals (`s08`) lowers the MSE to the
  stock branch but *raises* log-mel distance (+0.10 vs +0.01 at 384 in the 24 kHz decoder). Least squares shrinks
  towards the mean and the decoder punishes the lost energy. Likewise low-rank in the PCA basis of real
  activations is no better than plain weight SVD: the state really uses ~all directions.
* Freezing quiet gates, the LSTM's "constants", is a poor trade (few MACs, visible damage).
* A tiny LSTM plus a solved linear bypass does not rescue the idea either (24 kHz decoder, `s08` with
  `DWP_BYPASS=1`): 256 units +0.24 dB log-mel, 128 units +0.54, 64 units +0.80.
* Encoder and decoder replaced together (48 kHz, 384 + 384, re-loaded from the saved files by
  `s10_combined_listen.py`): 36 % identical codes, d SNR -0.16 dB, d log-mel +0.04 dB, d MR-STFT 0.00. Still
  inside the bar, with no margin left on log-mel.

### 4.3 Convolution blocks (24 kHz encoder, `s03_conv_blocks.py`)

| block | what was written | tag | encoder MAC saved | codes % | d log-mel |
|---|---|---|---|---|---|
| L1 (C=32) | branch deleted | W | 5.0 % | 2.8 | +4.39 |
| L1 | branch := its mean | W | 5.0 % | 10.8 | +2.50 |
| L1 | whole block := one solved 1x1 conv | LS | 5.0 % | 14.9 | +0.56 |
| L1 | whole block := solved 3-tap linear conv | LS | 0 % | 37.9 | +0.06 |
| L1 | shortcut := identity | W | 1.7 % | 1.1 | +8.25 |
| L1 | bottleneck 16 -> 12, mean folded | W | 0.8 % | 26.1 | +0.19 |
| L1 | bottleneck 16 -> 12, read-out re-solved | LS | 0.8 % | 68.3 | +0.01 |
| L1 | bottleneck 16 -> 8, read-out re-solved | LS | 1.7 % | 44.0 | +0.03 |
| L1 | bottleneck 16 -> 4, read-out re-solved | LS | 2.5 % | 23.8 | +0.12 |
| L4 (C=64) | branch deleted / := mean | W | 9.9 % | 5.0 / 7.8 | +5.33 / +5.00 |
| L4 | whole block := one solved 1x1 conv | LS | 9.9 % | 14.9 | +0.82 |
| L4 | bottleneck 32 -> 24, read-out re-solved | LS | 1.7 % | 51.5 | 0.00 |
| L4 | bottleneck 32 -> 16, read-out re-solved | LS | 3.3 % | 21.3 | +0.11 |

The deeper encoder blocks and the decoder blocks were not finished (the run was stopped when priorities moved to
the 48 kHz seam work). The pattern in the two measured blocks is uniform: no block is removable, and bottleneck
pruning buys 1 to 2 % of the MACs per block at best.

### 4.4 Later codebooks removed (stock ladder, test split)

| 48 kHz model | 1.5 kbps | 3 | 4.5 | 6 | 7.5 | 9 | 12 | 18 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| SNR dB | 2.30 | 4.24 | 5.29 | 6.13 | 6.78 | 7.32 | 8.16 | 9.27 | 9.92 |
| log-mel dB | 5.66 | 4.77 | 4.42 | 4.20 | 4.05 | 3.93 | 3.77 | 3.58 | 3.49 |
| MR-STFT | 2.05 | 1.88 | 1.81 | 1.77 | 1.73 | 1.71 | 1.67 | 1.63 | 1.61 |

No codebook can be dropped inside the perceptual bar: the last four (18 -> 24 kbps) are already worth
-0.09 dB log-mel / +0.65 dB SNR.

## 5. Compression findings

| idea | tag | result |
|---|---|---|
| Decode tables re-estimated per bitrate (conditional means, exact coordinate-wise least squares), same bitstream | LS | **No gain.** 48 kHz at 6 kbps: latent error -0.14 dB (worse on held-out audio), log-mel +0.02. Overfits with 1024 x n_q free vectors, and the stock centroids are already near-optimal for their own cells. |
| Missing residual predicted by a linear FIR of the dequantised latent (folds into decoder conv 0) | LS | **Negative.** Latent SNR improves (+0.5 dB at 6 kbps, 5 taps) but audio gets *worse*: log-mel +0.12 dB, SNR -0.18 dB. The MSE-optimal latent is a shrunken, smoothed one, and the GAN-trained decoder renders it duller. At half strength it is a wash (+0.02). |
| Both of the above at 1.5 / 3 / 12 kbps and on the 24 kHz model | LS | same sign everywhere (`results/s04_*.jsonl`) |
| Rare entries removed: M-entry codebooks, more stages at equal bitrate (new fixed-rate packing, stock tables still decode it) | W, fmt | **Negative.** 48 kHz at 6 kbps, 5 x 256 entries vs 4 x 1024: log-mel +0.60 dB, SNR -1.6 dB. 10 x 512 vs 9 x 1024 at 13.5 kbps: +0.24 dB. |
| Beam-search RVQ in the encoder (not a weight edit; stock decoder reads it) | enc | **Small positive.** 48 kHz, beam 16: 6 kbps +0.26 dB SNR, -0.05 dB log-mel; 24 kbps +0.24 dB, -0.04 dB. Worth roughly 8 to 10 % of bitrate (the fifth codebook, +25 % bits, buys -0.15 dB / +0.65 dB). Costs 16 quantiser searches instead of 1 (the quantiser is a negligible part of encoding). |

The residual predictor is saved in `out/decode_corrections_{24,48}khz.safetensors` for inspection, not for use.

## 6. Verdict on direct weight programming

* **Interpretation worked.** Reading weights plus a few minutes of recorded activations gives a concrete
  picture: skip + constant + gated feed-forward units + a few saturated clocks and loudness integrators, no
  low-dimensional structure, bottlenecks with a little slack, well-used codebooks.
* **Against the strict bar, direct weight programming loses to the exact optimisations already in place.**
  Nothing approximate comes within 20 dB of it (4.1); the quantiser turns 1e-3 relative errors into different codes.
* **Against the perceptual bar there is one real candidate: a 384-unit LSTM** written by unit selection plus one
  least-squares fold (section 4.2). It removes 38 % of the LSTM MACs and about 6 MB of the 16.8 MB of weights of each
  LSTM. If the LSTM is ~72 % of CPU time that is about **x1.35 to x1.4** for the whole codec on that side
  (`s06_speed.py`: measured single-thread time ratio 0.64 at 384, but the machine was too loaded to trust the
  third digit; the 72 % figure is the earlier agent's 24 kHz profile, the 48 kHz share was not profiled but its
  MAC split is identical). The cost is that the output is no longer bit-compatible with stock: 15 to 17 dB SNR against the
  stock decoder's output, 25 to 36 % identical codes, metrics against the source unchanged within the bar.
  Derived weights: `out/lstm_<model>_<side>_pruned384_ls.safetensors`.
* Everything else tried either does not pass (linear replacement, feed-forward rewrite, gate freezing, block
  removal, 320 units and below) or passes while saving nothing useful (low rank, bottleneck pruning, 16 to 64 units).
* **Is it worth a Rust change?** Only if a x1.35 decode speed-up matters more than exactness. The change would be:
  1. `nn.rs`: let `LstmLayer` be rectangular (`input: ConvKernel` of shape `4*H x D_in`, `w_hh` of `4*H x H`);
     `Lstm::push_layer` takes `(d_in, hidden)` instead of asserting `dim x dim`; `step` already only needs `hidden`.
  2. after the last layer apply the read-out `out.weight` (512 x 384, mostly a scatter: 384 one-hot rows and
     128 dense rows) and `out.bias`, then add the skip as now.
  3. `weights.rs` / `model.rs`: when a side-car file `lstm_*_pruned384_ls.safetensors` is present, build the LSTM
     from `l0.w_ih, l0.w_hh, l0.b, l1.*, out.*` (gate order i, f, g, o as in PyTorch; `b` is already `b_ih + b_hh`).
  Recommendation: **do not ship it.** Section 7 (written after this one) shows that for 48 kHz decode the LSTM is
  only 15 to 20 % of the time, so the gain is about x1.06 to x1.15, not x1.35, and that the 384-unit decoder misses the
  bar on a second set of clips. The seam fix in section 1 is worth far more to the listener than this is to the CPU.

## 7. Listening set for the pruned LSTM, and what it really buys (added last; supersedes the speed estimate in 6)

`s12_listen_lstm.py` -> `out/listen_lstm/`; per-file metrics in `results/s12_listen_lstm.md`.
Decoded with the real segmentation and Meta overlap-add; within a clip the only difference between files is the
LSTM weights. 48 kHz clips: the two `listen` clips (30 s) plus speech, piano and chiptune (25 s each).

Mean over clips (d = change against stock, measured against the source):

| set | variant | clips | SNR vs stock output | d SNR | d log-mel | d MR-STFT | worst clip d log-mel |
|---|---|---|---|---|---|---|---|
| 48 kHz 24 kbps | decoder 448 [W] | 5 | 24.0 dB | -0.08 | +0.05 | +0.01 | +0.10 |
| 48 kHz 24 kbps | decoder 384 [LS] | 5 | 22.1 | -0.21 | +0.09 | +0.01 | +0.20 |
| 48 kHz 24 kbps | decoder 320 [LS] | 5 | 18.2 | -0.60 | +0.20 | +0.02 | +0.34 |
| 48 kHz 24 kbps | enc + dec 448 [W] | 5 | 14.9 | +0.03 | +0.04 | 0.00 | +0.09 |
| 48 kHz 24 kbps | enc + dec 384 [LS] | 5 | 14.4 | -0.05 | +0.07 | +0.01 | +0.18 |
| 48 kHz 24 kbps | enc + dec 320 [LS] | 5 | 13.1 | -0.37 | +0.19 | +0.02 | +0.32 |
| 48 kHz 12 kbps | decoder 384 [LS] | 2 | 21.3 | -0.30 | +0.13 | +0.01 | +0.20 |
| 48 kHz 12 kbps | enc + dec 384 [LS] | 2 | 13.7 | -0.01 | +0.07 | 0.00 | +0.21 |
| 24 kHz 6 kbps | decoder 384 [LS] | 2 | 16.7 | -0.29 | +0.31 | +0.04 | +0.47 |
| 24 kHz 6 kbps | enc + dec 384 [LS] | 2 | 9.3 | -0.34 | +0.32 | +0.04 | +0.46 |

**This is worse than section 4.2 suggested.** On these clips the 384-unit decoder sits at +0.09 dB log-mel
(worst clip +0.20), outside the +0.05 bar that it met on the 12-clip test split (+0.02); only 448 [W] is at the bar.
The 24 kHz model at 6 kbps clearly fails (+0.31; +0.47 on the orchestral clip). Likely reasons, not verified:
the 24 kHz LSTM is streamed over 25 s here while the solve only ever saw 10 s clips (the clock / accumulator
units reach values outside the fitted range), and the solve used full-bitrate decoder inputs. Treat "384 units
passes" as true on the test split and unproven in general. Ears decide: `out/listen_lstm/abx/`.

Blind test: `out/listen_lstm/abx/trialNN_{A,B,X}.wav` (38 trials, 10 s excerpts, one of A/B is stock, X is a copy of
A or B). The key is in `out/listen_lstm/abx_answer_key/key.csv`; do not open it first.

### Speed for the 48 kHz decoder

* MACs per frame: stock LSTM 4.19 M; 384 units 2.61 M (x0.62); the sequential recurrent part alone x0.56.
* Share of decode time: `encodec-rs --threads 1 decode` of a 120 s 48 kHz file takes 75 to 77 CPU-seconds, and the
  earlier agent's instrumented build of the same code (72 CPU-s) attributes about **15 to 20 % of single-thread decode
  time to the LSTM** (`lstm-step` + `lstm-proj`; three runs on a loaded machine). The 72 % quoted for the 24 kHz model
  does not apply to 48 kHz decode: GroupNorm, ELU and the convolutions dominate there.
* Microbenchmark of the recurrent matrix-vector product, single thread (`s13_matvec_bench.py`,
  `results/s13_matvec_bench*.txt`): 512 -> 384 took x0.22 to x0.33 of the time in short quiet bursts (619 -> 139 us
  best case; the 4 MB matrix stops thrashing L2) and x0.47 to x0.86 in three longer runs under load. The machine was
  never quiet enough to pin this down; x0.56 (MAC-proportional) is the safe figure, x0.25 the optimistic one.
* **Estimated 48 kHz decode speed-up from a 384-unit decoder LSTM: x1.06 to x1.09 by MACs, up to about x1.15 if the
  cache effect holds in Rust.** 448 units: x1.03 to x1.08.

### Rust work for a rectangular LSTM (estimate, not started)

`nn.rs` assumes square layers in more places than the constructor: `TimeBlock.data` is overwritten in place with a
layer's output (so input and output widths must match), `LstmState`, `forward_stream`, `forward_group` and the
pipeline all use `self.dim`. Needed: per-layer `(d_in, hidden)`; `run_from` writing outputs into a block of the new
width; state vectors sized per layer; a final read-out (384 one-hot rows + 128 dense rows, then the skip); a side-car
weights loader; parity tests against the files here. About 250 to 350 changed lines and 1 to 2 days including the
streaming path and the wasm build. **Verdict: not worth it for gaim.** It buys roughly 6 to 15 % of 48 kHz decode time, costs
bit-exactness and, on this listening set, measurable quality. The decode profile says GroupNorm / ELU / conv are where
exact speed-ups would pay.

## 8. Files

| file | what |
|---|---|
| `common.py` | model loading, clip sets, metrics, manual LSTM, `Bench` |
| `s00_baseline.py` | stock quality ladder, MAC budget |
| `s01_lstm_interp.py`, `s05_integrators.py` | LSTM interpretation |
| `s02_lstm_replace.py` | LSTM ablations and rewrites (one-shot) |
| `s08_lstm_prune_refit.py` | LSTM pruning with every interface re-solved (worse; kept as a negative result) |
| `s03_conv_blocks.py` | residual block interpretation and rewrites |
| `s04_codebooks.py` | codebook statistics, decoder-side corrections, small codebooks |
| `s06_speed.py` | single-thread time model of a reduced LSTM |
| `s07_sensitivity.py` | how much error each bar tolerates |
| `s09_beam_rvq.py` | beam-search quantisation |
| `s10_combined_listen.py` | encoder + decoder LSTMs replaced together, listening samples |
| `s11_seam.py` | segment-seam measurements and fixes (section 1) |
| `s12_listen_lstm.py`, `s12_table.py` | pruned-LSTM listening set with real overlap-add, blind ABX folder, its metric table (section 7) |
| `s13_matvec_bench.py` | single-thread matvec microbenchmark, 512 vs smaller |
| `make_tables.py`, `show.py`, `show_seam.py` | result printing |
| `results/` | every measured row (`*.jsonl`), raw inspection output (`*.txt`), `tables.md` |
| `out/` | derived tensors and WAVs, see `out/README.md` |
| `cache/` | cached clips and stock activations; deleted, any script rebuilds what it needs (about 2 GB, `build_fit.py`) |

Reproduce: `python build_fit.py 48khz`, then any script with the arguments in its docstring.
