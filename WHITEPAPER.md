# picotone: a practical EnCodec for game audio in the browser

*White paper, November 2026. The project was called encodecx until it was merged into one crate and one
format, `.pico` (Ogg plus the cx2 entropy coder). Sections 2–6 describe the work that led there, with the
measurements as they were made; §3.5, §6.3 (last part), §7 and §8 give the numbers in the merged format.*

## Abstract

picotone is a reimplementation of Meta's EnCodec neural audio codec, written for one consumer: a local web
game that streams 1–1.5 minute stereo music loops and short sound effects to a browser. It has four parts:

- a pure-Rust encoder and decoder that is bit-exact with the reference;
- two browser decoders (WebGPU and WebAssembly);
- a lossless entropy coder (cx2) and its file format, `.pico`, which is Ogg;
- a study of what can be learned and changed by reading the network's weights directly.

Five changes matter most in practice:

1. **The seam fix.** Stock EnCodec at 48 kHz has an audible once-per-second "beat". It comes from the damaged
   edges of its independently coded 1 s segments. A longer segment overlap (stride 44800 instead of 47520)
   removes it, and costs nothing at 24 kbps.
2. **Beam search.** Beam-search residual vector quantisation gives a free +0.16 to +0.27 dB SNR. It costs only
   encode time.
3. **Gapless loops.** Each loop is coded with wrap-around padding that the decoder trims away.
4. **cx2.** A product-of-experts entropy coder with one small integer neural network. It makes music files
   13–30% smaller and decodes at about 60× real time on one CPU thread. Meta's language model saves more, but
   it runs at 0.2–0.3× real time.
5. **The weight study.** Direct weight programming, meaning editing weights by analysis rather than gradient
   descent, explained a lot about the network. It produced no edit that improves on exact engineering.

## 1. Background

EnCodec (Défossez et al., 2022) is a convolutional encoder and decoder with LSTMs, joined by a residual vector
quantiser (RVQ) of up to 32 codebooks of 1024 entries each.

| Model | Audio | Frames | Codebooks | Kbps per codebook | Segmentation |
|---|---|---|---|---|---|
| `encodec_24khz` | 24 kHz mono, causal | 75 per second | up to 32 | 0.75 | none (one stream) |
| `encodec_48khz` | 48 kHz stereo, non-causal | 150 per second | up to 16 | 1.5 | 1 s segments, 47520 samples apart, triangle cross-fade over the 480-sample overlap |

The 48 kHz model also normalises each segment by its RMS and stores that scale.

Meta's tool stores codes in an `.ecdc` file: a JSON header, then 10-bit packed codes. It can optionally add a
small transformer language model (LM) for entropy coding. This project started out reading and writing that
format, with a few added header fields; the merge replaced it with `.pico` (§7), and picotone no longer reads
or writes `.ecdc`.

The consumer's needs shaped every decision:

- music at 48 kHz stereo, around 24 kbps;
- playback in a browser, sometimes on a LAN device reached over plain HTTP, which is not a secure context;
- loops that join without a click;
- no requirement to stay compatible with Meta's tool.

## 2. The Rust implementation

### 2.1 Exactness

The encoder reproduces the Hugging Face `transformers` reference exactly in greedy mode (`--beam 1`):

- 100% of codes agree on both models;
- per-segment scales agree exactly;
- decoder output is within 117 dB (24 kHz) and 120 dB (48 kHz) SNR of the reference decoder fed the same
  codes.

The parity check (`tools/parity.py`) encodes synthetic audio with both and compares them. It is a test only;
picotone's files are its own format.

### 2.2 Speed and memory

The code is safe Rust. `unsafe` is forbidden, and clippy's `pedantic`, `nursery` and `cargo` groups are
denied. Its main techniques:

- **Convolutions:** run as GEMMs over an overlapping strided im2col view with virtual padding, so no padded copy
  is made. ELU is fused into the GEMM output.
- **GroupNorm:** moments are computed per tile, and normalisation, affine and ELU are collapsed into one pass.
- **Transposed convolutions:** run in polyphase form.
- **LSTM:** the input projection is batched over time, and the recurrent step works as a pipeline.
- **Parallelism:** rayon, over segments and over tiles.
- **Loading:** only what a command needs is read: the encoder, the decoder, and only the codebooks the file uses.
- **Streaming:** each stage releases its copy of the audio as soon as the next exists.

An optimisation pass cut CPU-seconds 2–4× and peak memory sharply (24 kHz decode: 806 MB to 102 MB). Output
stayed byte-identical.

A second pass targeted decode. Profiling a 120 s `music24` file on one thread showed the LSTM at only 26%
of the time; ELU, GroupNorm, allocation and the convolutions took the rest. Five changes followed:

- **Packed-panel GEMM:** a GEMM kernel written in safe `pulp` SIMD, AVX2+FMA natively and SIMD128 in wasm, with
  weights packed once at load. The wasm shape uses fewer registers, so V8 doesn't spill them.
- **LSTM:** one batched GEMM per time step, with vector sigmoid and tanh; segments in a chunk are batched together.
- **ELU, GroupNorm and the residual sum fused into the convolutions:** a vector `expm1` for ELU (about 1 ulp),
  then GroupNorm's affine, the residual sum and ELU applied as the next convolution reads each tile.
- **Faster GroupNorm statistics:** computed in short f32 blocks and summed in f64.
- **A buffer pool:** removes page-fault costs.

| 120 s file, one thread | Before | After | Gain |
|---|---|---|---|
| 48 kHz `music24` | 26.3 CPU-s | 10.9 CPU-s (11× real time) | 2.42× |
| 24 kHz | 15.2 CPU-s | 6.55 CPU-s (18× real time) | 2.32× |

| Profile stage, 48 kHz, one thread | Before | After |
|---|---|---|
| LSTM | 6.97 s | 1.88 s |
| Convolution GEMMs | 9.67 s | 7.37 s |
| ELU and GroupNorm | 7.30 s | 1.13 s |
| Allocation and copies | 1.81 s | 0 |

All threads (16) gain 2.1–2.2×. Scaling across threads is poor, possibly because of the laptop's power limit.

The output is 115–122 dB from the previous build's and bit-identical across thread counts. Greedy codes still
match the reference 100%.

GEMM is now 70% of the time and runs near the kernel's in-cache speed. With the LSTM at 17%, a pruned 384-unit
LSTM (§9.2) would give only about 1.07×.

### 2.3 Streaming APIs

- `stream_decoder()` decodes the causal 24 kHz model as it goes. It carries convolution history and LSTM
  state across pushes, buffers 7 frames of look-ahead at the start, and holds back the last 320 samples until
  `finish`.
- `decode_segment()` with `overlap_add()` does the same for the 48 kHz model. It emits only samples that no
  later segment can change.

Both are bit-identical to a whole-file decode. Their memory depends on the chunk size, not the file length.

## 3. The once-per-second seam

### 3.1 Diagnosis

At 6 kbps the 48 kHz model has a distracting pulse about once a second; listeners rejected that setting
because of it. Averaging the error envelope over segment phase shows its cause:

- every 1 s segment has a damaged start, about 30 ms;
- and a damaged end, about 10 ms.

The damage comes from reflect padding in the non-causal convolutions, and from the LSTMs starting at zero
state. Coding with hard cuts gives per-10 ms SNR of 3.9, 7.0, 8.3, 8.7 … dB after a cut, and 9.3, 8.6,
5.9 dB before one.

The stock overlap is only 10 ms, so it cross-fades two damaged pieces into each other. That produces a
2–3 dB quality dip at every seam, at every bitrate. It is a loss of waveform match, a texture change, not a
spectral bump, and it is most audible at low bitrates.

### 3.2 Things that do not fix it

None of these reduced the dip measurably (details in `dwp/README.md` §1):

- decoder-only changes, which keep the stock bitstream: a different cross-fade, a carried-over or "mean" LSTM
  state, 2–75 frames of latent context;
- encoder context;
- context on both sides.

Two segments rendered independently do not line up sample for sample. One reason is that the stock stride is
148.5 frames, a half-frame offset. The others are per-segment RMS and GroupNorm statistics, and the decoder's
generative nature. A cross-fade over *undamaged* audio is needed, and 10 ms of overlap contains none.

### 3.3 Fix: a longer overlap

Two stream parameters carry the fix (in the first container they were the optional `.ecdc` header fields
`ss` and `xf`; in `.pico` they are the ID header's stride and edge flag):

- the segment stride in samples;
- the cross-fade rule: EnCodec's triangle, or edge-avoiding.

The edge-avoiding rule matters only for short overlaps. With `ov = 48000 − stride`:

```
hi = ov − 480 if ov > 960 else ov
lo = max(0, min(1440, hi − 240))
out = old before lo, a linear ramp from lo to hi, new from hi on
```

At 5% overlap or more, the stock triangle does as well, because it gives the damaged edges almost no weight.

| Setting | Total kbps | Seam dip, two music clips | Seam dip, 12-clip test set |
|---|---|---|---|
| Stock, 16 codebooks | 24.24 | 2.71 dB | 1.44 dB |
| 15 codebooks, stride 45000 (6.25% overlap) | 24.00 | −0.33 dB (none) | 0.17 dB |
| Stock, 4 codebooks | 6.06 | 2.76 dB | 0.77 dB |
| 4 codebooks, 5% overlap | 6.32 | 0.17 dB | −0.03 dB |

At 24 kbps, dropping the 16th codebook pays for the overlap, so the fix is free. At lower bitrates it costs a
few percent of bits, which is far cheaper than adding a codebook (+25% bits for about 0.15 dB log-mel).

Longer segments (2–4 s) were rejected: the model was trained on 1 s, and mid-segment quality drops.

### 3.4 Presets

The user preferred more overlap over a lower bitrate, so all presets share one stride. It was 45000 at first
(24.03 / 12.83 / 6.43 kbps with 32-bit scales); the merge moved it to **44800**, exactly 140 latent frames, so
segment starts fall on frame boundaries:

| Preset | Codebooks | Bitrate before cx2 | Cross-fade | Beam |
|---|---|---|---|---|
| `music24` | 15 | 24.12 kbps | triangle | 8 |
| `music12` | 8 | 12.87 kbps | triangle | 8 |
| `music6` | 4 | 6.44 kbps | triangle | 8 |

The bitrate is `nc × 1500 × 48000 / stride` plus 12 bits of scale per segment (about 13 bit/s). After the
change, the seam dip is about 0 dB or negative at every preset (§3.5).

### 3.5 Stride 44800

The seam check was repeated at 44800 with the merged encoder (beam 8, 12-bit scales) on the two 120 s listening
clips. The dip is the median quality over the middle of the segment minus the worst bins at its edges, in
448-sample bins:

| Preset | orchestral | ambient-electronic | mean |
|---|---|---|---|
| `music24` | −0.31 dB | +0.08 dB | −0.12 dB |
| `music12` | −0.14 dB | +0.09 dB | −0.02 dB |
| `music6` | −0.25 dB | −0.02 dB | −0.14 dB |

No seam is measurable. The whole `.pico` files of those clips, headers and Ogg framing included:

| Preset | orchestral | ambient-electronic |
|---|---|---|
| `music24` | 17.51 kbps | 17.66 kbps |
| `music12` | 8.30 kbps | 7.79 kbps |
| `music6` | 4.02 kbps | 3.22 kbps |

## 4. Beam-search quantisation

Greedy RVQ picks the nearest entry at each stage. Beam search keeps the N best partial sums instead, and picks
the best full path at the end. The decoder and the bitstream do not change.

| Bitrate | SNR gain | Log-mel change |
|---|---|---|
| 6 kbps | +0.21 to +0.23 dB | −0.05 dB |
| 12 kbps | +0.27 dB | |
| 24 kbps | +0.16 dB | −0.03 dB |

That is worth roughly 8–10% of bitrate. Beam 8 costs about 2% more encode time at 6 kbps and 33% more at
24 kbps. Beam 16 adds almost nothing. Beam 8 is therefore the default; `--beam 1` restores the reference's
exact greedy codes.

Beam search also does not hurt cx2. On matched fits, beam-8 files come out −0.7% to +0.7% the size of beam-1
files (§6.5).

## 5. Gapless loops

EnCodec's first and last few milliseconds are not faithful, so a decoded loop clicks where it wraps. With
`--loop`, the encoder:

1. prepends the last 150 ms of the clip and appends its first 50 ms, then codes the result;
2. records a pre-skip of 7200 samples, an end trim of 2400 and the loop flag (at 48 kHz).

The decoder then:

1. drops the pre-skip and the end trim, so the output length equals the input's;
2. cross-fades the coded pre-roll, which is a faithful rendering of the loop's own end, into the last 100 ms of
   the output.

The last sample therefore runs straight into the first. The sample jump at the wrap fell:

- from 0.094 to 0.008 on an ambient clip at 6 kbps;
- from 0.149 to 0.006 on the WebAssembly test's synthetic loop.

## 6. cx2: lossless entropy coding

### 6.1 Why

At 10 bits per code, codes are far from random. Each codebook's usage is skewed, codes repeat in time, and each
RVQ stage depends on the ones before. Meta's LM exploits this, but at 0.18–0.34× real time on one thread.
That is too slow for browser playback, and it has to run on both encode and decode. The goal was most of the
LM's saving at a speed a browser can afford.

### 6.2 Model

cx2 is a product of experts over the 1024 symbols of each code, combined in the log domain with fitted weights.
(Earlier drafts called the coder family cx1; the stream scheme was always `cx2`.)
Its experts:

| Expert | What it uses |
|---|---|
| prior | per-codebook usage |
| geo | the RVQ geometry: how near each centroid is to the current partial reconstruction |
| rec | recency of codes |
| pair | the same codebook in the previous frame |
| stat | adaptive statistics |
| hist | a low-rank (SVD) model of pointwise mutual information |

A range coder then turns the combined probabilities into bits.

The constructed experts alone fell well short of the LM at 48 kHz. The missing part is non-linear dependence on
short context. The hybrid adds one small gradient-trained MLP for it:

- about 1.8 M parameters;
- two hidden layers of 384 and 256 units, with low-rank output heads of rank 64 and 32;
- trained for 200,000 steps on 6.7 hours of audio, a subset of the fit corpus.

The fit corpus is 1,118 clips, 24.3 hours, all public domain or CC0 and listed in `DATA_SOURCES.json`.
It spans game music and effects, rock, pop, electronic, hip-hop, jazz, ambient, folk, world, speech and vocals.
The constructed tables are fitted on all of it.

Everything that touches probabilities is integer arithmetic. That includes the network: int16 weights with
32-bit accumulation. Coding is therefore bit-identical on any CPU, in WebAssembly and on a GPU. Floats would
not be, because the encoder and decoder must agree on every probability exactly. The float-to-integer
conversion costs only 0.0001–0.0002 bits per code.

### 6.3 Results

**Phase 1**, constructed experts only, fitted on 2–3 hours of audio. Size saved against the plain file:

| Model and bitrate | cx2 | Meta's LM |
|---|---|---|
| 24 kHz, 6 kbps | 22.8% | 18.9% |
| 24 kHz, 24 kbps | 16.9% | 14.9% |
| 48 kHz, 6 kbps | 21.3% | 37.7% |
| 48 kHz, 24 kbps | 11.6% | 24.3% |

**Final tables, phase 2.** The files are beam-8 preset files in Ogg, page overhead included, from held-out
clips by creators not in the fit set. Size saved against the plain file, on the old test set (23 clips) / new
test set (30 clips):

| Preset (page length) | Best generic compressor | Constructed only | Hybrid (net + rec + pair) | Meta's LM, payload only |
|---|---|---|---|---|
| `music24` (1 s) | 1.2% / 1.8% | 10.4% / 6.4% | 19.4% / 12.7% | 25.7% / 20.8% |
| `music12` (2 s) | 3.2% / 2.7% | 14.5% / 9.7% | 25.5% / 16.7% | 32.2% / 25.5% |
| `music6` (4 s) | 7.7% / 3.7% | 20.1% / 14.2% | 29.2% / 19.4% | 37.6% / 30.0% |

The same files in bits per code; the plain file is about 10.0 bits:

| Preset | Constructed | Hybrid | Meta's LM |
|---|---|---|---|
| `music24` | 8.98 / 9.38 | 8.07 / 8.74 | 7.44 / 7.94 |
| `music12` | 8.58 / 9.06 | 7.47 / 8.36 | 6.80 / 7.48 |
| `music6` | 8.04 / 8.64 | 7.12 / 8.12 | 6.28 / 7.04 |

The trained network roughly doubles the constructed saving, and the hybrid closes about half the gap to Meta's
LM. The new test set is harder: it is more varied, with pop and vocals. Keeping every constructed expert beside
the network adds nothing at `music24` and is slower, so the shipped hybrid is the lean one.

Speed, single thread, on an idle machine, with the full determinism check (pack / unpack, in multiples of real
time):

| Table | `music24` | `music12` | `music6` |
|---|---|---|---|
| Constructed | 56× / 50× | 75× / 76× | 160× / 161× |
| Hybrid | 63× / 61× | 112× / 106× | 197× / 193× |

The 24 kHz model at 32 codebooks runs at 69× / 59×. The network costs 3.0 µs per code. The hybrid is faster
than constructed-only because its leaner expert set more than pays for the network.

**More data helps the constructed tables.** Refitting on 24.3 hours instead of 2 raised the saving on the new
test set by 2–7 points, for example from 11.1% to 16.0% at 4 codebooks. On the old test set it rose by under 2.
The generalisation gap shrank most for the 24 kHz model.

**In the `.pico` format.** The merge moved the tables to version 5 (the fixed-point centroids are stored in
the table, so no EnCodec weights are needed to decode) by conversion, not retraining: the predictions are the
same, and `build_tables.py` with the shipped settings rebuilds the net table byte for byte. The streams changed
(12-bit scales coded inside each chunk, leaner varint headers, the end trim and length in the header). Measured
with `picotone dev check` on the same preset test sets, bits per code including every header and page:

| Preset | old set: before / after | new set: before / after | saved vs plain, old / new set |
|---|---|---|---|
| `music24` | 8.07 / 8.05 | 8.74 / 8.72 | 19.6% / 13.0% |
| `music12` | 7.47 / 7.44 | 8.36 / 8.33 | 25.8% / 17.0% |
| `music6` | 7.12 / 7.07 | 8.12 / 8.05 | 29.7% / 19.9% |

That is 0.3% (music24) to 0.8% (music6) smaller than the previous container; the low bitrates gain most
because headers and scales were a larger share there. With 8 parallel jobs, pack and unpack ran at 33–101× real
time per file.

**The `net48v2` retrain.** The network was retrained on the full 24.3-hour corpus (13.1 M frames) for 180,000
steps at batch 2048 (about 28 passes, 40 minutes on an RTX 3090), with an embedding for 15 codebooks and a
learned position-in-segment input (33 buckets over the 150 frames). A learning-curve check first showed little
gain from data alone: at 40,000 steps, doubling from 50% to 100% of the corpus gained only 0.014–0.018 bits per
code at 15 codebooks. The network alone reached 7.75 / 8.05 bits per code at 15 codebooks (old / new set),
against 7.87 / 8.33 for the first network at 16, with a float-to-integer gap under 0.0003. In the shipped
hybrid table, same test sets and checks:

| Preset | old set: `net48` / `net48v2` | new set: `net48` / `net48v2` | saved vs plain, old / new set |
|---|---|---|---|
| `music24` | 8.047 / 7.998 | 8.717 / 8.635 | 20.1% / 13.8% |
| `music12` | 7.438 / 7.388 | 8.326 / 8.231 | 26.3% / 17.9% |
| `music6` | 7.068 / 7.001 | 8.052 / 7.950 | 30.4% / 20.9% |

Files are 0.6–1.3% smaller, most on the harder new set and at low bitrates; the other experts had already
captured much of what the network alone gained. Speed is unchanged (`music12`: 53–58× real time against 46–49×,
measured back to back). The table replaced the first network's; picotone ships only its newest tables.

The shipped tables are `cx2_encodec_48khz_net2_v5.ecxm` (hybrid with `net48v2`, the 48 kHz default),
`cx2_encodec_48khz_v5.ecxm` (constructed only) and `cx2_encodec_24khz_v5.ecxm`. A stream stores its table's
hash (FNV-1a 64 of the file), so shipped tables must never change in place.

### 6.4 Speed engineering

- **Network kernel:** an explicit pair-interleaved 16-bit multiply-add (`pmaddwd`, through pulp's safe AVX2
  wrappers, with a scalar fallback). It cut network time from 13.8 to 10.8 µs per code, with byte-identical
  output on every dispatch path.
- **Int8 was evaluated and rejected:**
  - int8 weights with today's activations cost about +0.5% in size, and were not clearly faster;
  - true int8 × int8, which is what VNNI and WebGPU's `dot4I8Packed` need, cost 7–20%, because narrow
    activations do the damage;
  - VNNI cannot be reached without `unsafe` in the current pulp.
- **Browser exactness:**
  - WebAssembly must use the standard `i32x4.dot_i16x8_s`. Relaxed-SIMD dot products are
    implementation-defined, and would break decoding across machines.
  - WGSL integer arithmetic is exact.
- **GPU page-parallel decoding was rejected:**
  - Inside a page, every code depends on the previous one, so a 75 s file is 2,250 sequential rounds.
  - GPU readback per round would be slower than the CPU.
  - Avoiding readback means porting the whole range decoder bit-exactly to WGSL.
  - Page-parallel Web Workers get most of the benefit simply.

### 6.5 Robustness of the fit

- **Plain fit encodes:** tables fitted on plain (greedy, stock-stride) encodes lose only 0.18–0.29% on preset
  files, so the fit set was not re-encoded.
- **Beam 8:** beam-8 against beam-1 files on matched fits ranges from −0.71% (`music24`) to +0.74% (`music6`,
  one set).
- **The network on beam-8 files:** it does slightly better there than on greedy encodes of the same clips,
  although it was trained on greedy codes.

## 7. The `.pico` format

The coder's output is real Ogg (RFC 3533), and follows Ogg Opus (RFC 7845) where it can. The full specification
is [FORMAT.md](FORMAT.md). In short:

- **Headers:** a `Pico` identification header and, only when there are tags, a `PicoTags` Vorbis-style comment
  header (`TITLE`, `ARTIST`, `LICENSE`, `SOURCE`, `LOOPSTART`, `LOOPLENGTH`). The ID header is versioned and
  lean: enum bytes for the model and scheme, the table hash, the codebook count, a flag byte (loop, comments,
  length, edge cross-fade), and varints for the pre-skip, the stride, the page length and, optionally, the
  playable length and end trim.
- **Extension room:** TLV fields after the fixed part, with odd types critical. Types are reserved for a
  per-segment codebook schedule, a longer predictor-reset interval, a stride schedule and enhancement layers
  (`.picol` sidecars that add codebooks to a base file; not implemented).
- **Self-contained pages:** each audio page is one chunk: the page's 12-bit scales, then its codes, coded with the
  predictor reset and the range coder restarted. A page therefore decodes alone; it is both the seek unit and
  the most a damaged page can cost. The page layout is derived from the header, not stored.
- **Page length** trades size against seek granularity. It is 1 segment at `music24`, 2 at `music12` and 4 at
  `music6`: lower bitrates lose proportionally more to cold starts and headers. On the `music6` test set,
  1-segment pages cost 4.5% against a single page, 4-segment pages 0.5% (`tools/reset_cost.py`).
- **Granule positions** count final decoded samples, and the last page's marks the trimmed end. Bisecting on
  them seeks as in any Ogg file; the WebAssembly decoder seeks bit-exactly.
- **Gapless and loops:** the pre-skip, the end trim (through the final granule position) and a loop flag.
- **Streaming writes:** without the length flag, the end trim moves to the last packet, so a stream can be
  written before its length is known.
- **Damage:** every page has a CRC-32 and a sequence number. A bad page loses only its own chunk, and the reader
  resynchronises on the next `OggS`.

The first container (`EncxHead`, written by the separate `ecdz` tool from `.ecdc` files) had fixed-width
fields, names as strings, 32-bit float scales, an always-present comment header and a per-packet header; it
round-tripped `.ecdc` byte for byte. That compatibility goal was dropped in the merge, which allowed the leaner
layout above.

A decoder must refuse a stream whose table hash differs from its own. With the wrong table, the decode
produces garbage, not an error.

## 8. Browser decoders

### 8.1 WebGPU (`web/`)

A TypeScript and WGSL port of the Rust decoder, with f32 or f16 weights:

| Model | f32 | f16 |
|---|---|---|
| `encodec_24khz` | 46.5 MB | 23.2 MB |
| `encodec_48khz` | 38.1 MB | 19.0 MB |

f16 is 59–69 dB SNR against the Rust decoder, and runs the LSTM 1.4–1.7× faster. f32 is about 120 dB.

- **Batching:** 48 kHz files decode in batches of up to 4 segments, the first batch a single segment for a
  quick start. Overlap-add runs on the GPU.
- **24 kHz:** files decode as one causal stream, 5 s at a time.
- **Memory:** GPU memory is fixed per chunk: about 160 MB for 48 kHz.
- **Speed:** 10–21× real time at 48 kHz, 23–33× at 24 kHz (f32 weights).
- **cx2:** the range decode runs in WebAssembly (the same Rust code as the CLI, so the codes are bit-exact), one
  Ogg page at a time, and the codes are fed to the GPU. Porting the range coder to WGSL was rejected (§6.4).
  Decoding page by page instead of the whole file first cut the time to first audio of a 75 s stereo file from
  3.3 s to 0.5 s.

### 8.2 WebAssembly (`wasm/`)

The Rust library compiled to wasm32 with SIMD. It comes in two builds:

- **Threaded:** a nightly build with shared memory and a rayon pool. It needs cross-origin isolation.
- **Single-threaded:** used everywhere else.

The worker picks the build at runtime from `crossOriginIsolated`.

Streaming decoding cut the time to first audio for a 75 s mono file:

| Build | Before | After |
|---|---|---|
| Threaded | 13.6 s | 1.6 s |
| Single-threaded | 48 s | 2.1 s |

For a 75 s 48 kHz stereo file in headless Chrome, before the merge (`.ecdc`, no entropy decode):

| Build | Decode time | Real-time factor |
|---|---|---|
| Threaded (16) | 4.06 s | about 18× |
| Single-threaded | 13.7 s | about 5.5× |
| Single-threaded with relaxed SIMD | 11.5 s | about 6.5× |

Before the decode-speed pass these took 8.28 s and 25–27 s. Earlier, slower figures were measured on a heavily
loaded machine.

After the merge the decoder reads `.pico`, so these times include the cx2 decode (75 s at `-b 6` / 120 s at
`music12`): threaded with 8 threads 15.8× / 12×, single-threaded 5.6× / 5.1×, and 6.8× / 6.0× with relaxed
SIMD. First audio arrives after 0.1–0.3 s. Decoding from a later sample (a seek) is bit-exact with the same
samples of a whole decode.

The relaxed-SIMD build (`pkg-st-fma`) is loaded only when a `WebAssembly.validate` probe finds relaxed SIMD.
Relaxed SIMD is implementation-defined, so it is acceptable for the float neural decoder, which needs only about
120 dB accuracy. It is never used for the cx2 decoder, which has to be bit-exact: that code is integer-only,
the relaxed-SIMD feature switches only the float kernels, and pulp's dispatch takes its portable path on
WebAssembly.

### 8.3 Secure contexts

Both WebGPU and threaded WebAssembly require a secure context: HTTPS or `localhost`. A LAN device reaching the
game over plain HTTP gets neither. It falls back to single-threaded WebAssembly, which still decodes 48 kHz
stereo at about 5–7× real time. HTTPS with a locally trusted certificate (an mkcert-style local CA) unlocks the
faster paths.

Both decoders are tested against the Rust decoder: SNR of at least 100 dB for WebAssembly, and a parity suite for
WebGPU. They clamp to ±0.99 before comparing, because the CLI's WAV writer limits its output.

## 9. Direct weight programming

The question: can a language model read EnCodec's weights and recorded activations, work out what the parts
compute, and write better weights without gradient descent? The method was inspection and closed-form least
squares on held-out audio, against two bars:

- **strict:** at least 99% identical codes, and at least 40 dB against stock;
- **perceptual:** against the source audio, log-mel +0.05 dB or less, MR-STFT +0.01 or less, SNR −0.25 dB or
  more.

Full results are in `dwp/README.md`.

### 9.1 What the parts do

- **The LSTM blocks:** the skip carries the signal, and the LSTM adds a correction that later layers amplify.
  - Much of that correction is a constant: replacing the LSTM by its mean output recovers most of the damage of
    deleting it.
  - Units fall into memoryless gated feed-forward units, long integrators, and data-dependent units.
  - The integrators include "clocks", whose cell value tracks the frame index (correlation 0.9–0.98), and
    loudness accumulators. At 48 kHz the clocks encode position within the segment.
  - Gates are driven mostly by the input, not the recurrence.
  - Nothing is low-dimensional: 99% of the variance needs 310–440 of 512 dimensions.
- **Convolutions:** no residual block is removable. Bottlenecks have about 25% slack, worth only 1–2% of the
  multiply-adds.
- **Codebooks:** they are well used, at 8.4–9.1 bits of entropy per 10-bit code on real audio. Deeper ones hold
  near-duplicates and dead entries. That headroom is what cx2 exploits.

### 9.2 What can be changed

- **Strict bar:** nothing approximate passes. The quantiser turns relative errors of 1e-3 into different codes.
  A rewritten encoder LSTM would need 60–70 dB accuracy, and errors a thousand times larger are inaudible to
  every metric. Exact engineering (§2) is the only way to speed things up at this bar.
- **Perceptual bar:** one candidate came close, a 384-unit LSTM (from 512). Units are kept by how unpredictable
  they are from the others; the dropped units are solved as a linear function of the kept ones and folded into
  the weights.
  - It passed on the 12-clip test split.
  - It missed on a second listening set: +0.09 dB log-mel for the 48 kHz decoder, +0.20 on the worst clip. The
    24 kHz model failed clearly, at +0.31.
  - It would speed up 48 kHz decode only about 1.06–1.15×, because the LSTM is just 15–20% of that time. It is
    not built.
  - A blind ABX set is in `dwp/out/listen_lstm/abx/`, so it can be judged by ear.
- **Compression ideas that failed:**
  - re-estimating the codebooks;
  - predicting the missing residual (it improves latent SNR, but makes the audio duller);
  - smaller codebooks with more stages.
- **Compression idea that worked:** beam-search RVQ (§4), which is not a weight edit.

### 9.3 Verdict

Interpretation worked: it explained the seam, located decode time, and identified codebook redundancy. Writing
weights did not beat exact engineering. The project's real gains came from format changes (overlap and loops),
search (beam) and entropy coding (cx2).

## 10. Limitations and future work

- **Bitrate gap:** cx2 still trails Meta's LM by about 0.5–0.9 bit per code. More data has stopped helping
  (§6.3, `net48v2`), so the remaining gap is more likely context: the network sees only four frames, and Meta's
  LM is a transformer over seconds. A recurrent trunk is being tested (`train/train_gru.py`). A design review
  proposed further ideas, from a slimmer container (the merge
  took 0.3–0.8% of it) to encoder-side rate-distortion optimisation against cx2 (an estimated 5–12%).
- **Enhancement layers:** a `.picol` sidecar holding the upper codebooks of the same encode, so a small base file
  plays alone and base plus sidecar plays at a higher preset. The format reserves room for it (FORMAT.md); the
  base would stay coded exactly as today.
- **WebGPU seeking:** only the WebAssembly decoder seeks so far.
- **Decode speed:** GEMM is now 70% of decode time, near the kernel's in-cache speed. AVX-512 was not tried;
  it needs pulp's `x86-v4` feature and a newer minimum Rust. The pruned LSTM is not worth building (about 1.07×).
- **Thread scaling:** 16 threads give only about 2.6× over one thread on this laptop.
- **No listening tests:** all quality figures are metrics against the source. Listening was informal.

## 11. Reproducing

| What | Command |
|---|---|
| Build and test | `cargo build --release`, `cargo test --release` |
| Parity with the reference | `uv run --project tools --extra parity tools/parity.py`: `encode --beam 1`, `dev codes`, compared with `transformers.EncodecModel` |
| Seam measurements | `dwp/s11_seam.py` |
| Beam search | `dwp/s09_beam_rvq.py` |
| Browser decoders | `web`: `npm run test:parity`; `wasm`: `node test/run.mjs` |
| cx2 and `.pico` | `cargo test --release`; tables with `tools/build_tables.py` and `tools/check_testset.py`; training in `train/` (uv with `pyproject.toml`) |

## References

- A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio Compression.* 2022. arXiv:2210.13438.
- RFC 3533, *The Ogg Encapsulation Format.* RFC 7845, *Ogg Encapsulation for the Opus Audio Codec.*
- Hugging Face `transformers`, `EncodecModel`, the reference used for parity.
