# picotone: neural audio codecs for game audio in the browser

*White paper, November 2026. It describes picotone as it is; the file format is specified in
[FORMAT.md](FORMAT.md).*

## Abstract

picotone is a neural audio codec in safe Rust, written for one consumer: a local web game that streams
1–1.5 minute stereo music loops and short sound effects to a browser. It has two codecs:

- **HQ**, the default: 44.1 kHz audio coded as DAC 44 kHz codes, made by picotone's STFT encoder and decoded
  by picotone's Vocos decoder, at 1 to 16 codebooks per channel (about 0.6 to 10.6 kbps per channel after
  entropy coding), in mono, left/right, mid/side or parametric stereo. A file can be split into a base that
  plays alone and an enhancement layer (`.picol`) that brings it up to more codebooks.
- **EnCodec**, the fast path (`--fast`): Meta's codec at 48 kHz stereo and 24 kHz mono, which encodes much
  faster than HQ, bit-exact with the reference implementation, with presets from about 3 to 18 kbps.

Both store their codes in **`.pico`** files: Ogg pages that each decode on their own, coded losslessly with
**cx2**, an integer-only predictive model and range coder that gives the same bits on every CPU, in
WebAssembly and on a GPU. Decoders run natively, in WebAssembly (both codecs) and on WebGPU (EnCodec).

The EnCodec path has three changes to the stock codec:

1. **A longer segment overlap** (stride 44800 instead of 47520), which removes stock EnCodec's audible
   once-per-second seam.
2. **Beam-search quantisation**, +0.16 to +0.27 dB SNR at no decode cost.
3. **Gapless loops**: wrap-around padding that the decoder trims away.

## 1. Background

The consumer's needs shape the design:

- music at 44.1–48 kHz stereo, at a few kbps per channel;
- playback in a browser, sometimes on a LAN device reached over plain HTTP, which is not a secure context;
- loops that join without a click;
- small files that can start playing before the rest has loaded.

**EnCodec** (Défossez et al., 2022) is a convolutional encoder and decoder with LSTMs, joined by a residual
vector quantiser (RVQ) of up to 32 codebooks of 1024 entries each.

| Model | Audio | Frames | Codebooks | Kbps per codebook | Segmentation |
|---|---|---|---|---|---|
| `encodec_24khz` | 24 kHz mono, causal | 75 per second | up to 32 | 0.75 | none (one stream) |
| `encodec_48khz` | 48 kHz stereo, non-causal | 150 per second | up to 16 | 1.5 | 1 s segments, triangle cross-fade over the overlap |

The 48 kHz model normalises each segment by its RMS and stores that scale.

**DAC** (Descript Audio Codec) 44 kHz codes one channel at 86 frames per second (512 samples per frame) with an
RVQ of 9 codebooks of 1024 entries. picotone keeps DAC's code space and quantiser but uses its own encoder and
decoder, and extends the quantiser to 16 codebooks (§2).

picotone uses its own container, `.pico` (§7), not Meta's `.ecdc`.

## 2. The HQ codec

### 2.1 Pipeline

- **Encoder:** picotone's STFT encoder (`dac44_stftenc`) maps 44.1 kHz audio to DAC's latent.
- **Quantiser:** DAC's own residual quantiser codes the latent: its 9 codebooks, then 7 more that picotone
  trained on DAC's residual, 16 in all (`dac44_quantizer`). A lower codebook count is a prefix of the full code,
  so one encode serves every rate.
- **Decoder:** a Vocos (iSTFT) decoder trained on those codes (`dac44_vocos`) decodes every codebook count.

A file names its code space by a hash of the codebooks, not a decoder, so any decoder of these codes can play
it. Each coded channel is coded separately.

### 2.2 Rates

After cx2 (§6.3), on the 23 test clips, kbps per coded channel, headers and Ogg framing included:

| Codebooks | 1 | 3 | 6 | 9 (default) | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kbps | 0.60 | 1.84 | 3.94 | 6.09 | 6.56 | 7.32 | 8.06 | 8.38 | 9.13 | 9.87 | 10.57 |

Before entropy coding each codebook is 0.86 kbps per channel.

### 2.3 Stereo

A stereo input is coded as left/right or mid/side, whichever scores better on log-mel distance over up to four
loud 3 s excerpts (or as `--stereo` says). Mid/side can give the side fewer codebooks than the mid. At 2
codebooks per channel or fewer the encoder also tries **parametric stereo**: the mid alone, at about twice the
codebooks, plus per-band level and correlation parameters per page (about 1 kbps), from which the decoder
rebuilds the side with a decorrelated copy of the mid. A stereo input whose side stays more than 40 dB under its
mid is coded as mono.

### 2.4 Enhancement layers (`.picol`)

`picotone encode --codebooks B --layer-codebooks N` writes a base `.pico` with codebooks `0..B`, which plays
alone, and a `.picol` layer with codebooks `B..N` of the same encode. Base plus layer decodes sample for sample
as a one-file encode at N codebooks. The layer's codes are coded with a predictor of all N codebooks, fed the
base's codes without coding them. Over a parametric base, the layer codes mid/side and carries the whole side.

Cost of base plus layer against one file at 16 codebooks, on the 23 test clips (`picotone dev layer-cost`):

| Base | Base kbps | Extra size |
|---|---|---|
| q3 | 3.49 | +2.8% |
| q5 | 5.88 | +2.7% |
| q9 | 10.90 | +1.5% |
| q12 | 14.34 | +1.1% |
| parametric (q1–q3 mid) | 1.9–3.2 | +5.5–5.8% |

(Base kbps here is per file over all channels.) About 0.6 points of the extra is the layer's framing; the rest
is the base's predictor, which sees only the base's codebooks. Over a parametric base, the stereo parameters go
unused once the layer is there.

A layer is refused as a whole when it is not the base's or any page is damaged or missing; the base then plays
alone. A player upgrades a playing file by restarting at the current position with the layer.

### 2.5 Speed

The WebAssembly decoder runs HQ at about 4.4–4.7× real time for single-threaded stereo, in about 77 MiB of
memory.

## 3. The EnCodec implementation

### 3.1 Exactness

In greedy mode (`--beam 1`) the encoder reproduces the Hugging Face `transformers` reference exactly:

- 100% of codes agree on both models;
- per-segment scales agree exactly;
- decoder output is within 117 dB (24 kHz) and 120 dB (48 kHz) SNR of the reference decoder fed the same
  codes.

The parity check (`tools/parity.py`) encodes synthetic audio with both and compares them. It is a test only.

### 3.2 Speed and memory

The code is safe Rust: `unsafe` is forbidden, and clippy's `pedantic`, `nursery` and `cargo` groups are denied.

- **GEMM:** a packed-panel kernel in safe `pulp` SIMD, AVX2+FMA natively and SIMD128 in wasm (with a
  register-light shape there, so V8 does not spill), with weights packed once at load.
- **Convolutions:** GEMMs over an overlapping strided im2col view with virtual padding, so no padded copy is
  made. Transposed convolutions run in polyphase form.
- **Fused passes:** ELU (a vector `expm1`, about 1 ulp), GroupNorm's affine and the residual sum are applied as
  the next convolution reads each tile. GroupNorm statistics are computed in short f32 blocks and summed in f64.
- **LSTM:** the input projection is batched over time; each step is one batched GEMM with vector sigmoid and
  tanh, and the segments of a chunk are batched together.
- **Memory:** a buffer pool; only what a command needs is loaded (encoder, decoder, the codebooks a file
  uses); each stage releases its copy of the audio as soon as the next exists.
- **Parallelism:** rayon, over segments and over tiles. Output is bit-identical across thread counts.

| 120 s file, one thread | Decode |
|---|---|
| 48 kHz `music24` | 10.9 CPU-s (11× real time) |
| 24 kHz | 6.55 CPU-s (18× real time) |

On one thread at 48 kHz, decode time splits into convolution GEMMs 7.37 s, LSTM 1.88 s, and ELU and GroupNorm
1.13 s. GEMM runs near the kernel's in-cache speed.

### 3.3 Streaming APIs

- `stream_decoder()` decodes the causal 24 kHz model as it goes. It carries convolution history and LSTM
  state across pushes, buffers 7 frames of look-ahead at the start, and holds back the last 320 samples until
  `finish`.
- `decode_segment()` with `overlap_add()` does the same for the 48 kHz model. It emits only samples that no
  later segment can change.

Both are bit-identical to a whole-file decode, and their memory depends on the chunk size, not the file length.

## 4. The segment overlap (EnCodec 48 kHz)

Stock EnCodec at 48 kHz has an audible pulse about once a second, worst at low bitrates. Averaging the error
envelope over segment phase shows its cause: every 1 s segment has a damaged start (about 30 ms) and a damaged
end (about 10 ms), from reflect padding in the non-causal convolutions and the LSTMs starting at zero state.
The stock overlap is only 10 ms, so it cross-fades two damaged pieces into each other, a 1.4–2.8 dB dip in
waveform match at every seam. Two segments rendered independently do not line up sample for sample, so the
cross-fade has to run over undamaged audio.

picotone's presets use a stride of **44800** samples (exactly 140 latent frames, so segment starts fall on frame
boundaries), an overlap of 3200 samples. Two stream fields carry it: the stride, and the cross-fade rule
(EnCodec's triangle, or `--crossfade edge`, which skips the damaged edges and matters only for short overlaps).
With `ov = 48000 − stride`:

```
hi = ov − 480 if ov > 960 else ov
lo = max(0, min(1440, hi − 240))
out = old before lo, a linear ramp from lo to hi, new from hi on
```

At 5% overlap or more the triangle does as well, because it gives the damaged edges almost no weight.

| Preset | Codebooks | Bitrate before cx2 | Cross-fade | Beam |
|---|---|---|---|---|
| `music24` | 15 | 24.12 kbps | triangle | 8 |
| `music12` | 8 | 12.87 kbps | triangle | 8 |
| `music6` | 4 | 6.44 kbps | triangle | 8 |

The bitrate is `nc × 1500 × 48000 / stride` plus 12 bits of scale per segment (about 13 bit/s). At 24 kbps
the 16th codebook pays for the overlap.

Seam dip (median quality over the middle of the segment minus the worst 448-sample bins at its edges) on two
120 s listening clips, and the whole `.pico` file sizes:

| Preset | Dip, orchestral | Dip, ambient-electronic | File, orchestral | File, ambient-electronic |
|---|---|---|---|---|
| `music24` | −0.31 dB | +0.08 dB | 17.51 kbps | 17.66 kbps |
| `music12` | −0.14 dB | +0.09 dB | 8.30 kbps | 7.79 kbps |
| `music6` | −0.25 dB | −0.02 dB | 4.02 kbps | 3.22 kbps |

No seam is measurable.

## 5. Beam-search quantisation (EnCodec)

Greedy RVQ picks the nearest entry at each stage. Beam search keeps the N best partial sums and picks the best
full path at the end. The decoder and the bitstream do not change.

| Bitrate | SNR gain | Log-mel change |
|---|---|---|
| 6 kbps | +0.21 to +0.23 dB | −0.05 dB |
| 12 kbps | +0.27 dB | |
| 24 kbps | +0.16 dB | −0.03 dB |

That is worth roughly 8–10% of bitrate. Beam 8, the default, costs about 2% more encode time at 6 kbps and 33%
at 24 kbps; `--beam 1` gives the reference's greedy codes. Beam-8 files code to within ±0.7% of the size of
beam-1 files with cx2.

## 6. cx2: lossless entropy coding

### 6.1 Approach

At 10 bits per code, codes are far from random: each codebook's usage is skewed, codes repeat in time, and each
RVQ stage depends on the ones before. A large language model can exploit this, but too slowly for browser
playback, since it must run on both encode and decode. cx2 aims for most of that saving at tens of times real
time on one thread.

Everything that touches probabilities is integer arithmetic, including the networks (`i16` weights, `i32`
accumulation), so coding is bit-identical on any CPU, in WebAssembly and on a GPU. In WebAssembly the integer
dot product is the standard `i32x4.dot_i16x8_s`, never a relaxed-SIMD one, whose results are
implementation-defined. A range coder turns the probabilities into bits. Each page is coded with the predictor
reset and the range coder restarted, so it decodes alone; the decode is sequential within a page, and runs on
the CPU (in WebAssembly in the browser).

A stream stores its table's hash (FNV-1a 64 of the file); a decoder needs that exact table, so shipped tables
never change in place. When a table is replaced, files are re-encoded.

### 6.2 EnCodec tables

A product of experts over the 1024 symbols of each code, combined in the log domain with fitted weights:

| Expert | What it uses |
|---|---|
| prior | per-codebook usage |
| geo | the RVQ geometry: how near each centroid is to the current partial reconstruction |
| rec | recency of codes |
| pair | the same codebook in the previous frame |
| stat | adaptive statistics |
| hist | a low-rank (SVD) model of pointwise mutual information |
| net | a gradient-trained MLP over four frames of context (48 kHz only) |

The `net48v2` network has about 1.8 M parameters: two hidden layers of 384 and 256 units with low-rank output
heads (rank 64 and 32), an embedding for 15 codebooks, and a position-in-segment input. It was trained on the
full fit corpus: 1,118 clips, 24.3 hours, all public domain or CC0 and listed in `DATA_SOURCES.json`, spanning
game music and effects, rock, pop, electronic, hip-hop, jazz, ambient, folk, world, speech and vocals. The
shipped 48 kHz hybrid keeps only net, rec and pair beside the prior; the network roughly doubles the saving of
the constructed experts alone.

| Table | Use |
|---|---|
| `cx2_encodec_48khz_net2_v5.ecxm` | 48 kHz default (hybrid with `net48v2`) |
| `cx2_encodec_48khz_v5.ecxm` | 48 kHz, constructed experts only |
| `cx2_encodec_24khz_v5.ecxm` | 24 kHz |

The tables hold the codebook centroids, so a decoder needs only the table and the decoder half of the weights.

Results on held-out clips by creators not in the fit set, two test sets (23 clips / a harder, more varied 30
clips with pop and vocals), bits per code including every header and page, against about 10.0 for plainly
packed codes:

| Preset (page length) | Bits per code | Saved vs plain | Meta's LM, payload only |
|---|---|---|---|
| `music24` (1 s) | 7.998 / 8.635 | 20.1% / 13.8% | 25.7% / 20.8% |
| `music12` (2 s) | 7.388 / 8.231 | 26.3% / 17.9% | 32.2% / 25.5% |
| `music6` (4 s) | 7.001 / 7.950 | 30.4% / 20.9% | 37.6% / 30.0% |

Meta's LM decodes at 0.2–0.3× real time; the hybrid at about 55× (`music12`) on one thread.

### 6.3 The HQ table

`models/cx2_dac44.ecxm` is one integer network and no other experts. It predicts each code from the channel's
previous 8 frames in the same page and the codebooks already coded in this frame, with a bias by the channel's
role (left/right, mid, or side). It is trained on codes dumped at 16 codebooks from the fit corpus, mixing
codebook counts per channel, so one table serves every count. On the 23 test clips it codes 7.854 bits per code
at 9 codebooks and 7.125 at 3 (rates in §2.2).

## 7. The `.pico` format

The files are real Ogg (RFC 3533) and follow Ogg Opus (RFC 7845) where they can. The full specification is
[FORMAT.md](FORMAT.md). In short:

- **Headers:** a `Pico` identification header and, only when there are tags, a `PicoTags` Vorbis-style comment
  header (`TITLE`, `ARTIST`, `LICENSE`, `SOURCE`, `LOOPSTART`, `LOOPLENGTH`). The ID header is versioned and
  lean: enum bytes for the model and scheme, the table hash, the codebook count (per coded channel for HQ, with
  the stereo layout), a flag byte, and varints for the pre-skip, the page length and, optionally, the playable
  length and end trim (EnCodec streams also store the stride).
- **Extensions:** TLV fields after the fixed part, with odd types critical. Type 7 marks an enhancement layer
  (`.picol`, HQ only; §2.4).
- **Self-contained pages:** each audio page is one chunk per coded channel (EnCodec: the page's 12-bit scales,
  then its codes), coded with the predictor reset and the range coder restarted. A page is both the seek unit and
  the most a damaged page can cost. The page layout is derived from the header, not stored.
- **Page length** trades size against seek granularity. EnCodec uses 1 segment at `music24`, 2 at `music12` and
  4 at `music6`, since lower bitrates lose proportionally more to cold starts (on the `music6` test set,
  1-segment pages cost 4.5% against a single page, 4-segment pages 0.5%). HQ uses 80 frames, about 0.93 s.
- **Granule positions** count final decoded samples, and the last page's marks the trimmed end. Bisecting on
  them seeks as in any Ogg file; the WebAssembly decoder seeks bit-exactly.
- **Gapless and loops:** the pre-skip, the end trim (through the final granule position) and a loop flag.
- **Streaming writes:** without the length flag, the end trim moves to the last packet, so a stream can be
  written before its length is known.
- **Damage:** every page has a CRC-32 and a sequence number. A bad page loses only its own chunk, and the reader
  resynchronises on the next `OggS`.

A decoder must refuse a stream whose table hash differs from its table's: with the wrong table, the decode
produces garbage, not an error.

## 8. Gapless loops

EnCodec's first and last few milliseconds are not faithful, so a decoded loop would click where it wraps. With
`--loop`, the encoder:

1. prepends the last 150 ms of the clip and appends its first 50 ms, then codes the result;
2. records the pre-skip, the end trim and the loop flag.

The decoder drops the pre-skip and the end trim, so the output length equals the input's, and cross-fades the
coded pre-roll, a faithful rendering of the loop's own end, into the last 100 ms of the output. The last sample
therefore runs straight into the first: the sample jump at the wrap is 0.008 on an ambient clip at 6 kbps (0.094
without `--loop`).

## 9. Browser decoders

Both decoders take the same `.pico` files, stream audio as it decodes, honour loops and trims, and are written in
strict TypeScript. The cx2 decode is the same Rust code in both, compiled to WebAssembly, so the codes are
bit-exact with the command line. Both are tested against the Rust decoder.

### 9.1 WebAssembly (`wasm/`)

The Rust library compiled to wasm32 with SIMD, in a Web Worker. It plays HQ and EnCodec files, seeks, and takes
an HQ file's `.picol` layer. Two builds:

- **Threaded:** shared memory and a rayon pool; needs cross-origin isolation.
- **Single-threaded:** used everywhere else.

The worker picks the build at runtime from `crossOriginIsolated`. A relaxed-SIMD single-threaded build is loaded
when a `WebAssembly.validate` probe finds relaxed SIMD; it changes only the float kernels of the neural decoder,
never the integer cx2 decode.

| EnCodec 48 kHz stereo (75 s at `-b 6` / 120 s at `music12`) | Real time |
|---|---|
| Threaded, 8 threads | 15.8× / 12× |
| Single-threaded | 5.6× / 5.1× |
| Single-threaded, relaxed SIMD | 6.8× / 6.0× |

First audio arrives after 0.1–0.3 s. HQ decodes at about 4.4–4.7× real time single-threaded (§2.5). A decode
from a later sample (a seek) equals the same samples of a whole decode.

### 9.2 WebGPU (`web/`)

A TypeScript and WGSL port of the EnCodec decoder, with f32 (about 120 dB SNR against the Rust decoder) or f16
weights (59–69 dB, LSTM 1.4–1.7× faster). It does not play HQ files yet.

| Model | f32 | f16 |
|---|---|---|
| `encodec_24khz` | 46.5 MB | 23.2 MB |
| `encodec_48khz` | 38.1 MB | 19.0 MB |

48 kHz files decode in batches of up to 4 segments (the first a single segment, for a quick start), with
overlap-add on the GPU; 24 kHz files decode as one causal stream, 5 s at a time. GPU memory is fixed per chunk
(about 160 MB at 48 kHz). Speed: 10–21× real time at 48 kHz, 23–33× at 24 kHz (f32). The range decode runs in
WebAssembly a page at a time, feeding the GPU; first audio arrives after about 0.5 s.

### 9.3 Secure contexts

WebGPU and threaded WebAssembly require a secure context: HTTPS or `localhost`. A LAN device reaching the game
over plain HTTP gets single-threaded WebAssembly, which still decodes faster than real time. HTTPS with a locally
trusted certificate unlocks the faster paths.

## 10. Limitations and future work

- **EnCodec bitrate gap:** cx2 trails Meta's LM by about 0.5–0.9 bit per code; the network sees only four
  frames, while the LM is a transformer over seconds.
- **Layers:** `.picol` exists for HQ files only, and a playing file picks up its layer by restarting at its
  position, not inside one decode.
- **WebGPU:** no HQ decoding and no seeking yet.
- **Thread scaling:** 16 threads give only about 2.6× over one thread on the test laptop.
- **Listening:** quality decisions are made by ear, informally; the figures here are metrics against the
  source.

## 11. Reproducing

| What | Command |
|---|---|
| Build and test | `cargo build --release`, `cargo test --release` |
| EnCodec parity with the reference | `uv run --project tools --extra parity tools/parity.py` |
| Seam measurements | `dwp/s11_seam.py` |
| Beam search | `dwp/s09_beam_rvq.py` |
| EnCodec cx2 tables | `tools/build_tables.py`, `tools/check_testset.py`, `picotone dev check`; network training in `train/` |
| HQ table | `tools/dac_codes.py` (dump codes), `train/hq_net.py` (train and export); `picotone dev dac-pack` (rates) |
| HQ layer cost | `picotone dev layer-cost` |
| Browser decoders | `web`: `npm run test:parity`; `wasm`: `node test/run.mjs` |

Python tooling uses uv with `pyproject.toml`.

## References

- A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio Compression.* 2022. arXiv:2210.13438.
- R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, K. Kumar. *High-Fidelity Audio Compression with Improved RVQGAN*
  (DAC). 2023. arXiv:2306.06546.
- H. Siuzdak. *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality
  audio synthesis.* 2023. arXiv:2306.00814.
- RFC 3533, *The Ogg Encapsulation Format.* RFC 7845, *Ogg Encapsulation for the Opus Audio Codec.*
- Hugging Face `transformers`, `EncodecModel`, the reference used for parity.
