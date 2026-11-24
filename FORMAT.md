# The `.pico` format

How picotone stores codec codes: EnCodec's (models 0 and 1, below) and the HQ codec's (model 2, in
[HQ streams](#hq-streams-model-2)). A `.pico` file is plain Ogg (RFC 3533) with one logical stream, and its
mapping follows Ogg Opus (RFC 7845) wherever it can: pre-skip, granule positions, an end trim through the final
granule position, and Vorbis-style comments. The implementation is `src/pico.rs`; `tests/pico.rs` covers round
trips, seeking, loops, streams without a length, damage and an independent Ogg reader (the `ogg` crate, and
`ffprobe` when it is installed). General-purpose tools recognise the file as Ogg but not the codec inside.

This document describes **version 1**. There is no compatibility with earlier picotone or encodecx files: a
decoder refuses any other version.

All fixed-width integers are little-endian. A **varint** is unsigned LEB128: 7 bits per byte, low bits first,
the top bit set on every byte but the last, at most 64 bits. "Frame" means one latent frame: 320 samples per
channel, carrying one code per codebook.

## Stream layout

| page | packet | contents |
|---|---|---|
| 0 (begin-of-stream flag) | `Pico` | identification header |
| 1, only when the comments flag is set | `PicoTags` | vendor string and `KEY=value` comments |
| then, one per page | audio packet | one independently decodable cx2 chunk |

Each packet starts its own page. A packet over 65,025 bytes continues on following pages with the "continued"
flag, as Ogg prescribes; at the shipped page lengths that does not happen. The last page carries the
end-of-stream flag. Header pages have granule position 0.

## Identification header (`Pico`)

| size | field | meaning |
|---|---|---|
| 4 | magic | `Pico` |
| 1 | version | 1 |
| varint | fixed length | bytes in the fixed part that follows |
| | **fixed part** | |
| 1 | model | 0 = `encodec_24khz` (24 kHz mono), 1 = `encodec_48khz` (48 kHz stereo), 2 = HQ (its own fixed part, [below](#hq-streams-model-2)) |
| 1 | scheme | 2 = `cx2` |
| 8 | table hash | the cx2 table (`.ecxm`) the stream was coded with: FNV-1a 64 of the table file |
| 1 | codebooks | codes per frame (1 to 64) |
| 1 | flags | bit 0: loop; bit 1: a comment header follows; bit 2: the length is stored; bit 3: edge-avoiding cross-fade |
| varint | pre-skip | decoded samples a player drops at the start |
| varint | stride | samples between segment starts; **48 kHz model only** (absent for 24 kHz) |
| varint | page length | segments per page (48 kHz) or frames per page (24 kHz); 0 = the stream is one page |
| varint | playable length | samples after trimming; only with flag bit 2 |
| varint | end trim | decoded samples after the end that are padding; only with flag bit 2 |
| | **extensions** | |
| ... | TLVs | to the end of the packet, see below |

A decoder must refuse a stream whose version, model, scheme or flags it does not know, or whose table hash
differs from every table it holds: decoding with another table produces garbage, not an error. A reader skips
fixed-part bytes beyond the fields it knows (later minor revisions may append fields), so the fixed length is
always honoured.

The 48 kHz model codes 1 s segments (48000 samples) `stride` apart, cross-faded over the overlap: the triangle
rule of EnCodec, or with flag bit 3 a linear ramp that avoids the damaged first 30 ms and last 10 ms of each
segment. The encoder's presets use stride **44800** (6.7% overlap). The stride must lie in 24000..=48000. The
24 kHz model codes the whole file as one causal stream.

### Extensions (TLV)

After the fixed part, until the end of the packet: varint type, varint length, then `length` bytes. **Odd types
are critical**: a decoder that does not know one must refuse the stream. Even types may be skipped.

| type | name | status | body |
|---|---|---|---|
| 1 | codebook schedule | reserved, critical | a codebook count per segment (4 bits each) |
| 2 | output gain | defined, optional | `i16`, dB in Q7.8, applied on playback (0 = none; parsed, not yet applied by picotone's decoders) |
| 3 | reset interval | reserved, critical | predictor resets (keyframes) less often than once per page |
| 5 | stride schedule | reserved, critical | a stride per segment |
| 7 | layer | reserved, critical | marks an enhancement layer (`.picol`); see below |

picotone's decoders refuse every reserved type today. They are listed so that later versions of the encoder can
use them without a new version number.

## Comment header (`PicoTags`)

`PicoTags`, then varint vendor length, vendor string (`picotone <version>`), varint comment count, then per
comment a varint length and a UTF-8 `KEY=value` string. Suggested keys: `TITLE`, `ARTIST`, `LICENSE`, `SOURCE`
(URL), and `LOOPSTART` / `LOOPLENGTH` (in samples, counted after the pre-skip) when the loop is not the whole
file. The header is written only when there are comments.

## Audio packets

| size | field |
|---|---|
| varint | end trim: **only in the last packet, and only when the length is not in the header** |
| rest | the cx2 chunk |

The page's layout is not stored: it follows from the header. For the 48 kHz model a page holds `page length`
whole segments (the last page may hold fewer), and the segment lengths follow from the stride and the total
length. For the 24 kHz model a page holds `page length` frames.

A **cx2 chunk** is one range-coded run, with the predictor fully reset and the range coder restarted, so it
decodes without any other page:

1. For the 48 kHz model, one **12-bit scale** per segment of the page, each coded uniformly over 4096 values.
   The scale is the segment's RMS level on a logarithmic grid with 128 steps per octave, from 2^-28 to just
   under 2^4 (`src/scale.rs`). The encoder quantises it before normalising the segment, so the decoder's
   multiplication undoes exactly what the encoder did. The 24 kHz model has no scales.
2. The **codes**, frame by frame, codebook 0 first within a frame, each coded with the frequencies the cx2
   model predicts from the codes before it in the chunk.

The cx2 coder is integer-only, so every platform (x86 with AVX2, the portable path, WebAssembly) decodes the
same codes.

## Granule position, length and seeking

A page's granule position is the number of decoded samples that are final once that page has been decoded:
`segments so far x stride` for the 48 kHz model, `frames so far x 320` for the 24 kHz one, capped at the end of
the audio. The last page's granule position is where the audio ends: `total decoded samples - end trim`.

The playable length is `final granule - pre-skip`. When the header holds the length (flag bit 2, the default), a
player knows it before reading the stream. A stream written without it (for example while it is being encoded)
stores the end trim at the start of its last packet instead, so it can be written without knowing its length.

To seek to playable sample `s`, find the page whose segments cover sample `s + pre-skip` and decode from there;
bisecting on granule positions works as in any Ogg file. With overlapping segments the start of a page's first
segment is cross-faded with the tail of the previous segment, so a player that needs those samples exactly
starts one segment earlier (picotone's WebAssembly decoder does this, and is bit-exact with a whole-file
decode). The causal 24 kHz model must decode from the start.

## Gapless playback and loops

A player outputs the decoded samples from `pre-skip` up to the final granule position. `--loop` codes the file
with its own end before it and its start after it (150 ms of pre-skip and 50 ms of end trim), so what remains
after trimming joins end to start without a click; the decoder folds the overlap onto the start. When flag bit 0
is set the trimmed audio repeats; `LOOPSTART` / `LOOPLENGTH` narrow the loop to part of it.

## Page length

One page is one chunk, so page length is both the seek granularity and the most audio a damaged page can cost.
Shorter pages cost size: the predictor starts cold on each page, and each page has a 27-byte header plus lacing.
The encoder picks:

| bitrate before cx2 | page length |
|---|---|
| 20 kbps and above (music24) | 1 segment (about 1 s) |
| 10 to 20 kbps (music12) | 2 segments |
| below 10 kbps (music6) | 4 segments |

For the 24 kHz model the same rule applies in seconds (75 frames per second).

## Damage

Every page has the `OggS` capture pattern, a sequence number and a CRC-32 (polynomial `0x04C11DB7`). A page that
fails its CRC or is missing loses exactly its own chunk; the reader resynchronises on the next capture pattern
and later pages decode normally. Damage to a header page makes the stream undecodable. A stream without the
length in its header needs an intact last page; a player that streams can instead stop at the last complete page.

## Limits a reader enforces

- Codebook counts above 64 and lengths above 2^40 samples are rejected, as is a varint longer than 64 bits.
- Decoding a chunk stops with an error once the range decoder has run past the end of the chunk's data, so a
  wrong length cannot produce codes from nothing.
- Lost pages are inferred from gaps in page sequence numbers, but never more of them than the file's size allows
  (27 bytes per page).
- Comment counts and lengths larger than the packet are rejected.

## HQ streams (model 2)

The HQ codec (`dac44-student`, the encoder's default) codes 44.1 kHz audio with DAC 44 kHz's encoder and
residual quantiser: one frame per 512 samples, up to 9 codebooks of 1024 entries per frame, a lower codebook
count being a prefix of the full one. picotone decodes it with a distilled student of DAC's decoder. Ogg
framing, the comment header, extensions (TLVs), pre-skip, granules, damage handling and loops work as for
EnCodec; only the fixed part and the audio packets differ. The implementation is `src/pico/dac.rs`.

### Fixed part

| size | field | meaning |
|---|---|---|
| 1 | model | 2 |
| 1 | scheme | 1 = raw: 10 bits per code, no entropy coding |
| 8 | codebook hash | FNV-1a 64 over DAC's 9 x 1024 x 8 quantiser codebooks (`f32`, little-endian, stage by stage) |
| 1 | layout | 0 = mono, 1 = left/right, 2 = mid/side (M = (L+R)/2, S = (L-R)/2; L = M+S, R = M-S), 3 = parametric stereo (the mid M alone, plus stereo parameters per page) |
| 1 | parametric settings | **only for layout 3**: bits 0-1 the band layout (0 = 20 bands, 1 = 10), bits 2-4 log2 of the frames per parameter set (1 to 16); other bits 0 |
| 1 per coded channel | codebooks | codes per frame for each coded channel (1 to 9); for mid/side, the mid's then the side's; for parametric stereo, the mid's |
| 1 | flags | bit 0: loop; bit 1: a comment header follows; bit 2: the length is stored. Other bits must be 0 |
| varint | pre-skip | decoded samples a player drops at the start |
| varint | page length | frames per page (the encoder writes 80, about 0.93 s); 0 = the stream is one page |
| varint | playable length | samples after trimming; only with flag bit 2 |
| varint | end trim | decoded samples after the playable end that are padding; only with flag bit 2 |

The hash names the code space, not a decoder: any decoder of DAC 44 kHz's codes (the student, a later student,
or DAC's own decoder) can play the stream, and a decoder must refuse a stream whose hash differs from its own
codebooks.

### Audio packets

| size | field |
|---|---|
| varint | end trim: **only in the last packet, and only when the length is not in the header** |
| varint, then that many bytes | **only for layout 3**: the size of the page's stereo parameters, then the parameters (below) |
| rest | the page's codes: channel 0's frames, then channel 1's; within a frame, codebook 0 first |

Each code is 10 bits, packed low bits first into one bit stream per page that is padded with zeros to a whole
byte. A page holds `page length` frames (the last may hold fewer), so its size in bytes is fixed by the header,
and a reader refuses a packet of any other size.

### Length and granules

The coded audio is `ceil(samples / 512)` frames; the encoder zero-pads the last frame, and that padding is
implicit (it is not part of the end trim). A page's granule position is `frames so far x 512`, capped at the end
of the audio. Every page decodes on its own (the codes need no state), but the decoder is a convolution
network, so a player seeking to a page decodes from a few frames earlier for an exact result.

### Stereo

The encoder codes a stereo input as left/right or mid/side, whichever scores better (log-mel distance, on up to
4 loud 3 s excerpts), unless `--stereo` says which. At 2 codebooks per channel or fewer it also tries parametric
stereo, with a mid of `2 x codebooks - 1` (about the same total rate). A stereo input whose side stays more than
40 dB under its mid is coded as mono with `--channel-format auto`.

### Parametric stereo (layout 3)

The implementation is `src/pstereo.rs`; the measurements behind it are in `tools/pstereo_proto.py` and
`distill/INTEGRATION.md` (M8).

- **Parameter sets.** Each page's frames are split into sets of `k` frames (the header's frames per set; the
  last set of a page may be shorter). A set holds, per band, an IID level (index into the dB grid -25, -18, -14,
  -10, -7, -4, -2, 0, 2, 4, 7, 10, 14, 18, 25; L over R in power) and an ICC level (index into 1, 0.997, 0.99,
  0.97, 0.937, 0.84118, 0.60092, 0.36764, 0, -0.589, -1).
- **Bands.** They are ranges of the bins of a 1024-point STFT (43.07 Hz per bin): 20 bands with edges 0, 3, 6, 9,
  12, 15, 18, 21, 24, 32, 40, 48, 56, 64, 72, 88, 112, 144, 184, 280, 513; or 10 bands with edges 0, 6, 12, 18,
  24, 40, 56, 72, 112, 184, 513.
- **Bits.** The parameters are read low bit of each byte first, as Huffman codes. For each set in order, the
  IID row then the ICC row:
  - in the page's first set, frequency deltas: band 0 against the grid's centre (IID 7, ICC 0), each later band
    against the band before;
  - in later sets, one bit first: 1 = time deltas (each band against the same band of the previous set),
    0 = frequency deltas.

  Deltas are coded with four static canonical Huffman codes (IID or ICC, frequency or time), whose code
  lengths are in `src/pstereo.rs`. The parameters are zero-padded to a whole byte. A reader refuses data that
  is truncated, has extra bytes, or decodes a level out of range. The parameters need nothing from other pages.
- **Upmix.** Run a 1024-point sine-window STFT, hop 512, on the decoded mid M and on D, the mid through the
  velvet-noise FIR in `src/pstereo.rs`. STFT frame `t` covers samples `[(t-1) x 512, (t+1) x 512)` and uses the
  set holding frame `min(t, frames - 1)`. Per band:
  - D is scaled by `min(1, sqrt(E_M / E_D))`, from the band's energies in that frame;
  - with `c = 10^(IID/20)`, `a = c^2 + 1`, `p = max(a + 2 ICC c, 0.1 a)`, `g = (c^2 - 1) / p` and
    `h = sqrt(max(0, (a - 2 ICC c) / p - g^2))`, the side is `S = g M + h D`.

  The decoder then computes `L = M + S` and `R = M - S`, which keeps `(L + R) / 2 = M`. Output near a page
  edge (within 512 samples) uses the neighbouring page's parameters, and D reaches 54 ms back into the mid.
  So, as with the codes, a player seeking to a page decodes from a little earlier.

## Enhancement layers (`.picol`), reserved

Not implemented yet; the format keeps room for it. A **base** `.pico` (for example 4 codebooks, music6) plays on
its own. A `.picol` sidecar holds codebooks `first..=last` of the **same encode** (same model, stride, segments
and trims), so base plus sidecar decodes with `last + 1` codebooks (music12 or music24). Copying only the
`.pico` gives the small file.

- A `.picol` is structurally a `.pico`: the same Ogg framing, one page per chunk, and the **same page boundaries
  and granule positions** as its base, so a decoder pairs pages by granule position.
- Its identification header repeats the base's fields, with `codebooks = last + 1`, and carries the critical
  **layer** TLV (type 7): varint layer index (1 for the first sidecar), `u32` serial number of the base stream,
  `u64` base stream id (FNV-1a 64 of the base's identification packet followed by its first audio packet), varint
  first codebook, varint last codebook. Because the type is critical, a decoder that does not know layers refuses
  a `.picol` on its own instead of playing it.
- Its chunks hold no scales (the base has them). Each chunk codes, per frame, codebooks `first..=last`, with the
  predictor conditioned on codebooks `0..first` of the same frame and the frames before it, as decoded from the
  base. The cx2 predictor already allows this: it is updated with every code of a frame in order, and a code
  need not be range-coded to be fed to it.
- The base stays coded exactly as today, with a predictor that sees only its own codebooks, so it always decodes
  alone.
