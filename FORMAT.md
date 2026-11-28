# The `.pico` format

How picotone stores codec codes: EnCodec's (models 0 and 1, below) and the HQ codec's (model 2, in
[HQ streams](#hq-streams-model-2)). A `.pico` file is plain Ogg (RFC 3533) with one logical stream, and its
mapping follows Ogg Opus (RFC 7845) wherever it can: pre-skip, granule positions, an end trim through the final
granule position, and Vorbis-style comments. The implementation is `src/pico.rs`; `tests/pico.rs` covers round
trips, seeking, loops, streams without a length, damage and an independent Ogg reader (the `ogg` crate, and
`ffprobe` when it is installed). General-purpose tools recognise the file as Ogg but not the codec inside.

This document describes **version 1**: a decoder refuses any other version.

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
| 7 | layer | defined (HQ), critical | marks an enhancement layer (`.picol`): base serial, base id, pages per group; see [below](#enhancement-layers-picol-hq-only) |

A stream that carries the layer type is a layer, never a stream of its own: a decoder reads it only as the layer
of its base. picotone's decoders refuse every reserved type today. They are listed so that later versions of the encoder can
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

The HQ codec (`--model hq`, the encoder's default) codes 44.1 kHz audio as DAC 44 kHz codes: picotone's
STFT encoder (`dac44_stftenc`) makes DAC's latent and DAC's own residual quantiser codes it, one frame per 512
samples, up to 16 codebooks of 1024 entries per frame, a lower codebook
count being a prefix of the full one. picotone decodes it with a Vocos (iSTFT) decoder trained on those codes
(`dac44_vocos`). Ogg
framing, the comment header, extensions (TLVs), pre-skip, granules, damage handling and loops work as for
EnCodec; only the fixed part and the audio packets differ. The implementation is `src/pico/dac.rs`.

The format stores codes only. What picotone's decoder does after the network is decoder-side and not part of
the format: at 1 to 5 codebooks per frame a post-filter on the decoder's spectrum and a declicker, and on every
decode (HQ and EnCodec) a look-ahead peak limiter (WHITEPAPER §2.6, §2.7, §3.4). Another decoder of these codes
may leave them out or do otherwise.

### Fixed part

| size | field | meaning |
|---|---|---|
| 1 | model | 2 |
| 1 | scheme | 2 = `cx2` (the only HQ scheme; below) |
| 8 | codebook hash | FNV-1a 64 over the quantiser's 16 x 1024 x 8 codebooks (DAC's 9, then picotone's 7; `f32`, little-endian, stage by stage) |
| 8 | table hash | FNV-1a 64 of the HQ cx2 table file (`.ecxm`, magic `ECXH`) the pages are coded with |
| 1 | layout | 0 = mono, 1 = left/right, 2 = mid/side (M = (L+R)/2, S = (L-R)/2; L = M+S, R = M-S), 3 = parametric stereo (the mid M alone, plus stereo parameters per page) |
| 1 | parametric settings | **only for layout 3**: bits 0-1 the band layout (0 = 20 bands, 1 = 10), bits 2-4 log2 of the frames per parameter set (1 to 16); other bits 0 |
| 1 per coded channel | codebooks | codes per frame for each coded channel (1 to 16); for mid/side, the mid's then the side's; for parametric stereo, the mid's |
| 1 | flags | bit 0: loop; bit 1: a comment header follows; bit 2: the length is stored. Other bits must be 0 |
| varint | pre-skip | decoded samples a player drops at the start |
| varint | page length | frames per page (the encoder writes 80, about 0.93 s); 0 = the stream is one page |
| varint | playable length | samples after trimming; only with flag bit 2 |
| varint | end trim | decoded samples after the playable end that are padding; only with flag bit 2 |

The hash names the code space, not a decoder: any decoder of DAC 44 kHz's codes (picotone's Vocos
decoder, a later one, or DAC's own decoder) can play the stream, and a decoder must refuse a stream whose hash differs from its own
codebooks. The table hash names the entropy coder's model: a decoder needs that exact table (picotone looks for
it by hash in its tables directory; it ships as `models/cx2_dac44.ecxm`) and refuses the stream without it.

### Audio packets

| size | field |
|---|---|
| varint | end trim: **only in the last packet, and only when the length is not in the header** |
| varint, then that many bytes | **only for layout 3**: the size of the page's stereo parameters, then the parameters (below) |
| per coded channel: varint, then that many bytes | the size of the channel's coded codes, then the codes, range-coded with cx2 (below) |

Nothing follows the last channel's codes; a reader refuses a packet with trailing bytes. A page holds
`page length` frames (the last may hold fewer).

### Code coding (`cx2`)

Each coded channel of each page is coded on its own: the predictor starts empty and the range coder (the
64-bit coder of the EnCodec streams' cx2, `src/cx2/rangecoder.rs`) starts afresh, so a page needs nothing from
other pages or channels. The codes go frame by frame, codebook 0 first, `codebooks` per frame. For each code the
model gives 1024 integer logits; the frequencies are `(EXP[d mod 2^f] >> (d div 2^f)) + floor`, where `d` is
the code's distance below the largest logit, `f` the table's fractional bits, `EXP` and `floor` from the table,
and the shift capped at 31.

The model (`src/cx2/hq.rs`, trained by `train/hq_net.py`) is an integer network over the channel's previous 8
frames *of the same page* and the codebooks already coded in this frame, with a bias chosen by the channel's
role: left/right (layout 1), mid (layout 0, the mid of layout 2, and layout 3), or side. Every operation is an
`i16` x `i16` product summed in `i32`, a right shift by a power of two and a clamp, with limits on the table's
values that keep the sums in range, so every decoder computes identical frequencies. The table format is
described in `src/cx2/hq.rs`; its exact arithmetic is the reference.

### Length and granules

The coded audio is `ceil(samples / 512)` frames; the encoder zero-pads the last frame, and that padding is
implicit (it is not part of the end trim). A page's granule position is `frames so far x 512`, capped at the end
of the audio. Every page decodes on its own (the codes' coder starts afresh per page), but the decoder is a convolution
network, so a player seeking to a page decodes from a few frames earlier for an exact result.

### Stereo

The encoder codes a stereo input as left/right or mid/side, whichever scores better (log-mel distance, on up to
4 loud 3 s excerpts), unless `--stereo` says which. At 2 codebooks per channel or fewer it also tries parametric
stereo, with a mid of `2 x codebooks - 1` (about the same total rate). A stereo input whose side stays more than
40 dB under its mid is coded as mono with `--channel-format auto`. `tools/stereo_scan.py` measures the side
level page by page, which sets that threshold, and `tools/stereo_pick_check.py` checks the sampled
left/right or mid/side choice against a whole-track comparison.

### Parametric stereo (layout 3)

The implementation is `src/pstereo.rs`; the prototype and measurements behind it are in
`tools/pstereo_proto.py`.

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

## Enhancement layers (`.picol`, HQ only)

A **base** `.pico` plays on its own. A `.picol` **layer** adds the upper codebooks of the **same encode**, so
base plus layer decodes as the full encode would: the same codes, so the same audio (picotone's output is
sample-for-sample the same as the one-file encode's). A player can start on the small base and pick the layer up
later (gaim's case: play at once, then upgrade when the layer has loaded). Copying only the `.pico` gives the
small file. There is one layer per base, and the base's codebook count is free (1 to 15).

picotone writes a layer with `encode --codebooks B --layer-codebooks N` (and `--layer-side-codebooks` for the
side): `song.pico` holds codebooks `0..B`, `song.picol` the rest up to `N`. The encoder refuses `N = 10`
(codebook 10 adds little on its own); the format allows it. `decode` and `info` pick up the `.picol` next to the
`.pico` (same name, extension `.picol`), unless `--no-layer`. The stereo layout is chosen at the base's codebooks.
On the 23 test clips (`picotone dev layer-cost --top 16`), a base plus its layer costs 1 to 3% more than one
file of the combined codebooks (q3 base +2.8%, q5 +2.7%, q9 +1.5%, q12 +1.1%): the layer's framing (about
0.6%), and the base's own predictor, which sees only the base's codebooks of the past frames. Over a parametric
base it costs about 6% more, as the base's stereo parameters (about 1 kbps) are not used with the layer.

### Base and combined layouts

The **combined** stream is what base plus layer decode as: its layout and its codebooks per coded channel.

| base layout | combined layout | the layer holds |
|---|---|---|
| mono (0) | mono | the mid's codebooks `b..n` |
| left/right (1) | left/right | each channel's codebooks `b..n` |
| mid/side (2) | mid/side | each channel's codebooks `b..n` (the side's base count may differ from the mid's) |
| parametric (3) | **mid/side** | the mid's codebooks `b..n`, and the whole side, codebooks `0..n_s` |

A combined channel has at least as many codebooks as the same channel in the base. With a parametric base the
encoder codes the full encode as mid/side, whatever the stereo pick would choose. The base's mid is then the
combined mid's first codebooks, and with the layer a decoder ignores the base's stereo parameters. A parametric
base cannot have a left/right layer.

### The `.picol` stream

- Ogg, with its own serial number (picotone writes the base's plus 1). Its identification header is an HQ header (model 2) whose layout and
  codebooks are the **combined** stream's. Its codebook hash and table hash equal the base's. Its flags,
  pre-skip, page length, playable length and end trim equal the base's, with no comment header (bit 1 clear).
- It carries the critical **layer** TLV (type 7): `u32` serial number of the base stream, `u64` base stream
  id (FNV-1a 64 over the base's identification packet followed by its first audio packet), then varint
  **group**, the base pages per layer page (1 or more; the encoder writes 4). A decoder refuses a layer whose
  base id does not match the base it is given. Because the type is critical, a decoder that does not know
  layers refuses a `.picol` given alone and does not play it.
- **Pages in groups:** layer page `j` covers base pages `j x group` to `j x group + group - 1` (the last layer
  page may cover fewer), and its granule position is that of the last base page it covers. A decoder pairs
  them by granule position. Grouping saves Ogg page overhead: on the test clips the layer's framing is about
  2.3% of a single file's size at group 1 and 0.8% at 4 (`picotone dev layer-cost`); the cost is that a damaged
  layer page drops `group` base pages (about 3.7 s at 4) to base quality.
- There is one layer packet per layer page, and a layer has exactly `ceil(base pages / group)` of them.

### Layer audio packets

For each base page the layer page covers, in order:

| size | field |
|---|---|
| per combined coded channel: varint, then that many bytes | the size of the channel's layer codes for that base page, then the codes (size 0 when the layer adds nothing to that channel) |

There are no stereo parameters and no end trim (the base has them). Each base page's chunks are coded on their
own, as below, so grouping changes only the framing. For combined channel `c` with `n` codebooks,
whose base holds its first `b` (0 for the side of a parametric base), the chunk range-codes, frame by frame, the
codes of codebooks `b..n`. It uses a fresh `n`-codebook predictor with the combined channel's role (left/right,
mid or side). Each frame's codebooks `0..b` come from the base's page: they are fed to the predictor in order, but
not coded. So the predictor sees exactly what it would see coding the full encode. The base is coded exactly as
any `.pico`, with a predictor that sees only its own codebooks, so it always decodes alone.

### Decoding

Base plus layer: per base page, decode the base's codes, then the layer's, and decode the combined codes. A
decoder may play the base pages that a missing or damaged layer page covers from the base alone, at the base's
codebooks (with a parametric base, with the base's stereo parameters); near such a page's edges the output
depends on the decoder's context (as for seeking) and is not specified exactly. **picotone** does not mix the
two: it refuses a layer with any missing or damaged page as a whole, and plays the base alone. A damaged base page
is lost with or without the layer.

A decoder must refuse:
- a layer given without its base;
- a layer whose base serial or base id, table hash or codebook hash differs from the base's (the base id needs
  the base's first audio packet intact);
- a layer whose flags (other than bit 1, which a layer leaves clear), pre-skip, page length or (when stored)
  length differ from the base's;
- a combined layout that the table above does not allow for the base's layout, a combined channel with fewer
  codebooks than the base's, or a layer that adds no codebooks;
- a layer with the wrong number of pages, a page whose granule position is not that of the last base page it
  covers, or a packet with missing or trailing bytes.
