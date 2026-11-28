# picotone

Small, seekable, gapless music and sound files for the web, in pure safe Rust. picotone's default **HQ** codec
codes 44.1 kHz audio as DAC 44 kHz codes (its own STFT encoder and Vocos decoder, 1 to 16 codebooks per
channel); Meta's **EnCodec** (48 kHz stereo, 24 kHz mono) is the fast path (`--fast`): it encodes much faster than HQ. Both code their output with **cx2**
(a small integer predictive model and a range coder, lossless on the codes) and store it in **`.pico`** files:
Ogg pages that each decode on their own. Decoders run natively and in the browser (WebAssembly; WebGPU for
EnCodec).

It was built to store music loops and sound effects for a local web game, so it is tuned for music that plays
in a browser, starts quickly and loops without a click.

> For the design and the measurements, read the **[white paper](WHITEPAPER.md)**. The file format is
> specified in **[FORMAT.md](FORMAT.md)**.

## What you get

| Part | Folder | What it does |
|---|---|---|
| `picotone` command and library | `src/` | Encode any audio file to `.pico`, decode it to WAV, show what a file holds |
| cx2 entropy coder | `src/cx2/` | Range-codes the codes with a predictive model (part of the library) |
| cx2 tables | `models/` | The fitted `.ecxm` tables the coder reads |
| WebAssembly decoder | `wasm/` | Decodes HQ and EnCodec files in a Web Worker on the CPU. Works everywhere, threaded when the page allows it |
| WebGPU decoder | `web/` | Decodes EnCodec files in the browser: cx2 in WebAssembly, the network on the GPU |
| Table tooling | `tools/`, `train/` | Python scripts that fit and check the tables and train their networks |
| Measurements | `dwp/` | Scripts behind the EnCodec seam and beam-search measurements |

The table and training tooling lives at the top level (`tools/`, `train/`) rather than inside the crate: it is
Python, it is only needed to make new tables, and it drives the `picotone dev` commands.

## Quick start

The model weights are stored with [Git LFS](https://git-lfs.com). Install it before cloning, or fetch them
afterwards:

```powershell
git lfs install
git clone https://github.com/rbroderi/picotone.git
cd picotone
git lfs pull          # only needed if LFS was installed after cloning
```

Build (Rust 1.88 or newer):

```powershell
cargo build --release
```

The weights are in `weights/`:

- HQ: `dac44_stftenc` (picotone's STFT encoder, which makes DAC's latent), `dac44_quantizer` (DAC 44 kHz's
  quantiser and picotone's 7 extra codebooks) and `dac44_vocos` (picotone's Vocos decoder), each a
  `.json` and `.safetensors` with a `.NOTICE.txt` and `.credits.jsonl` (training-data licences and credits);
  DAC's licence is `weights/DAC-MIT.txt`.
- EnCodec: `encodec_24khz` and `encodec_48khz` (`.json` and `.safetensors`), with Meta's licence notice in
  `weights/EnCodec-MIT.txt`.

To use another folder, point `--weights` or `PICOTONE_WEIGHTS` at it (`ENCODEC_WEIGHTS` is read when
`PICOTONE_WEIGHTS` is not set). The cx2 tables are read from `models/` (or `--tables` / `PICOTONE_TABLES`).

Encode with the HQ codec (the default: 44.1 kHz, about 6 kbps per channel at its default 9 codebooks, up to about 10.6 kbps at `--codebooks 16`), or with
EnCodec (`--fast`, or a preset):

```powershell
target\release\picotone encode song.flac song.pico
target\release\picotone encode song.flac song.pico --preset music24
target\release\picotone decode song.pico song.wav
target\release\picotone info song.pico
```

The WebAssembly decoder (`wasm/`) plays HQ and EnCodec files; the WebGPU one (`web/`) plays EnCodec files only.

An HQ file can be split into a small base that plays on its own and an enhancement layer (`.picol`) with the
rest of the same encode's codebooks; with both it decodes exactly as a one-file encode of all of them:

```powershell
target\release\picotone encode song.flac song.pico --codebooks 5 --layer-codebooks 16
target\release\picotone decode song.pico song.wav      # uses song.picol when it is there; --no-layer skips it
```

The two parts cost 1-3% more than one file of all the codebooks. See [FORMAT.md](FORMAT.md#enhancement-layers-picol-hq-only).

Encode a seamless loop, so the end joins the start with no click:

```powershell
target\release\picotone encode loop.wav loop.pico --preset music12 --loop --tag TITLE=Theme
```

Input can be WAV, FLAC, MP3, Ogg, M4A and most other common formats. It is resampled and up- or down-mixed
to the model automatically.

## Choosing a setting

| Setting | Codebooks | Before cx2 | Typical file | Use it for |
|---|---|---|---|---|
| (none: HQ) | 9 per channel | 7.75 kbps per channel | about 6.1 kbps per channel | 44.1 kHz music and effects, the default |
| `--codebooks N` (HQ) | 1–16 per channel | 0.86 kbps each | 0.6 (q1), 1.8 (q3), 3.9 (q6), 8.1 (q12), 10.6 (q16) kbps per channel | A lower or higher HQ rate |
| `--preset music24` | 15 | 24.1 kbps | about 17.5 kbps | EnCodec 48 kHz stereo music |
| `--preset music12` | 8 | 12.9 kbps | about 8 kbps | EnCodec, where size matters more |
| `--preset music6` | 4 | 6.4 kbps | about 3.2–4 kbps | EnCodec ambience, long beds; clearly lower quality |
| `--fast --channel-format mono -b 6` | 8 | 6 kbps | | Speech and sound effects (EnCodec 24 kHz mono) |

"Typical file" is the whole `.pico` file, headers and Ogg framing included (HQ: the 23 test clips; EnCodec: two
120 s music clips).

The EnCodec music presets use a longer segment overlap than EnCodec's own (stride 44800 instead of 47520), which
removes stock EnCodec's faint once-a-second "beat", and an 8-wide beam search in the quantiser, which gives
slightly better quality at no decode cost.

### Encoder options

| Option | Default | Meaning |
|---|---|---|
| `--fast` | off | EnCodec instead of HQ: `encodec-24k` for mono, `encodec-48k` for stereo |
| `--model` | HQ | The exact model: `hq`, `encodec-48k` or `encodec-24k`; overrides `--fast` and presets |
| `--channel-format` | `auto` | `auto` (follow the input), `mono` or `stereo`. With HQ, `auto` codes a stereo input whose side is over 40 dB under its mid as mono. A model that can't give the format is an error |
| `--stereo` | `pick` | HQ only: `lr`, `ms` (mid/side), `parametric`, or `pick` the better on sampled excerpts |
| `--codebooks` | HQ: 9; EnCodec: from `-b` | Codebooks per frame (HQ: per channel, 1–16, 0.86 kbps each before cx2) |
| `--side-codebooks` | `--codebooks` | HQ mid/side: codebooks for the side |
| `--layer-codebooks` | none | HQ: also write OUTPUT.picol, a layer up to this many codebooks (2–16, not 10) |
| `--layer-side-codebooks` | `--layer-codebooks` | HQ mid/side or parametric base: the side's codebooks with the layer |
| `-b, --bandwidth` | `6` | EnCodec only: target kbps; sets the number of codebooks |
| `--preset` | none | EnCodec only: `music24`, `music12` or `music6` (implies `encodec-48k`); the flags below override it |
| `--stride` | 47520 | EnCodec 48 kHz only (as are `--crossfade`, `--beam` and `--table`): samples between segment starts, 24000–48000 |
| `--crossfade` | `triangle` | `triangle` (EnCodec's) or `edge` (skips the damaged segment edges; helps with short overlaps) |
| `--beam` | `8` | Quantiser beam width. `--beam 1` gives the reference implementation's greedy codes |
| `--loop` | off | Encode as a gapless loop |
| `--tag KEY=value` | none | A comment (TITLE, ARTIST, LICENSE, ...); repeatable |
| `--table` | the model's default | The cx2 table to code with |
| `--threads` | all cores | Worker threads |

Every decode ends in a look-ahead peak limiter, which lowers the gain smoothly around any sample the decoder
puts over full scale (WHITEPAPER §3.4). `decode` takes `--float` (32-bit float WAV), `--rescale` (scale the
whole file down to fit, instead of limiting) and `--no-limit` (the decoder's raw output, for measurements). HQ files decode with the Vocos (iSTFT) decoder, `weights/dac44_vocos.safetensors`; at 1 to 5 codebooks its post-filter removes most frame-boundary clicks and a declicker repairs those left on the frame grid (WHITEPAPER §2.6–2.7). `roundtrip` encodes and decodes in one step. `info` prints the headers, the tags and the bitrate.

## The `.pico` format

A `.pico` file is an Ogg stream (RFC 3533) of the `cx2` scheme: an identification header, an optional comment
header, then one range-coded chunk per page. Each page decodes on its own, so a player can seek to any page;
granule positions, the pre-skip, the end trim and the loop flag give sample-exact trims and gapless loops, as in
Ogg Opus. The scales are quantised to 12 bits in the encoder. See [FORMAT.md](FORMAT.md).

The cx2 coder is lossless on the codes. HQ codes take about 7.9 bits each at 9 codebooks (7.1 at 3) instead
of 10. On EnCodec files it saves 14–30% against plainly packed 10-bit codes:

| Preset | Smaller by |
|---|---|
| `music24` | 14–20% |
| `music12` | 18–26% |
| `music6` | 21–30% |

The lower figure is on a harder, more varied test set (pop, vocals). It decodes far faster than the neural
decoders (tens of times real time on one thread).

Four tables ship in `models/`:

| Table | What it is |
|---|---|
| `cx2_dac44.ecxm` | HQ, every codebook count |
| `cx2_encodec_48khz_net2_v5.ecxm` | The EnCodec 48 kHz default, with the trained network expert (`net48v2`) |
| `cx2_encodec_48khz_v5.ecxm` | EnCodec 48 kHz without the network |
| `cx2_encodec_24khz_v5.ecxm` | EnCodec 24 kHz |

A file names its table by hash and decodes only with that exact table. picotone ships only its newest tables;
when a table is replaced, re-encode files made with the old one. The tables are self-contained (they hold
the codebook centroids), so a decoder needs only the table and the decoder half of the weights.

## In the browser

Both decoders take the same `.pico` files, stream audio as it decodes, honour loops and trims, and are written
in strict TypeScript. The cx2 decode is the same Rust code in both, compiled to WebAssembly, so the codes are
bit-exact with the command line.

- **WebAssembly** (`wasm/`): plays HQ and EnCodec files, works everywhere, can seek, and takes HQ files'
  `.picol` layers. The threaded build is used when the page is
  cross-origin isolated (COOP/COEP headers on HTTPS or `localhost`); otherwise a single-threaded build runs. Over
  plain HTTP on a LAN, only the single-threaded build is available. It still decodes 48 kHz stereo faster than
  real time, but serve over HTTPS (a local certificate is fine) to get WebGPU and threads. See
  [`wasm/README.md`](wasm/README.md) and [`wasm/INTEGRATION-gaim.md`](wasm/INTEGRATION-gaim.md).
- **WebGPU** (`web/`): EnCodec files only, fastest for them. It needs a secure context (HTTPS or `localhost`).
  See [`web/README.md`](web/README.md).

| Decoder | Stereo decode speed |
|---|---|
| WebAssembly, single thread, HQ | about 4.4–4.7× real time |
| WebAssembly, EnCodec 48 kHz, threaded (8 threads) | about 12× |
| WebAssembly, EnCodec 48 kHz, single thread | about 5× (6× in browsers with relaxed SIMD) |
| WebGPU, EnCodec 48 kHz (f32 weights) | 10–21× |

Playback starts while the rest decodes: first audio arrives after about 0.2–0.6 s.

## In foobar2000

`foo_input_pico` (`foobar/`) is a foobar2000 input component for Windows, for the 32-bit and 64-bit players. It plays
HQ and EnCodec `.pico` files, seeks, shows the tags, and uses an HQ file's `.picol` layer when one lies next to
it (`song.pico` + `song.picol`). The decoding is the library's streaming player, linked into the component, so
the output is the same as the command line's. Build and install it with
`powershell -ExecutionPolicy Bypass -File foobar\build.ps1 -Install`, or install the
`foobar\build\foo_input_pico.fb2k-component` it packs from foobar2000's Components page. See
[`foobar/README.md`](foobar/README.md).

## As a library

```toml
picotone = { path = "../encodecx", default-features = false }           # decoder only, no audio I/O
picotone = { path = "../encodecx", features = ["audio-io"] }            # plus file reading and WAV writing
```

The main entry points:

- `player::Player`: a streaming player for any `.pico` file (with its `.picol`), from any sample, a block at a
  time, limited (`limiter::Limiter`); the browser decoder and the foobar2000 component use it.
- `pico::DacStream::parse` (with `attach_layer` for a `.picol`) and `hqstream::HqStream`: HQ streams, decoded page by
  page.
- `Encodec::from_bytes` and `ModelKind::load_with(dir, LoadOptions)`: load only the EnCodec encoder, decoder or
  codebooks you need.
- `encode_with`, `decode_with`: whole files, as codes.
- `pico::write`, `pico::Stream::parse`, `Stream::decode_page`: the `.pico` container, a page at a time.
- `cx2::Model::from_table`: load a table.
- `decode_segment` with `overlap_add()`, and `stream_decoder()`: decode as you go, with memory bounded by the
  chunk size.

## Development

```powershell
cargo fmt --check
cargo clippy --all-targets
cargo test --release
```

The crate forbids `unsafe` and denies the clippy `pedantic`, `nursery` and `cargo` groups. Python tooling uses
[uv](https://docs.astral.sh/uv/) with `pyproject.toml` (`tools/`, `train/`). `tools/parity.py` checks the
greedy codes (`--beam 1`) and the decoder against the Hugging Face reference implementation; it is a test only.

## Licence

The code is licensed under the [GNU General Public License v3.0](LICENSE) or later. The foobar2000 component
(`foobar/`, and the DLL it builds with picotone's library inside) is licensed under the
[GNU Lesser General Public License v3.0](foobar/COPYING.LESSER) or later.

picotone runs Meta's EnCodec model weights and weights derived from Descript's DAC, both MIT-licensed. Any copy
of the weights, including the browser model files, must carry their notices (Meta's in [`LICENSES/EnCodec-MIT.txt`](LICENSES/EnCodec-MIT.txt), which the browser
exporters copy next to every model file; `weights/DAC-MIT.txt` and the HQ weights' `.NOTICE.txt` files). This is an independent project, not
affiliated with Meta or Descript.
For every credit and dependency licence, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
