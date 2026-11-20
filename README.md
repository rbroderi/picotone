# picotone

Small, seekable, gapless music and sound files for the web, in pure safe Rust. picotone runs the **EnCodec**
neural audio codec, codes its output with **cx2** (a small predictive model and a range coder, lossless on the
codes) and stores it in **`.pico`** files: Ogg pages that any page can be decoded from on its own. Decoders run in
the browser on WebGPU and in WebAssembly.

It was built to store music loops and sound effects for a local web game, so the defaults are tuned for 48 kHz
stereo music that plays in a browser, loops without a click and has no audible once-per-second seam.

> For the design, the measurements and the dead ends, read the **[white paper](WHITEPAPER.md)**. The file
> format is specified in **[FORMAT.md](FORMAT.md)**.

## What you get

| Part | Folder | What it does |
|---|---|---|
| `picotone` command and library | `src/` | Encode any audio file to `.pico`, decode it to WAV, show what a file holds |
| cx2 entropy coder | `src/cx2/` | Range-codes the codes with a predictive model (part of the library) |
| cx2 tables | `models/` | The fitted `.ecxm` tables the coder reads |
| WebGPU decoder | `web/` | Decodes `.pico` in the browser: cx2 in WebAssembly, the network on the GPU |
| WebAssembly decoder | `wasm/` | Decodes in a Web Worker on the CPU. Works everywhere, threaded when the page allows it |
| Table tooling | `tools/`, `train/` | Python scripts that fit and check the tables, and train the network expert |
| Weight study | `dwp/` | Research notes: what the network's parts do, and what can be cut |

The table and training tooling lives at the top level (`tools/`, `train/`) rather than inside the crate: it is
Python, it is only needed to make new tables, and it drives the `picotone dev` commands.

## Quick start

The model weights (Meta's EnCodec, MIT; about 160 MB) are stored with [Git LFS](https://git-lfs.com). Install
it before cloning, or fetch them afterwards:

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

The model weights are in `weights/` (`encodec_24khz.{json,safetensors}` and `encodec_48khz.{json,safetensors}`,
with Meta's licence notice in `weights/EnCodec-MIT.txt`). To use another folder, point `--weights` or `PICOTONE_WEIGHTS` at another folder (the older name `ENCODEC_WEIGHTS` still works
when `PICOTONE_WEIGHTS` is not set). The cx2 tables are read from `models/` (or `--tables` /
`PICOTONE_TABLES`).

Encode music for the game (48 kHz stereo):

```powershell
target\release\picotone encode song.flac song.pico --preset music24
target\release\picotone decode song.pico song.wav
target\release\picotone info song.pico
```

Encode a seamless loop, so the end joins the start with no click:

```powershell
target\release\picotone encode loop.wav loop.pico --preset music12 --loop --tag TITLE=Theme
```

Input can be WAV, FLAC, MP3, Ogg, M4A and most other common formats. It is resampled and up- or down-mixed
to the model automatically.

## Choosing a setting

| Preset | Codebooks | Before cx2 | Typical file | Use it for |
|---|---|---|---|---|
| `music24` | 15 | 24.1 kbps | about 17.5 kbps | Background music (recommended) |
| `music12` | 8 | 12.9 kbps | about 8 kbps | Music where size matters more |
| `music6` | 4 | 6.4 kbps | about 3.2–4 kbps | Ambience, long beds; clearly lower quality |
| `-m mono -b 6` | 8 | 6 kbps | | Speech and sound effects (24 kHz mono model) |

"Typical file" is the whole `.pico` file (headers and Ogg framing included) for two 120 s music clips.

All three music presets use a longer segment overlap than EnCodec's own (stride 44800 instead of 47520). This
removes the faint "beat" once a second that stock EnCodec has at low bitrates. They also use an 8-wide beam
search in the quantiser, which gives slightly better quality at no decode cost.

### Encoder options

| Option | Default | Meaning |
|---|---|---|
| `-m, --model` | `auto` | `mono` (24 kHz) or `stereo` (48 kHz). `auto` picks stereo for stereo input or any preset |
| `-b, --bandwidth` | `6` | Target kbps; sets the number of codebooks |
| `--preset` | none | `music24`, `music12` or `music6`; the flags below override it |
| `--codebooks` | from `-b` | Codebooks per frame, set directly |
| `--stride` | 47520 | Samples between segment starts (48 kHz model, 24000–48000) |
| `--crossfade` | `triangle` | `triangle` (EnCodec's) or `edge` (skips the damaged segment edges; helps with short overlaps) |
| `--beam` | `8` | Quantiser beam width. `--beam 1` gives the reference implementation's greedy codes |
| `--loop` | off | Encode as a gapless loop |
| `--tag KEY=value` | none | A comment (TITLE, ARTIST, LICENSE, ...); repeatable |
| `--table` | the model's default | The cx2 table to code with |
| `--threads` | all cores | Worker threads |

`decode` takes `--float` (32-bit float WAV) and `--rescale` (scale down to avoid clipping, instead of
limiting). `roundtrip` encodes and decodes in one step. `info` prints the headers, the tags and the bitrate.

## The `.pico` format

A `.pico` file is an Ogg stream (RFC 3533) of the `cx2` scheme: an identification header, an optional comment
header, then one range-coded chunk per page. Each page decodes on its own, so a player can seek to any page;
granule positions, the pre-skip, the end trim and the loop flag give sample-exact trims and gapless loops, as in
Ogg Opus. The scales are quantised to 12 bits in the encoder. See [FORMAT.md](FORMAT.md).

The cx2 coder is lossless on the codes. Against plainly packed 10-bit codes it saves 13–30%:

| Preset | Smaller by |
|---|---|
| `music24` | 13–20% |
| `music12` | 17–26% |
| `music6` | 20–30% |

The lower figure is on a harder, more varied test set (pop, vocals). It decodes far faster than the neural
decoder (tens of times real time on one thread).

Three tables ship in `models/`:

| Table | What it is |
|---|---|
| `cx2_encodec_48khz_net_v5.ecxm` | The 48 kHz default, with the trained network expert |
| `cx2_encodec_48khz_v5.ecxm` | The 48 kHz no-network fallback |
| `cx2_encodec_24khz_v5.ecxm` | For 24 kHz |

A file names its table by hash and decodes only with that exact table. The tables are self-contained (they hold
the codebook centroids), so a decoder needs only the table and the decoder half of the weights.

## In the browser

Both decoders take the same `.pico` files, stream audio as it decodes, honour loops and trims, and are written
in strict TypeScript. The cx2 decode is the same Rust code in both, compiled to WebAssembly, so the codes are
bit-exact with the command line.

- **WebGPU** (`web/`): fastest. It needs a secure context (HTTPS or `localhost`). See [`web/README.md`](web/README.md).
- **WebAssembly** (`wasm/`): works everywhere, and can seek. The threaded build is used when the page is
  cross-origin isolated (COOP/COEP headers on HTTPS or `localhost`); otherwise a single-threaded build runs. Over
  plain HTTP on a LAN, only the single-threaded build is available. It still decodes 48 kHz stereo faster than
  real time, but serve over HTTPS (a local certificate is fine) to get WebGPU and threads. See
  [`wasm/README.md`](wasm/README.md) and [`wasm/INTEGRATION-gaim.md`](wasm/INTEGRATION-gaim.md).

| Decoder | 48 kHz stereo decode speed |
|---|---|
| WebGPU (f32 weights) | 10–21× real time |
| WebAssembly, threaded (8 threads) | about 12× |
| WebAssembly, single thread | about 5× (6× in browsers with relaxed SIMD) |

Playback starts while the rest decodes: first audio arrives after about 0.2–0.6 s.

## As a library

```toml
picotone = { path = "../encodecx", default-features = false }           # decoder only, no audio I/O
picotone = { path = "../encodecx", features = ["audio-io"] }            # plus file reading and WAV writing
```

The main entry points:

- `Encodec::from_bytes` and `ModelKind::load_with(dir, LoadOptions)`: load only the encoder, the decoder or
  the codebooks you need.
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

The code is licensed under the [GNU General Public License v3.0](LICENSE) or later.

picotone runs Meta's EnCodec model weights, which are MIT-licensed. Any copy of the weights, including the
browser model files, must carry Meta's notice in [`LICENSES/EnCodec-MIT.txt`](LICENSES/EnCodec-MIT.txt); the
browser exporters copy it next to every model file. This is an independent project, not affiliated with Meta.
For every credit and dependency licence, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
