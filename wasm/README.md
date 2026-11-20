# picotone-wasm: the picotone decoder in the browser

A WebAssembly build of the picotone decoder. It decodes `.pico` files in a dedicated Web Worker and returns
planar `Float32Array`s. The cx2 range decode and the EnCodec network run in the same Rust code as the command
line. It ships in three builds:

| build | toolchain | features | used when |
|---|---|---|---|
| `pkg-mt` (threaded) | nightly, `-Z build-std=std,panic_abort` | `+atomics,+bulk-memory,+mutable-globals,+simd128`, shared memory, `wasm-bindgen-rayon` pool of `navigator.hardwareConcurrency` workers | `self.crossOriginIsolated` is true |
| `pkg-st-fma` (single, relaxed SIMD) | stable | `+simd128,+relaxed-simd`: the matrix products use the fused multiply-add | not isolated, and the browser has relaxed SIMD |
| `pkg-st` (single) | stable | `+simd128` (rayon runs everything on the calling thread) | everywhere else |

The worker picks the build and reports which one it chose and why (`decoder.build.kind`,
`decoder.build.relaxedSimd`, `decoder.build.reason`). `pkg-st-fma` is about 15–20% faster than `pkg-st` on
48 kHz stereo. Its output differs from `pkg-st`'s by float rounding in the network only (both stay above 100 dB
SNR against the native decoder). A browser without relaxed SIMD cannot compile it, so the worker checks first
(`WebAssembly.validate` of a probe module) and falls back to `pkg-st`;
`PicotoneDecoder.create({ relaxedSimd: false })` forces `pkg-st`.

The cx2 decode is integer-only and identical in every build: relaxed SIMD touches only the network's f32
kernels, and the coder's SIMD dispatch (`pulp`) takes its portable path on WebAssembly. The codes are bit-exact
with the native decoder.

## Build

Prerequisites (one-off):

```powershell
rustup target add wasm32-unknown-unknown
rustup +nightly target add wasm32-unknown-unknown
rustup +nightly component add rust-src
cargo install wasm-bindgen-cli --version 0.2.130 --locked   # must equal the crate's wasm-bindgen
cd wasm; npm install                                          # TypeScript + puppeteer-core
```

Then, from `wasm/`:

```powershell
powershell -File build.ps1          # all builds + TS API + model export (or: npm run build)
powershell -File build.ps1 -Only st # just one build (st, fma or mt)
node tools/export-models.mjs        # (re)write www/models/
npm run build:ts                    # only the TS API: ts/*.ts -> www/picotone/*.js, *.d.ts
```

Everything under `www/picotone/` and `www/models/` is build output and is not in git, so a fresh
checkout needs `build.ps1` (or at least `npm install` and `npm run build:ts` plus the wasm builds)
before the site or the tests work.

Every cargo build goes to `wasm/target/{st,st-fma,mt,native}`. Nothing is built in the root `target/`.
`wasm-opt` runs if it is on `PATH`; it is optional.

Output (`www/` is the deployable site):

```
www/index.html, demo.js            demo page
www/picotone/picotone.js (.d.ts)   main-thread API (compiled from ts/picotone.ts)
www/picotone/pico-header.js        the .pico header reader (no worker needed)
www/picotone/protocol.js (.d.ts)   message types
www/picotone/worker.js             the dedicated decoder worker
www/picotone/pkg-st/               single-threaded wasm-bindgen output
www/picotone/pkg-st-fma/           the same with relaxed SIMD
www/picotone/pkg-mt/               threaded wasm-bindgen output (+ rayon worker snippet)
www/models/encodec_{24,48}khz.{json,safetensors}   decoder-only weights
www/models/EnCodec-MIT.txt         the weights' licence (must travel with them)
www/models/cx2_*.ecxm              the default cx2 table of each model
www/models/cx2-tables.json         { tables: [{ file, model, hash }] }; the worker loads these
www/_headers                       COOP/COEP for Netlify / Cloudflare Pages
```

To decode files coded with another table, copy the `.ecxm` next to the models and add it to
`cx2-tables.json` (the hash is FNV-1a 64 of the file, as 16 hex digits; `export-models.mjs` computes it).

## Serve and try

```powershell
node serve.mjs                 # http://127.0.0.1:8080/  COOP+COEP on: threaded build
node serve.mjs --no-isolation  # headers off: single-threaded fallback
```

Drop a `.pico` file on the page. It shows the headers and the active build, streams playback (stereo starts
after the first segment), and offers a WAV download. `?build=single` forces the fallback.

## Test

```powershell
node test/run.mjs            # or: npm test     (add --quick to skip the long files)
```

The test synthesises audio and encodes it to `.pico` with `target\release\picotone.exe` (presets, edge
cross-fade, loops, and a stream with no length in its header), then writes a float reference with
`decode --float`. It decodes every file in headless Chrome with every build (served with and without
isolation), checks SNR ≥ 100 dB against the reference, seeks into five files at 37% and 81% and checks those
decodes against the reference too, and prints speed, time to first audio, underruns, wasm memory and the
loop-seam measurement. Env: `PICOTONE_EXE`, `PICOTONE_TEST_DIR`, `PICOTONE_TEST_THREADS` (threaded pool size),
`PICOTONE_LISTEN_DIR` (adds two 120 s music clips). `cargo test --release` (with
`CARGO_TARGET_DIR=target/native`) checks natively that the chunked stream, every seek and the codes-only paths
are bit-identical to a whole-file decode.

Measured on a 16-thread desktop (Chrome, 120 s 48 kHz stereo at music12): threaded (8 threads) about 12× real
time, single-threaded 6.0× with relaxed SIMD and 5.1× without; first audio after 0.2–0.3 s.

## API (TypeScript)

```ts
import { PicotoneDecoder, readPicoHeader, encodeWav } from "./picotone/picotone.js";

const info = readPicoHeader(bytes);          // model, audioLength, tags, bitrateKbps, ...
const decoder = await PicotoneDecoder.create({ baseUrl: "/static/picotone/" });
console.log(decoder.build.kind, decoder.build.reason);
await decoder.loadModel(info.model, { onProgress: (got, total) => {} });   // weights + cx2 tables

// Whole file (chunks also stream to onChunk as they become final):
const audio = await decoder.decode(bytes, { onProgress: (p) => {} });
// audio.sampleRate, audio.channels: Float32Array[] (planar), audio.stats

// From a later sample (a seek); audio.startSample says where it starts:
const tail = await decoder.decode(bytes, { startSample: 48000 * 30 });

// Streaming playback, starting after the first segment:
const playback = decoder.play(audioContext, bytes, { loop: true });
await playback.started;  // ms to first audio

// Cached effect:
const buffer = await decoder.decodeToAudioBuffer(audioContext, bytes);
```

All URLs (worker, the builds, models, tables) derive from `baseUrl`; `modelsUrl` and `workerUrl` override them.
Model files and tables are cached in Cache Storage (`cacheName`, default `picotone-models-v1`) when the context
is secure, and left to the HTTP cache otherwise.

For other decoders (the WebGPU one), the module also exports `readCodes(bytes, tables)` and `CodePages` (one Ogg
page at a time): the cx2 decode without the network.

### Streaming and overlap-add

The 48 kHz model codes 1 s segments `stride` apart (0.933 s at the presets' 44800) and cross-fades the overlaps.
The segments of a chunk are decoded together with `Encodec::decode_segments` (the LSTM advances them as one
batch, so a larger chunk costs less per segment; the threaded build also shares their convolutions) and pushed
in order into `Encodec::overlap_add()`, which returns only the samples no later segment can change. The 24 kHz
model is causal and unsegmented: its codes go through `Encodec::stream_decoder` one second of latent frames at a
time, carrying the convolution and LSTM state between pushes. Both are bit-identical to a whole-file decode (see
the native test), nothing is decoded twice, and memory depends on the chunk size, not the file length. Pages are
range-decoded only when the network needs their codes.

Chunks are 1 unit first, then 2, 4, …, up to `chunkSegments` (default 16 threaded, 8 single-threaded). A unit is
one segment (48 kHz) or one second of frames (24 kHz).

### Seeking

`decode(bytes, { startSample })` and `play()` after a seek start at the last segment whose start is at or before
the wanted sample and that no earlier segment overlaps there, with a fresh overlap-add, then trim to the exact
sample, so the result equals the same samples of a whole-file decode. Files of the causal 24 kHz model and
loops decode from the start and drop the samples before `startSample`.

### Loops

`play(ctx, bytes, { loop: true })` streams the file once, then hands over to a looping `AudioBufferSourceNode`
with sample-accurate timing. Encode loops with `picotone encode loop.wav loop.pico --preset music6 --loop`; no
padding and no loop points are needed. `--loop` codes the loop with 150 ms of its own end before it and 50 ms of
its start after it, and records the pre-skip, the end trim and the loop flag in the header. The decoder drops
those samples, so the decoded length equals the original, and folds the coded pre-roll into the last 100 ms, so
the last sample runs into the first. A file encoded without `--loop` still clicks slightly at the seam, because
EnCodec's first and last few ms are not faithful.

### Header fields

`readPicoHeader()` and the stream's getters expose: `model`, `audioLength` (always the trimmed length),
`codedLength`, `numCodebooks`, `stride`, `crossfade` (0 triangle, 1 edge-avoiding), `preSkip`, `endTrim`,
`loop`, `tableHash`, `pageLength`, `tags` and `outputGainDb`. See [`../FORMAT.md`](../FORMAT.md).

## GitHub Pages (no custom headers)

GitHub Pages cannot send COOP/COEP. Either accept the single-threaded fallback (it works out of the box), or add
[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker). Copy `coi-serviceworker.js` next to
`index.html` and load it first:

```html
<script src="coi-serviceworker.js"></script>
```

On first visit it registers a service worker that re-serves every response with COOP/COEP, then reloads the
page once. After that the page is cross-origin isolated and the threaded build is used. It needs a secure
context (HTTPS, which Pages provides), and the first load always runs without isolation.

## Requirements and limitations

- Browsers need WebAssembly SIMD (Chrome 91+, Firefox 89+, Safari 16.4+). The threaded build also needs
  cross-origin isolation, which only works in a **secure context**: HTTPS or `localhost`. Plain
  `http://<lan-ip>` always gets the single-threaded build.
- A file decodes only with the exact cx2 table it was coded with; the worker says which hash is missing.
- A loaded model or table stays in wasm memory for the worker's lifetime (streams borrow them across calls).
  `Decoder.free()` does not release the weights. The worker loads each model once, and
  `PicotoneDecoder.dispose()` terminates it.
- Decoder-only weights hold every codebook (any bandwidth). Files for a single bandwidth could drop the higher
  codebooks; the library supports that (`LoadOptions::codebooks`), but the exporter and bindings do not use it yet.
- The output gain field is read but not applied.
- `wasm-bindgen-rayon`'s no-bundler helper uses wasm-bindgen's deprecated positional init, so each pool worker
  logs one "deprecated parameters" warning. It is harmless.
