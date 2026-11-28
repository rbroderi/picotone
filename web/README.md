# picotone-webgpu: the WebGPU decoder

Decodes `.pico` files (EnCodec 24 kHz mono / 48 kHz stereo, cx2-coded) in the browser. The cx2 range decode
runs in WebAssembly, in the same Rust code as the command line (so the codes are bit-exact), one Ogg page at a
time; the codes are then fed to the EnCodec network on the GPU. The network is a port of the Rust decoder in
`../src`, and the parity tests check it against the exe. It refuses HQ (`dac44`) files: those play through the
WebAssembly decoder in `../wasm`.

## Build and run

```
npm install
npm run export-weights   # public/models/: the weights, the cx2 tables, cx2-tables.json and EnCodec-MIT.txt
npm run build            # tsc --noEmit, then public/build/ (demo + parity page), dist/picotone-webgpu.js (library),
                         # and copies the cx2 WebAssembly module to public/models/cx2/
npm run serve            # serves http://127.0.0.1:8080/ (the demo is index.html)
npm run test:parity      # makes the references with the exe, then checks SNR and speed in headless Chrome
```

`npm run build` needs the single-threaded WebAssembly build (`wasm/www/picotone/pkg-st/`). Make it with
`powershell -File ../wasm/build.ps1 -Only st -SkipModels` if it is missing.

`export-weights` writes, under `public/models/`:

```
encodec_{24,48}khz/{manifest.json,weights-f32.bin,weights-f16.bin,EnCodec-MIT.txt}
EnCodec-MIT.txt                      the weights' licence (must travel with them)
cx2_encodec_24khz_v5.ecxm, cx2_encodec_48khz_net2_v5.ecxm  each model's default cx2 table
cx2-tables.json                      { tables: [{ file, model, hash }] }
cx2/picotone.js, cx2/picotone_bg.wasm   (from npm run build)
```

Each `manifest.json` has `"format": "picotone-webgpu-weights"`. The loader refuses any other format.

`test:parity` and `make-refs` read `PICOTONE_TEST_DIR` (where the test audio goes; default
`%TEMP%/picotone-web-test`), `PICOTONE_EXE` (the path to `picotone.exe`) and `PICOTONE_LISTEN_DIR` (adds two
120 s music clips).

`node tools/profile.mjs <case>` prints the GPU time for each kernel.

## Library use

```ts
import { PicotoneWebGpu, isWebGpuAvailable, parseHeader } from './dist/picotone-webgpu.js';

const gpu = isWebGpuAvailable();          // cheap, synchronous; { available, reason }
if (!gpu.available) useWasmFallback(gpu.reason);

const header = parseHeader(bytes);        // header.model: 'encodec_24khz' | 'encodec_48khz'; no WebAssembly needed
const model = await PicotoneWebGpu.load({ model: header.model, baseUrl: '/static/picotone/models/', dtype: 'f16' });
const out = await model.decode(bytes, {
  onChunk: (c) => schedule(c.channels, c.offset, c.sampleRate), // optional streaming
});
// out.sampleRate, out.channels: Float32Array[] (planar)
```

- `baseUrl` is the folder that holds the `<model>/manifest.json` folders, `cx2-tables.json`, the tables and
  `cx2/`. `cx2Url` points elsewhere for the WebAssembly module. The module and the tables load once per page and
  are shared by both models.
- Pass `collect: false` to stream only, so the full PCM is not kept.
- Pass `signal` to abort a decode.
- WebGPU needs a secure context (HTTPS or localhost). On LAN clients over plain HTTP, `isWebGpuAvailable()`
  returns false and says why.
- Files carry their segment stride and cross-fade in the header (the presets use stride 44800). Strides from
  half a segment (24000) to a whole one (48000) are accepted.
- Files from `picotone encode --loop` carry a pre-skip, an end trim and the loop flag. `decode()` drops the
  pre-skip and end trim, so `out.length` is the original length and chunk offsets count trimmed samples. It also
  folds the coded pre-roll into the last 100 ms, so the last sample runs into the first. `out.loop` is then
  true, and `play(ctx, out)` loops the whole buffer with no click and no loop points
  (`AudioBufferSourceNode.loop = true`). Do not pad loops by hand.
- Every decode ends in the look-ahead peak limiter (`src/limit.ts`, a port of the library's `limiter::Limiter`
  with the same f32 arithmetic): it lowers the gain smoothly around any sample the network puts over full
  scale, and leaves the rest unchanged.
- A file decodes only with the exact cx2 table it was coded with; `PicoError` names the missing hash.
- This decoder always starts at the beginning of the file. For seeking, use the WebAssembly decoder.
- The library builds under `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`,
  `verbatimModuleSyntax`, `isolatedModules` and `noUnusedLocals`.
- A consumer whose tsconfig sets `types` also needs `@webgpu/types`. The source has a triple-slash reference to it.

## Speed

On a desktop GPU (Chrome, f32 weights): 48 kHz stereo 10–21× real time (75 s at music6: 21×), 24 kHz mono
23–33×. First audio after 0.1–0.6 s (`npm run test:parity` prints it cold and warm): only the first page is
range-decoded before the GPU starts.

## Memory bounds

- **24 kHz files:** decoded as one causal stream in chunks of 375 frames (5 s); the first chunk is 75 frames.
  - Each layer carries its history rows across chunks, along with the LSTM state.
  - GPU memory is fixed per chunk, whatever the file length.
- **48 kHz files:** split into 1 s segments, decoded in batches of up to 4 (the first batch is 1 segment, for a
  quick start).
  - Overlap-add runs on the GPU, and the overlap tail carries from batch to batch.
- **Readback:** uses a two-deep staging ring.
- **GPU totals:** about 91 MB for 24 kHz and 161 MB for 48 kHz in f32 (87 / 157 MB in f16).
- **cx2:** pages are range-decoded as the GPU needs them, so the codes held are about one batch.

## Weights

| model | f32 | f16 |
|---|---|---|
| encodec_24khz | 46.5 MB | 23.2 MB |
| encodec_48khz | 38.1 MB | 19.0 MB |

f16 is about 59–69 dB SNR against the exe, and f32 is about 120 dB. f16 also runs the LSTM from packed halves,
which is about 1.4–1.7× faster. The cx2 module adds about 0.7 MB (it also holds the CPU network, unused here),
and the tables 7.4 MB (48 kHz) and 12.5 MB (24 kHz).
