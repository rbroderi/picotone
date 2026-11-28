# Integrating picotone-wasm into gaim

How to play picotone `.pico` music loops and sound effects in gaim's console and table
pages, decoded in the browser by the WebAssembly decoder in this folder.

The files in §1 belong together: copy all of them from one build, and change `cacheName` (§2)
whenever you copy a new build, so no browser mixes files from two builds. Every asset must be
encoded with `picotone encode` against the tables of that build.

## 1. Files to copy

Build first (`powershell -File wasm\build.ps1`; it compiles the wasm builds and the `.js`
files of `www/picotone/`, none of which are in git, and exports `www/models/`), then copy:

| from (picotone repo) | to (gaim) | how it is used |
|---|---|---|
| `wasm/www/picotone/worker.js`, `protocol.js`, `pico-header.js` | `src/gaim/web/static/picotone/` | dedicated decoder worker, loaded by URL |
| `wasm/www/picotone/pkg-st/` (whole folder) | `src/gaim/web/static/picotone/pkg-st/` | single-threaded build |
| `wasm/www/picotone/pkg-st-fma/` (whole folder) | `src/gaim/web/static/picotone/pkg-st-fma/` | single-threaded build with relaxed SIMD, picked over `pkg-st` where the browser has it (about 15–20% faster) |
| `wasm/www/picotone/pkg-mt/` (whole folder, incl. `snippets/`) | `src/gaim/web/static/picotone/pkg-mt/` | threaded build + rayon pool worker |
| `wasm/www/models/encodec_*.json`, `encodec_*.safetensors` | `src/gaim/web/static/picotone/models/` | EnCodec decoder-only weights (63.4 MB mono, 46.6 MB stereo) |
| `wasm/www/models/EnCodec-MIT.txt` | `src/gaim/web/static/picotone/models/` | the EnCodec weights' MIT notice; it must travel with them |
| `wasm/www/models/dac44_vocos.safetensors`, `dac44_vocos.json` | `src/gaim/web/static/picotone/models/` | the HQ decoder (Vocos with its low-tier post-filter, 63.75 MB), for `.pico` files of model `dac44` (the default `picotone encode` output) |
| `wasm/www/models/dac44_vocos.NOTICE.txt`, `dac44_vocos.credits.jsonl`, `DAC-MIT.txt` | `src/gaim/web/static/picotone/models/` | the HQ decoder's notice, its training credits (10.9 MB) and DAC's MIT licence; the notice requires that they travel with the weights |
| `wasm/www/models/cx2-tables.json`, `cx2_*.ecxm` | `src/gaim/web/static/picotone/models/` | the cx2 tables (7.4 MB stereo, 12.5 MB mono, 8.7 MB HQ `cx2_dac44.ecxm`) and their list |
| `wasm/ts/picotone.ts`, `pico-header.ts`, `protocol.ts` | `web/src/picotone/` | typed client API, bundled by esbuild into console/table |

`web/tools/build.ts` only prunes `static/build/`, so `static/picotone/` is left alone. The
three `.ts` files compile cleanly under gaim's `tsconfig.json` (`strict`,
`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `verbatimModuleSyntax`,
`isolatedModules`). This folder's `tsconfig.json` uses the same flags. If you would rather
not bundle, `wasm/www/picotone/picotone.js` + `picotone.d.ts` is the same module, compiled.
Only copy the model(s) you use; if you drop one, also drop its table and its entry in
`cx2-tables.json` (the worker loads only the tables of the models you load).

The worker, the wasm builds and the tables are fetched by URL at run time, so esbuild never
sees them. That is why `baseUrl` must be given when the API is bundled.

## 2. Server: cross-origin isolation headers

The threaded build needs `SharedArrayBuffer`, which browsers grant only to
**cross-origin isolated** pages. gaim loads nothing from other origins, so
`COOP: same-origin` plus `COEP: require-corp` is safe. Add this pure-ASGI middleware
(it also covers `StaticFiles`) to `create_app`'s `middleware=[...]`:

```python
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

class CrossOriginIsolation:
    """Sends COOP/COEP on every HTTP response, for the threaded audio decoder."""
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        async def send_isolated(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Cross-Origin-Opener-Policy"] = "same-origin"
                headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            await send(message)
        await self.app(scope, receive, send_isolated)
```

```python
middleware=[Middleware(_HostCheck, hosts=...), Middleware(CrossOriginIsolation)]
```

**MIME types.** `StaticFiles` uses `mimetypes.guess_type`. Python 3.15 maps `.wasm`
to `application/wasm`, which `WebAssembly.instantiateStreaming` requires. On Windows
the registry can override `mimetypes`, and `.safetensors` and `.ecxm` are unknown (Starlette
would fall back to `text/plain`), so pin them once at import time:

```python
import mimetypes
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".safetensors")
mimetypes.add_type("application/octet-stream", ".ecxm")
mimetypes.add_type("audio/ogg", ".pico")
```

**Caching.** `StaticFiles` sends `ETag`/`Last-Modified`, so a repeat load revalidates
with a cheap `304`. On top of that, the worker stores the model files and tables in Cache
Storage (`cacheName`, default `picotone-models-v1`), so they download once per browser. Change
`cacheName` when you re-export the models or tables. Cache Storage only exists in secure
contexts. On plain-http origins the API silently relies on the HTTP cache alone.

## 3. The networked table screen: serve gaim over HTTPS (recommended)

With `web.table = "network"`, the table is another device opening
`http://<lan-ip>:8765/table`. That origin is **not a secure context**: browsers ignore
COOP/COEP there, so `crossOriginIsolated` is false and `SharedArrayBuffer`, Cache
Storage and WebGPU are all unavailable. The decoder then runs single-threaded (see §5 for
whether that keeps up). **The supported fix is to serve gaim over HTTPS with a locally
trusted certificate.**

### 3.1 Create a local CA and a certificate (mkcert)

On the machine that runs gaim:

```powershell
winget install FiloSottile.mkcert        # or: choco install mkcert / scoop install mkcert
mkcert -install                          # creates a local CA and trusts it on this machine
mkcert -cert-file gaim.pem -key-file gaim-key.pem `
  localhost 127.0.0.1 ::1 192.168.1.50 gaim-host gaim-host.local gaim.lan
```

The SANs must cover every name the browsers type: `localhost`, `127.0.0.1`, the server's
**LAN IP**, its **hostname** (and `.local` name if you use mDNS), and **every entry in
`web.allowed_hosts`**. A name that is missing gives a certificate error on that device.
If the LAN IP changes (DHCP), re-issue the cert, or give the server a DHCP reservation.

### 3.2 Trust the CA on the table device too

The table device must trust the same local CA, or it sees a certificate error.
Run `mkcert -CAROOT` to find the folder and copy **`rootCA.pem` only**. Never copy
`rootCA-key.pem`, which can mint certificates for any site. Install it as a trusted root:

- **Windows:** `certutil -addstore -f ROOT rootCA.pem`, or double-click it and choose
  *Trusted Root Certification Authorities*.
- **macOS:** open it in Keychain Access, then set *Trust → Always Trust*.
- **Android:** *Settings → Security → Encryption & credentials → Install a certificate →
  CA certificate*.
- **iOS / iPadOS:** install the profile, then enable it under *Settings → General → About →
  Certificate Trust Settings*.
- **Linux Chrome:** `certutil -d sql:$HOME/.pki/nssdb -A -t C,, -n mkcert -i rootCA.pem`.
- **Firefox** (any OS) keeps its own store. Import the CA under *Settings → Privacy &
  Security → Certificates*, or set `security.enterprise_roots.enabled`.

Clicking through a certificate warning is **not** a substitute. An origin with a
certificate error is not treated as a properly secure origin: some features stay
disabled (for example, service workers will not register), and browsers do not promise
that cross-origin isolation, Cache Storage or WebGPU work there. Install the CA.

### 3.3 Give uvicorn the certificate

uvicorn takes `ssl_certfile` and `ssl_keyfile`. In `gaim/play.py`, `serve` builds
`uvicorn.Config(...)`. Pass them through, for example from two new `WebConfig` fields:

```python
uvicorn.Config(
    runtime.app,
    host=web.host,
    port=web.port,
    access_log=False,
    log_level="warning",
    ssl_certfile=web.ssl_certfile or None,   # e.g. "C:/gaim/certs/gaim.pem"
    ssl_keyfile=web.ssl_keyfile or None,     # e.g. "C:/gaim/certs/gaim-key.pem"
)
```

(Standalone equivalent: `uvicorn app:app --ssl-certfile gaim.pem --ssl-keyfile gaim-key.pem`.)
The printed console/table URLs (`urls(web)`) should then say `https://`.

### 3.4 What changes for gaim's own checks

- **Websockets become `wss://`.** `web/src/common.ts` already derives the scheme from
  `location.protocol` (`https:` → `wss:`), so the pages need no change.
- **Same-origin check.** `same_origin` maps a `wss` socket to the `https` scheme and
  compares the `Origin` (`https://<host>:8765`) with the socket's own `Host`. Both now
  carry `https` and the explicit port, so they match. On port 443, the default-port logic
  (`_default_port("https") == 443`) handles an `Origin` without a port.
- **Host check.** `_HostCheck` normalises `Host` with the request's scheme
  (`scope["scheme"] == "https"`). The names allowed are unchanged, but a browser that
  reaches the server by a name missing from the certificate fails at TLS, before gaim
  sees the request.

### 3.5 Verify on the table device

Open `https://<lan-ip>:8765/table` on the table device. In its devtools console (for
Android, use `chrome://inspect` from a desktop), check:

```js
isSecureContext        // true
crossOriginIsolated    // true
!!navigator.gpu        // true (WebGPU; also needs a supported GPU/browser)
```

If you also log `decoder.build`, it should read `kind: "threaded"` and
`reason: "cross-origin isolated: running N rayon threads"`.

### 3.6 Without HTTPS: the single-threaded fallback

Without HTTPS, everything still works single-threaded. The worker detects the
situation and reports it in `decoder.build`:

- `isSecureContext: false` → *"this page is not a secure context (plain http on an address
  other than localhost), so browsers ignore COOP/COEP and threads are unavailable; serve
  it over HTTPS …"*
- `isSecureContext: true, crossOriginIsolated: false` → *"this page is not cross-origin
  isolated; serve it with COOP … COEP …"* (the middleware is missing).

Show `decoder.build.reason` in the table's diagnostics so the cause is obvious. For a
one-off test you can also start Chrome with
`--unsafely-treat-insecure-origin-as-secure=http://192.168.1.50:8765 --user-data-dir=<tmp>`
(or `chrome://flags/#unsafely-treat-insecure-origin-as-secure`). That is a developer
switch, not a deployment option.

## 4. Usage

```ts
import { PicotoneDecoder } from "./picotone/picotone.js";

// One AudioContext for everything; 48 kHz matches the stereo model (the 24 kHz
// effects are resampled by Web Audio). Resume it from a user gesture.
const ctx = new AudioContext({ sampleRate: 48_000 });
const music = ctx.createGain();
const sfx = ctx.createGain();
music.connect(ctx.destination);
sfx.connect(ctx.destination);

const decoder = await PicotoneDecoder.create({ baseUrl: "/static/picotone/" });
if (decoder.build.kind !== "threaded") console.warn(`picotone: ${decoder.build.reason}`);
// Weights and cx2 tables download once, then come from Cache Storage (secure contexts) or the HTTP cache.
await Promise.all([decoder.loadModel("encodec_48khz"), decoder.loadModel("encodec_24khz")]);

async function bytesOf(url: string): Promise<Uint8Array> {
  return new Uint8Array(await (await fetch(url)).arrayBuffer());
}

// Music: streams, starts after the first ~1 s segment (or once enough is buffered if
// decoding is slower than realtime), then loops the whole file gaplessly.
const track = decoder.play(ctx, await bytesOf("/static/audio/tavern.pico"), {
  destination: music,
  loop: true,               // encoded with --loop, see "Gapless loops" below
  onUnderrun: (gap) => console.warn(`music underrun ${gap.toFixed(2)} s`),
});
// later: track.stop();

// Effects: decode once into an AudioBuffer, then play as often as needed.
const effects = new Map<string, Promise<AudioBuffer>>();
function playEffect(url: string): void {
  let buffer = effects.get(url);
  if (buffer === undefined) {
    buffer = bytesOf(url).then((b) => decoder.decodeToAudioBuffer(ctx, b));
    effects.set(url, buffer);
  }
  void buffer.then((b) => {
    const source = ctx.createBufferSource();
    source.buffer = b;
    source.connect(sfx);
    source.start();
  });
}
```

For a loop you would rather hold fully in memory, use
`const b = await decoder.decodeToAudioBuffer(ctx, bytes)`, then an
`AudioBufferSourceNode` with `loop = true` and no loop points.

To start a long track part-way (resume a scene), pass `startSample` to `decode()` (in samples
of the model's rate, after the trims). Only the pages from that point are decoded, and the
result is sample-identical to the same part of a whole decode. Loops and 24 kHz files decode
from the start and drop the earlier samples.

`readPicoHeader(bytes)` gives the model, length, tags (`TITLE=...`) and bitrate without the
worker, for example to pick the model to load.

### Gapless loops: encode with `--loop`

EnCodec does not reproduce the first and last few milliseconds of a file faithfully,
because of reflect padding and the segment edges. A plain file looped end-to-start therefore
**clicks at the seam**, even when the source loop was seamless (measurements in §5).
Encode loops with `--loop`:

```sh
picotone encode tavern-loop.wav tavern.pico --preset music6 --loop --tag TITLE=Tavern
```

The encoder codes the loop with 150 ms of its own end before it and 50 ms of its start
after it. It records the pre-skip, the end trim and the loop flag in the header.
The decoder drops the extra samples, so the decoded length is exactly the original. It also
cross-fades the decoded pre-roll into the last 100 ms, so the last sample runs into the
first the way the source did. Then just loop the whole file:

```ts
decoder.play(ctx, bytes, { loop: true });                // no loopStart / loopEnd
```

Do not pad loops by hand. Short effects need nothing.

### Which preset

| preset | file size (120 s music) | use |
|---|---|---|
| `--preset music24` | about 17.5 kbps (260 KB) | background music |
| `--preset music12` | about 8 kbps (120 KB) | music where size matters |
| `--preset music6` | about 3.2–4 kbps (55 KB) | ambience beds |
| `--fast --channel-format mono -b 6` | | effects, speech (24 kHz mono) |

`picotone encode` with no preset or `--fast` writes the HQ codec (44.1 kHz, `model: "dac44"`
in `readPicoHeader`), at 1 to 16 codebooks per channel (`--codebooks`, default 9; after
entropy coding about 0.6 kbps per channel at 1, 1.8 at 3, 3.9 at 6, 6.1 at 9, 8.1 at 12 and
10.6 at 16) in mono, left/right, mid/side or parametric stereo. This decoder plays
it: `await decoder.loadModel("dac44")` (the worker fetches `dac44_vocos.*` and the HQ
table), then `play` / `decode` / `decodeToAudioBuffer` as for any file:

```ts
const bytes = await bytesOf("/static/audio/theme.pico");
const info = readPicoHeader(bytes);       // info.model "dac44", info.layout "ms", info.codebooks [9, 9]
await decoder.loadModel(info.model);      // dac44_vocos.* and cx2_dac44.ecxm, once per worker
const theme = decoder.play(ctx, bytes, { destination: music, loop: info.loop });
const ms = await theme.started;           // about 0.8 s at q1-q5, 0.2-0.3 s above
```

HQ output is 44.1 kHz, so a 48 kHz `AudioContext` resamples it. At 1 to 5 codebooks the
decoder also runs a post-filter on its spectrum and a declicker that repairs the clicks left
on the frame grid; every decode, HQ and EnCodec, ends in a look-ahead peak limiter. They are
part of the decode and need nothing from gaim. The declicker looks up to 2 s ahead, but in a
file's first 2 s only as far as the decode has gone, so first audio at q1–q5 comes after about
0.8 s (0.2–0.3 s at q6 and above). HQ decodes at about 4.2–4.7× realtime for single-threaded
stereo, the low end at q1–q5 (§5), so it suits plain-HTTP table pages too.
`readPicoHeader` also gives HQ files' `layout`, per-channel `codebooks` and `codebookHash`.
The WebGPU decoder in `web/` refuses HQ files.

### HQ enhancement layers (`.picol`): play small, upgrade later

An HQ file can come in two parts: a small `.pico` that plays on its own, and a `.picol` layer
with the rest of the same encode's codebooks. With both, the decode equals a one-file encode
of all the codebooks, sample for sample. The base's codebook count is free:

```sh
picotone encode song.flac song.pico --codebooks 5 --layer-codebooks 16
# song.pico: q5 alone (about 3.3 kbps per channel); song.picol: codebooks 6-16
```

The two parts cost 1-3% more than one q16 file (about 6% over a parametric-stereo base).
`--layer-side-codebooks` sets the side's count for mid/side; `--layer-codebooks 10` is
refused (use 9 or 11-16). `picotone decode` and `info` pick up `song.picol` next to `song.pico`;
`--no-layer` leaves it out.

In the browser, pass the layer's bytes as `layer` to `decode` / `play` / `decodeToAudioBuffer`:

```ts
const base = await (await fetch("song.pico")).arrayBuffer();
const startedAt = ctx.currentTime;                     // roughly; onStart gives the exact start
let playback = decoder.play(ctx, base);                 // starts at once
const layer = await (await fetch("song.picol")).arrayBuffer();
// Upgrade: restart from the current position with the layer.
const at = Math.floor((ctx.currentTime - startedAt) * 44100);
playback.stop();
playback = decoder.play(ctx, base, { layer, startSample: at });
```

A layer that is not the file's, or is damaged, is left out and the file plays alone; the
result's `stats.layered` is then false and `stats.layerError` says why. There is no in-place
upgrade inside one decode: restart from the position as above (an HQ seek decodes from a few
frames before it, so the join is the decoder's own).

An HQ file names its code space, not a decoder; the Vocos network (`dac44_vocos`) decodes
every `dac44` file at every codebook count. It needs about 100 MiB of wasm memory
single-threaded (§5).

## 5. Performance and caveats

Measured with `node wasm/test/run.mjs` (16 logical cores, headless Chrome, the threaded
build limited to 8 threads with `PICOTONE_TEST_THREADS=8`, lightly loaded). Speed is audio
seconds ÷ decode wall time, including the cx2 decode; SNR is against the native
`decode --float`.

| case | threaded (8) | first audio | single, relaxed SIMD | single, plain | first audio | SNR |
|---|---|---|---|---|---|---|
| mono effect 1.2 s | 17.7× | 0.11 s | 10.1× | 9.2× | 0.16 s | 123 dB |
| mono music 75 s (`-b 6`) | 22.2× | 0.12 s | 12.2× | 10.2× | 0.16–0.18 s | 119 dB |
| stereo 8 s, music24 | 9.1× | 0.16 s | 5.5× | 4.8× | 0.23–0.26 s | 125 dB |
| stereo 8 s, music6 | 11.8× | 0.20 s | 6.4× | 4.9× | 0.25–0.28 s | 125 dB |
| stereo music 75 s (`-b 6`) | 15.8× | 0.20 s | 6.8× | 5.6× | 0.27–0.31 s | 124 dB |
| stereo 120 s, music12 | 11.9–12.4× | 0.18–0.20 s | 5.9–6.0× | 5.1× | 0.25–0.27 s | 104–118 dB |

**HQ (`dac44`), the Vocos decoder**, same machine, all 16 threads for the threaded build, the
trained weights, 30 s of music, with the post-filter and declicker at q1–q5 and the limiter:

- **Speed:** single-threaded stereo about 4.2–4.7× realtime, the low end at q1–q5; threaded
  about 9–12×. Mono q9 decodes at about 12× single-threaded and 20× threaded. The network takes
  86–94% of the decode time, the cx2 decode the rest.
- **First audio:** about 0.8 s at q1–q5 (the declicker's look-ahead, see §4), 0.2–0.3 s at q6
  and above.
- **SNR** against the native `decode --float`: 113–117 dB, seeks included.

**Memory.** wasm memory never shrinks, so its peak is what a worker keeps. The weights are not
copied into wasm: `new Decoder(name, config, bytes)` takes the `Uint8Array` and reads each
tensor from it as it packs it, so the load holds only the packed network (plus one tensor).
Drop the buffer after the constructor returns. Peaks measured with `node wasm/test/memory.mjs`
(30 s q9 HQ files, 8 s / 6 s EnCodec; 16 threads):

| network | threaded: built / peak | single: built / peak |
|---|---|---|
| HQ, Vocos (stereo / mono) | 98 / 203, 159 MiB | 66 / 101, 94 MiB |
| EnCodec 48 kHz stereo | 71 / 172 MiB | 40 / 100 MiB |
| EnCodec 24 kHz mono | 79 / 129 MiB | 47 / 79 MiB |

"Built" includes the empty instance (34 MiB threaded with its thread stacks, 1 MiB single);
the cx2 tables add 14–32 MiB, and decoding the rest. It does not depend on file length.

No EnCodec case had an underrun. Model download and load take 0.3–0.5 s from a local server.

**Verdict for music loops.** With the threaded build (localhost, or HTTPS per §3) music
starts after about 0.2 s and stays far ahead of realtime. The single-threaded build, all the
plain-HTTP network table gets, decodes 48 kHz stereo at about 5–7× realtime here: ahead of
realtime, but a slow or busy device has less margin. When it falls behind, `play()`'s auto
prebuffer waits until it can play through. On a weak table device, decode music ahead of
time (`decodeToAudioBuffer` at scene load, and keep the result), or serve over HTTPS (§3),
which is the real fix. Effects should be decoded ahead and cached as `AudioBuffer`s anyway.

**Loop seam.** A plain file looped end-to-start clicks. In the test's 4 s stereo loop, the
jump across the seam is 0.236, 25× the 99.9th-percentile step (0.0096). With `--loop` it is
0.0053, below the 99.9th-percentile step (0.0102).

**Other caveats.**
- The first chunk is one unit (one segment, one second of mono frames, or one HQ page of
  80 frames, 0.93 s). Threaded, that is about 0.4–2 s.
- HQ seeks (`startSample`) decode from a few frames before the point; HQ loops decode from
  the start. Either way the samples equal a whole-file decode.
- A loaded model stays in wasm memory for the worker's lifetime. The worker loads each
  model once; `dispose()` terminates it.
- A file decodes only with the cx2 table it was coded with. The worker refuses others with
  the missing hash; files from `picotone encode` with its default tables always match the
  exported ones.
- The output gain field of the format is read but not applied.
