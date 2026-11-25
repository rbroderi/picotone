# Integrating picotone-wasm into gaim

How to play picotone `.pico` music loops and sound effects in gaim's console and table
pages, decoded in the browser by the WebAssembly decoder in this folder.

**What changed from the encodecx version of this guide:** files are `.pico` (not `.ecdc`),
the static folder is `picotone/` (not `encodecx/`), the module is `picotone.js`, the class
`PicotoneDecoder` is now `PicotoneDecoder` (and `PicotoneOptions` is `PicotoneOptions`),
`readEcdcHeader` is now `readPicoHeader`, and the worker also loads the cx2 tables listed in
`models/cx2-tables.json`. Re-encode every asset with `picotone encode`; old `.ecdc` files no
longer decode.

**gaim must re-copy everything in §1** (the worker and its modules, all three `pkg-*` builds,
the models, `EnCodec-MIT.txt`, the tables and `cx2-tables.json`, and the three `.ts` files),
then rename `PicotoneDecoder` to `PicotoneDecoder` in its own code and bump `cacheName` if it
sets one. Mixing old and new files does not work. (If gaim ever uses the WebGPU decoder
from `web/`, its class is now `PicotoneWebGpu` and its models must be re-exported: their
manifests now say `picotone-webgpu-weights`.)

**HQ (M3):** to play HQ files, gaim must also re-copy the worker, `protocol.js`,
`pico-header.js`, all three `pkg-*` builds and the three `.ts` files (the bindings changed),
plus `DAC-MIT.txt`, `cx2_dac44.ecxm` and the new `cx2-tables.json` (§1). `DecodeStats` now
also has `cx2Ms` and `networkMs`.

**Vocos is the only HQ decoder:** the distilled "student" decoder (`dac44_student.*`) is gone.
Re-copy the worker, `protocol.js`, all three `pkg-*` builds and the three `.ts` files, copy
`dac44_vocos.*` (§1), and delete any `dac44_student.*` from gaim's `models/`. `loadModel` no
longer takes `hqDecoder`, and the `HqDecoderName` type is gone; `ModelInfo.network` is
`dac44_vocos` for `dac44`.

## 1. Files to copy

Build first (`powershell -File wasm\build.ps1`; it compiles the wasm builds and the `.js`
files of `www/picotone/`, none of which are in git, and exports `www/models/`), then copy:

| from (encodecx repo) | to (gaim) | how it is used |
|---|---|---|
| `wasm/www/picotone/worker.js`, `protocol.js`, `pico-header.js` | `src/gaim/web/static/picotone/` | dedicated decoder worker, loaded by URL |
| `wasm/www/picotone/pkg-st/` (whole folder) | `src/gaim/web/static/picotone/pkg-st/` | single-threaded build |
| `wasm/www/picotone/pkg-st-fma/` (whole folder) | `src/gaim/web/static/picotone/pkg-st-fma/` | single-threaded build with relaxed SIMD, used instead of `pkg-st` where the browser has it (about 15–20% faster) |
| `wasm/www/picotone/pkg-mt/` (whole folder, incl. `snippets/`) | `src/gaim/web/static/picotone/pkg-mt/` | threaded build + rayon pool worker |
| `wasm/www/models/*.json`, `*.safetensors` | `src/gaim/web/static/picotone/models/` | decoder-only weights (63.4 MB mono, 46.6 MB stereo) |
| `wasm/www/models/EnCodec-MIT.txt` | `src/gaim/web/static/picotone/models/` | the weights' MIT notice; it must travel with them |
| `wasm/www/models/dac44_vocos.safetensors`, `dac44_vocos.json` | `src/gaim/web/static/picotone/models/` | the HQ decoder (Vocos, 56 MB), for `.pico` files of model `dac44` (the default `picotone encode` output) |
| `wasm/www/models/dac44_vocos.NOTICE.txt`, `dac44_vocos.credits.jsonl`, `DAC-MIT.txt` | `src/gaim/web/static/picotone/models/` | the HQ decoder's notice, its training credits (10.9 MB) and DAC's MIT licence; the notice requires that they travel with the weights |
| `wasm/www/models/cx2-tables.json`, `cx2_*.ecxm` | `src/gaim/web/static/picotone/models/` | the cx2 tables (7.4 MB stereo, 12.5 MB mono, 4.9 MB HQ `cx2_dac44.ecxm`) and their list |
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

Do not pad loops by hand any more. Short effects need nothing.

### Which preset

| preset | file size (120 s music) | use |
|---|---|---|
| `--preset music24` | about 17.5 kbps (260 KB) | background music |
| `--preset music12` | about 8 kbps (120 KB) | music where size matters |
| `--preset music6` | about 3.2–4 kbps (55 KB) | ambience beds |
| `--fast --channel-format mono -b 6` | | effects, speech (24 kHz mono) |

`picotone encode` with no preset or `--fast` writes the HQ codec (44.1 kHz, `model: "dac44"`
in `readPicoHeader`), at 1 to 9 codebooks per channel (`--codebooks`, about 0.86 kbps each
before entropy coding) in mono, left/right, mid/side or parametric stereo. This decoder plays
it: `await decoder.loadModel("dac44")` (the worker fetches `dac44_vocos.*` and the HQ
table), then `play` / `decode` / `decodeToAudioBuffer` as for any file. HQ output is 44.1 kHz,
so a 48 kHz `AudioContext` resamples it. It decodes at about 4.4–4.7× realtime for
single-threaded stereo (§5), so it suits plain-HTTP table pages too.
`readPicoHeader` also gives HQ files' `layout` and per-channel `codebooks`. The WebGPU
decoder in `web/` refuses HQ files for now.

An HQ file names its code space, not a decoder; the Vocos network (`dac44_vocos`) decodes
every `dac44` file at every codebook count. The same holds on the encoding side: `picotone encode`
makes HQ codes with picotone's STFT encoder in front of DAC's quantiser, and the files are the
same format and code space as before, so nothing changes in the decoder or its weights. It needs about 77 MiB of wasm memory
single-threaded.

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

**HQ (`dac44`), the Vocos decoder**, same machine, all 16 threads for the threaded build, 30 s
of music. cx2 is the entropy decode, net the network (with stereo and trim). Speeds were
measured with random-initialised stand-in weights (the speed does not depend on the values);
the SNR is that of the trained weights:

| case | threaded (16) | single, relaxed SIMD | single, plain | SNR |
|---|---|---|---|---|
| stereo q5 (L/R) | 13.0× | 4.7× (cx2 0.011 s, net 0.095 s per ch-s) | 3.8× | 116.7 dB |
| stereo q9 (L/R) | 12.4× | 4.4× (cx2 0.015 s, net 0.098 s per ch-s) | 3.6× | 116.9–117.0 dB |
| mono q9 | 18.0× | 8.9× | 6.6× | 117.1–117.2 dB |
| 2 single-threaded workers, one mono q9 file each, at once | | 15.7 channel-s/s | | |

wasm memory peaks at 71–77 MiB single-threaded and 125–151 MiB threaded (the weights are
56 MB, packed 63 MiB).

*History:* the retired distilled "student" decoder (`dac44_student`, DAC's decoder
architecture at width 512) measured, per channel-second of coded audio:

| case | threaded (16) | single, relaxed SIMD | single, plain | SNR |
|---|---|---|---|---|
| stereo q5 (L/R) | 3.9× | 0.83× (cx2 0.011 s, net 0.59 s) | 0.64× (cx2 0.012 s, net 0.77 s) | 115 dB |
| stereo q9 (L/R) | 4.1× | 0.84× (cx2 0.015 s, net 0.58 s) | 0.68× (cx2 0.018 s, net 0.72 s) | 116 dB |
| mono q9 | 7.5× | 1.45× | 1.29× | 117 dB |
| 2 single-threaded workers, one mono q9 file each, at once | | 2.88 channel-s/s (1.44× for stereo) | | |

Its network dominated its decode (98%) and single-threaded stereo was slower than realtime,
which is why Vocos (about 5× faster) replaced it. Threaded, it used about 200–220 MiB of wasm
memory; single-threaded 102 MiB.

**Memory.** wasm memory never shrinks, so its peak is what a worker keeps. The weights are not
copied into wasm: `new Decoder(name, config, bytes)` takes the `Uint8Array` and reads each
tensor from it as it packs it, so the load holds only the packed network (plus one tensor).
Drop the buffer after the constructor returns. Peaks measured with `node wasm/test/memory.mjs`
(30 s q9 HQ files, 8 s / 6 s EnCodec; 16 threads):

| network | threaded: built / peak | single: built / peak |
|---|---|---|
| HQ, Vocos (stereo / mono) | 97 / 151, 125 MiB | 64 / 77, 71 MiB |
| EnCodec 48 kHz stereo | 71 / 169 MiB | 40 / 100 MiB |
| EnCodec 24 kHz mono | 79 / 127 MiB | 47 / 79 MiB |

"Built" includes the empty instance (34 MiB threaded with its thread stacks, 1 MiB single);
the cx2 tables add 4–32 MiB, and decoding the rest. It does not depend on file length.

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
