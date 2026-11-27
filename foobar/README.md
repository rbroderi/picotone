# foo_input_pico: picotone in foobar2000

A foobar2000 input component that plays picotone's `.pico` files on Windows:

- HQ files at any codebook count and stereo layout, and EnCodec files (48 kHz stereo, 24 kHz mono).
- An HQ file's `.picol` enhancement layer, when it has the same name next to the file (`song.pico` +
  `song.picol`). A layer that belongs to another file is ignored, with a line in foobar2000's console.
- Seeking, the file's tags, and the codec, profile (`q5`, `q16 mid/side (with .picol)`, ...) and bitrate in the
  properties. The bitrate counts the layer when it is used.

Tags are read-only: the component does not write tags to `.pico` files.

## Layout

| Path | What it is |
|---|---|
| `picotone-ffi/` | A small C interface (`pt_init`, `pt_header`, `pt_tags`, `pt_open`, `pt_decode`, `pt_close`) to picotone's `player::Player`, built as a static library |
| `picotone-ffi/examples/ptcheck.rs` | Checks the C interface on a file: header, a whole decode, and seeks that must match the whole decode exactly |
| `foo_input_pico/` | The component (C++, foobar2000 SDK) and `picotone.h`, the C header |
| `build.ps1` | Builds both platforms and packs `build/foo_input_pico.fb2k-component` |
| `sdk/` | The foobar2000 SDK, which you download (not in the repository) |

## Building

You need:

- Rust, with the `i686-pc-windows-msvc` and `x86_64-pc-windows-msvc` targets (`rustup target add ...`).
- Visual Studio 2022 or its Build Tools, with the C++ workload (toolset v143).
- The foobar2000 SDK from <https://www.foobar2000.org/SDK>, unpacked into `foobar/sdk`, so that
  `foobar/sdk/foobar2000/SDK/foobar2000.h` exists.

Then, from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File foobar\build.ps1            # build and pack
powershell -ExecutionPolicy Bypass -File foobar\build.ps1 -Install   # also install into the foobar2000 profile
```

`-Install` copies the component into `%APPDATA%\foobar2000-v2\user-components` (32-bit foobar2000) or
`user-components-x64` (64-bit), for whichever is installed; restart foobar2000 afterwards. Otherwise, install the
`.fb2k-component` from foobar2000's Preferences > Components page.

The package holds both DLLs (32-bit at the root, 64-bit in `x64/`), the decoder weights (HQ's `dac44_vocos`,
EnCodec's 48 kHz and 24 kHz models, loaded decoder-only), every cx2 table, and the weights' notices and credits.

## How it finds its data

The component looks for the `weights` and `models` folders next to its DLL, then one folder up (the 64-bit DLL
sits in `x64/`). The environment variables `PICOTONE_WEIGHTS` and `PICOTONE_TABLES` override both. The tables load
when the first file plays, and each network loads the first time a file needs it. Scanning files into the library
reads only their headers and loads no network.

## Checking the C interface

```powershell
cargo run --release --manifest-path foobar\picotone-ffi\Cargo.toml --example ptcheck -- weights models song.pico [song.picol]
```

It prints the header and the decode speed, and fails if the decode is the wrong length, if two decodes differ,
or if a seek's samples differ from the whole decode's.

## Licence

The component (everything in `foobar/`, and the `foo_input_pico.dll` it builds, which includes picotone's library)
is licensed under the [GNU Lesser General Public License v3.0](COPYING.LESSER) or later, so it can link the
foobar2000 SDK (its own licence is in `sdk/sdk-license.txt`) and run inside foobar2000, which is closed source. The
LGPL adds permissions to the GPL, whose text is in [`LICENSE`](../LICENSE). The weights carry their own notices,
which the package includes.
