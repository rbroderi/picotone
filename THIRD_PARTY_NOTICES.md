# Third-party notices and attributions

picotone's own code is licensed under the [GNU General Public License v3.0 or later](LICENSE), with the
exceptions under [picotone's licences](#picotones-licences). It builds on the work below. Everything here is
compatible with GPL-3.0.

## picotone's licences

| Part | Licence |
|---|---|
| The library, the `picotone` command, the cx2 coder and tables, the browser decoders (`wasm/`, `web/`), the Python tooling (`tools/`, `train/`) | [GPL-3.0-or-later](LICENSE) |
| The C library (`ffi/`) and the foobar2000 component (`foobar/`), and the DLL they build with picotone's library inside | [LGPL-3.0-or-later](foobar/COPYING.LESSER) (the LGPL adds permissions to the GPL) |
| The FFmpeg demuxer and muxer (`ffmpeg/libavformat/libpicotone*`) | LGPL-2.1-or-later, like FFmpeg's. picotone itself is GPL-3.0-or-later, so an FFmpeg built with `--enable-libpicotone` needs `--enable-gpl --enable-version3`, and its programs are GPLv3 |
| The model weights in `weights/` | their own notices: see the EnCodec and DAC sections below |

The foobar2000 component builds against the **foobar2000 SDK** (Copyright (c) Peter Pawlowski), which is
downloaded into `foobar/sdk/` and is not in the repository. The SDK is under a BSD-style licence
(`foobar/sdk/sdk-license.txt`: redistributions keep its copyright notice, conditions and disclaimer, and the
author's name may not be used to endorse derived products); its `pfc` and `libPPUI` libraries are under a
zlib-style licence (`pfc-license.txt`, `libPPUI-license.txt`). A distributed component must carry those
notices.

## EnCodec (Meta), MIT

picotone implements the EnCodec neural audio codec and runs its pre-trained weights:

- **Code:** https://github.com/facebookresearch/encodec, MIT.
- **Weights:** `encodec_24khz` and `encodec_48khz`, from https://huggingface.co/facebook/encodec_24khz and
  https://huggingface.co/facebook/encodec_48khz. Their model cards give the licence as MIT.

> Copyright (c) Meta Platforms, Inc. and affiliates. Released under the MIT License.

The full text is in [`LICENSES/EnCodec-MIT.txt`](LICENSES/EnCodec-MIT.txt).

**Every copy of the weights must carry that notice.** That covers `weights/` and the converted browser models
(`web/public/models/`, `wasm/www/models/`, and any site that serves them).

The weights are stored unmodified in `weights/` with Git LFS, with the notice beside them in
`weights/EnCodec-MIT.txt`.

picotone is an independent project. It is not made, endorsed or supported by Meta. "EnCodec" is used only to
name the model it implements.

If you use this work in research, please cite the EnCodec paper:

> A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio Compression.* 2022.
> arXiv:2210.13438.

## DAC (Descript), MIT

The HQ codec codes DAC 44 kHz's codes, and its weights derive from or are trained on DAC's:

- **Code:** https://github.com/descriptinc/descript-audio-codec (descript-audio-codec 1.0.0), MIT. The HQ
  training code in `train/hq/` re-implements DAC's losses and discriminators.
- **Weights:** DAC's 44 kHz model. `dac44_quantizer` holds DAC's quantiser (its 9 codebooks, plus 7 that
  picotone trained on DAC's residual); `dac44_stftenc` is trained to feed that quantiser, matching DAC's encoder
  first; `dac44_vocos` is a Vocos decoder trained on DAC 44 kHz codes. All three are derivatives of DAC's MIT
  weights.

> Copyright (c) 2023-present, Descript. Released under the MIT License.

The full text is in [`weights/DAC-MIT.txt`](weights/DAC-MIT.txt). **Every copy of the HQ weights must carry it**,
with the weights' own `.NOTICE.txt` files; the WebAssembly exporter copies them next to `dac44_vocos` in
`wasm/www/models/`.

picotone is not made, endorsed or supported by Descript. If you use this work in research, please cite:

> R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, K. Kumar. *High-Fidelity Audio Compression with Improved
> RVQGAN.* 2023. arXiv:2306.06546.

## Vocos

picotone's HQ decoder follows the Vocos architecture (a ConvNeXt backbone that predicts an STFT, then an
inverse STFT). picotone implements it independently and trains its own weights; no Vocos code or weights are
used. Vocos is by Hubert Siuzdak (https://github.com/gemelo-ai/vocos, MIT):

> H. Siuzdak. *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality
> audio synthesis.* 2023. arXiv:2306.00814.

## Audio used to train the HQ networks

The HQ networks (`dac44_stftenc`, `dac44_vocos` with its post-filter, and the 7 extra codebooks of
`dac44_quantizer`) were trained on 44,042 third-party tracks licensed CC BY (9,423), CC BY-SA (32,630), the
Free Art Licence (663), CC0 (901) or public domain (425). Every track is credited, with its artist, title,
source URL and licence URL, in the `.credits.jsonl` next to each weights file
(`weights/dac44_quantizer.credits.jsonl`, `dac44_stftenc.credits.jsonl`, `dac44_vocos.credits.jsonl`), and
each weights file's `.NOTICE.txt` gives the counts.

**ShareAlike.** Whether trained weights are an adaptation of their training data is legally unsettled. Since
some of the audio is CC BY-SA or Free Art Licence, the HQ weights may be a ShareAlike derivative of it. Ship
each weights file with its `.NOTICE.txt` and `.credits.jsonl`. None of this audio is in the repository.

## Reference implementation used for testing

Hugging Face `transformers` (`EncodecModel`, Apache-2.0) was used as the reference in parity tests: codes and
decoder output are compared against it. It is not bundled or linked.

## Rust dependencies

The Rust crates picotone builds with are MIT, Apache-2.0, or dual-licensed under MIT and Apache-2.0 (with a few
also under BSD-3-Clause, Zlib, Unlicense or Unicode-3.0). The exception is **Symphonia** (audio file decoding,
the `audio-io` feature), which is **MPL-2.0**.

MPL-2.0 is compatible with GPL-3.0. picotone uses Symphonia unmodified from crates.io, where its source is
available: https://github.com/pdeljanov/Symphonia.

The **ogg** crate (BSD-3-Clause) is a dev-dependency only: the format tests use it as an independent Ogg reader.
It is not part of any build of picotone.

Distributed binaries should ship these crates' licence texts. Generate them with
[`cargo about`](https://github.com/EmbarkStudios/cargo-about), or with `cargo metadata` for the list.

## Browser tooling

- **wasm-bindgen** and **wasm-bindgen-rayon:** MIT or Apache-2.0. Every WebAssembly build bundles the
  wasm-bindgen glue (the WebGPU decoder too, for its cx2 decode); the threaded build also bundles
  wasm-bindgen-rayon.
- **coi-serviceworker:** MIT. It is suggested in `wasm/README.md` for GitHub Pages. If you copy it into a site,
  keep its licence header.
- **npm dev dependencies** (TypeScript, puppeteer-core, `@webgpu/types` and others) are build and test tools
  only, and are not shipped.

## Formats and specifications

The `.pico` container follows two specifications:

- **Ogg:** RFC 3533.
- **Ogg Opus:** RFC 7845. picotone follows it for pre-skip, granule positions and Vorbis-style comments.

picotone implements them independently; no Xiph.Org code is used. The Ogg specification is free for anyone to
use and implement. "Ogg" is a trademark of the Xiph.Org Foundation; picotone uses it only to describe the
container, and is not certified or endorsed by Xiph.Org.

## Audio used to fit and test the entropy coder

The tables and the small neural network in picotone's entropy coder (cx2, the `.ecxm` files in `models/`)
were fitted on audio that is public domain or CC0 only. The source items are on archive.org, and each is listed
with its URL and licence in [`DATA_SOURCES.json`](DATA_SOURCES.json). No attribution is legally required. With thanks to:

- **Kimiko Ishizaka:** *The Open Goldberg Variations* (CC0).
- **Musopen and Aaron Dunn:** *The Complete Chopin Collection*, and the Brahms symphony recordings.
- **The United States Navy Band, Air Force Band and Marine Band:** marches, waltzes and orchestral works
  (public domain, as US government works).
- **The Genetic Trance netlabel artists** (the `[GT…]` releases), including Timur Dzhafarov (The Rogue Sources) and
  Joachim Rontxelius (Noisesurfer).
- **LibriVox volunteer readers:** public-domain audiobooks of Twain, Melville, Stoker, Austen, Dickens, Doyle,
  Shakespeare and others.

None of this audio is included in the repository.

## The sample song

`samples/` holds **Ravel – Boléro** by **Jason Weinberger**
(<https://soundcloud.com/jasonweinberger/ravel-bolero>), licensed
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), with picotone encodings of it. The encodings
are adaptations of the recording and are licensed CC BY-SA 3.0 as well. See
[`samples/README.md`](samples/README.md).
