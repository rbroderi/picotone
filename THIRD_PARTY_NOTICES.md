# Third-party notices and attributions

picotone's own code is licensed under the [GNU General Public License v3.0 or later](LICENSE). It builds on
the work below. Everything here is compatible with GPL-3.0.

## EnCodec (Meta), MIT

picotone implements the EnCodec neural audio codec and runs its pre-trained weights:

- **Code:** https://github.com/facebookresearch/encodec, MIT.
- **Weights:** `encodec_24khz` and `encodec_48khz`, from https://huggingface.co/facebook/encodec_24khz and
  https://huggingface.co/facebook/encodec_48khz. Their model cards give the licence as MIT.

> Copyright (c) Meta Platforms, Inc. and affiliates. Released under the MIT License.

The full text is in [`LICENSES/EnCodec-MIT.txt`](LICENSES/EnCodec-MIT.txt).

**Every copy of the weights must carry that notice.** That covers `weights/` and the converted browser models
(`web/public/models/`, `wasm/www/models/`, and any site that serves them).

The weights are not stored in this repository.

picotone is an independent project. It is not made, endorsed or supported by Meta. "EnCodec" is used only to
name the model it implements.

If you use this work in research, please cite the EnCodec paper:

> A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio Compression.* 2022.
> arXiv:2210.13438.

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
