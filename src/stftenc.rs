//! The trained HQ encoder (`dac44_stftenc`): mono 44.1 kHz audio to the
//! 1024-wide latent `z` that DAC 44 kHz's quantiser turns into codes.
//!
//! It replaces DAC's own encoder (about 30.6 GMAC per second of audio) with
//! a Vocos-style network at the latent frame rate (about 1.3 GMAC/s). The
//! network is `STFTEncoder` in `distill/encoder.py`, converted by
//! `tools/export_stftenc.py`:
//!
//! - **The features.** Per resolution `n_fft` `w` (2048 and 512), hop
//!   `min(512, w / 4)` and zero padding `(w - hop) / 2` on each side without
//!   centring, so the `512 / hop` STFT frames of latent frame `f` are centred
//!   inside its samples `[512 f, 512 f + 512)`. Periodic Hann window. Each
//!   bin becomes `S |S|^(c - 1)` (`|S|^2 + 1e-8` under the power, `c = 0.3`),
//!   real parts then imaginary parts. A latent frame's sub-hop STFT frames
//!   follow each other in the channels (sub-frame `i`, value `v` at channel
//!   `i * C + v` within its resolution, `C = 2 (w / 2 + 1)`), and the
//!   resolutions follow each other: 2050 + 4 x 514 = 4106 channels.
//! - **The network.** A k1 conv to `dim` channels and a `LayerNorm`, the
//!   Vocos decoder's `ConvNeXt` blocks ([`crate::vocos`]), a final
//!   `LayerNorm` (folded into the output Linear at load) and a Linear to the
//!   latent.
//!
//! The features are computed in double precision (the FFT included) and
//! rounded once: the compression magnifies the rounding of quiet bins, and
//! the transforms are a small part of the work.
//!
//! **Length.** The network maps `T` samples to `T / 512` frames for `T` a
//! multiple of 512; this module takes any length and reads zeros past the
//! end, as if the caller had zero padded the audio to `ceil(T / 512)`
//! frames.
//!
//! **Chunking.** A frame's latent depends on the features within
//! `3 * blocks` frames of it (24 for 8 blocks: 3 per k7 depthwise conv), and
//! a frame's features on the samples from 768 before its first sample to
//! 768 after its last (for the 2048 window). The features of a frame read
//! the signal itself (zeros only outside the whole signal), and every other
//! value is computed per frame by the same operations wherever the frame
//! sits in a window, so any chunking equals a whole-signal encode bit for
//! bit.

use std::cell::RefCell;
use std::ops::Range;
use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};
use rayon::prelude::*;

use crate::dac::HOP;
use crate::fft::{RealForward, Scratch64};
use crate::kernels::{self, PackedMat};
use crate::nn::Seq;
use crate::vocos::{Block, TILE, fold_linear};
use crate::weights::Weights;

/// The encoder's name; its weight files are `dac44_stftenc.*`.
pub const NAME: &str = "dac44_stftenc";
/// `LayerNorm` epsilon.
const EPS: f32 = 1e-6;
/// Added to `|S|^2` under the compression's power (as `float32`, like
/// PyTorch's).
const POWER_EPS: f32 = 1e-8;
/// Most frames one network call encodes (before context).
pub const PIECE: usize = 1024;

/// One STFT resolution of the features.
#[derive(Debug, Clone)]
struct Resolution {
    /// Window length `n_fft`.
    n_fft: usize,
    /// STFT hop, `min(512, n_fft / 4)`.
    hop: usize,
    /// STFT frames per latent frame, `512 / hop`.
    sub: usize,
    /// Zero padding before the first sample, `(n_fft - hop) / 2`.
    pad: usize,
    /// The periodic Hann window, as PyTorch's `float32` values.
    window: Vec<f64>,
    fft: RealForward,
}

impl Resolution {
    /// Channels of one STFT frame: real and imaginary parts of each bin.
    const fn width(&self) -> usize {
        2 * (self.n_fft / 2 + 1)
    }

    /// Samples after the end of a latent frame that its features read.
    const fn reach_after(&self) -> usize {
        ((self.sub - 1) * self.hop + self.n_fft).saturating_sub(self.pad + HOP)
    }
}

/// The STFT encoder.
#[derive(Debug, Clone)]
pub struct StftEncoder {
    dim: usize,
    latent: usize,
    channels: usize,
    /// `(c - 1) / 2`, the power of `|S|^2 + 1e-8`.
    exponent: f64,
    resolutions: Vec<Resolution>,
    /// Samples before a latent frame's first sample that its features read.
    reach_before: usize,
    /// Samples after a latent frame's last sample that its features read.
    reach_after: usize,
    /// The k1 embed conv, `[channels, dim]`.
    embed: PackedMat,
    embed_bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    blocks: Vec<Block>,
    /// The output Linear with the final `LayerNorm` folded in,
    /// `[dim, latent]`.
    out: PackedMat,
    out_bias: Vec<f32>,
}

/// The samples a window of frames reads: `samples[i]` is sample
/// `start + i` of a signal of `len` samples, zero outside it.
#[derive(Debug, Clone, Copy)]
struct Signal<'a> {
    samples: &'a [f32],
    start: usize,
    len: usize,
}

impl Signal<'_> {
    /// Sample `i - offset` (zero before the start or past the end of the
    /// signal), so that positions before sample 0 need no signed numbers.
    fn at(&self, i: usize, offset: usize) -> f64 {
        i.checked_sub(offset)
            .filter(|&s| s < self.len)
            .and_then(|s| s.checked_sub(self.start))
            .and_then(|s| self.samples.get(s))
            .map_or(0.0, |&v| f64::from(v))
    }
}

impl StftEncoder {
    /// Builds the encoder from `tools/export_stftenc.py`'s tensors; the sizes
    /// come from the tensors' shapes.
    ///
    /// # Errors
    /// Fails when a tensor is missing or has the wrong shape, or the
    /// features do not fit the embed conv.
    pub fn from_weights(w: &Weights<'_>) -> Result<Self> {
        let first = |name: &str| -> Result<usize> {
            w.shape(name)
                .and_then(<[usize]>::first)
                .copied()
                .with_context(|| format!("checkpoint has no {name}: not an STFT encoder"))
        };
        let dim = first("embed.bias")?;
        let mlp = first("block0.pw1.bias")?;
        let latent = first("out.bias")?;
        let power = w.get("power", &[1])?[0];
        ensure!(
            power > 0.0 && power <= 1.0,
            "compression power {power} is out of range"
        );
        let mut resolutions = Vec::new();
        while let Some(&[n_fft]) = w.shape(&format!("window{}", resolutions.len())) {
            let hop = HOP.min(n_fft / 4);
            ensure!(
                n_fft >= 16
                    && n_fft.is_power_of_two()
                    && HOP.is_multiple_of(hop)
                    && (n_fft - hop).is_multiple_of(2),
                "unsupported STFT size {n_fft}"
            );
            let window = w.get(&format!("window{}", resolutions.len()), &[n_fft])?;
            resolutions.push(Resolution {
                n_fft,
                hop,
                sub: HOP / hop,
                pad: (n_fft - hop) / 2,
                window: window.into_iter().map(f64::from).collect(),
                fft: RealForward::new(n_fft)?,
            });
        }
        ensure!(
            !resolutions.is_empty(),
            "the checkpoint has no STFT windows"
        );
        let channels: usize = resolutions.iter().map(|r| r.sub * r.width()).sum();
        let Some(&[out_dim, in_channels, ..]) = w.shape("embed.weight") else {
            bail!("checkpoint has no embed.weight");
        };
        ensure!(
            out_dim == dim && in_channels == channels,
            "embed.weight is [{out_dim}, {in_channels}], expected [{dim}, {channels}] (a k1 conv \
             over the features)"
        );
        let embed_weight = w.get("embed.weight", &[dim, channels])?;
        let embed = PackedMat::from_fn(channels, dim, |i, o| embed_weight[o * channels + i]);
        drop(embed_weight);
        let mut blocks = Vec::new();
        while w.contains(&format!("block{}.gamma", blocks.len())) {
            blocks.push(Block::load(w, blocks.len(), dim, mlp)?);
        }
        ensure!(!blocks.is_empty(), "the checkpoint has no ConvNeXt blocks");
        let vector = |name: &str| w.get(name, &[dim]);
        let (final_w, final_b) = (vector("final_norm.weight")?, vector("final_norm.bias")?);
        let (out, out_bias) = fold_linear(
            &w.get("out.weight", &[latent, dim])?,
            &w.get("out.bias", &[latent])?,
            (latent, dim),
            Some((&final_w, &final_b)),
            None,
        )?;
        let reach_before = resolutions.iter().map(|r| r.pad).max().unwrap_or(0);
        let reach_after = resolutions
            .iter()
            .map(Resolution::reach_after)
            .max()
            .unwrap_or(0);
        Ok(Self {
            dim,
            latent,
            channels,
            exponent: (f64::from(power) - 1.0) / 2.0,
            resolutions,
            reach_before,
            reach_after,
            embed,
            embed_bias: vector("embed.bias")?,
            norm_weight: vector("norm.weight")?,
            norm_bias: vector("norm.bias")?,
            blocks,
            out,
            out_bias,
        })
    }

    /// Loads `dac44_stftenc.safetensors`.
    ///
    /// # Errors
    /// Fails when the file is missing or invalid.
    pub fn load(path: &Path) -> Result<Self> {
        let w = Weights::load(path)?;
        Self::from_weights(&w)
            .with_context(|| format!("building the STFT encoder from {}", path.display()))
    }

    /// Builds the encoder from the bytes of `dac44_stftenc.safetensors`.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid encoder.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_weights(&Weights::from_bytes(bytes)?).context("building the STFT encoder")
    }

    /// Channels of the residual stream.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// `ConvNeXt` blocks.
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Values per latent frame (1024 for DAC's quantiser).
    #[must_use]
    pub const fn latent_dim(&self) -> usize {
        self.latent
    }

    /// Feature channels per latent frame (4106 for STFTs of 2048 and 512).
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Latent frames of context per side for an exact chunked encode.
    #[must_use]
    pub const fn receptive_field(&self) -> usize {
        3 * self.blocks.len()
    }

    /// Latent frames of a signal of `len` samples: `ceil(len / 512)`.
    #[must_use]
    pub const fn frames(len: usize) -> usize {
        len.div_ceil(HOP)
    }

    /// The samples of a signal of `len` samples that
    /// [`StftEncoder::latent_span`] reads to encode `frames`.
    #[must_use]
    pub fn samples_needed(&self, frames: Range<usize>, len: usize) -> Range<usize> {
        let window = self.window_of(&frames, Self::frames(len));
        let start = (window.start * HOP).saturating_sub(self.reach_before);
        let end = (window.end * HOP + self.reach_after).min(len);
        start.min(end)..end
    }

    /// The frames computed for `frames`: `frames` with the receptive field
    /// on each side, within the signal.
    fn window_of(&self, frames: &Range<usize>, total: usize) -> Range<usize> {
        let context = self.receptive_field();
        frames.start.saturating_sub(context)..(frames.end + context).min(total)
    }

    /// The latent of mono 44.1 kHz `audio`, read as zero padded to a whole
    /// number of frames: `[ceil(len / 512), latent]`, encoded in parallel
    /// pieces.
    ///
    /// # Errors
    /// Fails on internal shape mismatches only.
    pub fn latent(&self, audio: &[f32]) -> Result<Seq> {
        // Pieces of at most PIECE frames, shorter (down to four contexts)
        // while threads would idle.
        let chunk = Self::frames(audio.len())
            .div_ceil(rayon::current_num_threads())
            .clamp(4 * self.receptive_field(), PIECE);
        self.latent_chunked(audio, chunk)
    }

    /// [`StftEncoder::latent`] in chunks of `chunk` frames, run in parallel;
    /// every chunk length gives the same latent, bit for bit.
    ///
    /// # Errors
    /// Fails when `chunk` is zero.
    pub fn latent_chunked(&self, audio: &[f32], chunk: usize) -> Result<Seq> {
        ensure!(chunk > 0, "chunk length must be positive");
        let frames = Self::frames(audio.len());
        let parts = (0..frames.div_ceil(chunk))
            .into_par_iter()
            .map(|i| {
                let start = i * chunk;
                let end = start.saturating_add(chunk).min(frames);
                self.latent_span(audio, 0, audio.len(), start..end)
            })
            .collect::<Result<Vec<_>>>()?;
        if parts.is_empty() {
            return Ok(Seq::zeros(0, self.latent));
        }
        Seq::concat(parts)
    }

    /// The latent of `frames` of a signal of `len` samples, of which
    /// `samples` holds samples `start..start + samples.len()`: at least
    /// [`StftEncoder::samples_needed`] of them. The same values, bit for bit,
    /// as those frames of a whole-signal encode. This is the streaming
    /// entry point: an encoder that has received samples up to `s` can
    /// encode every frame whose needed samples end by `s` (or by the end of
    /// the signal, once `len` is known).
    ///
    /// # Errors
    /// Fails when `frames` is past the end of the signal or `samples` does
    /// not cover the samples needed.
    pub fn latent_span(
        &self,
        samples: &[f32],
        start: usize,
        len: usize,
        frames: Range<usize>,
    ) -> Result<Seq> {
        let total = Self::frames(len);
        ensure!(
            frames.start <= frames.end && frames.end <= total,
            "frames {frames:?} are outside a signal of {total} frames"
        );
        if frames.is_empty() {
            return Ok(Seq::zeros(0, self.latent));
        }
        let needed = self.samples_needed(frames.clone(), len);
        ensure!(
            start <= needed.start && needed.end <= start + samples.len(),
            "frames {frames:?} need samples {needed:?}, got {start}..{}",
            start + samples.len()
        );
        let signal = Signal {
            samples,
            start,
            len,
        };
        let window = self.window_of(&frames, total);
        let z = SCRATCH.with(|cell| {
            let mut scratch = cell
                .try_borrow_mut()
                .context("the STFT encoder scratch is in use")?;
            self.run(&mut scratch, signal, window, frames.clone())
        })?;
        Seq::from_vec(frames.len(), self.latent, z)
    }

    /// The features of latent frame `frame` into `out` (`channels` values).
    fn features(&self, signal: Signal<'_>, frame: usize, out: &mut [f32], s: &mut FftScratch) {
        let FftScratch {
            buffer,
            re,
            im,
            fft,
        } = s;
        let mut base = 0;
        for r in &self.resolutions {
            let (n, bins, width) = (r.n_fft, r.n_fft / 2 + 1, r.width());
            buffer.resize(n, 0.0);
            re.resize(bins, 0.0);
            im.resize(bins, 0.0);
            for i in 0..r.sub {
                // STFT frame j starts at sample j * hop - pad.
                let first = (frame * r.sub + i) * r.hop;
                for ((v, &w), p) in buffer.iter_mut().zip(&r.window).zip(first..) {
                    *v = signal.at(p, r.pad) * w;
                }
                r.fft.run(buffer, re, im, fft);
                let dst = &mut out[base + i * width..base + (i + 1) * width];
                let (dst_re, dst_im) = dst.split_at_mut(bins);
                for (((&a, &b), y_re), y_im) in re.iter().zip(im.iter()).zip(dst_re).zip(dst_im) {
                    let scale = a.mul_add(a, b * b) + f64::from(POWER_EPS);
                    let scale = scale.powf(self.exponent);
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "features are rounded to single precision once"
                    )]
                    let (u, v) = ((a * scale) as f32, (b * scale) as f32);
                    *y_re = u;
                    *y_im = v;
                }
            }
            base += r.sub * width;
        }
    }

    /// Runs the network on frames `window` as if they were the whole signal
    /// and returns the latent of frames `keep` (inside `window`).
    fn run(
        &self,
        s: &mut Scratch,
        signal: Signal<'_>,
        window: Range<usize>,
        keep: Range<usize>,
    ) -> Result<Vec<f32>> {
        let Scratch {
            feats,
            x,
            h,
            wide,
            fft,
        } = s;
        let (dim, channels) = (self.dim, self.channels);
        let frames = window.len();
        let x = sized(x, frames * dim);
        let feats = sized(feats, TILE.min(frames) * channels);
        for (tile, x_tile) in x.chunks_mut(TILE * dim).enumerate() {
            let rows = x_tile.len() / dim;
            let prof = crate::prof::scope("stftenc stft", channels, 0, rows);
            for (r, out) in feats.chunks_exact_mut(channels).take(rows).enumerate() {
                self.features(signal, window.start + tile * TILE + r, out, fft);
            }
            prof.stop();
            let _prof = crate::prof::scope("stftenc embed", channels, dim, rows);
            self.embed.mul(
                &feats[..rows * channels],
                channels,
                Some(&self.embed_bias),
                x_tile,
            )?;
        }
        kernels::layer_norm_in_place(x, dim, EPS, Some((&self.norm_weight, &self.norm_bias)));
        let h = sized(h, frames * dim);
        let mlp = self.blocks.first().map_or(0, Block::mlp);
        let wide = sized(wide, TILE.min(frames) * mlp);
        for block in &self.blocks {
            block.forward(x, h, wide)?;
        }
        let kept = &mut x[(keep.start - window.start) * dim..(keep.end - window.start) * dim];
        kernels::layer_norm_in_place(kept, dim, EPS, None);
        let mut z = vec![0.0; keep.len() * self.latent];
        let _prof = crate::prof::scope("stftenc out", dim, self.latent, keep.len());
        for (x_tile, z_tile) in kept
            .chunks(TILE * dim)
            .zip(z.chunks_mut(TILE * self.latent))
        {
            self.out.mul(x_tile, dim, Some(&self.out_bias), z_tile)?;
        }
        Ok(z)
    }
}

/// `buffer` as a slice of `len` values, grown when needed; its contents are
/// unspecified.
fn sized(buffer: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    &mut buffer[..len]
}

/// Working memory of the features.
#[derive(Debug)]
struct FftScratch {
    buffer: Vec<f64>,
    re: Vec<f64>,
    im: Vec<f64>,
    fft: Scratch64,
}

/// Working memory of one thread's encodes, kept between calls.
#[derive(Debug)]
struct Scratch {
    feats: Vec<f32>,
    x: Vec<f32>,
    h: Vec<f32>,
    wide: Vec<f32>,
    fft: FftScratch,
}

impl Scratch {
    const fn new() -> Self {
        Self {
            feats: Vec::new(),
            x: Vec::new(),
            h: Vec::new(),
            wide: Vec::new(),
            fft: FftScratch {
                buffer: Vec::new(),
                re: Vec::new(),
                im: Vec::new(),
                fft: Scratch64::new(),
            },
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
}
