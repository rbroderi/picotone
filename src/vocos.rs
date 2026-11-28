//! The HQ decoder (`dac44_vocos`) of DAC 44 kHz codes.
//!
//! Its layers all run at the latent frame rate (86.13 Hz) and its samples
//! come from an inverse STFT: about a fifth of the work per second of the
//! DAC-architecture decoder it replaced. The design follows Vocos (Siuzdak 2023, MIT, Copyright (c) 2023 Charactr
//! Inc.); the network is `train/hq/vocos_student.py`, converted by
//! `tools/export_vocos.py`:
//!
//! - **The input.** The codebook projections are folded into the embed conv
//!   ([`crate::dac`]): 81 values per frame.
//! - **The backbone.** A k7 conv to `dim` channels and a `LayerNorm`, then
//!   `ConvNeXt` blocks: a depthwise k7 conv, `LayerNorm`, a Linear to `mlp`,
//!   exact GELU, a Linear back, a per-channel layer scale, plus the input.
//!   Then a final `LayerNorm`. Every `LayerNorm` is per frame (eps 1e-6).
//! - **The head.** A Linear to `n_fft + 2` values per frame: log-magnitudes
//!   and phases of `n_fft / 2 + 1` bins. The magnitude is `exp` clipped at
//!   100; an inverse real FFT, a Hann window and overlap-add with "same"
//!   padding give exactly 512 samples per frame, divided by the overlapped
//!   squared window. Only hop 512 (`n_fft` 2048, one STFT frame per latent
//!   frame) is built.
//! - **The post-filter** (tensors `pf.*`, when present; WHITEPAPER §2.6). At
//!   1 to [`POST_FILTER_MAX_Q`] codebooks, a small `ConvNeXt` (4 blocks) reads
//!   the head's spectrum `S`, the final hidden state and the tier, and gives
//!   one gain per bin and frame: `S' = S exp(tanh(o))`, phase kept. Above
//!   that tier it does not run and the decode is the network's own.
//!
//! At load, each `LayerNorm`'s scale and offset are folded into the Linear
//! after it, and the layer scale into the second Linear of its block.
//!
//! **Chunking.** An output frame depends on the codes within
//! `3 (blocks + 1) + 2` frames of it (29 for 8 blocks), 3 more per
//! post-filter block (41 with it): 3 per k7 conv, and 2 for the windows that
//! overlap its samples. Every value is computed per
//! frame by the same operations wherever the frame sits in a window (the
//! matrix products are row by row, the depthwise conv adds its taps in
//! order, the overlap-add sums frames in order), so a decode window needs no
//! alignment and any chunking equals a whole-file decode bit for bit.

use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};
use rayon::prelude::*;

use crate::dac::{self, CODEBOOK_DIM, CODEBOOK_SIZE, CODEBOOKS, FOLDED_INPUT, HOP};
use crate::fft::{self, RealInverse};
use crate::hqstream::{Context, HqDecoder};
use crate::kernels::{self, Affine, PackedMat};
use crate::weights::Weights;

/// The decoder's name; its weight files are `dac44_vocos.*`.
pub const NAME: &str = "dac44_vocos";
/// Kernel size of the embed and depthwise convs.
const KERNEL: usize = 7;
/// `LayerNorm` epsilon.
const EPS: f32 = 1e-6;
/// Frames per tile of the Linears: the wide activations of one tile stay in
/// cache, and only the residual stream spans the whole window.
pub(crate) const TILE: usize = 64;
/// Most frames one network call decodes (before context): with 29 frames of
/// context per side, long pieces recompute little of it.
pub const PIECE: usize = 1024;

/// One `ConvNeXt` block, folded (shared with [`crate::stftenc`]).
#[derive(Debug, Clone)]
pub(crate) struct Block {
    /// The depthwise conv, `[tap][channel]`.
    taps: Vec<f32>,
    dw_bias: Vec<f32>,
    /// The first Linear with the block's `LayerNorm` folded in, `[dim, mlp]`.
    pw1: PackedMat,
    pw1_bias: Vec<f32>,
    /// The second Linear with the layer scale folded in, `[mlp, dim]`; its
    /// bias is added to the residual stream first.
    pw2: PackedMat,
    pw2_bias: Affine,
}

/// The Vocos decoder.
#[derive(Debug, Clone)]
pub struct VocosDecoder {
    dim: usize,
    mlp: usize,
    n_fft: usize,
    /// `[codebook][entry][value]`.
    codebooks: Vec<f32>,
    /// The folded embed conv, `[(tap, input), dim]`.
    embed: PackedMat,
    embed_bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    blocks: Vec<Block>,
    /// The head with the final `LayerNorm` folded in, `[dim, n_fft + 2]`.
    head: PackedMat,
    head_bias: Vec<f32>,
    /// The low-tier post-filter, when the weights have one.
    post_filter: Option<PostFilter>,
    fft: RealInverse,
    /// The Hann window over `n_fft` (the FFT's normalisation folded in).
    window: Vec<f32>,
    /// The squared Hann window, for the overlap-add envelope.
    window_sq: Vec<f32>,
}

/// A PyTorch Linear `[out, in]` as a `[in, out]` matrix and bias, with an
/// optional `LayerNorm` affine part before it folded in (`w diag(scale)`,
/// `b + w shift`) and an optional per-output scale after it, in double
/// precision. Each folded value is written straight into the packed matrix.
pub(crate) fn fold_linear(
    weight: &[f32],
    bias: &[f32],
    (outputs, inputs): (usize, usize),
    before: Option<(&[f32], &[f32])>,
    after: Option<&[f32]>,
) -> Result<(PackedMat, Vec<f32>)> {
    ensure!(
        weight.len() == outputs * inputs
            && bias.len() == outputs
            && before.is_none_or(|(s, t)| s.len() == inputs && t.len() == inputs)
            && after.is_none_or(|s| s.len() == outputs),
        "linear layer size mismatch"
    );
    let affine = |i: usize| before.map_or((1.0, 0.0), |(s, t)| (f64::from(s[i]), f64::from(t[i])));
    let post = |o: usize| after.map_or(1.0, |s| f64::from(s[o]));
    let folded = (0..outputs)
        .map(|o| {
            let row = &weight[o * inputs..(o + 1) * inputs];
            let b = row
                .iter()
                .enumerate()
                .fold(f64::from(bias[o]), |b, (i, &w)| {
                    b + f64::from(w) * affine(i).1
                });
            #[expect(clippy::cast_possible_truncation, reason = "rounding the fold once")]
            let value = (b * post(o)) as f32;
            value
        })
        .collect();
    let matrix = PackedMat::from_fn(inputs, outputs, |i, o| {
        #[expect(clippy::cast_possible_truncation, reason = "rounding the fold once")]
        let value = (f64::from(weight[o * inputs + i]) * affine(i).0 * post(o)) as f32;
        value
    });
    Ok((matrix, folded))
}

/// Tensor `name` of shape `[len]`.
fn vector(w: &Weights<'_>, name: &str, len: usize) -> Result<Vec<f32>> {
    w.get(name, &[len])
}

/// The periodic Hann window over `n` (as `torch.hann_window`, in float32)
/// divided by `n` for the FFT's normalisation, and its square.
fn hann_windows(n: usize) -> (Vec<f32>, Vec<f32>) {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "window values, computed in double precision and rounded once"
    )]
    (0..n)
        .map(|j| {
            let hann = 0.5 - 0.5 * (std::f64::consts::TAU * j as f64 / n as f64).cos();
            let single = hann as f32;
            ((hann / n as f64) as f32, single * single)
        })
        .unzip()
}

impl Block {
    /// Width of the block's hidden layer.
    pub(crate) const fn mlp(&self) -> usize {
        self.pw1_bias.len()
    }

    /// The block on time-major frames `x` (`[frames, dim]`), in place: `h`
    /// is `x.len()` values of working memory, `wide` at least
    /// `TILE.min(frames) * mlp`.
    pub(crate) fn forward(&self, x: &mut [f32], h: &mut [f32], wide: &mut [f32]) -> Result<()> {
        let (dim, mlp) = (self.dw_bias.len(), self.pw1_bias.len());
        let frames = x.len() / dim;
        let prof = crate::prof::scope("vocos dw+ln", dim, KERNEL, frames);
        kernels::depthwise7_layer_norm(x, &self.taps, &self.dw_bias, EPS, h);
        prof.stop();
        for (x_tile, h_tile) in x.chunks_mut(TILE * dim).zip(h.chunks(TILE * dim)) {
            let rows = h_tile.len() / dim;
            let u = &mut wide[..rows * mlp];
            let prof = crate::prof::scope("vocos pw1", dim, mlp, rows);
            self.pw1.mul(h_tile, dim, Some(&self.pw1_bias), u)?;
            prof.stop();
            let prof = crate::prof::scope("vocos gelu", mlp, 0, rows);
            kernels::gelu_in_place(u);
            prof.stop();
            let _prof = crate::prof::scope("vocos pw2", mlp, dim, rows);
            kernels::map_in_place(x_tile, Some(&self.pw2_bias), false);
            self.pw2.mul(u, mlp, None, x_tile)?;
        }
        Ok(())
    }

    /// Block `index` of the checkpoint, folded.
    pub(crate) fn load(w: &Weights<'_>, index: usize, dim: usize, mlp: usize) -> Result<Self> {
        Self::load_named(w, &format!("block{index}"), dim, mlp)
    }

    /// The block whose tensors are `{p}.dwconv.weight` and so on, folded.
    pub(crate) fn load_named(w: &Weights<'_>, p: &str, dim: usize, mlp: usize) -> Result<Self> {
        let taps_in = w.get(&format!("{p}.dwconv.weight"), &[dim, KERNEL])?;
        let mut taps = vec![0.0; KERNEL * dim];
        for c in 0..dim {
            for j in 0..KERNEL {
                taps[j * dim + c] = taps_in[c * KERNEL + j];
            }
        }
        let norm_w = vector(w, &format!("{p}.norm.weight"), dim)?;
        let norm_b = vector(w, &format!("{p}.norm.bias"), dim)?;
        let (pw1, pw1_bias) = fold_linear(
            &w.get(&format!("{p}.pw1.weight"), &[mlp, dim])?,
            &vector(w, &format!("{p}.pw1.bias"), mlp)?,
            (mlp, dim),
            Some((&norm_w, &norm_b)),
            None,
        )?;
        let gamma = vector(w, &format!("{p}.gamma"), dim)?;
        let (pw2, pw2_bias) = fold_linear(
            &w.get(&format!("{p}.pw2.weight"), &[dim, mlp])?,
            &vector(w, &format!("{p}.pw2.bias"), dim)?,
            (dim, mlp),
            None,
            Some(&gamma),
        )?;
        Ok(Self {
            taps,
            dw_bias: vector(w, &format!("{p}.dwconv.bias"), dim)?,
            pw1,
            pw1_bias,
            pw2,
            pw2_bias: Affine::new(pw2_bias.into_iter().map(|b| (1.0, b))),
        })
    }
}

/// The low-tier post-filter (tensors `pf.*`): a real gain per bin and frame on
/// the head's spectrum at tiers 1 to [`POST_FILTER_MAX_Q`]; the phase is kept.
#[derive(Debug, Clone)]
struct PostFilter {
    dim: usize,
    mlp: usize,
    decoder_dim: usize,
    /// The spectrum features' Linear, `[2 bins, dim]`.
    embed: PackedMat,
    /// The decoder's hidden state's Linear, `[decoder dim, dim]`, with the
    /// decoder's final `LayerNorm` affine part folded in.
    h_embed: PackedMat,
    /// Per tier 1 to [`POST_FILTER_MAX_Q`]: both Linears' biases plus the
    /// tier's embedding, `[tier - 1][dim]`.
    bias: Vec<Vec<f32>>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    blocks: Vec<Block>,
    /// The head with the final `LayerNorm` folded in, `[dim, bins]`.
    head: PackedMat,
    head_bias: Vec<f32>,
}

/// The highest tier (codebooks per frame) the post-filter runs at; above it
/// the decode is the network's own, bit for bit.
pub const POST_FILTER_MAX_Q: usize = 5;
/// The power the post-filter's input features compress magnitudes with.
const PF_COMPRESSION: f32 = 0.3;
/// Added to the magnitude before the compression.
const PF_EPS: f32 = 1e-5;
/// The gain is `exp(PF_GAIN_BOUND tanh(o))`.
const PF_GAIN_BOUND: f32 = 1.0;

impl PostFilter {
    /// Loads `pf.*` for a decoder `decoder_dim` wide with final `LayerNorm`
    /// `final_norm`, and `bins` spectrum bins.
    fn load(
        w: &Weights<'_>,
        decoder_dim: usize,
        final_norm: (&[f32], &[f32]),
        bins: usize,
    ) -> Result<Self> {
        let dim = *w
            .shape("pf.embed.bias")
            .and_then(<[usize]>::first)
            .context("the post-filter has no pf.embed.bias")?;
        let mlp = *w
            .shape("pf.block0.pw1.bias")
            .and_then(<[usize]>::first)
            .context("the post-filter has no pf.block0.pw1.bias")?;
        let v = |name: &str, len: usize| vector(w, name, len);
        let (embed, embed_bias) = fold_linear(
            &w.get("pf.embed.weight", &[dim, 2 * bins])?,
            &v("pf.embed.bias", dim)?,
            (dim, 2 * bins),
            None,
            None,
        )?;
        let (h_embed, h_bias) = fold_linear(
            &w.get("pf.h_embed.weight", &[dim, decoder_dim])?,
            &v("pf.h_embed.bias", dim)?,
            (dim, decoder_dim),
            Some(final_norm),
            None,
        )?;
        let tiers = w.get("pf.tier_emb.weight", &[POST_FILTER_MAX_Q + 1, dim])?;
        let tier_bias = (1..=POST_FILTER_MAX_Q)
            .map(|q| {
                (0..dim)
                    .map(|c| {
                        let b = f64::from(embed_bias[c])
                            + f64::from(h_bias[c])
                            + f64::from(tiers[q * dim + c]);
                        #[expect(clippy::cast_possible_truncation, reason = "rounding the sum once")]
                        let value = b as f32;
                        value
                    })
                    .collect()
            })
            .collect();
        let mut blocks = Vec::new();
        while w.contains(&format!("pf.block{}.gamma", blocks.len())) {
            let name = format!("pf.block{}", blocks.len());
            blocks.push(Block::load_named(w, &name, dim, mlp)?);
        }
        let (final_w, final_b) = (v("pf.final_norm.weight", dim)?, v("pf.final_norm.bias", dim)?);
        let (head, head_bias) = fold_linear(
            &w.get("pf.head.weight", &[bins, dim])?,
            &v("pf.head.bias", bins)?,
            (bins, dim),
            Some((&final_w, &final_b)),
            None,
        )?;
        Ok(Self {
            dim,
            mlp,
            decoder_dim,
            embed,
            h_embed,
            bias: tier_bias,
            norm_weight: v("pf.norm.weight", dim)?,
            norm_bias: v("pf.norm.bias", dim)?,
            blocks,
            head,
            head_bias,
        })
    }

    /// Latent frames of context per side the post-filter adds.
    const fn receptive_field(&self) -> usize {
        (KERNEL / 2) * self.blocks.len()
    }

    /// Filters `spec` (`[frame][re of the bins, then im]`) in place at tier
    /// `q`, from the decoder's normalised hidden state `hidden`
    /// (`[frames, decoder dim]`, before its final affine part). `x`, `h`,
    /// `wide` and `tile` are working memory.
    fn apply(
        &self,
        spec: &mut [f32],
        hidden: &[f32],
        q: usize,
        [x, h, wide, tile]: [&mut Vec<f32>; 4],
    ) -> Result<()> {
        let (dim, decoder_dim) = (self.dim, self.decoder_dim);
        let bins = self.head_bias.len();
        let frames = spec.len() / (2 * bins);
        let x = sized(x, frames * dim);
        {
            let _prof = crate::prof::scope("postfilter embed", 2 * bins, dim, frames);
            for (t0, x_tile) in (0..frames).step_by(TILE).zip(x.chunks_mut(TILE * dim)) {
                let rows = x_tile.len() / dim;
                let feat = sized(tile, rows * 2 * bins);
                for (f, s) in feat
                    .chunks_exact_mut(2 * bins)
                    .zip(spec[t0 * 2 * bins..].chunks_exact(2 * bins))
                {
                    let (re, im) = s.split_at(bins);
                    let (fr, fi) = f.split_at_mut(bins);
                    for (((fr, fi), &a), &b) in fr.iter_mut().zip(fi).zip(re).zip(im) {
                        let scale = (a.hypot(b) + PF_EPS).powf(PF_COMPRESSION - 1.0);
                        (*fr, *fi) = (a * scale, b * scale);
                    }
                }
                self.embed.mul(feat, 2 * bins, Some(&self.bias[q - 1]), x_tile)?;
                self.h_embed.mul(
                    &hidden[t0 * decoder_dim..(t0 + rows) * decoder_dim],
                    decoder_dim,
                    None,
                    x_tile,
                )?;
            }
            kernels::layer_norm_in_place(x, dim, EPS, Some((&self.norm_weight, &self.norm_bias)));
        }
        let h = sized(h, frames * dim);
        let wide = sized(wide, TILE.min(frames) * self.mlp);
        for block in &self.blocks {
            block.forward(x, h, wide)?;
        }
        kernels::layer_norm_in_place(x, dim, EPS, None);
        let _prof = crate::prof::scope("postfilter head", dim, bins, frames);
        for (x_tile, s_tile) in x.chunks(TILE * dim).zip(spec.chunks_mut(TILE * 2 * bins)) {
            let rows = x_tile.len() / dim;
            let o = sized(tile, rows * bins);
            self.head.mul(x_tile, dim, Some(&self.head_bias), o)?;
            for (o, s) in o.chunks_exact(bins).zip(s_tile.chunks_exact_mut(2 * bins)) {
                let (re, im) = s.split_at_mut(bins);
                for ((&o, re), im) in o.iter().zip(re).zip(im) {
                    let gain = (PF_GAIN_BOUND * o.tanh()).exp();
                    *re *= gain;
                    *im *= gain;
                }
            }
        }
        Ok(())
    }
}

impl VocosDecoder {
    /// Builds the decoder from `tools/export_vocos.py`'s tensors; the sizes
    /// come from the tensors' shapes.
    ///
    /// # Errors
    /// Fails when a tensor is missing or has the wrong shape, or the head is
    /// not for hop 512.
    pub fn from_weights(w: &Weights<'_>) -> Result<Self> {
        let dim = *w
            .shape("embed.bias")
            .and_then(<[usize]>::first)
            .context("checkpoint has no embed.bias: not a Vocos decoder")?;
        let mlp = *w
            .shape("block0.pw1.bias")
            .and_then(<[usize]>::first)
            .context("checkpoint has no block0.pw1.bias")?;
        let width = *w
            .shape("head.bias")
            .and_then(<[usize]>::first)
            .context("checkpoint has no head.bias")?;
        let n_fft = width.saturating_sub(2);
        if n_fft != 4 * HOP {
            bail!(
                "the head gives {width} values per frame; this decoder supports hop 512 (n_fft 2048, \
                 2050 values) only"
            );
        }
        let codebooks = w.get("codebooks", &[CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM])?;
        let embed_weight = w.get("embed.weight", &[dim, FOLDED_INPUT, KERNEL])?;
        // Row (tap, input), column output.
        let embed = PackedMat::from_fn(KERNEL * FOLDED_INPUT, dim, |r, o| {
            let (j, i) = (r / FOLDED_INPUT, r % FOLDED_INPUT);
            embed_weight[(o * FOLDED_INPUT + i) * KERNEL + j]
        });
        drop(embed_weight);
        let vector = |name: &str, len: usize| vector(w, name, len);
        let mut blocks = Vec::new();
        while w.contains(&format!("block{}.gamma", blocks.len())) {
            blocks.push(Block::load(w, blocks.len(), dim, mlp)?);
        }
        ensure!(!blocks.is_empty(), "the checkpoint has no ConvNeXt blocks");
        let final_w = vector("final_norm.weight", dim)?;
        let final_b = vector("final_norm.bias", dim)?;
        let (head, head_bias) = fold_linear(
            &w.get("head.weight", &[width, dim])?,
            &vector("head.bias", width)?,
            (width, dim),
            Some((&final_w, &final_b)),
            None,
        )?;
        let post_filter = if w.contains("pf.embed.bias") {
            Some(
                PostFilter::load(w, dim, (&final_w, &final_b), n_fft / 2 + 1)
                    .context("loading the post-filter")?,
            )
        } else {
            None
        };
        let (window, window_sq) = hann_windows(n_fft);
        Ok(Self {
            dim,
            mlp,
            n_fft,
            codebooks,
            embed,
            embed_bias: vector("embed.bias", dim)?,
            norm_weight: vector("norm.weight", dim)?,
            norm_bias: vector("norm.bias", dim)?,
            blocks,
            head,
            head_bias,
            post_filter,
            fft: RealInverse::new(n_fft)?,
            window,
            window_sq,
        })
    }

    /// Loads `dac44_vocos.safetensors`.
    ///
    /// # Errors
    /// Fails when the file is missing or invalid.
    pub fn load(path: &Path) -> Result<Self> {
        let w = Weights::load(path)?;
        Self::from_weights(&w)
            .with_context(|| format!("building the Vocos decoder from {}", path.display()))
    }

    /// Builds the decoder from the bytes of `dac44_vocos.safetensors`.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid decoder.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_weights(&Weights::from_bytes(bytes)?).context("building the Vocos HQ decoder")
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

    /// Whether the weights include the low-tier post-filter.
    #[must_use]
    pub const fn has_post_filter(&self) -> bool {
        self.post_filter.is_some()
    }

    /// Latent frames of context per side for an exact chunked decode.
    #[must_use]
    pub const fn receptive_field(&self) -> usize {
        let post_filter = match &self.post_filter {
            Some(pf) => pf.receptive_field(),
            None => 0,
        };
        (KERNEL / 2) * (1 + self.blocks.len()) + post_filter + (self.n_fft / 2).div_ceil(HOP)
    }

    /// Decodes `codes`, `[frames][num_codebooks]` (codebooks 0 to
    /// `num_codebooks - 1`, each code below 1024): `frames * 512` samples.
    ///
    /// # Errors
    /// Fails when the codes do not fit the decoder.
    pub fn decode(&self, codes: &[u16], num_codebooks: usize) -> Result<Vec<f32>> {
        // Pieces of at most PIECE frames, shorter (down to four contexts)
        // while threads would idle, as `HqStream` splits a span.
        let frames = codes.len() / num_codebooks.max(1);
        let chunk = frames
            .div_ceil(rayon::current_num_threads())
            .clamp(4 * self.receptive_field(), PIECE);
        self.decode_chunked(codes, num_codebooks, chunk)
    }

    /// [`VocosDecoder::decode`] in chunks of `chunk` frames, run in
    /// parallel; every chunk length gives the same samples, bit for bit.
    ///
    /// # Errors
    /// See [`VocosDecoder::decode`].
    pub fn decode_chunked(
        &self,
        codes: &[u16],
        num_codebooks: usize,
        chunk: usize,
    ) -> Result<Vec<f32>> {
        dac::check_codes(codes, num_codebooks)?;
        ensure!(chunk > 0, "chunk length must be positive");
        let frames = codes.len() / num_codebooks;
        let parts = (0..frames.div_ceil(chunk))
            .into_par_iter()
            .map(|i| {
                let start = i * chunk;
                let end = (start + chunk).min(frames);
                self.decode_range(codes, num_codebooks, 0, start..end, frames)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(parts.concat())
    }

    /// Decodes one window of frames as if it were the whole file.
    fn decode_span(&self, codes: &[u16], num_codebooks: usize) -> Result<Vec<f32>> {
        let frames = codes.len() / num_codebooks;
        if frames == 0 {
            return Ok(Vec::new());
        }
        SCRATCH.with(|cell| {
            let mut scratch = cell
                .try_borrow_mut()
                .context("the Vocos scratch is in use")?;
            self.run(&mut scratch, codes, num_codebooks, frames)
        })
    }

    fn run(
        &self,
        s: &mut Scratch,
        codes: &[u16],
        num_codebooks: usize,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let Scratch {
            input,
            x,
            h,
            wide,
            head,
            spec,
            pf_x,
            frame,
            ola,
            fft,
        } = s;
        let (dim, pad) = (self.dim, KERNEL / 2);
        let input = sized(input, (frames + 2 * pad) * FOLDED_INPUT);
        input.fill(0.0);
        dac::write_folded_input(
            &self.codebooks,
            codes,
            num_codebooks,
            &mut input[pad * FOLDED_INPUT..(pad + frames) * FOLDED_INPUT],
        );
        let x = sized(x, frames * dim);
        {
            let _prof = crate::prof::scope("vocos embed", FOLDED_INPUT, dim, frames);
            self.embed
                .mul(input, FOLDED_INPUT, Some(&self.embed_bias), x)?;
            kernels::layer_norm_in_place(x, dim, EPS, Some((&self.norm_weight, &self.norm_bias)));
        }
        {
            let h = sized(h, frames * dim);
            let wide = sized(wide, TILE.min(frames) * self.mlp);
            for block in &self.blocks {
                block.forward(x, h, wide)?;
            }
        }
        kernels::layer_norm_in_place(x, dim, EPS, None);
        let (n_fft, bins) = (self.n_fft, self.n_fft / 2 + 1);
        let width = 2 * bins;
        // The spectrum of every frame, `[frame][re of the bins, then im]`.
        let spec = sized(spec, frames * width);
        {
            let o = sized(head, TILE.min(frames) * width);
            for (x_tile, s_tile) in x.chunks(TILE * dim).zip(spec.chunks_mut(TILE * width)) {
                let rows = x_tile.len() / dim;
                let o = &mut o[..rows * width];
                let _prof = crate::prof::scope("vocos head", dim, width, rows);
                self.head.mul(x_tile, dim, Some(&self.head_bias), o)?;
                for (row, s) in o.chunks_exact(width).zip(s_tile.chunks_exact_mut(width)) {
                    let (re, im) = s.split_at_mut(bins);
                    kernels::polar(&row[..bins], &row[bins..], re, im);
                }
            }
        }
        if let Some(pf) = &self.post_filter
            && (1..=POST_FILTER_MAX_Q).contains(&num_codebooks)
        {
            pf.apply(spec, x, num_codebooks, [pf_x, h, wide, head])?;
        }
        let ola = sized(ola, (frames - 1) * HOP + n_fft);
        ola.fill(0.0);
        let frame = sized(frame, n_fft);
        let prof = crate::prof::scope("vocos istft", n_fft, HOP, frames);
        for (t, s) in spec.chunks_exact(width).enumerate() {
            let (re, im) = s.split_at(bins);
            self.fft.run(re, im, frame, fft);
            for ((y, &v), &w) in ola[t * HOP..t * HOP + n_fft]
                .iter_mut()
                .zip(frame.iter())
                .zip(&self.window)
            {
                *y += v * w;
            }
        }
        prof.stop();
        // "Same" padding: sample s is at s + pad of the overlap-add, divided
        // by the squared windows that overlap there, added in frame order.
        let _prof = crate::prof::scope("vocos istft", n_fft, HOP, frames);
        let pad = (n_fft - HOP) / 2;
        let mut y = vec![0.0_f32; frames * HOP];
        for (s, v) in y.iter_mut().enumerate() {
            let p = s + pad;
            let first = (p + 1).saturating_sub(n_fft).div_ceil(HOP);
            let last = (p / HOP).min(frames - 1);
            let envelope = (first..=last).fold(0.0_f32, |e, u| e + self.window_sq[p - u * HOP]);
            *v = ola[p] / envelope;
        }
        Ok(y)
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

/// Working memory of one thread's decodes, kept between calls.
#[derive(Debug)]
struct Scratch {
    input: Vec<f32>,
    x: Vec<f32>,
    h: Vec<f32>,
    wide: Vec<f32>,
    head: Vec<f32>,
    spec: Vec<f32>,
    pf_x: Vec<f32>,
    frame: Vec<f32>,
    ola: Vec<f32>,
    fft: fft::Scratch,
}

impl Scratch {
    const fn new() -> Self {
        Self {
            input: Vec::new(),
            x: Vec::new(),
            h: Vec::new(),
            wide: Vec::new(),
            head: Vec::new(),
            spec: Vec::new(),
            pf_x: Vec::new(),
            frame: Vec::new(),
            ola: Vec::new(),
            fft: fft::Scratch::new(),
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
}

impl HqDecoder for VocosDecoder {
    fn name(&self) -> &'static str {
        NAME
    }

    fn codebook_hash(&self) -> u64 {
        dac::codebook_hash(&self.codebooks)
    }

    fn context(&self) -> Context {
        Context {
            frames: self.receptive_field(),
            align: 1,
        }
    }

    fn piece(&self) -> usize {
        PIECE
    }

    fn decode_window(&self, codes: &[u16], num_codebooks: usize) -> Result<Vec<f32>> {
        dac::check_codes(codes, num_codebooks)?;
        self.decode_span(codes, num_codebooks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fold as the loader did it before it packed straight from the
    /// tensors: a row-major `[in, out]` matrix first, then packed.
    fn fold_linear_via_matrix(
        weight: &[f32],
        bias: &[f32],
        (outputs, inputs): (usize, usize),
        before: Option<(&[f32], &[f32])>,
        after: Option<&[f32]>,
    ) -> Result<(PackedMat, Vec<f32>)> {
        let mut matrix = vec![0.0_f32; inputs * outputs];
        let mut folded = vec![0.0_f32; outputs];
        for o in 0..outputs {
            let row = &weight[o * inputs..(o + 1) * inputs];
            let post = after.map_or(1.0, |s| f64::from(s[o]));
            let mut b = f64::from(bias[o]);
            for (i, &w) in row.iter().enumerate() {
                let (scale, shift) =
                    before.map_or((1.0, 0.0), |(s, t)| (f64::from(s[i]), f64::from(t[i])));
                b += f64::from(w) * shift;
                #[expect(clippy::cast_possible_truncation, reason = "rounding the fold once")]
                let value = (f64::from(w) * scale * post) as f32;
                matrix[i * outputs + o] = value;
            }
            #[expect(clippy::cast_possible_truncation, reason = "rounding the fold once")]
            let value = (b * post) as f32;
            folded[o] = value;
        }
        Ok((PackedMat::pack(matrix, inputs, outputs)?, folded))
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The decoder's packed weights equal, bit for bit, those of the loader
    /// that went through row-major matrices.
    fn packs_like_the_matrix_loader(path: &Path) -> Result<()> {
        let bytes = std::fs::read(path)?;
        let w = Weights::from_bytes(&bytes)?;
        let d = VocosDecoder::from_weights(&w)?;
        let (dim, mlp) = (d.dim, d.mlp);
        let embed = w.get("embed.weight", &[dim, FOLDED_INPUT, KERNEL])?;
        let mut matrix = vec![0.0; KERNEL * FOLDED_INPUT * dim];
        for o in 0..dim {
            for i in 0..FOLDED_INPUT {
                for j in 0..KERNEL {
                    matrix[(j * FOLDED_INPUT + i) * dim + o] =
                        embed[(o * FOLDED_INPUT + i) * KERNEL + j];
                }
            }
        }
        let old = PackedMat::pack(matrix, KERNEL * FOLDED_INPUT, dim)?;
        assert!(old.bit_identical(&d.embed), "embed");
        let v = |name: &str, len: usize| w.get(name, &[len]);
        for (index, block) in d.blocks.iter().enumerate() {
            let p = format!("block{index}");
            let (norm_w, norm_b) = (
                v(&format!("{p}.norm.weight"), dim)?,
                v(&format!("{p}.norm.bias"), dim)?,
            );
            let args = (
                w.get(&format!("{p}.pw1.weight"), &[mlp, dim])?,
                v(&format!("{p}.pw1.bias"), mlp)?,
            );
            let before = Some((norm_w.as_slice(), norm_b.as_slice()));
            let (old, old_bias) =
                fold_linear_via_matrix(&args.0, &args.1, (mlp, dim), before, None)?;
            let (new, new_bias) = fold_linear(&args.0, &args.1, (mlp, dim), before, None)?;
            assert!(
                old.bit_identical(&block.pw1) && new.bit_identical(&block.pw1),
                "{p}.pw1"
            );
            assert!(same_bits(&old_bias, &new_bias) && same_bits(&old_bias, &block.pw1_bias));
            let gamma = v(&format!("{p}.gamma"), dim)?;
            let args = (
                w.get(&format!("{p}.pw2.weight"), &[dim, mlp])?,
                v(&format!("{p}.pw2.bias"), dim)?,
            );
            let (old, old_bias) =
                fold_linear_via_matrix(&args.0, &args.1, (dim, mlp), None, Some(&gamma))?;
            let (new, new_bias) = fold_linear(&args.0, &args.1, (dim, mlp), None, Some(&gamma))?;
            assert!(
                old.bit_identical(&block.pw2) && new.bit_identical(&block.pw2),
                "{p}.pw2"
            );
            assert!(same_bits(&old_bias, &new_bias), "{p}.pw2 bias");
        }
        let width = d.n_fft + 2;
        let (final_w, final_b) = (v("final_norm.weight", dim)?, v("final_norm.bias", dim)?);
        let (old, old_bias) = fold_linear_via_matrix(
            &w.get("head.weight", &[width, dim])?,
            &v("head.bias", width)?,
            (width, dim),
            Some((&final_w, &final_b)),
            None,
        )?;
        assert!(old.bit_identical(&d.head), "head");
        assert!(same_bits(&old_bias, &d.head_bias), "head bias");
        Ok(())
    }

    #[test]
    fn tiny_fixture_packs_like_the_matrix_loader() -> Result<()> {
        packs_like_the_matrix_loader(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/vocos_tiny.safetensors"),
        )
    }

    /// The full-size decoder too, when `weights/` has it (it is not committed).
    #[test]
    fn full_size_packs_like_the_matrix_loader() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/dac44_vocos.safetensors");
        if !path.exists() {
            return Ok(());
        }
        packs_like_the_matrix_loader(&path)
    }
}
