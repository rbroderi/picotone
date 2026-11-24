//! The Vocos HQ decoder (`dac44_vocos`) of DAC 44 kHz codes.
//!
//! Its layers all run at the latent frame rate (86.13 Hz) and its samples
//! come from an inverse STFT: about a fifth of the student's work per second.
//! The design follows Vocos (Siuzdak 2023, MIT, Copyright (c) 2023 Charactr
//! Inc.); the network is `distill/vocos_student.py`, converted by
//! `tools/export_vocos.py`:
//!
//! - **The input.** As for the student ([`crate::dac`]), the codebook
//!   projections are folded into the embed conv: 81 values per frame.
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
//!
//! At load, each `LayerNorm`'s scale and offset are folded into the Linear
//! after it, and the layer scale into the second Linear of its block.
//!
//! **Chunking.** An output frame depends on the codes within
//! `3 (blocks + 1) + 2` frames of it (29 for 8 blocks): 3 per k7 conv, and 2
//! for the windows that overlap its samples. Every value is computed per
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
const TILE: usize = 64;
/// Most frames one network call decodes (before context): with 29 frames of
/// context per side, long pieces recompute little of it.
pub const PIECE: usize = 1024;

/// One `ConvNeXt` block, folded.
#[derive(Debug, Clone)]
struct Block {
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
    fft: RealInverse,
    /// The Hann window over `n_fft` (the FFT's normalisation folded in).
    window: Vec<f32>,
    /// The squared Hann window, for the overlap-add envelope.
    window_sq: Vec<f32>,
}

/// A PyTorch Linear `[out, in]` as a `[in, out]` matrix and bias, with an
/// optional `LayerNorm` affine part before it folded in (`w diag(scale)`,
/// `b + w shift`) and an optional per-output scale after it, in double
/// precision.
fn fold_linear(
    weight: &[f32],
    bias: &[f32],
    (outputs, inputs): (usize, usize),
    before: Option<(&[f32], &[f32])>,
    after: Option<&[f32]>,
) -> Result<(PackedMat, Vec<f32>)> {
    ensure!(
        weight.len() == outputs * inputs && bias.len() == outputs,
        "linear layer size mismatch"
    );
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

/// Tensor `name` of shape `[len]`.
fn vector(w: &Weights, name: &str, len: usize) -> Result<Vec<f32>> {
    Ok(w.get(name, &[len])?.into_owned())
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
    /// Block `index` of the checkpoint, folded.
    fn load(w: &Weights, index: usize, dim: usize, mlp: usize) -> Result<Self> {
        let p = format!("block{index}");
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

impl VocosDecoder {
    /// Builds the decoder from `tools/export_vocos.py`'s tensors; the sizes
    /// come from the tensors' shapes.
    ///
    /// # Errors
    /// Fails when a tensor is missing or has the wrong shape, or the head is
    /// not for hop 512.
    pub fn from_weights(w: &Weights) -> Result<Self> {
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
        let codebooks = w
            .get("codebooks", &[CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM])?
            .into_owned();
        let embed_weight = w.get("embed.weight", &[dim, FOLDED_INPUT, KERNEL])?;
        let mut matrix = vec![0.0; KERNEL * FOLDED_INPUT * dim];
        for o in 0..dim {
            for i in 0..FOLDED_INPUT {
                for j in 0..KERNEL {
                    matrix[(j * FOLDED_INPUT + i) * dim + o] =
                        embed_weight[(o * FOLDED_INPUT + i) * KERNEL + j];
                }
            }
        }
        let embed = PackedMat::pack(matrix, KERNEL * FOLDED_INPUT, dim)?;
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
        let w = Weights::from_bytes(bytes)?;
        Self::from_weights(&w).context("building the Vocos HQ decoder")
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

    /// Latent frames of context per side for an exact chunked decode.
    #[must_use]
    pub const fn receptive_field(&self) -> usize {
        (KERNEL / 2) * (1 + self.blocks.len()) + (self.n_fft / 2).div_ceil(HOP)
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
            re,
            im,
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
        let h = sized(h, frames * dim);
        let wide = sized(wide, TILE.min(frames) * self.mlp);
        for block in &self.blocks {
            let prof = crate::prof::scope("vocos dw+ln", dim, KERNEL, frames);
            kernels::depthwise7_layer_norm(x, &block.taps, &block.dw_bias, EPS, h);
            prof.stop();
            for (x_tile, h_tile) in x.chunks_mut(TILE * dim).zip(h.chunks(TILE * dim)) {
                let rows = h_tile.len() / dim;
                let u = &mut wide[..rows * self.mlp];
                let prof = crate::prof::scope("vocos pw1", dim, self.mlp, rows);
                block.pw1.mul(h_tile, dim, Some(&block.pw1_bias), u)?;
                prof.stop();
                let prof = crate::prof::scope("vocos gelu", self.mlp, 0, rows);
                kernels::gelu_in_place(u);
                prof.stop();
                let _prof = crate::prof::scope("vocos pw2", self.mlp, dim, rows);
                kernels::map_in_place(x_tile, Some(&block.pw2_bias), false);
                block.pw2.mul(u, self.mlp, None, x_tile)?;
            }
        }
        kernels::layer_norm_in_place(x, dim, EPS, None);
        let (n_fft, bins) = (self.n_fft, self.n_fft / 2 + 1);
        let width = 2 * bins;
        let ola = sized(ola, (frames - 1) * HOP + n_fft);
        ola.fill(0.0);
        let head = sized(head, TILE.min(frames) * width);
        let (re, im, frame) = (sized(re, bins), sized(im, bins), sized(frame, n_fft));
        for (tile, x_tile) in x.chunks(TILE * dim).enumerate() {
            let rows = x_tile.len() / dim;
            let o = &mut head[..rows * width];
            let prof = crate::prof::scope("vocos head", dim, width, rows);
            self.head.mul(x_tile, dim, Some(&self.head_bias), o)?;
            prof.stop();
            let _prof = crate::prof::scope("vocos istft", n_fft, HOP, rows);
            for (r, row) in o.chunks_exact(width).enumerate() {
                let t = tile * TILE + r;
                kernels::polar(&row[..bins], &row[bins..], re, im);
                self.fft.run(re, im, frame, fft);
                for ((y, &v), &w) in ola[t * HOP..t * HOP + n_fft]
                    .iter_mut()
                    .zip(frame.iter())
                    .zip(&self.window)
                {
                    *y += v * w;
                }
            }
        }
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
    re: Vec<f32>,
    im: Vec<f32>,
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
            re: Vec::new(),
            im: Vec::new(),
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
