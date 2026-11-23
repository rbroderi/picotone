//! The HQ decoder: a 512-wide student of DAC 44 kHz's decoder (`dac44-student`).
//!
//! DAC (descript-audio-codec, MIT, Copyright (c) 2023-present, Descript) codes
//! mono 44.1 kHz audio as 9 codebooks of 1024 entries per latent frame, one
//! frame per 512 samples. The student was distilled from DAC's decoder in
//! `distill/` and exported by `distill/export_rust.py`:
//!
//! - **The input.** The codebook projections are folded into the input conv,
//!   so each frame is 81 values: for each codebook, its 8 embedding values and
//!   a 1 meaning "present". Codebooks above the tier give 9 zeros.
//! - **The layers.** A k7 conv to 512 channels, then four blocks: Snake, a
//!   transposed conv (stride 8, 8, 4, 2, halving the width), then three
//!   residual units (Snake, a k7 conv dilated 1, 3 or 9, Snake, a 1x1 conv,
//!   plus the input). Then Snake, a k7 conv to one channel and tanh. Every
//!   conv is zero padded to keep its length; there is no normalisation and no
//!   recurrence.
//! - **Chunking.** An output sample depends on about ±10 latent frames, so a
//!   long file is decoded in chunks with [`CONTEXT`] frames of context on each
//!   side, which gives the same output as one whole-file decode.

use anyhow::{Context as _, Result, ensure};

use crate::kernels;
use crate::nn::{ConvGeometry, ConvKernel, ConvTranspose1d, Seq};
use crate::weights::Weights;

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Samples per latent frame.
pub const HOP: usize = 512;
/// Codebooks per frame at the highest tier.
pub const CODEBOOKS: usize = 9;
/// Entries per codebook.
pub const CODEBOOK_SIZE: usize = 1024;
/// Values per codebook entry.
const CODEBOOK_DIM: usize = 8;
/// Folded input values per codebook: the entry, then "present".
const IN_PER_CODEBOOK: usize = CODEBOOK_DIM + 1;
/// Upsampling of each block.
const RATES: [usize; 4] = [8, 8, 4, 2];
/// Dilations of each block's residual units.
const DILATIONS: [usize; 3] = [1, 3, 9];
/// Kernel size of every non-transposed conv except the 1x1s.
const KERNEL: usize = 7;
/// Latent frames of context on each side of a chunk. The receptive field is
/// a little under 10 frames each way; the chunking test checks this margin.
pub const CONTEXT: usize = 12;
/// Latent frames per chunk, before context (about 3 s).
const CHUNK: usize = 256;

/// Snake's per-channel parameters.
#[derive(Debug, Clone)]
struct Snake {
    alpha: Vec<f32>,
    /// `1 / (alpha + 1e-9)`, computed as PyTorch does, in single precision.
    inv: Vec<f32>,
}

impl Snake {
    fn load(w: &Weights, name: &str, channels: usize) -> Result<Self> {
        let alpha = w.get(name, &[channels])?.into_owned();
        let inv = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
        Ok(Self { alpha, inv })
    }

    fn apply(&self, values: &mut [f32]) {
        kernels::snake_in_place(values, &self.alpha, &self.inv);
    }
}

/// A conv that keeps its input's length (zero padding `dilation * 3` on each
/// side), with Snake on its input.
#[derive(Debug, Clone)]
struct SnakeConv {
    snake: Snake,
    conv: ConvKernel,
    pad: usize,
}

impl SnakeConv {
    fn load(
        w: &Weights,
        snake: &str,
        conv: &str,
        (cout, cin, kernel): (usize, usize, usize),
        dilation: usize,
    ) -> Result<Self> {
        let weight = w.get(&format!("{conv}.weight"), &[cout, cin, kernel])?;
        let bias = w.get(&format!("{conv}.bias"), &[cout])?;
        Ok(Self {
            snake: Snake::load(w, snake, cin)?,
            conv: ConvKernel::from_torch(&weight, &bias, (cout, cin, kernel), 1, dilation)?,
            pad: (kernel - 1) * dilation / 2,
        })
    }

    fn forward(&self, x: &Seq) -> Result<Seq> {
        let c = x.channels();
        let mut padded = Seq::zeros(x.len() + 2 * self.pad, c);
        let interior = &mut padded.data_mut()[self.pad * c..(self.pad + x.len()) * c];
        interior.copy_from_slice(x.data());
        self.snake.apply(interior);
        self.conv.apply(&padded)
    }
}

/// `x + conv2(snake2(conv1(snake1(x))))`.
#[derive(Debug, Clone)]
struct Residual {
    first: SnakeConv,
    second: SnakeConv,
}

impl Residual {
    fn forward(&self, x: Seq) -> Result<Seq> {
        let h = self.first.forward(&x)?;
        let y = self.second.forward(&h)?;
        let mut out = x;
        out.add_assign(&y)?;
        Ok(out)
    }
}

/// Snake, a transposed conv, then three residual units.
#[derive(Debug, Clone)]
struct Block {
    snake: Snake,
    up: ConvTranspose1d,
    residuals: Vec<Residual>,
}

impl Block {
    fn forward(&self, x: Seq) -> Result<Seq> {
        let mut x = x;
        self.snake.apply(x.data_mut());
        let mut h = self.up.forward(&x, false)?;
        drop(x);
        for residual in &self.residuals {
            h = residual.forward(h)?;
        }
        Ok(h)
    }
}

/// The student decoder.
#[derive(Debug, Clone)]
pub struct DacDecoder {
    width: usize,
    /// `[codebook][entry][value]`.
    codebooks: Vec<f32>,
    input: ConvKernel,
    blocks: Vec<Block>,
    output: SnakeConv,
}

impl DacDecoder {
    /// Builds the decoder from `export_rust.py`'s tensors.
    ///
    /// # Errors
    /// Fails when a tensor is missing or has the wrong shape.
    pub fn from_weights(w: &Weights, width: usize) -> Result<Self> {
        ensure!(
            width.is_multiple_of(1 << RATES.len()),
            "width {width} does not halve {} times",
            RATES.len()
        );
        let codebooks = w
            .get("codebooks", &[CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM])?
            .into_owned();
        let cin = CODEBOOKS * IN_PER_CODEBOOK;
        let input = ConvKernel::from_torch(
            &w.get("input.weight", &[width, cin, KERNEL])?,
            &w.get("input.bias", &[width])?,
            (width, cin, KERNEL),
            1,
            1,
        )?;
        let mut blocks = Vec::with_capacity(RATES.len());
        let mut c = width;
        for (b, &stride) in RATES.iter().enumerate() {
            let cout = c / 2;
            let p = format!("block{b}");
            let up = ConvTranspose1d::new(
                &w.get(&format!("{p}.up.weight"), &[c, cout, 2 * stride])?,
                &w.get(&format!("{p}.up.bias"), &[cout])?,
                ConvGeometry {
                    in_channels: c,
                    out_channels: cout,
                    kernel: 2 * stride,
                    stride,
                    dilation: 1,
                },
                false,
                0.0,
                None,
            )?;
            let residuals = DILATIONS
                .iter()
                .enumerate()
                .map(|(r, &dilation)| -> Result<Residual> {
                    let q = format!("{p}.res{r}");
                    Ok(Residual {
                        first: SnakeConv::load(
                            w,
                            &format!("{q}.snake1.alpha"),
                            &format!("{q}.conv1"),
                            (cout, cout, KERNEL),
                            dilation,
                        )?,
                        second: SnakeConv::load(
                            w,
                            &format!("{q}.snake2.alpha"),
                            &format!("{q}.conv2"),
                            (cout, cout, 1),
                            1,
                        )?,
                    })
                })
                .collect::<Result<_>>()?;
            blocks.push(Block {
                snake: Snake::load(w, &format!("{p}.snake.alpha"), c)?,
                up,
                residuals,
            });
            c = cout;
        }
        let output = SnakeConv::load(w, "output.snake.alpha", "output.conv", (1, c, KERNEL), 1)?;
        Ok(Self {
            width,
            codebooks,
            input,
            blocks,
            output,
        })
    }

    /// Loads `dac44_student.safetensors` (width 512).
    ///
    /// # Errors
    /// Fails when the file is missing or invalid.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let w = Weights::load(path)?;
        Self::from_weights(&w, 512)
            .with_context(|| format!("building the decoder from {}", path.display()))
    }

    /// Channels after the input conv.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The folded input of frames `start..end`: `[frames, 81]`.
    fn input_frames(&self, codes: &[u16], num_codebooks: usize, start: usize, end: usize) -> Seq {
        let cin = CODEBOOKS * IN_PER_CODEBOOK;
        let mut x = Seq::zeros(end - start, cin);
        for (row, frame) in x
            .data_mut()
            .chunks_exact_mut(cin)
            .zip(codes[start * num_codebooks..end * num_codebooks].chunks_exact(num_codebooks))
        {
            for (i, (&code, dst)) in frame
                .iter()
                .zip(row.as_chunks_mut::<IN_PER_CODEBOOK>().0)
                .enumerate()
            {
                let entry = (i * CODEBOOK_SIZE + usize::from(code)) * CODEBOOK_DIM;
                dst[..CODEBOOK_DIM].copy_from_slice(&self.codebooks[entry..entry + CODEBOOK_DIM]);
                dst[CODEBOOK_DIM] = 1.0;
            }
        }
        x
    }

    /// Decodes one span of frames as if it were the whole file: `[frames * 512]` samples.
    fn decode_span(&self, x: &Seq) -> Result<Vec<f32>> {
        let c = x.channels();
        let pad = (KERNEL - 1) / 2;
        let mut padded = Seq::zeros(x.len() + 2 * pad, c);
        padded.data_mut()[pad * c..(pad + x.len()) * c].copy_from_slice(x.data());
        let mut h = self.input.apply(&padded)?;
        drop(padded);
        for block in &self.blocks {
            h = block.forward(h)?;
        }
        let mut y = self.output.forward(&h)?.into_data();
        for v in &mut y {
            *v = v.tanh();
        }
        Ok(y)
    }

    /// Decodes codes `[frames][num_codebooks]` (row-major) to mono samples,
    /// `frames * 512` of them, chunk by chunk.
    ///
    /// # Errors
    /// Fails when `num_codebooks` is not 1 to 9, a code is out of range or the
    /// length is not a whole number of frames.
    pub fn decode(&self, codes: &[u16], num_codebooks: usize) -> Result<Vec<f32>> {
        self.decode_chunked(codes, num_codebooks, CHUNK)
    }

    /// [`DacDecoder::decode`] with a given chunk length, for tests.
    ///
    /// # Errors
    /// See [`DacDecoder::decode`].
    pub fn decode_chunked(
        &self,
        codes: &[u16],
        num_codebooks: usize,
        chunk: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            (1..=CODEBOOKS).contains(&num_codebooks),
            "{num_codebooks} codebooks; the model has 1 to {CODEBOOKS}"
        );
        ensure!(
            codes.len().is_multiple_of(num_codebooks),
            "codes are not a whole number of frames"
        );
        ensure!(
            codes.iter().all(|&c| usize::from(c) < CODEBOOK_SIZE),
            "a code is out of range"
        );
        ensure!(chunk > 0, "chunk length must be positive");
        let frames = codes.len() / num_codebooks;
        let mut out = Vec::with_capacity(frames * HOP);
        let mut start = 0;
        while start < frames {
            let end = (start + chunk).min(frames);
            // Context is clipped at the file's ends, where the whole-file
            // decode pads with zeros too.
            let lo = start.saturating_sub(CONTEXT);
            let hi = (end + CONTEXT).min(frames);
            let x = self.input_frames(codes, num_codebooks, lo, hi);
            let y = self.decode_span(&x)?;
            out.extend_from_slice(&y[(start - lo) * HOP..(end - lo) * HOP]);
            start = end;
        }
        Ok(out)
    }
}
