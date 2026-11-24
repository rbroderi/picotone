//! The HQ codec (`dac44-student`): DAC 44 kHz's own encoder and quantiser
//! ([`DacEncoder`]), and a 512-wide student of its decoder ([`DacDecoder`]).
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
use rayon::prelude::*;

use crate::fastconv::FastConv;
use crate::hqstream::{Context, HqDecoder};
use crate::kernels;
use crate::nn::{ConvGeometry, ConvKernel, ConvTranspose1d, Seq};
use crate::weights::Weights;
use crate::winograd::WinogradConv;

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Samples per latent frame.
pub const HOP: usize = 512;
/// Codebooks per frame at the highest tier.
pub const CODEBOOKS: usize = 9;
/// Entries per codebook.
pub const CODEBOOK_SIZE: usize = 1024;
/// Values per codebook entry.
pub(crate) const CODEBOOK_DIM: usize = 8;
/// Folded input values per codebook: the entry, then "present".
const IN_PER_CODEBOOK: usize = CODEBOOK_DIM + 1;
/// Folded input values per frame, all codebooks.
pub(crate) const FOLDED_INPUT: usize = CODEBOOKS * IN_PER_CODEBOOK;
/// Upsampling of each block.
const RATES: [usize; 4] = [8, 8, 4, 2];
/// Dilations of each block's residual units.
const DILATIONS: [usize; 3] = [1, 3, 9];
/// Kernel size of every non-transposed conv except the 1x1s.
const KERNEL: usize = 7;
/// Latent frames of context on each side of a chunk. The receptive field is
/// a little under 10 frames each way; the chunking test checks this margin.
pub const CONTEXT: usize = 12;
/// A decoded span starts and ends at multiples of this many frames.
///
/// The Winograd tiles start at a span's first output and hold 4 outputs of
/// one dilation phase, so they repeat every `4 x 9` samples of the most
/// dilated layer. Every stage's rate (8, 64, 256 and 512 samples per frame)
/// times 9 frames is a multiple of 36, so spans on this grid tile every
/// layer as a whole-file decode does (a span's last, partial tiles too), and
/// a chunked decode equals it bit for bit.
pub const ALIGN: usize = 9;
/// Latent frames per chunk, before context (about 3 s).
pub const CHUNK: usize = 256;
/// Most frames (with their context) decoded at once across threads. The
/// last block's activations take 64 KiB per frame and copy, three or four
/// at a time, so this bounds the decode's memory at about 150 MiB above the
/// weights however many threads there are; past that, the threads share
/// each piece's convs. Unbounded, a 16-thread wasm decode of 30 s stereo
/// peaked at 430–670 MiB, against 215–230 MiB at 512 and at the same speed
/// within noise; 256 was about 15% slower.
const FRAMES_IN_FLIGHT: usize = 512;

/// FNV-1a 64 of codebook values as little-endian f32, `[codebook][entry][value]`.
///
/// A stream stores this hash of the codebooks its codes index. It names DAC's
/// quantiser, not a decoder, so any decoder of DAC codes plays the stream; the
/// encoder and decoder files each carry a copy of the codebooks.
pub(crate) fn codebook_hash<'a>(books: impl IntoIterator<Item = &'a f32>) -> u64 {
    books
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// Snake's per-channel parameters.
#[derive(Debug, Clone)]
struct Snake {
    alpha: Vec<f32>,
    /// `1 / (alpha + 1e-9)`, computed as PyTorch does, in single precision.
    inv: Vec<f32>,
}

impl Snake {
    fn load(w: &Weights<'_>, name: &str, channels: usize) -> Result<Self> {
        let alpha = w.get(name, &[channels])?;
        let inv = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
        Ok(Self { alpha, inv })
    }

    fn apply(&self, values: &mut [f32]) {
        let _prof = crate::prof::scope("snake", self.alpha.len(), 0, values.len());
        kernels::snake_in_place(values, &self.alpha, &self.inv);
    }
}

/// A conv that keeps its input's length (zero padding `dilation * 3` on each
/// side), with Snake on its input.
#[derive(Debug, Clone)]
struct SnakeConv {
    snake: Snake,
    conv: Conv,
    pad: usize,
}

/// Which kernel a conv runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Packed,
    Fast,
}

/// The stride-1 k7 convs run on Winograd in both models. Of the rest, the
/// decoder's are packed and the encoder's run on faer (faster there; see
/// [`crate::fastconv`]). All match PyTorch to float rounding.
#[derive(Debug, Clone)]
enum Conv {
    Packed(ConvKernel),
    Fast(FastConv),
    /// The residual units' k7 convs, in both models (see [`crate::winograd`]).
    Winograd(WinogradConv),
}

impl Conv {
    fn apply(&self, x: &Seq) -> Result<Seq> {
        match self {
            Self::Packed(c) => c.apply(x),
            Self::Fast(c) => c.apply(x),
            Self::Winograd(c) => c.apply(x),
        }
    }
}

/// Channels in and out from which a stride-1 k7 conv runs on Winograd; the
/// narrow ends of the models gain little from it.
const WINOGRAD_MIN_CHANNELS: usize = 16;

impl SnakeConv {
    /// A stride-1 conv that keeps its input's length.
    fn load(
        w: &Weights<'_>,
        snake: &str,
        conv: &str,
        (cout, cin, kernel): (usize, usize, usize),
        dilation: usize,
        backend: Backend,
    ) -> Result<Self> {
        Self::build(
            w,
            (snake, conv),
            (cout, cin, kernel),
            (1, dilation),
            (kernel - 1) * dilation / 2,
            backend,
        )
    }

    /// The encoder's downsampling conv: kernel `2 * stride`, padding
    /// `stride / 2`, which divides the length by `stride` exactly.
    fn load_down(
        w: &Weights<'_>,
        snake: &str,
        conv: &str,
        (cout, cin): (usize, usize),
        stride: usize,
    ) -> Result<Self> {
        Self::build(
            w,
            (snake, conv),
            (cout, cin, 2 * stride),
            (stride, 1),
            stride.div_ceil(2),
            Backend::Fast,
        )
    }

    fn build(
        w: &Weights<'_>,
        (snake, conv): (&str, &str),
        (cout, cin, kernel): (usize, usize, usize),
        (stride, dilation): (usize, usize),
        pad: usize,
        backend: Backend,
    ) -> Result<Self> {
        let weight = w.get(&format!("{conv}.weight"), &[cout, cin, kernel])?;
        let bias = w.get(&format!("{conv}.bias"), &[cout])?;
        Ok(Self {
            snake: Snake::load(w, snake, cin)?,
            conv: match backend {
                _ if kernel == KERNEL && stride == 1 && cin.min(cout) >= WINOGRAD_MIN_CHANNELS => {
                    Conv::Winograd(WinogradConv::from_torch(
                        &weight,
                        &bias,
                        (cout, cin),
                        dilation,
                    )?)
                }
                Backend::Packed => Conv::Packed(ConvKernel::from_torch(
                    &weight,
                    &bias,
                    (cout, cin, kernel),
                    stride,
                    dilation,
                )?),
                Backend::Fast => Conv::Fast(FastConv::from_torch(
                    &weight,
                    &bias,
                    (cout, cin, kernel),
                    stride,
                    dilation,
                )?),
            },
            pad,
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

    /// As [`SnakeConv::forward`], consuming `x`: an unpadded conv applies
    /// Snake in place rather than to a copy (the same values, one buffer
    /// fewer).
    fn forward_owned(&self, mut x: Seq) -> Result<Seq> {
        if self.pad > 0 {
            return self.forward(&x);
        }
        self.snake.apply(x.data_mut());
        self.conv.apply(&x)
    }
}

/// `x + conv2(snake2(conv1(snake1(x))))`.
#[derive(Debug, Clone)]
struct Residual {
    first: SnakeConv,
    second: SnakeConv,
}

impl Residual {
    /// Residual unit `r` of the block at `prefix`, `channels` wide.
    fn load(
        w: &Weights<'_>,
        prefix: &str,
        channels: usize,
        dilation: usize,
        backend: Backend,
    ) -> Result<Self> {
        Ok(Self {
            first: SnakeConv::load(
                w,
                &format!("{prefix}.snake1.alpha"),
                &format!("{prefix}.conv1"),
                (channels, channels, KERNEL),
                dilation,
                backend,
            )?,
            second: SnakeConv::load(
                w,
                &format!("{prefix}.snake2.alpha"),
                &format!("{prefix}.conv2"),
                (channels, channels, 1),
                1,
                backend,
            )?,
        })
    }

    /// The three residual units of the block at `prefix`.
    fn load_three(
        w: &Weights<'_>,
        prefix: &str,
        channels: usize,
        backend: Backend,
    ) -> Result<Vec<Self>> {
        DILATIONS
            .iter()
            .enumerate()
            .map(|(r, &dilation)| {
                Self::load(w, &format!("{prefix}.res{r}"), channels, dilation, backend)
            })
            .collect()
    }

    fn forward(&self, x: Seq) -> Result<Seq> {
        let y = self.second.forward_owned(self.first.forward(&x)?)?;
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
    pub fn from_weights(w: &Weights<'_>, width: usize) -> Result<Self> {
        ensure!(
            width.is_multiple_of(1 << RATES.len()),
            "width {width} does not halve {} times",
            RATES.len()
        );
        let codebooks = w.get("codebooks", &[CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM])?;
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
            let residuals = Residual::load_three(w, &p, cout, Backend::Packed)?;
            blocks.push(Block {
                snake: Snake::load(w, &format!("{p}.snake.alpha"), c)?,
                up,
                residuals,
            });
            c = cout;
        }
        let output = SnakeConv::load(
            w,
            "output.snake.alpha",
            "output.conv",
            (1, c, KERNEL),
            1,
            Backend::Packed,
        )?;
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

    /// Hash of the codebooks the decoder reads codes with; a stream must name
    /// the same.
    #[must_use]
    pub fn codebook_hash(&self) -> u64 {
        codebook_hash(&self.codebooks)
    }

    /// The folded input of frames `start..end`: `[frames, 81]`.
    fn input_frames(&self, codes: &[u16], num_codebooks: usize, start: usize, end: usize) -> Seq {
        let mut x = Seq::zeros(end - start, FOLDED_INPUT);
        write_folded_input(
            &self.codebooks,
            &codes[start * num_codebooks..end * num_codebooks],
            num_codebooks,
            x.data_mut(),
        );
        x
    }

    /// Decodes one span of frames as if it were the whole file: `[frames * 512]` samples.
    fn decode_span(&self, x: &Seq) -> Result<Vec<f32>> {
        let mut h = conv_same(&self.input, x)?;
        for block in &self.blocks {
            h = block.forward(h)?;
        }
        let mut y = self.output.forward(&h)?.into_data();
        for v in &mut y {
            *v = v.tanh();
        }
        Ok(y)
    }

    /// Builds the decoder from the bytes of `dac44_student.safetensors`.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid decoder.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let w = Weights::from_bytes(bytes)?;
        Self::from_weights(&w, 512).context("building the HQ decoder")
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

    /// [`DacDecoder::decode`] with a given chunk length. Every chunk length
    /// gives the same samples, bit for bit (see [`ALIGN`]).
    ///
    /// # Errors
    /// See [`DacDecoder::decode`].
    pub fn decode_chunked(
        &self,
        codes: &[u16],
        num_codebooks: usize,
        chunk: usize,
    ) -> Result<Vec<f32>> {
        check_codes(codes, num_codebooks)?;
        ensure!(chunk > 0, "chunk length must be positive");
        let frames = codes.len() / num_codebooks;
        // Chunks run in parallel: much of a chunk's work (Snake, padding,
        // the transposed convs) is element-wise and runs on one thread, so
        // splitting only the matrix products leaves most cores idle.
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
}

/// Writes the folded input of `codes` (`[frames][num_codebooks]`) into the
/// zeroed `out`, `[frames][81]`: for each codebook present, its entry's 8
/// values and a 1. `codebooks` is `[codebook][entry][value]`.
pub(crate) fn write_folded_input(
    codebooks: &[f32],
    codes: &[u16],
    num_codebooks: usize,
    out: &mut [f32],
) {
    for (row, frame) in out
        .as_chunks_mut::<FOLDED_INPUT>()
        .0
        .iter_mut()
        .zip(codes.chunks_exact(num_codebooks))
    {
        for (i, (&code, dst)) in frame
            .iter()
            .zip(row.as_chunks_mut::<IN_PER_CODEBOOK>().0)
            .enumerate()
        {
            let entry = (i * CODEBOOK_SIZE + usize::from(code)) * CODEBOOK_DIM;
            dst[..CODEBOOK_DIM].copy_from_slice(&codebooks[entry..entry + CODEBOOK_DIM]);
            dst[CODEBOOK_DIM] = 1.0;
        }
    }
}

/// Checks that `codes` are whole frames of `num_codebooks` valid codes.
pub(crate) fn check_codes(codes: &[u16], num_codebooks: usize) -> Result<()> {
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
    Ok(())
}

impl HqDecoder for DacDecoder {
    fn name(&self) -> &'static str {
        "dac44_student"
    }

    fn codebook_hash(&self) -> u64 {
        Self::codebook_hash(self)
    }

    fn context(&self) -> Context {
        Context {
            frames: CONTEXT,
            align: ALIGN,
        }
    }

    fn frames_in_flight(&self) -> usize {
        FRAMES_IN_FLIGHT
    }

    fn decode_window(&self, codes: &[u16], num_codebooks: usize) -> Result<Vec<f32>> {
        check_codes(codes, num_codebooks)?;
        let x = self.input_frames(codes, num_codebooks, 0, codes.len() / num_codebooks);
        self.decode_span(&x)
    }
}

/// A plain k7 conv that keeps its input's length (zero padding 3 each side).
fn conv_same(kernel: &ConvKernel, x: &Seq) -> Result<Seq> {
    let c = x.channels();
    let pad = (KERNEL - 1) / 2;
    let mut padded = Seq::zeros(x.len() + 2 * pad, c);
    padded.data_mut()[pad * c..(pad + x.len()) * c].copy_from_slice(x.data());
    kernel.apply(&padded)
}

/// Downsampling of each encoder block.
const ENCODER_STRIDES: [usize; 4] = [2, 4, 8, 8];
/// Channels after the encoder's input conv.
const ENCODER_DIM: usize = 64;
/// Channels of the latent the quantiser reads.
const LATENT_DIM: usize = ENCODER_DIM << ENCODER_STRIDES.len();

/// Three residual units, then Snake and a strided conv that doubles the width.
#[derive(Debug, Clone)]
struct EncoderBlock {
    residuals: Vec<Residual>,
    down: SnakeConv,
}

/// One stage of DAC's residual vector quantiser.
#[derive(Debug, Clone)]
struct Quantizer {
    /// 1x1 conv, latent -> 8.
    in_proj: ConvKernel,
    /// `[entry][value]`, as stored.
    codebook: Vec<f32>,
    /// `[entry][value]`, each entry scaled to unit length.
    unit: Vec<f32>,
    /// Squared length of each unit entry (1 up to rounding, kept for parity
    /// with DAC's distance).
    unit_sq: Vec<f32>,
    /// 1x1 conv, 8 -> latent.
    out_proj: ConvKernel,
}

impl Quantizer {
    fn load(w: &Weights<'_>, i: usize) -> Result<Self> {
        let p = format!("quant{i}");
        let conv = |name: &str, cout: usize, cin: usize| -> Result<ConvKernel> {
            ConvKernel::from_torch(
                &w.get(&format!("{p}.{name}.weight"), &[cout, cin, 1])?,
                &w.get(&format!("{p}.{name}.bias"), &[cout])?,
                (cout, cin, 1),
                1,
                1,
            )
        };
        let shape = [CODEBOOK_SIZE, CODEBOOK_DIM];
        let unit = w.get(&format!("{p}.codebook_unit"), &shape)?;
        let unit_sq = unit
            .as_chunks::<CODEBOOK_DIM>()
            .0
            .iter()
            .map(|e| e.iter().map(|v| v * v).sum())
            .collect();
        Ok(Self {
            in_proj: conv("in_proj", CODEBOOK_DIM, LATENT_DIM)?,
            codebook: w.get(&format!("{p}.codebook"), &shape)?,
            unit,
            unit_sq,
            out_proj: conv("out_proj", LATENT_DIM, CODEBOOK_DIM)?,
        })
    }

    /// The nearest entry to each frame of `residual` (by cosine, as DAC
    /// does), written to `codes[t * stride]`; then subtracts the chosen
    /// entries' projections from `residual`.
    fn step(&self, residual: &mut Seq, codes: &mut [u16], stride: usize) -> Result<()> {
        let projected = self.in_proj.apply(residual)?;
        let mut chosen = Seq::zeros(residual.len(), CODEBOOK_DIM);
        for ((e, dst), code) in projected
            .data()
            .as_chunks::<CODEBOOK_DIM>()
            .0
            .iter()
            .zip(chosen.data_mut().as_chunks_mut::<CODEBOOK_DIM>().0)
            .zip(codes.iter_mut().step_by(stride))
        {
            let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let e = e.map(|v| v / norm);
            let mut best = (f32::NEG_INFINITY, 0);
            for (j, (u, &sq)) in self
                .unit
                .as_chunks::<CODEBOOK_DIM>()
                .0
                .iter()
                .zip(&self.unit_sq)
                .enumerate()
            {
                let dot: f32 = e.iter().zip(u).map(|(a, b)| a * b).sum();
                let score = 2.0 * dot - sq;
                if score > best.0 {
                    best = (score, j);
                }
            }
            *code = u16::try_from(best.1)?;
            let entry = best.1 * CODEBOOK_DIM;
            dst.copy_from_slice(&self.codebook[entry..entry + CODEBOOK_DIM]);
        }
        let sub = self.out_proj.apply(&chosen)?;
        for (r, s) in residual.data_mut().iter_mut().zip(sub.data()) {
            *r -= s;
        }
        Ok(())
    }
}

/// DAC 44 kHz's encoder and quantiser (MIT, Descript), as exported by
/// `distill/export_encoder.py`. The HQ codec's codes are DAC's own: the
/// student replaces only the decoder.
///
/// The encoder is a k7 conv to 64 channels, four blocks (three residual
/// units like the decoder's, then Snake and a conv of stride 2, 4, 8, 8 that
/// doubles the width), then Snake and a k3 conv to the 1024-wide latent.
/// The quantiser is 9 residual stages, each projecting to 8 values and
/// picking the entry with the largest cosine.
#[derive(Debug, Clone)]
pub struct DacEncoder {
    input: ConvKernel,
    blocks: Vec<EncoderBlock>,
    output: SnakeConv,
    quantizers: Vec<Quantizer>,
}

impl DacEncoder {
    /// Builds the encoder from `export_encoder.py`'s tensors.
    ///
    /// # Errors
    /// Fails when a tensor is missing or has the wrong shape.
    pub fn from_weights(w: &Weights<'_>) -> Result<Self> {
        let input = ConvKernel::from_torch(
            &w.get("input.weight", &[ENCODER_DIM, 1, KERNEL])?,
            &w.get("input.bias", &[ENCODER_DIM])?,
            (ENCODER_DIM, 1, KERNEL),
            1,
            1,
        )?;
        let mut blocks = Vec::with_capacity(ENCODER_STRIDES.len());
        let mut c = ENCODER_DIM;
        for (b, &stride) in ENCODER_STRIDES.iter().enumerate() {
            let p = format!("block{b}");
            blocks.push(EncoderBlock {
                residuals: Residual::load_three(w, &p, c, Backend::Fast)?,
                down: SnakeConv::load_down(
                    w,
                    &format!("{p}.snake.alpha"),
                    &format!("{p}.down"),
                    (2 * c, c),
                    stride,
                )?,
            });
            c *= 2;
        }
        let output = SnakeConv::load(
            w,
            "output.snake.alpha",
            "output.conv",
            (LATENT_DIM, c, 3),
            1,
            Backend::Fast,
        )?;
        let quantizers = (0..CODEBOOKS)
            .map(|i| Quantizer::load(w, i))
            .collect::<Result<_>>()?;
        Ok(Self {
            input,
            blocks,
            output,
            quantizers,
        })
    }

    /// Loads `dac44_encoder.safetensors`.
    ///
    /// # Errors
    /// Fails when the file is missing or invalid.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let w = Weights::load(path)?;
        Self::from_weights(&w)
            .with_context(|| format!("building the encoder from {}", path.display()))
    }

    /// Hash of the codebooks the codes index (the same as the decoder's).
    #[must_use]
    pub fn codebook_hash(&self) -> u64 {
        codebook_hash(self.quantizers.iter().flat_map(|q| &q.codebook))
    }

    /// The latent of one span of samples (a whole number of frames) as if it
    /// were the whole file: `[samples / 512, 1024]`.
    fn latent_span(&self, x: &Seq) -> Result<Seq> {
        let mut h = conv_same(&self.input, x)?;
        for block in &self.blocks {
            for residual in &block.residuals {
                h = residual.forward(h)?;
            }
            h = block.down.forward(&h)?;
        }
        self.output.forward(&h)
    }

    /// The latent of mono 44.1 kHz `audio`, zero padded at the end to a
    /// whole number of frames: `[frames, 1024]`, chunk by chunk.
    ///
    /// # Errors
    /// Fails when `chunk` is zero.
    pub fn latent_chunked(&self, audio: &[f32], chunk: usize) -> Result<Seq> {
        ensure!(chunk > 0, "chunk length must be positive");
        let frames = audio.len().div_ceil(HOP);
        // Chunks are independent, so they run in parallel: one chunk's
        // activations then stay in a core's cache, where a single chunk
        // split across all cores would wait on memory.
        let parts = (0..frames.div_ceil(chunk))
            .into_par_iter()
            .map(|i| {
                let start = i * chunk;
                let end = (start + chunk).min(frames);
                let lo = start.saturating_sub(CONTEXT);
                let hi = (end + CONTEXT).min(frames);
                let mut x = Seq::zeros((hi - lo) * HOP, 1);
                let src = &audio[(lo * HOP).min(audio.len())..(hi * HOP).min(audio.len())];
                x.data_mut()[..src.len()].copy_from_slice(src);
                Ok(self.latent_span(&x)?.slice_rows(start - lo, end - lo))
            })
            .collect::<Result<Vec<_>>>()?;
        if parts.is_empty() {
            return Ok(Seq::zeros(0, LATENT_DIM));
        }
        Seq::concat(parts)
    }

    /// Codes of a latent: `[frames][9]`, row-major. A tier's codes are the
    /// first codebooks of each frame.
    ///
    /// # Errors
    /// Fails when the latent is not 1024 wide.
    pub fn quantize(&self, latent: Seq) -> Result<Vec<u16>> {
        ensure!(
            latent.channels() == LATENT_DIM,
            "latent is {} wide, expected {LATENT_DIM}",
            latent.channels()
        );
        let mut codes = vec![0; latent.len() * CODEBOOKS];
        let mut residual = latent;
        for (i, q) in self.quantizers.iter().enumerate() {
            q.step(&mut residual, &mut codes[i..], CODEBOOKS)?;
        }
        Ok(codes)
    }

    /// Encodes mono 44.1 kHz `audio` to codes `[frames][9]` (row-major),
    /// `ceil(len / 512)` frames.
    ///
    /// # Errors
    /// Fails on internal shape mismatches only.
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<u16>> {
        self.quantize(self.latent_chunked(audio, CHUNK)?)
    }
}
