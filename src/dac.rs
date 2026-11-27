//! The HQ codec's codes: its residual vector quantiser ([`DacQuantizer`]), and
//! the code-space helpers its decoder ([`crate::vocos::VocosDecoder`]) shares.
//!
//! The quantiser codes mono 44.1 kHz audio as up to 16 codebooks of 1024
//! entries per latent frame, one frame per 512 samples. picotone's encoder
//! ([`crate::stftenc`]) makes the 1024-wide latent. Codebooks 1-9 are DAC 44
//! kHz's own (descript-audio-codec, MIT, Copyright (c) 2023-present,
//! Descript); codebooks 10-16 were trained with the decoder on what DAC's
//! leave. A decoder reads the codes as a folded input: the codebook
//! projections are folded into its first conv, so each frame is 144 values:
//! for each codebook, its 8 embedding values and a 1 meaning "present".
//! Codebooks above the tier give 9 zeros.

use anyhow::{Context as _, Result, ensure};
use rayon::prelude::*;

use crate::nn::{ConvKernel, Seq};
use crate::weights::Weights;

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Samples per latent frame.
pub const HOP: usize = 512;
/// Codebooks per frame at the highest tier.
pub const CODEBOOKS: usize = 16;
/// DAC's own codebooks, the first of each frame.
pub const DAC_CODEBOOKS: usize = 9;
/// Codebooks an encode uses unless told otherwise.
pub const DEFAULT_CODEBOOKS: usize = DAC_CODEBOOKS;
/// Entries per codebook.
pub const CODEBOOK_SIZE: usize = 1024;
/// Values per codebook entry.
pub(crate) const CODEBOOK_DIM: usize = 8;
/// Values per frame of the latent the quantiser reads.
pub const LATENT_DIM: usize = 1024;
/// Latent frames the quantiser codes per parallel piece.
const QUANTIZE_PIECE: usize = 256;
/// Folded input values per codebook: the entry, then "present".
const IN_PER_CODEBOOK: usize = CODEBOOK_DIM + 1;
/// Folded input values per frame, all codebooks.
pub(crate) const FOLDED_INPUT: usize = CODEBOOKS * IN_PER_CODEBOOK;

/// FNV-1a 64 of codebook values as little-endian f32, `[codebook][entry][value]`.
///
/// A stream stores this hash of all 16 codebooks its codes index. It names
/// the quantiser, not an encoder or a decoder, so any decoder of these codes
/// plays the stream; the quantiser and decoder files each carry a copy of the
/// codebooks.
pub(crate) fn codebook_hash<'a>(books: impl IntoIterator<Item = &'a f32>) -> u64 {
    books
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// Writes the folded input of `codes` (`[frames][num_codebooks]`) into the
/// zeroed `out`, `[frames][144]`: for each codebook present, its entry's 8
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

/// One stage of DAC's residual vector quantiser.
#[derive(Debug, Clone)]
struct Stage {
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

impl Stage {
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
    /// entries' projections from `residual` and adds them to `sum`.
    fn step(
        &self,
        residual: &mut Seq,
        sum: &mut Seq,
        codes: &mut [u16],
        stride: usize,
    ) -> Result<()> {
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
        for ((r, z), s) in residual
            .data_mut()
            .iter_mut()
            .zip(sum.data_mut())
            .zip(sub.data())
        {
            *r -= s;
            *z += s;
        }
        Ok(())
    }
}

/// The HQ codec's residual vector quantiser: DAC 44 kHz's 9 stages (MIT,
/// Descript), then 7 of picotone's.
///
/// Exported by `tools/export_quantizer.py`: 16 stages, each projecting the
/// residual to 8 values and picking the entry with the largest cosine. The
/// first 9 codes of a frame are DAC's own. Stage 10 reads the residual
/// recomputed as the latent minus the sum of stages 1-9's projections, as the
/// training code does (the running residual differs by rounding, which can
/// flip near ties).
#[derive(Debug, Clone)]
pub struct DacQuantizer {
    stages: Vec<Stage>,
}

impl DacQuantizer {
    /// Builds the quantiser from `export_quantizer.py`'s tensors.
    ///
    /// # Errors
    /// Fails when a tensor is missing or has the wrong shape.
    pub fn from_weights(w: &Weights<'_>) -> Result<Self> {
        let stages = (0..CODEBOOKS)
            .map(|i| Stage::load(w, i))
            .collect::<Result<_>>()?;
        Ok(Self { stages })
    }

    /// Loads `dac44_quantizer.safetensors`.
    ///
    /// # Errors
    /// Fails when the file is missing or invalid.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let w = Weights::load(path)?;
        Self::from_weights(&w)
            .with_context(|| format!("building the quantiser from {}", path.display()))
    }

    /// Hash of the codebooks the codes index (the same as the decoder's,
    /// [`crate::hqstream::HqDecoder::codebook_hash`]).
    #[must_use]
    pub fn codebook_hash(&self) -> u64 {
        codebook_hash(self.stages.iter().flat_map(|q| &q.codebook))
    }

    /// Codes of a latent: `[frames][16]`, row-major. A tier's codes are the
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
        if latent.len() <= QUANTIZE_PIECE {
            return self.quantize_piece(latent);
        }
        // Frames are quantised independently, so pieces run in parallel.
        let pieces = (0..latent.len().div_ceil(QUANTIZE_PIECE))
            .into_par_iter()
            .map(|i| {
                let start = i * QUANTIZE_PIECE;
                let end = (start + QUANTIZE_PIECE).min(latent.len());
                self.quantize_piece(latent.slice_rows(start, end))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(pieces.concat())
    }

    fn quantize_piece(&self, latent: Seq) -> Result<Vec<u16>> {
        let mut codes = vec![0; latent.len() * CODEBOOKS];
        let mut sum = Seq::zeros(latent.len(), LATENT_DIM);
        let mut residual = latent.clone();
        for (i, q) in self.stages.iter().enumerate() {
            if i == DAC_CODEBOOKS {
                for ((r, z), s) in residual.data_mut().iter_mut().zip(latent.data()).zip(sum.data()) {
                    *r = z - s;
                }
            }
            q.step(&mut residual, &mut sum, &mut codes[i..], CODEBOOKS)?;
        }
        Ok(codes)
    }
}
