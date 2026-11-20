//! cx2: deterministic entropy coding of EnCodec code streams.
//!
//! A small model (see [`model`] and [`features`]) predicts each code from the
//! codes already coded, in integer arithmetic only, and a range coder (see
//! [`rangecoder`]) turns those predictions into bits.
//!
//! A stream is cut into chunks, one per Ogg page of a `.pico` file (see
//! [`crate::pico`]). Every chunk is coded on its own: the predictor forgets
//! everything and the range coder restarts, so a decoder can start at any
//! chunk and a damaged chunk cannot hurt the others. A chunk first holds the
//! 12-bit volume scale of each of its segments (see [`crate::scale`]), coded
//! as uniform symbols, then its codes, frame-major.
//!
//! The tables of the model live in `.ecxm` files; [`fit`] holds the offline
//! steps that build them.

pub mod features;
pub mod fit;
pub mod kernels;
pub mod model;
pub mod net;
pub mod rangecoder;

use anyhow::{Context as _, Result, ensure};

pub use crate::cx2::model::Model;
use crate::cx2::model::Predictor;
use crate::cx2::rangecoder::{Decoder, Encoder};
use crate::scale;

/// Name of the scheme.
pub const SCHEME: &str = "cx2";

/// Range-codes one chunk with a freshly reset predictor: `scales` (12-bit
/// indices), then `codes` (frame-major, `num_codebooks` per frame).
///
/// # Errors
/// Fails when the model does not cover the stream or a code or scale is out of range.
pub fn encode_chunk(
    model: &Model,
    num_codebooks: usize,
    scales: &[u16],
    codes: &[u16],
) -> Result<Vec<u8>> {
    encode_with(&mut model.predictor(num_codebooks)?, scales, codes)
}

fn encode_with(predictor: &mut Predictor<'_>, scales: &[u16], codes: &[u16]) -> Result<Vec<u8>> {
    predictor.reset();
    let mut encoder = Encoder::new();
    for &index in scales {
        ensure!(index < scale::LEVELS, "scale index {index} is out of range");
        encoder.encode(u64::from(index), 1, u64::from(scale::LEVELS))?;
    }
    for &code in codes {
        let (freqs, total) = predictor.frequencies();
        let index = usize::from(code);
        ensure!(index < freqs.len(), "code {code} is out of range");
        let cum: u64 = freqs[..index].iter().map(|&f| u64::from(f)).sum();
        encoder.encode(cum, u64::from(freqs[index]), total)?;
        predictor.update(code)?;
    }
    Ok(encoder.finish())
}

/// Decodes one chunk holding `scale_count` scales and `code_count` codes.
///
/// # Errors
/// Fails when the model does not cover the stream or the chunk is damaged or truncated.
pub fn decode_chunk(
    model: &Model,
    num_codebooks: usize,
    scale_count: usize,
    code_count: usize,
    bytes: &[u8],
) -> Result<(Vec<u16>, Vec<u16>)> {
    let mut predictor = model.predictor(num_codebooks)?;
    let mut decoder = Decoder::new(bytes);
    // The counts come from a header; reserve only what the bytes could plausibly hold.
    let plausible = bytes.len().saturating_mul(8) + 64;
    let mut scales = Vec::with_capacity(scale_count.min(plausible));
    for _ in 0..scale_count {
        let index = decoder.target(u64::from(scale::LEVELS))?;
        decoder.consume(index, 1);
        ensure!(!decoder.overrun(), "entropy-coded chunk is truncated");
        scales.push(u16::try_from(index).context("corrupt scale")?);
    }
    let mut codes = Vec::with_capacity(code_count.min(plausible));
    for _ in 0..code_count {
        let (freqs, total) = predictor.frequencies();
        let target = decoder.target(total)?;
        let mut cum = 0_u64;
        let mut found = None;
        for (index, &freq) in freqs.iter().enumerate() {
            let next = cum + u64::from(freq);
            if target < next {
                found = Some((index, freq));
                break;
            }
            cum = next;
        }
        let (index, freq) = found.context("corrupt entropy-coded stream")?;
        decoder.consume(cum, u64::from(freq));
        // Without this a damaged frame count would keep decoding codes out of nothing.
        ensure!(!decoder.overrun(), "entropy-coded chunk is truncated");
        let code = u16::try_from(index)?;
        predictor.update(code)?;
        codes.push(code);
    }
    Ok((scales, codes))
}

/// Range-codes `codes` (frame-major, `num_codebooks` per frame) as independent
/// chunks of `chunk_frames` frames each (one chunk when zero), without scales.
///
/// This is the raw coder the fitting tools measure with; files use [`encode_chunk`].
///
/// # Errors
/// Fails when the model does not cover the stream or a code is out of range.
pub fn encode_codes(
    model: &Model,
    num_codebooks: usize,
    chunk_frames: usize,
    codes: &[u16],
) -> Result<Vec<Vec<u8>>> {
    let mut predictor = model.predictor(num_codebooks)?;
    let len = if chunk_frames == 0 {
        codes.len().max(1)
    } else {
        chunk_frames.saturating_mul(num_codebooks)
    };
    codes
        .chunks(len)
        .map(|chunk| encode_with(&mut predictor, &[], chunk))
        .collect()
}

/// Converts a count to `f64` for reporting; counts here stay far below 2^52.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "code and byte counts are far below 2^52"
)]
pub const fn to_f64(value: usize) -> f64 {
    value as f64
}
