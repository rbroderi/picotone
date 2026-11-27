//! The HQ model: integer prediction of DAC 44 kHz codes (HQ streams, model 2).
//!
//! One trained network predicts each code of a coded channel from that
//! channel's previous frames and the codebooks already coded in the current
//! frame. Channels are independent, and so is every page: the predictor
//! forgets everything at a page start (see [`crate::pico::DacStream`]).
//!
//! ```text
//! e[q][c]  table embedding of code c of codebook q, DIM i16 values, |e| <= 1023
//! x        e of codebooks 0..n of the H previous frames of this page, most recent
//!          first (zero for frames before the page start and codebooks past n)   H x 9 x DIM
//! h1       clamp((W1 x + b1 + r[role]) >> s1, 0, 1023)                          once per frame
//! h2       clamp((W2 h1 + b2) >> s2, 0, 1023)                                   once per frame
//! g_k      clamp((A_k [h2, e of codebooks 0..k of this frame] + c_k) >> sa_k, 0, 1023)
//! logit_j  d_k[j] + ((O_k g_k)[j] >> so_k)                     units of 2^-frac_bits bits
//! freq_j   (EXP[frac] >> int) + floor of the distance below the best logit, as in cx2
//! ```
//!
//! Weights are `i16`. A layer with `n` inputs and a bias accepts weights up to
//! `(2^31 - 1 - 2^29) / (n x 1023)` and biases up to `2^29` (for the first
//! layer, `b1 + r` together); an output layer `O_k` accepts `(2^31 - 1) / (n x 1023)`.
//! With activations and embeddings at most 1023 in magnitude, no `i32`
//! accumulator can overflow, so every machine computes the same integers.
//!
//! The table (`.ecxm`, magic `ECXH`) is self-contained and its FNV-1a 64 hash
//! names it in HQ stream headers. `train/hq_net.py` trains the network and
//! writes the table; nothing in it is specific to one encoder beyond what it
//! learnt from that encoder's codes.

use anyhow::{Context as _, Result, ensure};

use crate::cx2::kernels::{Matrix, Simd};
use crate::cx2::rangecoder::{Decoder, Encoder};

/// Codebook size.
pub const CARD: usize = 1024;
/// Most codebooks per frame.
pub const CODEBOOKS: usize = 16;
const MAGIC: &[u8; 4] = b"ECXH";
const VERSION: u32 = 1;
/// Largest activation and embedding magnitude.
const LIMIT: i32 = 1023;
const BIAS_LIMIT: u32 = 1 << 29;
/// Codes per block of frequency sums, which the coder searches first.
const BLOCK: usize = 32;
/// Largest right shift applied to the exponential table.
const MAX_SHIFT: u32 = 31;
/// Largest shift a layer may declare.
const MAX_LAYER_SHIFT: u32 = 30;

/// What a coded channel carries; selects a bias of the first layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Left or right of a left/right stream.
    LeftRight = 0,
    /// A mono stream, the mid of a mid/side stream, or a parametric stream's mid.
    Mid = 1,
    /// The side of a mid/side stream.
    Side = 2,
}

/// Largest weight of a layer with `inputs` inputs and a bias.
fn hidden_limit(inputs: usize) -> i32 {
    let bound = (2_i64.pow(31) - 1 - i64::from(BIAS_LIMIT))
        / (i64::try_from(inputs.max(1)).unwrap_or(i64::MAX / 2048) * i64::from(LIMIT));
    i32::try_from(bound.min(32767)).unwrap_or(0)
}

/// Largest weight of an output layer with `inputs` inputs.
fn out_limit(inputs: usize) -> i32 {
    let bound = (2_i64.pow(31) - 1)
        / (i64::try_from(inputs.max(1)).unwrap_or(i64::MAX / 2048) * i64::from(LIMIT));
    i32::try_from(bound.min(32767)).unwrap_or(0)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("HQ table size overflow")?;
        let out = self
            .bytes
            .get(self.pos..end)
            .context("HQ table is truncated")?;
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn size(&mut self) -> Result<usize> {
        Ok(usize::try_from(self.u32()?)?)
    }

    fn shift(&mut self) -> Result<u32> {
        let s = self.u32()?;
        ensure!(s <= MAX_LAYER_SHIFT, "implausible HQ table shift {s}");
        Ok(s)
    }

    fn i16s(&mut self, n: usize, limit: i32) -> Result<Vec<i16>> {
        let values: Vec<i16> = self
            .take(n.checked_mul(2).context("HQ table size overflow")?)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect();
        ensure!(
            values.iter().all(|&v| i32::from(v).abs() <= limit),
            "HQ table weight is out of range"
        );
        Ok(values)
    }

    fn i32s(&mut self, n: usize, limit: u32) -> Result<Vec<i32>> {
        let values: Vec<i32> = self
            .take(n.checked_mul(4).context("HQ table size overflow")?)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b))
            .collect();
        ensure!(
            values.iter().all(|v| v.unsigned_abs() <= limit),
            "HQ table bias is out of range"
        );
        Ok(values)
    }

    fn matrix(&mut self, inputs: usize, rows: usize, limit: i32) -> Result<Matrix> {
        let columns = self.i16s(inputs * rows, limit)?;
        Matrix::from_columns(&columns, rows).context("implausible HQ layer shape")
    }
}

/// One codebook's output head.
#[derive(Debug)]
struct Head {
    /// `rank` rows; columns: the trunk's outputs, then this frame's earlier codebooks' embeddings.
    hidden: Matrix,
    bias: Vec<i32>,
    hidden_shift: u32,
    /// `CARD` rows, `rank` columns.
    out: Matrix,
    out_shift: u32,
    out_bias: Vec<i32>,
}

/// A loaded HQ table.
#[derive(Debug)]
pub struct HqModel {
    dim: usize,
    history: usize,
    h1: usize,
    h2: usize,
    rank: usize,
    frac_bits: u32,
    /// Frequency by distance below the best logit.
    freq: Vec<u32>,
    /// `[codebook][code][dim]`.
    emb: Vec<i16>,
    w1: Matrix,
    b1: Vec<i32>,
    /// One `h1`-long bias per [`Role`].
    roles: Vec<i32>,
    s1: u32,
    w2: Matrix,
    b2: Vec<i32>,
    s2: u32,
    heads: Vec<Head>,
    hash: u64,
    simd: Simd,
}

/// FNV-1a 64-bit hash.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The hash an HQ table file gives its model, without parsing it; `None` for other files.
#[must_use]
pub fn table_hash(table: &[u8]) -> Option<u64> {
    (table.get(..4)? == MAGIC).then(|| fnv1a(table))
}

impl HqModel {
    /// Parses an HQ table.
    ///
    /// # Errors
    /// Fails on a malformed or implausible table.
    pub fn from_table(table: &[u8]) -> Result<Self> {
        let mut r = Reader {
            bytes: table,
            pos: 0,
        };
        ensure!(r.take(4)? == MAGIC, "not an HQ cx2 table (bad magic)");
        let version = r.u32()?;
        ensure!(version == VERSION, "unsupported HQ table version {version}");
        let codebooks = r.size()?;
        let card = r.size()?;
        let dim = r.size()?;
        let history = r.size()?;
        let h1 = r.size()?;
        let h2 = r.size()?;
        let rank = r.size()?;
        let roles = r.size()?;
        ensure!(
            codebooks == CODEBOOKS
                && card == CARD
                && (1..=32).contains(&dim)
                && (1..=32).contains(&history)
                && (1..=1024).contains(&h1)
                && (1..=1024).contains(&h2)
                && (1..=256).contains(&rank)
                && roles == 3,
            "implausible HQ table dimensions"
        );
        let frac_bits = r.u32()?;
        let floor = r.u32()?;
        ensure!(
            frac_bits <= 12 && (1..=1 << 16).contains(&floor),
            "implausible HQ table parameters"
        );
        let exp = (0..1_usize << frac_bits)
            .map(|_| r.u32())
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            exp.iter().all(|&e| e <= 1 << 24),
            "exponential table entry is too large"
        );
        let freq: Vec<u32> = (0..=MAX_SHIFT)
            .flat_map(|shift| exp.iter().map(move |&e| (e >> shift) + floor))
            .collect();
        let emb = r.i16s(CODEBOOKS * CARD * dim, LIMIT)?;
        let inputs = history * CODEBOOKS * dim;
        let s1 = r.shift()?;
        let w1 = r.matrix(inputs, h1, hidden_limit(inputs))?;
        let b1 = r.i32s(h1, BIAS_LIMIT)?;
        let role_bias = r.i32s(roles * h1, BIAS_LIMIT)?;
        ensure!(
            role_bias.chunks_exact(h1).all(|row| row
                .iter()
                .zip(&b1)
                .all(|(a, b)| a.unsigned_abs() + b.unsigned_abs() <= BIAS_LIMIT)),
            "HQ table bias is out of range"
        );
        let s2 = r.shift()?;
        let w2 = r.matrix(h1, h2, hidden_limit(h1))?;
        let b2 = r.i32s(h2, BIAS_LIMIT)?;
        let mut heads = Vec::with_capacity(CODEBOOKS);
        for k in 0..CODEBOOKS {
            let hidden_shift = r.shift()?;
            let out_shift = r.shift()?;
            let n = h2 + k * dim;
            heads.push(Head {
                hidden: r.matrix(n, rank, hidden_limit(n))?,
                bias: r.i32s(rank, BIAS_LIMIT)?,
                hidden_shift,
                out: r.matrix(rank, CARD, out_limit(rank))?,
                out_shift,
                out_bias: r.i32s(CARD, 1 << 20)?,
            });
        }
        ensure!(r.pos == table.len(), "HQ table has trailing bytes");
        Ok(Self {
            dim,
            history,
            h1,
            h2,
            rank,
            frac_bits,
            freq,
            emb,
            w1,
            b1,
            roles: role_bias,
            s1,
            w2,
            b2,
            s2,
            heads,
            hash: fnv1a(table),
            simd: Simd::new(),
        })
    }

    /// Hash of the table, as stored in HQ stream headers.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Fractional bits of a logit.
    #[must_use]
    pub const fn frac_bits(&self) -> u32 {
        self.frac_bits
    }

    /// Chooses the CPU dispatch of the dense kernels; every choice codes identically.
    pub const fn set_simd(&mut self, simd: Simd) {
        self.simd = simd;
    }

    /// Multiply-adds per frame of a channel with `codebooks` codebooks.
    #[must_use]
    pub fn macs_per_frame(&self, codebooks: usize) -> usize {
        let trunk = self.history * CODEBOOKS * self.dim * self.h1 + self.h1 * self.h2;
        trunk
            + (0..codebooks)
                .map(|k| (self.h2 + k * self.dim) * self.rank + self.rank * CARD)
                .sum::<usize>()
    }

    /// Starts predicting a channel of `codebooks` codes per frame.
    ///
    /// # Errors
    /// Fails when `codebooks` is not 1 to 16.
    pub fn predictor(&self, role: Role, codebooks: usize) -> Result<HqPredictor<'_>> {
        ensure!(
            (1..=CODEBOOKS).contains(&codebooks),
            "{codebooks} codebooks; the HQ model codes 1 to {CODEBOOKS}"
        );
        Ok(HqPredictor {
            model: self,
            role,
            codebooks,
            past: vec![0; self.history * CODEBOOKS * self.dim],
            current: vec![0; self.h2 + CODEBOOKS * self.dim],
            next: 0,
            ready: false,
            acc: vec![0; self.h1.max(self.h2).max(self.rank).max(CARD)],
            hidden: vec![0; self.h1.max(self.rank)],
            logits: vec![0; CARD],
            freqs: vec![0; CARD],
            blocks: vec![0; CARD / BLOCK],
        })
    }

    fn embedding(&self, codebook: usize, code: u16) -> &[i16] {
        let start = (codebook * CARD + usize::from(code)) * self.dim;
        &self.emb[start..start + self.dim]
    }
}

/// Clamps `acc >> shift` into the activation range.
fn activate(out: &mut [i16], acc: &[i32], shift: u32) {
    for (o, &a) in out.iter_mut().zip(acc) {
        *o = i16::try_from((a >> shift).clamp(0, LIMIT)).unwrap_or(0);
    }
}

/// Streaming predictor for one coded channel: frequencies of the next code, then the code.
#[derive(Debug)]
pub struct HqPredictor<'a> {
    model: &'a HqModel,
    role: Role,
    codebooks: usize,
    /// Embeddings of the previous frames of this page, most recent first; zero before the page start.
    past: Vec<i16>,
    /// The trunk's output, then the embeddings of this frame's codes so far.
    current: Vec<i16>,
    /// Codebook of the next code.
    next: usize,
    /// Whether the trunk's output for this frame is in `current`.
    ready: bool,
    acc: Vec<i32>,
    hidden: Vec<i16>,
    logits: Vec<i32>,
    freqs: Vec<u32>,
    /// Sum of each `BLOCK` frequencies (each at most `2^24 + 2^16`, so a block fits a `u32`).
    blocks: Vec<u32>,
}

impl HqPredictor<'_> {
    /// Forgets everything, as at a page start.
    pub fn reset(&mut self) {
        self.past.fill(0);
        self.next = 0;
        self.ready = false;
    }

    /// Codebook of the next code.
    #[must_use]
    pub const fn codebook(&self) -> usize {
        self.next
    }

    /// The integer logits of the next code, in units of `2^-frac_bits` bits.
    pub fn logits(&mut self) -> &[i32] {
        self.compute();
        &self.logits
    }

    fn compute(&mut self) {
        let m = self.model;
        let simd = m.simd;
        if !self.ready {
            let acc = &mut self.acc[..m.h1];
            let role = &m.roles[self.role as usize * m.h1..][..m.h1];
            for ((a, &b), &r) in acc.iter_mut().zip(&m.b1).zip(role) {
                *a = b + r;
            }
            simd.multiply_add(acc, &m.w1, &self.past);
            activate(&mut self.hidden[..m.h1], acc, m.s1);
            let acc = &mut self.acc[..m.h2];
            acc.copy_from_slice(&m.b2);
            simd.multiply_add(acc, &m.w2, &self.hidden[..m.h1]);
            activate(&mut self.current[..m.h2], acc, m.s2);
            self.ready = true;
        }
        let k = self.next;
        let head = &m.heads[k];
        let acc = &mut self.acc[..m.rank];
        acc.copy_from_slice(&head.bias);
        simd.multiply_add(acc, &head.hidden, &self.current[..m.h2 + k * m.dim]);
        activate(&mut self.hidden[..m.rank], acc, head.hidden_shift);
        let acc = &mut self.acc[..CARD];
        acc.fill(0);
        simd.multiply_add(acc, &head.out, &self.hidden[..m.rank]);
        for ((logit, &dot), &bias) in self.logits.iter_mut().zip(acc.iter()).zip(&head.out_bias) {
            *logit = bias + (dot >> head.out_shift);
        }
    }

    /// Frequencies of the next code (all positive) and their total.
    pub fn frequencies(&mut self) -> (&[u32], u64) {
        self.compute();
        let table = &self.model.freq;
        let last = table.len() - 1;
        let best = self.logits.iter().copied().max().unwrap_or(0);
        let mut total = 0_u64;
        for ((freqs, logits), block) in self
            .freqs
            .as_chunks_mut::<BLOCK>()
            .0
            .iter_mut()
            .zip(self.logits.as_chunks::<BLOCK>().0)
            .zip(&mut self.blocks)
        {
            let mut sum = 0_u32;
            for (freq, &logit) in freqs.iter_mut().zip(logits) {
                let below = usize::try_from(best.abs_diff(logit)).unwrap_or(usize::MAX);
                *freq = table[below.min(last)];
                sum += *freq;
            }
            *block = sum;
            total += u64::from(sum);
        }
        (&self.freqs, total)
    }

    /// The frequencies of the codes before `code`, summed, from the last [`Self::frequencies`].
    #[must_use]
    pub fn cumulative(&self, code: usize) -> u64 {
        let block = code / BLOCK;
        let whole: u64 = self.blocks[..block].iter().map(|&b| u64::from(b)).sum();
        whole
            + self.freqs[block * BLOCK..code]
                .iter()
                .map(|&f| u64::from(f))
                .sum::<u64>()
    }

    /// The code whose frequency range holds `target`, with the range's start and
    /// size, from the last [`Self::frequencies`]; `None` past the total.
    #[must_use]
    pub fn locate(&self, target: u64) -> Option<(usize, u64, u32)> {
        let mut cum = 0_u64;
        for (b, &sum) in self.blocks.iter().enumerate() {
            let next = cum + u64::from(sum);
            if target < next {
                for (i, &freq) in self.freqs[b * BLOCK..(b + 1) * BLOCK].iter().enumerate() {
                    let end = cum + u64::from(freq);
                    if target < end {
                        return Some((b * BLOCK + i, cum, freq));
                    }
                    cum = end;
                }
                return None;
            }
            cum = next;
        }
        None
    }

    /// Records the code just coded and moves to the next one.
    ///
    /// # Errors
    /// Fails when `code` is not a codebook entry.
    pub fn update(&mut self, code: u16) -> Result<()> {
        ensure!(usize::from(code) < CARD, "code {code} is out of range");
        let m = self.model;
        let k = self.next;
        let start = m.h2 + k * m.dim;
        self.current[start..start + m.dim].copy_from_slice(m.embedding(k, code));
        self.next += 1;
        if self.next == self.codebooks {
            // The finished frame becomes the most recent past frame.
            let frame = CODEBOOKS * m.dim;
            self.past.copy_within(..(m.history - 1) * frame, frame);
            let (head, _) = self.past.split_at_mut(frame);
            let n = self.codebooks * m.dim;
            head[..n].copy_from_slice(&self.current[m.h2..m.h2 + n]);
            head[n..].fill(0);
            self.next = 0;
            self.ready = false;
        }
        Ok(())
    }
}

/// Range-codes one page of one channel (`codes` frame-major, `codebooks` per frame)
/// with a fresh predictor.
///
/// # Errors
/// Fails when a code is out of range or `codebooks` is not 1 to 16.
pub fn encode_page(
    model: &HqModel,
    role: Role,
    codebooks: usize,
    codes: &[u16],
) -> Result<Vec<u8>> {
    let mut predictor = model.predictor(role, codebooks)?;
    let mut encoder = Encoder::new();
    for &code in codes {
        let index = usize::from(code);
        let (freqs, total) = predictor.frequencies();
        ensure!(index < freqs.len(), "code {code} is out of range");
        let freq = u64::from(freqs[index]);
        encoder.encode(predictor.cumulative(index), freq, total)?;
        predictor.update(code)?;
    }
    Ok(encoder.finish())
}

/// Decodes `count` codes of one page of one channel.
///
/// # Errors
/// Fails when the bytes are damaged or truncated.
pub fn decode_page(
    model: &HqModel,
    role: Role,
    codebooks: usize,
    count: usize,
    bytes: &[u8],
) -> Result<Vec<u16>> {
    let mut predictor = model.predictor(role, codebooks)?;
    let mut decoder = Decoder::new(bytes);
    // The count comes from a header; reserve only what the bytes could plausibly hold.
    let mut codes = Vec::with_capacity(count.min(bytes.len().saturating_mul(8) + 64));
    for _ in 0..count {
        let (_, total) = predictor.frequencies();
        let target = decoder.target(total)?;
        let (index, cum, freq) = predictor
            .locate(target)
            .context("corrupt entropy-coded page")?;
        decoder.consume(cum, u64::from(freq));
        ensure!(!decoder.overrun(), "entropy-coded page is truncated");
        let code = u16::try_from(index)?;
        predictor.update(code)?;
        codes.push(code);
    }
    Ok(codes)
}

/// Range-codes the layer part of one page of one channel.
///
/// Of each frame of `codes` (frame-major, `codebooks` per frame) it codes
/// codebooks `base..codebooks`. Codebooks `0..base` are fed to the predictor
/// but not coded, so it sees what coding the whole channel would show it.
/// Empty when `base == codebooks`.
///
/// # Errors
/// Fails when a code is out of range, `codebooks` is not 1 to 16, `base` is
/// more than `codebooks`, or `codes` is not whole frames.
pub fn encode_layer_page(
    model: &HqModel,
    role: Role,
    codebooks: usize,
    base: usize,
    codes: &[u16],
) -> Result<Vec<u8>> {
    ensure!(base <= codebooks, "a layer below its base");
    ensure!(
        codes.len().is_multiple_of(codebooks.max(1)),
        "layer codes are not whole frames"
    );
    if base == codebooks {
        return Ok(Vec::new());
    }
    let mut predictor = model.predictor(role, codebooks)?;
    let mut encoder = Encoder::new();
    for (i, &code) in codes.iter().enumerate() {
        if i % codebooks >= base {
            let index = usize::from(code);
            let (freqs, total) = predictor.frequencies();
            ensure!(index < freqs.len(), "code {code} is out of range");
            let freq = u64::from(freqs[index]);
            encoder.encode(predictor.cumulative(index), freq, total)?;
        }
        predictor.update(code)?;
    }
    Ok(encoder.finish())
}

/// Decodes the layer part of `frames` frames of one channel over its base
/// codes (`base` per frame); returns the whole codes, `codebooks` per frame.
///
/// # Errors
/// Fails when the bytes are damaged or truncated, or do not fit the counts.
pub fn decode_layer_page(
    model: &HqModel,
    role: Role,
    codebooks: usize,
    base: usize,
    base_codes: &[u16],
    frames: usize,
    bytes: &[u8],
) -> Result<Vec<u16>> {
    ensure!(base <= codebooks, "a layer below its base");
    ensure!(
        base_codes.len() == frames * base,
        "the base page has {} codes, expected {frames} frames x {base}",
        base_codes.len()
    );
    if base == codebooks {
        ensure!(bytes.is_empty(), "a layer chunk with nothing to code");
        return Ok(base_codes.to_vec());
    }
    let mut predictor = model.predictor(role, codebooks)?;
    let mut decoder = Decoder::new(bytes);
    let mut codes = Vec::with_capacity(
        (frames * codebooks).min(base_codes.len() + bytes.len().saturating_mul(8) + 64),
    );
    for frame in 0..frames {
        for k in 0..codebooks {
            let code = if k < base {
                base_codes[frame * base + k]
            } else {
                let (_, total) = predictor.frequencies();
                let target = decoder.target(total)?;
                let (index, cum, freq) = predictor
                    .locate(target)
                    .context("corrupt entropy-coded layer page")?;
                decoder.consume(cum, u64::from(freq));
                ensure!(!decoder.overrun(), "entropy-coded layer page is truncated");
                u16::try_from(index)?
            };
            predictor.update(code)?;
            codes.push(code);
        }
    }
    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_keep_accumulators_in_range() {
        for n in [1_i64, 128, 256, 384, 512, 1152] {
            let inputs = usize::try_from(n).unwrap_or(0);
            let worst =
                i64::from(hidden_limit(inputs)) * i64::from(LIMIT) * n + i64::from(BIAS_LIMIT);
            assert!(worst <= i64::from(i32::MAX), "{n}");
            let worst = i64::from(out_limit(inputs)) * i64::from(LIMIT) * n;
            assert!(worst <= i64::from(i32::MAX), "{n}");
        }
    }
}
