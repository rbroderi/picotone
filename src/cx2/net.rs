//! The trained expert: a small multilayer perceptron evaluated in integer arithmetic.
//!
//! Unlike every other table of the model, these weights come from gradient
//! descent. Inference is still exact integer arithmetic, so it is as
//! deterministic as the rest:
//!
//! ```text
//! x    = centroid sums of the previous 4 frames, clamp(sum >> 3, +-1023)      4 x 128
//! h1   = clamp((W1 x + b1 + e[K] + p[bucket(t mod P)]) >> 7, 0, 1023)        once per frame
//! h2   = clamp((W2 h1 + b2)       >> 10, 0, 1023)                             once per frame
//! g    = clamp((A_k h2 + B_k s + c_k) >> 10, 0, 1023)    s = this frame's partial sum
//! logit_j += d_k[j] + ((O_k g)[j] >> 11)                                      once per code
//! ```
//!
//! Weights are `i16` with magnitude at most 2047 (16383 for `O_k`, which has at
//! most 128 inputs), activations are at most 1023, and no layer has more than
//! 512 inputs, so every accumulator stays inside an `i32` (the three first-layer
//! biases are each at most 2^28: 3 x 2^28 + 512 x 2047 x 1023 < 2^31).
//!
//! The position term exists only in the second net format (table version 5).
//! `t` counts frames since the last reset and `P` is the table's period (150 for
//! the 48 kHz model, one 1 s segment; 0 switches the term off). Resets of the
//! segmented model always fall on segment starts, so `t mod P` is the frame's
//! index within its segment (except in a file's last segment when the one before
//! it is short, where it is still the same on both sides of the coder).

use anyhow::{Context as _, Result, ensure};

use crate::cx2::features::{CARD, DIM};
use crate::cx2::kernels::{Matrix, Simd};

/// Previous frames the trunk looks at.
pub const PAST: usize = 4;
const INPUT_SHIFT: u32 = 3;
const LIMIT: i32 = 1023;
/// Largest weight magnitude of the layers with up to 512 inputs.
const WEIGHT_LIMIT: i16 = 2047;
/// Largest weight magnitude of the output layers, which have at most 128 inputs.
const OUT_LIMIT: i16 = 16383;

/// One codebook's output head.
#[derive(Debug)]
struct Head {
    rank: usize,
    /// `rank` rows; the columns take `h2` trunk outputs, then the `DIM` entries of the
    /// current partial sum.
    hidden: Matrix,
    bias: Vec<i32>,
    /// `CARD` rows, `rank` columns.
    out: Matrix,
    out_bias: Vec<i32>,
}

/// The network's weights.
#[derive(Debug)]
pub struct Net {
    h1: usize,
    h2: usize,
    /// Codebook counts the network was trained for, ascending.
    bandwidths: Vec<usize>,
    /// One `h1`-long bias per entry of `bandwidths`.
    bandwidth_bias: Vec<i32>,
    /// Frames per position cycle; zero when the net has no position input.
    period: usize,
    /// Bucket of each position `0..period`.
    position_bucket: Vec<u8>,
    /// One `h1`-long bias per position bucket.
    position_bias: Vec<i32>,
    w1: Matrix,
    b1: Vec<i32>,
    w2: Matrix,
    b2: Vec<i32>,
    heads: Vec<Head>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("net table size overflow")?;
        let out = self
            .bytes
            .get(self.pos..end)
            .context("net table is truncated")?;
        self.pos = end;
        Ok(out)
    }

    fn size(&mut self) -> Result<usize> {
        let b = self.take(4)?;
        Ok(usize::try_from(u32::from_le_bytes([
            b[0], b[1], b[2], b[3],
        ]))?)
    }

    fn weights(&mut self, n: usize, limit: i16) -> Result<Vec<i16>> {
        let values: Vec<i16> = self
            .take(n.checked_mul(2).context("net table size overflow")?)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect();
        ensure!(
            values.iter().all(|v| (-limit..=limit).contains(v)),
            "net weight is out of range"
        );
        Ok(values)
    }

    fn biases(&mut self, n: usize) -> Result<Vec<i32>> {
        let values: Vec<i32> = self
            .take(n.checked_mul(4).context("net table size overflow")?)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b))
            .collect();
        ensure!(
            values.iter().all(|v| v.unsigned_abs() <= 1 << 28),
            "net bias is out of range"
        );
        Ok(values)
    }
}

/// Longest position cycle a table may declare, in frames.
const MAX_PERIOD: usize = 4096;

impl Net {
    /// Parses a network for `codebooks` codebooks from the start of `bytes`.
    ///
    /// `positioned` selects the second format, which carries the position input
    /// after the bandwidth biases: u32 period, u32 bucket count, one u8 bucket per
    /// position, then one `h1`-long i32 bias per bucket.
    ///
    /// Returns the network and the number of bytes it occupied.
    ///
    /// # Errors
    /// Fails on truncated or implausible input.
    pub fn parse(bytes: &[u8], codebooks: usize, positioned: bool) -> Result<(Self, usize)> {
        let mut r = Reader { bytes, pos: 0 };
        let h1 = r.size()?;
        let h2 = r.size()?;
        let count = r.size()?;
        // With weights up to 4095 and activations up to 1023, 512 inputs fit an i32.
        ensure!(
            (1..=512).contains(&h1) && (1..=256).contains(&h2) && (1..=8).contains(&count),
            "implausible net dimensions"
        );
        let bandwidths = (0..count).map(|_| r.size()).collect::<Result<Vec<_>>>()?;
        let bandwidth_bias = r.biases(count * h1)?;
        let (period, position_bucket, position_bias) = if positioned {
            let period = r.size()?;
            let buckets = r.size()?;
            ensure!(
                period <= MAX_PERIOD && buckets <= 256 && (period == 0) == (buckets == 0),
                "implausible net position input"
            );
            let map = r.take(period)?.to_vec();
            ensure!(
                map.iter().all(|&b| usize::from(b) < buckets),
                "net position bucket is out of range"
            );
            (period, map, r.biases(buckets * h1)?)
        } else {
            (0, Vec::new(), Vec::new())
        };
        let w1 = matrix(&r.weights(PAST * DIM * h1, WEIGHT_LIMIT)?, h1)?;
        let b1 = r.biases(h1)?;
        let w2 = matrix(&r.weights(h1 * h2, WEIGHT_LIMIT)?, h2)?;
        let b2 = r.biases(h2)?;
        let mut heads = Vec::with_capacity(codebooks);
        for _ in 0..codebooks {
            let rank = r.size()?;
            ensure!((1..=128).contains(&rank), "implausible net head rank");
            // On file: the trunk's columns, then the partial sum's.
            let hidden = matrix(&r.weights((h2 + DIM) * rank, WEIGHT_LIMIT)?, rank)?;
            heads.push(Head {
                rank,
                hidden,
                bias: r.biases(rank)?,
                out: matrix(&r.weights(rank * CARD, OUT_LIMIT)?, CARD)?,
                out_bias: r.biases(CARD)?,
            });
        }
        Ok((
            Self {
                h1,
                h2,
                bandwidths,
                bandwidth_bias,
                period,
                position_bucket,
                position_bias,
                w1,
                b1,
                w2,
                b2,
                heads,
            },
            r.pos,
        ))
    }

    /// Starts the per-stream state for `num_codebooks` codes per frame.
    #[must_use]
    pub fn state(&self, num_codebooks: usize) -> NetState {
        NetState {
            // A stream with an untrained codebook count uses the nearest trained one
            // above it (its inputs then simply lack the last codebooks' centroids).
            bandwidth: self
                .bandwidths
                .iter()
                .position(|&k| k >= num_codebooks)
                .or_else(|| self.bandwidths.len().checked_sub(1)),
            period: self.period,
            position: 0,
            past: vec![0; PAST * DIM],
            input: vec![0; self.h2 + DIM],
            ready: false,
            hidden: vec![0; self.h1.max(128)],
            acc: vec![0; self.h1.max(self.h2).max(128)],
            dots: vec![0; CARD],
        }
    }
}

/// Lays out columns of `rows` entries for the multiply-add kernel.
fn matrix(columns: &[i16], rows: usize) -> Result<Matrix> {
    Matrix::from_columns(columns, rows).context("implausible net layer shape")
}

/// Clamps `acc >> shift` into the activation range.
fn activate(out: &mut [i16], acc: &[i32], shift: u32) {
    for (o, &a) in out.iter_mut().zip(acc) {
        *o = i16::try_from((a >> shift).clamp(0, LIMIT)).unwrap_or(0);
    }
}

/// Fixed-point input from a centroid sum.
fn squash(out: &mut [i16], sums: &[i32]) {
    for (o, &s) in out.iter_mut().zip(sums) {
        *o = i16::try_from((s >> INPUT_SHIFT).clamp(-LIMIT, LIMIT)).unwrap_or(0);
    }
}

/// What the network remembers about a stream.
#[derive(Debug)]
pub struct NetState {
    /// Index into the network's bandwidth list.
    bandwidth: Option<usize>,
    /// The network's position period (zero: no position input).
    period: usize,
    /// Frames since the last reset, modulo `period`.
    position: usize,
    /// Inputs from the previous frames, most recent first.
    past: Vec<i16>,
    /// The trunk's output, then the current partial sum: the heads' input.
    input: Vec<i16>,
    ready: bool,
    hidden: Vec<i16>,
    acc: Vec<i32>,
    dots: Vec<i32>,
}

impl NetState {
    /// Forgets the previous frames.
    pub fn reset(&mut self) {
        self.past.fill(0);
        self.position = 0;
        self.ready = false;
    }

    /// Records a finished frame's full centroid sum.
    pub fn push_frame(&mut self, sums: &[i32]) {
        self.past.copy_within(..(PAST - 1) * DIM, DIM);
        squash(&mut self.past[..DIM], sums);
        if self.period > 0 {
            self.position = (self.position + 1) % self.period;
        }
        self.ready = false;
    }

    /// Adds the network's logits for codebook `k` to `out`.
    ///
    /// `partial` is the centroid sum of the codebooks already coded in this frame.
    pub fn add(&mut self, net: &Net, simd: Simd, k: usize, partial: &[i32], out: &mut [i32]) {
        let (Some(bandwidth), Some(head)) = (self.bandwidth, net.heads.get(k)) else {
            return;
        };
        if !self.ready {
            let acc = &mut self.acc[..net.h1];
            let extra = &net.bandwidth_bias[bandwidth * net.h1..(bandwidth + 1) * net.h1];
            for ((a, &b), &e) in acc.iter_mut().zip(&net.b1).zip(extra) {
                *a = b + e;
            }
            if let Some(&bucket) = net.position_bucket.get(self.position) {
                let start = usize::from(bucket) * net.h1;
                for (a, &p) in acc
                    .iter_mut()
                    .zip(&net.position_bias[start..start + net.h1])
                {
                    *a += p;
                }
            }
            simd.multiply_add(acc, &net.w1, &self.past);
            activate(&mut self.hidden[..net.h1], acc, 7);
            let acc = &mut self.acc[..net.h2];
            acc.copy_from_slice(&net.b2);
            simd.multiply_add(acc, &net.w2, &self.hidden[..net.h1]);
            activate(&mut self.input[..net.h2], acc, 10);
            self.ready = true;
        }
        squash(&mut self.input[net.h2..], partial);
        let acc = &mut self.acc[..head.rank];
        acc.copy_from_slice(&head.bias);
        simd.multiply_add(acc, &head.hidden, &self.input);
        let hidden = &mut self.hidden[..head.rank];
        activate(hidden, acc, 10);
        self.dots.fill(0);
        simd.multiply_add(&mut self.dots, &head.out, hidden);
        for ((value, &dot), &bias) in out.iter_mut().zip(&self.dots).zip(&head.out_bias) {
            *value += bias + (dot >> 11);
        }
    }
}
