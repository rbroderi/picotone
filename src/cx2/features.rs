//! Integer features of an EnCodec code stream.
//!
//! [`State`] walks a stream code by code. Before each code it can describe
//! every one of the 1024 candidates with small integers ("buckets"); the
//! model turns those into log-probabilities by table lookup. Nothing here
//! uses floating point after the centroids are rounded to fixed point.
//!
//! Features of candidate `j` for codebook `k` of frame `t`:
//!
//! * **geo**: squared distance between centroid `j` and the residual predicted
//!   from the previous frame: the centroid sum of codebooks `0..=k` of frame
//!   `t-1` minus the centroid sum of codebooks `0..k` of frame `t`.
//! * **rec**: how many frames ago `j` last appeared in codebook `k`, and how
//!   often it has appeared so far.
//! * **pair** (three contexts): how often `j` has followed the current context
//!   code so far in this stream.

use anyhow::{Context as _, Result, ensure};

use crate::cx2::kernels::Simd;

/// Codebook size.
pub const CARD: usize = 1024;
/// Latent dimension of the RVQ centroids.
pub const DIM: usize = 128;
/// Number of code contexts.
pub const CONTEXTS: usize = 3;
/// Fixed-point shift of the centroids (values are multiples of 1/128).
const CENTROID_SHIFT: u32 = 7;
/// Clamp of the predicted residual, in fixed-point units (about 32.0).
const RESIDUAL_CLAMP: i32 = 4095;
/// Buckets of the squared-distance feature: eight per octave.
pub const GEO_BUCKETS: usize = 8 * 34;
const REC_COUNT_BUCKETS: usize = 16;
const REC_COUNT_STEP: u16 = 16;
/// Recency classes: never seen, then 28 logarithmic steps.
const REC_CLASSES: usize = 29;
/// Buckets of the recency feature: 29 recency classes by 16 count classes.
pub const REC_BUCKETS: usize = REC_CLASSES * REC_COUNT_BUCKETS;
const PAIR_CONTEXT_BUCKETS: usize = 12;
/// Buckets of a pair feature: 8 hit-count classes by 12 context-count classes.
pub const PAIR_BUCKETS: usize = 8 * PAIR_CONTEXT_BUCKETS;

/// `floor(2 * log2(value))` for `value >= 1`, in integer arithmetic.
#[must_use]
pub fn half_log2(value: u32) -> usize {
    let value = u64::from(value.max(1));
    let exponent = 63 - value.leading_zeros();
    let half = u32::from(value * value >= 1_u64 << (2 * exponent + 1));
    usize::try_from(2 * exponent + half).unwrap_or(0)
}

/// Bucket of a squared distance: eight logarithmic steps per octave.
fn geo_bucket(distance: u64) -> u16 {
    let value = distance + 1;
    let exponent = 63 - value.leading_zeros();
    let mantissa = if exponent >= 3 {
        (value >> (exponent - 3)) & 7
    } else {
        (value << (3 - exponent)) & 7
    };
    let bucket = u64::from(exponent) * 8 + mantissa;
    u16::try_from(bucket.min(GEO_BUCKETS as u64 - 1)).unwrap_or(0)
}

/// `min(half_log2(hits + 1), 7)` for `hits` up to 11; larger counts are class 7.
const HIT_CLASS: [u8; 12] = [0, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7];
/// Recency values at or beyond this many frames share the last class.
const RECENCY_TABLE: usize = 1 << 14;
/// Counts at or beyond this share the last class.
const COUNT_TABLE: usize = 256;

/// Bucket of a pair feature from a hit count and the context's total count.
#[must_use]
pub fn pair_bucket(hits: u32, context_total: u32) -> usize {
    let index = usize::try_from(hits)
        .unwrap_or(usize::MAX)
        .min(HIT_CLASS.len() - 1);
    usize::from(HIT_CLASS[index]) * PAIR_CONTEXT_BUCKETS + context_bucket(context_total)
}

fn context_bucket(context_total: u32) -> usize {
    half_log2(context_total.saturating_add(1)).min(PAIR_CONTEXT_BUCKETS - 1)
}

/// EnCodec's RVQ centroids in fixed point, with their squared norms.
#[derive(Debug)]
pub struct Centroids {
    codebooks: usize,
    /// `[codebook][code][dimension]`.
    data: Vec<i16>,
    /// The same values as `[codebook][dimension][code]`, for the distance kernel.
    columns: Vec<i16>,
    norms: Vec<i64>,
    simd: Simd,
}

/// Rounds `value * 2^CENTROID_SHIFT` to the nearest integer, saturating at the residual clamp.
///
/// Scaling by a power of two is exact and rounding half away from zero is
/// fully specified, so every IEEE-754 platform gets the same integer.
fn to_fixed(value: f32) -> i16 {
    let scaled = (f64::from(value) * f64::from(1_u32 << CENTROID_SHIFT)).round();
    let clamped = scaled.clamp(-f64::from(RESIDUAL_CLAMP), f64::from(RESIDUAL_CLAMP));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is clamped well inside the i16 range and is integral"
    )]
    let fixed = clamped as i16;
    fixed
}

impl Centroids {
    /// Reads the centroids of the first `codebooks` RVQ layers from EnCodec `safetensors` bytes.
    ///
    /// # Errors
    /// Fails when the file is malformed or lacks a codebook.
    pub fn from_safetensors(weights: &[u8], codebooks: usize) -> Result<Self> {
        ensure!(weights.len() >= 8, "weights file is too short");
        let (len_bytes, rest) = weights.split_at(8);
        let mut len = [0_u8; 8];
        len.copy_from_slice(len_bytes);
        let header_len = usize::try_from(u64::from_le_bytes(len))?;
        ensure!(rest.len() >= header_len, "weights header is truncated");
        let (header, tensors) = rest.split_at(header_len);
        let header: serde_json::Value =
            serde_json::from_slice(header).context("parsing the safetensors header")?;
        let mut data = Vec::with_capacity(codebooks * CARD * DIM);
        for k in 0..codebooks {
            let name = format!("quantizer.layers.{k}.codebook.embed");
            let info = header
                .get(&name)
                .with_context(|| format!("weights have no tensor {name}"))?;
            ensure!(
                info.get("dtype").and_then(serde_json::Value::as_str) == Some("F32"),
                "{name} is not f32"
            );
            let offsets = info
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .with_context(|| format!("{name} has no data offsets"))?;
            let bound = |i: usize| -> Result<usize> {
                let v = offsets
                    .get(i)
                    .and_then(serde_json::Value::as_u64)
                    .with_context(|| format!("{name} has malformed data offsets"))?;
                Ok(usize::try_from(v)?)
            };
            let (start, end) = (bound(0)?, bound(1)?);
            ensure!(
                start <= end && end <= tensors.len() && end - start == CARD * DIM * 4,
                "{name} has an unexpected size"
            );
            data.extend(
                tensors[start..end]
                    .chunks_exact(4)
                    .map(|b| to_fixed(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
            );
        }
        Ok(Self::from_values(data, codebooks))
    }

    /// Takes centroids that are already in fixed point, `[codebook][code][dimension]`,
    /// as a model table stores them.
    ///
    /// # Errors
    /// Fails when the count does not match or a value is outside the residual clamp.
    pub fn from_fixed(values: Vec<i16>, codebooks: usize) -> Result<Self> {
        ensure!(
            values.len() == codebooks * CARD * DIM,
            "centroid table has the wrong size"
        );
        ensure!(
            values
                .iter()
                .all(|v| (-RESIDUAL_CLAMP..=RESIDUAL_CLAMP).contains(&i32::from(*v))),
            "centroid value is out of range"
        );
        Ok(Self::from_values(values, codebooks))
    }

    fn from_values(data: Vec<i16>, codebooks: usize) -> Self {
        let norms = data
            .chunks_exact(DIM)
            .map(|row| row.iter().map(|&v| i64::from(v) * i64::from(v)).sum())
            .collect();
        let mut columns = vec![0; data.len()];
        for (k, book) in data.chunks_exact(CARD * DIM).enumerate() {
            for (code, row) in book.chunks_exact(DIM).enumerate() {
                for (d, &value) in row.iter().enumerate() {
                    columns[(k * DIM + d) * CARD + code] = value;
                }
            }
        }
        Self {
            codebooks,
            data,
            columns,
            norms,
            simd: Simd::new(),
        }
    }

    /// Number of codebooks loaded.
    #[must_use]
    pub const fn codebooks(&self) -> usize {
        self.codebooks
    }

    /// Fixed-point values, for hashing.
    #[must_use]
    pub fn values(&self) -> &[i16] {
        &self.data
    }

    /// Chooses the CPU dispatch used by the distance kernel.
    pub const fn set_simd(&mut self, simd: Simd) {
        self.simd = simd;
    }

    fn row(&self, codebook: usize, code: usize) -> &[i16] {
        let start = (codebook * CARD + code) * DIM;
        &self.data[start..start + DIM]
    }
}

/// Position in a code stream plus everything remembered about its past.
#[derive(Debug)]
pub struct State<'a> {
    centroids: &'a Centroids,
    num_codebooks: usize,
    frame: u32,
    /// Codebook of the next code within its frame.
    next: usize,
    /// `prev_sums[k]`: centroid sum of codebooks `0..=k` of the previous frame.
    prev_sums: Vec<i32>,
    /// Centroid sum of the codebooks already coded in the current frame.
    cur_sum: Vec<i32>,
    prev_codes: Vec<usize>,
    /// Codebook 0 two frames back.
    older_code: usize,
    cur_codes: Vec<usize>,
    /// Frame number plus one of each code's last appearance; zero when never seen.
    last_seen: Vec<u32>,
    counts: Vec<u32>,
    /// Successor counts per (context slot, codebook, context code).
    pairs: Vec<Vec<(u16, u32)>>,
    context_totals: Vec<u32>,
    residual: Vec<i16>,
    dots: Vec<i32>,
    /// Recency class by frames since the last appearance.
    recency_class: Vec<u8>,
    /// Count class by number of appearances.
    count_class: Vec<u8>,
}

impl<'a> State<'a> {
    /// Starts a stream with `num_codebooks` codes per frame.
    ///
    /// # Errors
    /// Fails when the centroids do not cover that many codebooks.
    pub fn new(centroids: &'a Centroids, num_codebooks: usize) -> Result<Self> {
        ensure!(
            (1..=centroids.codebooks).contains(&num_codebooks),
            "model covers {} codebooks, stream has {num_codebooks}",
            centroids.codebooks
        );
        Ok(Self {
            centroids,
            num_codebooks,
            frame: 0,
            next: 0,
            prev_sums: vec![0; num_codebooks * DIM],
            cur_sum: vec![0; DIM],
            prev_codes: vec![0; num_codebooks],
            older_code: 0,
            cur_codes: vec![0; num_codebooks],
            last_seen: vec![0; num_codebooks * CARD],
            counts: vec![0; num_codebooks * CARD],
            pairs: vec![Vec::new(); CONTEXTS * num_codebooks * CARD],
            context_totals: vec![0; CONTEXTS * num_codebooks * CARD],
            residual: vec![0; DIM],
            dots: vec![0; CARD],
            recency_class: (0..RECENCY_TABLE)
                .map(|frames| {
                    let class = 1 + half_log2(u32::try_from(frames).unwrap_or(u32::MAX)).min(27);
                    u8::try_from(class).unwrap_or(u8::MAX)
                })
                .collect(),
            count_class: (0..COUNT_TABLE)
                .map(|count| {
                    let class = half_log2(u32::try_from(count + 1).unwrap_or(u32::MAX))
                        .min(REC_COUNT_BUCKETS - 1);
                    u8::try_from(class).unwrap_or(u8::MAX)
                })
                .collect(),
        })
    }

    /// Forgets everything, as at the start of a stream. Used at reset points.
    pub fn reset(&mut self) {
        self.frame = 0;
        self.next = 0;
        self.prev_sums.fill(0);
        self.cur_sum.fill(0);
        self.prev_codes.fill(0);
        self.older_code = 0;
        self.cur_codes.fill(0);
        self.last_seen.fill(0);
        self.counts.fill(0);
        for list in &mut self.pairs {
            list.clear();
        }
        self.context_totals.fill(0);
    }

    /// Centroid sum of the codebooks already coded in the current frame.
    #[must_use]
    pub fn partial_sum(&self) -> &[i32] {
        &self.cur_sum
    }

    /// Centroid sum of all codebooks of the previous frame.
    #[must_use]
    pub fn previous_sum(&self) -> &[i32] {
        &self.prev_sums[(self.num_codebooks - 1) * DIM..]
    }

    /// Frames completed since the start or the last reset.
    #[must_use]
    pub const fn frame(&self) -> u32 {
        self.frame
    }

    /// Codebook of the next code.
    #[must_use]
    pub const fn codebook(&self) -> usize {
        self.next
    }

    /// Codes per frame.
    #[must_use]
    pub const fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// The three context codes of the next code.
    ///
    /// | codebook | slot 0        | slot 1          | slot 2        |
    /// |----------|---------------|-----------------|---------------|
    /// | 0        | `c[t-1][0]`   | `c[t-1][1]`     | `c[t-2][0]`   |
    /// | 1        | `c[t-1][1]`   | `c[t][0]`       | `c[t-1][0]`   |
    /// | k >= 2   | `c[t-1][k]`   | `c[t][k-1]`     | `c[t][0]`     |
    #[must_use]
    pub fn contexts(&self) -> [usize; CONTEXTS] {
        let k = self.next;
        let same = self.prev_codes[k];
        match k {
            0 => [
                same,
                self.prev_codes.get(1).copied().unwrap_or(0),
                self.older_code,
            ],
            1 => [same, self.cur_codes[0], self.prev_codes[0]],
            _ => [same, self.cur_codes[k - 1], self.cur_codes[0]],
        }
    }

    const fn pair_index(&self, slot: usize, context: usize) -> usize {
        (slot * self.num_codebooks + self.next) * CARD + context
    }

    /// Writes the geometric bucket of every candidate into `out`.
    pub fn geo_buckets(&mut self, out: &mut [u16]) {
        let k = self.next;
        let prev = &self.prev_sums[k * DIM..(k + 1) * DIM];
        let mut own = 0_i64;
        for ((r, &p), &c) in self.residual.iter_mut().zip(prev).zip(&self.cur_sum) {
            let value = (p - c).clamp(-RESIDUAL_CLAMP, RESIDUAL_CLAMP);
            *r = i16::try_from(value).unwrap_or(0);
            own += i64::from(value) * i64::from(value);
        }
        // |residual| and |centroid| are at most 4095, so 128 products fit an i32.
        let columns = &self.centroids.columns[k * CARD * DIM..(k + 1) * CARD * DIM];
        self.dots.fill(0);
        self.centroids
            .simd
            .accumulate(&mut self.dots, columns, &self.residual);
        let norms = &self.centroids.norms[k * CARD..(k + 1) * CARD];
        for ((bucket, &dot), &norm) in out.iter_mut().zip(&self.dots).zip(norms) {
            let distance = own - 2 * i64::from(dot) + norm;
            *bucket = geo_bucket(u64::try_from(distance).unwrap_or(0));
        }
    }

    /// Writes the recency bucket of every candidate into `out`.
    pub fn rec_buckets(&self, out: &mut [u16]) {
        let k = self.next;
        for (code, bucket) in out.iter_mut().enumerate() {
            *bucket = self.rec_bucket(k, code);
        }
    }

    /// Recency bucket of one candidate of codebook `k`, as seen from the current frame.
    #[must_use]
    pub fn rec_bucket(&self, k: usize, code: usize) -> u16 {
        let seen = self.last_seen[k * CARD + code];
        let recency = if seen == 0 {
            0
        } else {
            let frames = usize::try_from(self.frame + 1 - seen).unwrap_or(usize::MAX);
            self.recency_class[frames.min(RECENCY_TABLE - 1)]
        };
        let count = usize::try_from(self.counts[k * CARD + code]).unwrap_or(usize::MAX);
        let count = self.count_class[count.min(COUNT_TABLE - 1)];
        u16::from(recency) * REC_COUNT_STEP + u16::from(count)
    }

    /// Frame distances at which a code's recency class changes.
    ///
    /// A code last seen `d` frames ago keeps its bucket until `d` reaches the next
    /// entry, which lets a caller refresh only the few codes that crossed one.
    #[must_use]
    pub fn recency_steps(&self) -> Vec<u32> {
        (1..RECENCY_TABLE)
            .filter(|&d| self.recency_class[d] != self.recency_class[d - 1])
            .filter_map(|d| u32::try_from(d).ok())
            .collect()
    }

    /// Successors seen so far after context `slot`: the context's total and `(code, hits)` pairs.
    #[must_use]
    pub fn pair_hits(&self, slot: usize) -> (u32, &[(u16, u32)]) {
        let index = self.pair_index(slot, self.contexts()[slot]);
        (self.context_totals[index], &self.pairs[index])
    }

    /// Records the code at the current position and advances.
    ///
    /// # Errors
    /// Fails when `code` is not a valid codebook entry.
    pub fn update(&mut self, code: u16) -> Result<()> {
        let index = usize::from(code);
        ensure!(index < CARD, "code {code} is out of range");
        let k = self.next;
        for (slot, context) in self.contexts().into_iter().enumerate() {
            let at = self.pair_index(slot, context);
            self.context_totals[at] = self.context_totals[at].saturating_add(1);
            let list = &mut self.pairs[at];
            if let Some(position) = list.iter().position(|entry| entry.0 == code) {
                list[position].1 = list[position].1.saturating_add(1);
                // Drift frequent successors towards the front so later searches are short.
                // The order never affects the model's output.
                if position > 0 {
                    list.swap(position, position - 1);
                }
            } else {
                list.push((code, 1));
            }
        }
        for (sum, &c) in self.cur_sum.iter_mut().zip(self.centroids.row(k, index)) {
            *sum += i32::from(c);
        }
        // The previous frame's sum for this codebook has been used; the next frame
        // needs the current one.
        self.prev_sums[k * DIM..(k + 1) * DIM].copy_from_slice(&self.cur_sum);
        self.cur_codes[k] = index;
        self.last_seen[k * CARD + index] = self.frame + 1;
        self.counts[k * CARD + index] = self.counts[k * CARD + index].saturating_add(1);
        self.next += 1;
        if self.next == self.num_codebooks {
            self.next = 0;
            self.frame = self.frame.saturating_add(1);
            self.older_code = self.prev_codes[0];
            self.prev_codes.copy_from_slice(&self.cur_codes);
            self.cur_sum.fill(0);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_log2_matches_its_definition() {
        let expected = [0, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8];
        for (value, &want) in (1_u32..).zip(&expected) {
            assert_eq!(half_log2(value), want, "value {value}");
        }
        assert_eq!(half_log2(u32::MAX), 63);
    }

    #[test]
    fn geo_buckets_are_monotone() {
        let mut last = 0;
        for distance in [0_u64, 1, 2, 3, 7, 8, 100, 10_000, 1 << 30, 1 << 40] {
            let bucket = geo_bucket(distance);
            assert!(bucket >= last);
            last = bucket;
        }
        assert_eq!(usize::from(geo_bucket(u64::MAX - 1)), GEO_BUCKETS - 1);
    }
}
