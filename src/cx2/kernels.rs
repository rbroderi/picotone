//! Dense integer kernels, with runtime CPU dispatch.
//!
//! The model's heavy steps are matrix-vector products of small integers. They
//! are written column by column so the inner loop is a plain
//! "accumulator += column * scalar" that compilers vectorise. [`Simd`] runs
//! that loop under the widest instruction set the CPU has; because the
//! arithmetic is exact integer arithmetic, every path gives the same result.

/// CPU feature dispatcher, detected once.
#[derive(Debug, Clone, Copy)]
pub struct Simd {
    arch: pulp::Arch,
    /// The AVX2 instruction set, when the CPU has it.
    #[cfg(target_arch = "x86_64")]
    v3: Option<pulp::x86::V3>,
    /// Forces the baseline code path; used to test that all paths agree.
    baseline: bool,
}

impl Default for Simd {
    fn default() -> Self {
        Self::new()
    }
}

impl Simd {
    /// Detects the CPU's features.
    #[must_use]
    pub fn new() -> Self {
        Self {
            arch: pulp::Arch::new(),
            #[cfg(target_arch = "x86_64")]
            v3: pulp::x86::V3::try_new(),
            baseline: false,
        }
    }

    /// A dispatcher that always runs the baseline (compile-time) code path.
    #[must_use]
    pub fn baseline() -> Self {
        Self {
            arch: pulp::Arch::new(),
            #[cfg(target_arch = "x86_64")]
            v3: None,
            baseline: true,
        }
    }

    /// Whether this dispatcher always runs the baseline code path.
    #[must_use]
    pub const fn is_baseline(self) -> bool {
        self.baseline
    }

    /// Adds `matrix * weights` to `acc`.
    ///
    /// `matrix` holds `weights.len()` columns of `acc.len()` entries each. The
    /// caller keeps the values small enough that no sum leaves the `i32` range.
    /// Columns whose weight is zero are skipped.
    pub fn accumulate(self, acc: &mut [i32], matrix: &[i16], weights: &[i16]) {
        if self.baseline {
            accumulate(acc, matrix, weights);
        } else {
            self.arch.dispatch(|| accumulate(acc, matrix, weights));
        }
    }

    /// Adds `matrix * input` to `acc`, which has one entry per matrix row.
    ///
    /// The caller keeps the values small enough that no sum of two products, and
    /// no row total, leaves the `i32` range. Input pairs that are both zero are
    /// skipped. Entries of `input` beyond the matrix's columns are ignored and
    /// missing ones count as zero.
    pub fn multiply_add(self, acc: &mut [i32], matrix: &Matrix, input: &[i16]) {
        // Most matrices are narrow; only wide ones pay for the large list.
        if matrix.pairs <= SMALL_PAIRS {
            self.multiply_add_with::<SMALL_PAIRS>(acc, matrix, input);
        } else {
            self.multiply_add_with::<MAX_PAIRS>(acc, matrix, input);
        }
    }

    fn multiply_add_with<const PAIRS: usize>(
        self,
        acc: &mut [i32],
        matrix: &Matrix,
        input: &[i16],
    ) {
        let mut active = [(0_usize, 0_i16, 0_i16); PAIRS];
        let mut used = 0;
        for (pair, values) in input.chunks(2).take(matrix.pairs).enumerate() {
            let left = values.first().copied().unwrap_or(0);
            let right = values.get(1).copied().unwrap_or(0);
            if (left != 0 || right != 0)
                && let Some(slot) = active.get_mut(used)
            {
                *slot = (pair, left, right);
                used += 1;
            }
        }
        let active = &active[..used];
        #[cfg(target_arch = "x86_64")]
        if let Some(v3) = self.v3 {
            v3.vectorize(|| multiply_add_v3(v3, acc, matrix, active));
            return;
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if !self.baseline {
            multiply_add_simd128(acc, matrix, active);
            return;
        }
        multiply_add(acc, matrix, active);
    }
}

/// Rows per block of a [`Matrix`].
const GROUP: usize = 32;
/// Most column pairs a [`Matrix`] may have.
const MAX_PAIRS: usize = 1024;
/// Column pairs up to which [`Simd::multiply_add`] keeps its work list small.
const SMALL_PAIRS: usize = 256;

/// A matrix laid out for [`Simd::multiply_add`].
///
/// Rows are taken in groups of 32 and columns in pairs; a block holds, for one
/// group and one pair, the two entries of each row side by side. One 16-bit
/// multiply-add instruction then handles two columns of eight rows at once,
/// and a group's accumulators stay in registers across all pairs.
#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    pairs: usize,
    /// `pairs` blocks per group of rows.
    blocks: Vec<[i16; 2 * GROUP]>,
}

impl Matrix {
    /// Builds the layout from `columns`, which holds columns of `rows` entries.
    ///
    /// Returns `None` if there are no rows or more than 2048 columns.
    #[must_use]
    pub fn from_columns(columns: &[i16], rows: usize) -> Option<Self> {
        if rows == 0 || !columns.len().is_multiple_of(rows) {
            return None;
        }
        let pairs = (columns.len() / rows).div_ceil(2);
        if pairs > MAX_PAIRS {
            return None;
        }
        let groups = rows.div_ceil(GROUP);
        let mut blocks = vec![[0_i16; 2 * GROUP]; groups * pairs];
        for (column, values) in columns.chunks_exact(rows).enumerate() {
            for (row, &value) in values.iter().enumerate() {
                let block = &mut blocks[(row / GROUP) * pairs + column / 2];
                block[2 * (row % GROUP) + column % 2] = value;
            }
        }
        Some(Self {
            rows,
            pairs,
            blocks,
        })
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }
}

/// Portable form of [`Simd::multiply_add`].
fn multiply_add(acc: &mut [i32], matrix: &Matrix, active: &[(usize, i16, i16)]) {
    for (sums, blocks) in acc
        .chunks_mut(GROUP)
        .zip(matrix.blocks.chunks_exact(matrix.pairs.max(1)))
    {
        for &(pair, left, right) in active {
            let Some(block) = blocks.get(pair) else {
                continue;
            };
            for (sum, entries) in sums.iter_mut().zip(block.as_chunks::<2>().0) {
                *sum = sum.wrapping_add(
                    (i32::from(entries[0]) * i32::from(left))
                        .wrapping_add(i32::from(entries[1]) * i32::from(right)),
                );
            }
        }
    }
}

/// AVX2 form of [`Simd::multiply_add`]: the same sums, eight rows per instruction.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::inline_always,
    reason = "the instruction set only applies to code inlined into `vectorize`"
)]
#[inline(always)]
fn multiply_add_v3(
    v3: pulp::x86::V3,
    acc: &mut [i32],
    matrix: &Matrix,
    active: &[(usize, i16, i16)],
) {
    for (sums, blocks) in acc
        .chunks_mut(GROUP)
        .zip(matrix.blocks.chunks_exact(matrix.pairs.max(1)))
    {
        let mut padded = [0_i32; GROUP];
        padded[..sums.len()].copy_from_slice(sums);
        let mut lanes: [pulp::i32x8; GROUP / 8] = pulp::cast(padded);
        for &(pair, left, right) in active {
            let Some(block) = blocks.get(pair) else {
                continue;
            };
            let mut weights = [0_i16; 16];
            for both in weights.as_chunks_mut::<2>().0 {
                both[0] = left;
                both[1] = right;
            }
            let weights: pulp::i16x16 = pulp::cast(weights);
            let entries: [pulp::i16x16; GROUP / 8] = pulp::cast(*block);
            for (lane, &entry) in lanes.iter_mut().zip(&entries) {
                let products = v3.multiply_wrapping_add_adjacent_i16x16(entry, weights);
                *lane = pulp::cast(
                    v3.avx2
                        ._mm256_add_epi32(pulp::cast(*lane), pulp::cast(products)),
                );
            }
        }
        let padded: [i32; GROUP] = pulp::cast(lanes);
        sums.copy_from_slice(&padded[..sums.len()]);
    }
}

/// WebAssembly SIMD128 form of [`Simd::multiply_add`]: the same sums, four
/// rows per instruction. `i32x4.dot_i16x8_s` adds the two products of a row
/// with wrapping, exactly as the other forms do (it is not relaxed SIMD).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn multiply_add_simd128(acc: &mut [i32], matrix: &Matrix, active: &[(usize, i16, i16)]) {
    use core::arch::wasm32::{
        i16x8, i32x4, i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_splat, v128,
    };
    for (sums, blocks) in acc
        .chunks_mut(GROUP)
        .zip(matrix.blocks.chunks_exact(matrix.pairs.max(1)))
    {
        let mut padded = [0_i32; GROUP];
        padded[..sums.len()].copy_from_slice(sums);
        let mut lanes: [v128; GROUP / 4] = core::array::from_fn(|j| {
            i32x4(
                padded[4 * j],
                padded[4 * j + 1],
                padded[4 * j + 2],
                padded[4 * j + 3],
            )
        });
        for &(pair, left, right) in active {
            let Some(block) = blocks.get(pair) else {
                continue;
            };
            let [l0, l1] = left.to_le_bytes();
            let [r0, r1] = right.to_le_bytes();
            let weights = i32x4_splat(i32::from_le_bytes([l0, l1, r0, r1]));
            for (lane, e) in lanes.iter_mut().zip(block.as_chunks::<8>().0) {
                let entries = i16x8(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
                *lane = i32x4_add(*lane, i32x4_dot_i16x8(entries, weights));
            }
        }
        for (j, lane) in lanes.iter().enumerate() {
            padded[4 * j] = i32x4_extract_lane::<0>(*lane);
            padded[4 * j + 1] = i32x4_extract_lane::<1>(*lane);
            padded[4 * j + 2] = i32x4_extract_lane::<2>(*lane);
            padded[4 * j + 3] = i32x4_extract_lane::<3>(*lane);
        }
        sums.copy_from_slice(&padded[..sums.len()]);
    }
}

#[expect(
    clippy::inline_always,
    reason = "the dispatcher only recompiles code that is inlined into it"
)]
#[inline(always)]
fn accumulate(acc: &mut [i32], matrix: &[i16], weights: &[i16]) {
    let len = acc.len().max(1);
    for (column, &weight) in matrix.chunks_exact(len).zip(weights) {
        if weight != 0 {
            for (sum, &value) in acc.iter_mut().zip(column) {
                *sum += i32::from(value) * i32::from(weight);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_add_agrees_with_a_plain_loop_on_every_path() {
        for (rows, columns) in [
            (1024, 64),
            (37, 5),
            (64, 384),
            (1, 1),
            (384, 512),
            (96, 1152),
        ] {
            let matrix: Vec<i16> = (0..rows * columns)
                .map(|i| i16::try_from((i * 7919) % 4095).unwrap_or(0) - 2047)
                .collect();
            let input: Vec<i16> = (0..columns)
                .map(|i| {
                    if i % 5 < 2 {
                        0
                    } else {
                        i16::try_from((i * 104_729) % 2047).unwrap_or(0) - 1023
                    }
                })
                .collect();
            let mut expected = vec![-7_i32; rows];
            for (j, sum) in expected.iter_mut().enumerate() {
                for (i, &w) in input.iter().enumerate() {
                    *sum += i32::from(matrix[i * rows + j]) * i32::from(w);
                }
            }
            let packed = Matrix::from_columns(&matrix, rows);
            assert!(packed.is_some());
            if let Some(packed) = packed {
                assert_eq!(packed.rows(), rows);
                for simd in [Simd::new(), Simd::baseline()] {
                    let mut acc = vec![-7_i32; rows];
                    simd.multiply_add(&mut acc, &packed, &input);
                    assert_eq!(acc, expected);
                }
            }
        }
    }

    #[test]
    fn dispatch_paths_agree_with_a_plain_loop() {
        for columns in [37, 8, 1] {
            let rows = 1024;
            let matrix: Vec<i16> = (0..rows * columns)
                .map(|i| i16::try_from((i * 7919) % 8191).unwrap_or(0) - 4095)
                .collect();
            let weights: Vec<i16> = (0..columns)
                .map(|i| i16::try_from((i * 104_729) % 8191).unwrap_or(0) - 4095)
                .collect();
            let mut expected = vec![5_i32; rows];
            for (j, sum) in expected.iter_mut().enumerate() {
                for (i, &w) in weights.iter().enumerate() {
                    *sum += i32::from(matrix[i * rows + j]) * i32::from(w);
                }
            }
            for simd in [Simd::new(), Simd::baseline()] {
                let mut acc = vec![5_i32; rows];
                simd.accumulate(&mut acc, &matrix, &weights);
                assert_eq!(acc, expected);
            }
        }
    }
}
