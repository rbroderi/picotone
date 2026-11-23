//! Winograd F(m, 7) convolutions, for the HQ codec's k7 layers.
//!
//! A k7 conv computes each output from 7 inputs. Winograd's minimal filtering
//! computes `m` neighbouring outputs from `m + 6` transformed inputs instead
//! of `7 m` products: the input tile is transformed (`B^T d`), multiplied
//! point by point with the transformed filter (`G g`, precomputed), and
//! transformed back (`A^T`). Over channels each of the `m + 6` points is one
//! matrix product `[tiles, cin] x [cin, cout]`, run on the packed kernel, so
//! the arithmetic drops by `7 m / (m + 6)`: 2.8x for m = 4.
//!
//! A dilated conv is an undilated one on each of its `dilation` phases (every
//! `dilation`-th frame), so a tile's outputs and inputs are `dilation` frames
//! apart.
//!
//! The transforms are built in double precision from the interpolation points
//! below, with the scale of each point folded into the filter transform. The
//! result differs from a direct conv by float rounding only, and the tiles
//! start at each call's first output, so a chunked decode matches a whole-file
//! one to rounding rather than bit for bit.

#![expect(
    clippy::inline_always,
    reason = "the transforms must be inlined into the function compiled for the detected instruction set"
)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use anyhow::{Result, ensure};
use pulp::{Simd, WithSimd};
use rayon::prelude::*;

use crate::kernels::{self, PackedMat};
use crate::nn::Seq;

/// Filter taps.
const TAPS: usize = 7;
/// Outputs per tile.
pub(crate) const TILE_OUTPUTS: usize = 4;
/// Finite interpolation points, in the order they are used; infinity is the
/// last point of every transform.
const POINTS: [f64; 9] = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
/// Values (rows times channels, in and out) a group of tiles aims to keep, so
/// its transformed tiles stay in cache through the products.
const GROUP_VALUES: usize = 1 << 16;

/// Points per tile: inputs, and products per channel pair.
const ALPHA: usize = TILE_OUTPUTS + TAPS - 1;
/// Finite points used.
const FINITE: usize = ALPHA - 1;

/// `x^e`.
const fn pow(x: f64, e: usize) -> f64 {
    let mut p = 1.0;
    let mut i = 0;
    while i < e {
        p *= x;
        i += 1;
    }
    p
}

/// `prod_{l != k} (a_k - a_l)`: the scale of point `k`, folded into `G`.
const fn scale(k: usize) -> f64 {
    let mut f = 1.0;
    let mut l = 0;
    while l < FINITE {
        if l != k {
            f *= POINTS[k] - POINTS[l];
        }
        l += 1;
    }
    f
}

/// `G [alpha][7]`, the filter transform.
const G: [[f64; TAPS]; ALPHA] = {
    let mut g = [[0.0; TAPS]; ALPHA];
    let mut k = 0;
    while k < FINITE {
        let mut j = 0;
        while j < TAPS {
            g[k][j] = pow(POINTS[k], j) / scale(k);
            j += 1;
        }
        k += 1;
    }
    g[FINITE][TAPS - 1] = 1.0;
    g
};

/// `A^T [m][alpha]`, the output transform.
const AT: [[f64; ALPHA]; TILE_OUTPUTS] = {
    let mut at = [[0.0; ALPHA]; TILE_OUTPUTS];
    let mut i = 0;
    while i < TILE_OUTPUTS {
        let mut k = 0;
        while k < FINITE {
            at[i][k] = pow(POINTS[k], i);
            k += 1;
        }
        i += 1;
    }
    at[TILE_OUTPUTS - 1][FINITE] = 1.0;
    at
};

/// `B^T [alpha][alpha]`, the input transform: the inverse of `V`, where
/// `V[s][k] = a_k^s / f_k` sums the products `A^T[i][k] G[k][j]` for `i + j = s`.
const BT: [[f64; ALPHA]; ALPHA] = {
    let mut v = [[0.0; ALPHA]; ALPHA];
    let mut s = 0;
    while s < ALPHA {
        let mut k = 0;
        while k < FINITE {
            v[s][k] = pow(POINTS[k], s) / scale(k);
            k += 1;
        }
        s += 1;
    }
    v[ALPHA - 1][FINITE] = 1.0;
    invert(v)
};

/// Inverse of a square matrix, by Gauss-Jordan elimination with partial pivoting.
const fn invert(mut a: [[f64; ALPHA]; ALPHA]) -> [[f64; ALPHA]; ALPHA] {
    let mut inv = [[0.0; ALPHA]; ALPHA];
    let mut i = 0;
    while i < ALPHA {
        inv[i][i] = 1.0;
        i += 1;
    }
    let mut col = 0;
    while col < ALPHA {
        let mut pivot = col;
        let mut r = col + 1;
        while r < ALPHA {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
            r += 1;
        }
        assert!(a[pivot][col].abs() > 1e-12, "singular Winograd transform");
        let (row, inv_row) = (a[pivot], inv[pivot]);
        a[pivot] = a[col];
        inv[pivot] = inv[col];
        a[col] = row;
        inv[col] = inv_row;
        let p = a[col][col];
        let mut j = 0;
        while j < ALPHA {
            a[col][j] /= p;
            inv[col][j] /= p;
            j += 1;
        }
        let mut r = 0;
        while r < ALPHA {
            let factor = a[r][col];
            if r != col && factor != 0.0 {
                let mut j = 0;
                while j < ALPHA {
                    a[r][j] -= factor * a[col][j];
                    inv[r][j] -= factor * inv[col][j];
                    j += 1;
                }
            }
            r += 1;
        }
        col += 1;
    }
    inv
}

/// A transform in single precision. Its exact entries are multiples of 1/64
/// (the points are integers and halves); those within rounding of one (from
/// the inversion) are snapped to it, so zeros are exact and drop out.
const fn single<const R: usize, const C: usize>(m: [[f64; C]; R]) -> [[f32; C]; R] {
    let mut out = [[0.0; C]; R];
    let mut i = 0;
    while i < R {
        let mut j = 0;
        while j < C {
            let v = m[i][j];
            // `f64::round` is not const at the supported Rust version.
            #[expect(clippy::cast_possible_truncation, reason = "small transform entries")]
            #[expect(clippy::cast_precision_loss, reason = "small transform entries")]
            let r = (if v < 0.0 {
                v * 64.0 - 0.5
            } else {
                v * 64.0 + 0.5
            }) as i64 as f64
                / 64.0;
            #[expect(clippy::cast_possible_truncation, reason = "transform coefficients")]
            let s = if (v - r).abs() < 1e-9 {
                r as f32
            } else {
                v as f32
            };
            out[i][j] = s;
            j += 1;
        }
        i += 1;
    }
    out
}

const BT32: [[f32; ALPHA]; ALPHA] = single(BT);
const AT32: [[f32; ALPHA]; TILE_OUTPUTS] = single(AT);

/// Channel counts must be a multiple of this, so rows split into whole
/// registers for every instruction set used (8 lanes for AVX2, 4 for simd128).
const LANES: usize = 8;

/// A buffer as whole registers.
#[inline(always)]
fn registers<S: Simd>(values: &[f32]) -> Result<&[S::f32s]> {
    let (whole, tail) = S::as_simd_f32s(values);
    ensure!(tail.is_empty(), "rows do not fill whole registers");
    Ok(whole)
}

/// A buffer as whole registers, writable.
#[inline(always)]
fn registers_mut<S: Simd>(values: &mut [f32]) -> Result<&mut [S::f32s]> {
    let (whole, tail) = S::as_mut_simd_f32s(values);
    ensure!(tail.is_empty(), "rows do not fill whole registers");
    Ok(whole)
}

/// `sum_k coefficients[k] * x[k]`; the coefficients are constants, so the
/// zeros drop out once the loops unroll.
#[inline(always)]
fn combine<S: Simd, const N: usize>(
    simd: S,
    start: S::f32s,
    coefficients: &[f32; N],
    x: &[S::f32s; N],
) -> S::f32s {
    let mut sum = start;
    for k in 0..N {
        if coefficients[k] != 0.0 {
            sum = simd.mul_add_e_f32s(simd.splat_f32s(coefficients[k]), x[k], sum);
        }
    }
    sum
}

/// Where a group's tiles sit: tile `q` starts at frame
/// `start + (q / dilation) * block + q % dilation` of the input and of the output.
#[derive(Debug, Clone, Copy)]
struct Tiles {
    start: usize,
    block: usize,
    dilation: usize,
    count: usize,
}

impl Tiles {
    #[inline(always)]
    const fn origin(&self, q: usize) -> usize {
        self.start + (q / self.dilation) * self.block + q % self.dilation
    }
}

/// The input transform of a group's tiles, into `[point][tile][cin]`.
struct InputPass<'a> {
    tiles: Tiles,
    input: &'a [f32],
    cin: usize,
    v: &'a mut [f32],
}

impl WithSimd for InputPass<'_> {
    type Output = Result<()>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Result<()> {
        let Self {
            tiles,
            input,
            cin,
            v,
        } = self;
        let width = registers::<S>(&input[..cin])?.len();
        let input = registers::<S>(input)?;
        let v = registers_mut::<S>(v)?;
        let frames = input.len() / width;
        let zero = simd.splat_f32s(0.0);
        for q in 0..tiles.count {
            let t0 = tiles.origin(q);
            // Inputs past the end are zeros (the last tiles of a call).
            let inside = (0..ALPHA)
                .take_while(|l| t0 + l * tiles.dilation < frames)
                .count();
            for c in 0..width {
                let mut d = [zero; ALPHA];
                for (l, x) in d[..inside].iter_mut().enumerate() {
                    *x = input[(t0 + l * tiles.dilation) * width + c];
                }
                for (p, coefficients) in BT32.iter().enumerate() {
                    v[(p * tiles.count + q) * width + c] = combine(simd, zero, coefficients, &d);
                }
            }
        }
        Ok(())
    }
}

/// The output transform of a group's tiles, from `[point][tile][cout]` into
/// the group's output frames.
struct OutputPass<'a> {
    tiles: Tiles,
    prod: &'a [f32],
    bias: &'a [f32],
    dst: &'a mut [f32],
}

impl WithSimd for OutputPass<'_> {
    type Output = Result<()>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Result<()> {
        let Self {
            tiles,
            prod,
            bias,
            dst,
        } = self;
        let bias = registers::<S>(bias)?;
        let width = bias.len();
        let prod = registers::<S>(prod)?;
        let dst = registers_mut::<S>(dst)?;
        let rows = dst.len() / width;
        for q in 0..tiles.count {
            let local0 = tiles.origin(q) - tiles.start;
            let inside = (0..TILE_OUTPUTS)
                .take_while(|i| local0 + i * tiles.dilation < rows)
                .count();
            for (c, &b) in bias.iter().enumerate() {
                let points: [S::f32s; ALPHA] =
                    std::array::from_fn(|p| prod[(p * tiles.count + q) * width + c]);
                for (i, coefficients) in AT32[..inside].iter().enumerate() {
                    dst[(local0 + i * tiles.dilation) * width + c] =
                        combine(simd, b, coefficients, &points);
                }
            }
        }
        Ok(())
    }
}

/// A stride-1 k7 conv, dilated, over an already padded `[time, channels]` input.
#[derive(Debug, Clone)]
pub(crate) struct WinogradConv {
    cin: usize,
    cout: usize,
    dilation: usize,
    /// The transformed filter at each point: `[cin, cout]`.
    u: Vec<PackedMat>,
    bias: Vec<f32>,
    /// A zero bias for the products, so they overwrite rather than add.
    zeros: Vec<f32>,
}

impl WinogradConv {
    /// From PyTorch's `[cout, cin, 7]` weight.
    pub(crate) fn from_torch(
        weight: &[f32],
        bias: &[f32],
        (cout, cin): (usize, usize),
        dilation: usize,
    ) -> Result<Self> {
        ensure!(
            weight.len() == cout * cin * TAPS && bias.len() == cout,
            "conv weight has {} values, expected {cout} x {cin} x {TAPS}",
            weight.len()
        );
        ensure!(dilation > 0, "dilation must be positive");
        ensure!(
            cin.is_multiple_of(LANES) && cout.is_multiple_of(LANES),
            "Winograd convs need multiples of {LANES} channels, not {cin} -> {cout}"
        );
        let u = G
            .iter()
            .map(|g| {
                let mut mat = vec![0.0_f32; cin * cout];
                for (o, filters) in weight.chunks_exact(cin * TAPS).enumerate() {
                    for (c, taps) in filters.as_chunks::<TAPS>().0.iter().enumerate() {
                        let v: f64 = taps.iter().zip(g).map(|(&w, &gj)| f64::from(w) * gj).sum();
                        #[expect(clippy::cast_possible_truncation, reason = "f64 to f32 weights")]
                        let v = v as f32;
                        mat[c * cout + o] = v;
                    }
                }
                PackedMat::pack(mat, cin, cout)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            cin,
            cout,
            dilation,
            u,
            bias: bias.to_vec(),
            zeros: vec![0.0; cout],
        })
    }

    /// Convolves `x` (no further padding): `[len - 6 * dilation, cout]`.
    pub(crate) fn apply(&self, x: &Seq) -> Result<Seq> {
        let (cin, cout, d) = (self.cin, self.cout, self.dilation);
        ensure!(
            x.channels() == cin,
            "conv input is {} wide, expected {cin}",
            x.channels()
        );
        let span = (TAPS - 1) * d;
        ensure!(
            x.len() > span,
            "conv input has {} frames, needs {}",
            x.len(),
            span + 1
        );
        let tout = x.len() - span;
        // A block is `TILE_OUTPUTS` tiles of each phase: that many times
        // `d` consecutive outputs.
        let block = TILE_OUTPUTS * d;
        let target_tiles = (GROUP_VALUES / (ALPHA * (cin + cout))).clamp(32, 256);
        let blocks_per_group = (target_tiles / d).max(1);
        let mut out = vec![0.0; tout * cout];
        out.par_chunks_mut(blocks_per_group * block * cout)
            .enumerate()
            // Scratch is reused across a job's groups: fresh large buffers
            // cost a page fault per page on first touch.
            .try_for_each_init(
                || (Vec::new(), Vec::new()),
                |(v, prod): &mut (Vec<f32>, Vec<f32>), (group, dst)| {
                let rows = dst.len() / cout;
                let tiles = Tiles {
                    start: group * blocks_per_group * block,
                    block,
                    dilation: d,
                    count: rows.div_ceil(block) * d,
                };
                let n = tiles.count;
                // Every value of both is written before it is read.
                v.resize(ALPHA * n * cin, 0.0);
                prod.resize(ALPHA * n * cout, 0.0);
                {
                    let _prof = crate::prof::scope("wino-in", cin, cout, n * ALPHA * cin);
                    kernels::dispatch(InputPass {
                        tiles,
                        input: x.data(),
                        cin,
                        v,
                    })?;
                }
                {
                    let _prof = crate::prof::scope("wino-mul", cin, cout, n * ALPHA * cout);
                    for (p, u) in self.u.iter().enumerate() {
                        u.mul(
                            &v[p * n * cin..(p + 1) * n * cin],
                            cin,
                            Some(&self.zeros),
                            &mut prod[p * n * cout..(p + 1) * n * cout],
                        )?;
                    }
                }
                let _prof = crate::prof::scope("wino-out", cin, cout, rows * cout);
                kernels::dispatch(OutputPass {
                    tiles,
                    prod,
                    bias: &self.bias,
                    dst,
                })
            },
            )?;
        Seq::from_vec(tout, cout, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::ConvKernel;

    fn noise(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                f32::from(u16::try_from(*seed >> 48).unwrap_or(0)) / 65536.0 - 0.5
            })
            .collect()
    }

    /// Largest |difference| from the direct conv, relative to the largest output.
    fn error((cout, cin): (usize, usize), dilation: usize, len: usize) -> Result<f32> {
        let mut seed = 11;
        let w: Vec<f32> = noise(cout * cin * TAPS, &mut seed)
            .iter()
            .map(|v| v / 4.0)
            .collect();
        let b = noise(cout, &mut seed);
        let x = Seq::from_vec(len, cin, noise(len * cin, &mut seed))?;
        let fast = WinogradConv::from_torch(&w, &b, (cout, cin), dilation)?.apply(&x)?;
        let direct = ConvKernel::from_torch(&w, &b, (cout, cin, TAPS), 1, dilation)?.apply(&x)?;
        ensure!(
            fast.len() == direct.len(),
            "length {} vs {}",
            fast.len(),
            direct.len()
        );
        let peak = direct.data().iter().fold(0.0_f32, |a, v| a.max(v.abs()));
        let diff = fast
            .data()
            .iter()
            .zip(direct.data())
            .fold(0.0_f32, |a, (p, q)| a.max((p - q).abs()));
        Ok(diff / peak)
    }

    #[test]
    fn matches_the_direct_conv() -> Result<()> {
        for (shape, dilation, len) in [
            ((16, 8), 1, 101),
            ((8, 16), 3, 257),
            ((24, 8), 9, 400),
            ((8, 8), 9, 55),
            ((64, 64), 1, 3000),
            ((512, 512), 3, 2000),
        ] {
            let e = error(shape, dilation, len)?;
            assert!(e < 5e-5, "{shape:?} d{dilation} len {len}: {e:.2e}");
        }
        Ok(())
    }

    /// `A^T (G g . B^T d)` is the conv of `d` with `g`, in double precision.
    #[test]
    fn transforms_are_a_conv() {
        let g: [f64; TAPS] = [0.3, -1.2, 0.7, 2.0, -0.4, 0.9, -1.5];
        let d: [f64; ALPHA] = [1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 0.1, -0.3, 1.5, -2.5];
        for i in 0..TILE_OUTPUTS {
            let direct: f64 = (0..TAPS).map(|j| g[j] * d[i + j]).sum();
            let fast: f64 = (0..ALPHA)
                .map(|p| {
                    let u: f64 = (0..TAPS).map(|j| G[p][j] * g[j]).sum();
                    let v: f64 = (0..ALPHA).map(|l| BT[p][l] * d[l]).sum();
                    AT[i][p] * u * v
                })
                .sum();
            assert!(
                (direct - fast).abs() < 1e-9,
                "output {i}: {direct} vs {fast}"
            );
        }
    }
}
