//! SIMD kernels: the matrix product against constant weights that every
//! convolution and the LSTM lower to, and the element-wise passes around it.
//!
//! Everything is safe Rust on top of `pulp`, which picks the widest
//! instruction set of the running CPU once (AVX2 with FMA on x86-64, SIMD128
//! in WebAssembly) and falls back to scalar code elsewhere.
//!
//! Every value is computed by the same sequence of operations wherever it
//! sits in a slice, tile or batch, so results do not depend on the number of
//! threads or on how a signal is cut into chunks.

#![expect(
    clippy::inline_always,
    reason = "the kernels must be inlined into the function compiled for the detected instruction set"
)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder as _};
use pulp::{Arch, Simd, WithSimd};

/// Outputs up to which each one is computed as a plain dot product.
const MAX_DOT_OUTPUTS: usize = 4;
/// Rows of a weight panel multiplied in one go. A block of a panel stays in
/// the first-level cache while the rows of the left operand stream past it.
const PANEL_BLOCK: usize = 128;
/// Registers of at least this many values use the wide kernel shape.
const WIDE_LANES: usize = 8;
/// Shortest repeating scale and offset pattern of an [`Affine`], in values.
const MIN_AFFINE_PERIOD: usize = 256;
/// Registers whose values are added in single precision before the total
/// goes on in double precision.
const MOMENT_BLOCK: usize = 16;
/// `1.5 * 2^23`: adding it to a small float rounds that float to an integer
/// and leaves the integer in the low mantissa bits.
const ROUND_MAGIC: f32 = 12_582_912.0;
/// High and low parts of `ln 2`; the high part has few enough bits for
/// `n * LN2_HI` to be exact.
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
/// Below this `exp(x) - 1` is `-1` in single precision.
const EXPM1_FLOOR: f32 = -32.0;
/// Largest magnitude whose exponential is formed without overflow.
const EXP_LIMIT: f32 = 87.0;

/// The instruction set to run on, detected once.
fn arch() -> Arch {
    static ARCH: OnceLock<Arch> = OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

/// `f32` values per SIMD register of `S`.
const fn lanes_of<S: Simd>() -> usize {
    size_of::<S::f32s>() / size_of::<f32>()
}

/// `f32` values per SIMD register on this CPU; 1 without SIMD.
pub(crate) fn lanes() -> usize {
    struct Lanes;
    impl WithSimd for Lanes {
        type Output = usize;

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) -> usize {
            lanes_of::<S>()
        }
    }
    arch().dispatch(Lanes)
}

/// Runs `pass` compiled for the detected instruction set; plain element-wise
/// loops in an `#[inline(always)]` `with_simd` vectorise for it.
pub(crate) fn dispatch<P: WithSimd>(pass: P) -> P::Output {
    arch().dispatch(pass)
}

/// High and low parts of pi; the high part (201/64) has few enough bits for
/// `k * PI_HI` to be exact for any `k` below `2^16`.
const PI_HI: f32 = 3.140_625;
const PI_LO: f32 = 9.676_536e-4;
/// Taylor coefficients of `sin(r)`, odd terms 3 to 11. On `|r| <= pi/2` the
/// truncation error is below `6e-8`.
const SIN_C3: f32 = -1.0 / 6.0;
const SIN_C5: f32 = 1.0 / 120.0;
const SIN_C7: f32 = -1.0 / 5040.0;
const SIN_C9: f32 = 1.0 / 362_880.0;
const SIN_C11: f32 = -1.0 / 39_916_800.0;

/// DAC's Snake activation for one value: `x + sin²(a x) / a`, with `inv`
/// the precomputed `1 / (a + 1e-9)`.
///
/// `sin²` has period pi, so `a x` is reduced to `[-pi/2, pi/2]` (round to the
/// nearest multiple of pi, two-part subtraction) before the polynomial. The
/// rounding is exact while `|a x| < 2^22`.
#[inline(always)]
fn snake_value(x: f32, a: f32, inv: f32) -> f32 {
    let y = a * x;
    let k = (y * std::f32::consts::FRAC_1_PI + ROUND_MAGIC) - ROUND_MAGIC;
    let r = (y - k * PI_HI) - k * PI_LO;
    let r2 = r * r;
    let s =
        r * (1.0 + r2 * (SIN_C3 + r2 * (SIN_C5 + r2 * (SIN_C7 + r2 * (SIN_C9 + r2 * SIN_C11)))));
    x + s * s * inv
}

/// Snake in place over time-major frames of `alpha.len()` channels.
///
/// Plain element-wise code, compiled for the detected instruction set so it
/// vectorises across channels. Every value goes through the same operations
/// wherever it sits, so results do not depend on chunking.
pub(crate) fn snake_in_place(values: &mut [f32], alpha: &[f32], inv: &[f32]) {
    struct SnakePass<'a> {
        values: &'a mut [f32],
        alpha: &'a [f32],
        inv: &'a [f32],
    }
    impl WithSimd for SnakePass<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) {
            let channels = self.alpha.len();
            for frame in self.values.chunks_exact_mut(channels) {
                for ((v, &a), &inv) in frame.iter_mut().zip(self.alpha).zip(self.inv) {
                    *v = snake_value(*v, a, inv);
                }
            }
        }
    }
    if alpha.is_empty() {
        return;
    }
    arch().dispatch(SnakePass { values, alpha, inv });
}

/// `out[m, n] += a · b`, where row `r` of `a` is `a[r * a_stride..][..k]` (rows
/// may overlap) and `b`, `out` are row-major. Portable, without SIMD of ours.
pub(crate) fn gemm_acc(
    a: &[f32],
    a_stride: usize,
    b: &[f32],
    (m, k, n): (usize, usize, usize),
    out: &mut [f32],
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let a = ArrayView2::from_shape((m, k).strides((a_stride.max(1), 1)), a)?;
    let b = ArrayView2::from_shape((k, n), b)?;
    let mut c = ArrayViewMut2::from_shape((m, n), out)?;
    general_mat_mul(1.0, &a, &b, 1.0, &mut c);
    Ok(())
}

/// How a [`PackedMat`] stores its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Row-major `[k][n]`, for CPUs without SIMD support here.
    Plain,
    /// Column-major `[n][k]`: a few outputs, each a dot product.
    Columns,
    /// Panels of `width` columns, `[panel][k][width]`, zero padded, so the
    /// micro-kernel reads a panel front to back.
    Panels { width: usize },
}

/// Columns per weight panel for registers of `lanes` values: sixteen, which
/// is two wide registers or four narrow ones (see [`Mul`]).
const fn panel_width(lanes: usize) -> usize {
    if lanes >= WIDE_LANES {
        2 * lanes
    } else {
        4 * lanes
    }
}

/// A constant `[k, n]` matrix laid out for multiplying from the left.
#[derive(Debug, Clone)]
pub(crate) struct PackedMat {
    k: usize,
    n: usize,
    layout: Layout,
    data: Vec<f32>,
}

impl PackedMat {
    /// Packs a row-major `[k, n]` matrix.
    pub(crate) fn pack(weight: Vec<f32>, k: usize, n: usize) -> Result<Self> {
        ensure!(weight.len() == k * n, "matrix size mismatch");
        let lanes = lanes();
        let (layout, data) = if lanes == 1 || k == 0 || n == 0 {
            (Layout::Plain, weight)
        } else if n <= MAX_DOT_OUTPUTS {
            let mut data = vec![0.0; k * n];
            for (o, column) in data.chunks_exact_mut(k).enumerate() {
                for (dst, row) in column.iter_mut().zip(weight.chunks_exact(n)) {
                    *dst = row[o];
                }
            }
            (Layout::Columns, data)
        } else {
            let width = panel_width(lanes);
            let mut data = vec![0.0; n.div_ceil(width) * k * width];
            for (p, panel) in data.chunks_exact_mut(k * width).enumerate() {
                let first = p * width;
                let cols = width.min(n - first);
                for (dst, row) in panel.chunks_exact_mut(width).zip(weight.chunks_exact(n)) {
                    dst[..cols].copy_from_slice(&row[first..first + cols]);
                }
            }
            (Layout::Panels { width }, data)
        };
        Ok(Self { k, n, layout, data })
    }

    /// The matrix in row-major order again.
    pub(crate) fn unpack(&self) -> Vec<f32> {
        let (k, n) = (self.k, self.n);
        match self.layout {
            Layout::Plain => self.data.clone(),
            Layout::Columns => {
                let mut out = vec![0.0; k * n];
                for (o, column) in self.data.chunks_exact(k).enumerate() {
                    for (row, &v) in out.chunks_exact_mut(n).zip(column) {
                        row[o] = v;
                    }
                }
                out
            }
            Layout::Panels { width } => {
                let mut out = vec![0.0; k * n];
                for (p, panel) in self.data.chunks_exact(k * width).enumerate() {
                    let first = p * width;
                    let cols = width.min(n - first);
                    for (row, src) in out.chunks_exact_mut(n).zip(panel.chunks_exact(width)) {
                        row[first..first + cols].copy_from_slice(&src[..cols]);
                    }
                }
                out
            }
        }
    }

    /// `out = start + a · self`, where row `r` of `a` is
    /// `a[r * a_stride..][..k]` (rows may overlap), `out` is row-major
    /// `[rows, n]`, and `start` is `bias` in every row or, without a bias,
    /// what `out` holds on entry.
    ///
    /// # Errors
    /// Fails when the slices do not match the matrix.
    pub(crate) fn mul(
        &self,
        a: &[f32],
        a_stride: usize,
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        let (k, n) = (self.k, self.n);
        if n == 0 || out.is_empty() {
            return Ok(());
        }
        let rows = out.len() / n;
        ensure!(
            out.len() == rows * n
                && a.len() >= (rows - 1) * a_stride + k
                && bias.is_none_or(|bias| bias.len() == n),
            "matrix product size mismatch"
        );
        match self.layout {
            Layout::Plain => {
                if let Some(bias) = bias {
                    for row in out.chunks_exact_mut(n) {
                        row.copy_from_slice(bias);
                    }
                }
                gemm_acc(a, a_stride, &self.data, (rows, k, n), out)
            }
            Layout::Columns | Layout::Panels { .. } => {
                ensure!(
                    self.layout == Layout::Columns
                        || self.layout
                            == Layout::Panels {
                                width: panel_width(lanes())
                            },
                    "matrix was packed for another instruction set"
                );
                arch().dispatch(Mul {
                    mat: self,
                    a,
                    a_stride,
                    bias,
                    out,
                });
                Ok(())
            }
        }
    }
}

/// A [`PackedMat::mul`] call, checked.
///
/// The micro-kernel keeps a small block of outputs in registers, `R` rows of
/// the left operand by one panel of columns, and walks down the panel: every
/// weight row is loaded once and multiplied by one broadcast value per row.
/// Its shape follows the registers. Sixteen wide registers hold three rows
/// by two registers, in two running sums each (for even and for odd weight
/// rows), with room for the weights. Narrow registers hold two rows by four
/// registers in one sum each; anything more spills to memory where only
/// sixteen registers exist.
struct Mul<'a> {
    mat: &'a PackedMat,
    a: &'a [f32],
    a_stride: usize,
    bias: Option<&'a [f32]>,
    out: &'a mut [f32],
}

impl WithSimd for Mul<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let (k, n) = (self.mat.k, self.mat.n);
        if self.mat.layout == Layout::Columns {
            for (r, out_row) in self.out.chunks_exact_mut(n).enumerate() {
                let row = &self.a[r * self.a_stride..][..k];
                for (o, (dst, column)) in out_row
                    .iter_mut()
                    .zip(self.mat.data.chunks_exact(k))
                    .enumerate()
                {
                    *dst = self.bias.map_or(*dst, |bias| bias[o]) + dot(simd, row, column);
                }
            }
        } else if lanes_of::<S>() >= WIDE_LANES {
            self.panels::<S, 2, 2, 3>(simd);
        } else {
            self.panels::<S, 4, 1, 2>(simd);
        }
    }
}

impl Mul<'_> {
    /// The product for panels of `V` registers, with `U` running sums per
    /// output and up to `R` rows of the left operand at a time.
    #[inline(always)]
    fn panels<S: Simd, const V: usize, const U: usize, const R: usize>(self, simd: S) {
        let Self {
            mat,
            a,
            a_stride,
            bias,
            out,
        } = self;
        let (k, n) = (mat.k, mat.n);
        let width = V * lanes_of::<S>();
        let rows = out.len() / n;
        for (p, panel) in mat.data.chunks_exact(k * width).enumerate() {
            let first = p * width;
            let cols = width.min(n - first);
            let (vectors, _) = S::as_simd_f32s(panel);
            let (b, _) = vectors.as_chunks::<V>();
            let bias = bias.map(|bias| load_group::<S, V>(simd, &bias[first..first + cols]));
            for start in (0..k).step_by(PANEL_BLOCK) {
                let block = Block {
                    simd,
                    b: &b[start..(start + PANEL_BLOCK).min(k)],
                    a,
                    a_offset: start,
                    a_stride,
                    n,
                    first,
                    cols,
                    init: if start == 0 { bias } else { None },
                };
                let mut r = 0;
                while rows - r >= R {
                    block.rows::<R, U>(r, out);
                    r += R;
                }
                match rows - r {
                    2 => block.rows::<2, U>(r, out),
                    1 => block.rows::<1, U>(r, out),
                    _ => {}
                }
            }
        }
    }
}

/// One block of one weight panel, about to meet the rows of the left operand.
struct Block<'a, S: Simd, const V: usize> {
    simd: S,
    /// The block's rows, each `V` registers wide.
    b: &'a [[S::f32s; V]],
    a: &'a [f32],
    /// First value of each row of `a` the block multiplies.
    a_offset: usize,
    a_stride: usize,
    n: usize,
    /// First output column of the panel and how many are real.
    first: usize,
    cols: usize,
    /// What the outputs start from, when not from themselves.
    init: Option<[S::f32s; V]>,
}

impl<S: Simd, const V: usize> Block<'_, S, V> {
    /// Adds the block's products into rows `r..r + R` of `out`.
    ///
    /// Weight row `j` goes to running sum `j % U`, and the sums are joined at
    /// the end, for every `R`: several sums hide the latency of the adds,
    /// and the fixed order makes a row's result independent of the rows
    /// computed with it.
    #[inline(always)]
    fn rows<const R: usize, const U: usize>(&self, r: usize, out: &mut [f32]) {
        let simd = self.simd;
        let len = self.b.len();
        let rows: [&[f32]; R] =
            std::array::from_fn(|i| &self.a[(r + i) * self.a_stride + self.a_offset..][..len]);
        let mut acc = [[[simd.splat_f32s(0.0); V]; R]; U];
        for (i, start) in acc[0].iter_mut().enumerate() {
            *start = self.init.unwrap_or_else(|| {
                load_group::<S, V>(simd, &out[(r + i) * self.n + self.first..][..self.cols])
            });
        }
        let (steps, last) = self.b.as_chunks::<U>();
        for (s, step) in steps.iter().enumerate() {
            for u in 0..U {
                for i in 0..R {
                    let x = simd.splat_f32s(rows[i][U * s + u]);
                    for v in 0..V {
                        acc[u][i][v] = simd.mul_add_e_f32s(x, step[u][v], acc[u][i][v]);
                    }
                }
            }
        }
        for (u, b) in last.iter().enumerate() {
            for i in 0..R {
                let x = simd.splat_f32s(rows[i][U * steps.len() + u]);
                for v in 0..V {
                    acc[u][i][v] = simd.mul_add_e_f32s(x, b[v], acc[u][i][v]);
                }
            }
        }
        for i in 0..R {
            let mut sum = acc[0][i];
            for extra in &acc[1..] {
                for v in 0..V {
                    sum[v] = simd.add_f32s(sum[v], extra[i][v]);
                }
            }
            store_group::<S, V>(
                simd,
                &mut out[(r + i) * self.n + self.first..][..self.cols],
                sum,
            );
        }
    }
}

/// `V` registers from up to `V * lanes` values; missing values read as zero.
#[inline(always)]
fn load_group<S: Simd, const V: usize>(simd: S, values: &[f32]) -> [S::f32s; V] {
    let lanes = lanes_of::<S>();
    let (vectors, tail) = S::as_simd_f32s(values);
    match (vectors.first_chunk::<V>(), tail) {
        (Some(group), []) => *group,
        _ => std::array::from_fn(|v| {
            let end = values.len().min((v + 1) * lanes);
            simd.partial_load_f32s(&values[end.min(v * lanes)..end])
        }),
    }
}

/// Stores the first `values.len()` lanes of `V` registers.
#[inline(always)]
fn store_group<S: Simd, const V: usize>(simd: S, values: &mut [f32], group: [S::f32s; V]) {
    let lanes = lanes_of::<S>();
    if values.len() == V * lanes
        && let Some(whole) = S::as_mut_simd_f32s(values).0.first_chunk_mut::<V>()
    {
        *whole = group;
    } else {
        for (chunk, register) in values.chunks_mut(lanes).zip(group) {
            simd.partial_store_f32s(chunk, register);
        }
    }
}

/// Dot product of two slices of the same length, with four running sums.
#[inline(always)]
fn dot<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    let (a_vectors, a_tail) = S::as_simd_f32s(a);
    let (b_vectors, b_tail) = S::as_simd_f32s(b);
    let (a_blocks, a_rest) = a_vectors.as_chunks::<4>();
    let (b_blocks, b_rest) = b_vectors.as_chunks::<4>();
    let mut acc = [simd.splat_f32s(0.0); 4];
    for (x, y) in a_blocks.iter().zip(b_blocks) {
        for lane in 0..4 {
            acc[lane] = simd.mul_add_e_f32s(x[lane], y[lane], acc[lane]);
        }
    }
    for (&x, &y) in a_rest.iter().zip(b_rest) {
        acc[0] = simd.mul_add_e_f32s(x, y, acc[0]);
    }
    acc[0] = simd.mul_add_e_f32s(
        simd.partial_load_f32s(a_tail),
        simd.partial_load_f32s(b_tail),
        acc[0],
    );
    simd.reduce_sum_f32s(
        simd.add_f32s(simd.add_f32s(acc[0], acc[1]), simd.add_f32s(acc[2], acc[3])),
    )
}

/// `(expm1(r), 2^n)` for `x = n ln 2 + r` with `|r| <= ln 2 / 2`, from which
/// `exp(x) = 2^n expm1(r) + 2^n` and `expm1(x) = 2^n expm1(r) + (2^n - 1)`.
/// `x` must lie within `-EXP_LIMIT..=EXP_LIMIT`. Good to about an ulp.
#[inline(always)]
fn exp_parts<S: Simd>(simd: S, x: S::f32s) -> (S::f32s, S::f32s) {
    let magic = simd.splat_f32s(ROUND_MAGIC);
    let shifted = simd.mul_add_e_f32s(x, simd.splat_f32s(std::f32::consts::LOG2_E), magic);
    let n = simd.sub_f32s(shifted, magic);
    let r = simd.mul_add_e_f32s(n, simd.splat_f32s(-LN2_HI), x);
    let r = simd.mul_add_e_f32s(n, simd.splat_f32s(-LN2_LO), r);
    // expm1(r) = r + r^2 (1/2 + r/6 + ... + r^5/5040).
    let mut poly = simd.splat_f32s(1.0 / 5040.0);
    for coefficient in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5] {
        poly = simd.mul_add_e_f32s(poly, r, simd.splat_f32s(coefficient));
    }
    let expm1 = simd.mul_add_e_f32s(poly, simd.mul_f32s(r, r), r);
    // The low bits of `shifted` hold n; move n + 127 into the exponent field.
    let exponent = simd.sub_u32s(
        simd.transmute_u32s_f32s(shifted),
        simd.splat_u32s(ROUND_MAGIC.to_bits() - 127),
    );
    let scale = simd.transmute_f32s_u32s(simd.mul_u32s(exponent, simd.splat_u32s(1 << 23)));
    (expm1, scale)
}

/// `exp(x) - 1` for `x <= 0`.
#[inline(always)]
fn expm1_negative<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let x = simd.max_f32s(x, simd.splat_f32s(EXPM1_FLOOR));
    let (expm1, scale) = exp_parts(simd, x);
    simd.mul_add_e_f32s(scale, expm1, simd.sub_f32s(scale, simd.splat_f32s(1.0)))
}

/// ELU with alpha = 1.
#[inline(always)]
fn elu_vector<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let zero = simd.splat_f32s(0.0);
    let negative = expm1_negative(simd, simd.min_f32s(x, zero));
    simd.select_f32s(simd.greater_than_f32s(x, zero), x, negative)
}

/// The logistic function.
#[inline(always)]
fn sigmoid_vector<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let limit = simd.splat_f32s(EXP_LIMIT);
    let x = simd.min_f32s(simd.max_f32s(simd.neg_f32s(x), simd.neg_f32s(limit)), limit);
    let (expm1, scale) = exp_parts(simd, x);
    let exp = simd.mul_add_e_f32s(scale, expm1, scale);
    let one = simd.splat_f32s(1.0);
    simd.div_f32s(one, simd.add_f32s(one, exp))
}

/// The hyperbolic tangent, as `-e / (e + 2)` with `e = expm1(-2 |x|)`.
#[inline(always)]
fn tanh_vector<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let magnitude = simd.abs_f32s(x);
    let e = expm1_negative(simd, simd.mul_f32s(magnitude, simd.splat_f32s(-2.0)));
    let positive = simd.div_f32s(simd.neg_f32s(e), simd.add_f32s(e, simd.splat_f32s(2.0)));
    // Give the result the sign of `x`.
    let sign = simd.and_u32s(simd.transmute_u32s_f32s(x), simd.splat_u32s(1 << 31));
    simd.transmute_f32s_u32s(simd.or_u32s(simd.transmute_u32s_f32s(positive), sign))
}

/// A per-channel scale and offset for time-major data, repeated to a whole
/// number of registers so a pass needs no per-frame tail.
#[derive(Debug, Clone)]
pub(crate) struct Affine {
    scale: Vec<f32>,
    offset: Vec<f32>,
}

impl Affine {
    /// Builds the pattern from one `(scale, offset)` per channel.
    pub(crate) fn new(channels: impl ExactSizeIterator<Item = (f32, f32)> + Clone) -> Self {
        let count = channels.len().max(1);
        let lanes = lanes();
        // A common multiple of the channel count and the lanes, long enough
        // for the work per repetition to dwarf its set-up.
        let mut period = count;
        while !period.is_multiple_of(lanes) || period < MIN_AFFINE_PERIOD {
            period += count;
        }
        let repeated = channels.cycle().take(period);
        Self {
            scale: repeated.clone().map(|(scale, _)| scale).collect(),
            offset: repeated.map(|(_, offset)| offset).collect(),
        }
    }

    /// The transform that changes nothing.
    pub(crate) fn identity(channels: usize) -> Self {
        Self::new(std::iter::repeat_n((1.0, 0.0), channels))
    }

    /// Values after which the pattern repeats: a whole number of frames.
    pub(crate) const fn period(&self) -> usize {
        self.scale.len()
    }
}

/// `elu?(v * scale + offset)` for one register, the affine part optional.
#[inline(always)]
fn map_vector<S: Simd>(
    simd: S,
    v: S::f32s,
    affine: Option<(S::f32s, S::f32s)>,
    elu: bool,
) -> S::f32s {
    let y = match affine {
        Some((scale, offset)) => simd.mul_add_e_f32s(v, scale, offset),
        None => v,
    };
    if elu { elu_vector(simd, y) } else { y }
}

/// `elu?((a * scale_a + offset_a) + (b * scale_b + offset_b))` for one
/// register; each operand comes with its scale and offset.
#[inline(always)]
fn sum_vector<S: Simd>(
    simd: S,
    (a, a_scale, a_offset): (S::f32s, S::f32s, S::f32s),
    (b, b_scale, b_offset): (S::f32s, S::f32s, S::f32s),
    elu: bool,
) -> S::f32s {
    let a = simd.mul_add_e_f32s(a, a_scale, a_offset);
    let b = simd.mul_add_e_f32s(b, b_scale, b_offset);
    map_vector(simd, simd.add_f32s(a, b), None, elu)
}

/// The registers of an optional affine pattern, or empty slices.
#[inline(always)]
fn pattern<S: Simd>(affine: Option<&Affine>) -> (&[S::f32s], &[S::f32s], usize) {
    affine.map_or((&[], &[], usize::MAX), |affine| {
        (
            S::as_simd_f32s(&affine.scale).0,
            S::as_simd_f32s(&affine.offset).0,
            affine.period(),
        )
    })
}

/// `values = elu?(values * scale + offset)`, in place. `values` starts at the
/// start of a frame; the affine part and the ELU are both optional.
pub(crate) fn map_in_place(values: &mut [f32], affine: Option<&Affine>, elu: bool) {
    struct InPlace<'a> {
        values: &'a mut [f32],
        affine: Option<&'a Affine>,
        elu: bool,
    }
    impl WithSimd for InPlace<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let (scale, offset, period) = pattern::<S>(self.affine);
            let elu = self.elu;
            for chunk in self.values.chunks_mut(period) {
                let (vectors, tail) = S::as_mut_simd_f32s(chunk);
                let done = vectors.len();
                if self.affine.is_some() {
                    for ((v, &scale), &offset) in vectors.iter_mut().zip(scale).zip(offset) {
                        *v = map_vector(simd, *v, Some((scale, offset)), elu);
                    }
                } else {
                    for v in vectors {
                        *v = map_vector(simd, *v, None, elu);
                    }
                }
                if !tail.is_empty() {
                    let affine = self.affine.map(|_| (scale[done], offset[done]));
                    let v = map_vector(simd, simd.partial_load_f32s(tail), affine, elu);
                    simd.partial_store_f32s(tail, v);
                }
            }
        }
    }
    arch().dispatch(InPlace {
        values,
        affine,
        elu,
    });
}

/// `dst = elu?(src * scale + offset)`; see [`map_in_place`].
pub(crate) fn map_into(src: &[f32], dst: &mut [f32], affine: Option<&Affine>, elu: bool) {
    struct Into<'a> {
        src: &'a [f32],
        dst: &'a mut [f32],
        affine: Option<&'a Affine>,
        elu: bool,
    }
    impl WithSimd for Into<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let (scale, offset, period) = pattern::<S>(self.affine);
            let elu = self.elu;
            for (dst, src) in self.dst.chunks_mut(period).zip(self.src.chunks(period)) {
                let (dst_vectors, dst_tail) = S::as_mut_simd_f32s(dst);
                let (src_vectors, src_tail) = S::as_simd_f32s(src);
                let done = dst_vectors.len();
                if self.affine.is_some() {
                    for (((d, &v), &scale), &offset) in dst_vectors
                        .iter_mut()
                        .zip(src_vectors)
                        .zip(scale)
                        .zip(offset)
                    {
                        *d = map_vector(simd, v, Some((scale, offset)), elu);
                    }
                } else {
                    for (d, &v) in dst_vectors.iter_mut().zip(src_vectors) {
                        *d = map_vector(simd, v, None, elu);
                    }
                }
                if !dst_tail.is_empty() {
                    let affine = self.affine.map(|_| (scale[done], offset[done]));
                    let v = map_vector(simd, simd.partial_load_f32s(src_tail), affine, elu);
                    simd.partial_store_f32s(dst_tail, v);
                }
            }
        }
    }
    if affine.is_none() && !elu {
        dst.copy_from_slice(src);
        return;
    }
    arch().dispatch(Into {
        src,
        dst,
        affine,
        elu,
    });
}

/// `dst = elu?((a * scale_a + offset_a) + (b * scale_b + offset_b))`, the
/// patterns of the same period; see [`map_in_place`].
pub(crate) fn sum_into(
    (a, a_affine): (&[f32], &Affine),
    (b, b_affine): (&[f32], &Affine),
    dst: &mut [f32],
    elu: bool,
) {
    struct Sum<'a> {
        a: (&'a [f32], &'a Affine),
        b: (&'a [f32], &'a Affine),
        dst: &'a mut [f32],
        elu: bool,
    }
    impl WithSimd for Sum<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let (a_scale, a_offset, period) = pattern::<S>(Some(self.a.1));
            let (b_scale, b_offset, _) = pattern::<S>(Some(self.b.1));
            let elu = self.elu;
            for ((dst, a), b) in self
                .dst
                .chunks_mut(period)
                .zip(self.a.0.chunks(period))
                .zip(self.b.0.chunks(period))
            {
                let (dst_vectors, dst_tail) = S::as_mut_simd_f32s(dst);
                let (a_vectors, a_tail) = S::as_simd_f32s(a);
                let (b_vectors, b_tail) = S::as_simd_f32s(b);
                let done = dst_vectors.len();
                for (at, ((d, &a), &b)) in dst_vectors
                    .iter_mut()
                    .zip(a_vectors)
                    .zip(b_vectors)
                    .enumerate()
                {
                    *d = sum_vector(
                        simd,
                        (a, a_scale[at], a_offset[at]),
                        (b, b_scale[at], b_offset[at]),
                        elu,
                    );
                }
                if !dst_tail.is_empty() {
                    let v = sum_vector(
                        simd,
                        (
                            simd.partial_load_f32s(a_tail),
                            a_scale[done],
                            a_offset[done],
                        ),
                        (
                            simd.partial_load_f32s(b_tail),
                            b_scale[done],
                            b_offset[done],
                        ),
                        elu,
                    );
                    simd.partial_store_f32s(dst_tail, v);
                }
            }
        }
    }
    arch().dispatch(Sum {
        a: (a, a_affine),
        b: (b, b_affine),
        dst,
        elu,
    });
}

/// Sum and sum of squares of `values`.
///
/// Single-precision registers add up short blocks, whose totals are then
/// added in double precision: a block is too short for its rounding to
/// matter, and the long sums carry no single-precision error.
pub(crate) fn moments(values: &[f32]) -> (f64, f64) {
    struct Moments<'a>(&'a [f32]);
    impl WithSimd for Moments<'_> {
        type Output = (f64, f64);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> (f64, f64) {
            let (vectors, tail) = S::as_simd_f32s(self.0);
            let (mut sum, mut sum_sq) = (0.0_f64, 0.0_f64);
            let zero = simd.splat_f32s(0.0);
            for block in vectors.chunks(MOMENT_BLOCK) {
                let (mut s, mut q) = ([zero; 2], [zero; 2]);
                let (pairs, last) = block.as_chunks::<2>();
                for &[a, b] in pairs {
                    s[0] = simd.add_f32s(s[0], a);
                    s[1] = simd.add_f32s(s[1], b);
                    q[0] = simd.mul_add_e_f32s(a, a, q[0]);
                    q[1] = simd.mul_add_e_f32s(b, b, q[1]);
                }
                if let [a] = last {
                    s[0] = simd.add_f32s(s[0], *a);
                    q[0] = simd.mul_add_e_f32s(*a, *a, q[0]);
                }
                sum += f64::from(simd.reduce_sum_f32s(simd.add_f32s(s[0], s[1])));
                sum_sq += f64::from(simd.reduce_sum_f32s(simd.add_f32s(q[0], q[1])));
            }
            for &v in tail {
                let v = f64::from(v);
                sum += v;
                sum_sq += v * v;
            }
            (sum, sum_sq)
        }
    }
    arch().dispatch(Moments(values))
}

/// One LSTM time step for one sequence, given the gate pre-activations
/// `pre` (`[input | forget | cell | output]`, each `c.len()` wide): updates
/// the cell state `c` and writes the new hidden state to `h`.
pub(crate) fn lstm_gates(pre: &[f32], c: &mut [f32], h: &mut [f32]) {
    arch().dispatch(Gates { pre, c, h });
}

/// New cell and hidden state from the `[input, forget, cell, output]` gate
/// pre-activations and the old cell state.
#[inline(always)]
fn gates_vector<S: Simd>(simd: S, [i, f, g, o]: [S::f32s; 4], c: S::f32s) -> (S::f32s, S::f32s) {
    let i = sigmoid_vector(simd, i);
    let f = sigmoid_vector(simd, f);
    let g = tanh_vector(simd, g);
    let o = sigmoid_vector(simd, o);
    let c = simd.mul_add_e_f32s(f, c, simd.mul_f32s(i, g));
    (c, simd.mul_f32s(o, tanh_vector(simd, c)))
}

struct Gates<'a> {
    pre: &'a [f32],
    c: &'a mut [f32],
    h: &'a mut [f32],
}

impl WithSimd for Gates<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let d = self.c.len();
        let (input, rest) = self.pre.split_at(d);
        let (forget, rest) = rest.split_at(d);
        let (cell, output) = rest.split_at(d);
        let (i_vectors, i_tail) = S::as_simd_f32s(input);
        let (f_vectors, f_tail) = S::as_simd_f32s(forget);
        let (g_vectors, g_tail) = S::as_simd_f32s(cell);
        let (o_vectors, o_tail) = S::as_simd_f32s(output);
        let (c_vectors, c_tail) = S::as_mut_simd_f32s(self.c);
        let (h_vectors, h_tail) = S::as_mut_simd_f32s(self.h);
        for (((((c, h), &i), &f), &g), &o) in c_vectors
            .iter_mut()
            .zip(h_vectors)
            .zip(i_vectors)
            .zip(f_vectors)
            .zip(g_vectors)
            .zip(o_vectors)
        {
            (*c, *h) = gates_vector(simd, [i, f, g, o], *c);
        }
        if !c_tail.is_empty() {
            let gates = [i_tail, f_tail, g_tail, o_tail].map(|tail| simd.partial_load_f32s(tail));
            let (c, h) = gates_vector(simd, gates, simd.partial_load_f32s(c_tail));
            simd.partial_store_f32s(c_tail, c);
            simd.partial_store_f32s(h_tail, h);
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "small test values"
)]
mod tests {
    use super::*;

    #[test]
    fn snake_matches_exact_sine() {
        // x from -60 to 60 and alphas from 0.05 to 20: |a x| up to 1200.
        let alpha = [0.05_f32, 0.7, 1.0, 3.3, 20.0];
        let inv: Vec<f32> = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
        let xs: Vec<f32> = (0..24_001).map(|i| (i as f32 - 12_000.0) * 0.005).collect();
        let mut values: Vec<f32> = xs.iter().flat_map(|&x| [x; 5]).collect();
        snake_in_place(&mut values, &alpha, &inv);
        let mut worst = 0.0_f64;
        for (frame, &x) in values.as_chunks::<5>().0.iter().zip(&xs) {
            for ((&got, &a), &i) in frame.iter().zip(&alpha).zip(&inv) {
                let ax = f64::from(a * x);
                let exact = f64::from(x) + ax.sin().powi(2) * f64::from(i);
                worst = worst.max((f64::from(got) - exact).abs() / (1.0 + exact.abs()));
            }
        }
        assert!(worst < 2e-6, "worst relative error {worst:e}");
    }

    fn pattern(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f64).mul_add(0.37, f64::from(seed)).sin() * 0.8) as f32)
            .collect()
    }

    #[test]
    fn packed_product_matches_naive_for_every_shape() -> Result<()> {
        for &(rows, k, n, stride) in &[
            (1, 5, 3, 5),
            (7, 9, 2, 3),
            (4, 16, 16, 16),
            (11, 33, 40, 6),
            (5, 1030, 70, 2),
            (2, 7, 1, 7),
        ] {
            let weight = pattern(k * n, 0.3);
            let a = pattern((rows - 1) * stride + k, 1.1);
            let bias = pattern(n, 2.0);
            let mat = PackedMat::pack(weight.clone(), k, n)?;
            assert_eq!(mat.unpack(), weight);
            let mut out = vec![0.0; rows * n];
            mat.mul(&a, stride, Some(&bias), &mut out)?;
            // Accumulating on top of the bias gives the same values.
            let mut again: Vec<f32> = bias.iter().copied().cycle().take(rows * n).collect();
            mat.mul(&a, stride, None, &mut again)?;
            for r in 0..rows {
                for o in 0..n {
                    let want: f64 = f64::from(bias[o])
                        + (0..k)
                            .map(|j| f64::from(a[r * stride + j]) * f64::from(weight[j * n + o]))
                            .sum::<f64>();
                    let got = f64::from(out[r * n + o]);
                    assert!((got - want).abs() < 1e-4, "{rows}x{k}x{n}: {got} vs {want}");
                }
            }
            if mat.layout != Layout::Plain {
                assert_eq!(out, again);
                // A row's values do not depend on the rows computed with it.
                let mut single = vec![0.0; n];
                mat.mul(&a[(rows - 1) * stride..], stride, Some(&bias), &mut single)?;
                assert_eq!(single, out[(rows - 1) * n..]);
            }
        }
        Ok(())
    }

    #[test]
    fn element_wise_functions_match_the_library() {
        let mut xs: Vec<f32> = (-400..=400).map(|i| i as f32 * 0.11).collect();
        xs.extend([0.0, -0.0, 1e-8, -1e-8, -1e-3, 3e-4, -100.0, 100.0, -17.4]);
        let mut got = xs.clone();
        map_in_place(&mut got, None, true);
        for (&x, &y) in xs.iter().zip(&got) {
            let want = if x > 0.0 { x } else { x.exp_m1() };
            assert!(
                (y - want).abs() <= 3.0 * f32::EPSILON * want.abs(),
                "elu({x}) = {y}, want {want}"
            );
        }
        // The gates give sigmoid and tanh: with c = 0, forget = anything,
        // h = sigmoid(o) * tanh(sigmoid(i) * tanh(g)).
        let d = xs.len();
        let mut pre = vec![0.0; 4 * d];
        pre[..d].fill(40.0);
        pre[2 * d..3 * d].copy_from_slice(&xs);
        pre[3 * d..].fill(40.0);
        let (mut c, mut h) = (vec![0.0; d], vec![0.0; d]);
        lstm_gates(&pre, &mut c, &mut h);
        for ((&x, &cell), &hidden) in xs.iter().zip(&c).zip(&h) {
            assert!(
                (cell - x.tanh()).abs() <= 4.0 * f32::EPSILON * x.tanh().abs(),
                "tanh({x}) = {cell}"
            );
            assert!((hidden - x.tanh().tanh()).abs() <= 8.0 * f32::EPSILON);
        }
        pre[..d].copy_from_slice(&xs);
        pre[2 * d..3 * d].fill(40.0);
        c.fill(0.0);
        lstm_gates(&pre, &mut c, &mut h);
        for (&x, &cell) in xs.iter().zip(&c) {
            let want = 1.0 / (1.0 + (-x).exp());
            assert!(
                (cell - want).abs() <= (4.0 * f32::EPSILON).mul_add(want, 1e-37),
                "sigmoid({x}) = {cell}"
            );
        }
    }

    #[test]
    fn affine_pass_repeats_the_channel_pattern() {
        for channels in [1, 2, 3, 16, 24] {
            let scale = pattern(channels, 0.5);
            let offset = pattern(channels, 1.5);
            let affine = Affine::new(scale.iter().copied().zip(offset.iter().copied()));
            let xs = pattern(channels * 13, 0.9);
            for with_elu in [false, true] {
                let mut got = xs.clone();
                map_in_place(&mut got, Some(&affine), with_elu);
                let mut copy = vec![0.0; xs.len()];
                map_into(&xs, &mut copy, Some(&affine), with_elu);
                assert_eq!(got, copy);
                for (index, (&x, &y)) in xs.iter().zip(&got).enumerate() {
                    let v = x.mul_add(scale[index % channels], offset[index % channels]);
                    let want = if with_elu && v <= 0.0 { v.exp_m1() } else { v };
                    assert!(
                        (y - want).abs() <= 1e-6,
                        "{channels} channels: {y} vs {want}"
                    );
                }
                // The sum of two transformed operands, here twice the same.
                let mut twice = vec![0.0; xs.len()];
                sum_into((&xs, &affine), (&xs, &affine), &mut twice, false);
                let mut plain = xs.clone();
                map_in_place(&mut plain, Some(&affine), false);
                for (&sum, &v) in twice.iter().zip(&plain) {
                    assert!(2.0_f32.mul_add(-v, sum).abs() <= 1e-6);
                }
                let identity = Affine::identity(channels);
                map_into(&xs, &mut copy, Some(&identity), false);
                assert_eq!(copy, xs);
            }
        }
    }
}
