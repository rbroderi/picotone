//! Time-major tensors and the neural-network layers EnCodec needs: padded
//! 1-D convolutions, transposed convolutions, group normalisation, ELU and LSTM.
//!
//! All convolutions are lowered to a matrix product over tiles of output
//! frames, against weights packed once when the model is built. With unit
//! dilation the `im2col` matrix is a strided, overlapping view of the input, so
//! nothing is copied. The LSTM recurrence runs over a batch of sequences, one
//! matrix product per time step for the whole batch.

use std::borrow::Cow;
use std::cmp::Reverse;

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;

use crate::config::PadMode;
use crate::kernels::{self, Affine, PackedMat, gemm_acc};
use crate::{pool, prof};

/// Most output frames computed per tile: the frames a tile reads stay in the
/// second-level cache while every panel of the weights passes over them.
const MAX_TILE_ROWS: usize = 256;
/// Output frames whose statistics are summed together, and the unit tiles
/// are made of. Statistics are added block by block in order, so they do not
/// depend on how the frames are cut into tiles.
const BLOCK_ROWS: usize = 16;
/// Fewest sequences an LSTM worker advances together. A time step reads all
/// the recurrent weights whatever the batch, so a few sequences more cost
/// little.
const MIN_LSTM_GROUP: usize = 4;
/// Values handled per task in element-wise passes.
const ELEMENT_CHUNK: usize = 16 * 1024;
/// Rows of LSTM input projected per GEMM block (time steps times batch).
const LSTM_PROJECTION_ROWS: usize = 512;

thread_local! {
    /// The buffer a worker thread computes the input frames of a tile into.
    static TILE_INPUT: std::cell::Cell<Vec<f32>> = const { std::cell::Cell::new(Vec::new()) };
}

/// Output frames per tile so `rows` spread over the worker threads.
fn tile_rows(rows: usize) -> usize {
    rows.div_ceil(rayon::current_num_threads().max(1))
        .next_multiple_of(BLOCK_ROWS)
        .clamp(BLOCK_ROWS, MAX_TILE_ROWS)
}

/// A sequence of `len` frames with `channels` values each, stored time-major:
/// value `c` of frame `t` lives at `data[t * channels + c]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Seq {
    len: usize,
    channels: usize,
    data: Vec<f32>,
}

impl Seq {
    /// A zero-filled sequence.
    #[must_use]
    pub fn zeros(len: usize, channels: usize) -> Self {
        Self {
            len,
            channels,
            data: vec![0.0; len * channels],
        }
    }

    /// A sequence whose values are unspecified until written: for outputs
    /// that are filled completely.
    pub(crate) fn scratch(len: usize, channels: usize) -> Self {
        Self {
            len,
            channels,
            data: pool::take(len * channels),
        }
    }

    /// Wraps time-major data.
    ///
    /// # Errors
    /// Fails when `data.len() != len * channels`.
    pub fn from_vec(len: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        ensure!(
            Some(data.len()) == len.checked_mul(channels),
            "sequence data has {} values, expected {len} x {channels}",
            data.len()
        );
        Ok(Self {
            len,
            channels,
            data,
        })
    }

    /// Joins sequences end to end.
    ///
    /// # Errors
    /// Fails when `parts` is empty or the channel counts differ.
    pub fn concat(parts: Vec<Self>) -> Result<Self> {
        let total: usize = parts.iter().map(|part| part.data.len()).sum();
        let mut parts = parts.into_iter();
        let Some(mut out) = parts.next() else {
            bail!("cannot concatenate zero sequences");
        };
        out.data.reserve_exact(total - out.data.len());
        for part in parts {
            ensure!(
                part.channels == out.channels,
                "cannot concatenate {} and {} channels",
                out.channels,
                part.channels
            );
            out.len += part.len;
            out.data.extend_from_slice(&part.data);
        }
        Ok(out)
    }

    /// Number of frames.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence has no frames.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Values per frame.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Time-major values.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable time-major values.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consumes the sequence, returning its time-major values.
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Hands the sequence's buffer back for the next layer to write into.
    pub(crate) fn recycle(self) {
        pool::give(self.data);
    }

    /// The values of frame `t`.
    #[must_use]
    pub fn row(&self, t: usize) -> &[f32] {
        &self.data[t * self.channels..(t + 1) * self.channels]
    }

    /// Frames `start..end` as a new sequence.
    #[must_use]
    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        Self {
            len: end - start,
            channels: self.channels,
            data: self.data[start * self.channels..end * self.channels].to_vec(),
        }
    }

    /// Drops `left` frames from the start and `right` frames from the end, in place.
    pub fn trim_rows(&mut self, left: usize, right: usize) {
        let keep = self.len.saturating_sub(left + right);
        let _prof = prof::scope("trim", self.channels, 0, self.data.len());
        self.data.truncate((left + keep) * self.channels);
        drop(self.data.drain(..left.min(self.len) * self.channels));
        self.len = keep;
    }

    /// Applies ELU (alpha = 1) in place.
    #[must_use]
    pub fn elu(mut self) -> Self {
        let _prof = prof::scope("elu", self.channels, 0, self.data.len());
        map_chunks(&mut self.data, None, true);
        self
    }

    /// Adds `other` element-wise.
    ///
    /// # Errors
    /// Fails when the shapes differ.
    pub fn add_assign(&mut self, other: &Self) -> Result<()> {
        ensure!(
            self.len == other.len && self.channels == other.channels,
            "cannot add {}x{} to {}x{}",
            other.len,
            other.channels,
            self.len,
            self.channels
        );
        let _prof = prof::scope("add", self.channels, 0, self.data.len());
        self.data
            .par_chunks_mut(ELEMENT_CHUNK)
            .zip(other.data.par_chunks(ELEMENT_CHUNK))
            .for_each(|(a, b)| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            });
        Ok(())
    }

    /// Pads `left` and `right` frames using `mode`, following PyTorch / EnCodec
    /// semantics (reflect padding of very short inputs pads with zeros first).
    /// Borrows the sequence when there is nothing to pad.
    #[must_use]
    pub fn pad(&self, left: usize, right: usize, mode: PadMode) -> Cow<'_, Self> {
        if left == 0 && right == 0 {
            return Cow::Borrowed(self);
        }
        let c = self.channels;
        Cow::Owned(match mode {
            PadMode::Constant => {
                let mut out = Self::zeros(self.len + left + right, c);
                out.data[left * c..(left + self.len) * c].copy_from_slice(&self.data);
                out
            }
            PadMode::Replicate => {
                let mut out = Self::zeros(self.len + left + right, c);
                for t in 0..out.len {
                    let src = t.saturating_sub(left).min(self.len.saturating_sub(1));
                    if self.len > 0 {
                        out.data[t * c..(t + 1) * c].copy_from_slice(self.row(src));
                    }
                }
                out
            }
            PadMode::Reflect => {
                let max_pad = left.max(right);
                let extra = if self.len <= max_pad {
                    max_pad - self.len + 1
                } else {
                    0
                };
                let base = if extra > 0 {
                    self.pad(0, extra, PadMode::Constant)
                } else {
                    Cow::Borrowed(self)
                };
                let n = base.len;
                let total = n + left + right - extra;
                let mut out = Self::zeros(total, c);
                // The middle is one straight copy; only the edges mirror.
                let body = n.min(total - left);
                out.data[left * c..(left + body) * c].copy_from_slice(&base.data[..body * c]);
                for t in (0..left).chain(left + body..total) {
                    let src = if t < left {
                        left - t
                    } else {
                        2 * (n - 1) - (t - left)
                    };
                    out.data[t * c..(t + 1) * c].copy_from_slice(base.row(src));
                }
                out
            }
        })
    }
}

/// Row-major `a[m, k] · b[k, n]` into `out[m, n]`, on the calling thread.
///
/// # Errors
/// Fails when slice sizes do not match the dimensions.
pub fn matmul_into(
    a: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
    out: &mut [f32],
) -> Result<()> {
    ensure!(
        a.len() == m * k && b.len() == k * n && out.len() == m * n,
        "matmul size mismatch"
    );
    out.fill(0.0);
    gemm_acc(a, k, b, (m, k, n), out)
}

/// Row-major `a[m, k] · b[k, n]`, computed in parallel tiles of rows.
///
/// # Errors
/// Fails when slice sizes do not match the dimensions.
pub fn matmul(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize)) -> Result<Vec<f32>> {
    ensure!(a.len() == m * k && b.len() == k * n, "matmul size mismatch");
    let mut out = vec![0.0; m * n];
    if n == 0 || k == 0 {
        return Ok(out);
    }
    let tile = tile_rows(m);
    out.par_chunks_mut(tile * n)
        .zip(a.par_chunks(tile * k))
        .try_for_each(|(out_tile, a_tile)| {
            gemm_acc(a_tile, k, b, (out_tile.len() / n, k, n), out_tile)
        })?;
    Ok(out)
}

/// A dense strided convolution over an already padded time-major input,
/// computed as `im2col` tiles times a `[kernel * in, out]` weight matrix.
#[derive(Debug, Clone)]
pub struct ConvKernel {
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    /// Row `j * in_channels + i`, column `o`.
    weight: PackedMat,
    bias: Vec<f32>,
}

impl ConvKernel {
    /// Builds a kernel from a PyTorch `[out, in, kernel]` weight.
    ///
    /// # Errors
    /// Fails when the weight or bias size is inconsistent.
    pub fn from_torch(
        weight: &[f32],
        bias: &[f32],
        (out_channels, in_channels, kernel): (usize, usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        ensure!(
            weight.len() == out_channels * in_channels * kernel,
            "conv weight has {} values, expected {out_channels}x{in_channels}x{kernel}",
            weight.len()
        );
        ensure!(bias.len() == out_channels, "conv bias size mismatch");
        // Row (tap, input), column output, read straight from the weight.
        let matrix = PackedMat::from_fn(kernel * in_channels, out_channels, |r, o| {
            let (j, i) = (r / in_channels, r % in_channels);
            weight[(o * in_channels + i) * kernel + j]
        });
        Self::from_packed(
            matrix,
            bias.to_vec(),
            (out_channels, in_channels, kernel),
            stride,
            dilation,
        )
    }

    /// Builds a kernel from a weight already in `[(j, i), o]` matrix layout.
    ///
    /// # Errors
    /// Fails when sizes are inconsistent or stride/dilation are zero.
    pub fn from_matrix(
        weight: Vec<f32>,
        bias: Vec<f32>,
        (out_channels, in_channels, kernel): (usize, usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        ensure!(
            weight.len() == kernel * in_channels * out_channels,
            "conv matrix size mismatch"
        );
        let weight = PackedMat::pack(weight, kernel * in_channels, out_channels)?;
        Self::from_packed(
            weight,
            bias,
            (out_channels, in_channels, kernel),
            stride,
            dilation,
        )
    }

    /// Builds a kernel from its packed `[(j, i), o]` matrix.
    fn from_packed(
        weight: PackedMat,
        bias: Vec<f32>,
        (out_channels, in_channels, kernel): (usize, usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        ensure!(
            stride > 0 && dilation > 0 && kernel > 0,
            "invalid conv geometry"
        );
        ensure!(
            weight.shape() == (kernel * in_channels, out_channels) && bias.len() == out_channels,
            "conv matrix size mismatch"
        );
        Ok(Self {
            in_channels,
            out_channels,
            kernel,
            stride,
            dilation,
            weight,
            bias,
        })
    }

    /// Fuses two 1x1 convolutions that are summed, `a(x) + b(y)`, into one
    /// 1x1 convolution over the concatenated frame `[x | y]`.
    ///
    /// # Errors
    /// Fails unless both kernels are 1x1 with the same output width.
    pub fn fuse_sum(a: &Self, b: &Self) -> Result<Self> {
        ensure!(
            a.kernel == 1 && b.kernel == 1 && a.stride == 1 && b.stride == 1,
            "only 1x1 convolutions can be fused"
        );
        ensure!(
            a.out_channels == b.out_channels,
            "fused convolutions differ in output width"
        );
        let mut weight = a.weight.unpack();
        weight.extend_from_slice(&b.weight.unpack());
        let bias = a.bias.iter().zip(&b.bias).map(|(x, y)| x + y).collect();
        Self::from_matrix(
            weight,
            bias,
            (a.out_channels, a.in_channels + b.in_channels, 1),
            1,
            1,
        )
    }

    /// Span of input frames one output frame reads.
    #[must_use]
    pub const fn effective_kernel(&self) -> usize {
        (self.kernel - 1) * self.dilation + 1
    }

    /// Convolution stride.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Input channels.
    #[must_use]
    pub const fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Applies a 1x1 kernel to the frames in `x`, writing `out`, on the
    /// calling thread only.
    ///
    /// # Errors
    /// Fails when the kernel is not 1x1 or the slices do not match it.
    pub fn apply_frames(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let (cin, cout) = (self.in_channels, self.out_channels);
        ensure!(
            self.kernel == 1 && self.stride == 1,
            "apply_frames needs a 1x1 kernel"
        );
        ensure!(
            x.len().is_multiple_of(cin) && out.len() == x.len() / cin * cout,
            "apply_frames size mismatch"
        );
        for (x_tile, out_tile) in x
            .chunks(MAX_TILE_ROWS * cin)
            .zip(out.chunks_mut(MAX_TILE_ROWS * cout))
        {
            self.weight.mul(x_tile, cin, Some(&self.bias), out_tile)?;
        }
        Ok(())
    }

    /// Applies the convolution with no implicit padding.
    ///
    /// # Errors
    /// Fails when the input is shorter than the kernel or has the wrong width.
    pub fn apply(&self, x: &Seq) -> Result<Seq> {
        self.apply_input(&Input::plain(x))
    }

    /// [`ConvKernel::apply`] for an input that is computed as it is read.
    pub(crate) fn apply_input(&self, x: &Input<'_>) -> Result<Seq> {
        Ok(self.apply_tiles(&Padded::none(*x), TileOp::None)?.0)
    }

    /// Applies the convolution to a virtually padded input and, while each
    /// tile of output is still in cache, the extra `op`. Returns the output
    /// and the sum and sum of squares of its values (zero unless `op` is
    /// [`TileOp::Moments`]).
    ///
    /// The result does not depend on the number of threads: every output
    /// value is computed the same way wherever its tile starts, and the
    /// statistics are summed over fixed blocks of frames, in order.
    fn apply_tiles(&self, input: &Padded<'_>, op: TileOp) -> Result<(Seq, (f64, f64))> {
        let cin = self.in_channels;
        let cout = self.out_channels;
        ensure!(
            input.x.channels() == cin,
            "conv expects {cin} channels, got {}",
            input.x.channels()
        );
        let span = self.effective_kernel();
        let in_len = input.len();
        if in_len < span {
            bail!("conv input has {in_len} frames, kernel needs {span}");
        }
        let out_len = (in_len - span) / self.stride + 1;
        let kc = self.kernel * cin;
        let tile = tile_rows(out_len);
        let mut out = pool::take(out_len * cout);
        let per_tile = out
            .par_chunks_mut(tile * cout)
            .enumerate()
            .map(|(index, out_tile)| -> Result<Vec<(f64, f64)>> {
                let rows = out_tile.len() / cout;
                let first = index * tile;
                let bias = Some(self.bias.as_slice());
                // Input frames this tile reads, in padded coordinates.
                let start = first * self.stride;
                let end = (first + rows - 1) * self.stride + span;
                // The frames are read in place when they are stored as they
                // are needed; otherwise they are computed into a buffer small
                // enough to stay in cache until the product has read it.
                let mut computed = Vec::new();
                let frames = if let Some(direct) = input.interior(start, end) {
                    direct
                } else {
                    let fill = prof::scope("tile-input", cin, 0, (end - start) * cin);
                    computed = TILE_INPUT.take();
                    computed.resize((end - start) * cin, 0.0);
                    input.fill(start, end, &mut computed);
                    fill.stop();
                    &computed
                };
                if self.dilation == 1 {
                    // Each im2col row is a contiguous window of the frames.
                    let _prof = prof::scope("gemm", kc, cout, rows * kc * cout);
                    self.weight.mul(frames, self.stride * cin, bias, out_tile)?;
                } else {
                    let im2col = prof::scope("im2col", kc, cout, rows * kc);
                    let mut col = vec![0.0; rows * kc];
                    for (r, col_row) in col.chunks_exact_mut(kc).enumerate() {
                        for (j, dst) in col_row.chunks_exact_mut(cin).enumerate() {
                            let frame = r * self.stride + j * self.dilation;
                            dst.copy_from_slice(&frames[frame * cin..(frame + 1) * cin]);
                        }
                    }
                    im2col.stop();
                    let _prof = prof::scope("gemm", kc, cout, rows * kc * cout);
                    self.weight.mul(&col, kc, bias, out_tile)?;
                }
                if computed.capacity() > 0 {
                    TILE_INPUT.set(computed);
                }
                Ok(match op {
                    TileOp::None => Vec::new(),
                    TileOp::Elu => {
                        let _prof = prof::scope("elu-tile", cout, 0, out_tile.len());
                        kernels::map_in_place(out_tile, None, true);
                        Vec::new()
                    }
                    TileOp::Moments => {
                        let _prof = prof::scope("moments", cout, 0, out_tile.len());
                        out_tile
                            .chunks(BLOCK_ROWS * cout)
                            .map(kernels::moments)
                            .collect()
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let total = per_tile
            .iter()
            .flatten()
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        Ok((Seq::from_vec(out_len, cout, out)?, total))
    }
}

/// Part of a layer input: the frames of `seq` from `skip` on, each value
/// still to go through the per-channel `affine` map, if any.
#[derive(Debug, Clone, Copy)]
struct Operand<'a> {
    seq: &'a Seq,
    skip: usize,
    affine: Option<&'a Affine>,
}

impl<'a> Operand<'a> {
    /// The stored values of frames `start..end`.
    fn values(&self, start: usize, end: usize) -> &'a [f32] {
        let c = self.seq.channels;
        &self.seq.data[(self.skip + start) * c..(self.skip + end) * c]
    }
}

/// How the second operand of an [`Input`] joins the first.
#[derive(Debug, Clone, Copy)]
enum Join<'a> {
    /// There is no second operand.
    Alone,
    /// The operands are added, after their affine maps.
    Sum(Operand<'a>),
    /// The stored frames of the operands sit side by side.
    Concat(Operand<'a>),
}

/// The input of a layer, computed from stored sequences while the layer
/// reads it.
///
/// A normalisation, the sum of a residual block and an activation each
/// cost a pass over a signal far larger than the caches when done on their
/// own. Left pending and applied to the few frames a tile of the next
/// convolution is about to read, they cost almost nothing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input<'a> {
    first: Operand<'a>,
    join: Join<'a>,
    /// Whether ELU is applied last.
    elu: bool,
    len: usize,
}

impl<'a> Input<'a> {
    /// A stored sequence as it is.
    pub(crate) const fn plain(seq: &'a Seq) -> Self {
        Self {
            first: Operand {
                seq,
                skip: 0,
                affine: None,
            },
            join: Join::Alone,
            elu: false,
            len: seq.len,
        }
    }

    /// The frames of `a` and of `b` side by side. `b` must be stored as it
    /// is, apart from skipped frames.
    ///
    /// # Errors
    /// Fails when `b` has pending work or another length.
    pub(crate) fn concat(a: &'a Seq, b: &Self) -> Result<Self> {
        ensure!(
            b.is_stored() && b.len == a.len,
            "cannot join {} and {} frames side by side",
            a.len,
            b.len
        );
        Ok(Self {
            join: Join::Concat(b.first),
            ..Self::plain(a)
        })
    }

    /// The same input followed by ELU.
    ///
    /// # Errors
    /// Fails when an ELU is pending already.
    pub(crate) fn with_elu(mut self) -> Result<Self> {
        ensure!(
            !self.elu && !matches!(self.join, Join::Concat(_)),
            "input cannot take another activation"
        );
        self.elu = true;
        Ok(self)
    }

    /// Whether the values are stored exactly as they are read.
    const fn is_stored(&self) -> bool {
        matches!(self.join, Join::Alone) && self.first.affine.is_none() && !self.elu
    }

    const fn len(&self) -> usize {
        self.len
    }

    const fn channels(&self) -> usize {
        match self.join {
            Join::Alone | Join::Sum(_) => self.first.seq.channels,
            Join::Concat(other) => self.first.seq.channels + other.seq.channels,
        }
    }

    /// Frames `start..end` when they are stored as they are read.
    fn direct(&self, start: usize, end: usize) -> Option<&'a [f32]> {
        self.is_stored().then(|| self.first.values(start, end))
    }

    /// Computes frames `start..end` into `dst`.
    fn fill(&self, start: usize, end: usize, dst: &mut [f32]) {
        let a = self.first.values(start, end);
        match self.join {
            Join::Alone => kernels::map_into(a, dst, self.first.affine, self.elu),
            Join::Sum(other) => {
                let b = other.values(start, end);
                if let (Some(a_affine), Some(b_affine)) = (self.first.affine, other.affine) {
                    kernels::sum_into((a, a_affine), (b, b_affine), dst, self.elu);
                } else {
                    let identity = Affine::identity(self.first.seq.channels);
                    kernels::sum_into(
                        (a, self.first.affine.unwrap_or(&identity)),
                        (b, other.affine.unwrap_or(&identity)),
                        dst,
                        self.elu,
                    );
                }
            }
            Join::Concat(other) => {
                let (ca, cb) = (self.first.seq.channels, other.seq.channels);
                let b = other.values(start, end);
                for ((row, a), b) in dst
                    .chunks_exact_mut(ca + cb)
                    .zip(a.chunks_exact(ca))
                    .zip(b.chunks_exact(cb))
                {
                    let (left, right) = row.split_at_mut(ca);
                    left.copy_from_slice(a);
                    right.copy_from_slice(b);
                }
            }
        }
    }

    /// The whole input as a stored sequence.
    fn to_seq(self) -> Seq {
        let c = self.channels();
        let mut out = Seq::scratch(self.len, c);
        let rows = (ELEMENT_CHUNK / c.max(1)).max(1);
        let _prof = prof::scope("settle", c, 0, out.data.len());
        out.data
            .par_chunks_mut((rows * c).max(1))
            .enumerate()
            .for_each(|(index, chunk)| {
                let start = index * rows;
                self.fill(start, start + chunk.len() / c.max(1), chunk);
            });
        out
    }
}

/// One stored sequence of a [`Lazy`] signal with what is pending on it.
#[derive(Debug)]
struct Part {
    seq: Seq,
    /// Leading frames that are not part of the signal.
    skip: usize,
    affine: Option<Affine>,
}

impl Part {
    const fn operand(&self) -> Operand<'_> {
        Operand {
            seq: &self.seq,
            skip: self.skip,
            affine: self.affine.as_ref(),
        }
    }
}

/// The output of a layer with the cheap work that follows it left pending:
/// trimming, the normalisation's scale and offset, the sum with a residual
/// branch and ELU. The next layer does that work as it reads its
/// [`Input`]; [`Lazy::settle`] does it for anything else.
#[derive(Debug)]
pub(crate) struct Lazy {
    first: Part,
    /// A second signal to add.
    second: Option<Part>,
    elu: bool,
    len: usize,
}

impl Lazy {
    /// A finished sequence.
    pub(crate) const fn new(seq: Seq) -> Self {
        Self {
            len: seq.len,
            first: Part {
                seq,
                skip: 0,
                affine: None,
            },
            second: None,
            elu: false,
        }
    }

    /// A sequence whose normalisation is pending.
    const fn normalised(seq: Seq, affine: Affine) -> Self {
        Self {
            len: seq.len,
            first: Part {
                seq,
                skip: 0,
                affine: Some(affine),
            },
            second: None,
            elu: false,
        }
    }

    /// The signal as a layer input.
    pub(crate) const fn input(&self) -> Input<'_> {
        Input {
            first: self.first.operand(),
            join: match &self.second {
                Some(second) => Join::Sum(second.operand()),
                None => Join::Alone,
            },
            elu: self.elu,
            len: self.len,
        }
    }

    /// Whether the signal is stored as it is read, apart from skipped frames.
    pub(crate) const fn is_stored(&self) -> bool {
        self.input().is_stored()
    }

    /// Whether an ELU is pending.
    pub(crate) const fn has_elu(&self) -> bool {
        self.elu
    }

    /// The signal followed by ELU.
    pub(crate) fn then_elu(self) -> Self {
        let mut out = if self.elu {
            Self::new(self.settle())
        } else {
            self
        };
        out.elu = true;
        out
    }

    /// The sum of two signals.
    ///
    /// # Errors
    /// Fails when the shapes differ.
    pub(crate) fn plus(self, other: Self) -> Result<Self> {
        ensure!(
            self.len == other.len && self.first.seq.channels == other.first.seq.channels,
            "cannot add {}x{} to {}x{}",
            other.len,
            other.first.seq.channels,
            self.len,
            self.first.seq.channels
        );
        let single = |lazy: Self| {
            let lazy = if lazy.second.is_some() || lazy.elu {
                Self::new(lazy.settle())
            } else {
                lazy
            };
            let channels = lazy.first.seq.channels;
            Part {
                affine: Some(
                    lazy.first
                        .affine
                        .unwrap_or_else(|| Affine::identity(channels)),
                ),
                ..lazy.first
            }
        };
        Ok(Self {
            len: self.len,
            first: single(self),
            second: Some(single(other)),
            elu: false,
        })
    }

    /// Does the pending work and returns the stored result.
    pub(crate) fn settle(self) -> Seq {
        let whole = self.first.skip == 0 && self.first.seq.len == self.len;
        if self.second.is_none() && whole {
            let Part {
                mut seq, affine, ..
            } = self.first;
            if affine.is_some() || self.elu {
                let _prof = prof::scope("settle", seq.channels, 1, seq.data.len());
                map_chunks(&mut seq.data, affine.as_ref(), self.elu);
            }
            return seq;
        }
        let out = self.input().to_seq();
        self.recycle();
        out
    }

    /// Hands the buffers back for the next layer to write into.
    pub(crate) fn recycle(self) {
        self.first.seq.recycle();
        if let Some(second) = self.second {
            second.seq.recycle();
        }
    }
}

/// An input with `left` and `right` frames of padding that are never
/// materialised for the whole signal: the convolution reads the interior
/// directly where it can and builds the padded frames only at the two ends.
#[derive(Debug, Clone, Copy)]
struct Padded<'a> {
    x: Input<'a>,
    left: usize,
    right: usize,
    mode: PadMode,
}

impl<'a> Padded<'a> {
    /// `x` without padding.
    const fn none(x: Input<'a>) -> Self {
        Self {
            x,
            left: 0,
            right: 0,
            mode: PadMode::Constant,
        }
    }

    /// Whether padding `len` frames as [`Seq::pad`] would can be expressed
    /// frame by frame. It cannot for an input too short for the mode:
    /// reflection of a very short input pads with zeros first.
    const fn expressible(len: usize, left: usize, right: usize, mode: PadMode) -> bool {
        let max_pad = if left > right { left } else { right };
        match mode {
            PadMode::Constant => true,
            PadMode::Replicate => len > 0,
            PadMode::Reflect => len > max_pad,
        }
    }

    const fn len(&self) -> usize {
        self.left + self.x.len() + self.right
    }

    /// Padded frames `start..end` when they all lie inside the real signal
    /// and are stored as they are read.
    fn interior(&self, start: usize, end: usize) -> Option<&'a [f32]> {
        if start >= self.left && end <= self.left + self.x.len() {
            self.x.direct(start - self.left, end - self.left)
        } else {
            None
        }
    }

    /// Computes padded frames `start..end` into `dst`.
    fn fill(&self, start: usize, end: usize, dst: &mut [f32]) {
        let c = self.x.channels();
        let n = self.x.len();
        let mut t = start;
        while t < end {
            if t >= self.left && t - self.left < n {
                // The real signal, as one run.
                let run_end = end.min(self.left + n);
                self.x.fill(
                    t - self.left,
                    run_end - self.left,
                    &mut dst[(t - start) * c..(run_end - start) * c],
                );
                t = run_end;
                continue;
            }
            let source = if t < self.left {
                match self.mode {
                    PadMode::Constant => None,
                    PadMode::Replicate => Some(0),
                    PadMode::Reflect => Some(self.left - t),
                }
            } else {
                let past = t - self.left - n;
                match self.mode {
                    PadMode::Constant => None,
                    PadMode::Replicate => Some(n - 1),
                    PadMode::Reflect => Some(n - 2 - past),
                }
            };
            let row = &mut dst[(t - start) * c..(t - start + 1) * c];
            match source {
                Some(source) => self.x.fill(source, source + 1, row),
                None => row.fill(0.0),
            }
            t += 1;
        }
    }
}

/// Extra work done on each tile of convolution output while it is in cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileOp {
    /// Nothing.
    None,
    /// Apply ELU.
    Elu,
    /// Accumulate the sum and sum of squares for a following `GroupNorm`.
    Moments,
}

/// `GroupNorm` with a single group: normalises over every frame and channel,
/// then applies a per-channel affine transform.
#[derive(Debug, Clone)]
pub struct GroupNorm1 {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f64,
}

impl GroupNorm1 {
    /// Creates the layer from its affine parameters.
    ///
    /// # Errors
    /// Fails when `weight` and `bias` differ in size.
    pub fn new(weight: Vec<f32>, bias: Vec<f32>) -> Result<Self> {
        ensure!(
            weight.len() == bias.len(),
            "group norm parameter size mismatch"
        );
        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    /// Normalises `x` in place.
    ///
    /// # Errors
    /// Fails when the channel count does not match.
    pub fn apply(&self, x: &mut Seq) -> Result<()> {
        ensure!(
            x.channels == self.weight.len(),
            "group norm channel mismatch"
        );
        let affine = self.affine(x.data.len(), chunked_moments(&x.data));
        let _prof = prof::scope("gn-affine", x.channels, 0, x.data.len());
        map_chunks(&mut x.data, Some(&affine), false);
        Ok(())
    }

    /// The scale and offset per channel that normalise `count` values with
    /// the given sum and sum of squares.
    ///
    /// The statistics and the affine terms are folded into one scale and one
    /// offset per channel, in double precision, so normalisation and affine
    /// transform are one multiply-add per value.
    #[expect(
        clippy::suspicious_operation_groupings,
        reason = "variance is E[x^2] - mean^2"
    )]
    fn affine(&self, count: usize, (sum, sum_sq): (f64, f64)) -> Affine {
        let n = crate::config::to_f64(count.max(1));
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        let inv = 1.0 / (var + self.eps).sqrt();
        Affine::new(self.weight.iter().zip(&self.bias).map(|(&g, &b)| {
            let scale = inv * f64::from(g);
            (narrow(scale), narrow(f64::from(b) - mean * scale))
        }))
    }
}

/// `values = elu?(values * scale + offset)` in parallel chunks of whole
/// frames; the affine part and the ELU are both optional.
fn map_chunks(values: &mut [f32], affine: Option<&Affine>, elu: bool) {
    if affine.is_none() && !elu {
        return;
    }
    // Whole periods of the scale pattern per task.
    let period = affine.map_or(1, Affine::period);
    let chunk = (ELEMENT_CHUNK / period).max(1) * period;
    values
        .par_chunks_mut(chunk)
        .for_each(|values| kernels::map_in_place(values, affine, elu));
}

/// Sum and sum of squares of `values`, summed in fixed chunks.
fn chunked_moments(values: &[f32]) -> (f64, f64) {
    let _prof = prof::scope("moments", 0, 1, values.len());
    let per_chunk: Vec<(f64, f64)> = values
        .par_chunks(ELEMENT_CHUNK)
        .map(kernels::moments)
        .collect();
    per_chunk
        .iter()
        .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "normalisation factors are well inside f32 range"
)]
const fn narrow(v: f64) -> f32 {
    v as f32
}

/// Shape of a convolution layer.
#[derive(Debug, Clone, Copy)]
pub struct ConvGeometry {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size.
    pub kernel: usize,
    /// Stride.
    pub stride: usize,
    /// Dilation.
    pub dilation: usize,
}

/// EnCodec's padded 1-D convolution (`SConv1d`).
#[derive(Debug, Clone)]
pub struct Conv1d {
    kernel: ConvKernel,
    causal: bool,
    pad_mode: PadMode,
    norm: Option<GroupNorm1>,
}

impl Conv1d {
    /// Creates the layer.
    #[must_use]
    pub const fn new(
        kernel: ConvKernel,
        causal: bool,
        pad_mode: PadMode,
        norm: Option<GroupNorm1>,
    ) -> Self {
        Self {
            kernel,
            causal,
            pad_mode,
            norm,
        }
    }

    /// The convolution without its padding and normalisation.
    #[must_use]
    pub const fn kernel(&self) -> &ConvKernel {
        &self.kernel
    }

    /// `(effective kernel, stride)` when an output frame depends only on
    /// current and earlier input frames and on nothing global, which is what
    /// lets a long signal be processed in chunks.
    #[must_use]
    pub const fn causal_window(&self) -> Option<(usize, usize)> {
        if self.causal && self.norm.is_none() {
            Some((self.kernel.effective_kernel(), self.kernel.stride()))
        } else {
            None
        }
    }

    /// Applies padding, convolution, the optional normalisation and, when
    /// `elu` is set, the ELU that follows the layer.
    ///
    /// # Errors
    /// Fails on shape mismatches.
    pub fn forward(&self, x: &Seq, elu: bool) -> Result<Seq> {
        Ok(self.forward_lazy(&Input::plain(x), elu)?.settle())
    }

    /// [`Conv1d::forward`] for an input that is computed as it is read,
    /// leaving the normalisation and the ELU after it pending.
    pub(crate) fn forward_lazy(&self, x: &Input<'_>, elu: bool) -> Result<Lazy> {
        let k = self.kernel.effective_kernel();
        let s = self.kernel.stride();
        let padding_total = k - s;
        let extra = extra_padding(x.len(), k, s, padding_total);
        let (left, right) = if self.causal {
            (padding_total, extra)
        } else {
            let right = padding_total / 2;
            (padding_total - right, right + extra)
        };
        // Very short inputs get a real padded copy; everything else is
        // padded virtually.
        let materialised = (!Padded::expressible(x.len(), left, right, self.pad_mode))
            .then(|| x.to_seq().pad(left, right, self.pad_mode).into_owned());
        let padded = materialised.as_ref().map_or(
            Padded {
                x: *x,
                left,
                right,
                mode: self.pad_mode,
            },
            |seq| Padded::none(Input::plain(seq)),
        );
        if let Some(norm) = &self.norm {
            ensure!(
                self.kernel.out_channels == norm.weight.len(),
                "group norm channel mismatch"
            );
            let (seq, moments) = self.kernel.apply_tiles(&padded, TileOp::Moments)?;
            let affine = norm.affine(seq.data.len(), moments);
            Ok(Lazy {
                elu,
                ..Lazy::normalised(seq, affine)
            })
        } else {
            let op = if elu { TileOp::Elu } else { TileOp::None };
            Ok(Lazy::new(self.kernel.apply_tiles(&padded, op)?.0))
        }
    }
}

/// Extra right padding so the last window is complete
/// (`get_extra_padding_for_conv1d` in EnCodec).
fn extra_padding(len: usize, kernel: usize, stride: usize, padding_total: usize) -> usize {
    // n_frames = ceil((len - kernel + padding_total) / stride) + 1, in exact integers.
    let num = (len + padding_total).cast_signed() - kernel.cast_signed();
    let frames = num.div_euclid(stride.cast_signed())
        + isize::from(num.rem_euclid(stride.cast_signed()) != 0)
        + 1;
    let ideal = (frames - 1) * stride.cast_signed() + (kernel - padding_total).cast_signed();
    (ideal - len.cast_signed()).max(0).cast_unsigned()
}

/// EnCodec's transposed convolution (`SConvTranspose1d`), lowered to a
/// polyphase convolution: each input window produces `stride` output frames.
#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    poly: ConvKernel,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    taps: usize,
    /// Frames trimmed from the start and the end of the full output.
    trim: (usize, usize),
    causal: bool,
    norm: Option<GroupNorm1>,
}

impl ConvTranspose1d {
    /// Builds the layer from a PyTorch `[in, out, kernel]` weight.
    ///
    /// # Errors
    /// Fails on inconsistent sizes.
    pub fn new(
        weight: &[f32],
        bias: &[f32],
        geometry: ConvGeometry,
        causal: bool,
        trim_right_ratio: f64,
        norm: Option<GroupNorm1>,
    ) -> Result<Self> {
        let ConvGeometry {
            in_channels: cin,
            out_channels: cout,
            kernel: k,
            stride: s,
            ..
        } = geometry;
        ensure!(s > 0 && k >= s, "invalid transposed conv geometry");
        ensure!(
            weight.len() == cin * cout * k,
            "transposed conv weight size mismatch"
        );
        ensure!(bias.len() == cout, "transposed conv bias size mismatch");
        let taps = k.div_ceil(s);
        let rows = taps * cin;
        let cols = s * cout;
        // Row (tap, input), column (phase, output): kernel position
        // `p + (taps - 1 - tap) * s`, zero past the kernel.
        let matrix = PackedMat::from_fn(rows, cols, |r, c| {
            let (tap, i) = (r / cin, r % cin);
            let (p, o) = (c / cout, c % cout);
            let j = p + (taps - 1 - tap) * s;
            if j < k {
                weight[(i * cout + o) * k + j]
            } else {
                0.0
            }
        });
        let poly_bias = bias.repeat(s);
        let poly = ConvKernel::from_packed(matrix, poly_bias, (cols, cin, taps), 1, 1)?;
        let padding_total = k - s;
        let right = if causal {
            crate::config::floor_to_usize(
                (crate::config::to_f64(padding_total) * trim_right_ratio).ceil(),
            )?
            .min(padding_total)
        } else {
            padding_total / 2
        };
        Ok(Self {
            poly,
            out_channels: cout,
            kernel: k,
            stride: s,
            taps,
            trim: (padding_total - right, right),
            causal,
            norm,
        })
    }

    /// `(taps, stride)` when an output frame depends only on current and
    /// earlier input frames and on nothing global (see [`Conv1d::causal_window`]).
    #[must_use]
    pub const fn causal_window(&self) -> Option<(usize, usize)> {
        if self.causal && self.norm.is_none() && self.trim.0 == 0 {
            Some((self.taps, self.stride))
        } else {
            None
        }
    }

    /// Applies the transposed convolution and the optional normalisation,
    /// trims padding and, when `elu` is set, applies the ELU that follows.
    ///
    /// # Errors
    /// Fails on shape mismatches.
    pub fn forward(&self, x: &Seq, elu: bool) -> Result<Seq> {
        Ok(self.forward_lazy(&Input::plain(x), elu)?.settle())
    }

    /// [`ConvTranspose1d::forward`] for an input that is computed as it is
    /// read, leaving trimming, normalisation and the ELU after it pending.
    pub(crate) fn forward_lazy(&self, x: &Input<'_>, elu: bool) -> Result<Lazy> {
        ensure!(x.len() > 0, "transposed conv input is empty");
        let padded = Padded {
            x: *x,
            left: self.taps - 1,
            right: self.taps - 1,
            mode: PadMode::Constant,
        };
        let full_len = (x.len() - 1) * self.stride + self.kernel;
        // The statistics cover the output before trimming, as in PyTorch. The
        // phases are exactly that output unless the kernel is not a multiple
        // of the stride.
        let whole = full_len == (x.len() + self.taps - 1) * self.stride;
        let op = match (&self.norm, elu) {
            (Some(_), _) if whole => TileOp::Moments,
            (None, true) => TileOp::Elu,
            _ => TileOp::None,
        };
        let (phases, moments) = self.poly.apply_tiles(&padded, op)?;
        let mut data = phases.into_data();
        data.truncate(full_len * self.out_channels);
        let seq = Seq::from_vec(full_len, self.out_channels, data)?;
        let mut out = match &self.norm {
            Some(norm) => {
                ensure!(
                    self.out_channels == norm.weight.len(),
                    "group norm channel mismatch"
                );
                let total = if whole {
                    moments
                } else {
                    chunked_moments(&seq.data)
                };
                let affine = norm.affine(seq.data.len(), total);
                Lazy {
                    elu,
                    ..Lazy::normalised(seq, affine)
                }
            }
            None => Lazy::new(seq),
        };
        out.first.skip = self.trim.0;
        out.len = full_len.saturating_sub(self.trim.0 + self.trim.1);
        Ok(out)
    }
}

/// A multi-layer unidirectional LSTM with EnCodec's skip connection
/// (output of the last layer plus the input).
#[derive(Debug, Clone)]
pub struct Lstm {
    dim: usize,
    layers: Vec<LstmLayer>,
}

/// One LSTM layer. A time step is one matrix product of the hidden states of
/// every running sequence with the recurrent weights, added to the projected
/// inputs, followed by the gates. Gate pre-activations keep PyTorch's
/// gate-major order `[i | f | g | o]`, so the gates vectorise over the units.
#[derive(Debug, Clone)]
struct LstmLayer {
    /// Input projection `x · W_ih^T + b_ih + b_hh`.
    input: ConvKernel,
    /// Recurrent weights `W_hh^T`, `[hidden, 4 * hidden]`.
    w_hh: PackedMat,
    hidden: usize,
}

/// Parameters of one PyTorch LSTM layer.
#[derive(Debug, Clone, Copy)]
pub struct LstmParams<'a> {
    /// `weight_ih_l{n}`, `[4 * hidden, input]`.
    pub w_ih: &'a [f32],
    /// `weight_hh_l{n}`, `[4 * hidden, hidden]`.
    pub w_hh: &'a [f32],
    /// `bias_ih_l{n}`.
    pub b_ih: &'a [f32],
    /// `bias_hh_l{n}`.
    pub b_hh: &'a [f32],
}

/// The hidden and cell state of every layer of an [`Lstm`], carried between
/// calls of [`Lstm::forward_stream`].
#[derive(Debug, Clone, PartialEq)]
pub struct LstmState {
    /// `(hidden, cell)` per layer.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

/// A run of consecutive time steps for a group of sequences that advance
/// together. Row `step * running + b` of `data` belongs to time
/// `start + step` of the group's `b`-th sequence; rows of sequences that have
/// already ended are padding.
#[derive(Debug)]
struct TimeBlock {
    start: usize,
    steps: usize,
    running: usize,
    data: Vec<f32>,
}

/// Where a pipeline stage hands its finished blocks.
type BlockSink<'a> = dyn FnMut(TimeBlock) + Send + 'a;

impl Lstm {
    /// Builds the LSTM from per-layer parameters; every layer is `dim -> dim`.
    ///
    /// # Errors
    /// Fails on inconsistent sizes.
    pub fn new(dim: usize, layers: &[LstmParams<'_>]) -> Result<Self> {
        ensure!(dim > 0, "LSTM width is zero");
        let mut lstm = Self {
            dim,
            layers: Vec::with_capacity(layers.len()),
        };
        for params in layers {
            lstm.push_layer(params)?;
        }
        Ok(lstm)
    }

    /// Appends a layer on top of the existing ones.
    ///
    /// # Errors
    /// Fails on inconsistent sizes.
    pub fn push_layer(&mut self, p: &LstmParams<'_>) -> Result<()> {
        let dim = self.dim;
        ensure!(
            p.w_ih.len() == 4 * dim * dim
                && p.w_hh.len() == 4 * dim * dim
                && p.b_ih.len() == 4 * dim
                && p.b_hh.len() == 4 * dim,
            "LSTM parameter size mismatch"
        );
        let bias: Vec<f32> = p.b_ih.iter().zip(p.b_hh).map(|(a, b)| a + b).collect();
        // Row unit, column gate: the transpose of `w_hh`.
        let w_hh = PackedMat::from_fn(dim, 4 * dim, |unit, gate| p.w_hh[gate * dim + unit]);
        self.layers.push(LstmLayer {
            input: ConvKernel::from_torch(p.w_ih, &bias, (4 * dim, dim, 1), 1, 1)?,
            w_hh,
            hidden: dim,
        });
        Ok(())
    }

    /// Runs every layer over the sequence and adds the input.
    ///
    /// # Errors
    /// Fails on shape mismatches.
    pub fn forward(&self, x: &Seq) -> Result<Seq> {
        let mut out = self.forward_batch(std::slice::from_ref(x))?;
        out.pop()
            .ok_or_else(|| anyhow::anyhow!("LSTM produced no output"))
    }

    /// Runs every layer over independent sequences, which may differ in
    /// length, and adds each input to its output.
    ///
    /// The batch is split into groups, at most one per worker thread. A
    /// group's sequences advance through time together, so one pass over the
    /// recurrent weights serves all of them, and nothing is synchronised
    /// between groups. Within a group the layers form a pipeline: layer `n`
    /// starts on a block of time steps as soon as layer `n - 1` has finished
    /// it, on another thread when one is free.
    ///
    /// # Errors
    /// Fails on shape mismatches.
    pub fn forward_batch(&self, xs: &[Seq]) -> Result<Vec<Seq>> {
        ensure!(
            xs.iter().all(|x| x.channels == self.dim),
            "LSTM expects {} channels",
            self.dim
        );
        // Longest first, so each group's running sequences are a prefix of it.
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by_key(|&i| Reverse(xs[i].len));
        let group = xs
            .len()
            .div_ceil(rayon::current_num_threads().max(1))
            .max(MIN_LSTM_GROUP);
        let groups = order
            .par_chunks(group)
            .map(|members| self.forward_group(xs, members))
            .collect::<Result<Vec<_>>>()?;
        let mut outs: Vec<Option<Seq>> = xs.iter().map(|_| None).collect();
        for (&index, out) in order.iter().zip(groups.into_iter().flatten()) {
            outs[index] = Some(out);
        }
        outs.into_iter()
            .map(|out| out.ok_or_else(|| anyhow::anyhow!("LSTM lost a sequence")))
            .collect()
    }

    /// The state before any input: zeros for every layer.
    #[must_use]
    pub fn initial_state(&self) -> LstmState {
        LstmState {
            layers: self
                .layers
                .iter()
                .map(|_| (vec![0.0; self.dim], vec![0.0; self.dim]))
                .collect(),
        }
    }

    /// Runs every layer over the next frames of one sequence, continuing
    /// from `state` and updating it, and adds the input. Feeding a sequence
    /// in pieces gives the same values as [`Lstm::forward`] on the whole.
    ///
    /// # Errors
    /// Fails on shape mismatches.
    pub fn forward_stream(&self, state: &mut LstmState, x: &Seq) -> Result<Seq> {
        let d = self.dim;
        ensure!(x.channels == d, "LSTM expects {d} channels");
        ensure!(
            state.layers.len() == self.layers.len(),
            "LSTM state belongs to another network"
        );
        let len = x.len;
        let mut current = x.data.clone();
        for (layer, (h, c)) in self.layers.iter().zip(&mut state.layers) {
            let mut next = Vec::with_capacity(current.len());
            let source =
                current
                    .chunks(LSTM_PROJECTION_ROWS * d)
                    .enumerate()
                    .map(|(index, rows)| {
                        let mut data = pool::take(rows.len());
                        data.copy_from_slice(rows);
                        TimeBlock {
                            start: index * LSTM_PROJECTION_ROWS,
                            steps: rows.len() / d,
                            running: 1,
                            data,
                        }
                    });
            layer.run_from(&[len], (h, c), source, &mut |block| {
                next.extend_from_slice(&block.data);
                pool::give(block.data);
            })?;
            current = next;
        }
        for (out, input) in current.iter_mut().zip(&x.data) {
            *out += input;
        }
        Seq::from_vec(len, d, current)
    }

    /// Runs the sequences `xs[members]`, sorted longest first.
    fn forward_group(&self, xs: &[Seq], members: &[usize]) -> Result<Vec<Seq>> {
        let d = self.dim;
        let lens: Vec<usize> = members.iter().map(|&i| xs[i].len).collect();
        let max_len = lens.first().copied().unwrap_or(0);
        let block_steps = (LSTM_PROJECTION_ROWS / members.len().max(1)).max(1);
        let lens = lens.as_slice();
        let source = (0..max_len).step_by(block_steps).map(move |start| {
            let steps = block_steps.min(max_len - start);
            let running = lens.partition_point(|&l| l > start);
            let mut data = pool::take(steps * running * d);
            for (r, row) in data.chunks_exact_mut(d).enumerate() {
                let (t, b) = (start + r / running, r % running);
                if t < lens[b] {
                    row.copy_from_slice(xs[members[b]].row(t));
                } else {
                    row.fill(0.0);
                }
            }
            TimeBlock {
                start,
                steps,
                running,
                data,
            }
        });
        let mut outs: Vec<Seq> = lens.iter().map(|&len| Seq::zeros(len, d)).collect();
        let mut sink = |block: TimeBlock| {
            for (r, row) in block.data.chunks_exact(d).enumerate() {
                let (t, b) = (block.start + r / block.running, r % block.running);
                if t < lens[b] {
                    let input = xs[members[b]].row(t);
                    for ((out, h), x) in outs[b].data[t * d..(t + 1) * d]
                        .iter_mut()
                        .zip(row)
                        .zip(input)
                    {
                        *out = h + x;
                    }
                }
            }
            pool::give(block.data);
        };
        pipeline(&self.layers, lens, source, &mut sink)?;
        Ok(outs)
    }
}

/// Feeds `source` through `layers`, each layer a pipeline stage connected to
/// the next by a queue, and hands the last layer's blocks to `sink`.
///
/// A stage never waits on anything but the stage before it and never starts
/// nested parallel work, so stages cannot deadlock; on a single thread they
/// simply run one after another.
fn pipeline<I>(
    layers: &[LstmLayer],
    lens: &[usize],
    source: I,
    sink: &mut BlockSink<'_>,
) -> Result<()>
where
    I: Iterator<Item = TimeBlock> + Send,
{
    match layers {
        [] => {
            source.for_each(sink);
            Ok(())
        }
        [last] => last.run(lens, source, sink),
        [first, rest @ ..] => {
            let (sender, receiver) = std::sync::mpsc::channel();
            let (produced, consumed) = rayon::join(
                move || {
                    // A send only fails when the consumer has already failed.
                    first.run(lens, source, &mut |block| drop(sender.send(block)))
                },
                || pipeline(rest, lens, receiver.into_iter(), sink),
            );
            produced.and(consumed)
        }
    }
}

impl LstmLayer {
    /// Advances a group of sequences (lengths `lens`, longest first) through
    /// the blocks of `source`, replacing each block's inputs by the outputs.
    fn run(
        &self,
        lens: &[usize],
        source: impl Iterator<Item = TimeBlock>,
        sink: &mut BlockSink<'_>,
    ) -> Result<()> {
        let mut h = vec![0.0_f32; lens.len() * self.hidden];
        let mut c = h.clone();
        self.run_from(lens, (&mut h, &mut c), source, sink)
    }

    /// Like [`LstmLayer::run`], starting from the hidden and cell state in
    /// `state` (`[sequence][hidden]` each) and leaving the final state there.
    fn run_from(
        &self,
        lens: &[usize],
        (h, c): (&mut [f32], &mut [f32]),
        source: impl Iterator<Item = TimeBlock>,
        sink: &mut BlockSink<'_>,
    ) -> Result<()> {
        let d = self.hidden;
        ensure!(
            h.len() == lens.len() * d && c.len() == h.len(),
            "LSTM state size mismatch"
        );
        let mut h_next = vec![0.0_f32; lens.len() * d];
        let mut projected = Vec::new();
        for mut block in source {
            let rows = block.steps * block.running;
            ensure!(
                block.data.len() == rows * d && block.running <= lens.len(),
                "LSTM block size mismatch"
            );
            projected.resize(rows * 4 * d, 0.0);
            let projection = prof::scope("lstm-proj", d, 4 * d, rows * 4 * d * d);
            self.input.apply_frames(&block.data, &mut projected)?;
            projection.stop();
            let _prof = prof::scope("lstm-steps", d, 4 * d, rows * 4 * d * d);
            for step in 0..block.steps {
                let t = block.start + step;
                let active = lens[..block.running].partition_point(|&l| l > t);
                let first = step * block.running;
                let n = active * d;
                self.step(
                    &mut projected[first * 4 * d..][..active * 4 * d],
                    &h[..n],
                    &mut c[..n],
                    &mut h_next[..n],
                )?;
                h[..n].copy_from_slice(&h_next[..n]);
                block.data[first * d..][..n].copy_from_slice(&h_next[..n]);
            }
            sink(block);
        }
        Ok(())
    }

    /// One time step for `h.len() / hidden` sequences. `pre` holds their
    /// projected inputs (`[sequence][4 * hidden]`, gate-major) and is used
    /// up; `h`, `c` and `h_next` are `[sequence][hidden]`.
    fn step(&self, pre: &mut [f32], h: &[f32], c: &mut [f32], h_next: &mut [f32]) -> Result<()> {
        let d = self.hidden;
        self.w_hh.mul(h, d, None, pre)?;
        for ((gates, cell), hidden) in pre
            .chunks_exact(4 * d)
            .zip(c.chunks_exact_mut(d))
            .zip(h_next.chunks_exact_mut(d))
        {
            kernels::lstm_gates(gates, cell, hidden);
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss, reason = "small test indices")]
mod tests {
    use super::*;

    fn seq(len: usize, channels: usize, f: impl Fn(usize) -> f32) -> Seq {
        Seq::from_vec(len, channels, (0..len * channels).map(f).collect())
            .unwrap_or_else(|_| Seq::zeros(len, channels))
    }

    fn values(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32).mul_add(0.731, seed)).sin())
            .collect()
    }

    /// The layers pack their weights straight from PyTorch's layout into the
    /// same matrices, bit for bit, as the loader that went through a
    /// row-major matrix first.
    #[test]
    fn layers_pack_like_the_matrix_loader() -> Result<()> {
        // Conv: rows (tap, input), columns output.
        let (cout, cin, kernel) = (37, 19, 7);
        let (weight, bias) = (values(cout * cin * kernel, 0.1), values(cout, 0.2));
        let mut matrix = vec![0.0; kernel * cin * cout];
        for o in 0..cout {
            for i in 0..cin {
                for j in 0..kernel {
                    matrix[(j * cin + i) * cout + o] = weight[(o * cin + i) * kernel + j];
                }
            }
        }
        let old = PackedMat::pack(matrix, kernel * cin, cout)?;
        let conv = ConvKernel::from_torch(&weight, &bias, (cout, cin, kernel), 1, 3)?;
        assert!(conv.weight.bit_identical(&old), "conv");
        // Transposed conv: rows (tap, input), columns (phase, output).
        // Kernel 12, stride 8: some positions lie past the kernel.
        let (cin, cout, k, s) = (24, 12, 12, 8);
        let (weight, bias) = (values(cin * cout * k, 0.3), values(cout, 0.4));
        let taps = k.div_ceil(s);
        let cols = s * cout;
        let mut matrix = vec![0.0; taps * cin * cols];
        for tap in 0..taps {
            for p in 0..s {
                let j = p + (taps - 1 - tap) * s;
                if j >= k {
                    continue;
                }
                for i in 0..cin {
                    for o in 0..cout {
                        matrix[(tap * cin + i) * cols + p * cout + o] =
                            weight[(i * cout + o) * k + j];
                    }
                }
            }
        }
        let old = PackedMat::pack(matrix, taps * cin, cols)?;
        let geometry = ConvGeometry {
            in_channels: cin,
            out_channels: cout,
            kernel: k,
            stride: s,
            dilation: 1,
        };
        let up = ConvTranspose1d::new(&weight, &bias, geometry, false, 0.0, None)?;
        assert!(up.poly.weight.bit_identical(&old), "transposed conv");
        // LSTM recurrent weights: the transpose of `w_hh`.
        let dim = 9;
        let w_hh = values(4 * dim * dim, 0.5);
        let mut recurrent = vec![0.0; 4 * dim * dim];
        for (gate, row) in w_hh.chunks_exact(dim).enumerate() {
            for (unit, &v) in row.iter().enumerate() {
                recurrent[unit * 4 * dim + gate] = v;
            }
        }
        let old = PackedMat::pack(recurrent, dim, 4 * dim)?;
        let (input, b) = (values(4 * dim * dim, 0.6), values(4 * dim, 0.7));
        let lstm = Lstm::new(
            dim,
            &[LstmParams {
                w_ih: &input,
                w_hh: &w_hh,
                b_ih: &b,
                b_hh: &b,
            }],
        )?;
        assert!(lstm.layers[0].w_hh.bit_identical(&old), "LSTM");
        Ok(())
    }

    #[test]
    fn reflect_pad_matches_pytorch() -> Result<()> {
        let x = Seq::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])?;
        assert_eq!(
            x.pad(2, 3, PadMode::Reflect).data(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
        );
        // Input shorter than the pad: zeros are appended before reflecting.
        let short = Seq::from_vec(2, 1, vec![1.0, 2.0])?;
        assert_eq!(
            short.pad(0, 3, PadMode::Reflect).data(),
            &[1.0, 2.0, 0.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn extra_padding_matches_reference() {
        // Reference values from EnCodec's get_extra_padding_for_conv1d.
        assert_eq!(extra_padding(10, 4, 2, 2), 0);
        assert_eq!(extra_padding(11, 4, 2, 2), 1);
        assert_eq!(extra_padding(1, 16, 8, 8), 7);
        assert_eq!(extra_padding(5, 7, 1, 6), 0);
    }

    /// Direct transposed convolution used to check the polyphase lowering.
    fn naive_conv_transpose(
        x: &Seq,
        weight: &[f32],
        bias: &[f32],
        (cin, cout, k, s): (usize, usize, usize, usize),
    ) -> Seq {
        let full = (x.len() - 1) * s + k;
        let mut y = Seq::zeros(full, cout);
        for t in 0..full {
            y.data[t * cout..(t + 1) * cout].copy_from_slice(bias);
        }
        for t in 0..x.len() {
            for j in 0..k {
                for i in 0..cin {
                    for o in 0..cout {
                        y.data[(t * s + j) * cout + o] +=
                            x.data[t * cin + i] * weight[(i * cout + o) * k + j];
                    }
                }
            }
        }
        y
    }

    #[test]
    fn polyphase_transposed_conv_matches_naive() -> Result<()> {
        for &(cin, cout, k, s) in &[(3, 2, 4, 2), (2, 3, 16, 8), (1, 1, 5, 2), (2, 2, 3, 3)] {
            let x = seq(7, cin, |i| ((i * 37) % 11) as f32 - 5.0);
            let w: Vec<f32> = (0..cin * cout * k)
                .map(|i| ((i * 13) % 7) as f32 - 3.0)
                .collect();
            let b: Vec<f32> = (0..cout).map(|o| o as f32 * 0.5).collect();
            let layer = ConvTranspose1d::new(
                &w,
                &b,
                ConvGeometry {
                    in_channels: cin,
                    out_channels: cout,
                    kernel: k,
                    stride: s,
                    dilation: 1,
                },
                true,
                0.0,
                None,
            )?;
            // trim_right_ratio 0 with causal trims everything from the left.
            let got = layer.forward(&x, false)?;
            let want = naive_conv_transpose(&x, &w, &b, (cin, cout, k, s));
            let want = want.slice_rows(k - s, want.len());
            assert_eq!(got.len(), want.len());
            for (a, b) in got.data().iter().zip(want.data()) {
                assert!((a - b).abs() < 1e-3, "{a} != {b}");
            }
        }
        Ok(())
    }

    #[test]
    fn conv_matches_naive() -> Result<()> {
        let (cin, cout, k, s, d) = (3, 4, 3, 2, 2);
        let x = seq(20, cin, |i| ((i * 7) % 5) as f32 - 2.0);
        let w: Vec<f32> = (0..cout * cin * k)
            .map(|i| ((i * 3) % 4) as f32 - 1.5)
            .collect();
        let b = vec![0.25; cout];
        let conv = ConvKernel::from_torch(&w, &b, (cout, cin, k), s, d)?;
        let y = conv.apply(&x)?;
        let span = (k - 1) * d + 1;
        assert_eq!(y.len(), (x.len() - span) / s + 1);
        for t in 0..y.len() {
            for o in 0..cout {
                let mut want = b[o];
                for i in 0..cin {
                    for j in 0..k {
                        want += x.data[(t * s + j * d) * cin + i] * w[(o * cin + i) * k + j];
                    }
                }
                assert!((y.data[t * cout + o] - want).abs() < 1e-4);
            }
        }
        Ok(())
    }

    #[test]
    fn strided_conv_without_copy_matches_naive() -> Result<()> {
        // Unit dilation takes the zero-copy path with overlapping windows.
        for &(cin, cout, k, s) in &[(3, 4, 3, 1), (2, 5, 4, 2), (4, 3, 1, 1), (1, 2, 7, 1)] {
            let x = seq(700, cin, |i| (((i * 7) % 13) as f32).mul_add(0.25, -1.5));
            let w: Vec<f32> = (0..cout * cin * k)
                .map(|i| (((i * 5) % 9) as f32).mul_add(0.125, -0.5))
                .collect();
            let b: Vec<f32> = (0..cout).map(|o| (o as f32).mul_add(0.5, -1.0)).collect();
            let y = ConvKernel::from_torch(&w, &b, (cout, cin, k), s, 1)?.apply(&x)?;
            assert_eq!(y.len(), (x.len() - k) / s + 1);
            for t in 0..y.len() {
                for o in 0..cout {
                    let mut want = b[o];
                    for i in 0..cin {
                        for j in 0..k {
                            want += x.data[(t * s + j) * cin + i] * w[(o * cin + i) * k + j];
                        }
                    }
                    assert!((y.data[t * cout + o] - want).abs() < 1e-3);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fused_sum_matches_two_convolutions() -> Result<()> {
        let x = seq(40, 3, |i| ((i * 7) % 5) as f32 - 2.0);
        let y = seq(40, 2, |i| ((i * 3) % 7) as f32 - 3.0);
        let wa: Vec<f32> = (0..12).map(|i| (i as f32).mul_add(0.25, -1.0)).collect();
        let wb: Vec<f32> = (0..8).map(|i| (i as f32).mul_add(-0.125, 0.5)).collect();
        let a = ConvKernel::from_torch(&wa, &[0.5, -0.5, 0.25, 1.0], (4, 3, 1), 1, 1)?;
        let b = ConvKernel::from_torch(&wb, &[1.0, 2.0, 3.0, 4.0], (4, 2, 1), 1, 1)?;
        let mut want = a.apply(&x)?;
        want.add_assign(&b.apply(&y)?)?;
        let joined: Vec<f32> = (0..40)
            .flat_map(|t| x.row(t).iter().chain(y.row(t)).copied())
            .collect();
        let got = ConvKernel::fuse_sum(&a, &b)?.apply(&Seq::from_vec(40, 5, joined)?)?;
        for (g, w) in got.data().iter().zip(want.data()) {
            assert!((g - w).abs() < 1e-4, "{g} != {w}");
        }
        Ok(())
    }

    #[test]
    fn trim_and_concat_round_trip() -> Result<()> {
        let x = seq(6, 2, |i| i as f32);
        let mut head = x.clone();
        head.trim_rows(0, 4);
        let mut tail = x.clone();
        tail.trim_rows(2, 0);
        assert_eq!(head.len(), 2);
        assert_eq!(tail.data(), &x.data()[4..]);
        assert_eq!(Seq::concat(vec![head, tail])?, x);
        Ok(())
    }

    #[test]
    fn group_norm_matches_definition() -> Result<()> {
        let mut x = seq(50, 3, |i| (((i * 11) % 17) as f32).mul_add(0.5, -3.0));
        let reference = x.clone();
        let (gamma, beta) = (vec![1.5, -0.5, 2.0], vec![0.25, 1.0, -1.0]);
        GroupNorm1::new(gamma.clone(), beta.clone())?.apply(&mut x)?;
        let n = reference.data().len() as f64;
        let mean = reference.data().iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = reference
            .data()
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        for (i, (&got, &v)) in x.data().iter().zip(reference.data()).enumerate() {
            let normed = (f64::from(v) - mean) / (var + 1e-5).sqrt();
            let want = normed.mul_add(f64::from(gamma[i % 3]), f64::from(beta[i % 3]));
            assert!((f64::from(got) - want).abs() < 1e-5, "{got} != {want}");
        }
        Ok(())
    }

    /// A normalised layer with weights that follow a fixed pattern.
    fn test_conv(cin: usize, cout: usize, k: usize, seed: usize) -> Result<Conv1d> {
        let w: Vec<f32> = (0..cout * cin * k)
            .map(|i| (((i * 7 + seed) % 13) as f32).mul_add(0.05, -0.3))
            .collect();
        let b: Vec<f32> = (0..cout).map(|o| (o as f32).mul_add(0.1, -0.2)).collect();
        let gamma: Vec<f32> = (0..cout).map(|o| (o as f32).mul_add(0.07, 0.8)).collect();
        let beta: Vec<f32> = (0..cout).map(|o| (o as f32).mul_add(-0.05, 0.1)).collect();
        Ok(Conv1d::new(
            ConvKernel::from_torch(&w, &b, (cout, cin, k), 1, 1)?,
            false,
            PadMode::Reflect,
            Some(GroupNorm1::new(gamma, beta)?),
        ))
    }

    #[test]
    fn pending_work_gives_the_values_of_settled_work() -> Result<()> {
        // More frames than a tile, and a channel count that is not a whole
        // number of registers.
        let x = seq(700, 6, |i| (((i * 11) % 23) as f32).mul_add(0.1, -1.1));
        let (first, second, third) = (
            test_conv(6, 10, 3, 1)?,
            test_conv(6, 10, 1, 2)?,
            test_conv(10, 4, 7, 3)?,
        );
        // Normalisation and ELU applied by the reading layer, or beforehand.
        let lazy = first.forward_lazy(&Input::plain(&x), true)?;
        let read = third.forward_lazy(&lazy.input(), false)?.settle();
        let settled = lazy.settle();
        assert_eq!(read, third.forward(&settled, false)?);
        assert_eq!(settled, first.forward(&x, true)?);

        // The sum of two branches followed by ELU, as in a residual block.
        let mut want = first.forward(&x, false)?;
        want.add_assign(&second.forward(&x, false)?)?;
        let want = want.elu();
        let sum = first
            .forward_lazy(&Input::plain(&x), false)?
            .plus(second.forward_lazy(&Input::plain(&x), false)?)?
            .then_elu();
        assert_eq!(
            third.forward_lazy(&sum.input(), false)?.settle(),
            third.forward(&want, false)?
        );
        assert_eq!(sum.settle(), want);

        // A transposed convolution whose trimming and normalisation are pending.
        let w: Vec<f32> = (0..6 * 4 * 4)
            .map(|i| ((i % 9) as f32).mul_add(0.1, -0.4))
            .collect();
        let up = ConvTranspose1d::new(
            &w,
            &[0.1, -0.1, 0.2, 0.0],
            ConvGeometry {
                in_channels: 6,
                out_channels: 4,
                kernel: 4,
                stride: 2,
                dilation: 1,
            },
            false,
            1.0,
            Some(GroupNorm1::new(
                vec![1.0, 0.9, 1.1, 1.2],
                vec![0.0, 0.1, -0.1, 0.2],
            )?),
        )?;
        let last = test_conv(4, 3, 3, 4)?;
        let lazy = up.forward_lazy(&Input::plain(&x), false)?.then_elu();
        let read = last.forward_lazy(&lazy.input(), false)?.settle();
        let settled = lazy.settle();
        assert_eq!(settled.len(), 2 * x.len());
        assert_eq!(read, last.forward(&settled, false)?);
        Ok(())
    }

    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Textbook LSTM layer in PyTorch's gate-major layout.
    fn naive_lstm_layer(input: &Seq, params: &LstmParams<'_>, d: usize) -> Seq {
        let mut out = Seq::zeros(input.len(), d);
        let (mut hidden, mut cell) = (vec![0.0_f32; d], vec![0.0_f32; d]);
        for t in 0..input.len() {
            let gates: Vec<f32> = (0..4 * d)
                .map(|r| {
                    let wx: f32 = (0..d)
                        .map(|k| params.w_ih[r * d + k] * input.row(t)[k])
                        .sum();
                    let wh: f32 = (0..d).map(|k| params.w_hh[r * d + k] * hidden[k]).sum();
                    wx + wh + params.b_ih[r] + params.b_hh[r]
                })
                .collect();
            for j in 0..d {
                let (i, f) = (sigmoid(gates[j]), sigmoid(gates[d + j]));
                let (g, o) = (gates[2 * d + j].tanh(), sigmoid(gates[3 * d + j]));
                cell[j] = f.mul_add(cell[j], i * g);
                hidden[j] = o * cell[j].tanh();
            }
            out.data[t * d..(t + 1) * d].copy_from_slice(&hidden);
        }
        out
    }

    #[test]
    fn batched_lstm_matches_naive() -> Result<()> {
        let d = 40;
        let gen_weights = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 31 + seed * 17) % 23) as f32 - 11.0) * 0.02)
                .collect()
        };
        let weights: Vec<[Vec<f32>; 4]> = (0..2)
            .map(|l| {
                [
                    gen_weights(l, 4 * d * d),
                    gen_weights(l + 5, 4 * d * d),
                    gen_weights(l + 9, 4 * d),
                    gen_weights(l + 13, 4 * d),
                ]
            })
            .collect();
        let params: Vec<LstmParams<'_>> = weights
            .iter()
            .map(|w| LstmParams {
                w_ih: &w[0],
                w_hh: &w[1],
                b_ih: &w[2],
                b_hh: &w[3],
            })
            .collect();
        let lstm = Lstm::new(d, &params)?;
        // Different lengths, not sorted, including one longer than a projection block.
        let xs: Vec<Seq> = [7_usize, 600, 1, 33]
            .iter()
            .enumerate()
            .map(|(b, &len)| {
                seq(len, d, |i| {
                    ((((i + b * 3) * 13) % 19) as f32).mul_add(0.1, -0.9)
                })
            })
            .collect();
        let got = lstm.forward_batch(&xs)?;
        for (x, y) in xs.iter().zip(&got) {
            let mut want = naive_lstm_layer(&naive_lstm_layer(x, &params[0], d), &params[1], d);
            want.add_assign(x)?;
            assert_eq!(y.len(), want.len());
            for (a, b) in y.data().iter().zip(want.data()) {
                assert!((a - b).abs() < 1e-4, "{a} != {b}");
            }
            let alone = lstm.forward(x)?;
            assert_eq!(&alone, y, "batching must not change a sequence's result");
            // Fed in uneven pieces with carried state: still the same values.
            let mut state = lstm.initial_state();
            let mut pieces = Vec::new();
            let mut start = 0;
            for size in [1, 2, 5, 513, 40].into_iter().cycle() {
                if start >= x.len() {
                    break;
                }
                let end = (start + size).min(x.len());
                pieces.push(lstm.forward_stream(&mut state, &x.slice_rows(start, end))?);
                start = end;
            }
            assert_eq!(&Seq::concat(pieces)?, y, "streaming must match one pass");
        }
        Ok(())
    }
}
