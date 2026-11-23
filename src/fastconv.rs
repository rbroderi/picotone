//! Convolutions on faer's matrix product, for the HQ encoder.
//!
//! On one large product faer's blocked kernel is 1.2 to 2x faster than
//! [`crate::nn::ConvKernel`]'s; inside the encoder (256-row tiles, the laptop
//! at its power limit) the gain is about 10%, measured with alternating runs.
//! Elsewhere it gains nothing (EnCodec, the HQ decoder) or loses (the LSTM's
//! few-row products), so only the encoder uses it. Its summation order
//! differs from PyTorch's and the packed kernel's, so the encoder's latent
//! matches PyTorch to rounding (the parity check allows 1e-4).

#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use anyhow::{Result, ensure};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;

use crate::nn::Seq;

/// Output rows per matrix product (one rayon task).
const TILE: usize = 256;

/// A 1-D convolution over an already padded `[time, channels]` input.
#[derive(Debug, Clone)]
pub(crate) struct FastConv {
    /// `[kernel * cin, cout]`, row-major: row `tap * cin + c`.
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
}

impl FastConv {
    /// From PyTorch's `[cout, cin, kernel]` weight.
    pub(crate) fn from_torch(
        weight: &[f32],
        bias: &[f32],
        (cout, cin, kernel): (usize, usize, usize),
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        ensure!(
            weight.len() == cout * cin * kernel && bias.len() == cout,
            "conv weight has {} values, expected {cout} x {cin} x {kernel}",
            weight.len()
        );
        ensure!(stride > 0 && dilation > 0 && kernel > 0, "degenerate conv");
        let mut packed = vec![0.0; kernel * cin * cout];
        for o in 0..cout {
            for c in 0..cin {
                for j in 0..kernel {
                    packed[(j * cin + c) * cout + o] = weight[(o * cin + c) * kernel + j];
                }
            }
        }
        Ok(Self {
            weight: packed,
            bias: bias.to_vec(),
            cin,
            cout,
            kernel,
            stride,
            dilation,
        })
    }

    /// Convolves `x` (no further padding): `[(len - span) / stride + 1, cout]`.
    pub(crate) fn apply(&self, x: &Seq) -> Result<Seq> {
        ensure!(
            x.channels() == self.cin,
            "conv input is {} wide, expected {}",
            x.channels(),
            self.cin
        );
        let span = (self.kernel - 1) * self.dilation + 1;
        let rows = if x.len() >= span {
            (x.len() - span) / self.stride + 1
        } else {
            0
        };
        let (k, n, cin) = (self.kernel * self.cin, self.cout, self.cin);
        let mut out = vec![0.0; rows * n];
        let w = MatRef::from_row_major_slice(&self.weight, k, n);
        let input = x.data();
        out.par_chunks_mut(TILE * n)
            .enumerate()
            .for_each(|(t, dst)| {
                let first = t * TILE;
                let m = dst.len() / n;
                for row in dst.chunks_exact_mut(n) {
                    row.copy_from_slice(&self.bias);
                }
                let dst = MatMut::from_row_major_slice_mut(dst, m, n);
                let start = first * self.stride * cin;
                if self.dilation == 1 {
                    // Row r's taps are contiguous: the input read in place.
                    let len = (m - 1) * self.stride * cin + k;
                    let a = MatRef::from_row_major_slice_with_stride(
                        &input[start..start + len],
                        m,
                        k,
                        self.stride * cin,
                    );
                    matmul(dst, Accum::Add, a, w, 1.0, Par::Seq);
                } else {
                    let mut cols = vec![0.0; m * k];
                    for (r, row) in cols.chunks_exact_mut(k).enumerate() {
                        let base = start + r * self.stride * cin;
                        for (j, tap) in row.chunks_exact_mut(cin).enumerate() {
                            let at = base + j * self.dilation * cin;
                            tap.copy_from_slice(&input[at..at + cin]);
                        }
                    }
                    let a = MatRef::from_row_major_slice(&cols, m, k);
                    matmul(dst, Accum::Add, a, w, 1.0, Par::Seq);
                }
            });
        Seq::from_vec(rows, n, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::ConvKernel;

    #[test]
    fn matches_the_packed_kernel() -> Result<()> {
        let mut s = 7_u64;
        let mut noise = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    f32::from(u16::try_from(s >> 48).unwrap_or(0)) / 65536.0 - 0.5
                })
                .collect()
        };
        for (cout, cin, kernel, stride, dilation) in [
            (16, 8, 7, 1, 3),
            (16, 8, 4, 2, 1),
            (8, 16, 1, 1, 1),
            (12, 6, 7, 1, 9),
        ] {
            let w = noise(cout * cin * kernel);
            let b = noise(cout);
            let x = Seq::from_vec(700, cin, noise(700 * cin))?;
            let ours =
                FastConv::from_torch(&w, &b, (cout, cin, kernel), stride, dilation)?.apply(&x)?;
            let reference =
                ConvKernel::from_torch(&w, &b, (cout, cin, kernel), stride, dilation)?.apply(&x)?;
            assert_eq!(ours.len(), reference.len());
            let diff = ours
                .data()
                .iter()
                .zip(reference.data())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                diff < 1e-4,
                "{cout}x{cin}x{kernel} s{stride} d{dilation}: {diff}"
            );
        }
        Ok(())
    }
}
