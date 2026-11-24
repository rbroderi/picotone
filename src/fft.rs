//! The inverse real FFT behind the Vocos decoder's iSTFT.
//!
//! One plan per size (a power of two): a complex inverse FFT of half the
//! size (Stockham radix 2, natural order in and out, no bit reversal) plus
//! the usual split step that turns `n / 2 + 1` Hermitian bins into `n` real
//! samples. Twiddles are computed in double precision and rounded once.
//!
//! It is plain Rust rather than a crate (`rustfft`/`realfft` would do): the
//! decoder needs one fixed size, the transform is well under 1% of the
//! decoder's work (about 0.06 MFLOP per frame against 28 MMAC for the
//! network), and a few dozen lines keep the WebAssembly build small, add no
//! dependency, and compute every frame with the same operations on every
//! target and in every chunk.

#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use anyhow::{Result, ensure};

/// `(cos, sin)` of `2 pi k / n` in double precision, rounded to `f32`.
fn twiddle(k: usize, n: usize) -> (f32, f32) {
    #[expect(clippy::cast_precision_loss, reason = "FFT sizes are far below 2^52")]
    let angle = std::f64::consts::TAU * k as f64 / n as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounding the twiddle to single precision"
    )]
    let pair = (angle.cos() as f32, angle.sin() as f32);
    pair
}

/// An inverse real FFT of a fixed size `n`: `n / 2 + 1` complex bins to `n`
/// real samples, without the `1 / n` normalisation (`n` times
/// `numpy.fft.irfft`). The imaginary parts of the first and last bin are ignored.
#[derive(Debug, Clone)]
pub(crate) struct RealInverse {
    n: usize,
    /// Per stage of the half-size complex FFT, `exp(+2 pi i p / len)` for
    /// `p < len / 2`, as separate real and imaginary parts.
    stages: Vec<(Vec<f32>, Vec<f32>)>,
    /// `exp(+2 pi i k / n)` for `k < n / 2`.
    split: (Vec<f32>, Vec<f32>),
}

/// Working memory of [`RealInverse::run`], reused across calls.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    re: [Vec<f32>; 2],
    im: [Vec<f32>; 2],
}

impl Scratch {
    /// Empty buffers, sized on first use.
    pub(crate) const fn new() -> Self {
        Self {
            re: [Vec::new(), Vec::new()],
            im: [Vec::new(), Vec::new()],
        }
    }
}

impl RealInverse {
    /// A plan for size `n`, a power of two of at least 4.
    pub(crate) fn new(n: usize) -> Result<Self> {
        ensure!(
            n >= 4 && n.is_power_of_two(),
            "FFT size {n} is not a power of two of at least 4"
        );
        let m = n / 2;
        let mut stages = Vec::new();
        let mut len = m;
        while len > 1 {
            let (re, im) = (0..len / 2).map(|p| twiddle(p, len)).unzip();
            stages.push((re, im));
            len /= 2;
        }
        Ok(Self {
            n,
            stages,
            split: (0..m).map(|k| twiddle(k, n)).unzip(),
        })
    }

    /// Bins `n / 2 + 1` (`re`, `im`) to `n` samples in `out`.
    pub(crate) fn run(&self, re: &[f32], im: &[f32], out: &mut [f32], scratch: &mut Scratch) {
        let m = self.n / 2;
        debug_assert!(re.len() > m && im.len() > m && out.len() >= self.n);
        for buffer in scratch.re.iter_mut().chain(&mut scratch.im) {
            buffer.resize(m, 0.0);
        }
        let ([zr, yr], [zi, yi]) = (&mut scratch.re, &mut scratch.im);
        // Z[k] = (X[k] + conj X[m - k]) + i (X[k] - conj X[m - k]) w^k, the
        // half-size spectrum of z[j] = x[2j] + i x[2j + 1] (times 2).
        let (wr, wi) = &self.split;
        for k in 0..m {
            let (ar, ai) = (re[k], if k == 0 { 0.0 } else { im[k] });
            let (br, bi) = (re[m - k], if k == 0 { 0.0 } else { -im[m - k] });
            let (er, ei) = (ar + br, ai + bi);
            let (dr, di) = (ar - br, ai - bi);
            let (tr, ti) = (dr * wr[k] - di * wi[k], dr * wi[k] + di * wr[k]);
            zr[k] = er - ti;
            zi[k] = ei + tr;
        }
        // Stockham autosort: stage s reads one buffer and writes the other.
        let mut len = m;
        let mut stride = 1;
        let mut in_z = true;
        for (tr, ti) in &self.stages {
            let half = len / 2;
            let (src_r, src_i, dst_r, dst_i) = if in_z {
                (&*zr, &*zi, &mut *yr, &mut *yi)
            } else {
                (&*yr, &*yi, &mut *zr, &mut *zi)
            };
            for p in 0..half {
                let (wr, wi) = (tr[p], ti[p]);
                let a = stride * p;
                let b = stride * (p + half);
                let even = stride * 2 * p;
                let odd = even + stride;
                for q in 0..stride {
                    let (xr, xi) = (src_r[a + q], src_i[a + q]);
                    let (vr, vi) = (src_r[b + q], src_i[b + q]);
                    dst_r[even + q] = xr + vr;
                    dst_i[even + q] = xi + vi;
                    let (dr, di) = (xr - vr, xi - vi);
                    dst_r[odd + q] = dr * wr - di * wi;
                    dst_i[odd + q] = dr * wi + di * wr;
                }
            }
            len = half;
            stride *= 2;
            in_z = !in_z;
        }
        let (fr, fi) = if in_z { (&*zr, &*zi) } else { (&*yr, &*yi) };
        for (pair, (&r, &i)) in out.as_chunks_mut::<2>().0.iter_mut().zip(fr.iter().zip(fi)) {
            *pair = [r, i];
        }
    }
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss, reason = "small test values")]
mod tests {
    use super::*;

    #[test]
    fn matches_a_naive_inverse_dft() -> Result<()> {
        for n in [4, 8, 32, 2048] {
            let m = n / 2;
            let re: Vec<f32> = (0..=m).map(|k| ((k as f32) * 0.37).sin() * 2.0).collect();
            let im: Vec<f32> = (0..=m).map(|k| ((k as f32) * 0.91 + 0.2).cos()).collect();
            let mut out = vec![0.0; n];
            RealInverse::new(n)?.run(&re, &im, &mut out, &mut Scratch::default());
            let mut worst = 0.0_f64;
            for (j, &got) in out.iter().enumerate() {
                // x[j] = sum over the Hermitian spectrum, DC and Nyquist real.
                let mut want =
                    f64::from(re[0]) + f64::from(re[m]) * if j % 2 == 0 { 1.0 } else { -1.0 };
                for k in 1..m {
                    let a = std::f64::consts::TAU * (k * j) as f64 / n as f64;
                    want += 2.0 * (f64::from(re[k]) * a.cos() - f64::from(im[k]) * a.sin());
                }
                worst = worst.max((f64::from(got) - want).abs());
            }
            assert!(worst < 1e-5 * n as f64, "n {n}: worst {worst:e}");
        }
        Ok(())
    }
}
