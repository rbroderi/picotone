//! Stereo decisions for the HQ codec: the automatic mono flag and the
//! left/right against mid/side pick (`distill/INTEGRATION.md`, decision 2).
//!
//! - **Mono flag** (`distill/stereo_scan.py`'s rule). Pages of 40,960
//!   samples. A page where both channels are below -60 dBFS is silent and
//!   skipped. A side below -70 dBFS counts as none. The file is mono when
//!   every other page's side, `(L - R) / 2`, is more than 40 dB below its
//!   mid, `(L + R) / 2`.
//! - **Layout pick** (`distill/stereo_pick_check.py`). Up to 4 excerpts of
//!   3 s, the loudest window in each of 4 equal regions, each coded both ways
//!   with 1 s of context on each side and scored on its centre. The score is
//!   the log-mel distance of `distill/metrics.py` (80 mels, 2048-point FFT,
//!   floor 80 dB under the peak), averaged over L, R and the side. Sampling
//!   stops after 2 excerpts when the layouts differ by more than 5%.
//!   Checked against coding whole tracks both ways: 35 of 36 tracks agreed,
//!   and the miss cost 0.001.

use anyhow::{Result, ensure};

use crate::config::to_f64;
use crate::pico::ChannelLayout;

/// Sample rate the rules are defined at.
const RATE: usize = 44_100;
/// Samples per page of the mono check (80 HQ frames).
const PAGE: usize = 40_960;
/// Mean square below which both channels make a page silent (-60 dBFS).
const SILENCE: f64 = 1e-6;
/// Mean square below which the side counts as none (-70 dBFS).
const SIDE_FLOOR: f64 = 1e-7;
/// A file is mono when its loudest side is this far below its mid (-40 dB).
const MONO_RATIO: f64 = 1e-4;

/// Whether a stereo signal is mono by the rule above.
#[must_use]
pub fn is_mono(left: &[f32], right: &[f32]) -> bool {
    let n = left.len().min(right.len());
    let mut start = 0;
    while start + PAGE / 2 < n {
        let end = (start + PAGE).min(n);
        let (mut l2, mut r2, mut m2, mut s2) = (0.0, 0.0, 0.0, 0.0);
        for (&l, &r) in left[start..end].iter().zip(&right[start..end]) {
            let (l, r) = (f64::from(l), f64::from(r));
            l2 += l * l;
            r2 += r * r;
            m2 += (l + r) * (l + r) / 4.0;
            s2 += (l - r) * (l - r) / 4.0;
        }
        let len = to_f64(end - start);
        let (l2, r2, m2, s2) = (l2 / len, r2 / len, m2 / len, s2 / len);
        if l2.max(r2) >= SILENCE && s2 >= SIDE_FLOOR && (m2 <= 0.0 || s2 / m2 >= MONO_RATIO) {
            return false;
        }
        start += PAGE;
    }
    true
}

/// FFT size of the log-mel score.
const N_FFT: usize = 2048;
/// Hop of the log-mel score.
const STFT_HOP: usize = N_FFT / 4;
/// Mel bands of the log-mel score.
const MELS: usize = 80;

/// The log-mel spectrogram's filters and window.
#[derive(Debug, Clone)]
pub struct LogMel {
    /// `[mel][bin]` triangle weights.
    filters: Vec<f64>,
    /// Periodic Hann window.
    window: Vec<f64>,
    /// Twiddle factors `e^{-2 pi i k / N}` for `k < N / 2`.
    twiddles: Vec<(f64, f64)>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    /// Builds the filters for 44.1 kHz.
    #[must_use]
    pub fn new() -> Self {
        let bins = N_FFT / 2 + 1;
        let hz2mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let mel2hz = |m: f64| 700.0 * (10_f64.powf(m / 2595.0) - 1.0);
        let nyquist = to_f64(RATE) / 2.0;
        let (lo, hi) = (hz2mel(30.0), hz2mel(nyquist));
        let points: Vec<f64> = (0..MELS + 2)
            .map(|i| mel2hz(lo + (hi - lo) * to_f64(i) / to_f64(MELS + 1)))
            .collect();
        let mut filters = vec![0.0; MELS * bins];
        for (m, row) in filters.chunks_exact_mut(bins).enumerate() {
            let (a, c, b) = (points[m], points[m + 1], points[m + 2]);
            for (k, w) in row.iter_mut().enumerate() {
                let f = nyquist * to_f64(k) / to_f64(bins - 1);
                *w = ((f - a) / (c - a)).min((b - f) / (b - c)).max(0.0);
            }
        }
        let window = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * to_f64(i) / to_f64(N_FFT)).cos())
            .collect();
        let twiddles = (0..N_FFT / 2)
            .map(|k| {
                let a = -std::f64::consts::TAU * to_f64(k) / to_f64(N_FFT);
                (a.cos(), a.sin())
            })
            .collect();
        Self {
            filters,
            window,
            twiddles,
        }
    }

    /// In-place radix-2 FFT of `N_FFT` points.
    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.twiddles[k * step];
                    let (a, b) = (start + k, start + k + len / 2);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }

    /// Mel energies `[frame][mel]` of `x`, framed as `torch.stft` with
    /// `center=True` (reflect padding).
    fn mel(&self, x: &[f32]) -> Vec<f64> {
        let pad = N_FFT / 2;
        let n = x.len();
        let at = |i: isize| -> f64 {
            // Reflect without repeating the edge; inputs shorter than the pad
            // reflect as far as they can, then read zeros.
            let len = isize::try_from(n).unwrap_or(isize::MAX);
            let j = if i < 0 {
                -i
            } else if i >= len {
                2 * (len - 1) - i
            } else {
                i
            };
            usize::try_from(j)
                .ok()
                .and_then(|j| x.get(j))
                .map_or(0.0, |&v| f64::from(v))
        };
        let frames = 1 + n / STFT_HOP;
        let bins = N_FFT / 2 + 1;
        let mut out = Vec::with_capacity(frames * MELS);
        let (mut re, mut im) = (vec![0.0; N_FFT], vec![0.0; N_FFT]);
        let mut mag = vec![0.0; bins];
        for t in 0..frames {
            let origin = isize::try_from(t * STFT_HOP).unwrap_or(isize::MAX)
                - isize::try_from(pad).unwrap_or(0);
            for (i, (r, w)) in re.iter_mut().zip(&self.window).enumerate() {
                *r = at(origin + isize::try_from(i).unwrap_or(0)) * w;
            }
            im.fill(0.0);
            self.fft(&mut re, &mut im);
            for (k, m) in mag.iter_mut().enumerate() {
                *m = re[k].hypot(im[k]);
            }
            for row in self.filters.chunks_exact(bins) {
                out.push(row.iter().zip(&mag).map(|(w, m)| w * m).sum());
            }
        }
        out
    }

    /// Log-mel distance in dB of `decoded` from `reference`; `None` when the
    /// reference is digital silence.
    #[must_use]
    pub fn distance(&self, reference: &[f32], decoded: &[f32]) -> Option<f64> {
        let n = reference.len().min(decoded.len());
        let (reference, decoded) = (&reference[..n], &decoded[..n]);
        if reference.iter().all(|v| v.abs() < 1e-4) {
            return None;
        }
        let (mr, my) = (self.mel(reference), self.mel(decoded));
        let floor = mr.iter().copied().fold(0.0, f64::max) * 1e-4;
        let total: f64 = mr
            .iter()
            .zip(&my)
            .map(|(&r, &y)| (20.0 * (y.max(floor).log10() - r.max(floor).log10())).abs())
            .sum();
        Some(total / to_f64(mr.len()))
    }

    /// The stereo score: the mean distance over L, R and the side (lower is better).
    #[must_use]
    pub fn stereo_score(&self, reference: [&[f32]; 2], decoded: [&[f32]; 2]) -> f64 {
        let side = |x: [&[f32]; 2]| -> Vec<f32> {
            x[0].iter().zip(x[1]).map(|(l, r)| (l - r) / 2.0).collect()
        };
        let scores: Vec<f64> = [
            self.distance(reference[0], decoded[0]),
            self.distance(reference[1], decoded[1]),
            self.distance(&side(reference), &side(decoded)),
        ]
        .into_iter()
        .flatten()
        .collect();
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / to_f64(scores.len())
        }
    }
}

/// Excerpts the pick samples.
const EXCERPTS: usize = 4;
/// Samples per excerpt (3 s, whole frames).
const EXCERPT: usize = 3 * RATE / 512 * 512;
/// Context on each side of an excerpt (1 s, whole frames).
const CONTEXT: usize = RATE / 512 * 512;
/// Excerpts scored before the early exit may stop.
const MIN_EXCERPTS: usize = 2;
/// Relative margin that ends sampling early.
const MARGIN: f64 = 0.05;

/// Start of each excerpt: the loudest `EXCERPT`-sample window (searched in
/// half-second steps, frame aligned) in each of `EXCERPTS` equal regions,
/// leaving `CONTEXT` free on each side where the file allows.
fn excerpt_starts(left: &[f32], right: &[f32]) -> Vec<usize> {
    let n = left.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    let mut acc = 0.0;
    for (&l, &r) in left.iter().zip(right) {
        acc += f64::midpoint(f64::from(l) * f64::from(l), f64::from(r) * f64::from(r));
        prefix.push(acc);
    }
    let energy = |s: usize| prefix[(s + EXCERPT).min(n)] - prefix[s.min(n)];
    (0..EXCERPTS)
        .map(|k| {
            let (r0, r1) = (n * k / EXCERPTS, n * (k + 1) / EXCERPTS);
            let (mut lo, mut hi) = (
                r0.max(CONTEXT),
                r1.min(n.saturating_sub(CONTEXT)).saturating_sub(EXCERPT),
            );
            if hi < lo {
                (lo, hi) = (r0, r1.saturating_sub(EXCERPT).max(r0));
            }
            let mut best = (lo, f64::NEG_INFINITY);
            for s in (lo..=hi).step_by(RATE / 2) {
                let e = energy(s);
                if e > best.1 {
                    best = (s, e);
                }
            }
            best.0 / 512 * 512
        })
        .collect()
}

/// Mean left/right and mid/side scores so far.
#[derive(Debug, Clone, Copy, Default)]
pub struct PickScores {
    /// Mean score coding left/right.
    pub left_right: f64,
    /// Mean score coding mid/side.
    pub mid_side: f64,
    /// Excerpts scored (0: the whole file was compared).
    pub excerpts: usize,
}

/// Picks the stereo layout.
///
/// `roundtrip(left, right, layout)` codes a stereo excerpt in `layout` and
/// returns the decoded left and right, the same length as the input. Files
/// shorter than the excerpts with their context are compared whole.
///
/// # Errors
/// Fails when `roundtrip` does, or returns the wrong length.
pub fn pick_layout<F>(
    left: &[f32],
    right: &[f32],
    mut roundtrip: F,
) -> Result<(ChannelLayout, PickScores)>
where
    F: FnMut(&[f32], &[f32], ChannelLayout) -> Result<[Vec<f32>; 2]>,
{
    let mel = LogMel::new();
    let n = left.len().min(right.len());
    let mut score = |lo: usize, hi: usize, start: usize, end: usize| -> Result<(f64, f64)> {
        let mut out = [0.0; 2];
        for (slot, layout) in out
            .iter_mut()
            .zip([ChannelLayout::LeftRight, ChannelLayout::MidSide])
        {
            let [l, r] = roundtrip(&left[lo..hi], &right[lo..hi], layout)?;
            ensure!(
                l.len() == hi - lo && r.len() == hi - lo,
                "the layout round trip changed the length"
            );
            let clip = |v: &[f32]| -> Vec<f32> {
                v[start - lo..end - lo]
                    .iter()
                    .map(|x| x.clamp(-1.0, 1.0))
                    .collect()
            };
            *slot = mel.stereo_score(
                [&left[start..end], &right[start..end]],
                [&clip(&l), &clip(&r)],
            );
        }
        Ok(out.into())
    };
    let pick = |s: &PickScores| {
        if s.left_right <= s.mid_side {
            ChannelLayout::LeftRight
        } else {
            ChannelLayout::MidSide
        }
    };
    if n <= EXCERPTS * (EXCERPT + 2 * CONTEXT) {
        let (lr, ms) = score(0, n, 0, n)?;
        let s = PickScores {
            left_right: lr,
            mid_side: ms,
            excerpts: 0,
        };
        return Ok((pick(&s), s));
    }
    let (mut sum_lr, mut sum_ms) = (0.0, 0.0);
    let mut s = PickScores::default();
    for (k, start) in excerpt_starts(&left[..n], &right[..n])
        .into_iter()
        .enumerate()
    {
        let end = (start + EXCERPT).min(n);
        let (lr, ms) = score(
            start.saturating_sub(CONTEXT),
            (end + CONTEXT).min(n),
            start,
            end,
        )?;
        sum_lr += lr;
        sum_ms += ms;
        s = PickScores {
            left_right: sum_lr / to_f64(k + 1),
            mid_side: sum_ms / to_f64(k + 1),
            excerpts: k + 1,
        };
        let best = s.left_right.min(s.mid_side);
        if k + 1 >= MIN_EXCERPTS && best > 0.0 && (s.left_right - s.mid_side).abs() / best > MARGIN
        {
            break;
        }
    }
    Ok((pick(&s), s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, freq: f64, gain: f64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = gain * (std::f64::consts::TAU * freq * to_f64(i) / to_f64(RATE)).sin();
                #[expect(clippy::cast_possible_truncation, reason = "test tone in f32 range")]
                let v = v as f32;
                v
            })
            .collect()
    }

    #[test]
    fn mono_rule() {
        let n = RATE * 3;
        let l = tone(n, 440.0, 0.5);
        assert!(is_mono(&l, &l));
        // A side 50 dB under the mid is still mono, 30 dB under is stereo.
        let quiet: Vec<f32> = l
            .iter()
            .zip(tone(n, 660.0, 0.5 * 10_f64.powf(-50.0 / 20.0)))
            .map(|(a, b)| a + b)
            .collect();
        assert!(is_mono(&l, &quiet));
        let wide: Vec<f32> = l
            .iter()
            .zip(tone(n, 660.0, 0.5 * 10_f64.powf(-30.0 / 20.0)))
            .map(|(a, b)| a + b)
            .collect();
        assert!(!is_mono(&l, &wide));
        // Silence is mono.
        assert!(is_mono(&vec![0.0; n], &vec![1e-5; n]));
    }

    /// Two tones and a chirp, with the given gains.
    fn mix(gains: [f64; 3]) -> Vec<f32> {
        (0..RATE)
            .map(|i| {
                let t = to_f64(i);
                let w = std::f64::consts::TAU / to_f64(RATE);
                let v = gains[0] * (w * 440.0 * t).sin()
                    + gains[1] * (w * 3000.0 * t).sin()
                    + gains[2] * (t * t * 0.001).sin();
                #[expect(clippy::cast_possible_truncation, reason = "test signal in f32 range")]
                let v = v as f32;
                v
            })
            .collect()
    }

    #[test]
    fn log_mel_distance_matches_the_python_metric() {
        let mel = LogMel::new();
        let x = mix([0.3, 0.2, 0.05]);
        assert!(mel.distance(&x, &x).is_some_and(|d| d.abs() < 1e-9));
        // distill/metrics.py's score(x, y)["logmel"] on the same signals.
        let y = mix([0.25, 0.1, 0.025]);
        let d = mel.distance(&x, &y).unwrap_or(f64::NAN);
        assert!((d - 4.778_193).abs() < 1e-3, "{d}");
        let half: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        let d = mel.distance(&x, &half).unwrap_or(f64::NAN);
        assert!((d - 5.253_406).abs() < 1e-3, "{d}");
        assert!(mel.distance(&[0.0; 4096], &x).is_none());
    }

    #[test]
    fn excerpts_find_the_loud_parts() {
        let n = RATE * 60;
        let mut l = vec![0.0_f32; n];
        // One loud second in each quarter.
        for k in 0..4 {
            let at = n * k / 4 + RATE * 7;
            for v in &mut l[at..at + RATE] {
                *v = 0.5;
            }
        }
        let starts = excerpt_starts(&l, &l);
        assert_eq!(starts.len(), 4);
        for (k, s) in starts.into_iter().enumerate() {
            let at = n * k / 4 + RATE * 7;
            assert!(
                s <= at && s + EXCERPT >= at + RATE,
                "excerpt {k} at {s} misses {at}"
            );
        }
    }
}
