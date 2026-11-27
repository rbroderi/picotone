//! The look-ahead peak limiter at the end of every decode.
//!
//! A decoder network can overshoot full scale where the original did not
//! (HQ at low tiers, in loud passages). Instead of hard-clipping those
//! samples, the limiter lowers the gain smoothly around them: it starts
//! [`ATTACK`] samples before an over, holds for [`HOLD`] samples after it,
//! then returns to 1 over another [`ATTACK`] samples. Stereo channels share
//! one gain, so the image does not move.
//!
//! The gain at a sample depends only on the input from [`LOOKBACK`] samples
//! before it to [`LOOKAHEAD`] samples after it (before the start and after
//! the end, as if silent), and every sum runs in a fixed order. So the output
//! is the same bit for bit however the input is split into chunks, and a
//! decode that starts [`LOOKBACK`] samples before a seek point gives exactly
//! the samples a whole-file decode gives from that point. Where no sample
//! within that span is over [`CEILING`], the gain is exactly 1 and the audio
//! passes through unchanged.

use anyhow::{Result, ensure};

use crate::nn::Seq;

/// The highest output magnitude (−0.09 dBFS, under the WAV writer's guard).
pub const CEILING: f32 = 0.99;

/// Samples over which the gain ramps down before an over, and back up after
/// the hold (2.5 ms at 44.1 kHz; the ramp is twice this, 5 ms).
pub const ATTACK: usize = 110;

/// Samples the lowered gain is held after an over (50 ms at 44.1 kHz), so
/// a loud passage is turned down once, not at every peak.
pub const HOLD: usize = 2205;

/// Samples of input after a sample that its output depends on: the
/// limiter's delay.
pub const LOOKAHEAD: usize = 2 * ATTACK;

/// Samples of input before a sample that its output depends on: how far
/// before a seek point decoding must start.
pub const LOOKBACK: usize = 2 * ATTACK + HOLD;

/// A limiter running over one stream of interleaved samples.
#[derive(Debug, Clone)]
pub struct Limiter {
    channels: usize,
    /// Input samples not returned yet, `[sample][channel]`.
    pending: Vec<f32>,
    /// The gain each sample needs on its own, from [`LOOKBACK`] samples
    /// before the first pending one (1 before the start) to the last pushed.
    need: Vec<f32>,
}

impl Limiter {
    /// A limiter for `channels` interleaved channels, at the start of a
    /// stream.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self {
            channels: channels.max(1),
            pending: Vec::new(),
            need: vec![1.0; LOOKBACK],
        }
    }

    /// Takes the next input samples and returns the output samples that no
    /// later input can change: all but the last [`LOOKAHEAD`] pushed.
    ///
    /// # Errors
    /// Fails when `audio` has another channel count.
    pub fn push(&mut self, audio: &Seq) -> Result<Seq> {
        ensure!(
            audio.channels() == self.channels,
            "the limiter runs on {} channels, not {}",
            self.channels,
            audio.channels()
        );
        self.pending.extend_from_slice(audio.data());
        self.need
            .extend(audio.data().chunks_exact(self.channels).map(needed_gain));
        let ready = self.pending_len().saturating_sub(LOOKAHEAD);
        self.emit(ready)
    }

    /// Ends the stream and returns the samples still held back.
    ///
    /// # Errors
    /// Fails only on an internal size mismatch.
    pub fn finish(mut self) -> Result<Seq> {
        self.need.extend(std::iter::repeat_n(1.0, LOOKAHEAD));
        let all = self.pending_len();
        self.emit(all)
    }

    const fn pending_len(&self) -> usize {
        self.pending.len() / self.channels
    }

    /// Returns the first `n` pending samples, limited. Each needs the gains
    /// from [`LOOKBACK`] samples before it to [`LOOKAHEAD`] after it, which
    /// `need` holds.
    fn emit(&mut self, n: usize) -> Result<Seq> {
        let c = self.channels;
        let mut out: Vec<f32> = self.pending.drain(..n * c).collect();
        if n > 0 {
            let span = &self.need[..n + LOOKBACK + LOOKAHEAD];
            if span.iter().any(|&g| g < 1.0) {
                for (frame, g) in out.chunks_exact_mut(c).zip(gains(span, n)) {
                    if g < 1.0 {
                        for v in frame {
                            *v = (*v * g).clamp(-CEILING, CEILING);
                        }
                    }
                }
            }
            drop(self.need.drain(..n));
        }
        Seq::from_vec(n, c, out)
    }
}

/// Limits a whole signal: the same samples as pushing it in any chunks.
///
/// # Errors
/// Fails only on an internal size mismatch.
pub fn limit(audio: &Seq) -> Result<Seq> {
    let mut limiter = Limiter::new(audio.channels());
    let head = limiter.push(audio)?;
    Seq::concat(vec![head, limiter.finish()?])
}

/// The gain one sample needs on its own: 1 unless a channel is over the
/// ceiling.
fn needed_gain(frame: &[f32]) -> f32 {
    let peak = frame.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    if peak > CEILING { CEILING / peak } else { 1.0 }
}

/// The gains of `n` samples, the first at `need[LOOKBACK]`.
///
/// First the held minimum: at each sample the lowest need from
/// `ATTACK + HOLD` before it to `ATTACK` after it. Then its mean over
/// `ATTACK` either side, which ramps it. Each held minimum the mean takes is
/// at or below the sample's own need, so the gain never lets a sample over
/// the ceiling.
fn gains(need: &[f32], n: usize) -> Vec<f32> {
    let width = HOLD + 2 * ATTACK + 1;
    let count = n + 2 * ATTACK;
    // held[k] is the minimum of need[k .. k + width], by a sliding window of
    // indices whose needs only rise.
    let mut held = Vec::with_capacity(count);
    let mut window = std::collections::VecDeque::new();
    for j in 0..count + width - 1 {
        while window.back().is_some_and(|&b: &usize| need[b] >= need[j]) {
            let _: Option<usize> = window.pop_back();
        }
        window.push_back(j);
        if j + 1 >= width {
            let k = j + 1 - width;
            while window.front().is_some_and(|&f| f < k) {
                let _: Option<usize> = window.pop_front();
            }
            held.push(window.front().map_or(1.0, |&f| need[f]));
        }
    }
    let taps = 2 * ATTACK + 1;
    #[expect(clippy::cast_precision_loss, reason = "221 is exact in f32")]
    let scale = taps as f32;
    (0..n)
        .map(|i| held[i..i + taps].iter().sum::<f32>() / scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stereo test signal with a few overs of up to 1.5 in quiet music.
    fn signal(len: usize) -> Result<Seq> {
        let mut data = Vec::with_capacity(len * 2);
        let mut state = 12_345_u32;
        for t in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "test signal")]
            let (noise, phase) = (
                (state >> 8) as f32 / 16_777_216.0 - 0.5,
                t as f32 * 0.031,
            );
            let mut v = 0.5 * phase.sin() + 0.2 * noise;
            if (20_000..20_400).contains(&t) || (31_000..31_050).contains(&t) {
                v *= 3.0;
            }
            data.extend([v, -0.8 * v]);
        }
        Seq::from_vec(len, 2, data)
    }

    fn bits(s: &Seq) -> Vec<u32> {
        s.data().iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn keeps_quiet_audio_bit_for_bit_and_limits_overs() -> Result<()> {
        let x = signal(60_000)?;
        let y = limit(&x)?;
        assert_eq!(y.len(), x.len());
        assert!(y.data().iter().all(|v| v.abs() <= CEILING));
        // Far from the overs nothing changes.
        let far = |t: usize| t + LOOKAHEAD < 20_000 || t > 31_050 + LOOKBACK;
        for t in (0..x.len()).filter(|&t| far(t)) {
            assert_eq!(x.row(t), y.row(t), "sample {t}");
        }
        // Next to them the gain is lowered, not clipped: the ratio of the
        // channels stays (to the rounding of the mean, which the clamp
        // catches).
        for t in 19_000..33_000 {
            let (a, b) = (y.row(t)[0], y.row(t)[1]);
            assert!(
                (b + 0.8 * a).abs() <= 1e-5,
                "sample {t}: {a} {b} from {:?}",
                x.row(t)
            );
        }
        Ok(())
    }

    #[test]
    fn chunks_and_seeks_give_the_whole_signal_bit_for_bit() -> Result<()> {
        let x = signal(45_000)?;
        let want = bits(&limit(&x)?);
        for chunk in [1, 7, 219, 220, 221, 4096, 45_000] {
            let mut limiter = Limiter::new(2);
            let mut parts = Vec::new();
            for start in (0..x.len()).step_by(chunk) {
                let end = (start + chunk).min(x.len());
                parts.push(limiter.push(&x.slice_rows(start, end))?);
            }
            parts.push(limiter.finish()?);
            assert_eq!(bits(&Seq::concat(parts)?), want, "chunk {chunk}");
        }
        for seek in [1, LOOKBACK, 19_990, 20_300, 31_020, 44_999] {
            let from = seek.saturating_sub(LOOKBACK);
            let got = limit(&x.slice_rows(from, x.len()))?;
            let got = got.slice_rows(seek - from, got.len());
            assert_eq!(bits(&got), want[seek * 2..], "seek {seek}");
        }
        Ok(())
    }
}
