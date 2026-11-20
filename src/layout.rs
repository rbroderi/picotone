//! How a file's audio is laid out in coded segments and trimmed afterwards:
//! the segment stride, the cross-fade between segments, and the pre-skip,
//! end trim and loop fold applied to the decoded samples.
//!
//! A `.pico` file records all of them in its identification header (see
//! [`crate::pico::Info`]).

use anyhow::{Context as _, Result, ensure};

use crate::config::{Config, to_f64};
use crate::nn::Seq;

/// Samples of the old segment's tail the edge-avoiding cross-fade never plays
/// (when the overlap is long enough).
const EDGE_TAIL: usize = 480;
/// Samples of the new segment's head the edge-avoiding cross-fade skips at most.
const EDGE_HEAD: usize = 1440;
/// Shortest ramp the edge-avoiding cross-fade keeps before trimming the head.
const EDGE_RAMP: usize = 240;

/// How overlapping segments are blended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Crossfade {
    /// EnCodec's triangle-weighted overlap-add (code 0).
    #[default]
    Triangle,
    /// A linear ramp that avoids the damaged edges of both segments (code 1).
    EdgeAvoiding,
}

impl Crossfade {
    /// The header code: 0 triangle, 1 edge-avoiding.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Triangle => 0,
            Self::EdgeAvoiding => 1,
        }
    }

    /// Parses a header code.
    ///
    /// # Errors
    /// Fails on codes other than 0 and 1.
    pub fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Triangle),
            1 => Ok(Self::EdgeAvoiding),
            other => anyhow::bail!("unknown cross-fade {other} (expected 0 or 1)"),
        }
    }
}

/// Segment stride and cross-fade of one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Segmenting {
    /// Distance between segment starts; `None` uses the model's default.
    pub stride: Option<usize>,
    /// Cross-fade between overlapping segments.
    pub crossfade: Crossfade,
}

impl Segmenting {
    /// EnCodec's own layout: the model's stride and the triangle cross-fade.
    pub const STOCK: Self = Self {
        stride: None,
        crossfade: Crossfade::Triangle,
    };
}

/// Checks a stride and cross-fade against the model. Strides run from half a
/// segment to a whole one, so that at most two segments overlap anywhere.
///
/// # Errors
/// Fails on a stride for an unsegmented model or out of range.
pub fn check_segmenting(config: &Config, segmenting: Segmenting) -> Result<()> {
    if let Some(stride) = segmenting.stride {
        let length = config
            .segment_length()?
            .context("this model codes the whole file as one segment, so it has no stride")?;
        ensure!(
            (length.div_ceil(2)..=length).contains(&stride),
            "segment stride {stride} is out of range ({}..={length})",
            length.div_ceil(2)
        );
    }
    if segmenting.crossfade != Crossfade::Triangle {
        ensure!(
            config.segment_length()?.is_some(),
            "this model codes the whole file as one segment, so it has no cross-fade"
        );
    }
    Ok(())
}

/// The edge-avoiding cross-fade for an overlap of `ov` samples.
///
/// Within the overlap, at `t` samples from the new segment's start, the
/// output is the old segment for `t < lo`, a linear ramp from old to new over
/// `lo..hi`, and the new segment from `hi` on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRamp {
    /// Overlap length in samples.
    pub overlap: usize,
    /// First sample of the ramp.
    pub lo: usize,
    /// First sample past the ramp.
    pub hi: usize,
}

impl EdgeRamp {
    /// The ramp for segments of `length` samples starting `stride` apart.
    #[must_use]
    pub const fn new(length: usize, stride: usize) -> Self {
        let overlap = length.saturating_sub(stride);
        let hi = if overlap > 2 * EDGE_TAIL {
            overlap - EDGE_TAIL
        } else {
            overlap
        };
        let lo = {
            let head = hi.saturating_sub(EDGE_RAMP);
            if head < EDGE_HEAD { head } else { EDGE_HEAD }
        };
        Self { overlap, lo, hi }
    }

    /// Weight of the new segment at `t` samples from its start (`t < overlap`).
    #[must_use]
    pub fn new_weight(&self, t: usize) -> f64 {
        if t < self.lo {
            0.0
        } else if t < self.hi {
            to_f64(t - self.lo + 1) / to_f64(self.hi - self.lo + 1)
        } else {
            1.0
        }
    }

    /// Weight of a segment at `t` samples from its own start, when segments
    /// start `stride` apart: its head ramps in (unless it is the first one)
    /// and its tail ramps out.
    #[must_use]
    pub fn segment_weight(&self, t: usize, stride: usize, first: bool) -> f64 {
        let head = if !first && t < self.overlap {
            self.new_weight(t)
        } else {
            1.0
        };
        let tail = if t >= stride {
            1.0 - self.new_weight(t - stride)
        } else {
            1.0
        };
        head * tail
    }
}

/// Removes the pre-skip and end trim from decoded samples and, for loops,
/// folds the pre-roll into the end so that the last sample runs into the
/// first.
///
/// Decoded samples go in in order and the final ones come out. For a loop of
/// `n` samples with pre-skip `ps`, the fold length is
/// `f = min(ps, n, sample_rate / 10)`, and for `t < f`
/// `out[n - f + t] = (1 - a) * raw[ps + n - f + t] + a * raw[ps - f + t]`
/// with `a = (t + 1) / (f + 1)`: the end of the loop cross-fades into the
/// decoded copy of the audio that precedes the loop's start, so `out[n - 1]`
/// is followed by `out[0]` exactly as `raw[ps - 1]` is by `raw[ps]`.
#[derive(Debug, Clone)]
pub struct Trimmer {
    pre_skip: usize,
    length: usize,
    fold: usize,
    channels: usize,
    /// Raw samples taken so far.
    taken: usize,
    /// The `fold` raw samples just before the pre-skip ends, `[sample][channel]`.
    preroll: Vec<f32>,
}

impl Trimmer {
    /// A trimmer for `pre_skip + length + end_trim` decoded samples. `looped`
    /// enables the fold, whose length depends on `sample_rate`.
    ///
    /// # Errors
    /// Fails when `channels` is zero.
    pub fn new(
        pre_skip: usize,
        length: usize,
        looped: bool,
        sample_rate: usize,
        channels: usize,
    ) -> Result<Self> {
        ensure!(channels > 0, "trimmer needs at least one channel");
        let fold = if looped {
            pre_skip.min(length).min(sample_rate / 10)
        } else {
            0
        };
        Ok(Self {
            pre_skip,
            length,
            fold,
            channels,
            taken: 0,
            preroll: Vec::with_capacity(fold * channels),
        })
    }

    /// Samples per channel of the trimmed output.
    #[must_use]
    pub const fn output_length(&self) -> usize {
        self.length
    }

    /// Length of the loop fold, 0 for files that do not loop.
    #[must_use]
    pub const fn fold_length(&self) -> usize {
        self.fold
    }

    /// Whether the trimmer changes nothing (no pre-skip, no fold).
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        self.pre_skip == 0 && self.fold == 0
    }

    /// Takes the next decoded samples and returns the output samples they complete.
    ///
    /// # Errors
    /// Fails on a channel mismatch.
    pub fn push(&mut self, raw: &Seq) -> Result<Seq> {
        let channels = self.channels;
        ensure!(raw.channels() == channels, "trimmer channel mismatch");
        let (ps, n, f) = (self.pre_skip, self.length, self.fold);
        let mut out = Vec::with_capacity(raw.data().len());
        for row in raw.data().chunks_exact(channels) {
            let r = self.taken;
            self.taken += 1;
            if r < ps {
                if r + f >= ps {
                    self.preroll.extend_from_slice(row);
                }
            } else if r < ps + n - f {
                out.extend_from_slice(row);
            } else if r < ps + n {
                let t = r - (ps + n - f);
                let a = to_f64(t + 1) / to_f64(f + 1);
                let pre = self
                    .preroll
                    .get(t * channels..(t + 1) * channels)
                    .ok_or_else(|| anyhow::anyhow!("loop fold is missing its pre-roll"))?;
                out.extend(
                    row.iter()
                        .zip(pre)
                        .map(|(&v, &p)| narrow((1.0 - a).mul_add(f64::from(v), a * f64::from(p)))),
                );
            }
        }
        let len = out.len() / channels;
        Seq::from_vec(len, channels, out)
    }

    /// Checks that exactly the expected number of samples went through.
    ///
    /// # Errors
    /// Fails when fewer than `pre_skip + length` samples were pushed.
    pub fn finish(self) -> Result<()> {
        ensure!(
            self.taken >= self.pre_skip + self.length,
            "decoded {} samples, the file needs {}",
            self.taken,
            self.pre_skip + self.length
        );
        Ok(())
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a blend of two f32 samples is in f32 range"
)]
const fn narrow(v: f64) -> f32 {
    v as f32
}

/// The audio a loop encode codes: the loop with `pre` samples of its own end
/// before it and `post` samples of its own start after it (wrapping around as
/// often as needed for short loops).
///
/// # Errors
/// Fails on empty audio.
pub fn wrap_loop(audio: &Seq, pre: usize, post: usize) -> Result<Seq> {
    let n = audio.len();
    ensure!(n > 0, "cannot loop empty audio");
    let channels = audio.channels();
    let total = pre + n + post;
    let shift = n - pre % n;
    let mut data = Vec::with_capacity(total * channels);
    for i in 0..total {
        data.extend_from_slice(audio.row((i + shift) % n));
    }
    Seq::from_vec(total, channels, data)
}

/// Pre-roll and post-roll for a loop encode at `sample_rate`.
///
/// 150 ms of the loop's end go before it (the fold uses the last 100 ms of
/// them, clear of the damaged first 30 ms of the first segment) and 50 ms of
/// its start after it (clear of the damaged last 10 ms of the final segment).
#[must_use]
pub const fn loop_padding(sample_rate: usize) -> (usize, usize) {
    (sample_rate * 3 / 20, sample_rate / 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_ramp_matches_the_spec() {
        let r = EdgeRamp::new(48_000, 45_000);
        assert_eq!((r.overlap, r.lo, r.hi), (3000, 1440, 2520));
        let r = EdgeRamp::new(48_000, 47_520);
        assert_eq!((r.overlap, r.lo, r.hi), (480, 240, 480));
        let r = EdgeRamp::new(48_000, 46_560);
        assert_eq!((r.overlap, r.lo, r.hi), (1440, 720, 960));
        assert!(r.new_weight(719).abs() < 1e-12);
        assert!((r.new_weight(720) - 1.0 / 241.0).abs() < 1e-12);
        assert!((r.new_weight(959) - 240.0 / 241.0).abs() < 1e-12);
        assert!((r.new_weight(960) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trimmer_trims_and_folds() -> Result<()> {
        let raw = Seq::from_vec(20, 1, (0..20_u8).map(f32::from).collect())?;
        // Plain trim: 3 before, 12 kept, 5 after.
        let mut t = Trimmer::new(3, 12, false, 48_000, 1)?;
        let mut out = t.push(&raw.slice_rows(0, 7))?.into_data();
        out.extend(t.push(&raw.slice_rows(7, 20))?.into_data());
        t.finish()?;
        assert_eq!(out, (3..15_u8).map(f32::from).collect::<Vec<_>>());
        // Loop fold at 40 Hz: fold = min(3, 12, 4) = 3.
        let mut t = Trimmer::new(3, 12, true, 40, 1)?;
        assert_eq!(t.fold_length(), 3);
        let out = t.push(&raw)?.into_data();
        assert_eq!(out.len(), 12);
        assert_eq!(
            &out[..9],
            &(3..12_u8).map(f32::from).collect::<Vec<_>>()[..]
        );
        // out[9 + k] = (1 - a) raw[12 + k] + a raw[k], a = (k + 1) / 4.
        for (k, &got) in (0..3_u8).zip(&out[9..]) {
            let a = (f64::from(k) + 1.0) / 4.0;
            let want = (1.0 - a).mul_add(f64::from(12 + k), a * f64::from(k));
            assert!((f64::from(got) - want).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn wrap_loop_wraps() -> Result<()> {
        let x = Seq::from_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0])?;
        let w = wrap_loop(&x, 6, 5)?;
        assert_eq!(
            w.data(),
            &[
                2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0
            ]
        );
        Ok(())
    }
}
