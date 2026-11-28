//! The declicker after the HQ decoder at low tiers.
//!
//! At one to [`MAX_Q`] codebooks, the HQ decoder's overlap-add can leave short
//! clicks where a frame's phase does not match its neighbours'. They fall in
//! the first quarter of a 512-sample frame. The declicker finds them and
//! rebuilds those samples from the audio around them.
//!
//! - **Detection.** The audio is cut into 128-sample windows, four per frame.
//!   A window is a hit when the energy of the mono mix's second difference
//!   is [`THRESHOLD_DB`] over the median of the windows within [`MEDIAN`]
//!   windows of it. Only a hit in a frame's first quarter is a candidate.
//! - **The grid gate.** Speech consonants and drum hits are hits too, but
//!   they fall in all four quarters alike. So a candidate is repaired only
//!   if, among the windows within [`SPAN`] windows of it (2 s), the first
//!   quarters hold at least [`RATIO`] times as many hits as the mean of the
//!   other three. In the file's first 2 s the gate looks ahead only as far
//!   as the file has gone, so playback starts without waiting 2 s.
//! - **The repair.** The [`GAP`] samples around the window are rebuilt per
//!   channel by Janssen's iterative autoregressive interpolation (Janssen,
//!   Veldhuis and Vries 1986): fit an order-[`ORDER`] predictor to
//!   [`CONTEXT`] samples either side and the current fill, then choose the
//!   gap samples that minimise the prediction error, [`ITERATIONS`] times.
//!
//! Each repair reads the decoder's output, not earlier repairs, so an output
//! sample depends only on the input from [`LOOKBACK`] samples before it to
//! about as many after it (window edges are where the stream starts and
//! ends; the gate's shorter look-ahead is set by the file's start, not the
//! stream's). The output is the same bit for bit however the input is split into
//! chunks, and a decode that starts [`LOOKBACK`] samples before a seek point
//! gives exactly the samples of a whole-file decode from that point.

use anyhow::{Result, ensure};

use crate::nn::Seq;

/// The highest tier (codebooks per frame) the declicker runs at.
pub const MAX_Q: usize = 5;
/// Samples per HQ frame.
const FRAME: usize = 512;
/// Samples per detection window.
const WIN: usize = 128;
/// Windows per frame.
const QUARTERS: usize = FRAME / WIN;
/// A window is a hit this far over its neighbourhood's median, in dB.
pub const THRESHOLD_DB: f64 = 3.0;
/// Windows per side of the median.
pub const MEDIAN: usize = 50;
/// Windows per side of the grid gate (2 s at 44.1 kHz).
pub const SPAN: usize = 689;
/// How many times more hits the first quarters need than the others.
pub const RATIO: f64 = 1.5;
/// Samples rebuilt around a click.
pub const GAP: usize = 192;
/// Order of the autoregressive model.
pub const ORDER: usize = 32;
/// Samples of context on each side of a gap.
pub const CONTEXT: usize = 1024;
/// Rounds of model fit and least-squares fill.
pub const ITERATIONS: usize = 4;
/// Samples of input before a sample that its output depends on: how far
/// before a seek point decoding must start.
pub const LOOKBACK: usize = (SPAN + MEDIAN + 2) * WIN + GAP;
/// Where the gap starts relative to its window.
const GAP_BEFORE: usize = GAP / 2 - WIN / 2;

/// Whether the declicker runs on a decode at `num_codebooks` per frame.
#[must_use]
pub const fn applies(num_codebooks: usize) -> bool {
    num_codebooks >= 1 && num_codebooks <= MAX_Q
}

/// A declicker over one stream of interleaved samples. Positions are in the
/// decoder's sample coordinates, where frame `f` starts at `512 f`.
#[derive(Debug, Clone)]
pub struct Declicker {
    channels: usize,
    /// Position of the first sample pushed.
    start: usize,
    /// The file's first window wholly inside it, which sets the gate's
    /// look-ahead near the start.
    origin_window: usize,
    /// Position of `input[0]`.
    base: usize,
    /// The decoder's samples from `base`, `[sample][channel]`.
    input: Vec<f32>,
    /// Position of `output[0]`: the next sample to return.
    emitted: usize,
    /// The repaired samples from `emitted`.
    output: Vec<f32>,
    /// The first window wholly inside the stream (with the two samples
    /// before it that its second difference needs).
    first_window: usize,
    /// Window of `energy[0]` and `hit[0]`.
    window_base: usize,
    /// Each window's second-difference energy in dB, from `window_base`.
    energy: Vec<f64>,
    /// Whether each window is a hit, from `window_base`; shorter than
    /// `energy` until the windows after it are known.
    hit: Vec<bool>,
    /// The next first-quarter window to decide.
    candidate: usize,
    /// The stream's end, once known.
    end: Option<usize>,
}

impl Declicker {
    /// A declicker for `channels` interleaved channels of a file whose first
    /// sample is at position `origin`, for a stream whose first sample is at
    /// position `start` (`origin` itself unless the stream starts at a seek).
    #[must_use]
    pub fn new(channels: usize, origin: usize, start: usize) -> Self {
        let first_window = (start + 2).div_ceil(WIN);
        Self {
            channels: channels.max(1),
            start,
            origin_window: (origin + 2).div_ceil(WIN),
            base: start,
            input: Vec::new(),
            emitted: start,
            output: Vec::new(),
            first_window,
            window_base: first_window,
            energy: Vec::new(),
            hit: Vec::new(),
            candidate: first_window.next_multiple_of(QUARTERS),
            end: None,
        }
    }

    /// Takes the next input samples and returns the output samples that no
    /// later input can change.
    ///
    /// # Errors
    /// Fails when `audio` has another channel count.
    pub fn push(&mut self, audio: &Seq) -> Result<Seq> {
        ensure!(
            audio.channels() == self.channels,
            "the declicker runs on {} channels, not {}",
            self.channels,
            audio.channels()
        );
        self.input.extend_from_slice(audio.data());
        self.output.extend_from_slice(audio.data());
        self.measure();
        self.run()
    }

    /// Ends the stream and returns the samples still held back.
    ///
    /// # Errors
    /// Fails only on an internal size mismatch.
    pub fn finish(mut self) -> Result<Seq> {
        self.end = Some(self.pushed());
        self.run()
    }

    /// Position after the last sample pushed.
    const fn pushed(&self) -> usize {
        self.base + self.input.len() / self.channels
    }

    /// The mono mix of the input at position `p`.
    fn mono(&self, p: usize) -> f64 {
        let c = self.channels;
        let at = (p - self.base) * c;
        let sum = self.input[at..at + c]
            .iter()
            .fold(0.0_f64, |s, &v| s + f64::from(v));
        #[expect(clippy::cast_precision_loss, reason = "a channel count")]
        let mean = sum / c as f64;
        mean
    }

    /// Measures the energy of every window that the pushed input completes.
    fn measure(&mut self) {
        let end = self.pushed();
        loop {
            let w = self.window_base + self.energy.len();
            if (w + 1) * WIN > end {
                break;
            }
            let lo = w * WIN;
            let energy = (lo..lo + WIN).fold(0.0_f64, |e, t| {
                let h = self.mono(t) - 2.0 * self.mono(t - 1) + self.mono(t - 2);
                e + h * h
            });
            self.energy.push(10.0 * (energy + 1e-14).log10());
        }
    }

    /// Windows measured so far: `first_window..last_window`.
    const fn last_window(&self) -> usize {
        self.window_base + self.energy.len()
    }

    /// Marks the hits of every window whose neighbourhood is known.
    fn mark(&mut self) {
        let last = self.last_window();
        loop {
            let w = self.window_base + self.hit.len();
            if w >= last || (self.end.is_none() && w + MEDIAN >= last) {
                break;
            }
            let mut around: Vec<f64> = (0..=2 * MEDIAN)
                .map(|j| {
                    let v = (w + j).saturating_sub(MEDIAN).clamp(self.first_window, last - 1);
                    self.energy[v - self.window_base]
                })
                .collect();
            let (_, median, _) = around.select_nth_unstable_by(MEDIAN, f64::total_cmp);
            let over = self.energy[w - self.window_base] - *median;
            self.hit.push(over > THRESHOLD_DB);
        }
    }

    /// Decides every candidate whose gate and context are known, then
    /// returns the samples before the next undecided one.
    fn run(&mut self) -> Result<Seq> {
        self.mark();
        let marked = self.window_base + self.hit.len();
        let pushed = self.pushed();
        loop {
            let w = self.candidate;
            let gap_end = w * WIN + GAP_BEFORE + GAP;
            let known = match self.end {
                Some(_) => true,
                None => w + self.ahead(w) < marked && gap_end + CONTEXT <= pushed,
            };
            if !known || (self.end.is_some() && w >= self.last_window()) {
                break;
            }
            if w < marked && self.hit[w - self.window_base] && self.gate(w) {
                self.repair(w);
            }
            self.candidate += QUARTERS;
        }
        let ready = match self.end {
            Some(end) => end,
            None => (self.candidate * WIN - GAP_BEFORE).clamp(self.emitted, pushed),
        };
        self.emit(ready)
    }

    /// How many windows past `w` the gate reads: [`SPAN`], or in the file's
    /// first [`SPAN`] windows as many as come before `w`.
    fn ahead(&self, w: usize) -> usize {
        SPAN.min(w.saturating_sub(self.origin_window))
    }

    /// Whether, around first-quarter window `w`, the hits bunch in the first
    /// quarters.
    fn gate(&self, w: usize) -> bool {
        let lo = w.saturating_sub(SPAN).max(self.first_window);
        let hi = (w + self.ahead(w) + 1).min(self.window_base + self.hit.len());
        let mut counts = [0_u32; QUARTERS];
        for v in lo..hi {
            if self.hit[v - self.window_base] {
                counts[v % QUARTERS] += 1;
            }
        }
        let others = counts[1..].iter().map(|&n| f64::from(n)).sum::<f64>();
        #[expect(clippy::cast_precision_loss, reason = "three quarters")]
        let mean = others / (QUARTERS - 1) as f64;
        f64::from(counts[0]) >= RATIO * mean.max(1.0)
    }

    /// Rebuilds the gap around window `w` in every channel, when its context
    /// is inside the stream.
    fn repair(&mut self, w: usize) {
        let lo = w * WIN - GAP_BEFORE;
        let Some(first) = lo.checked_sub(CONTEXT) else {
            return;
        };
        let last = lo + GAP + CONTEXT;
        if first < self.start || self.end.is_some_and(|end| last > end) {
            return;
        }
        let c = self.channels;
        for ch in 0..c {
            let block: Vec<f64> = (first..last)
                .map(|p| f64::from(self.input[(p - self.base) * c + ch]))
                .collect();
            let fill = janssen(block);
            for (j, v) in fill.into_iter().enumerate() {
                #[expect(clippy::cast_possible_truncation, reason = "back to the output's f32")]
                let v = v as f32;
                self.output[(lo + j - self.emitted) * c + ch] = v;
            }
        }
    }

    /// Returns the samples up to position `ready` and drops what no later
    /// decision needs.
    fn emit(&mut self, ready: usize) -> Result<Seq> {
        let c = self.channels;
        let n = ready - self.emitted;
        let out: Vec<f32> = self.output.drain(..n * c).collect();
        self.emitted = ready;
        // Repairs read CONTEXT samples before their gap, which starts at or
        // after `emitted`; new windows read the two samples before them.
        let keep_from = self.emitted.saturating_sub(CONTEXT + 2).max(self.base);
        drop(self.input.drain(..(keep_from - self.base) * c));
        self.base = keep_from;
        // The gate and the median look back SPAN + MEDIAN windows.
        let keep_window = self
            .candidate
            .saturating_sub(SPAN + MEDIAN)
            .clamp(self.window_base, self.window_base + self.hit.len());
        let cut = keep_window - self.window_base;
        drop(self.energy.drain(..cut));
        drop(self.hit.drain(..cut));
        self.window_base = keep_window;
        Seq::from_vec(n, c, out)
    }
}

/// Declicks a whole stream, of a file starting at position `origin`, whose
/// first sample is at position `start`: the same samples as pushing it in any
/// chunks.
///
/// # Errors
/// Fails only on an internal size mismatch.
pub fn declick(audio: &Seq, origin: usize, start: usize) -> Result<Seq> {
    let mut declicker = Declicker::new(audio.channels(), origin, start);
    let head = declicker.push(audio)?;
    Seq::concat(vec![head, declicker.finish()?])
}

/// The order-[`ORDER`] linear predictor of `x` (autocorrelation method,
/// Levinson-Durbin): `x[t] ≈ sum_k a[k] x[t - 1 - k]`.
fn predictor(x: &[f64]) -> [f64; ORDER] {
    let mut r = [0.0_f64; ORDER + 1];
    for (k, r) in r.iter_mut().enumerate() {
        *r = x[k..].iter().zip(x).fold(0.0, |s, (a, b)| s + a * b);
    }
    r[0] = r[0] * 1.0001 + 1e-12;
    let mut a = [0.0_f64; ORDER];
    let mut error = r[0];
    for i in 0..ORDER {
        let acc = (0..i).fold(r[i + 1], |s, j| s - a[j] * r[i - j]);
        let k = acc / error;
        let previous = a;
        a[i] = k;
        for j in 0..i {
            a[j] = previous[j] - k * previous[i - 1 - j];
        }
        error *= 1.0 - k * k;
    }
    a
}

/// `n` samples predicted on from the end of `history` by `a`.
fn extrapolate(history: &[f64], a: &[f64; ORDER], n: usize) -> Vec<f64> {
    let mut buffer = history[history.len() - ORDER..].to_vec();
    for _ in 0..n {
        let t = buffer.len();
        let v = a
            .iter()
            .enumerate()
            .fold(0.0, |s, (k, &a)| s + a * buffer[t - 1 - k]);
        buffer.push(v);
    }
    buffer.split_off(ORDER)
}

/// Janssen's interpolation of the middle [`GAP`] samples of `block`
/// ([`CONTEXT`] samples, the gap, [`CONTEXT`] samples).
fn janssen(mut block: Vec<f64>) -> Vec<f64> {
    let (g0, g1) = (CONTEXT, CONTEXT + GAP);
    // Start from a cross-fade of the forward and backward predictions.
    let forward = extrapolate(&block[..g0], &predictor(&block[..g0]), GAP);
    let after: Vec<f64> = block[g1..].iter().rev().copied().collect();
    let mut backward = extrapolate(&after, &predictor(&after), GAP);
    backward.reverse();
    #[expect(clippy::cast_precision_loss, reason = "sample offsets")]
    for j in 0..GAP {
        let w = 0.5 - 0.5 * (std::f64::consts::PI * j as f64 / (GAP - 1) as f64).cos();
        block[g0 + j] = (1.0 - w) * forward[j] + w * backward[j];
    }
    for _ in 0..ITERATIONS {
        let a = predictor(&block);
        let mut c = [0.0_f64; ORDER + 1];
        c[0] = 1.0;
        for (c, a) in c[1..].iter_mut().zip(a) {
            *c = -a;
        }
        // Prediction error e[r] = sum_k c[k] x[r - k] for r in g0..g1 + ORDER;
        // `known[r]` is its part from the samples outside the gap.
        let rows = GAP + ORDER;
        let known: Vec<f64> = (g0..g0 + rows)
            .map(|r| {
                (0..=ORDER)
                    .filter(|&k| !(g0..g1).contains(&(r - k)))
                    .fold(0.0, |s, k| s + c[k] * block[r - k])
            })
            .collect();
        // Normal equations: (A^T A) u = -A^T known, where A^T A is the
        // banded Toeplitz matrix of c's autocorrelation.
        let auto: Vec<f64> = (0..=ORDER)
            .map(|d| (0..=ORDER - d).fold(0.0, |s, k| s + c[k] * c[k + d]))
            .collect();
        let rhs: Vec<f64> = (0..GAP)
            .map(|i| -(0..=ORDER).fold(0.0, |s, k| s + c[k] * known[i + k]))
            .collect();
        let fill = solve_banded(&auto, rhs);
        block[g0..g1].copy_from_slice(&fill);
    }
    block[g0..g1].to_vec()
}

/// Solves `T u = rhs` for the symmetric positive definite banded Toeplitz
/// matrix `T[i][j] = auto[|i - j|]` (zero past `auto`), by Cholesky.
fn solve_banded(auto: &[f64], mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    let band = auto.len() - 1;
    // l[i][d]: row i, column i - d.
    let mut l = vec![[0.0_f64; ORDER + 1]; n];
    for i in 0..n {
        for d in (0..=band.min(i)).rev() {
            let j = i - d;
            let mut s = auto[d];
            for e in 1..=band.min(j) {
                let k = j - e;
                if i - k <= band {
                    s -= l[i][i - k] * l[j][e];
                }
            }
            if d == 0 {
                l[i][0] = s.max(1e-300).sqrt();
            } else {
                l[i][d] = s / l[j][0];
            }
        }
    }
    for i in 0..n {
        let s = (1..=band.min(i)).fold(rhs[i], |s, d| s - l[i][d] * rhs[i - d]);
        rhs[i] = s / l[i][0];
    }
    for i in (0..n).rev() {
        let s = (1..=band.min(n - 1 - i)).fold(rhs[i], |s, d| s - l[i + d][d] * rhs[i + d]);
        rhs[i] = s / l[i][0];
    }
    rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone with a click at the first quarter of every frame.
    fn clicky(len: usize, start: usize) -> Seq {
        let mut data = Vec::with_capacity(len * 2);
        for t in 0..len {
            let p = start + t;
            #[expect(clippy::cast_precision_loss, reason = "test signal")]
            let mut v = 0.3 * (p as f32 * 0.031).sin() + 0.1 * (p as f32 * 0.17).sin();
            if p % FRAME == 40 {
                v += 0.2;
            }
            data.extend([v, 0.8 * v]);
        }
        Seq::from_vec(len, 2, data).unwrap_or_else(|_| Seq::zeros(0, 2))
    }

    fn same_bits(a: &Seq, b: &Seq) -> bool {
        a.len() == b.len()
            && a
                .data()
                .iter()
                .zip(b.data())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn solves_a_banded_toeplitz_system() {
        let auto = [4.0, -1.0, 0.5];
        let want = vec![1.0, -2.0, 0.5, 3.0, 0.0, -1.0];
        let n = want.len();
        let rhs: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| i.abs_diff(j) < auto.len())
                    .map(|j| auto[i.abs_diff(j)] * want[j])
                    .sum()
            })
            .collect();
        let got = solve_banded(&auto, rhs);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "{got:?}");
        }
    }

    #[test]
    fn rebuilds_a_gap_in_a_tone() {
        #[expect(clippy::cast_precision_loss, reason = "test signal")]
        let tone: Vec<f64> = (0..2 * CONTEXT + GAP)
            .map(|t| (t as f64 * 0.05).sin() + 0.3 * (t as f64 * 0.21).cos())
            .collect();
        let mut block = tone.clone();
        for v in &mut block[CONTEXT..CONTEXT + GAP] {
            *v = 0.0;
        }
        let fill = janssen(block);
        let err = fill
            .iter()
            .zip(&tone[CONTEXT..])
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        // The predictor's slight regularisation keeps it from being exact.
        assert!(err < 1e-2, "max error {err}");
    }

    #[test]
    fn removes_grid_clicks_and_chunks_give_the_whole_signal() -> Result<()> {
        let start = 300;
        let x = clicky(6 * 44_100, start);
        let whole = declick(&x, start, start)?;
        assert_eq!(whole.len(), x.len());
        // The clicks in the middle are gone.
        let mid = 3 * 44_100 / FRAME * FRAME + 40 - start;
        let before = (x.row(mid)[0] - x.row(mid - 1)[0]).abs();
        let after = (whole.row(mid)[0] - whole.row(mid - 1)[0]).abs();
        assert!(after < before / 4.0, "click {before} -> {after}");
        for chunk in [1, 97, 4096, 50_000] {
            let mut d = Declicker::new(2, start, start);
            let mut parts = Vec::new();
            let mut t = 0;
            while t < x.len() {
                let e = (t + chunk).min(x.len());
                parts.push(d.push(&x.slice_rows(t, e))?);
                t = e;
            }
            parts.push(d.finish()?);
            assert!(same_bits(&whole, &Seq::concat(parts)?), "chunk {chunk}");
        }
        // A stream starting LOOKBACK samples before a point gives the whole
        // signal's samples from there.
        let point = 4 * 44_100 + 77;
        let first = point - LOOKBACK;
        let tail = declick(&x.slice_rows(first, x.len()), start, start + first)?;
        assert!(same_bits(
            &tail.slice_rows(LOOKBACK, tail.len()),
            &whole.slice_rows(point, whole.len())
        ));
        Ok(())
    }

    #[test]
    fn starts_playing_without_the_full_look_ahead() -> Result<()> {
        let start = 300;
        let x = clicky(6 * 44_100, start);
        let whole = declick(&x, start, start)?;
        // Half a second in, a third of it can already be played: in the
        // first 2 s the gate reads as far ahead as the file has gone, so the
        // output keeps up with half the input.
        let mut d = Declicker::new(2, start, start);
        let half = 44_100 / 2;
        let head = d.push(&x.slice_rows(0, half))?;
        assert!(head.len() > half / 3, "only {} of {half} samples out", head.len());
        assert!(same_bits(&head, &whole.slice_rows(0, head.len())));
        // Clicks early in the file are still repaired.
        let early = 3 * FRAME + 40 - start + 20 * FRAME;
        let before = (x.row(early)[0] - x.row(early - 1)[0]).abs();
        let after = (whole.row(early)[0] - whole.row(early - 1)[0]).abs();
        assert!(after < before / 4.0, "click {before} -> {after}");
        Ok(())
    }

    #[test]
    fn leaves_clicks_off_the_grid_alone() -> Result<()> {
        // Clicks in every quarter alike, as speech and drums give.
        let len = 4 * 44_100;
        let data: Vec<f32> = (0..len)
            .flat_map(|t| {
                #[expect(clippy::cast_precision_loss, reason = "test signal")]
                let mut v = 0.3 * (t as f32 * 0.031).sin();
                if t % WIN == 40 {
                    v += 0.2;
                }
                [v, v]
            })
            .collect();
        let x = Seq::from_vec(len, 2, data)?;
        assert!(same_bits(&declick(&x, 0, 0)?, &x));
        Ok(())
    }
}
