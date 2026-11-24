//! Parametric stereo for the HQ codec, in the style of HE-AAC v2
//! (`distill/INTEGRATION.md`, M8; prototype and measurements:
//! `tools/pstereo_proto.py`).
//!
//! The file codes the mid `(L + R) / 2` as one HQ channel, plus, per band and
//! per parameter set of `frames` HQ frames, the intensity difference (IID)
//! and the coherence (ICC) of L and R.
//!
//! - **Analysis**: a 1024-point sine-window STFT with hop 512 (one HQ frame);
//!   frame `t` covers samples `[(t - 1) * 512, (t + 1) * 512)`, for
//!   `t = 0..=frames`. Each STFT frame takes the parameters of HQ frame
//!   `min(t, frames - 1)`.
//! - **Quantisation**: IID on HE-AAC's 15-level grid (dB). ICC on HE-AAC's
//!   8 levels plus 0.997, 0.99 and 0.97, nearest by the side-to-mid level.
//!   At equal powers HE-AAC's first step under 1 (0.937) already puts the
//!   side 15 dB under the mid, and centred mixes have sides 20–30 dB down in
//!   many bands. The extra levels cut the side's log-mel error from 5.28 to
//!   4.75 for 0.025 kbps.
//! - **Upmix**: `L = M + S` and `R = M - S` with `S = g M + h D` per band, so
//!   `(L + R) / 2` is exactly the decoded mid and no overall phase is needed.
//!   `g` and `h` give L and R the coded powers and correlation. `D` is the
//!   mid through a velvet-noise FIR (50 ms, 75 taps), ducked per band and
//!   STFT frame so that its energy never exceeds the mid's (transients).
//!   No IPD: in the prototype it cost 0.15 kbps and made the side worse.
//! - **Coding**: per page, per set, IID then ICC. The first set of a page
//!   codes frequency deltas (the first band against the grid's centre).
//!   Later sets write one flag bit (1 = time deltas against the previous
//!   set), then the deltas, with static canonical Huffman codes. Every page's
//!   parameters decode on their own.
//!
//! The decoder's output near a page edge (within 512 samples) uses the
//! neighbouring page's parameters, and `D` reaches 54 ms back into the mid:
//! like the HQ decoder's own context, a seek decodes a little earlier.

use std::ops::Range;

use anyhow::{Context as _, Result, bail, ensure};

use crate::dac::HOP;

/// STFT size.
const N: usize = 1024;
/// STFT bins, 0 to Nyquist.
const BINS: usize = N / 2 + 1;

/// Band edges in STFT bins of the 20-band layout (about 129 Hz wide to
/// 1 kHz, then widening as HE-AAC's 20-band layout).
const EDGES_20: [usize; 21] = [
    0, 3, 6, 9, 12, 15, 18, 21, 24, 32, 40, 48, 56, 64, 72, 88, 112, 144, 184, 280, BINS,
];
/// Band edges of the 10-band layout (every other edge of the 20).
const EDGES_10: [usize; 11] = [0, 6, 12, 18, 24, 40, 56, 72, 112, 184, BINS];

/// IID levels in dB (HE-AAC's default grid).
const IID_DB: [f64; 15] = [
    -25.0, -18.0, -14.0, -10.0, -7.0, -4.0, -2.0, 0.0, 2.0, 4.0, 7.0, 10.0, 14.0, 18.0, 25.0,
];
/// Index of 0 dB.
const IID_CENTRE: u8 = 7;
/// ICC levels: HE-AAC's 8 plus three near 1.
const ICC: [f64; 11] = [
    1.0, 0.997, 0.99, 0.97, 0.937, 0.841_18, 0.600_92, 0.367_64, 0.0, -0.589, -1.0,
];

/// Huffman code lengths of IID frequency deltas, -14 to 14 (fitted on the
/// prototype's 15 clips, add-one smoothed).
const IID_FREQ: [u8; 29] = [
    17, 17, 17, 17, 15, 12, 12, 10, 8, 7, 6, 5, 4, 3, 1, 3, 4, 5, 6, 7, 7, 9, 12, 13, 15, 17, 17,
    17, 17,
];
/// Code lengths of IID time deltas, -14 to 14.
const IID_TIME: [u8; 29] = [
    15, 15, 15, 15, 14, 13, 10, 10, 9, 6, 6, 5, 4, 3, 1, 3, 4, 5, 6, 7, 9, 10, 12, 12, 15, 15, 15,
    15, 14,
];
/// Code lengths of ICC frequency deltas, -10 to 10.
const ICC_FREQ: [u8; 21] = [
    14, 14, 11, 12, 10, 9, 7, 6, 4, 3, 1, 3, 4, 5, 6, 6, 7, 7, 6, 8, 13,
];
/// Code lengths of ICC time deltas, -10 to 10.
const ICC_TIME: [u8; 21] = [
    16, 16, 14, 13, 12, 9, 9, 6, 5, 3, 1, 2, 4, 7, 9, 10, 12, 13, 13, 16, 16,
];

/// The decorrelator: `(delay, gain)` taps of a velvet-noise FIR, 50 ms from
/// a 3 ms start, 1500 taps per second, decaying 10 dB, unit energy.
const VELVET: [(usize, f64); 75] = [
    (149, 1.829_655_0e-1),
    (186, 1.794_647_6e-1),
    (212, -1.770_449_3e-1),
    (226, 1.757_554_8e-1),
    (258, 1.728_433_4e-1),
    (303, 1.688_295_9e-1),
    (308, 1.683_894_1e-1),
    (361, 1.637_934_9e-1),
    (389, -1.614_163_1e-1),
    (409, -1.597_394_9e-1),
    (434, 1.576_679_2e-1),
    (463, -1.552_985_4e-1),
    (492, 1.529_647_7e-1),
    (526, 1.502_732_5e-1),
    (557, -1.478_605_1e-1),
    (588, 1.454_865_1e-1),
    (630, -1.423_308_0e-1),
    (654, -1.405_583_7e-1),
    (678, -1.388_080_2e-1),
    (718, 1.359_390_6e-1),
    (726, 1.353_724_2e-1),
    (753, 1.334_774_0e-1),
    (796, 1.305_140_2e-1),
    (809, 1.296_311_4e-1),
    (838, -1.276_830_9e-1),
    (881, -1.248_483_5e-1),
    (909, 1.230_363_9e-1),
    (951, -1.203_676_5e-1),
    (973, -1.189_929_2e-1),
    (999, -1.173_884_7e-1),
    (1028, 1.156_243_9e-1),
    (1050, -1.143_038_4e-1),
    (1073, 1.129_393_8e-1),
    (1107, 1.109_521_3e-1),
    (1151, 1.084_322_1e-1),
    (1166, -1.075_862_9e-1),
    (1200, 1.056_932_4e-1),
    (1219, 1.046_499_0e-1),
    (1272, -1.017_936_5e-1),
    (1282, -1.012_635_4e-1),
    (1315, -9.953_369_0e-2),
    (1362, 9.712_085_7e-2),
    (1381, -9.616_213_9e-2),
    (1420, -9.422_379_2e-2),
    (1443, 9.309_903_1e-2),
    (1476, -9.150_865_4e-2),
    (1486, 9.103_210_7e-2),
    (1529, -8.901_107_1e-2),
    (1557, 8.771_923_2e-2),
    (1597, -8.590_620_3e-2),
    (1612, -8.523_602_0e-2),
    (1648, 8.364_883_6e-2),
    (1662, -8.303_960_9e-2),
    (1701, 8.136_577_4e-2),
    (1728, 8.022_677_0e-2),
    (1753, -7.918_635_9e-2),
    (1801, -7.722_644_1e-2),
    (1818, 7.654_399_9e-2),
    (1864, -7.472_747_5e-2),
    (1883, 7.398_981_1e-2),
    (1913, 7.283_987_6e-2),
    (1943, -7.170_781_3e-2),
    (1974, -7.055_649_5e-2),
    (1988, 7.004_262_2e-2),
    (2026, -6.866_661_1e-2),
    (2049, 6.784_692_9e-2),
    (2083, 6.665_311_4e-2),
    (2104, -6.592_627_5e-2),
    (2158, 6.409_344_9e-2),
    (2166, 6.382_628_7e-2),
    (2209, 6.240_925_7e-2),
    (2227, 6.182_546_4e-2),
    (2273, 6.035_823_7e-2),
    (2297, 5.960_660_2e-2),
    (2311, 5.917_247_9e-2),
];

/// Band layout of the parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bands {
    /// 20 bands (the default).
    Twenty,
    /// 10 bands: about half the rate.
    Ten,
}

/// Parametric stereo settings, stored in the HQ header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Band layout.
    pub bands: Bands,
    /// HQ frames per parameter set: 1, 2, 4, 8 or 16.
    pub frames: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Config {
    /// 20 bands, a set every 8 frames (10.8 per second): about 1 kbps.
    pub const DEFAULT: Self = Self {
        bands: Bands::Twenty,
        frames: 8,
    };

    /// Settings from a band count (20 or 10) and frames per set.
    ///
    /// # Errors
    /// Fails on other band counts, or frames that are not 1, 2, 4, 8 or 16.
    pub fn new(bands: usize, frames: usize) -> Result<Self> {
        let bands = match bands {
            20 => Bands::Twenty,
            10 => Bands::Ten,
            other => bail!("{other} parametric stereo bands; there are layouts of 20 and 10"),
        };
        ensure!(
            frames.is_power_of_two() && frames <= 16,
            "{frames} frames per parameter set; use 1, 2, 4, 8 or 16"
        );
        Ok(Self { bands, frames })
    }

    /// Band count.
    #[must_use]
    pub const fn band_count(self) -> usize {
        self.edges().len() - 1
    }

    const fn edges(self) -> &'static [usize] {
        match self.bands {
            Bands::Twenty => &EDGES_20,
            Bands::Ten => &EDGES_10,
        }
    }

    /// The header byte: bits 0-1 the band layout (0 = 20, 1 = 10), bits 2-4
    /// log2 of the frames per set.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        let layout = match self.bands {
            Bands::Twenty => 0,
            Bands::Ten => 1,
        };
        // `frames` is at most 16, so its log2 fits 3 bits.
        #[expect(clippy::cast_possible_truncation, reason = "log2 of at most 16")]
        let log = self.frames.trailing_zeros() as u8;
        layout | (log << 2)
    }

    /// Reads the header byte.
    ///
    /// # Errors
    /// Fails on unknown layouts or bits.
    pub fn from_byte(byte: u8) -> Result<Self> {
        ensure!(
            byte & !0x1f == 0,
            "unknown parametric stereo bits {byte:#04x}"
        );
        let bands = match byte & 3 {
            0 => 20,
            1 => 10,
            other => bail!("unknown parametric stereo band layout {other}"),
        };
        Self::new(bands, 1 << (byte >> 2))
    }

    /// For messages: `20 bands, 8 frames per set`.
    #[must_use]
    pub fn describe(self) -> String {
        format!(
            "{} bands, {} frame{} per set",
            self.band_count(),
            self.frames,
            if self.frames == 1 { "" } else { "s" }
        )
    }
}

/// Quantised parameters of a whole stream: IID and ICC level indices,
/// `[set][band]` row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params {
    /// IID level per set and band (`0..15`).
    pub iid: Vec<u8>,
    /// ICC level per set and band (`0..11`).
    pub icc: Vec<u8>,
}

/// Per HQ frame its parameter set, and per page its range of sets.
fn frame_sets(spans: &[Range<usize>], per: usize) -> (Vec<usize>, Vec<Range<usize>>) {
    let mut of_frame = Vec::new();
    let mut pages = Vec::with_capacity(spans.len());
    let mut next = 0;
    for span in spans {
        let count = span.len().div_ceil(per);
        of_frame.extend((0..span.len()).map(|i| next + i / per));
        pages.push(next..next + count);
        next += count;
    }
    (of_frame, pages)
}

// ---------------------------------------------------------------- FFT

/// A 1024-point complex FFT and the sine window.
struct Fft {
    /// `e^{-2 pi i k / N}` for `k < N / 2`.
    twiddles: Vec<(f32, f32)>,
    /// Bit-reversed index of each position.
    reverse: Vec<usize>,
    /// `sin(pi (i + 0.5) / N)`: squared, it overlaps to 1 at hop N / 2.
    window: Vec<f32>,
}

impl Fft {
    fn new() -> Self {
        let n = crate::config::to_f64(N);
        #[expect(clippy::cast_possible_truncation, reason = "f32 twiddles and window")]
        let twiddles = (0..N / 2)
            .map(|k| {
                let a = -std::f64::consts::TAU * crate::config::to_f64(k) / n;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let bits = N.trailing_zeros();
        let reverse = (0..N)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        #[expect(clippy::cast_possible_truncation, reason = "f32 window")]
        let window = (0..N)
            .map(|i| (std::f64::consts::PI * (crate::config::to_f64(i) + 0.5) / n).sin() as f32)
            .collect();
        Self {
            twiddles,
            reverse,
            window,
        }
    }

    /// In-place forward transform.
    fn run(&self, re: &mut [f32], im: &mut [f32]) {
        for (i, &j) in self.reverse.iter().enumerate() {
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= N {
            let step = N / len;
            for start in (0..N).step_by(len) {
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

    /// Spectra (bins 0 to Nyquist) of two real frames, windowed, in one
    /// complex transform. `frame(i)` gives sample `i` of both.
    fn forward_pair(
        &self,
        frame: impl Fn(usize) -> (f32, f32),
        re: &mut [f32],
        im: &mut [f32],
        out: &mut [[(f32, f32); BINS]; 2],
    ) {
        for i in 0..N {
            let (a, b) = frame(i);
            re[i] = a * self.window[i];
            im[i] = b * self.window[i];
        }
        self.run(re, im);
        for k in 0..BINS {
            let m = (N - k) % N;
            let (zr, zi, nr, ni) = (re[k], im[k], re[m], im[m]);
            out[0][k] = (f32::midpoint(zr, nr), 0.5 * (zi - ni));
            out[1][k] = (f32::midpoint(zi, ni), -0.5 * (zr - nr));
        }
    }

    /// The windowed inverse of a real frame's spectrum (bins 0 to Nyquist),
    /// added into `out[..N]`.
    fn inverse_add(
        &self,
        spec: &[(f32, f32); BINS],
        re: &mut [f32],
        im: &mut [f32],
        out: &mut [f32],
    ) {
        // x = Re(FFT(conj(X))) / N for a Hermitian X.
        for k in 0..BINS {
            re[k] = spec[k].0;
            im[k] = -spec[k].1;
        }
        for k in BINS..N {
            re[k] = spec[N - k].0;
            im[k] = spec[N - k].1;
        }
        self.run(re, im);
        #[expect(clippy::cast_precision_loss, reason = "N = 1024 is exact")]
        let scale = 1.0 / N as f32;
        for (i, o) in out.iter_mut().take(N).enumerate() {
            *o += re[i] * scale * self.window[i];
        }
    }
}

/// Sample `i` of `x` with zeros outside.
fn at(x: &[f32], i: isize) -> f32 {
    usize::try_from(i)
        .ok()
        .and_then(|i| x.get(i))
        .copied()
        .unwrap_or(0.0)
}

/// First sample of STFT frame `t`.
fn frame_origin(t: usize) -> isize {
    isize::try_from(t * HOP).unwrap_or(isize::MAX) - isize::try_from(HOP).unwrap_or(0)
}

// ---------------------------------------------------------------- encoder

/// The parameters of `left` and `right`, coded as pages of frames `spans`
/// (the HQ stream's pages).
#[must_use]
pub fn analyse(left: &[f32], right: &[f32], spans: &[Range<usize>], cfg: Config) -> Params {
    let edges = cfg.edges();
    let nb = cfg.band_count();
    let (of_frame, pages) = frame_sets(spans, cfg.frames);
    let sets = pages.last().map_or(0, |p| p.end);
    let frames = of_frame.len();
    let (mut pl, mut pr, mut x) = (
        vec![0.0_f64; sets * nb],
        vec![0.0; sets * nb],
        vec![0.0; sets * nb],
    );
    let fft = Fft::new();
    let (mut re, mut im) = (vec![0.0; N], vec![0.0; N]);
    let mut spec = [[(0.0, 0.0); BINS]; 2];
    let stft_frames = if frames == 0 { 0 } else { frames + 1 };
    for t in 0..stft_frames {
        let origin = frame_origin(t);
        fft.forward_pair(
            |i| {
                let j = origin + isize::try_from(i).unwrap_or(0);
                (at(left, j), at(right, j))
            },
            &mut re,
            &mut im,
            &mut spec,
        );
        let set = of_frame[t.min(frames - 1)];
        for b in 0..nb {
            let (mut a, mut c, mut d) = (0.0_f64, 0.0_f64, 0.0_f64);
            for k in edges[b]..edges[b + 1] {
                let (lr, li) = spec[0][k];
                let (rr, ri) = spec[1][k];
                let (lr, li, rr, ri) = (f64::from(lr), f64::from(li), f64::from(rr), f64::from(ri));
                a += lr * lr + li * li;
                c += rr * rr + ri * ri;
                d += lr * rr + li * ri;
            }
            pl[set * nb + b] += a;
            pr[set * nb + b] += c;
            x[set * nb + b] += d;
        }
    }
    let side_db = |r: f64| {
        let r = r.clamp(-1.0, 1.0);
        10.0 * ((1.0005 - r) / (1.0005 + r)).log10()
    };
    let icc_db: Vec<f64> = ICC.iter().map(|&r| side_db(r)).collect();
    let nearest = |levels: &[f64], v: f64| -> u8 {
        let mut best = (0, f64::INFINITY);
        for (i, &l) in levels.iter().enumerate() {
            let e = (v - l).abs();
            if e < best.1 {
                best = (i, e);
            }
        }
        u8::try_from(best.0).unwrap_or(0)
    };
    let silent = 1e-7 * crate::config::to_f64(N);
    let eps = 1e-12;
    let mut out = Params {
        iid: vec![IID_CENTRE; sets * nb],
        icc: vec![0; sets * nb],
    };
    for i in 0..sets * nb {
        if pl[i] + pr[i] < silent {
            continue;
        }
        out.iid[i] = nearest(&IID_DB, 10.0 * ((pl[i] + eps) / (pr[i] + eps)).log10());
        out.icc[i] = nearest(&icc_db, side_db(x[i] / ((pl[i] * pr[i]).sqrt() + eps)));
    }
    out
}

// ---------------------------------------------------------------- coding

/// A canonical Huffman code over symbols `0..lengths.len()`.
struct Code {
    /// `(code, length)` of each symbol.
    codes: Vec<(u32, u8)>,
    /// Symbols in code order.
    sorted: Vec<u8>,
    /// Per length: the first code, and the index in `sorted` of its symbol.
    first: Vec<(u32, usize)>,
    /// Per length: how many codes have it.
    count: Vec<usize>,
}

impl Code {
    fn new(lengths: &[u8]) -> Self {
        let max = usize::from(lengths.iter().copied().max().unwrap_or(0));
        let mut sorted: Vec<u8> = (0..lengths.len())
            .filter_map(|s| u8::try_from(s).ok())
            .collect();
        sorted.sort_by_key(|&s| (lengths[usize::from(s)], s));
        let mut codes = vec![(0, 0); lengths.len()];
        let mut count = vec![0; max + 1];
        let mut first = vec![(0, 0); max + 1];
        let (mut code, mut len) = (0_u32, 0_u8);
        for (index, &s) in sorted.iter().enumerate() {
            let l = lengths[usize::from(s)];
            code <<= l - len;
            len = l;
            if count[usize::from(l)] == 0 {
                first[usize::from(l)] = (code, index);
            }
            count[usize::from(l)] += 1;
            codes[usize::from(s)] = (code, l);
            code += 1;
        }
        Self {
            codes,
            sorted,
            first,
            count,
        }
    }

    fn put(&self, out: &mut Bits, symbol: usize) {
        let (code, len) = self.codes[symbol];
        for i in (0..len).rev() {
            out.put((code >> i) & 1 != 0);
        }
    }

    fn len(&self, symbol: usize) -> usize {
        usize::from(self.codes[symbol].1)
    }

    fn get(&self, input: &mut Reader<'_>) -> Result<usize> {
        let mut code = 0_u32;
        for len in 1..self.count.len() {
            code = (code << 1) | u32::from(input.get()?);
            let (start, index) = self.first[len];
            if self.count[len] > 0 && code >= start {
                let offset = usize::try_from(code - start)?;
                if offset < self.count[len] {
                    return Ok(usize::from(self.sorted[index + offset]));
                }
            }
        }
        bail!("bad parametric stereo code")
    }
}

/// Bits, low bits of each byte first.
#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    used: u32,
}

impl Bits {
    fn put(&mut self, bit: bool) {
        if self.used.is_multiple_of(8) {
            self.out.push(0);
        }
        if let (true, Some(last)) = (bit, self.out.last_mut()) {
            *last |= 1 << (self.used % 8);
        }
        self.used += 1;
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn get(&mut self) -> Result<bool> {
        let byte = self
            .data
            .get(self.pos / 8)
            .context("parametric stereo data is truncated")?;
        let bit = byte >> (self.pos % 8) & 1 != 0;
        self.pos += 1;
        Ok(bit)
    }
}

/// One parameter type: its levels, codes and grid centre.
struct Kind {
    levels: usize,
    centre: u8,
    freq: Code,
    time: Code,
}

fn kinds() -> [Kind; 2] {
    [
        Kind {
            levels: IID_DB.len(),
            centre: IID_CENTRE,
            freq: Code::new(&IID_FREQ),
            time: Code::new(&IID_TIME),
        },
        Kind {
            levels: ICC.len(),
            centre: 0,
            freq: Code::new(&ICC_FREQ),
            time: Code::new(&ICC_TIME),
        },
    ]
}

/// Symbol of delta `d` for a type with `levels` levels.
fn symbol(d: i32, levels: usize) -> usize {
    usize::try_from(d + i32::try_from(levels).unwrap_or(0) - 1).unwrap_or(0)
}

/// Codes the parameter sets `sets` (one page's) of `params`.
#[must_use]
pub fn write_page(params: &Params, sets: Range<usize>, cfg: Config) -> Vec<u8> {
    let nb = cfg.band_count();
    let mut out = Bits::default();
    let kinds = kinds();
    for s in sets.clone() {
        for (kind, all) in kinds.iter().zip([&params.iid, &params.icc]) {
            let row = &all[s * nb..(s + 1) * nb];
            let freq: Vec<usize> = (0..nb)
                .map(|b| {
                    let prev = if b == 0 { kind.centre } else { row[b - 1] };
                    symbol(i32::from(row[b]) - i32::from(prev), kind.levels)
                })
                .collect();
            if s == sets.start {
                for &f in &freq {
                    kind.freq.put(&mut out, f);
                }
                continue;
            }
            let prev = &all[(s - 1) * nb..s * nb];
            let time: Vec<usize> = row
                .iter()
                .zip(prev)
                .map(|(&v, &p)| symbol(i32::from(v) - i32::from(p), kind.levels))
                .collect();
            let cost_f: usize = freq.iter().map(|&f| kind.freq.len(f)).sum();
            let cost_t: usize = time.iter().map(|&t| kind.time.len(t)).sum();
            if cost_t < cost_f {
                out.put(true);
                for &t in &time {
                    kind.time.put(&mut out, t);
                }
            } else {
                out.put(false);
                for &f in &freq {
                    kind.freq.put(&mut out, f);
                }
            }
        }
    }
    out.out
}

/// Reads one page's `sets` parameter sets: `(iid, icc)`, `[set][band]`.
///
/// # Errors
/// Fails when the data is truncated or codes a level out of range.
pub fn read_page(data: &[u8], sets: usize, cfg: Config) -> Result<(Vec<u8>, Vec<u8>)> {
    let nb = cfg.band_count();
    let mut input = Reader { data, pos: 0 };
    let mut out = [vec![0_u8; sets * nb], vec![0_u8; sets * nb]];
    for s in 0..sets {
        for (kind, all) in kinds().iter().zip(out.iter_mut()) {
            let time = s > 0 && input.get()?;
            let levels = i32::try_from(kind.levels)?;
            for b in 0..nb {
                let code = if time { &kind.time } else { &kind.freq };
                let d = i32::try_from(code.get(&mut input)?)? - (levels - 1);
                let base = if time {
                    all[(s - 1) * nb + b]
                } else if b == 0 {
                    kind.centre
                } else {
                    all[s * nb + b - 1]
                };
                let v = i32::from(base) + d;
                ensure!(
                    (0..levels).contains(&v),
                    "parametric stereo level {v} is out of range"
                );
                all[s * nb + b] = u8::try_from(v)?;
            }
        }
    }
    ensure!(
        data.len() == input.pos.div_ceil(8),
        "parametric stereo data has {} bytes, its sets need {}",
        data.len(),
        input.pos.div_ceil(8)
    );
    Ok(out.into())
}

/// Codes every page's parameters.
#[must_use]
pub fn write_pages(params: &Params, spans: &[Range<usize>], cfg: Config) -> Vec<Vec<u8>> {
    frame_sets(spans, cfg.frames)
        .1
        .into_iter()
        .map(|sets| write_page(params, sets, cfg))
        .collect()
}

/// Reads every page's parameters.
///
/// # Errors
/// Fails when a page's data is wrong.
pub fn read_pages(pages: &[&[u8]], spans: &[Range<usize>], cfg: Config) -> Result<Params> {
    let sets = frame_sets(spans, cfg.frames).1;
    ensure!(
        pages.len() == sets.len(),
        "parametric stereo data for {} of {} pages",
        pages.len(),
        sets.len()
    );
    let mut out = Params {
        iid: Vec::new(),
        icc: Vec::new(),
    };
    for (index, (data, range)) in pages.iter().zip(sets).enumerate() {
        let (iid, icc) = read_page(data, range.len(), cfg)
            .with_context(|| format!("page {index}'s parametric stereo data"))?;
        out.iid.extend(iid);
        out.icc.extend(icc);
    }
    Ok(out)
}

// ---------------------------------------------------------------- decoder

/// `(g, h)` of the side `S = g M + h D` for each IID and ICC level,
/// `[iid][icc]`.
fn gains() -> Vec<(f32, f32)> {
    let mut out = Vec::with_capacity(IID_DB.len() * ICC.len());
    for &db in &IID_DB {
        let c = 10_f64.powf(db / 20.0);
        for &rho in &ICC {
            // Powers L = c^2, R = 1, correlation rho c; M = (L + R) / 2.
            let c2 = c * c;
            let a = c2 + 1.0;
            // A mid at least 10 dB under the channels: bounds the gains.
            let pm4 = (a + 2.0 * rho * c).max(0.1 * a);
            let g = (c2 - 1.0) / pm4;
            let ps4 = a - 2.0 * rho * c;
            let g2 = g * g;
            let h = (ps4 / pm4 - g2).max(0.0).sqrt();
            #[expect(clippy::cast_possible_truncation, reason = "f32 gains")]
            out.push((g as f32, h as f32));
        }
    }
    out
}

/// Rebuilds left and right from the decoded mid (any length) and the
/// parameters of the stream's pages `spans`.
///
/// # Errors
/// Fails when the parameters do not match the pages.
pub fn upmix(
    mid: &[f32],
    params: &Params,
    spans: &[Range<usize>],
    cfg: Config,
) -> Result<[Vec<f32>; 2]> {
    let nb = cfg.band_count();
    let edges = cfg.edges();
    let (of_frame, pages) = frame_sets(spans, cfg.frames);
    let sets = pages.last().map_or(0, |p| p.end);
    ensure!(
        params.iid.len() == sets * nb && params.icc.len() == sets * nb,
        "parametric stereo parameters for the wrong number of sets"
    );
    let n = mid.len();
    if of_frame.is_empty() || n == 0 {
        return Ok([mid.to_vec(), mid.to_vec()]);
    }
    // The decorrelated mid.
    #[expect(clippy::cast_possible_truncation, reason = "f32 taps")]
    let taps: Vec<(usize, f32)> = VELVET.iter().map(|&(d, g)| (d, g as f32)).collect();
    let mut wet = vec![0.0_f32; n];
    for (i, w) in wet.iter_mut().enumerate() {
        let mut acc = 0.0;
        for &(delay, gain) in &taps {
            if delay > i {
                break;
            }
            acc += gain * mid[i - delay];
        }
        *w = acc;
    }
    let table = gains();
    let fft = Fft::new();
    let (mut re, mut im) = (vec![0.0; N], vec![0.0; N]);
    let mut spec = [[(0.0, 0.0); BINS]; 2];
    let mut side_spec = [(0.0_f32, 0.0_f32); BINS];
    let frames = n.div_ceil(HOP);
    let mut side = vec![0.0_f32; (frames + 1) * HOP + N];
    for t in 0..=frames {
        let origin = frame_origin(t);
        fft.forward_pair(
            |i| {
                let j = origin + isize::try_from(i).unwrap_or(0);
                (at(mid, j), at(&wet, j))
            },
            &mut re,
            &mut im,
            &mut spec,
        );
        let set = of_frame[t.min(of_frame.len() - 1)];
        for b in 0..nb {
            let range = edges[b]..edges[b + 1];
            let (mut em, mut ed) = (0.0_f32, 0.0_f32);
            for k in range.clone() {
                let ((mr, mi), (dr, di)) = (spec[0][k], spec[1][k]);
                em += mr * mr + mi * mi;
                ed += dr * dr + di * di;
            }
            let duck = ((em + 1e-12) / (ed + 1e-12)).sqrt().min(1.0);
            let i = set * nb + b;
            let (g, h) = table[usize::from(params.iid[i]) * ICC.len() + usize::from(params.icc[i])];
            let hd = h * duck;
            for k in range {
                let ((mr, mi), (dr, di)) = (spec[0][k], spec[1][k]);
                side_spec[k] = (g * mr + hd * dr, g * mi + hd * di);
            }
        }
        // Frame t starts at sample (t - 1) * HOP: offset by HOP in `side`.
        fft.inverse_add(&side_spec, &mut re, &mut im, &mut side[t * HOP..]);
    }
    let side = &side[HOP..HOP + n];
    let left = mid.iter().zip(side).map(|(m, s)| m + s).collect();
    let right = mid.iter().zip(side).map(|(m, s)| m - s).collect();
    Ok([left, right])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[expect(clippy::cast_precision_loss, reason = "test noise")]
                let v = (state >> 8) as f32 / 16_777_216.0 - 0.5;
                v * 0.5
            })
            .collect()
    }

    fn spans(frames: usize, per: usize) -> Vec<Range<usize>> {
        (0..frames)
            .step_by(per)
            .map(|s| s..(s + per).min(frames))
            .collect()
    }

    fn power(x: &[f32]) -> f64 {
        x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / crate::config::to_f64(x.len())
    }

    #[test]
    fn codes_are_complete_prefix_codes() {
        for lengths in [&IID_FREQ[..], &IID_TIME, &ICC_FREQ, &ICC_TIME] {
            let kraft: f64 = lengths.iter().map(|&l| 0.5_f64.powi(i32::from(l))).sum();
            assert!((kraft - 1.0).abs() < 1e-12, "{kraft}");
            let code = Code::new(lengths);
            let mut bits = Bits::default();
            for s in 0..lengths.len() {
                code.put(&mut bits, s);
            }
            let mut input = Reader {
                data: &bits.out,
                pos: 0,
            };
            for s in 0..lengths.len() {
                assert_eq!(code.get(&mut input).ok(), Some(s));
            }
        }
    }

    #[test]
    fn config_bytes_round_trip() -> Result<()> {
        for bands in [20, 10] {
            for frames in [1, 2, 4, 8, 16] {
                let c = Config::new(bands, frames)?;
                assert_eq!(Config::from_byte(c.to_byte())?, c);
            }
        }
        assert!(Config::new(12, 8).is_err());
        assert!(Config::new(20, 3).is_err());
        assert!(Config::new(20, 32).is_err());
        assert!(Config::from_byte(0x80).is_err());
        assert!(Config::from_byte(2).is_err());
        Ok(())
    }

    /// A stereo test signal: a source panned left, plus independent noise
    /// on each side in the second half.
    fn scene(n: usize) -> [Vec<f32>; 2] {
        let a = noise(n, 1);
        let (b, c) = (noise(n, 2), noise(n, 3));
        let left = (0..n)
            .map(|i| 0.8 * a[i] + if i >= n / 2 { 0.5 * b[i] } else { 0.0 })
            .collect();
        let right = (0..n)
            .map(|i| 0.4 * a[i] + if i >= n / 2 { 0.5 * c[i] } else { 0.0 })
            .collect();
        [left, right]
    }

    #[test]
    fn parameters_round_trip_per_page() -> Result<()> {
        let frames = 300;
        let [l, r] = scene(frames * HOP - 100);
        for cfg in [
            Config::DEFAULT,
            Config::new(10, 4)?,
            Config::new(20, 16)?,
            Config::new(20, 1)?,
        ] {
            let sp = spans(frames, 80);
            let params = analyse(&l, &r, &sp, cfg);
            let pages = write_pages(&params, &sp, cfg);
            let refs: Vec<&[u8]> = pages.iter().map(Vec::as_slice).collect();
            assert_eq!(read_pages(&refs, &sp, cfg)?, params);
            // Each page alone gives its own sets.
            let sets = frame_sets(&sp, cfg.frames).1;
            let nb = cfg.band_count();
            for (data, range) in pages.iter().zip(&sets) {
                let (iid, icc) = read_page(data, range.len(), cfg)?;
                assert_eq!(iid, params.iid[range.start * nb..range.end * nb]);
                assert_eq!(icc, params.icc[range.start * nb..range.end * nb]);
            }
            // Truncated or padded data is refused.
            if let Some(data) = pages.first().filter(|d| d.len() > 1) {
                assert!(read_page(&data[..data.len() - 1], sets[0].len(), cfg).is_err());
                let mut longer = data.clone();
                longer.push(0);
                assert!(read_page(&longer, sets[0].len(), cfg).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn upmix_restores_the_image() -> Result<()> {
        let frames = 400;
        let n = frames * HOP;
        let [l, r] = scene(n);
        let mid: Vec<f32> = l.iter().zip(&r).map(|(a, b)| (a + b) / 2.0).collect();
        let sp = spans(frames, 80);
        let cfg = Config::DEFAULT;
        let params = analyse(&l, &r, &sp, cfg);
        let [yl, yr] = upmix(&mid, &params, &sp, cfg)?;
        // The mid is kept.
        for i in 0..n {
            assert!((f32::midpoint(yl[i], yr[i]) - mid[i]).abs() < 1e-5);
        }
        // First half: one source panned 6 dB left, fully coherent.
        let q = n / 4;
        let ild = |a: &[f32], b: &[f32]| 10.0 * (power(a) / power(b)).log10();
        let ref_ild = ild(&l[q / 2..q], &r[q / 2..q]);
        let got = ild(&yl[q / 2..q], &yr[q / 2..q]);
        assert!((ref_ild - got).abs() < 1.0, "{ref_ild} {got}");
        let corr = |a: &[f32], b: &[f32]| {
            let x: f64 = a
                .iter()
                .zip(b)
                .map(|(&p, &q)| f64::from(p) * f64::from(q))
                .sum();
            x / (power(a) * power(b)).sqrt() / crate::config::to_f64(a.len())
        };
        assert!(corr(&yl[q / 2..q], &yr[q / 2..q]) > 0.95);
        // Second half: much less coherent, as the input.
        let (s, e) = (n / 2 + n / 8, n - n / 8);
        let (want, got) = (corr(&l[s..e], &r[s..e]), corr(&yl[s..e], &yr[s..e]));
        assert!((want - got).abs() < 0.15, "{want} {got}");
        assert!((ild(&l[s..e], &r[s..e]) - ild(&yl[s..e], &yr[s..e])).abs() < 1.0);
        Ok(())
    }

    #[test]
    fn upmix_is_deterministic_and_local() -> Result<()> {
        let frames = 320;
        let n = frames * HOP;
        let [l, r] = scene(n);
        let mid: Vec<f32> = l.iter().zip(&r).map(|(a, b)| (a + b) / 2.0).collect();
        let sp = spans(frames, 80);
        let cfg = Config::DEFAULT;
        let params = analyse(&l, &r, &sp, cfg);
        let a = upmix(&mid, &params, &sp, cfg)?;
        let b = upmix(&mid, &params, &sp, cfg)?;
        assert!(
            a[0].iter()
                .zip(&b[0])
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        // Changing page 3's parameters leaves pages 0 and 1 alone, and page 2
        // up to one hop before its end.
        let mut other = params;
        let nb = cfg.band_count();
        let sets = frame_sets(&sp, cfg.frames).1;
        for v in &mut other.iid[sets[3].start * nb..sets[3].end * nb] {
            *v = 0;
        }
        let c = upmix(&mid, &other, &sp, cfg)?;
        let edge = 3 * 80 * HOP - HOP;
        assert!(
            a[0][..edge]
                .iter()
                .zip(&c[0][..edge])
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert!(
            a[0][edge + HOP..]
                .iter()
                .zip(&c[0][edge + HOP..])
                .any(|(x, y)| x.to_bits() != y.to_bits())
        );
        Ok(())
    }

    #[test]
    fn mono_input_stays_mono() -> Result<()> {
        let frames = 100;
        let x = noise(frames * HOP, 9);
        let sp = spans(frames, 80);
        let cfg = Config::DEFAULT;
        let params = analyse(&x, &x, &sp, cfg);
        assert!(params.iid.iter().all(|&v| v == IID_CENTRE));
        assert!(params.icc.iter().all(|&v| v == 0));
        let [l, r] = upmix(&x, &params, &sp, cfg)?;
        assert!(l.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(r.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-5));
        Ok(())
    }
}
