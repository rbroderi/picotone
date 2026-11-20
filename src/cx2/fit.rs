//! Offline helpers that derive the model tables: counting and weight selection.
//!
//! Nothing here runs when a file is packed or unpacked. [`histograms`] counts
//! how feature buckets are distributed for the code that actually occurred
//! and for all candidate codes; the table builder turns the two counts into
//! log-likelihood ratios. [`tune`] picks one scalar weight per expert and
//! codebook by coordinate search on held-out streams. Floating point is used
//! here to measure code lengths, never to produce coded bits.

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::cx2::features::{
    CARD, CONTEXTS, Centroids, GEO_BUCKETS, PAIR_BUCKETS, REC_BUCKETS, State, pair_bucket,
};
use crate::cx2::model::{EXPERTS, Model};

/// Bucket counts of one feature: for the code that occurred, and over all candidates.
#[derive(Debug, Clone, Serialize)]
pub struct Counts {
    /// Bucket of the code that occurred, one count per coded symbol.
    #[serde(rename = "true")]
    pub hit: Vec<u64>,
    /// Buckets of every candidate code, 1024 counts per coded symbol.
    pub all: Vec<u64>,
}

impl Counts {
    fn new(buckets: usize) -> Self {
        Self {
            hit: vec![0; buckets],
            all: vec![0; buckets],
        }
    }

    fn add(&mut self, other: &Self) {
        for (a, b) in self.hit.iter_mut().zip(&other.hit) {
            *a += b;
        }
        for (a, b) in self.all.iter_mut().zip(&other.all) {
            *a += b;
        }
    }
}

/// Feature histograms of one codebook.
#[derive(Debug, Clone, Serialize)]
pub struct LayerCounts {
    /// Squared-distance feature.
    pub geo: Counts,
    /// Recency feature.
    pub rec: Counts,
    /// Pair features, one per context slot.
    pub pair: Vec<Counts>,
}

/// Counts feature buckets over `streams` (frame-major codes, `num_codebooks` per frame).
///
/// The squared-distance feature is the expensive one; it is counted only for the
/// first `geo_codebooks` codebooks and left at zero (expert off) for the rest.
/// The state is reset every `reset_frames` frames (never when zero), as the coder does.
///
/// # Errors
/// Fails when a stream has an out-of-range code or the centroids do not cover it.
pub fn histograms(
    centroids: &Centroids,
    num_codebooks: usize,
    geo_codebooks: usize,
    reset_frames: usize,
    streams: &[Vec<u16>],
) -> Result<Vec<LayerCounts>> {
    let threads = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .min(8);
    let chunk = streams.len().div_ceil(threads).max(1);
    let parts: Vec<Result<Vec<LayerCounts>>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for part in streams.chunks(chunk) {
                handles.push(scope.spawn(move || {
                    count(centroids, num_codebooks, geo_codebooks, reset_frames, part)
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("a counting thread panicked")))
                })
                .collect()
        });
    let mut total = count(centroids, num_codebooks, geo_codebooks, reset_frames, &[])?;
    for part in parts {
        for (sum, layer) in total.iter_mut().zip(part?) {
            sum.geo.add(&layer.geo);
            sum.rec.add(&layer.rec);
            for (a, b) in sum.pair.iter_mut().zip(&layer.pair) {
                a.add(b);
            }
        }
    }
    Ok(total)
}

/// Whether a reset is due before the next code.
fn due(reset_frames: usize, codebook: usize, frame: u32) -> bool {
    reset_frames > 0 && codebook == 0 && usize::try_from(frame).is_ok_and(|f| f == reset_frames)
}

/// Single-threaded worker of [`histograms`].
fn count(
    centroids: &Centroids,
    num_codebooks: usize,
    geo_codebooks: usize,
    reset_frames: usize,
    streams: &[Vec<u16>],
) -> Result<Vec<LayerCounts>> {
    let mut layers: Vec<LayerCounts> = (0..num_codebooks)
        .map(|_| LayerCounts {
            geo: Counts::new(GEO_BUCKETS),
            rec: Counts::new(REC_BUCKETS),
            pair: (0..CONTEXTS).map(|_| Counts::new(PAIR_BUCKETS)).collect(),
        })
        .collect();
    let mut buckets = vec![0_u16; CARD];
    for stream in streams {
        let mut state = State::new(centroids, num_codebooks)?;
        for &code in stream {
            if due(reset_frames, state.codebook(), state.frame()) {
                state.reset();
            }
            let index = usize::from(code);
            ensure!(index < CARD, "code {code} is out of range");
            let layer = &mut layers[state.codebook()];
            if state.codebook() < geo_codebooks {
                state.geo_buckets(&mut buckets);
                for &b in &buckets {
                    layer.geo.all[usize::from(b)] += 1;
                }
                layer.geo.hit[usize::from(buckets[index])] += 1;
            }
            state.rec_buckets(&mut buckets);
            for &b in &buckets {
                layer.rec.all[usize::from(b)] += 1;
            }
            layer.rec.hit[usize::from(buckets[index])] += 1;
            for (slot, counts) in layer.pair.iter_mut().enumerate() {
                let (total, hits) = state.pair_hits(slot);
                let mut own = 0;
                for &(seen, count) in hits {
                    counts.all[pair_bucket(count, total)] += 1;
                    if seen == code {
                        own = count;
                    }
                }
                counts.all[pair_bucket(0, total)] += u64::try_from(CARD - hits.len())?;
                counts.hit[pair_bucket(own, total)] += 1;
            }
            state.update(code)?;
        }
    }
    Ok(layers)
}

/// Result of [`tune`] for one codebook.
#[derive(Debug, Clone, Serialize)]
pub struct Tuned {
    /// Weight of each expert in sixteenths, in [`crate::cx2::model::EXPERT_NAMES`] order.
    pub weights: Vec<u32>,
    /// Sampled symbols used.
    pub samples: usize,
    /// Mean bits per code with the prior only.
    pub bits_prior: f64,
    /// Mean bits per code with every expert at weight one.
    pub bits_unit: f64,
    /// Mean bits per code with the chosen weights.
    pub bits_tuned: f64,
}

/// Weight grid, in sixteenths.
const GRID: [u32; 13] = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24];

/// Sampled symbols of one codebook: per-expert logit contributions and the true code.
#[derive(Default)]
struct Samples {
    bias: Vec<i32>,
    /// `EXPERTS * CARD` values per sample.
    values: Vec<i16>,
    labels: Vec<u16>,
}

impl Samples {
    /// Mean code length in bits under the given weights.
    #[expect(
        clippy::cast_precision_loss,
        reason = "logit sums and sample counts are far below 2^52"
    )]
    fn cost(&self, weights: &[u32; EXPERTS], frac_bits: u32) -> f64 {
        let unit = f64::from(1_u32 << frac_bits);
        let mut logits = vec![0_i64; CARD];
        let mut total = 0.0;
        for (sample, &label) in self.values.chunks_exact(EXPERTS * CARD).zip(&self.labels) {
            for (l, &b) in logits.iter_mut().zip(&self.bias) {
                *l = 16 * i64::from(b);
            }
            for (expert, &weight) in sample.chunks_exact(CARD).zip(weights) {
                if weight != 0 {
                    for (l, &v) in logits.iter_mut().zip(expert) {
                        *l += i64::from(weight) * i64::from(v);
                    }
                }
            }
            let best = logits.iter().copied().max().unwrap_or(0);
            let sum: f64 = logits
                .iter()
                .map(|&l| ((l - best) as f64 / (16.0 * unit)).exp2())
                .sum();
            let own = (logits[usize::from(label)] - best) as f64 / (16.0 * unit);
            total += sum.log2() - own;
        }
        total / self.labels.len().max(1) as f64
    }

    fn tune(&self, frac_bits: u32) -> Tuned {
        // The cost is convex in the weights, so coordinate search from a cautious start
        // settles within a few sweeps.
        let mut weights = [4_u32; EXPERTS];
        let mut best = self.cost(&weights, frac_bits);
        for _ in 0..5 {
            let before = weights;
            for expert in 0..EXPERTS {
                for &candidate in &GRID {
                    let mut trial = weights;
                    trial[expert] = candidate;
                    let cost = self.cost(&trial, frac_bits);
                    if cost < best {
                        best = cost;
                        weights = trial;
                    }
                }
            }
            if weights == before {
                break;
            }
        }
        Tuned {
            weights: weights.to_vec(),
            samples: self.labels.len(),
            bits_prior: self.cost(&[0; EXPERTS], frac_bits),
            bits_unit: self.cost(&[16; EXPERTS], frac_bits),
            bits_tuned: best,
        }
    }
}

/// Chooses per-codebook expert weights that minimise the code length of sampled symbols.
///
/// `model` should hold unit-weight tables; about `per_codebook` frames are sampled
/// evenly from `streams`.
///
/// # Errors
/// Fails when a stream does not fit the model.
pub fn tune(
    model: &Model,
    num_codebooks: usize,
    streams: &[Vec<u16>],
    per_codebook: usize,
    reset_frames: usize,
) -> Result<Vec<Tuned>> {
    let frames: usize = streams.iter().map(|s| s.len() / num_codebooks.max(1)).sum();
    let stride = (frames / per_codebook.max(1)).max(1);
    let mut samples: Vec<Samples> = (0..num_codebooks).map(|_| Samples::default()).collect();
    let mut scratch = vec![0_i32; CARD];
    let mut frame = 0_usize;
    for stream in streams {
        let mut predictor = model.predictor(num_codebooks)?;
        for (index, codes) in stream.chunks_exact(num_codebooks).enumerate() {
            if reset_frames > 0 && index > 0 && index % reset_frames == 0 {
                predictor.reset();
            }
            let keep = frame % stride == stride / 2;
            frame += 1;
            for &code in codes {
                if keep {
                    let store = &mut samples[predictor.codebook()];
                    if store.bias.is_empty() {
                        store.bias = predictor.bias().to_vec();
                    }
                    for expert in 0..EXPERTS {
                        scratch.fill(0);
                        predictor.add_expert(expert, &mut scratch);
                        store.values.extend(scratch.iter().map(|&v| {
                            i16::try_from(v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)))
                                .unwrap_or(0)
                        }));
                    }
                    store.labels.push(code);
                }
                predictor.update(code)?;
            }
        }
    }
    let frac_bits = model.frac_bits();
    let threads = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .min(8);
    let mut tuned: Vec<Option<Tuned>> = (0..num_codebooks).map(|_| None).collect();
    let chunk = num_codebooks.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (out, input) in tuned.chunks_mut(chunk).zip(samples.chunks(chunk)) {
            let _handle = scope.spawn(move || {
                for (slot, store) in out.iter_mut().zip(input) {
                    *slot = Some(store.tune(frac_bits));
                }
            });
        }
    });
    Ok(tuned.into_iter().flatten().collect())
}
