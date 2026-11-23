//! Integer-only probability model for EnCodec codes.
//!
//! The model is a product of experts evaluated once per code. Each expert
//! adds a table value to the logit of every candidate code `j`:
//!
//! ```text
//! logit_j = bias_k[j]                                   static prior of codebook k
//!         + GEO_k[ bucket(|residual - centroid_j|^2) ]  radial unit on EnCodec's own centroids
//!         + REC_k[ bucket(recency_j, count_j) ]         adaptive cache
//!         + PAIR_k,s[ bucket(hits of context_s -> j) ]  adaptive successor counts, 3 contexts
//!         + STAT_k,s[ context_s ][ j ]                  static sparse pair table, 3 contexts
//!         + (OUT_k,h[j] . decayed sum of EMB_k,h[recent codes])   low-rank history, 2 sources
//!         + NET_k(previous 4 frames, this frame so far)[j]         trained MLP, see `net`
//! freq_j  = (EXP[frac] >> int) + floor                  2^-x by table lookup
//! ```
//!
//! Logits are integers in units of `2^-frac_bits` bits. All tables except
//! the optional network are log-likelihood ratios (or an SVD of them)
//! obtained by counting on a fit set; the network is trained by gradient
//! descent. Any expert can be absent from a table file. A version 5 table also
//! stores the fixed-point centroids, so it alone defines the coder (and its hash);
//! a version 4 table takes them from the codec's weights. No step uses
//! floating point, so the frequencies, and therefore the coded bitstream,
//! are identical on every machine.

use anyhow::{Context as _, Result, ensure};

use crate::cx2::features::{
    CARD, CONTEXTS, Centroids, DIM, GEO_BUCKETS, PAIR_BUCKETS, REC_BUCKETS, State, pair_bucket,
};
use crate::cx2::kernels::Simd;
use crate::cx2::net::{Net, NetState};

const TABLE_MAGIC: &[u8; 4] = b"ECXM";
/// Table whose centroids come from the codec's weights; its net has no position input.
const VERSION_WEIGHTS: u32 = 4;
/// Table that carries its own fixed-point centroids (right after the exponential
/// table, `[codebook][code][dimension]` as i16) and whose net, if any, is in the
/// second format with the position input (see [`crate::cx2::net`]).
const VERSION_SELF_CONTAINED: u32 = 5;
/// History experts per codebook.
const HISTORIES: usize = 2;
/// Right shift of the history decay: each frame keeps 7/8 of the old sum.
const HISTORY_DECAY: u32 = 3;
/// Largest embedding magnitude; eight of them (the decay's fixed point) fit an `i16`.
const HISTORY_LIMIT: i16 = 4095;
/// Largest right shift applied to the exponential table.
const MAX_SHIFT: u32 = 31;
/// Largest magnitude of any single logit contribution.
const LOGIT_LIMIT: u32 = 1 << 20;
/// Number of experts: geo, rec, three pair experts, three static experts.
pub const EXPERTS: usize = 2 + 2 * CONTEXTS + HISTORIES + 1;
/// Names of the experts, in index order.
pub const EXPERT_NAMES: [&str; EXPERTS] = [
    "geo", "rec", "pair0", "pair1", "pair2", "stat0", "stat1", "stat2", "hist0", "hist1", "net",
];

/// History expert: a decayed sum of embeddings of recent codes, times an output matrix.
///
/// The two matrices are the truncated SVD of a co-occurrence PMI table, so this is
/// a low-rank bag-of-recent-codes model.
#[derive(Debug)]
struct History {
    rank: usize,
    /// `CARD x rank` embedding of a source code.
    emb: Vec<i16>,
    /// Output matrix, stored as `rank` columns of `CARD` entries.
    out: Vec<i16>,
    /// Right shift that turns a dot product into logit units.
    shift: u32,
}

/// Static sparse table: for each context code, a few `(code, value)` entries.
#[derive(Debug, Default)]
struct Sparse {
    offsets: Vec<u32>,
    codes: Vec<u16>,
    values: Vec<i16>,
}

impl Sparse {
    fn row(&self, context: usize) -> impl Iterator<Item = (usize, i32)> + '_ {
        let bound = |i: usize| {
            self.offsets
                .get(i)
                .map_or(0, |&v| usize::try_from(v).unwrap_or(0))
        };
        let (start, end) = (bound(context), bound(context + 1));
        self.codes[start..end]
            .iter()
            .zip(&self.values[start..end])
            .map(|(&c, &v)| (usize::from(c), i32::from(v)))
    }
}

/// Tables for one codebook.
#[derive(Debug)]
struct Layer {
    bias: Vec<i32>,
    /// `None` when the expert is switched off (all-zero table).
    geo: Option<Vec<i32>>,
    rec: Option<Vec<i32>>,
    pair: [Option<Vec<i32>>; CONTEXTS],
    stat: [Sparse; CONTEXTS],
    hist: [Option<History>; HISTORIES],
}

/// A loaded model: fixed-point centroids plus per-codebook tables.
#[derive(Debug)]
pub struct Model {
    centroids: Centroids,
    frac_bits: u32,
    /// Frequency by distance below the best logit: `(EXP[frac] >> int) + floor`.
    freq: Vec<u32>,
    layers: Vec<Layer>,
    hash: u64,
    /// See [`State::recency_steps`].
    recency_steps: Vec<u32>,
    simd: Simd,
    /// The trained expert, when the table carries one.
    net: Option<Net>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .context("model table size overflow")?;
        ensure!(end <= self.bytes.len(), "model table is truncated");
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn size(&mut self) -> Result<usize> {
        Ok(usize::try_from(self.u32()?)?)
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>> {
        Ok(self
            .take(n * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect())
    }

    fn i16s(&mut self, n: usize) -> Result<Vec<i16>> {
        Ok(self
            .take(n * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect())
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect())
    }

    /// Reads `n` logit values, bounded so that sums of all experts cannot overflow.
    fn logits(&mut self, n: usize) -> Result<Vec<i32>> {
        let values: Vec<i32> = self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b))
            .collect();
        ensure!(
            values.iter().all(|v| v.unsigned_abs() <= LOGIT_LIMIT),
            "model table value is out of range"
        );
        Ok(values)
    }

    /// Reads a table, mapping an all-zero one to `None`.
    fn expert(&mut self, n: usize) -> Result<Option<Vec<i32>>> {
        let values = self.logits(n)?;
        Ok(values.iter().any(|&v| v != 0).then_some(values))
    }

    fn i8s(&mut self, n: usize) -> Result<Vec<i16>> {
        Ok(self
            .take(n)?
            .iter()
            .map(|&b| i16::from(i8::from_le_bytes([b])))
            .collect())
    }

    /// Reads a history expert; rank zero means the expert is switched off.
    fn history(&mut self) -> Result<Option<History>> {
        let rank = self.size()?;
        let shift = self.u32()?;
        // 64 products of an i8 and a decayed sum below 2^15 stay far inside an i32.
        ensure!(rank <= 64 && shift <= 24, "implausible history expert");
        let emb = self.i16s(CARD * rank)?;
        ensure!(
            emb.iter().all(|v| v.abs() <= HISTORY_LIMIT),
            "history embedding is out of range"
        );
        // The file stores one row per code; the kernel wants one column per rank.
        let rows = self.i8s(CARD * rank)?;
        let mut out = vec![0; rows.len()];
        for (code, row) in rows.chunks_exact(rank.max(1)).enumerate() {
            for (i, &value) in row.iter().enumerate() {
                out[i * CARD + code] = value;
            }
        }
        Ok((rank > 0).then_some(History {
            rank,
            emb,
            out,
            shift,
        }))
    }

    fn sparse(&mut self) -> Result<Sparse> {
        let offsets = self.u32s(CARD + 1)?;
        let count = usize::try_from(offsets[CARD])?;
        ensure!(
            offsets[0] == 0 && offsets.windows(2).all(|w| w[0] <= w[1]),
            "static table offsets are not increasing"
        );
        let codes = self.u16s(count)?;
        ensure!(
            codes.iter().all(|&c| usize::from(c) < CARD),
            "static table code is out of range"
        );
        let values = self.i16s(count)?;
        Ok(Sparse {
            offsets,
            codes,
            values,
        })
    }
}

/// FNV-1a 64-bit hash.
fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The hash a self-contained (version 5) table file gives its model, without
/// parsing the rest of the table; `None` for other files.
///
/// Players use it to pick the table a stream names from a set of files.
#[must_use]
pub fn self_contained_hash(table: &[u8]) -> Option<u64> {
    let version = table.get(4..8)?;
    (table.get(..4)? == TABLE_MAGIC && version == VERSION_SELF_CONTAINED.to_le_bytes())
        .then(|| fnv1a(0xcbf2_9ce4_8422_2325, table))
}

impl Model {
    /// Builds a model from a table file and the codec's `safetensors` weights.
    ///
    /// A version 5 table carries its own centroids; for it the weights are only
    /// checked (they must round to exactly the stored centroids) and may be empty.
    ///
    /// # Errors
    /// Fails when either input is malformed or they do not fit together.
    pub fn load(table: &[u8], weights: &[u8]) -> Result<Self> {
        Self::parse(table, Some(weights))
    }

    /// Builds a model from a self-contained (version 5) table alone.
    ///
    /// # Errors
    /// Fails when the table is malformed or needs the codec's weights (version 4).
    pub fn from_table(table: &[u8]) -> Result<Self> {
        Self::parse(table, None)
    }

    fn parse(table: &[u8], weights: Option<&[u8]>) -> Result<Self> {
        let mut cur = Cursor {
            bytes: table,
            pos: 0,
        };
        ensure!(cur.take(4)? == TABLE_MAGIC, "not a model table (bad magic)");
        let version = cur.u32()?;
        ensure!(
            version == VERSION_WEIGHTS || version == VERSION_SELF_CONTAINED,
            "unsupported model table version {version}"
        );
        let self_contained = version == VERSION_SELF_CONTAINED;
        let codebooks = cur.size()?;
        let frac_bits = cur.u32()?;
        let floor = cur.u32()?;
        ensure!(
            (1..=64).contains(&codebooks) && frac_bits <= 12 && (1..=1 << 16).contains(&floor),
            "implausible model parameters"
        );
        let exp = cur.u32s(1 << frac_bits)?;
        ensure!(
            exp.iter().all(|&e| e <= 1 << 24),
            "exponential table entry is too large"
        );
        // Beyond a shift of 31 only the floor remains, so the table can stop there.
        let freq: Vec<u32> = (0..=MAX_SHIFT)
            .flat_map(|shift| exp.iter().map(move |&e| (e >> shift) + floor))
            .collect();
        let stored = if self_contained {
            Some(Centroids::from_fixed(
                cur.i16s(codebooks * CARD * DIM)?,
                codebooks,
            )?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(codebooks);
        for _ in 0..codebooks {
            let bias = cur.logits(CARD)?;
            let geo = cur.expert(GEO_BUCKETS)?;
            let rec = cur.expert(REC_BUCKETS)?;
            let pair = [
                cur.expert(PAIR_BUCKETS)?,
                cur.expert(PAIR_BUCKETS)?,
                cur.expert(PAIR_BUCKETS)?,
            ];
            let stat = [cur.sparse()?, cur.sparse()?, cur.sparse()?];
            let hist = [cur.history()?, cur.history()?];
            layers.push(Layer {
                bias,
                geo,
                rec,
                pair,
                stat,
                hist,
            });
        }
        let net = if cur.u32()? == 0 {
            None
        } else {
            let (net, used) = Net::parse(&table[cur.pos..], codebooks, self_contained)?;
            cur.pos += used;
            Some(net)
        };
        ensure!(cur.pos == table.len(), "model table has trailing bytes");
        let table_hash = fnv1a(0xcbf2_9ce4_8422_2325, table);
        let (centroids, hash) = if let Some(stored) = stored {
            // The table holds everything the coder computes with, so it alone is hashed.
            if let Some(weights) = weights.filter(|w| !w.is_empty()) {
                ensure!(
                    Centroids::from_safetensors(weights, codebooks)?.values() == stored.values(),
                    "the weights do not round to the table's centroids"
                );
            }
            (stored, table_hash)
        } else {
            let weights = weights.context("this model table version needs the codec's weights")?;
            let centroids = Centroids::from_safetensors(weights, codebooks)?;
            let hash = centroids
                .values()
                .iter()
                .fold(table_hash, |h, v| fnv1a(h, &v.to_le_bytes()));
            (centroids, hash)
        };
        let recency_steps = State::new(&centroids, 1)?.recency_steps();
        Ok(Self {
            centroids,
            frac_bits,
            freq,
            layers,
            hash,
            recency_steps,
            simd: Simd::new(),
            net,
        })
    }

    /// Chooses the CPU dispatch of the dense kernels; every choice codes identically.
    pub const fn set_simd(&mut self, simd: Simd) {
        self.simd = simd;
        self.centroids.set_simd(simd);
    }

    /// Number of codebooks the model covers.
    #[must_use]
    pub const fn codebooks(&self) -> usize {
        self.layers.len()
    }

    /// Fractional bits of a logit.
    #[must_use]
    pub const fn frac_bits(&self) -> u32 {
        self.frac_bits
    }

    /// Hash of the table and the fixed-point centroids, recorded in coded files.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        format!("{:016x}", self.hash)
    }

    /// Hash of the table and the fixed-point centroids, as stored in `.pico` headers.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Starts predicting a stream with `num_codebooks` codes per frame.
    ///
    /// # Errors
    /// Fails when the model does not cover that many codebooks.
    pub fn predictor(&self, num_codebooks: usize) -> Result<Predictor<'_>> {
        Ok(Predictor {
            model: self,
            state: State::new(&self.centroids, num_codebooks)?,
            history: self
                .layers
                .iter()
                .take(num_codebooks)
                .flat_map(|layer| &layer.hist)
                .map(|hist| vec![0; hist.as_ref().map_or(0, |h| h.rank)])
                .collect(),
            buckets: vec![0; CARD],
            dots: vec![0; CARD],
            rec_values: self
                .layers
                .iter()
                .take(num_codebooks)
                .flat_map(|layer| {
                    let never = layer.rec.as_ref().map_or(0, |table| table[0]);
                    std::iter::repeat_n(never, CARD)
                })
                .collect(),
            rec_done: vec![0; num_codebooks],
            code_log: vec![Vec::new(); num_codebooks],
            net: self.net.as_ref().map(|net| net.state(num_codebooks)),
            logits: vec![0; CARD],
            freqs: vec![0; CARD],
        })
    }
}

/// Streaming predictor: yields integer frequencies for the next code, then learns it.
#[derive(Debug)]
pub struct Predictor<'a> {
    model: &'a Model,
    state: State<'a>,
    /// Decayed embedding sums, one per (codebook, history expert).
    history: Vec<Vec<i16>>,
    buckets: Vec<u16>,
    dots: Vec<i32>,
    /// Current recency-expert value of every candidate, per codebook.
    rec_values: Vec<i32>,
    /// Per codebook: the frame up to which `rec_values` has been brought up to date.
    rec_done: Vec<u32>,
    /// Codes since the last reset, per codebook.
    code_log: Vec<Vec<u16>>,
    net: Option<NetState>,
    logits: Vec<i32>,
    freqs: Vec<u32>,
}

fn add_table(out: &mut [i32], buckets: &[u16], table: &[i32]) {
    for (value, &bucket) in out.iter_mut().zip(buckets) {
        *value += table.get(usize::from(bucket)).copied().unwrap_or(0);
    }
}

impl Predictor<'_> {
    /// Forgets everything, as at the start of a stream. Used at reset points.
    pub fn reset(&mut self) {
        self.state.reset();
        for sums in &mut self.history {
            sums.fill(0);
        }
        if let Some(net) = &mut self.net {
            net.reset();
        }
        for (k, values) in self
            .rec_values
            .as_chunks_mut::<CARD>()
            .0
            .iter_mut()
            .enumerate()
        {
            let never = self.model.layers[k]
                .rec
                .as_ref()
                .map_or(0, |table| table[0]);
            values.fill(never);
        }
        self.rec_done.fill(0);
        for log in &mut self.code_log {
            log.clear();
        }
    }

    /// Brings the recency values of the next code's codebook up to the current frame.
    ///
    /// A candidate's bucket only changes when it is coded or when the distance to
    /// its last appearance crosses a class boundary, so only the codes that sat at
    /// those distances need a fresh lookup.
    fn refresh_recency(&mut self, table: &[i32]) {
        let k = self.state.codebook();
        let now = self.state.frame();
        let log = &self.code_log[k];
        for frame in self.rec_done[k]..=now {
            for &step in &self.model.recency_steps {
                let Some(code) = frame
                    .checked_sub(step)
                    .and_then(|at| log.get(usize::try_from(at).ok()?))
                else {
                    break;
                };
                let code = usize::from(*code);
                let bucket = usize::from(self.state.rec_bucket(k, code));
                self.rec_values[k * CARD + code] = table.get(bucket).copied().unwrap_or(0);
            }
        }
        self.rec_done[k] = now + 1;
    }

    /// Whether the incrementally kept recency values equal a full recomputation.
    fn recency_is_current(&self, table: &[i32]) -> bool {
        let k = self.state.codebook();
        self.rec_values[k * CARD..(k + 1) * CARD]
            .iter()
            .enumerate()
            .all(|(code, &value)| {
                let bucket = usize::from(self.state.rec_bucket(k, code));
                value == table.get(bucket).copied().unwrap_or(0)
            })
    }

    /// Codebook of the next code.
    #[must_use]
    pub const fn codebook(&self) -> usize {
        self.state.codebook()
    }

    /// Prior logits of the next code's codebook.
    #[must_use]
    pub fn bias(&self) -> &[i32] {
        &self.model.layers[self.state.codebook()].bias
    }

    /// Adds expert `index`'s logit contributions for the next code to `out`.
    ///
    /// Experts are numbered as in [`EXPERT_NAMES`]; `out` has one entry per code.
    pub fn add_expert(&mut self, index: usize, out: &mut [i32]) {
        let layer = &self.model.layers[self.state.codebook()];
        match index {
            0 => {
                if let Some(table) = &layer.geo {
                    self.state.geo_buckets(&mut self.buckets);
                    add_table(out, &self.buckets, table);
                }
            }
            1 => {
                if let Some(table) = &layer.rec {
                    self.refresh_recency(table);
                    debug_assert!(self.recency_is_current(table));
                    let k = self.state.codebook();
                    let values = &self.rec_values[k * CARD..(k + 1) * CARD];
                    for (value, &rec) in out.iter_mut().zip(values) {
                        *value += rec;
                    }
                }
            }
            _ if index < 2 + CONTEXTS => {
                let slot = index - 2;
                if let Some(table) = &layer.pair[slot] {
                    let (total, hits) = self.state.pair_hits(slot);
                    // Codes never seen after this context all share one bucket; only
                    // the difference to it matters after normalisation.
                    let base = table[pair_bucket(0, total)];
                    // Hit counts above 11 share a class, so twelve differences cover all.
                    let mut deltas = [0_i32; 12];
                    for (count, delta) in (0..).zip(deltas.iter_mut()) {
                        *delta = table[pair_bucket(count, total)] - base;
                    }
                    for &(code, count) in hits {
                        let class = usize::try_from(count).unwrap_or(usize::MAX).min(11);
                        if let Some(value) = out.get_mut(usize::from(code)) {
                            *value += deltas[class];
                        }
                    }
                }
            }
            _ if index < 2 + 2 * CONTEXTS => {
                let slot = index - 2 - CONTEXTS;
                let context = self.state.contexts()[slot];
                for (code, value) in layer.stat[slot].row(context) {
                    if let Some(entry) = out.get_mut(code) {
                        *entry += value;
                    }
                }
            }
            _ if index < 2 + 2 * CONTEXTS + HISTORIES => {
                let slot = index - 2 - 2 * CONTEXTS;
                if let Some(hist) = &layer.hist[slot] {
                    let sums = &self.history[self.state.codebook() * HISTORIES + slot];
                    self.dots.fill(0);
                    self.model.simd.accumulate(&mut self.dots, &hist.out, sums);
                    for (value, &dot) in out.iter_mut().zip(&self.dots) {
                        *value += dot >> hist.shift;
                    }
                }
            }
            _ if index == EXPERTS - 1 => {
                if let (Some(net), Some(state)) = (&self.model.net, &mut self.net) {
                    state.add(
                        net,
                        self.model.simd,
                        self.state.codebook(),
                        self.state.partial_sum(),
                        out,
                    );
                }
            }
            _ => {}
        }
    }

    /// Folds `code` into the history sum that `target`'s expert `slot` reads.
    fn remember(&mut self, target: usize, slot: usize, code: u16) {
        let Some(hist) = &self.model.layers[target].hist[slot] else {
            return;
        };
        let start = usize::from(code) * hist.rank;
        let embedding = &hist.emb[start..start + hist.rank];
        for (sum, &e) in self.history[target * HISTORIES + slot]
            .iter_mut()
            .zip(embedding)
        {
            let old = i32::from(*sum);
            let new = old - (old >> HISTORY_DECAY) + i32::from(e);
            *sum = i16::try_from(new.clamp(i32::from(i16::MIN), i32::from(i16::MAX))).unwrap_or(0);
        }
    }

    /// Frequencies of the next code (all positive) and their total.
    pub fn frequencies(&mut self) -> (&[u32], u64) {
        let mut logits = std::mem::take(&mut self.logits);
        logits.copy_from_slice(self.bias());
        for index in 0..EXPERTS {
            self.add_expert(index, &mut logits);
        }
        let table = &self.model.freq;
        let last = table.len() - 1;
        let best = logits.iter().copied().max().unwrap_or(0);
        let mut total = 0_u64;
        for (freq, &logit) in self.freqs.iter_mut().zip(&logits) {
            let below = usize::try_from(best.abs_diff(logit)).unwrap_or(usize::MAX);
            *freq = table[below.min(last)];
            total += u64::from(*freq);
        }
        self.logits = logits;
        (&self.freqs, total)
    }

    /// Records the code that was coded and advances to the next position.
    ///
    /// # Errors
    /// Fails when `code` is not a valid codebook entry.
    pub fn update(&mut self, code: u16) -> Result<()> {
        let k = self.state.codebook();
        self.state.update(code)?;
        if let Some(table) = &self.model.layers[k].rec {
            // The code just coded moves to the most recent class; its count may change.
            let bucket = usize::from(self.state.rec_bucket(k, usize::from(code)));
            self.rec_values[k * CARD + usize::from(code)] = table.get(bucket).copied().unwrap_or(0);
            self.code_log[k].push(code);
        }
        // Slot 0 follows the codebook itself. Slot 1 follows codebook 0 (including the
        // current frame), except for codebook 0, whose slot 1 follows codebook 1.
        self.remember(k, 0, code);
        match k {
            0 => {
                for target in 1..self.state.num_codebooks() {
                    self.remember(target, 1, code);
                }
            }
            1 => self.remember(0, 1, code),
            _ => {}
        }
        if self.state.codebook() == 0
            && let Some(net) = &mut self.net
        {
            net.push_frame(self.state.previous_sum());
        }
        Ok(())
    }
}
