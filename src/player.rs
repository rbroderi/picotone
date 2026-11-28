//! Playback of `.pico` files a few units at a time, for players that start
//! before the whole file is decoded and seek: the WebAssembly decoder and the
//! native player plugin.
//!
//! A [`Tables`] holds the cx2 entropy tables, found by the hash a stream
//! stores. A [`Net`] is one decoder network: a decoder-only EnCodec model, or
//! an HQ decoder of DAC 44 kHz codes. [`Player::open`] parses a file (and an
//! HQ file's `.picol` layer), and [`Player::decode_units`] decodes the next
//! part, returning only samples that later input can no longer change.
//!
//! Segmented models (`encodec_48khz`) decode each segment with
//! [`Encodec::decode_segment`] and stitch them with [`OverlapAdd`]; causal
//! models (`encodec_24khz`) run a [`StreamDecoder`] over the codes; HQ files
//! run through [`HqStream`], and at one to five codebooks through the
//! [`Declicker`]. The decoded audio then goes through the [`Limiter`]. Every
//! way, the output equals a whole-file decode bit for bit (a seek starts
//! decoding [`limiter::LOOKBACK`] samples early for the limiter, plus
//! [`declick::LOOKBACK`] when the declicker runs), and memory depends on the
//! chunk size, not the file length.

use std::collections::VecDeque;

use anyhow::{Context as _, Result, ensure};

use crate::ModelKind;
use crate::cx2::{self, hq::HqModel};
use crate::dac;
use crate::declick::{self, Declicker};
use crate::hqstream::{HqDecoder, HqStream};
use crate::layout::Trimmer;
use crate::limiter::{self, Limiter};
use crate::model::{Encodec, EncodedFrame, OverlapAdd, StreamDecoder};
use crate::nn::Seq;
use crate::pico::{self, Codec, DacInfo, DacStream, Info};

/// Name of the HQ code space in headers (`model`), whatever decodes it.
pub const HQ_MODEL: &str = "dac44";

/// Milliseconds from an arbitrary origin, for the decode-time statistics.
pub type Clock = fn() -> f64;

/// A [`Clock`] that does not time (always 0).
#[must_use]
pub const fn no_clock() -> f64 {
    0.0
}

/// The cx2 entropy tables a page can decode with, keyed by their hash:
/// EnCodec tables (`ECXM` version 5) and HQ tables (`ECXH`).
///
/// Tables are leaked to `'static` when added (players borrow them for as long
/// as they run), so add each once per process.
#[derive(Debug, Default)]
pub struct Tables {
    list: Vec<&'static cx2::Model>,
    hq: Vec<&'static HqModel>,
}

impl Tables {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a self-contained (version 5) `.ecxm` table or an HQ table and
    /// returns its hash. Adding a table twice keeps one copy.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid table of either kind.
    pub fn add(&mut self, table: &[u8]) -> Result<u64> {
        if let Some(hash) = cx2::hq::table_hash(table) {
            if self.find_hq(hash).is_none() {
                let model = HqModel::from_table(table)?;
                self.hq.push(Box::leak(Box::new(model)));
            }
            return Ok(hash);
        }
        let hash = cx2::model::self_contained_hash(table)
            .context("not a self-contained (version 5) cx2 table or an HQ table")?;
        if self.find(hash).is_none() {
            let model = cx2::Model::from_table(table)?;
            self.list.push(Box::leak(Box::new(model)));
        }
        Ok(hash)
    }

    /// Whether the table with this hash is loaded.
    #[must_use]
    pub fn has(&self, hash: u64) -> bool {
        self.find(hash).is_some() || self.find_hq(hash).is_some()
    }

    /// The EnCodec table with this hash.
    #[must_use]
    pub fn find(&self, hash: u64) -> Option<&'static cx2::Model> {
        self.list.iter().copied().find(|t| t.hash() == hash)
    }

    /// The HQ table with this hash.
    #[must_use]
    pub fn find_hq(&self, hash: u64) -> Option<&'static HqModel> {
        self.hq.iter().copied().find(|t| t.hash() == hash)
    }

    /// The table an EnCodec stream was coded with.
    ///
    /// # Errors
    /// Fails when it is not loaded.
    pub fn for_stream(&self, info: &Info) -> Result<&'static cx2::Model> {
        self.find(info.table_hash).with_context(|| {
            format!(
                "the cx2 table {:016x} this file was coded with is not loaded",
                info.table_hash
            )
        })
    }

    /// The table an HQ stream was coded with.
    ///
    /// # Errors
    /// Fails when it is not loaded.
    pub fn for_hq_stream(&self, info: &DacInfo) -> Result<&'static HqModel> {
        self.find_hq(info.table_hash).with_context(|| {
            format!(
                "the HQ cx2 table {:016x} this file was coded with is not loaded",
                info.table_hash
            )
        })
    }
}

/// A decoder network.
#[derive(Debug, Clone, Copy)]
pub enum Net {
    /// A decoder-only EnCodec model.
    Encodec(&'static Encodec),
    /// A decoder of DAC 44 kHz codes, for HQ files.
    Hq(&'static dyn HqDecoder),
}

impl Net {
    /// The code space of the files it decodes, as headers name it: an EnCodec
    /// model's name, or [`HQ_MODEL`].
    #[must_use]
    pub fn model(&self) -> &'static str {
        match self {
            Self::Encodec(model) => model.name(),
            Self::Hq(_) => HQ_MODEL,
        }
    }
}

/// The code space a file needs, as [`Net::model`] names it.
///
/// # Errors
/// Fails when the bytes are not a `.pico` file.
pub fn file_model(bytes: &[u8]) -> Result<&'static str> {
    Ok(match pico::codec_of(bytes)? {
        Codec::Encodec(kind) => kind.name(),
        Codec::Dac44 => HQ_MODEL,
    })
}

/// Decodes an EnCodec stream's pages on demand.
#[derive(Debug)]
pub struct Pages {
    stream: pico::Stream,
    table: &'static cx2::Model,
    next: usize,
    clock: Clock,
    cx2_ms: f64,
}

impl Pages {
    /// Starts at page 0 of `stream`.
    #[must_use]
    pub const fn new(stream: pico::Stream, table: &'static cx2::Model, clock: Clock) -> Self {
        Self {
            stream,
            table,
            next: 0,
            clock,
            cx2_ms: 0.0,
        }
    }

    /// The stream being read.
    #[must_use]
    pub const fn stream(&self) -> &pico::Stream {
        &self.stream
    }

    /// Milliseconds spent in the cx2 decode so far.
    #[must_use]
    pub const fn cx2_ms(&self) -> f64 {
        self.cx2_ms
    }

    /// The next page's segments (causal model: its frames), or `None` at the
    /// end.
    ///
    /// # Errors
    /// Fails when the page is damaged or missing.
    pub fn next_page(&mut self) -> Result<Option<Vec<EncodedFrame>>> {
        if self.next == self.stream.pages.len() {
            return Ok(None);
        }
        let start = (self.clock)();
        let frames = self.stream.decode_page(self.table, self.next)?;
        self.cx2_ms += (self.clock)() - start;
        self.next += 1;
        Ok(Some(frames))
    }
}

/// Final decoded samples from playable sample `offset` on, interleaved.
#[derive(Debug)]
pub struct Chunk {
    /// First sample, per channel, from the start of the playable audio.
    pub offset: usize,
    /// The samples, `[sample][channel]`.
    pub audio: Seq,
}

/// A `.pico` file being decoded a few units at a time.
///
/// A unit is one coded segment (about 1 s) for segmented models, one second of
/// latent frames for causal models, and one page of frames (80 by default,
/// 0.93 s) for HQ files.
#[derive(Debug)]
pub struct Player {
    inner: Inner,
    model: &'static str,
    sample_rate: usize,
    channels: usize,
    audio_length: usize,
    looped: bool,
    num_codebooks: usize,
    layered: bool,
    /// Declicks HQ decodes at low tiers, before the limiter.
    declicker: Option<Declicker>,
    /// Limits the decoded audio; `None` once it has finished.
    limiter: Option<Limiter>,
    /// Limited samples still to drop: the pre-roll of a seek.
    skip: usize,
    /// Playable sample the next returned chunk starts at.
    emitted: usize,
    clock: Clock,
    cx2_ms: f64,
    network_ms: f64,
}

/// The decode behind a [`Player`].
#[derive(Debug)]
enum Inner {
    Encodec(Box<EncodecStream>),
    Hq(Box<HqStream<'static>>),
}

impl Player {
    /// Parses `bytes` (and, for an HQ file, its enhancement layer `layer`)
    /// for network `net`; the first returned sample is playable sample
    /// `start_sample`. Decoding starts [`limiter::LOOKBACK`] samples before
    /// it, for the limiter: the 48 kHz model at the segment that sample needs
    /// and HQ files a few frames before it; the causal 24 kHz model and loops
    /// decode from the start and drop what comes before.
    ///
    /// # Errors
    /// Fails when the file is malformed or truncated, needs another model, is
    /// coded with a table `tables` lacks, or `start_sample` is past the end;
    /// also when `layer` is given for an EnCodec file, is not this file's, or
    /// any of its pages is damaged or missing (the caller then opens the file
    /// alone).
    pub fn open(
        net: Net,
        bytes: &[u8],
        layer: Option<&[u8]>,
        tables: &Tables,
        start_sample: usize,
        clock: Clock,
    ) -> Result<Self> {
        let model = file_model(bytes)?;
        ensure!(
            model == net.model(),
            "this file needs model {model}, but {} is loaded",
            net.model()
        );
        let mut pre_roll = start_sample.min(limiter::LOOKBACK);
        let mut first = start_sample - pre_roll;
        let mut declicker = None;
        let (inner, sample_rate, channels, num_codebooks, audio_length, looped) = match net {
            Net::Encodec(encodec) => {
                ensure!(layer.is_none(), "only HQ files have enhancement layers");
                let stream = EncodecStream::new(encodec, bytes, tables, first, clock)?;
                ensure!(
                    start_sample < stream.info.playable_samples(),
                    "start sample {start_sample} is past the end"
                );
                let (rate, channels) = match stream.info.model {
                    ModelKind::Mono24k => (24_000, 1),
                    ModelKind::Stereo48k => (48_000, 2),
                };
                let info = &stream.info;
                let facts = (
                    rate,
                    channels,
                    info.num_codebooks,
                    info.playable_samples(),
                    info.looped,
                );
                (
                    Inner::Encodec(Box::new(stream)),
                    facts.0,
                    facts.1,
                    facts.2,
                    facts.3,
                    facts.4,
                )
            }
            Net::Hq(decoder) => {
                let mut stream = DacStream::parse(bytes)?;
                if let Some(layer) = layer {
                    stream.attach_layer(layer)?;
                }
                let table = tables.for_hq_stream(&stream.info)?;
                let info = &stream.info;
                let facts = (
                    info.layout.output_channels(),
                    info.codebooks.iter().copied().max().unwrap_or(0),
                    info.playable_samples(),
                    info.looped,
                );
                ensure!(
                    start_sample < facts.2,
                    "start sample {start_sample} is past the end ({} samples)",
                    facts.2
                );
                if declick::applies(facts.1) {
                    pre_roll = start_sample.min(limiter::LOOKBACK + declick::LOOKBACK);
                    first = start_sample - pre_roll;
                    declicker = Some(Declicker::new(facts.0, info.pre_skip, first + info.pre_skip));
                }
                let stream = HqStream::new(stream, table, decoder, first)?;
                (
                    Inner::Hq(Box::new(stream)),
                    usize::try_from(dac::SAMPLE_RATE)?,
                    facts.0,
                    facts.1,
                    facts.2,
                    facts.3,
                )
            }
        };
        Ok(Self {
            inner,
            model,
            sample_rate,
            channels,
            audio_length,
            looped,
            num_codebooks,
            layered: layer.is_some(),
            declicker,
            limiter: Some(Limiter::new(channels)),
            skip: pre_roll,
            emitted: start_sample,
            clock,
            cx2_ms: 0.0,
            network_ms: 0.0,
        })
    }

    /// Decodes up to `max_units` more units and returns the samples that
    /// became final, or `None` once the whole file has been returned. A
    /// returned chunk is never empty.
    ///
    /// # Errors
    /// Fails when a page is damaged or the codes do not fit the network.
    pub fn decode_units(&mut self, max_units: usize) -> Result<Option<Chunk>> {
        loop {
            if self.limiter.is_none() {
                return Ok(None);
            }
            let raw = self.decode_raw(max_units)?;
            let start = (self.clock)();
            let mut audio = if let Some(chunk) = &raw {
                let declicked = match self.declicker.as_mut() {
                    Some(d) => Some(d.push(&chunk.audio)?),
                    None => None,
                };
                self.limiter
                    .as_mut()
                    .context("player already finished")?
                    .push(declicked.as_ref().unwrap_or(&chunk.audio))?
            } else {
                let mut limiter = self.limiter.take().context("player already finished")?;
                let head = match self.declicker.take() {
                    Some(d) => limiter.push(&d.finish()?)?,
                    None => Seq::zeros(0, self.channels),
                };
                Seq::concat(vec![head, limiter.finish()?])?
            };
            drop(raw);
            if self.skip > 0 {
                let cut = self.skip.min(audio.len());
                self.skip -= cut;
                audio = audio.slice_rows(cut, audio.len());
            }
            self.network_ms += (self.clock)() - start;
            if !audio.is_empty() {
                let offset = self.emitted;
                self.emitted += audio.len();
                return Ok(Some(Chunk { offset, audio }));
            }
        }
    }

    /// The decoder's samples before the limiter: up to `max_units` more
    /// units, or `None` at the end.
    fn decode_raw(&mut self, max_units: usize) -> Result<Option<Chunk>> {
        let start = (self.clock)();
        match &mut self.inner {
            Inner::Encodec(s) => {
                let before = s.pages.cx2_ms();
                let chunk = s.decode_units(max_units)?;
                let cx2 = s.pages.cx2_ms() - before;
                self.cx2_ms += cx2;
                self.network_ms += (self.clock)() - start - cx2;
                Ok(chunk)
            }
            Inner::Hq(s) => {
                s.fetch(max_units)?;
                let fetched = (self.clock)();
                let part = s.decode_next(max_units)?;
                self.cx2_ms += fetched - start;
                self.network_ms += (self.clock)() - fetched;
                Ok(part.map(|(offset, audio)| Chunk { offset, audio }))
            }
        }
    }

    /// The file's code space, as [`Net::model`] names it.
    #[must_use]
    pub const fn model(&self) -> &'static str {
        self.model
    }

    /// Output sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Output channel count.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Audio length in samples per channel, after the pre-skip and end trim.
    #[must_use]
    pub const fn audio_length(&self) -> usize {
        self.audio_length
    }

    /// Whether the file is a seamless loop.
    #[must_use]
    pub const fn is_loop(&self) -> bool {
        self.looped
    }

    /// Codebooks per latent frame; for HQ files the largest count of any
    /// coded channel (with the layer, when one is attached).
    #[must_use]
    pub const fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Whether an enhancement layer is attached.
    #[must_use]
    pub const fn is_layered(&self) -> bool {
        self.layered
    }

    /// Milliseconds the cx2 (entropy) decode has taken so far.
    #[must_use]
    pub const fn cx2_ms(&self) -> f64 {
        self.cx2_ms
    }

    /// Milliseconds everything else (the network, stereo, trim) has taken.
    #[must_use]
    pub const fn network_ms(&self) -> f64 {
        self.network_ms
    }

    /// Units this player decodes.
    #[must_use]
    pub const fn unit_count(&self) -> usize {
        match &self.inner {
            Inner::Encodec(s) => s.unit_count(),
            Inner::Hq(s) => s.unit_count(),
        }
    }

    /// Units decoded so far.
    #[must_use]
    pub const fn units_done(&self) -> usize {
        match &self.inner {
            Inner::Encodec(s) => s.units_done(),
            Inner::Hq(s) => s.units_done(),
        }
    }
}

/// How an EnCodec stream turns codes into audio.
#[derive(Debug)]
enum Source {
    /// Normalised segments, decoded one by one and overlap-added.
    Segmented {
        /// Decoded segments not yet run through the network.
        queue: VecDeque<EncodedFrame>,
        /// Segments to drop from the first page (a seek into the page).
        skip: usize,
        /// Segments run through the network so far, counted from `first`.
        done: usize,
        /// Segments from the first decoded one to the end.
        count: usize,
        overlap: Option<OverlapAdd>,
    },
    /// One causal stream of latent frames.
    Causal {
        /// Codes decoded but not yet pushed, frame-major.
        codes: Vec<u16>,
        num_codebooks: usize,
        /// Latent frames per unit (one second).
        unit_frames: usize,
        /// Latent frames in the file.
        total: usize,
        /// Latent frames pushed so far.
        next: usize,
        decoder: Option<StreamDecoder<'static>>,
    },
}

/// An EnCodec `.pico` file being decoded a few units at a time.
#[derive(Debug)]
struct EncodecStream {
    model: &'static Encodec,
    info: Info,
    pages: Pages,
    source: Source,
    /// Removes the pre-skip and end trim and joins loops.
    trimmer: Option<Trimmer>,
    /// Coded samples the raw output stops at, counted from its first sample.
    raw_length: usize,
    /// Trimmed samples still to drop before the first returned one.
    drop: usize,
    /// Playable sample the next returned chunk starts at.
    emitted: usize,
}

impl EncodecStream {
    fn new(
        model: &'static Encodec,
        bytes: &[u8],
        tables: &Tables,
        start_sample: usize,
        clock: Clock,
    ) -> Result<Self> {
        let stream = pico::Stream::parse(bytes)?;
        let info = stream.info.clone();
        ensure!(
            info.model.name() == model.name(),
            "this file needs model {}, but {} is loaded",
            info.model.name(),
            model.name()
        );
        let playable = info.playable_samples();
        ensure!(playable > 0, "the file holds no audio");
        ensure!(
            start_sample < playable,
            "start sample {start_sample} is past the end ({playable} samples)"
        );
        let table = tables.for_stream(&info)?;
        let cfg = model.config();
        let rate = cfg.sampling_rate;
        let channels = cfg.audio_channels;
        // A seek starts at a later segment only where nothing before it matters:
        // segmented, and not a loop (a loop's end blends into its start).
        let first_segment = match (info.stride, info.looped) {
            (Some(stride), false) if start_sample > 0 => {
                first_needed_segment(&info, stride, start_sample + info.pre_skip)?
            }
            _ => 0,
        };
        let raw_start = first_segment * info.stride.unwrap_or(0);
        let (trimmer, drop) = if first_segment == 0 {
            let trimmer = Trimmer::new(info.pre_skip, playable, info.looped, rate, channels)?;
            (trimmer, start_sample)
        } else {
            let pre_skip = start_sample + info.pre_skip - raw_start;
            let trimmer = Trimmer::new(pre_skip, playable - start_sample, false, rate, channels)?;
            (trimmer, 0)
        };
        let first_page = stream
            .spans
            .iter()
            .position(|span| span.segments.contains(&first_segment))
            .context("no page holds the start segment")?;
        let skip = first_segment - stream.spans[first_page].segments.start;
        let segment_count = stream.segment_frames.len();
        let total_frames: usize = stream.segment_frames.iter().sum();
        let source = if info.stride.is_some() {
            Source::Segmented {
                queue: VecDeque::new(),
                skip,
                done: 0,
                count: segment_count - first_segment,
                overlap: Some(model.overlap_add_with(info.segmenting())?),
            }
        } else {
            Source::Causal {
                codes: Vec::new(),
                num_codebooks: info.num_codebooks,
                unit_frames: cfg.frame_rate().max(1),
                total: total_frames,
                next: 0,
                decoder: Some(model.stream_decoder(info.num_codebooks)?),
            }
        };
        let mut pages = Pages::new(stream, table, clock);
        pages.next = first_page;
        Ok(Self {
            model,
            raw_length: info.audio_length - raw_start,
            info,
            pages,
            source,
            trimmer: Some(trimmer),
            drop,
            emitted: start_sample,
        })
    }

    /// Decodes up to `max_units` more units and trims them; `None` once
    /// everything was returned. Keeps decoding while the trim swallows
    /// everything, so a returned chunk is never empty.
    fn decode_units(&mut self, max_units: usize) -> Result<Option<Chunk>> {
        loop {
            let Some((raw, finished)) = self.decode_raw(max_units)? else {
                return Ok(None);
            };
            let trimmer = self.trimmer.as_mut().context("stream already finished")?;
            let mut audio = trimmer.push(&raw)?;
            drop(raw);
            if finished {
                self.trimmer
                    .take()
                    .context("stream already finished")?
                    .finish()?;
            }
            if self.drop > 0 {
                let cut = self.drop.min(audio.len());
                self.drop -= cut;
                audio = audio.slice_rows(cut, audio.len());
            }
            if audio.is_empty() {
                if finished {
                    return Ok(None);
                }
                continue;
            }
            let offset = self.emitted;
            self.emitted += audio.len();
            return Ok(Some(Chunk { offset, audio }));
        }
    }

    /// Units this stream decodes.
    const fn unit_count(&self) -> usize {
        match &self.source {
            Source::Segmented { count, .. } => *count,
            Source::Causal {
                total, unit_frames, ..
            } => total.div_ceil(*unit_frames),
        }
    }

    /// Units decoded so far.
    const fn units_done(&self) -> usize {
        match &self.source {
            Source::Segmented { done, .. } => *done,
            Source::Causal {
                next, unit_frames, ..
            } => next.div_ceil(*unit_frames),
        }
    }

    /// Decodes up to `max_units` more units of coded samples, untrimmed, and
    /// whether they are the last; `None` once everything was returned.
    fn decode_raw(&mut self, max_units: usize) -> Result<Option<(Seq, bool)>> {
        let max_units = max_units.max(1);
        let raw_length = self.raw_length;
        let model = self.model;
        let pages = &mut self.pages;
        let audio = match &mut self.source {
            Source::Segmented {
                queue,
                skip,
                done,
                count,
                overlap,
            } => {
                let Some(acc) = overlap.as_mut() else {
                    return Ok(None);
                };
                let want = max_units.min(*count - *done);
                while queue.len() < want {
                    let frames = pages.next_page()?.context("the stream ended early")?;
                    queue.extend(frames.into_iter().skip(*skip));
                    *skip = 0;
                }
                let batch: Vec<EncodedFrame> = queue.drain(..want).collect();
                let mut parts = Vec::with_capacity(batch.len() + 1);
                // Together: the LSTM advances all segments of the batch in
                // one pass over its weights per time step, and the pool
                // shares the convolutions of a few segments at a time.
                for segment in &model.decode_segments(&batch)? {
                    parts.push(acc.push(segment)?);
                }
                *done += want;
                if *done == *count {
                    let acc = overlap.take().context("overlap-add already finished")?;
                    parts.push(acc.finish(raw_length)?);
                }
                Seq::concat(parts)?
            }
            Source::Causal {
                codes,
                num_codebooks,
                unit_frames,
                total,
                next,
                decoder,
            } => {
                let Some(stream) = decoder.as_mut() else {
                    return Ok(None);
                };
                let nc = (*num_codebooks).max(1);
                let step = max_units.saturating_mul(*unit_frames);
                let mut parts = Vec::new();
                // The first frames are held back until the reflection padding
                // is known, so keep pushing until some audio comes out.
                loop {
                    let end = next.saturating_add(step).min(*total);
                    while codes.len() < (end - *next) * nc {
                        let frames = pages.next_page()?.context("the stream ended early")?;
                        codes.extend(frames.into_iter().flat_map(|f| f.codes));
                    }
                    let take = (end - *next) * nc;
                    let audio = stream.push(&codes[..take])?;
                    drop(codes.drain(..take));
                    *next = end;
                    let got = audio.len();
                    parts.push(audio);
                    if end == *total {
                        let stream = decoder.take().context("stream already finished")?;
                        parts.push(stream.finish(raw_length)?);
                        break;
                    }
                    if got > 0 {
                        break;
                    }
                }
                Seq::concat(parts)?
            }
        };
        let finished = match &self.source {
            Source::Segmented { overlap, .. } => overlap.is_none(),
            Source::Causal { decoder, .. } => decoder.is_none(),
        };
        Ok(Some((audio, finished)))
    }
}

/// The first segment whose output reaches coded sample `coded_start`
/// unchanged by the segments before it: the last one that starts after the
/// previous segment ends, at or before `coded_start`.
fn first_needed_segment(info: &Info, stride: usize, coded_start: usize) -> Result<usize> {
    let layout = info.layout()?;
    let length = layout
        .segment_length()
        .context("a strided model has a segment length")?;
    let count = layout.segment_count(info.audio_length);
    // Segment j > 0 is alone from (j - 1) * stride + length on.
    let mut first = 0;
    for j in 1..count {
        if (j - 1) * stride + length <= coded_start {
            first = j;
        } else {
            break;
        }
    }
    Ok(first)
}
