//! Decoding HQ streams (model 2) a part at a time, for playback.
//!
//! An [`HqStream`] range-decodes a [`DacStream`]'s pages as it needs them,
//! runs the codes through an [`HqDecoder`] (the network that turns DAC codes
//! into audio), rebuilds the output channels from the coded ones (mid/side,
//! parametric stereo) and trims the result for playback. Each call returns
//! only samples that later input can no longer change, so playback can start
//! before the whole file is decoded, and the output equals a whole-file
//! decode bit for bit (with the same arithmetic), whatever the part sizes.
//! It can also start at a later sample (a seek), decoding only a little
//! before it.
//!
//! [`HqDecoder`] is the seam between the stream and the network:
//! [`crate::vocos::VocosDecoder`] implements it, and any other decoder of
//! DAC 44 kHz codes can too. The stream needs only to know
//! how much context the network needs around a span of frames.

use std::fmt;
use std::ops::Range;

use anyhow::{Context as _, Result, ensure};
use rayon::prelude::*;

use crate::cx2::hq::HqModel;
use crate::dac::{self, HOP};
use crate::layout::Trimmer;
use crate::nn::Seq;
use crate::pico::{ChannelLayout, DEFAULT_PAGE_FRAMES, DacInfo, DacStream};
use crate::pstereo::{self, Upmixer};

/// How much of the code stream around a span of frames a decoder reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    /// Frames of context on each side of a span, enough that its samples
    /// equal those of a whole-file decode.
    pub frames: usize,
    /// A decoded window starts and ends at multiples of this many frames (or
    /// at the stream's end), so that every window computes in step with a
    /// whole-file decode (1 when the decoder does not care).
    pub align: usize,
}

/// A network that turns DAC 44 kHz codes into mono 44.1 kHz audio, 512
/// samples per frame. The output at a frame depends only on the codes within
/// [`HqDecoder::context`] frames of it.
pub trait HqDecoder: fmt::Debug + Send + Sync {
    /// The decoder's name (its weight files are `<name>.safetensors`).
    fn name(&self) -> &str;

    /// FNV-1a 64 of the DAC codebooks the decoder reads codes with; a stream
    /// names the same ([`DacInfo::codebook_hash`]).
    fn codebook_hash(&self) -> u64;

    /// The context it needs around a span.
    fn context(&self) -> Context;

    /// Most frames one [`HqDecoder::decode_window`] call should decode
    /// (before context); longer spans are split into pieces of at most this
    /// many, decoded in parallel. A decoder with a wide context wants long
    /// pieces, which recompute less of it.
    fn piece(&self) -> usize;

    /// Decodes `codes`, `[frames][num_codebooks]`, as if they were a whole
    /// file: `frames * 512` samples.
    ///
    /// # Errors
    /// Fails when the codes do not fit the decoder.
    fn decode_window(&self, codes: &[u16], num_codebooks: usize) -> Result<Vec<f32>>;

    /// The frames to decode for frames `range` of a stream of `frames`.
    fn window(&self, range: Range<usize>, frames: usize) -> Range<usize> {
        let context = self.context();
        let align = context.align.max(1);
        let start = range.start.saturating_sub(context.frames) / align * align;
        start..((range.end + context.frames).div_ceil(align) * align).min(frames)
    }

    /// Decodes frames `range` of a stream of `frames`: `range.len() * 512`
    /// samples. `codes` holds the stream's codes from frame `first` on.
    ///
    /// # Errors
    /// Fails when `codes` lack the frames the window needs, or as
    /// [`HqDecoder::decode_window`].
    fn decode_range(
        &self,
        codes: &[u16],
        num_codebooks: usize,
        first: usize,
        range: Range<usize>,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let window = self.window(range.clone(), frames);
        let nc = num_codebooks.max(1);
        ensure!(
            window.start >= first && (window.end - first) * nc <= codes.len(),
            "the codes of frames {window:?} are not all decoded"
        );
        let y = self.decode_window(
            &codes[(window.start - first) * nc..(window.end - first) * nc],
            nc,
        )?;
        Ok(y[(range.start - window.start) * HOP..(range.end - window.start) * HOP].to_vec())
    }
}

/// How the coded channels become output channels.
#[derive(Debug)]
enum Stereo {
    /// Mono or left/right: as coded.
    Plain,
    /// Mid and side to left and right.
    MidSide,
    /// The mid, upmixed with the stream's parameters.
    Parametric(Box<Upmixer>),
}

/// An HQ stream being decoded a part at a time.
#[derive(Debug)]
pub struct HqStream<'a> {
    stream: DacStream,
    table: &'a HqModel,
    decoder: &'a dyn HqDecoder,
    /// Frames per coded channel.
    frames: usize,
    /// Frames per unit of [`HqStream::decode_next`]: the page length.
    unit: usize,
    /// First frame the network decodes.
    first_frame: usize,
    /// Next frame the network decodes.
    next_frame: usize,
    /// Per coded channel, the codes decoded so far from frame `codes_first`.
    codes: Vec<Vec<u16>>,
    codes_first: usize,
    /// Frame the decoded codes reach.
    codes_end: usize,
    /// Next page to range-decode.
    next_page: usize,
    stereo: Stereo,
    /// Removes the pre-skip and end trim and joins loops.
    trimmer: Option<Trimmer>,
    /// Trimmed samples still to drop before the first returned one.
    drop: usize,
    /// Playable sample the next returned part starts at.
    emitted: usize,
}

/// A span is split for idle threads only into pieces of at least this many
/// contexts, so that the recomputed context stays a small share.
const MIN_PIECE_CONTEXTS: usize = 4;

impl<'a> HqStream<'a> {
    /// Opens `stream` for decoding with `table` and `decoder`; the first
    /// returned sample is playable sample `start_sample`. Loops (whose end
    /// blends into their start) decode from the start and drop what comes
    /// before.
    ///
    /// # Errors
    /// Fails when the table or the decoder's codebooks are not the stream's,
    /// the stream holds no audio, `start_sample` is past its end, or (for
    /// parametric stereo) a page's parameters are damaged.
    pub fn new(
        stream: DacStream,
        table: &'a HqModel,
        decoder: &'a dyn HqDecoder,
        start_sample: usize,
    ) -> Result<Self> {
        let info = &stream.info;
        ensure!(
            table.hash() == info.table_hash,
            "the stream was coded with HQ cx2 table {:016x}, not {:016x}",
            info.table_hash,
            table.hash()
        );
        ensure!(
            decoder.codebook_hash() == info.codebook_hash,
            "the file's codes index codebooks {:016x}, the decoder {} has {:016x}",
            info.codebook_hash,
            decoder.name(),
            decoder.codebook_hash()
        );
        let playable = info.playable_samples();
        ensure!(playable > 0, "the file holds no audio");
        ensure!(
            start_sample < playable,
            "start sample {start_sample} is past the end ({playable} samples)"
        );
        let frames = info.frames();
        let rate = usize::try_from(dac::SAMPLE_RATE)?;
        let channels = info.layout.output_channels();
        // The first frame whose samples are returned; a seek starts there.
        let block = if start_sample > 0 && !info.looped {
            (start_sample + info.pre_skip) / HOP
        } else {
            0
        };
        let (trimmer, drop) = if block == 0 {
            let trimmer = Trimmer::new(info.pre_skip, playable, info.looped, rate, channels)?;
            (trimmer, start_sample)
        } else {
            let pre_skip = start_sample + info.pre_skip - block * HOP;
            let trimmer = Trimmer::new(pre_skip, playable - start_sample, false, rate, channels)?;
            (trimmer, 0)
        };
        let (stereo, first_frame) = match info.layout {
            ChannelLayout::Mono | ChannelLayout::LeftRight => (Stereo::Plain, block),
            ChannelLayout::MidSide => (Stereo::MidSide, block),
            ChannelLayout::Parametric(config) => {
                let pages = (0..stream.pages.len())
                    .map(|p| stream.page_parameters(p))
                    .collect::<Result<Vec<_>>>()?;
                let params = pstereo::read_pages(&pages, &stream.spans, config)?;
                let upmixer = Upmixer::new(params, &stream.spans, config, frames * HOP, block)?;
                let first = upmixer.first_input() / HOP;
                (Stereo::Parametric(Box::new(upmixer)), first)
            }
        };
        let window = decoder.window(first_frame..(first_frame + 1).min(frames), frames);
        let next_page = stream
            .spans
            .iter()
            .position(|span| span.contains(&window.start))
            .unwrap_or(0);
        let codes_first = stream.spans.get(next_page).map_or(0, |s| s.start);
        let unit = match info.page_length {
            0 => DEFAULT_PAGE_FRAMES,
            n => n,
        };
        Ok(Self {
            codes: vec![Vec::new(); info.codebooks.len()],
            codes_first,
            codes_end: codes_first,
            next_page,
            frames,
            unit,
            first_frame,
            next_frame: first_frame,
            stream,
            table,
            decoder,
            stereo,
            trimmer: Some(trimmer),
            drop,
            emitted: start_sample,
        })
    }

    /// The stream's header.
    #[must_use]
    pub const fn info(&self) -> &DacInfo {
        &self.stream.info
    }

    /// Units (pages of frames) the stream decodes from its first frame.
    #[must_use]
    pub const fn unit_count(&self) -> usize {
        (self.frames - self.first_frame).div_ceil(self.unit)
    }

    /// Units decoded so far.
    #[must_use]
    pub const fn units_done(&self) -> usize {
        (self.next_frame - self.first_frame).div_ceil(self.unit)
    }

    /// The frames the next [`HqStream::decode_next`] of `max_units` decodes.
    fn next_range(&self, max_units: usize) -> Range<usize> {
        let end = self
            .next_frame
            .saturating_add(max_units.max(1).saturating_mul(self.unit))
            .min(self.frames);
        self.next_frame..end
    }

    /// Range-decodes the pages the next [`HqStream::decode_next`] of
    /// `max_units` needs (that call does it otherwise; this lets a caller
    /// time the entropy decode on its own). Pages decode in parallel.
    ///
    /// # Errors
    /// Fails when a page is damaged or missing.
    pub fn fetch(&mut self, max_units: usize) -> Result<()> {
        let range = self.next_range(max_units);
        if range.is_empty() {
            return Ok(());
        }
        let need = self.decoder.window(range, self.frames).end;
        let mut last = self.next_page;
        while last < self.stream.spans.len() && self.stream.spans[last].start < need {
            last += 1;
        }
        let pages = (self.next_page..last)
            .into_par_iter()
            .map(|index| self.stream.decode_page(self.table, index))
            .collect::<Result<Vec<_>>>()?;
        for page in pages {
            for (all, codes) in self.codes.iter_mut().zip(page) {
                all.extend_from_slice(&codes);
            }
        }
        if last > self.next_page {
            self.codes_end = self.stream.spans[last - 1].end;
        }
        self.next_page = last;
        Ok(())
    }

    /// Decodes up to `max_units` more units (pages of frames) and returns the
    /// samples that became final, trimmed, `[sample][channel]`, with the
    /// playable sample they start at; `None` once everything was returned.
    /// Keeps decoding while the trim swallows everything, so a returned part
    /// is never empty.
    ///
    /// # Errors
    /// Fails when a page is damaged or missing, or the decoder fails.
    pub fn decode_next(&mut self, max_units: usize) -> Result<Option<(usize, Seq)>> {
        loop {
            if self.trimmer.is_none() {
                return Ok(None);
            }
            let raw = self.decode_raw(max_units)?;
            let finished = self.next_frame == self.frames && !self.upmix_pending();
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
            return Ok(Some((offset, audio)));
        }
    }

    /// Decodes everything left at once (the pieces run in parallel).
    ///
    /// # Errors
    /// As [`HqStream::decode_next`].
    pub fn decode_rest(&mut self) -> Result<Seq> {
        let mut parts = Vec::new();
        while let Some((_, audio)) = self.decode_next(usize::MAX)? {
            parts.push(audio);
        }
        if parts.is_empty() {
            return Ok(Seq::zeros(0, self.info().layout.output_channels()));
        }
        Seq::concat(parts)
    }

    /// Whether the upmix still holds samples (it runs one frame behind).
    const fn upmix_pending(&self) -> bool {
        matches!(self.stereo, Stereo::Parametric(_))
    }

    /// Decodes the next frames through the network and rebuilds the output
    /// channels: untrimmed samples, `[sample][channel]`.
    fn decode_raw(&mut self, max_units: usize) -> Result<Seq> {
        let range = self.next_range(max_units);
        let channels = self.info().layout.output_channels();
        if range.is_empty() {
            // Only the upmix's last frames are left.
            return match &mut self.stereo {
                Stereo::Parametric(upmixer) => interleave(upmixer.finish()?),
                _ => Ok(Seq::zeros(0, channels)),
            };
        }
        self.fetch(max_units)?;
        let signals = self.network(range.clone())?;
        self.next_frame = range.end;
        // Codes before the next window are not needed again.
        if range.end < self.frames {
            let keep = self
                .decoder
                .window(range.end..range.end + 1, self.frames)
                .start;
            if keep > self.codes_first {
                for (codes, &n) in self.codes.iter_mut().zip(&self.stream.info.codebooks) {
                    drop(codes.drain(..(keep - self.codes_first) * n));
                }
                self.codes_first = keep;
            }
        }
        let last = range.end == self.frames;
        match &mut self.stereo {
            Stereo::Plain => interleave(signals),
            Stereo::MidSide => {
                let mut it = signals.into_iter();
                let (mid, side) = (it.next().unwrap_or_default(), it.next().unwrap_or_default());
                let left = mid.iter().zip(&side).map(|(m, s)| m + s).collect();
                let right = mid.iter().zip(&side).map(|(m, s)| m - s).collect();
                interleave([left, right])
            }
            Stereo::Parametric(upmixer) => {
                let mid = signals.into_iter().next().unwrap_or_default();
                let [mut left, mut right] = upmixer.push(&mid)?;
                if last {
                    let [l, r] = upmixer.finish()?;
                    left.extend(l);
                    right.extend(r);
                    // Nothing is held back any more.
                    self.stereo = Stereo::Plain;
                }
                interleave([left, right])
            }
        }
    }

    /// Each coded channel's samples of frames `range`, decoded in pieces of
    /// at most [`HqDecoder::piece`] frames (more, shorter ones while threads
    /// would idle), all pieces of all channels in parallel.
    fn network(&self, range: Range<usize>) -> Result<Vec<Vec<f32>>> {
        let channels = self.codes.len().max(1);
        let context = self.decoder.context();
        let shortest = MIN_PIECE_CONTEXTS * context.frames.max(1);
        // The frames a piece decodes beyond its own: context both sides,
        // rounded out to the alignment.
        let spread = rayon::current_num_threads()
            .max(1)
            .div_ceil(channels)
            .min(range.len() / shortest)
            .max(1);
        let pieces = range
            .len()
            .div_ceil(self.decoder.piece().max(1))
            .max(spread);
        let size = range.len().div_ceil(pieces.max(1));
        let jobs: Vec<(usize, Range<usize>)> = (0..self.codes.len())
            .flat_map(|c| {
                (0..pieces).map(move |p| {
                    let start = range.start + p * size;
                    (c, start..(start + size).min(range.end))
                })
            })
            .collect();
        let decoded = jobs
            .par_iter()
            .map(|(c, piece)| {
                self.decoder.decode_range(
                    &self.codes[*c],
                    self.stream.info.codebooks[*c],
                    self.codes_first,
                    piece.clone(),
                    self.frames,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(decoded
            .chunks(pieces.max(1))
            .map(<[Vec<f32>]>::concat)
            .collect())
    }
}

/// Planar channels to one `[sample][channel]` sequence.
fn interleave<C: AsRef<[Vec<f32>]>>(channels: C) -> Result<Seq> {
    let channels = channels.as_ref();
    let length = channels.first().map_or(0, Vec::len);
    let count = channels.len().max(1);
    let mut data = vec![0.0_f32; length * count];
    for (c, channel) in channels.iter().enumerate() {
        ensure!(channel.len() == length, "channels of different lengths");
        for (t, &v) in channel.iter().enumerate() {
            data[t * count + c] = v;
        }
    }
    Seq::from_vec(length, count, data)
}
