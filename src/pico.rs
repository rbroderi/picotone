//! The `.pico` file format: EnCodec codes, cx2-coded, in Ogg (RFC 3533).
//!
//! ```text
//! page 0 (BOS)  identification header: "Pico", version, model, scheme, table hash,
//!               codebooks, flags, pre-skip, stride, page length, [length], TLVs
//! page 1        comment header, only when the flags say so: "PicoTags", vendor,
//!               KEY=value comments (Vorbis comment layout)
//! page 2..      one packet per page: one independently decodable cx2 chunk
//!               (the segments' 12-bit scales, then their codes). Only the last
//!               packet of a stream without a length field starts with its end trim.
//! ```
//!
//! Every page has the `OggS` capture pattern, a sequence number and a CRC, so
//! damage is detected, confined to one page, and a reader can resynchronise.
//! The granule position of a page is the number of decoded samples that are
//! final once that page is decoded; the last page's granule position is where
//! the audio ends. A player drops the first `pre_skip` decoded samples and
//! everything from the final granule position on, as in Ogg Opus; with the
//! loop flag set, what remains repeats seamlessly. `FORMAT.md` in the
//! repository describes every field.

use std::ops::Range;

use anyhow::{Context as _, Result, bail, ensure};

use crate::ModelKind;
use crate::cx2;
use crate::layout::{Crossfade, Segmenting};
use crate::model::EncodedFrame;
use crate::scale;

const CAPTURE: &[u8; 4] = b"OggS";
/// Start of the identification header.
pub const HEAD_MAGIC: &[u8; 4] = b"Pico";
/// Start of the comment header.
pub const TAGS_MAGIC: &[u8; 8] = b"PicoTags";
/// Version of the stream format this module reads and writes.
pub const VERSION: u8 = 1;
/// Scheme byte of the cx2 entropy coder.
pub const SCHEME_CX2: u8 = 2;
/// Vendor string of the comment header.
const VENDOR: &str = concat!("picotone ", env!("CARGO_PKG_VERSION"));
/// Serial number of the logical stream.
const SERIAL: u32 = 0x5049_434f;
const FLAG_CONTINUED: u8 = 1;
const FLAG_BOS: u8 = 2;
const FLAG_EOS: u8 = 4;
/// Granule position of a page on which no packet ends.
const NO_GRANULE: u64 = u64::MAX;
/// Samples per latent frame.
const HOP: usize = 320;
/// Longest audio a stream may claim, in samples.
const MAX_AUDIO_LENGTH: u64 = 1 << 40;
/// Most codebooks a stream may claim.
const MAX_CODEBOOKS: usize = 64;

/// Header flag: the trimmed audio loops seamlessly.
const HEAD_LOOP: u8 = 1;
/// Header flag: a comment header follows.
const HEAD_COMMENTS: u8 = 2;
/// Header flag: the header holds the playable length and the end trim.
const HEAD_LENGTH: u8 = 4;
/// Header flag: segments use the edge-avoiding cross-fade.
const HEAD_EDGE: u8 = 8;
const HEAD_KNOWN: u8 = HEAD_LOOP | HEAD_COMMENTS | HEAD_LENGTH | HEAD_EDGE;

/// TLV type of a per-segment codebook count (4 bits per segment). Reserved; critical.
pub const TLV_CODEBOOK_SCHEDULE: u64 = 1;
/// TLV type of the playback gain in dB, Q7.8 (`i16`, little-endian). Optional.
pub const TLV_OUTPUT_GAIN: u64 = 2;
/// TLV type of a predictor reset (keyframe) interval longer than a page. Reserved; critical.
pub const TLV_RESET_INTERVAL: u64 = 3;
/// TLV type of a segment stride schedule. Reserved; critical.
pub const TLV_STRIDE_SCHEDULE: u64 = 5;
/// TLV type of an enhancement layer (reserved, not implemented).
///
/// A `.picol` sidecar holds codebooks `first..=last` of its base stream's
/// encode. Body: varint layer index, `u32` base
/// serial number, `u64` base stream id (FNV-1a 64 of the base's identification
/// packet then its first audio packet; both little-endian), varint first
/// codebook, varint last codebook. Reserved; critical, so a layer is never
/// played as a stream of its own. See `FORMAT.md`.
pub const TLV_LAYER: u64 = 7;

/// Byte of a model in the identification header.
const fn model_code(model: ModelKind) -> u8 {
    match model {
        ModelKind::Mono24k => 0,
        ModelKind::Stereo48k => 1,
    }
}

/// How a model lays audio out in coded segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// The model.
    pub model: ModelKind,
    /// Segment stride in samples; `None` for the unsegmented 24 kHz model.
    pub stride: Option<usize>,
}

/// Where one page's chunk sits in the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSpan {
    /// Segments whose codes the page holds (all of one for the unsegmented model).
    pub segments: Range<usize>,
    /// Latent frames the page holds, counted from the start of the stream.
    pub frames: Range<usize>,
}

impl Layout {
    /// Output sample rate.
    #[must_use]
    pub const fn sample_rate(self) -> usize {
        match self.model {
            ModelKind::Mono24k => 24_000,
            ModelKind::Stereo48k => 48_000,
        }
    }

    /// Output channels.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self.model {
            ModelKind::Mono24k => 1,
            ModelKind::Stereo48k => 2,
        }
    }

    /// Latent frames per second.
    #[must_use]
    pub const fn frame_rate(self) -> usize {
        self.sample_rate() / HOP
    }

    /// Segment length in samples; `None` codes the whole file as one segment.
    #[must_use]
    pub const fn segment_length(self) -> Option<usize> {
        match self.model {
            ModelKind::Mono24k => None,
            ModelKind::Stereo48k => Some(48_000),
        }
    }

    /// Whether every segment carries a volume scale.
    #[must_use]
    pub const fn normalizes(self) -> bool {
        matches!(self.model, ModelKind::Stereo48k)
    }

    /// The layout of `model` with `stride`, checked: segmented models need a
    /// stride of half a segment to a whole one, the unsegmented model none.
    ///
    /// # Errors
    /// Fails on a missing, needless or out-of-range stride.
    pub fn new(model: ModelKind, stride: Option<usize>) -> Result<Self> {
        let layout = Self { model, stride };
        match (layout.segment_length(), stride) {
            (Some(length), Some(stride)) => ensure!(
                (length.div_ceil(2)..=length).contains(&stride),
                "segment stride {stride} is out of range ({}..={length})",
                length.div_ceil(2)
            ),
            (Some(_), None) => bail!("a segmented model needs a stride"),
            (None, Some(_)) => bail!("this model codes the whole file as one segment"),
            (None, None) => {}
        }
        Ok(layout)
    }

    /// Number of segments of an `audio_length`-sample stream.
    #[must_use]
    pub fn segment_count(self, audio_length: usize) -> usize {
        self.stride.map_or_else(
            || audio_length.min(1),
            |stride| audio_length.div_ceil(stride.max(1)),
        )
    }

    /// Latent frames in each segment of an `audio_length`-sample stream.
    ///
    /// This allocates one entry per segment; readers of untrusted input bound
    /// [`Self::segment_count`] first.
    #[must_use]
    pub fn segment_frames(self, audio_length: usize) -> Vec<usize> {
        let length = self.segment_length().unwrap_or(audio_length);
        let stride = self.stride.unwrap_or(audio_length).max(1);
        (0..audio_length)
            .step_by(stride)
            .map(|offset| (audio_length - offset).min(length).div_ceil(HOP))
            .collect()
    }

    /// The pages of an `audio_length`-sample stream with `page_length` segments
    /// (unsegmented model: frames) per page; zero means a single page.
    #[must_use]
    pub fn pages(self, audio_length: usize, page_length: usize) -> Vec<PageSpan> {
        let frames = self.segment_frames(audio_length);
        let mut out = Vec::new();
        if self.stride.is_some() {
            let per = if page_length == 0 {
                frames.len().max(1)
            } else {
                page_length
            };
            let mut frame = 0;
            for (index, group) in frames.chunks(per).enumerate() {
                let count: usize = group.iter().sum();
                out.push(PageSpan {
                    segments: index * per..index * per + group.len(),
                    frames: frame..frame + count,
                });
                frame += count;
            }
        } else if let Some(&total) = frames.first() {
            let per = if page_length == 0 {
                total.max(1)
            } else {
                page_length
            };
            out.extend((0..total).step_by(per).map(|start| PageSpan {
                segments: 0..1,
                frames: start..(start + per).min(total),
            }));
        }
        out
    }

    /// Decoded samples that are final once the pages up to `page` are decoded.
    fn granule(self, page: &PageSpan) -> usize {
        self.stride
            .map_or(page.frames.end * HOP, |stride| page.segments.end * stride)
    }
}

/// Stream parameters stored in the identification header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// The EnCodec model the codes are for.
    pub model: ModelKind,
    /// Hash of the cx2 table the stream was coded with.
    pub table_hash: u64,
    /// Codebooks per latent frame.
    pub num_codebooks: usize,
    /// Segment stride in samples; `None` for the unsegmented model.
    pub stride: Option<usize>,
    /// Cross-fade between segments.
    pub crossfade: Crossfade,
    /// Whether the trimmed audio loops seamlessly.
    pub looped: bool,
    /// Decoded samples to drop at the start (encoder priming or loop pre-roll).
    pub pre_skip: usize,
    /// Segments per page (frames for the unsegmented model); zero means one page.
    pub page_length: usize,
    /// Coded samples per channel, the pre-skip and end trim included.
    pub audio_length: usize,
    /// Decoded samples to drop at the end.
    pub end_trim: usize,
    /// Whether the length is in the identification header (otherwise only the
    /// last page gives it).
    pub length_in_header: bool,
    /// Playback gain in dB, Q7.8 fixed point.
    pub output_gain: i16,
    /// `KEY=value` comments; the comment header is written only when there are some.
    pub tags: Vec<String>,
}

impl Info {
    /// The layout of the stream.
    ///
    /// # Errors
    /// Fails when the stride does not suit the model.
    pub fn layout(&self) -> Result<Layout> {
        Layout::new(self.model, self.stride)
    }

    /// Stride and cross-fade, as the decoder takes them.
    #[must_use]
    pub const fn segmenting(&self) -> Segmenting {
        Segmenting {
            stride: self.stride,
            crossfade: self.crossfade,
        }
    }

    /// Samples per channel a player outputs: the coded length without the trims.
    #[must_use]
    pub const fn playable_samples(&self) -> usize {
        self.audio_length
            .saturating_sub(self.pre_skip)
            .saturating_sub(self.end_trim)
    }

    fn check_trims(&self) -> Result<()> {
        let trims = self.pre_skip.saturating_add(self.end_trim);
        ensure!(
            trims < self.audio_length || (trims == 0 && self.audio_length == 0),
            "pre-skip and end trim ({trims} samples) leave none of the {} samples",
            self.audio_length
        );
        Ok(())
    }
}

/// Page length suited to a stream's bitrate.
///
/// One page per chunk costs about 28 bytes plus one per 255 payload bytes, and
/// each chunk restart costs some compression, so low bitrates get longer pages:
/// one segment (unsegmented model: one second) at 20 kbps and above, two from
/// 10 kbps, four below.
#[must_use]
pub const fn default_page_length(model: ModelKind, num_codebooks: usize) -> usize {
    let layout = Layout {
        model,
        stride: None,
    };
    let kbps = num_codebooks * layout.frame_rate() / 100;
    let units = if kbps >= 20 {
        1
    } else if kbps >= 10 {
        2
    } else {
        4
    };
    if layout.segment_length().is_some() {
        units
    } else {
        units * layout.frame_rate()
    }
}

/// CRC-32 as Ogg defines it: polynomial 0x04C11DB7, no reflection, zero start and end.
fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0_u32; 256];
        let mut index = 0;
        let mut top = 0_u32;
        while index < 256 {
            let mut value = top << 24;
            let mut bit = 0;
            while bit < 8 {
                value = if value & 0x8000_0000 == 0 {
                    value << 1
                } else {
                    (value << 1) ^ 0x04C1_1DB7
                };
                bit += 1;
            }
            table[index] = value;
            index += 1;
            top += 1;
        }
        table
    };
    bytes.iter().fold(0, |crc, &byte| {
        (crc << 8) ^ TABLE[usize::from(crc.to_be_bytes()[0] ^ byte)]
    })
}

/// Writes pages of one logical stream.
struct PageWriter {
    out: Vec<u8>,
    serial: u32,
    sequence: u32,
}

impl PageWriter {
    /// Appends one packet as its own page(s); `granule` goes on the page where it ends.
    fn packet(&mut self, packet: &[u8], granule: u64, first: bool, last: bool) {
        // A packet is a run of 255-byte segments ended by a shorter (possibly empty) one.
        let mut lacing: Vec<u8> = vec![255; packet.len() / 255];
        lacing.push(u8::try_from(packet.len() % 255).unwrap_or(0));
        let mut data = packet;
        let pages = lacing.chunks(255).count();
        for (index, table) in lacing.chunks(255).enumerate() {
            let ends = index + 1 == pages;
            let size: usize = table.iter().map(|&l| usize::from(l)).sum();
            let (body, rest) = data.split_at(size);
            data = rest;
            let mut flags = 0;
            if index > 0 {
                flags |= FLAG_CONTINUED;
            }
            if first && index == 0 {
                flags |= FLAG_BOS;
            }
            if last && ends {
                flags |= FLAG_EOS;
            }
            let start = self.out.len();
            self.out.extend_from_slice(CAPTURE);
            self.out.push(0);
            self.out.push(flags);
            let position = if ends { granule } else { NO_GRANULE };
            self.out.extend_from_slice(&position.to_le_bytes());
            self.out.extend_from_slice(&self.serial.to_le_bytes());
            self.out.extend_from_slice(&self.sequence.to_le_bytes());
            self.out.extend_from_slice(&[0; 4]);
            self.out.push(u8::try_from(table.len()).unwrap_or(0));
            self.out.extend_from_slice(table);
            self.out.extend_from_slice(body);
            let crc = crc32(&self.out[start..]);
            self.out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
            self.sequence = self.sequence.wrapping_add(1);
        }
    }
}

/// One page as found in a file.
#[derive(Debug)]
struct Page<'a> {
    flags: u8,
    granule: u64,
    sequence: u32,
    lacing: &'a [u8],
    body: &'a [u8],
}

/// Reads the page starting at `pos`, checking its CRC; returns it and its total size.
fn read_page(bytes: &[u8], pos: usize) -> Option<(Page<'_>, usize)> {
    let head = bytes.get(pos..pos + 27)?;
    if &head[..4] != CAPTURE || head[4] != 0 {
        return None;
    }
    let segments = usize::from(head[26]);
    let lacing = bytes.get(pos + 27..pos + 27 + segments)?;
    let size: usize = lacing.iter().map(|&l| usize::from(l)).sum();
    let total = 27 + segments + size;
    let whole = bytes.get(pos..pos + total)?;
    let mut copy = whole.to_vec();
    copy[22..26].fill(0);
    let stored = u32::from_le_bytes([head[22], head[23], head[24], head[25]]);
    if crc32(&copy) != stored {
        return None;
    }
    let mut granule = [0; 8];
    granule.copy_from_slice(&head[6..14]);
    Some((
        Page {
            flags: head[5],
            granule: u64::from_le_bytes(granule),
            sequence: u32::from_le_bytes([head[18], head[19], head[20], head[21]]),
            lacing,
            body: &whole[27 + segments..],
        },
        total,
    ))
}

/// A packet recovered from the pages, or the sequence gap where one was lost.
#[derive(Debug)]
struct Packet {
    /// `None` when pages were damaged or missing here.
    data: Option<Vec<u8>>,
    granule: u64,
    last: bool,
}

/// Splits a file into packets, resynchronising on the capture pattern after damage.
fn packets(bytes: &[u8]) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut pos = 0;
    let mut expected = 0_u32;
    let mut partial: Option<Vec<u8>> = None;
    while pos < bytes.len() {
        let Some((page, size)) = read_page(bytes, pos) else {
            // Damaged or foreign bytes: drop any packet in progress and look for the next page.
            partial = None;
            pos += 1;
            while pos < bytes.len() && !bytes[pos..].starts_with(CAPTURE) {
                pos += 1;
            }
            continue;
        };
        pos += size;
        // Every chunk is one page here, so each missing sequence number is one lost packet.
        // A page is at least 27 bytes, which bounds how many can be missing from this
        // file; without the bound a damaged sequence number could ask for billions.
        while expected < page.sequence && out.len() < bytes.len() / 27 {
            out.push(Packet {
                data: None,
                granule: NO_GRANULE,
                last: false,
            });
            expected += 1;
            partial = None;
        }
        expected = page.sequence.wrapping_add(1);
        let mut data = if page.flags & FLAG_CONTINUED == 0 {
            Vec::new()
        } else {
            partial.take().unwrap_or_default()
        };
        let mut body = page.body;
        let mut complete = false;
        for &lace in page.lacing {
            let (segment, rest) = body.split_at(usize::from(lace));
            data.extend_from_slice(segment);
            body = rest;
            complete = lace < 255;
        }
        if complete {
            out.push(Packet {
                data: Some(data),
                granule: page.granule,
                last: page.flags & FLAG_EOS != 0,
            });
        } else {
            partial = Some(data);
        }
    }
    out
}

/// Appends `value` as an unsigned LEB128 varint.
fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = u8::try_from(value & 0x7f).unwrap_or(0);
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn put_size(out: &mut Vec<u8>, value: usize) -> Result<()> {
    put_varint(out, u64::try_from(value)?);
    Ok(())
}

/// A cursor over a header packet.
struct Fields<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Fields<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("header size overflow")?;
        let out = self
            .bytes
            .get(self.pos..end)
            .context("header packet is truncated")?;
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = self.u8()?;
            let bits = u64::from(byte & 0x7f);
            ensure!(
                shift < 63 || bits <= 1,
                "header varint does not fit 64 bits"
            );
            value |= bits << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("header varint is too long")
    }

    /// A varint that must be a plausible count of samples or similar.
    fn size(&mut self) -> Result<usize> {
        let value = self.varint()?;
        ensure!(
            value <= MAX_AUDIO_LENGTH,
            "header value {value} is implausible"
        );
        Ok(usize::try_from(value)?)
    }

    const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}

fn head_packet(info: &Info) -> Result<Vec<u8>> {
    let mut fixed = vec![model_code(info.model), SCHEME_CX2];
    fixed.extend_from_slice(&info.table_hash.to_le_bytes());
    fixed.push(u8::try_from(info.num_codebooks)?);
    let mut flags = 0;
    if info.looped {
        flags |= HEAD_LOOP;
    }
    if !info.tags.is_empty() {
        flags |= HEAD_COMMENTS;
    }
    if info.length_in_header {
        flags |= HEAD_LENGTH;
    }
    if info.crossfade == Crossfade::EdgeAvoiding {
        flags |= HEAD_EDGE;
    }
    fixed.push(flags);
    put_size(&mut fixed, info.pre_skip)?;
    if let Some(stride) = info.stride {
        put_size(&mut fixed, stride)?;
    }
    put_size(&mut fixed, info.page_length)?;
    if info.length_in_header {
        put_size(&mut fixed, info.playable_samples())?;
        put_size(&mut fixed, info.end_trim)?;
    }
    let mut out = HEAD_MAGIC.to_vec();
    out.push(VERSION);
    put_size(&mut out, fixed.len())?;
    out.extend_from_slice(&fixed);
    if info.output_gain != 0 {
        put_varint(&mut out, TLV_OUTPUT_GAIN);
        put_varint(&mut out, 2);
        out.extend_from_slice(&info.output_gain.to_le_bytes());
    }
    Ok(out)
}

/// Parses the identification header; returns it and whether comments follow.
///
/// Without the length flag, `audio_length` and `end_trim` are zero until the
/// last page is read.
fn parse_head(packet: &[u8]) -> Result<(Info, bool)> {
    let mut f = Fields::new(packet);
    ensure!(f.take(4)? == HEAD_MAGIC, "not a .pico stream");
    let version = f.u8()?;
    ensure!(
        version == VERSION,
        "unsupported .pico version {version} (this decoder reads version {VERSION})"
    );
    let fixed_len = f.size()?;
    let fixed = f.take(fixed_len)?;
    let mut h = Fields::new(fixed);
    let model = match h.u8()? {
        0 => ModelKind::Mono24k,
        1 => ModelKind::Stereo48k,
        other => bail!("unknown model {other}"),
    };
    let scheme = h.u8()?;
    ensure!(scheme == SCHEME_CX2, "unknown entropy coder {scheme}");
    let mut hash = [0; 8];
    hash.copy_from_slice(h.take(8)?);
    let num_codebooks = usize::from(h.u8()?);
    ensure!(
        (1..=MAX_CODEBOOKS).contains(&num_codebooks),
        "implausible codebook count {num_codebooks}"
    );
    let flags = h.u8()?;
    ensure!(
        flags & !HEAD_KNOWN == 0,
        "unknown header flags {flags:#04x}"
    );
    let pre_skip = h.size()?;
    let segmented = Layout {
        model,
        stride: None,
    }
    .segment_length()
    .is_some();
    let stride = if segmented { Some(h.size()?) } else { None };
    let page_length = h.size()?;
    let length_in_header = flags & HEAD_LENGTH != 0;
    let (audio_length, end_trim) = if length_in_header {
        let playable = h.size()?;
        let end_trim = h.size()?;
        let total = pre_skip
            .checked_add(playable)
            .and_then(|t| t.checked_add(end_trim))
            .filter(|&t| u64::try_from(t).is_ok_and(|t| t <= MAX_AUDIO_LENGTH))
            .context("stream has an implausible length")?;
        (total, end_trim)
    } else {
        (0, 0)
    };
    // Fixed fields past the ones known here belong to later minor revisions; skip them.
    let mut output_gain = 0;
    while f.remaining() > 0 {
        let kind = f.varint()?;
        let len = f.size()?;
        let body = f.take(len)?;
        match kind {
            TLV_OUTPUT_GAIN => {
                ensure!(body.len() == 2, "output gain field has the wrong size");
                output_gain = i16::from_le_bytes([body[0], body[1]]);
            }
            TLV_CODEBOOK_SCHEDULE => {
                bail!("stream uses a per-segment codebook count, which this decoder does not read")
            }
            TLV_RESET_INTERVAL => {
                bail!("stream uses a predictor reset interval, which this decoder does not read")
            }
            TLV_STRIDE_SCHEDULE => {
                bail!("stream uses a stride schedule, which this decoder does not read")
            }
            TLV_LAYER => {
                bail!("stream is an enhancement layer (.picol); decode its base .pico instead")
            }
            // Odd types are critical: a decoder that does not know one cannot play the stream.
            other if other & 1 == 1 => bail!("stream uses unknown critical field {other}"),
            _ => {}
        }
    }
    let info = Info {
        model,
        table_hash: u64::from_le_bytes(hash),
        num_codebooks,
        stride,
        crossfade: if flags & HEAD_EDGE == 0 {
            Crossfade::Triangle
        } else {
            Crossfade::EdgeAvoiding
        },
        looped: flags & HEAD_LOOP != 0,
        pre_skip,
        page_length,
        audio_length,
        end_trim,
        length_in_header,
        output_gain,
        tags: Vec::new(),
    };
    let _layout: Layout = info.layout()?;
    Ok((info, flags & HEAD_COMMENTS != 0))
}

fn tags_packet(tags: &[String]) -> Result<Vec<u8>> {
    let mut out = TAGS_MAGIC.to_vec();
    put_size(&mut out, VENDOR.len())?;
    out.extend_from_slice(VENDOR.as_bytes());
    put_size(&mut out, tags.len())?;
    for tag in tags {
        put_size(&mut out, tag.len())?;
        out.extend_from_slice(tag.as_bytes());
    }
    Ok(out)
}

fn parse_tags(packet: &[u8]) -> Result<Vec<String>> {
    let mut f = Fields::new(packet);
    ensure!(f.take(8)? == TAGS_MAGIC, "missing .pico comment header");
    let vendor = f.size()?;
    let _vendor: &[u8] = f.take(vendor)?;
    let count = f.size()?;
    ensure!(count <= f.remaining(), "comment count is implausible");
    (0..count)
        .map(|_| {
            let len = f.size()?;
            Ok(String::from_utf8_lossy(f.take(len)?).into_owned())
        })
        .collect()
}

/// Writes codes as a `.pico` stream.
///
/// `info` gives every header field (its `table_hash` must be `table`'s);
/// `frames` holds one entry per segment as [`Layout::segment_frames`] lays out
/// `info.audio_length` samples, each scale on the 12-bit grid of [`scale`].
///
/// # Errors
/// Fails when the frames do not match the layout or the table does not cover them.
pub fn write(table: &cx2::Model, info: &Info, frames: &[EncodedFrame]) -> Result<Vec<u8>> {
    ensure!(
        info.table_hash == table.hash(),
        "the header names table {:016x}, the coder is table {}",
        info.table_hash,
        table.hash_hex()
    );
    ensure!(
        (1..=MAX_CODEBOOKS).contains(&info.num_codebooks),
        "implausible codebook count {}",
        info.num_codebooks
    );
    info.check_trims()?;
    let layout = info.layout()?;
    let expected = layout.segment_frames(info.audio_length);
    ensure!(
        frames.len() == expected.len(),
        "{} segments given, the layout has {}",
        frames.len(),
        expected.len()
    );
    let mut scales = Vec::with_capacity(frames.len());
    for (frame, &count) in frames.iter().zip(&expected) {
        ensure!(
            frame.num_codebooks == info.num_codebooks && frame.frames() == count,
            "a segment does not match the layout"
        );
        ensure!(
            frame.codes.len() == count * info.num_codebooks,
            "a segment has the wrong number of codes"
        );
        match (layout.normalizes(), frame.scale) {
            (true, Some(level)) => {
                let index = scale::quantize(level);
                ensure!(
                    scale::dequantize(index).to_bits() == level.to_bits(),
                    "segment scale {level} is not on the 12-bit grid"
                );
                scales.push(index);
            }
            (false, None) => {}
            _ => bail!("segment scales do not match the model"),
        }
    }
    let codes: Vec<u16> = frames
        .iter()
        .flat_map(|f| f.codes.iter().copied())
        .collect();
    let pages = layout.pages(info.audio_length, info.page_length);
    let mut writer = PageWriter {
        out: Vec::new(),
        serial: SERIAL,
        sequence: 0,
    };
    let with_tags = !info.tags.is_empty();
    writer.packet(&head_packet(info)?, 0, true, pages.is_empty() && !with_tags);
    if with_tags {
        writer.packet(&tags_packet(&info.tags)?, 0, false, pages.is_empty());
    }
    let end = info.audio_length - info.end_trim;
    let nc = info.num_codebooks;
    for (index, page) in pages.iter().enumerate() {
        let last = index + 1 == pages.len();
        let page_scales = if layout.normalizes() {
            &scales[page.segments.clone()]
        } else {
            &[]
        };
        let chunk = cx2::encode_chunk(
            table,
            nc,
            page_scales,
            &codes[page.frames.start * nc..page.frames.end * nc],
        )?;
        let mut packet = Vec::with_capacity(chunk.len() + 4);
        if last && !info.length_in_header {
            put_size(&mut packet, info.end_trim)?;
        }
        packet.extend_from_slice(&chunk);
        let granule = if last {
            end
        } else {
            layout.granule(page).min(end)
        };
        writer.packet(&packet, u64::try_from(granule)?, false, last);
    }
    Ok(writer.out)
}

/// A parsed `.pico` stream.
#[derive(Debug)]
pub struct Stream {
    /// Identification header and comments, with the length filled in.
    pub info: Info,
    /// Where each page's chunk sits.
    pub spans: Vec<PageSpan>,
    /// Per page: the range-coded chunk, or `None` when the page was damaged or missing.
    pub pages: Vec<Option<Vec<u8>>>,
    /// Per page: decoded samples that are final once it is decoded.
    pub granules: Vec<u64>,
    /// Latent frames per segment.
    pub segment_frames: Vec<usize>,
}

impl Stream {
    /// Parses a stream, verifying every page's CRC.
    ///
    /// Damaged audio pages become `None`; damaged headers, or a damaged final
    /// page in a stream without a length field, are errors.
    ///
    /// # Errors
    /// Fails when the headers or the length of the stream cannot be read.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut found = packets(bytes).into_iter();
        let head = found.next().and_then(|p| p.data);
        let (mut info, comments) =
            parse_head(&head.context("identification header is missing or damaged")?)?;
        if comments {
            let tags = found.next().and_then(|p| p.data);
            info.tags = parse_tags(&tags.context("comment header is missing or damaged")?)?;
        }
        let audio: Vec<Packet> = found.collect();
        let mut chunks = Vec::with_capacity(audio.len());
        let mut final_page = None;
        for (index, packet) in audio.iter().enumerate() {
            let mut data = packet.data.as_deref();
            if packet.last && index + 1 == audio.len() {
                final_page = Some(packet.granule);
                if !info.length_in_header {
                    let mut f =
                        Fields::new(data.context("the final page of the stream is damaged")?);
                    info.end_trim = f.size()?;
                    data = data.map(|d| &d[f.pos..]);
                }
            }
            chunks.push(data.map(<[u8]>::to_vec));
        }
        if info.length_in_header {
            if let Some(granule) = final_page {
                ensure!(
                    u64::try_from(info.audio_length - info.end_trim)? == granule,
                    "the final page does not end where the header says"
                );
            }
        } else if audio.is_empty() {
            ensure!(packets_end_stream(bytes), "stream has no intact final page");
        } else {
            let granule = final_page.context("stream has no intact final page")?;
            ensure!(
                granule <= MAX_AUDIO_LENGTH,
                "stream has an implausible length"
            );
            info.audio_length = usize::try_from(granule)?
                .checked_add(info.end_trim)
                .filter(|&t| u64::try_from(t).is_ok_and(|t| t <= MAX_AUDIO_LENGTH))
                .context("stream has an implausible length")?;
        }
        info.check_trims()?;
        let layout = info.layout()?;
        // Every segment costs at least 12 bits of scale (and every page 27 bytes), so an
        // honest file is never shorter than this; the bound keeps damaged lengths from
        // sizing allocations.
        let segments = layout.segment_count(info.audio_length);
        ensure!(
            !layout.normalizes() || segments <= bytes.len(),
            "stream length does not match its size"
        );
        let spans = layout.pages(info.audio_length, info.page_length);
        ensure!(
            spans.len() <= bytes.len() / 27 + 1,
            "stream length does not match its size"
        );
        ensure!(
            chunks.len() <= spans.len(),
            "stream has more pages than its length allows"
        );
        chunks.resize(spans.len(), None);
        let end = info.audio_length - info.end_trim;
        let granules = spans
            .iter()
            .enumerate()
            .map(|(index, span)| {
                let granule = if index + 1 == spans.len() {
                    end
                } else {
                    layout.granule(span).min(end)
                };
                u64::try_from(granule).unwrap_or(u64::MAX)
            })
            .collect();
        Ok(Self {
            segment_frames: layout.segment_frames(info.audio_length),
            info,
            spans,
            pages: chunks,
            granules,
        })
    }

    /// Samples per channel a player outputs.
    #[must_use]
    pub const fn playable_samples(&self) -> usize {
        self.info.playable_samples()
    }

    /// Index of the page to start decoding at to reach playable sample `sample`
    /// (0 is the first sample after the pre-skip), found from granule positions alone.
    #[must_use]
    pub fn page_for_sample(&self, sample: u64) -> usize {
        let position = sample.saturating_add(u64::try_from(self.info.pre_skip).unwrap_or(0));
        self.granules
            .partition_point(|&g| g <= position)
            .min(self.pages.len().saturating_sub(1))
    }

    /// Checks that `table` is the one the stream was coded with.
    ///
    /// # Errors
    /// Fails on a mismatch.
    pub fn check_table(&self, table: &cx2::Model) -> Result<()> {
        ensure!(
            self.info.table_hash == table.hash(),
            "stream was coded with cx2 table {:016x}, not {}",
            self.info.table_hash,
            table.hash_hex()
        );
        Ok(())
    }

    /// Decodes page `index` on its own into its segments (for the unsegmented
    /// model, one frame holding the page's codes).
    ///
    /// # Errors
    /// Fails when the page is damaged, out of range, or was coded with another table.
    pub fn decode_page(&self, table: &cx2::Model, index: usize) -> Result<Vec<EncodedFrame>> {
        self.check_table(table)?;
        let span = self
            .spans
            .get(index)
            .context("page index is out of range")?;
        let Some(Some(chunk)) = self.pages.get(index) else {
            bail!("page {index} is damaged or missing");
        };
        let nc = self.info.num_codebooks;
        let normalizes = self.info.layout()?.normalizes();
        let scale_count = if normalizes { span.segments.len() } else { 0 };
        let (scales, codes) =
            cx2::decode_chunk(table, nc, scale_count, span.frames.len() * nc, chunk)?;
        if !normalizes && self.info.stride.is_none() {
            return Ok(vec![EncodedFrame {
                scale: None,
                num_codebooks: nc,
                codes,
            }]);
        }
        let mut out = Vec::with_capacity(span.segments.len());
        let mut pos = 0;
        for (offset, segment) in span.segments.clone().enumerate() {
            let len = self.segment_frames[segment] * nc;
            out.push(EncodedFrame {
                scale: scales.get(offset).map(|&q| scale::dequantize(q)),
                num_codebooks: nc,
                codes: codes[pos..pos + len].to_vec(),
            });
            pos += len;
        }
        Ok(out)
    }

    /// Decodes every page into one entry per segment. Every page must be intact.
    ///
    /// # Errors
    /// Fails when a page is damaged or the table does not match.
    pub fn decode_all(&self, table: &cx2::Model) -> Result<Vec<EncodedFrame>> {
        let mut out: Vec<EncodedFrame> = Vec::with_capacity(self.segment_frames.len());
        for index in 0..self.pages.len() {
            let frames = self.decode_page(table, index)?;
            if self.info.stride.is_none()
                && let (Some(first), Some(more)) = (out.first_mut(), frames.first())
            {
                first.codes.extend_from_slice(&more.codes);
                continue;
            }
            out.extend(frames);
        }
        Ok(out)
    }
}

/// Whether the last intact page of a header-only stream ends it.
fn packets_end_stream(bytes: &[u8]) -> bool {
    packets(bytes).last().is_some_and(|p| p.last)
}

/// Reads only the identification and comment headers.
///
/// Without a length field in the header the length is zero; [`Stream::parse`]
/// reads it from the last page.
///
/// # Errors
/// Fails when the headers are missing, damaged or of another format.
pub fn read_info(bytes: &[u8]) -> Result<Info> {
    let mut pos = 0;
    let mut wanted = Vec::new();
    while wanted.len() < 2 {
        let Some((page, size)) = read_page(bytes, pos) else {
            break;
        };
        pos += size;
        // Headers are short: each is one page with a final lacing value below 255.
        if page.lacing.last().is_some_and(|&l| l < 255) && page.flags & FLAG_CONTINUED == 0 {
            wanted.push(page.body.to_vec());
        } else {
            break;
        }
        if wanted.len() == 1 {
            let (_, comments) = parse_head(&wanted[0])?;
            if !comments {
                break;
            }
        }
    }
    let head = wanted
        .first()
        .context("identification header is missing or damaged")?;
    let (mut info, comments) = parse_head(head)?;
    if comments {
        info.tags = parse_tags(
            wanted
                .get(1)
                .context("comment header is missing or damaged")?,
        )?;
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_ogg_definition() {
        // Check value of CRC-32/MPEG-2 with a zero start (the Ogg variant) for "123456789".
        assert_eq!(crc32(b"123456789"), 0x89A1_897F);
    }

    #[test]
    fn packets_survive_page_splitting() {
        let mut writer = PageWriter {
            out: Vec::new(),
            serial: 7,
            sequence: 0,
        };
        let sizes = [0_usize, 1, 254, 255, 256, 70_000];
        for (index, &size) in sizes.iter().enumerate() {
            let packet: Vec<u8> = (0..size).map(|i| (i % 251).to_le_bytes()[0]).collect();
            let last = index + 1 == sizes.len();
            writer.packet(&packet, 100, index == 0, last);
        }
        let found = packets(&writer.out);
        assert_eq!(found.len(), sizes.len());
        for (packet, &size) in found.iter().zip(&sizes) {
            assert_eq!(packet.data.as_ref().map(Vec::len), Some(size));
        }
        assert!(found.last().is_some_and(|p| p.last));
    }

    #[test]
    fn varints_round_trip() -> Result<()> {
        for value in [0, 1, 127, 128, 300, 44_800, u64::from(u32::MAX), u64::MAX] {
            let mut out = Vec::new();
            put_varint(&mut out, value);
            let mut f = Fields::new(&out);
            assert_eq!(f.varint()?, value);
            assert_eq!(f.remaining(), 0);
        }
        assert!(Fields::new(&[0xff; 11]).varint().is_err());
        assert!(
            Fields::new(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02])
                .varint()
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn pages_cover_the_stream_in_whole_segments() -> Result<()> {
        let layout = Layout::new(ModelKind::Stereo48k, Some(44_800))?;
        // A length just past a stride step leaves a short penultimate segment.
        for length in [1, 44_800, 44_801, 48_000, 92_000, 200_000, 1_000_000] {
            let frames = layout.segment_frames(length);
            for per in [0, 1, 2, 4] {
                let pages = layout.pages(length, per);
                assert_eq!(pages.first().map(|p| p.frames.start), Some(0));
                assert_eq!(
                    pages.last().map(|p| p.frames.end),
                    Some(frames.iter().sum())
                );
                assert_eq!(pages.last().map(|p| p.segments.end), Some(frames.len()));
                for pair in pages.windows(2) {
                    assert_eq!(pair[0].frames.end, pair[1].frames.start);
                    assert_eq!(pair[0].segments.end, pair[1].segments.start);
                }
            }
        }
        let mono = Layout::new(ModelKind::Mono24k, None)?;
        let pages = mono.pages(24_000 * 3 + 5, 75);
        assert_eq!(pages.len(), 4);
        assert_eq!(pages.last().map(|p| p.frames.clone()), Some(225..226));
        assert!(Layout::new(ModelKind::Stereo48k, Some(23_999)).is_err());
        assert!(Layout::new(ModelKind::Mono24k, Some(24_000)).is_err());
        Ok(())
    }
}
