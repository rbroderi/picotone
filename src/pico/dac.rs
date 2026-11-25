//! HQ streams (model 2, `dac44`): DAC 44 kHz codes, entropy-coded with cx2
//! ([`crate::cx2::hq`]) one page at a time, in the same Ogg framing as the
//! EnCodec streams.
//!
//! Each coded channel is an independent mono stream of DAC frames (512
//! samples at 44.1 kHz). A page holds the same frame range of every coded
//! channel: for each channel in turn, its size in bytes (a varint) and its
//! range-coded codes, each frame codebook 0 first. Every page and channel is
//! coded with a fresh predictor, so pages decode on their own. Stereo is coded as left/right or as mid/side, `(L + R) / 2` and
//! `(L - R) / 2`, chosen per file; the two may use different codebook counts.
//! Parametric stereo codes the mid alone, and each packet starts with that
//! page's stereo parameters ([`crate::pstereo`]).

use std::ops::Range;

use anyhow::{Context as _, Result, bail, ensure};

use super::{
    Extent, Fields, HEAD_COMMENTS, HEAD_LENGTH, HEAD_LOOP, PageWriter, SERIAL, head_from_fixed,
    packets, parse_tags, put_size, read_audio, read_extensions, read_length, split_head,
    tags_packet,
};
use rayon::prelude::*;

use crate::cx2::hq::{self, HqModel, Role};
use crate::dac::{CODEBOOK_SIZE, CODEBOOKS, HOP};
use crate::pstereo;

/// Model byte of the HQ codec.
pub(super) const MODEL_DAC44: u8 = 2;
/// Scheme byte of cx2-coded codes, the only HQ scheme.
const SCHEME_CX2: u8 = 2;
/// Flags an HQ stream may set (no cross-fade: it is not segmented).
const DAC_KNOWN: u8 = HEAD_LOOP | HEAD_COMMENTS | HEAD_LENGTH;
/// Frames per page the encoder uses: 80 frames, 40,960 samples, about 0.93 s.
pub const DEFAULT_PAGE_FRAMES: usize = 80;

/// How a stream's coded channels make up its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// One channel.
    Mono,
    /// Two channels, left then right.
    LeftRight,
    /// Two channels, mid `(L + R) / 2` then side `(L - R) / 2`.
    MidSide,
    /// The mid alone, and stereo parameters per page: two output channels.
    Parametric(pstereo::Config),
}

impl ChannelLayout {
    /// Coded channels.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Mono | Self::Parametric(_) => 1,
            Self::LeftRight | Self::MidSide => 2,
        }
    }

    /// Output channels.
    #[must_use]
    pub const fn output_channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::LeftRight | Self::MidSide | Self::Parametric(_) => 2,
        }
    }

    /// Name for messages: `mono`, `left/right`, `mid/side` or `parametric`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::LeftRight => "left/right",
            Self::MidSide => "mid/side",
            Self::Parametric(_) => "parametric",
        }
    }

    /// The cx2 role of each coded channel.
    #[must_use]
    pub const fn roles(self) -> &'static [Role] {
        match self {
            Self::Mono | Self::Parametric(_) => &[Role::Mid],
            Self::LeftRight => &[Role::LeftRight, Role::LeftRight],
            Self::MidSide => &[Role::Mid, Role::Side],
        }
    }

    /// The layout byte, and for parametric stereo its settings byte.
    fn write(self, out: &mut Vec<u8>) {
        match self {
            Self::Mono => out.push(0),
            Self::LeftRight => out.push(1),
            Self::MidSide => out.push(2),
            Self::Parametric(config) => out.extend([3, config.to_byte()]),
        }
    }

    fn read(h: &mut Fields<'_>) -> Result<Self> {
        match h.u8()? {
            0 => Ok(Self::Mono),
            1 => Ok(Self::LeftRight),
            2 => Ok(Self::MidSide),
            3 => Ok(Self::Parametric(pstereo::Config::from_byte(h.u8()?)?)),
            other => bail!("unknown channel layout {other}"),
        }
    }
}

/// Parameters of an HQ stream, as stored in its identification header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DacInfo {
    /// FNV-1a 64 of the DAC codebooks the codes index
    /// ([`crate::dac::DacEncoder::codebook_hash`]).
    pub codebook_hash: u64,
    /// FNV-1a 64 of the cx2 table the pages are coded with ([`HqModel::hash`]).
    pub table_hash: u64,
    /// How the coded channels make up the output.
    pub layout: ChannelLayout,
    /// Codebooks per frame of each coded channel, 1 to 9.
    pub codebooks: Vec<usize>,
    /// Whether the trimmed audio loops seamlessly.
    pub looped: bool,
    /// Decoded samples to drop at the start.
    pub pre_skip: usize,
    /// Frames per page; zero means one page.
    pub page_length: usize,
    /// Coded samples per channel, the pre-skip and end trim included. The
    /// last frame is padded to 512 samples; a player drops the padding along
    /// with everything else past the final granule.
    pub audio_length: usize,
    /// Coded samples to drop at the end (a loop's post-roll).
    pub end_trim: usize,
    /// Whether the length is in the identification header.
    pub length_in_header: bool,
    /// Playback gain in dB, Q7.8 fixed point.
    pub output_gain: i16,
    /// `KEY=value` comments.
    pub tags: Vec<String>,
}

impl DacInfo {
    /// Frames per coded channel.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.audio_length.div_ceil(HOP)
    }

    /// Samples per channel a player outputs.
    #[must_use]
    pub const fn playable_samples(&self) -> usize {
        self.audio_length
            .saturating_sub(self.pre_skip)
            .saturating_sub(self.end_trim)
    }

    /// The frame range of each page.
    #[must_use]
    pub fn pages(&self) -> Vec<Range<usize>> {
        let frames = self.frames();
        let per = if self.page_length == 0 {
            frames.max(1)
        } else {
            self.page_length
        };
        (0..frames)
            .step_by(per)
            .map(|start| start..(start + per).min(frames))
            .collect()
    }

    fn check(&self) -> Result<()> {
        ensure!(
            self.codebooks.len() == self.layout.channels(),
            "{} codebook counts for {} channels",
            self.codebooks.len(),
            self.layout.channels()
        );
        for &n in &self.codebooks {
            ensure!(
                (1..=CODEBOOKS).contains(&n),
                "{n} codebooks; the HQ codec has 1 to {CODEBOOKS}"
            );
        }
        let trims = self.pre_skip.saturating_add(self.end_trim);
        ensure!(
            trims < self.audio_length || (trims == 0 && self.audio_length == 0),
            "pre-skip and end trim ({trims} samples) leave none of the {} samples",
            self.audio_length
        );
        Ok(())
    }

    /// Final granule of each page.
    fn granules(&self, spans: &[Range<usize>]) -> Vec<u64> {
        let end = self.audio_length - self.end_trim;
        spans
            .iter()
            .enumerate()
            .map(|(index, span)| {
                let granule = if index + 1 == spans.len() {
                    end
                } else {
                    (span.end * HOP).min(end)
                };
                u64::try_from(granule).unwrap_or(u64::MAX)
            })
            .collect()
    }
}

fn head_packet(info: &DacInfo) -> Result<Vec<u8>> {
    let mut fixed = vec![MODEL_DAC44, SCHEME_CX2];
    fixed.extend_from_slice(&info.codebook_hash.to_le_bytes());
    fixed.extend_from_slice(&info.table_hash.to_le_bytes());
    info.layout.write(&mut fixed);
    for &n in &info.codebooks {
        fixed.push(u8::try_from(n)?);
    }
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
    fixed.push(flags);
    put_size(&mut fixed, info.pre_skip)?;
    put_size(&mut fixed, info.page_length)?;
    if info.length_in_header {
        put_size(&mut fixed, info.playable_samples())?;
        put_size(&mut fixed, info.end_trim)?;
    }
    head_from_fixed(&fixed, info.output_gain)
}

/// Parses an HQ identification header; returns it and whether comments follow.
fn parse_head(packet: &[u8]) -> Result<(DacInfo, bool)> {
    let (fixed, mut f) = split_head(packet)?;
    let mut h = Fields::new(fixed);
    ensure!(h.u8()? == MODEL_DAC44, "not an HQ (dac44) stream");
    let scheme = h.u8()?;
    ensure!(scheme == SCHEME_CX2, "unknown HQ code scheme {scheme}");
    let mut hash = [0; 8];
    hash.copy_from_slice(h.take(8)?);
    let mut table = [0; 8];
    table.copy_from_slice(h.take(8)?);
    let layout = ChannelLayout::read(&mut h)?;
    let codebooks = (0..layout.channels())
        .map(|_| h.u8().map(usize::from))
        .collect::<Result<Vec<_>>>()?;
    let flags = h.u8()?;
    ensure!(flags & !DAC_KNOWN == 0, "unknown header flags {flags:#04x}");
    let pre_skip = h.size()?;
    let page_length = h.size()?;
    let length_in_header = flags & HEAD_LENGTH != 0;
    let (audio_length, end_trim) = if length_in_header {
        read_length(&mut h, pre_skip)?
    } else {
        (0, 0)
    };
    let output_gain = read_extensions(&mut f)?;
    let info = DacInfo {
        codebook_hash: u64::from_le_bytes(hash),
        table_hash: u64::from_le_bytes(table),
        layout,
        codebooks,
        looped: flags & HEAD_LOOP != 0,
        pre_skip,
        page_length,
        audio_length,
        end_trim,
        length_in_header,
        output_gain,
        tags: Vec::new(),
    };
    for &n in &info.codebooks {
        ensure!(
            (1..=CODEBOOKS).contains(&n),
            "implausible codebook count {n}"
        );
    }
    Ok((info, flags & HEAD_COMMENTS != 0))
}

/// Writes an HQ stream, coding it with `table`. `channels` holds each coded
/// channel's codes, `[frame][codebook]` row-major, with `info.codebooks[c]`
/// codes per frame and [`DacInfo::frames`] frames.
///
/// # Errors
/// Fails when the codes do not match `info`, `table` is not the one
/// `info.table_hash` names, or the layout is parametric (see
/// [`write_dac_parametric`]).
pub fn write_dac(info: &DacInfo, table: &HqModel, channels: &[Vec<u16>]) -> Result<Vec<u8>> {
    write_dac_parametric(info, table, channels, &[])
}

/// Writes an HQ stream like [`write_dac`], with each page's parametric
/// stereo data ([`crate::pstereo::write_pages`]) for that layout; `parameters`
/// must be empty for the others.
///
/// # Errors
/// Fails when the codes or the parameters do not match `info`.
pub fn write_dac_parametric(
    info: &DacInfo,
    table: &HqModel,
    channels: &[Vec<u16>],
    parameters: &[Vec<u8>],
) -> Result<Vec<u8>> {
    info.check()?;
    ensure!(
        table.hash() == info.table_hash,
        "the header names cx2 table {:016x}, the coder has {:016x}",
        info.table_hash,
        table.hash()
    );
    let frames = info.frames();
    ensure!(
        channels.len() == info.layout.channels(),
        "{} coded channels given, the layout has {}",
        channels.len(),
        info.layout.channels()
    );
    for (codes, &n) in channels.iter().zip(&info.codebooks) {
        ensure!(
            codes.len() == frames * n,
            "a channel has {} codes, expected {frames} frames x {n}",
            codes.len()
        );
        ensure!(
            codes.iter().all(|&c| usize::from(c) < CODEBOOK_SIZE),
            "a code is out of range"
        );
    }
    let pages = info.pages();
    let parametric = matches!(info.layout, ChannelLayout::Parametric(_));
    ensure!(
        parameters.len() == if parametric { pages.len() } else { 0 },
        "parametric stereo data for {} of {} pages",
        parameters.len(),
        pages.len()
    );
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
    let granules = info.granules(&pages);
    let roles = info.layout.roles();
    // Pages and channels code independently, so they code in parallel.
    let coded = pages
        .par_iter()
        .map(|span| {
            channels
                .iter()
                .zip(&info.codebooks)
                .zip(roles)
                .map(|((codes, &n), &role)| {
                    hq::encode_page(table, role, n, &codes[span.start * n..span.end * n])
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, (chunks, &granule)) in coded.iter().zip(&granules).enumerate() {
        let last = index + 1 == pages.len();
        let mut out = Vec::with_capacity(chunks.iter().map(|c| c.len() + 3).sum::<usize>() + 8);
        if last && !info.length_in_header {
            put_size(&mut out, info.end_trim)?;
        }
        if let Some(data) = parameters.get(index) {
            put_size(&mut out, data.len())?;
            out.extend_from_slice(data);
        }
        for chunk in chunks {
            put_size(&mut out, chunk.len())?;
            out.extend_from_slice(chunk);
        }
        writer.packet(&out, granule, false, last);
    }
    Ok(writer.out)
}

/// A parsed HQ stream.
#[derive(Debug)]
pub struct DacStream {
    /// Identification header and comments, with the length filled in.
    pub info: DacInfo,
    /// Frames of each page.
    pub spans: Vec<Range<usize>>,
    /// Per page: the packet after its end trim, or `None` when the page was damaged or missing.
    pub pages: Vec<Option<Vec<u8>>>,
    /// Per page: decoded samples that are final once it is decoded.
    pub granules: Vec<u64>,
}

impl DacStream {
    /// Parses a stream, verifying every page's CRC.
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
        let mut length = Extent {
            in_header: info.length_in_header,
            audio_length: info.audio_length,
            end_trim: info.end_trim,
        };
        let mut chunks = read_audio(&found.collect::<Vec<_>>(), bytes, &mut length)?;
        info.audio_length = length.audio_length;
        info.end_trim = length.end_trim;
        info.check()?;
        // Every page costs at least an Ogg page header, so an honest file has
        // more bytes than pages; the bound keeps damaged lengths from sizing
        // allocations (decoding reserves only what a page's bytes could hold).
        let per = if info.page_length == 0 {
            info.frames().max(1)
        } else {
            info.page_length
        };
        ensure!(
            info.frames().div_ceil(per) <= bytes.len(),
            "stream length does not match its size"
        );
        let spans = info.pages();
        ensure!(
            chunks.len() <= spans.len(),
            "stream has more pages than its length allows"
        );
        chunks.resize(spans.len(), None);
        let granules = info.granules(&spans);
        Ok(Self {
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

    /// Index of the page holding playable sample `sample`.
    #[must_use]
    pub fn page_for_sample(&self, sample: u64) -> usize {
        let position = sample.saturating_add(u64::try_from(self.info.pre_skip).unwrap_or(0));
        self.granules
            .partition_point(|&g| g <= position)
            .min(self.pages.len().saturating_sub(1))
    }

    /// A page's packet split into its parametric stereo data (empty for the
    /// other layouts) and its codes. Parametric packets start with the
    /// data's size in bytes (a varint), so the codes may have any size.
    fn split_packet(&self, index: usize) -> Result<(&[u8], &[u8])> {
        let Some(Some(data)) = self.pages.get(index) else {
            bail!("page {index} is damaged or missing");
        };
        if !matches!(self.info.layout, ChannelLayout::Parametric(_)) {
            return Ok((&[], data));
        }
        let mut f = Fields::new(data);
        let size = f.size()?;
        let parameters = f
            .take(size)
            .with_context(|| format!("page {index}'s stereo data is truncated"))?;
        Ok((parameters, &data[f.pos..]))
    }

    /// Page `index`'s parametric stereo data (empty for the other layouts).
    ///
    /// # Errors
    /// Fails when the page is damaged, missing or truncated.
    pub fn page_parameters(&self, index: usize) -> Result<&[u8]> {
        Ok(self.split_packet(index)?.0)
    }

    /// Page `index`'s coded channels: each one's range-coded bytes.
    fn page_chunks(&self, index: usize) -> Result<Vec<&[u8]>> {
        let data = self.split_packet(index)?.1;
        let mut f = Fields::new(data);
        let mut chunks = Vec::with_capacity(self.info.codebooks.len());
        for _ in &self.info.codebooks {
            let size = f.size()?;
            chunks.push(
                f.take(size)
                    .with_context(|| format!("page {index}'s codes are truncated"))?,
            );
        }
        ensure!(f.pos == data.len(), "page {index} has trailing bytes");
        Ok(chunks)
    }

    /// The codes of page `index`: per coded channel, `[frame][codebook]`.
    ///
    /// # Errors
    /// Fails when `table` is not the stream's, or the page is damaged,
    /// missing or truncated.
    pub fn decode_page(&self, table: &HqModel, index: usize) -> Result<Vec<Vec<u16>>> {
        ensure!(
            table.hash() == self.info.table_hash,
            "the stream was coded with cx2 table {:016x}, not {:016x}",
            self.info.table_hash,
            table.hash()
        );
        let span = self
            .spans
            .get(index)
            .context("page index is out of range")?;
        self.page_chunks(index)?
            .into_iter()
            .zip(&self.info.codebooks)
            .zip(self.info.layout.roles())
            .map(|((bytes, &n), &role)| {
                hq::decode_page(table, role, n, span.len() * n, bytes)
                    .with_context(|| format!("page {index}"))
            })
            .collect()
    }

    /// The codes of every page, joined: per coded channel, `[frame][codebook]`.
    /// Pages decode in parallel.
    ///
    /// # Errors
    /// Fails when a page is damaged or missing.
    pub fn decode_all(&self, table: &HqModel) -> Result<Vec<Vec<u16>>> {
        let pages = (0..self.pages.len())
            .into_par_iter()
            .map(|index| self.decode_page(table, index))
            .collect::<Result<Vec<_>>>()?;
        let mut out: Vec<Vec<u16>> = self
            .info
            .codebooks
            .iter()
            .map(|&n| Vec::with_capacity(self.info.frames() * n))
            .collect();
        for page in pages {
            for (all, codes) in out.iter_mut().zip(page) {
                all.extend_from_slice(&codes);
            }
        }
        Ok(out)
    }

    /// Bytes of range-coded codes per coded channel, over the intact pages.
    #[must_use]
    pub fn coded_bytes(&self) -> Vec<usize> {
        let mut out = vec![0; self.info.codebooks.len()];
        for index in 0..self.pages.len() {
            if let Ok(chunks) = self.page_chunks(index) {
                for (total, chunk) in out.iter_mut().zip(chunks) {
                    *total += chunk.len();
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;

    /// The shipped HQ table.
    fn table() -> Result<&'static HqModel> {
        static TABLE: OnceLock<Option<HqModel>> = OnceLock::new();
        TABLE
            .get_or_init(|| {
                let path =
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models/cx2_dac44.ecxm");
                HqModel::from_table(&std::fs::read(path).ok()?).ok()
            })
            .as_ref()
            .context("models/cx2_dac44.ecxm is missing or malformed")
    }

    fn info(layout: ChannelLayout, codebooks: Vec<usize>, audio_length: usize) -> DacInfo {
        DacInfo {
            codebook_hash: 0x0123_4567_89ab_cdef,
            table_hash: table().map_or(0, HqModel::hash),
            layout,
            codebooks,
            looped: false,
            pre_skip: 0,
            page_length: DEFAULT_PAGE_FRAMES,
            audio_length,
            end_trim: 0,
            length_in_header: true,
            output_gain: 0,
            tags: Vec::new(),
        }
    }

    fn codes(frames: usize, n: usize, seed: u32) -> Vec<u16> {
        let mut state = seed;
        (0..frames * n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                u16::try_from(state >> 22).unwrap_or(0)
            })
            .collect()
    }

    #[test]
    fn streams_round_trip() -> Result<()> {
        for (layout, books, length, with_length, tags) in [
            (ChannelLayout::Mono, vec![9], 44_100 * 3 + 7, true, false),
            (ChannelLayout::LeftRight, vec![9, 9], 512 * 80, true, true),
            (
                ChannelLayout::MidSide,
                vec![7, 3],
                512 * 161 - 1,
                false,
                true,
            ),
            (ChannelLayout::MidSide, vec![1, 9], 1, false, false),
        ] {
            let mut i = info(layout, books, length);
            i.length_in_header = with_length;
            if tags {
                i.tags = vec!["TITLE=x".to_owned()];
            }
            let frames = i.frames();
            let channels: Vec<Vec<u16>> = i
                .codebooks
                .iter()
                .zip(1..)
                .map(|(&n, seed)| codes(frames, n, seed))
                .collect();
            let bytes = write_dac(&i, table()?, &channels)?;
            assert_eq!(super::super::codec_of(&bytes)?, super::super::Codec::Dac44);
            let stream = DacStream::parse(&bytes)?;
            assert_eq!(stream.info, i);
            assert_eq!(stream.decode_all(table()?)?, channels);
            assert_eq!(
                stream.granules.last().copied(),
                u64::try_from(i.playable_samples()).ok()
            );
        }
        Ok(())
    }

    #[test]
    fn parametric_streams_round_trip() -> Result<()> {
        let config = pstereo::Config::new(10, 4)?;
        for (length, with_length) in [(512 * 200 + 3, true), (512 * 80, false), (5, true)] {
            let mut i = info(ChannelLayout::Parametric(config), vec![3], length);
            i.length_in_header = with_length;
            let frames = i.frames();
            let channels = vec![codes(frames, 3, 7)];
            let parameters: Vec<Vec<u8>> = (0..i.pages().len())
                .map(|p| {
                    (0..p * 37 % 200)
                        .map(|b| u8::try_from(b % 256).unwrap_or(0))
                        .collect()
                })
                .collect();
            assert!(write_dac(&i, table()?, &channels).is_err());
            let bytes = write_dac_parametric(&i, table()?, &channels, &parameters)?;
            let stream = DacStream::parse(&bytes)?;
            assert_eq!(stream.info, i);
            assert_eq!(stream.decode_all(table()?)?, channels);
            for (p, want) in parameters.iter().enumerate() {
                assert_eq!(stream.page_parameters(p)?, want.as_slice());
            }
        }
        let i = info(ChannelLayout::LeftRight, vec![2, 2], 512 * 10);
        let two = vec![codes(10, 2, 1), codes(10, 2, 2)];
        assert!(write_dac_parametric(&i, table()?, &two, &[vec![1]]).is_err());
        let none: &[u8] = &[];
        assert_eq!(
            DacStream::parse(&write_dac(&i, table()?, &two)?)?.page_parameters(0)?,
            none
        );
        Ok(())
    }

    #[test]
    fn a_damaged_page_costs_only_itself() -> Result<()> {
        let i = info(ChannelLayout::LeftRight, vec![4, 4], 512 * 400);
        let channels = vec![codes(400, 4, 1), codes(400, 4, 2)];
        let mut bytes = write_dac(&i, table()?, &channels)?;
        // Flip a byte in the middle of the file: inside one audio page.
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x55;
        let stream = DacStream::parse(&bytes)?;
        let damaged: Vec<usize> = (0..stream.pages.len())
            .filter(|&p| stream.pages[p].is_none())
            .collect();
        assert_eq!(damaged.len(), 1);
        for page in 0..stream.pages.len() {
            if damaged.contains(&page) {
                assert!(stream.decode_page(table()?, page).is_err());
            } else {
                let got = stream.decode_page(table()?, page)?;
                let span = &stream.spans[page];
                assert_eq!(got[1], channels[1][span.start * 4..span.end * 4]);
            }
        }
        Ok(())
    }

    #[test]
    fn coded_pages_are_smaller_than_raw_codes_and_checked() -> Result<()> {
        // Real music codes (the parity fixture's) cost under 10 bits each; random ones about 10.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/cx2_dac44_fixture.json");
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let mid: Vec<u16> = json["cases"][0]["codes"]
            .as_array()
            .context("fixture codes")?
            .iter()
            .filter_map(|v| v.as_u64().and_then(|c| u16::try_from(c).ok()))
            .collect();
        let frames = mid.len() / 9;
        let i = info(ChannelLayout::MidSide, vec![9, 3], 512 * frames);
        let side = codes(frames, 3, 5);
        let bytes = write_dac(&i, table()?, &[mid.clone(), side.clone()])?;
        let stream = DacStream::parse(&bytes)?;
        assert_eq!(stream.decode_all(table()?)?, vec![mid, side]);
        let coded = stream.coded_bytes();
        assert!(coded[0] * 8 < frames * 9 * 10, "{coded:?}");
        // Another table is refused.
        let mut other = i;
        other.table_hash ^= 1;
        assert!(
            write_dac(
                &other,
                table()?,
                &[codes(frames, 9, 1), codes(frames, 3, 2)]
            )
            .is_err()
        );
        // A truncated chunk is refused, not decoded into made-up codes.
        let mut cut = DacStream::parse(&bytes)?;
        if let Some(Some(page)) = cut.pages.get_mut(0) {
            let keep = page.len() - 4;
            page.truncate(keep);
        }
        assert!(cut.decode_page(table()?, 0).is_err());
        assert!(cut.decode_page(table()?, 1).is_ok());
        Ok(())
    }

    #[test]
    fn bad_headers_are_refused() -> Result<()> {
        let mut i = info(ChannelLayout::LeftRight, vec![9], 1000);
        assert!(write_dac(&i, table()?, &[codes(2, 9, 1)]).is_err());
        i.codebooks = vec![9, 10];
        assert!(write_dac(&i, table()?, &[codes(2, 9, 1), codes(2, 10, 1)]).is_err());
        i.codebooks = vec![9, 0];
        assert!(write_dac(&i, table()?, &[codes(2, 9, 1), Vec::new()]).is_err());
        Ok(())
    }
}
