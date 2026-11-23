//! The HQ codec in the command line: DAC 44 kHz's encoder, the distilled
//! decoder (`dac44-student`), and the stereo decisions.

#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use picotone::audio::Audio;
use picotone::dac::{self, DacDecoder, DacEncoder};
use picotone::layout::{self, Trimmer};
use picotone::nn::Seq;
use picotone::pico::{self, ChannelLayout, DacInfo, DacStream};
use picotone::stereo;
use rayon::prelude::*;

/// Encoder weights in the weights directory.
pub(crate) const ENCODER_FILE: &str = "dac44_encoder.safetensors";
/// Decoder weights in the weights directory.
pub(crate) const DECODER_FILE: &str = "dac44_student.safetensors";

/// How to choose the stereo layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StereoMode {
    /// Always mid/side.
    MidSide,
    /// Always left/right.
    LeftRight,
    /// Compare both on sampled excerpts.
    Pick,
}

/// Settings of an HQ encode.
#[derive(Debug)]
pub(crate) struct Settings<'a> {
    /// Output channels, 1 or 2.
    pub channels: usize,
    /// Code a stereo input that is mono by the rule as one channel.
    pub auto_mono: bool,
    /// Stereo layout choice.
    pub stereo: StereoMode,
    /// Codebooks per frame (left/right: each channel; mid/side: the mid).
    pub codebooks: usize,
    /// Codebooks per frame of the side.
    pub side_codebooks: usize,
    /// Code the input as a seamless loop.
    pub looped: bool,
    /// Comments.
    pub tags: &'a [String],
    /// Frames per page.
    pub page_length: Option<usize>,
    /// Leave the length out of the header.
    pub no_length: bool,
}

fn file(weights: &Path, name: &str) -> Result<PathBuf> {
    let path = weights.join(name);
    ensure!(
        path.is_file(),
        "HQ weights not found: expected {} (pass --weights, or use --fast for EnCodec)",
        path.display()
    );
    Ok(path)
}

fn load_encoder(weights: &Path) -> Result<DacEncoder> {
    let start = Instant::now();
    let encoder = DacEncoder::load(&file(weights, ENCODER_FILE)?)?;
    eprintln!(
        "loaded the HQ encoder in {:.2} s",
        start.elapsed().as_secs_f64()
    );
    Ok(encoder)
}

fn load_decoder(weights: &Path) -> Result<DacDecoder> {
    let start = Instant::now();
    let decoder = DacDecoder::load(&file(weights, DECODER_FILE)?)?;
    eprintln!(
        "loaded the HQ decoder in {:.2} s",
        start.elapsed().as_secs_f64()
    );
    Ok(decoder)
}

/// The signals a layout codes: the channels, or mid and side.
fn coded_signals(channels: &[Vec<f32>], layout: ChannelLayout) -> Vec<Vec<f32>> {
    match layout {
        ChannelLayout::MidSide => {
            let (l, r) = (&channels[0], &channels[1]);
            vec![
                l.iter().zip(r).map(|(a, b)| (a + b) / 2.0).collect(),
                l.iter().zip(r).map(|(a, b)| (a - b) / 2.0).collect(),
            ]
        }
        ChannelLayout::Mono | ChannelLayout::LeftRight => channels.to_vec(),
    }
}

/// Decoded coded signals back to output channels.
fn output_channels(mut decoded: Vec<Vec<f32>>, layout: ChannelLayout) -> Vec<Vec<f32>> {
    if layout == ChannelLayout::MidSide && decoded.len() == 2 {
        let side = decoded.pop().unwrap_or_default();
        let mid = decoded.pop().unwrap_or_default();
        let left = mid.iter().zip(&side).map(|(m, s)| m + s).collect();
        let right = mid.iter().zip(&side).map(|(m, s)| m - s).collect();
        vec![left, right]
    } else {
        decoded
    }
}

/// Codes of each signal, cut to its tier: `[frame][codebook]`.
fn encode_signals(
    encoder: &DacEncoder,
    signals: &[Vec<f32>],
    codebooks: &[usize],
) -> Result<Vec<Vec<u16>>> {
    signals
        .iter()
        .zip(codebooks)
        .map(|(signal, &n)| {
            let all = encoder.encode(signal)?;
            Ok(all
                .as_chunks::<{ dac::CODEBOOKS }>()
                .0
                .iter()
                .flat_map(|frame| frame[..n].iter().copied())
                .collect())
        })
        .collect()
}

/// Encodes `source` as an HQ stream; returns its header and bytes.
pub(crate) fn encode(
    weights: &Path,
    source: &Audio,
    s: &Settings<'_>,
) -> Result<(DacInfo, Vec<u8>)> {
    for n in [s.codebooks, s.side_codebooks] {
        ensure!(
            (1..=dac::CODEBOOKS).contains(&n),
            "{n} codebooks requested; the HQ codec has 1 to {}",
            dac::CODEBOOKS
        );
    }
    let encoder = load_encoder(weights)?;
    let mut channels = source.convert(dac::SAMPLE_RATE, s.channels)?.channels;
    if channels.len() == 2 && s.auto_mono && stereo::is_mono(&channels[0], &channels[1]) {
        eprintln!("the input is mono (its side is over 40 dB under its mid): coding one channel");
        channels = coded_signals(&channels, ChannelLayout::MidSide);
        channels.truncate(1);
    }
    let layout = if channels.len() == 1 {
        ChannelLayout::Mono
    } else {
        match s.stereo {
            StereoMode::LeftRight => ChannelLayout::LeftRight,
            StereoMode::MidSide => ChannelLayout::MidSide,
            StereoMode::Pick => pick(weights, &encoder, &channels, s)?,
        }
    };
    let codebooks = match layout {
        ChannelLayout::Mono => vec![s.codebooks],
        ChannelLayout::LeftRight => vec![s.codebooks; 2],
        ChannelLayout::MidSide => vec![s.codebooks, s.side_codebooks],
    };
    let mut info = DacInfo {
        codebook_hash: encoder.codebook_hash(),
        layout,
        codebooks: codebooks.clone(),
        looped: false,
        pre_skip: 0,
        page_length: s.page_length.unwrap_or(pico::DEFAULT_PAGE_FRAMES),
        audio_length: 0,
        end_trim: 0,
        length_in_header: !s.no_length,
        output_gain: 0,
        tags: s.tags.to_vec(),
    };
    if s.looped {
        let (pre, post) = layout::loop_padding(usize::try_from(dac::SAMPLE_RATE)?);
        let seq = Audio {
            sample_rate: dac::SAMPLE_RATE,
            channels,
        }
        .to_seq()?;
        channels = Audio::from_seq(&layout::wrap_loop(&seq, pre, post)?, dac::SAMPLE_RATE).channels;
        info.pre_skip = pre;
        info.end_trim = post;
        info.looped = true;
    }
    info.audio_length = channels.first().map_or(0, Vec::len);
    let start = Instant::now();
    let codes = encode_signals(&encoder, &coded_signals(&channels, layout), &codebooks)?;
    let bytes = pico::write_dac(&info, &codes)?;
    eprintln!(
        "encoded with dac44-student ({}, {} codebooks{}) in {:.2} s",
        layout.name(),
        codebooks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("+"),
        if s.looped { ", loop" } else { "" },
        start.elapsed().as_secs_f64()
    );
    Ok((info, bytes))
}

/// Picks left/right or mid/side from sampled excerpts.
fn pick(
    weights: &Path,
    encoder: &DacEncoder,
    channels: &[Vec<f32>],
    s: &Settings<'_>,
) -> Result<ChannelLayout> {
    let decoder = load_decoder(weights)?;
    let start = Instant::now();
    let (layout, scores) = stereo::pick_layout(&channels[0], &channels[1], |l, r, layout| {
        let books = if layout == ChannelLayout::MidSide {
            [s.codebooks, s.side_codebooks]
        } else {
            [s.codebooks; 2]
        };
        let excerpt = [l.to_vec(), r.to_vec()];
        let codes = encode_signals(encoder, &coded_signals(&excerpt, layout), &books)?;
        let signals = codes
            .iter()
            .zip(books)
            .map(|(c, n)| {
                let mut y = decoder.decode(c, n)?;
                y.truncate(l.len());
                Ok(y)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut out = output_channels(signals, layout).into_iter();
        Ok([
            out.next().unwrap_or_default(),
            out.next().unwrap_or_default(),
        ])
    })?;
    eprintln!(
        "stereo: {} (log-mel {:.3} left/right, {:.3} mid/side, {}) in {:.2} s",
        layout.name(),
        scores.left_right,
        scores.mid_side,
        if scores.excerpts == 0 {
            "whole file".to_owned()
        } else {
            format!("{} excerpt(s)", scores.excerpts)
        },
        start.elapsed().as_secs_f64()
    );
    Ok(layout)
}

/// Decodes an HQ stream to audio, trimmed for playback.
pub(crate) fn decode(weights: &Path, stream: &DacStream) -> Result<Audio> {
    let decoder = load_decoder(weights)?;
    let info = &stream.info;
    ensure!(
        decoder.codebook_hash() == info.codebook_hash,
        "the file's codes index codebooks {:016x}, the decoder has {:016x}",
        info.codebook_hash,
        decoder.codebook_hash()
    );
    let start = Instant::now();
    let all_codes = stream.decode_all()?;
    let signals = all_codes
        .par_iter()
        .zip(&info.codebooks)
        .map(|(c, &n)| decoder.decode(c, n))
        .collect::<Result<Vec<_>>>()?;
    let channels = output_channels(signals, info.layout);
    let rate = usize::try_from(dac::SAMPLE_RATE)?;
    let raw = Audio {
        sample_rate: dac::SAMPLE_RATE,
        channels,
    }
    .to_seq()?;
    let mut trimmer = Trimmer::new(
        info.pre_skip,
        info.playable_samples(),
        info.looped,
        rate,
        raw.channels(),
    )?;
    let out: Seq = trimmer.push(&raw)?;
    drop(raw);
    trimmer.finish()?;
    let audio = Audio::from_seq(&out, dac::SAMPLE_RATE);
    eprintln!(
        "decoded {:.2} s with dac44-student in {:.2} s",
        audio.seconds(),
        start.elapsed().as_secs_f64()
    );
    Ok(audio)
}

/// Reads an HQ stream.
pub(crate) fn parse(bytes: &[u8]) -> Result<DacStream> {
    DacStream::parse(bytes).context("reading the HQ stream")
}

/// Prints what an HQ stream holds.
pub(crate) fn print_info(stream: &DacStream, size: usize) {
    let info = &stream.info;
    let playable = info.playable_samples();
    let seconds = picotone::config::to_f64(playable) / f64::from(dac::SAMPLE_RATE);
    eprintln!("model:      dac44-student (HQ)");
    eprintln!(
        "samples:    {playable} ({seconds:.2} s at {} Hz)",
        dac::SAMPLE_RATE
    );
    eprintln!("channels:   {}", info.layout.name());
    eprintln!(
        "codebooks:  {}",
        info.codebooks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("+")
    );
    eprintln!("code hash:  {:016x}", info.codebook_hash);
    eprintln!(
        "pages:      {} of {} frames ({} damaged)",
        stream.pages.len(),
        info.page_length,
        stream.pages.iter().filter(|p| p.is_none()).count()
    );
    if info.pre_skip != 0 || info.end_trim != 0 {
        eprintln!(
            "trim:       {} at the start, {} at the end",
            info.pre_skip, info.end_trim
        );
    }
    if info.looped {
        eprintln!("loop:       yes");
    }
    for tag in &info.tags {
        eprintln!("tag:        {tag}");
    }
    let kbps = if seconds > 0.0 {
        picotone::config::to_f64(size) * 8.0 / seconds / 1000.0
    } else {
        0.0
    };
    eprintln!("size:       {size} bytes ({kbps:.2} kbps)");
}
