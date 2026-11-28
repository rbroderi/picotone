//! The HQ encoder: the STFT encoder (`dac44_stftenc`) with DAC 44 kHz's
//! quantiser, and the stereo decisions (the layout pick decodes excerpts with
//! the Vocos decoder, `dac44-vocos`).

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use rayon::prelude::*;

use super::{Log, Models};
use crate::audio::Audio;
use crate::cx2::hq::HqModel;
use crate::dac::{self, DacQuantizer};
use crate::layout;
use crate::pico::{self, ChannelLayout, DacInfo, DacStream};
use crate::stftenc::StftEncoder;
use crate::vocos::VocosDecoder;
use crate::{pstereo, stereo};

/// The HQ encoder's (STFT encoder) weights in the weights directory.
pub const ENCODER_FILE: &str = "dac44_stftenc.safetensors";
/// DAC's quantiser's weights in the weights directory.
pub const QUANTIZER_FILE: &str = "dac44_quantizer.safetensors";
/// The HQ decoder's (Vocos) weights in the weights directory.
pub const DECODER_FILE: &str = "dac44_vocos.safetensors";
/// The HQ decoder's name in messages.
pub const DECODER_NAME: &str = "dac44-vocos";

/// The cx2 table HQ streams are coded with, in the tables directory.
pub const TABLE_FILE: &str = "cx2_dac44.ecxm";

/// Loads an HQ cx2 table.
///
/// # Errors
/// Fails when the file is missing or not an HQ table.
pub fn load_table(path: &Path) -> Result<HqModel> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    HqModel::from_table(&bytes).with_context(|| format!("loading {}", path.display()))
}

/// How to choose the stereo layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StereoMode {
    /// Always mid/side.
    MidSide,
    /// Always left/right.
    LeftRight,
    /// Always parametric stereo: the mid at `codebooks`, plus parameters.
    Parametric,
    /// Compare left/right and mid/side on sampled excerpts, and parametric
    /// stereo too when its mid ([`ps_pick_codebooks`]) is at 3 codebooks or
    /// fewer.
    Pick,
}

/// The pick's parametric mid: about the same total rate as the other
/// layouts (the parameters cost about one codebook).
const fn ps_pick_codebooks(s: &Settings<'_>) -> usize {
    let total = s.codebooks + s.side_codebooks;
    if total <= 2 { 1 } else { total - 1 }
}

/// The pick tries parametric stereo when its mid has at most this many
/// codebooks: above, mid/side scored better (`tools/pstereo_proto.py`).
const PS_PICK_MAX: usize = 3;

/// Settings of an HQ encode.
#[derive(Debug)]
#[expect(clippy::struct_excessive_bools, reason = "independent settings")]
pub(super) struct Settings<'a> {
    /// Output channels, 1 or 2.
    pub(super) channels: usize,
    /// Code a stereo input that is mono by the rule as one channel.
    pub(super) auto_mono: bool,
    /// Stereo layout choice.
    pub(super) stereo: StereoMode,
    /// Codebooks per frame (left/right: each channel; mid/side: the mid).
    pub(super) codebooks: usize,
    /// Codebooks per frame of the side.
    pub(super) side_codebooks: usize,
    /// Parametric stereo settings (for that layout, and for the pick).
    pub(super) parametric: pstereo::Config,
    /// Code the input as a seamless loop.
    pub(super) looped: bool,
    /// Comments.
    pub(super) tags: &'a [String],
    /// Frames per page.
    pub(super) page_length: Option<usize>,
    /// Leave the length out of the header.
    pub(super) no_length: bool,
    /// The cx2 table to code with.
    pub(super) table: &'a HqModel,
    /// Return the codes instead of coding them; the encode then returns no
    /// bytes.
    pub(super) codes_only: bool,
    /// Also write an enhancement layer up to this many codebooks per channel
    /// (the mid for mid/side).
    pub(super) layer_codebooks: Option<usize>,
    /// The layer's side codebooks (mid/side, or a parametric base's layer).
    pub(super) layer_side_codebooks: Option<usize>,
}

/// An encoded HQ file: the stream, and its enhancement layer when asked for;
/// or with [`Settings::codes_only`], the codes.
#[derive(Debug)]
pub(super) struct Encoded {
    pub(super) info: DacInfo,
    pub(super) bytes: Vec<u8>,
    pub(super) layer: Option<Vec<u8>>,
    pub(super) codes: Option<Codes>,
}

/// The combined layout and codebooks of the layer `s` asks for over a base
/// of `layout` with `codebooks`.
fn layer_plan(
    layout: ChannelLayout,
    codebooks: &[usize],
    s: &Settings<'_>,
) -> Result<Option<(ChannelLayout, Vec<usize>)>> {
    let Some(n) = s.layer_codebooks else {
        ensure!(
            s.layer_side_codebooks.is_none(),
            "--layer-side-codebooks needs --layer-codebooks"
        );
        return Ok(None);
    };
    for m in [Some(n), s.layer_side_codebooks].into_iter().flatten() {
        ensure!(
            (1..=dac::CODEBOOKS).contains(&m),
            "{m} layer codebooks requested; the HQ codec has 1 to {}",
            dac::CODEBOOKS
        );
    }
    ensure!(
        n != dac::DAC_CODEBOOKS + 1,
        "--layer-codebooks {n}: codebook {n} adds little on its own; use {} or {} to {}",
        dac::DAC_CODEBOOKS,
        dac::DAC_CODEBOOKS + 2,
        dac::CODEBOOKS
    );
    let side = s.layer_side_codebooks.unwrap_or(n);
    let plan = match layout {
        ChannelLayout::Mono => (ChannelLayout::Mono, vec![n]),
        ChannelLayout::LeftRight => (ChannelLayout::LeftRight, vec![n; 2]),
        ChannelLayout::MidSide | ChannelLayout::Parametric(_) => {
            (ChannelLayout::MidSide, vec![n, side])
        }
    };
    let _bases = pico::layer_bases(layout, codebooks, plan.0, &plan.1)
        .context("the layer must hold more codebooks than the base")?;
    Ok(Some(plan))
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

/// The HQ encoder: the STFT encoder makes the latent, DAC's quantiser its
/// codes.
#[derive(Debug)]
pub struct Encoder {
    /// The STFT encoder.
    pub net: StftEncoder,
    /// DAC 44 kHz's residual vector quantiser.
    pub quantizer: DacQuantizer,
}

impl Encoder {
    /// Hash of the codebooks the codes index.
    #[must_use]
    pub fn codebook_hash(&self) -> u64 {
        self.quantizer.codebook_hash()
    }

    /// Codes of mono 44.1 kHz `audio`, `[frames][16]` (row-major),
    /// `ceil(len / 512)` frames (the audio read as zero padded at the end).
    ///
    /// # Errors
    /// Fails when the networks reject the input.
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<u16>> {
        self.quantizer.quantize(&self.net.latent(audio)?)
    }
}

/// Loads the HQ encoder from the weights directory.
///
/// # Errors
/// Fails when the weights are missing or invalid.
pub fn load_encoder(weights: &Path, log: Log<'_>) -> Result<Encoder> {
    let start = Instant::now();
    let net = StftEncoder::load(&file(weights, ENCODER_FILE)?)?;
    ensure!(
        net.latent_dim() == dac::LATENT_DIM,
        "{ENCODER_FILE} makes a latent {} wide; DAC's quantiser reads {}",
        net.latent_dim(),
        dac::LATENT_DIM
    );
    let quantizer = DacQuantizer::load(&file(weights, QUANTIZER_FILE)?)?;
    log(&format!(
        "loaded the HQ encoder ({}) in {:.2} s",
        crate::stftenc::NAME,
        start.elapsed().as_secs_f64()
    ));
    Ok(Encoder { net, quantizer })
}

/// Loads the HQ decoder from the weights directory.
///
/// # Errors
/// Fails when the weights are missing or invalid.
pub fn load_decoder(weights: &Path, log: Log<'_>) -> Result<VocosDecoder> {
    let start = Instant::now();
    let decoder = VocosDecoder::load(&file(weights, DECODER_FILE)?)?;
    log(&format!(
        "loaded the HQ decoder ({DECODER_NAME}) in {:.2} s",
        start.elapsed().as_secs_f64()
    ));
    Ok(decoder)
}

/// The signals a layout codes: the channels, mid and side, or the mid.
fn coded_signals(channels: &[Vec<f32>], layout: ChannelLayout) -> Vec<Vec<f32>> {
    let mid = || -> Vec<f32> {
        channels[0]
            .iter()
            .zip(&channels[1])
            .map(|(a, b)| (a + b) / 2.0)
            .collect()
    };
    match layout {
        ChannelLayout::MidSide => {
            let (l, r) = (&channels[0], &channels[1]);
            vec![mid(), l.iter().zip(r).map(|(a, b)| (a - b) / 2.0).collect()]
        }
        ChannelLayout::Parametric(_) => vec![mid()],
        ChannelLayout::Mono | ChannelLayout::LeftRight => channels.to_vec(),
    }
}

/// Parametric stereo parameters of a stream, and its pages.
type Parameters<'a> = (pstereo::Params, &'a [Range<usize>]);

/// Decoded coded signals back to output channels.
fn output_channels(
    mut decoded: Vec<Vec<f32>>,
    layout: ChannelLayout,
    parameters: Option<&Parameters<'_>>,
) -> Result<Vec<Vec<f32>>> {
    Ok(match layout {
        ChannelLayout::MidSide if decoded.len() == 2 => {
            let side = decoded.pop().unwrap_or_default();
            let mid = decoded.pop().unwrap_or_default();
            let left = mid.iter().zip(&side).map(|(m, s)| m + s).collect();
            let right = mid.iter().zip(&side).map(|(m, s)| m - s).collect();
            vec![left, right]
        }
        ChannelLayout::Parametric(config) => {
            let (params, spans) = parameters.context("parametric stereo without its parameters")?;
            let mid = decoded
                .first()
                .context("parametric stereo without its mid")?;
            pstereo::upmix(mid, params, spans, config)?.into()
        }
        _ => decoded,
    })
}

/// Codebooks per coded channel of `layout`.
fn layout_codebooks(layout: ChannelLayout, s: &Settings<'_>, picked: bool) -> Vec<usize> {
    match layout {
        ChannelLayout::Parametric(_) if picked => vec![ps_pick_codebooks(s)],
        ChannelLayout::Mono | ChannelLayout::Parametric(_) => vec![s.codebooks],
        ChannelLayout::LeftRight => vec![s.codebooks; 2],
        ChannelLayout::MidSide => vec![s.codebooks, s.side_codebooks],
    }
}

/// Each page's coded parametric stereo parameters of `channels` (none for
/// the other layouts).
fn stereo_parameters(
    channels: &[Vec<f32>],
    layout: ChannelLayout,
    spans: &[Range<usize>],
) -> Vec<Vec<u8>> {
    match layout {
        ChannelLayout::Parametric(config) => {
            let params = pstereo::analyse(&channels[0], &channels[1], spans, config);
            pstereo::write_pages(&params, spans, config)
        }
        _ => Vec::new(),
    }
}

/// Codes of each signal, cut to its tier: `[frame][codebook]`.
fn encode_signals(
    encoder: &Encoder,
    signals: &[Vec<f32>],
    codebooks: &[usize],
) -> Result<Vec<Vec<u16>>> {
    signals
        .par_iter()
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

/// Encodes `source` as an HQ stream, and its layer when `s` asks for one.
pub(super) fn encode(
    models: &mut Models,
    source: Audio,
    s: &Settings<'_>,
    log: Log<'_>,
) -> Result<Encoded> {
    for n in [s.codebooks, s.side_codebooks] {
        ensure!(
            (1..=dac::CODEBOOKS).contains(&n),
            "{n} codebooks requested; the HQ codec has 1 to {}",
            dac::CODEBOOKS
        );
    }
    let encoder = models.hq_encoder(log)?;
    let mut channels = source.convert(dac::SAMPLE_RATE, s.channels)?.channels;
    drop(source);
    if channels.len() == 2 && s.auto_mono && stereo::is_mono(&channels[0], &channels[1]) {
        log("the input is mono (its side is over 40 dB under its mid): coding one channel");
        channels = coded_signals(&channels, ChannelLayout::MidSide);
        channels.truncate(1);
    }
    let (layout, picked) = choose_layout(models, &encoder, &channels, s, log)?;
    let codebooks = layout_codebooks(layout, s, picked);
    let layer = layer_plan(layout, &codebooks, s)?;
    ensure!(
        layer.is_none() || !s.codes_only,
        "--dump-codes writes no layer"
    );
    let mut info = DacInfo {
        codebook_hash: encoder.codebook_hash(),
        table_hash: s.table.hash(),
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
    let (codes, combined) = all_codes(&encoder, &channels, layout, &codebooks, layer.as_ref())?;
    let parameters = stereo_parameters(&channels, layout, &info.pages());
    if s.codes_only {
        return Ok(Encoded {
            info,
            bytes: Vec::new(),
            layer: None,
            codes: Some(codes),
        });
    }
    report_parameters(layout, &parameters, info.audio_length, log);
    let coding = Instant::now();
    let bytes = pico::write_dac_parametric(&info, s.table, &codes, &parameters)?;
    let layer_bytes = match (&layer, combined) {
        (Some((combined_layout, books)), Some(all)) => Some(pico::write_dac_layer(
            &DacStream::parse(&bytes)?,
            *combined_layout,
            books.clone(),
            s.table,
            &all,
            pico::DEFAULT_LAYER_GROUP,
        )?),
        _ => None,
    };
    let cx2_seconds = coding.elapsed().as_secs_f64();
    log(&format!(
        "encoded HQ ({}, {} codebooks{}{}) in {:.2} s (cx2 {:.2} s)",
        layout.name(),
        join_books(&codebooks),
        layer
            .as_ref()
            .map_or_else(String::new, |(l, books)| format!(
                "; layer to {}, {}",
                l.name(),
                join_books(books)
            )),
        if s.looped { ", loop" } else { "" },
        start.elapsed().as_secs_f64(),
        cx2_seconds
    ));
    Ok(Encoded {
        info,
        bytes,
        layer: layer_bytes,
        codes: None,
    })
}

/// The base's codes of each coded channel, and with a layer the combined
/// stream's. With a layer the combined stream's signals are coded and the
/// base's codes cut from them (each base channel is the same combined
/// channel's first codebooks), so base plus layer are one encode.
fn all_codes(
    encoder: &Encoder,
    channels: &[Vec<f32>],
    layout: ChannelLayout,
    codebooks: &[usize],
    layer: Option<&(ChannelLayout, Vec<usize>)>,
) -> Result<(Codes, Option<Codes>)> {
    let Some((combined, books)) = layer else {
        let codes = encode_signals(encoder, &coded_signals(channels, layout), codebooks)?;
        return Ok((codes, None));
    };
    let all = encode_signals(encoder, &coded_signals(channels, *combined), books)?;
    let base = all
        .iter()
        .zip(books)
        .zip(codebooks)
        .map(|((c, &n), &b)| {
            c.chunks(n)
                .flat_map(|frame| frame[..b].iter().copied())
                .collect()
        })
        .collect();
    Ok((base, Some(all)))
}

/// Codes of each coded channel, `[frame][codebook]`.
pub(super) type Codes = Vec<Vec<u16>>;

/// The layout to code `channels` with, and whether the pick chose it.
fn choose_layout(
    models: &mut Models,
    encoder: &Encoder,
    channels: &[Vec<f32>],
    s: &Settings<'_>,
    log: Log<'_>,
) -> Result<(ChannelLayout, bool)> {
    if channels.len() == 1 {
        return Ok((ChannelLayout::Mono, false));
    }
    Ok(match s.stereo {
        StereoMode::LeftRight => (ChannelLayout::LeftRight, false),
        StereoMode::MidSide => (ChannelLayout::MidSide, false),
        StereoMode::Parametric => (ChannelLayout::Parametric(s.parametric), false),
        StereoMode::Pick => {
            let decoder = models.hq_decoder(log)?;
            (pick(&decoder, encoder, channels, s, log)?, true)
        }
    })
}

/// Reports the rate of a parametric stream's parameters.
fn report_parameters(
    layout: ChannelLayout,
    parameters: &[Vec<u8>],
    audio_length: usize,
    log: Log<'_>,
) {
    if let ChannelLayout::Parametric(config) = layout {
        let bits = 8 * parameters.iter().map(Vec::len).sum::<usize>();
        let seconds = crate::config::to_f64(audio_length) / f64::from(dac::SAMPLE_RATE);
        log(&format!(
            "parametric stereo: {}, {:.2} kbps of parameters",
            config.describe(),
            if seconds > 0.0 {
                crate::config::to_f64(bits) / seconds / 1000.0
            } else {
                0.0
            }
        ));
    }
}

/// Codebook counts as `9+4`.
#[must_use]
pub fn join_books(books: &[usize]) -> String {
    books
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("+")
}

/// Picks left/right or mid/side from sampled excerpts.
fn pick(
    decoder: &VocosDecoder,
    encoder: &Encoder,
    channels: &[Vec<f32>],
    s: &Settings<'_>,
    log: Log<'_>,
) -> Result<ChannelLayout> {
    let start = Instant::now();
    let mut candidates = vec![ChannelLayout::LeftRight, ChannelLayout::MidSide];
    if ps_pick_codebooks(s) <= PS_PICK_MAX {
        candidates.push(ChannelLayout::Parametric(s.parametric));
    }
    let page = s.page_length.unwrap_or(pico::DEFAULT_PAGE_FRAMES);
    let (layout, scores) =
        stereo::pick_layout(&channels[0], &channels[1], &candidates, |l, r, layout| {
            let books = layout_codebooks(layout, s, true);
            let excerpt = [l.to_vec(), r.to_vec()];
            let frames = l.len().div_ceil(dac::HOP);
            let per = if page == 0 { frames.max(1) } else { page };
            let spans: Vec<Range<usize>> = (0..frames)
                .step_by(per)
                .map(|f| f..(f + per).min(frames))
                .collect();
            let parameters = match layout {
                ChannelLayout::Parametric(config) => {
                    Some((pstereo::analyse(l, r, &spans, config), spans.as_slice()))
                }
                _ => None,
            };
            let codes = encode_signals(encoder, &coded_signals(&excerpt, layout), &books)?;
            let signals = codes
                .par_iter()
                .zip(&books)
                .map(|(c, &n)| {
                    let mut y = decoder.decode(c, n)?;
                    y.truncate(l.len());
                    Ok(y)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut out = output_channels(signals, layout, parameters.as_ref())?.into_iter();
            Ok([
                out.next().unwrap_or_default(),
                out.next().unwrap_or_default(),
            ])
        })?;
    log(&format!(
        "stereo: {} (log-mel {}, {}) in {:.2} s",
        layout.name(),
        scores
            .layouts
            .iter()
            .map(|(l, v)| format!("{v:.3} {}", l.name()))
            .collect::<Vec<_>>()
            .join(", "),
        if scores.excerpts == 0 {
            "whole file".to_owned()
        } else {
            format!("{} excerpt(s)", scores.excerpts)
        },
        start.elapsed().as_secs_f64()
    ));
    Ok(layout)
}
