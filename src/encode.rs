//! Encoding audio to `.pico` files: the HQ codec (the default) or one of the
//! EnCodec models.
//!
//! [`Options`] holds the settings `picotone encode` takes, with the same
//! defaults and checks; [`encode`] codes an [`Audio`] with them and returns the
//! file's bytes (and an HQ enhancement layer's). The networks and cx2 tables
//! load on first use into a [`Models`], which keeps them for later encodes.
//! Progress messages go to a [`Log`] callback.

pub mod hq;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result, bail, ensure};

use crate::audio::Audio;
use crate::config::Config;
use crate::cx2::hq::HqModel;
use crate::layout::{self, Crossfade, Segmenting};
use crate::model::{EncodeParams, Encodec, LoadOptions};
use crate::pico::{self, DacInfo, Info};
use crate::vocos::VocosDecoder;
use crate::{ModelKind, cx2, dac, pstereo};

/// Receives each progress message of an encode (one line, no newline).
pub type Log<'a> = &'a mut dyn FnMut(&str);

/// The model to encode with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// HQ: picotone's STFT encoder with DAC 44 kHz's quantiser; 44.1 kHz, mono
    /// or stereo, 1 to 16 codebooks per channel.
    Hq,
    /// `encodec_48khz`: 48 kHz stereo.
    Encodec48k,
    /// `encodec_24khz`: 24 kHz mono.
    Encodec24k,
}

/// The channels to code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelFormat {
    /// Follow the input. For HQ, a stereo input that is mono in all but name
    /// (its side over 40 dB under its mid) is coded as one channel.
    #[default]
    Auto,
    /// One channel (a down-mix).
    Mono,
    /// Two channels.
    Stereo,
}

/// The HQ stereo layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoChoice {
    /// Compare left/right and mid/side on sampled excerpts and keep the
    /// better; at 2 codebooks or fewer per channel, also parametric stereo.
    Pick,
    /// Mid/side.
    Ms,
    /// Left/right.
    Lr,
    /// Parametric stereo: the mid plus per-band level and coherence
    /// parameters.
    Parametric,
}

/// Recommended EnCodec 48 kHz music settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// 15 codebooks, stride 44800, triangle cross-fade, beam 8.
    Music24,
    /// 8 codebooks, stride 44800, triangle cross-fade, beam 8.
    Music12,
    /// 4 codebooks, stride 44800, triangle cross-fade, beam 8.
    Music6,
}

/// The EnCodec cross-fade between segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeChoice {
    /// EnCodec's triangle-weighted overlap-add.
    Triangle,
    /// A linear ramp that avoids the damaged segment edges.
    Edge,
}

/// EnCodec bandwidth in kbps when neither `bandwidth` nor `codebooks` sets one.
pub const DEFAULT_BANDWIDTH: f64 = 6.0;

/// Beam width used when neither `beam` nor a preset sets one.
pub const DEFAULT_BEAM: usize = 8;

/// Parametric stereo bands when `ps_bands` is not set.
pub const DEFAULT_PS_BANDS: usize = 20;

/// Segment stride of the music presets: 140 latent frames.
const PRESET_STRIDE: usize = 44_800;

/// Beam width the presets use.
const PRESET_BEAM: usize = 8;

impl Preset {
    /// Codebooks, stride, cross-fade and beam width.
    const fn settings(self) -> (usize, usize, CrossfadeChoice, usize) {
        match self {
            Self::Music24 => (15, PRESET_STRIDE, CrossfadeChoice::Triangle, PRESET_BEAM),
            Self::Music12 => (8, PRESET_STRIDE, CrossfadeChoice::Triangle, PRESET_BEAM),
            Self::Music6 => (4, PRESET_STRIDE, CrossfadeChoice::Triangle, PRESET_BEAM),
        }
    }
}

/// The settings of an encode, as `picotone encode` takes them.
///
/// `None` and `false` are the command's defaults: HQ, the channels of the
/// input, 9 codebooks per channel, the stereo pick; with `fast`, EnCodec at
/// 6 kbps with beam 8.
#[derive(Debug, Clone, PartialEq, Default)]
#[expect(clippy::struct_excessive_bools, reason = "independent flags")]
pub struct Options {
    /// Use EnCodec instead of HQ: encodec-24k for mono, encodec-48k for
    /// stereo (`--fast`).
    pub fast: bool,
    /// The exact model; overrides `fast`'s and `preset`'s choice.
    pub model: Option<Model>,
    /// The channels to code.
    pub channel_format: ChannelFormat,
    /// HQ stereo layout (default: the pick).
    pub stereo: Option<StereoChoice>,
    /// EnCodec target bandwidth in kbps (default 6).
    pub bandwidth: Option<f64>,
    /// EnCodec 48 kHz music settings; implies encodec-48k. The explicit
    /// fields override them.
    pub preset: Option<Preset>,
    /// Codebooks per frame. HQ: per channel (the mid for mid/side), 1 to 16,
    /// default 9. EnCodec: instead of the count `bandwidth` implies.
    pub codebooks: Option<usize>,
    /// HQ mid/side: the side's codebooks (default: `codebooks`).
    pub side_codebooks: Option<usize>,
    /// HQ: also code an enhancement layer up to this many codebooks per
    /// channel (2 to 16 but not 10).
    pub layer_codebooks: Option<usize>,
    /// HQ layer of a mid/side or parametric file: the side's codebooks with
    /// the layer (default: `layer_codebooks`).
    pub layer_side_codebooks: Option<usize>,
    /// HQ parametric stereo: parameter bands, 20 (default) or 10.
    pub ps_bands: Option<usize>,
    /// HQ parametric stereo: frames per parameter set, 1, 2, 4, 8 (default)
    /// or 16.
    pub ps_frames: Option<usize>,
    /// EnCodec segment stride in samples (48 kHz model: 24000 to 48000).
    pub stride: Option<usize>,
    /// EnCodec cross-fade between segments (default: triangle).
    pub crossfade: Option<CrossfadeChoice>,
    /// EnCodec beam width of the quantiser search (default 8; 1 is greedy).
    pub beam: Option<usize>,
    /// Code the input as a seamless loop (`--loop`).
    pub looped: bool,
    /// `KEY=value` comments.
    pub tags: Vec<String>,
    /// cx2 table to code with, instead of the model's in the tables
    /// directory.
    pub table: Option<PathBuf>,
    /// Segments (EnCodec 24 kHz and HQ: frames) per Ogg page (default from
    /// the bitrate; HQ: 80).
    pub page_length: Option<usize>,
    /// Leave the length out of the header, as a streaming encoder would.
    pub no_length: bool,
    /// HQ: return the codes in [`Encoded::codes`] instead of coding a file
    /// (EnCodec ignores it).
    pub codes_only: bool,
}

impl Options {
    /// Sets the option named as `picotone encode`'s long flag without the
    /// dashes (`model`, `fast`, `channel-format`, `stereo`, `bandwidth`,
    /// `preset`, `codebooks`, `side-codebooks`, `layer-codebooks`,
    /// `layer-side-codebooks`, `ps-bands`, `ps-frames`, `stride`,
    /// `crossfade`, `beam`, `loop`) from its text; `fast` and `loop` take `1`
    /// or `0`.
    ///
    /// # Errors
    /// Fails for an unknown key, or a value that does not parse; combinations
    /// are checked by [`encode`].
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        fn number<T: std::str::FromStr>(key: &str, value: &str) -> Result<Option<T>> {
            value
                .parse()
                .map(Some)
                .ok()
                .with_context(|| format!("--{key} {value}: not a number"))
        }
        fn flag(key: &str, value: &str) -> Result<bool> {
            match value {
                "1" | "true" => Ok(true),
                "0" | "false" => Ok(false),
                _ => bail!("--{key} {value}: expected 1 or 0"),
            }
        }
        fn choice<T: Copy>(key: &str, value: &str, names: &[(&str, T)]) -> Result<Option<T>> {
            names
                .iter()
                .find(|(name, _)| *name == value)
                .map(|&(_, v)| Some(v))
                .with_context(|| {
                    let all = names.iter().map(|(n, _)| *n).collect::<Vec<_>>();
                    format!("--{key} {value}: expected one of {}", all.join(", "))
                })
        }
        match key {
            "fast" => self.fast = flag(key, value)?,
            "loop" => self.looped = flag(key, value)?,
            "model" => {
                self.model = choice(
                    key,
                    value,
                    &[
                        ("hq", Model::Hq),
                        ("encodec-48k", Model::Encodec48k),
                        ("encodec-24k", Model::Encodec24k),
                    ],
                )?;
            }
            "channel-format" => {
                self.channel_format = choice(
                    key,
                    value,
                    &[
                        ("auto", ChannelFormat::Auto),
                        ("mono", ChannelFormat::Mono),
                        ("stereo", ChannelFormat::Stereo),
                    ],
                )?
                .unwrap_or_default();
            }
            "stereo" => {
                self.stereo = choice(
                    key,
                    value,
                    &[
                        ("pick", StereoChoice::Pick),
                        ("ms", StereoChoice::Ms),
                        ("lr", StereoChoice::Lr),
                        ("parametric", StereoChoice::Parametric),
                        ("ps", StereoChoice::Parametric),
                    ],
                )?;
            }
            "preset" => {
                self.preset = choice(
                    key,
                    value,
                    &[
                        ("music24", Preset::Music24),
                        ("music12", Preset::Music12),
                        ("music6", Preset::Music6),
                    ],
                )?;
            }
            "crossfade" => {
                self.crossfade = choice(
                    key,
                    value,
                    &[
                        ("triangle", CrossfadeChoice::Triangle),
                        ("edge", CrossfadeChoice::Edge),
                    ],
                )?;
            }
            "bandwidth" => self.bandwidth = number(key, value)?,
            "codebooks" => self.codebooks = number(key, value)?,
            "side-codebooks" => self.side_codebooks = number(key, value)?,
            "layer-codebooks" => self.layer_codebooks = number(key, value)?,
            "layer-side-codebooks" => self.layer_side_codebooks = number(key, value)?,
            "ps-bands" => self.ps_bands = number(key, value)?,
            "ps-frames" => self.ps_frames = number(key, value)?,
            "stride" => self.stride = number(key, value)?,
            "beam" => self.beam = number(key, value)?,
            other => bail!("unknown encode option {other}"),
        }
        Ok(())
    }

    /// Adds a `KEY=value` comment (`--tag`).
    ///
    /// # Errors
    /// Fails when it is not `KEY=value`.
    pub fn add_tag(&mut self, tag: &str) -> Result<()> {
        check_tag(tag)?;
        self.tags.push(tag.to_owned());
        Ok(())
    }
}

/// Checks that a comment is `KEY=value`.
///
/// # Errors
/// Fails when it is not.
pub fn check_tag(tag: &str) -> Result<()> {
    ensure!(tag.contains('='), "--tag {tag} is not KEY=value");
    Ok(())
}

/// What an encode runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    /// HQ with this many channels; `auto_mono` codes a mono-like stereo input as one.
    Hq { channels: usize, auto_mono: bool },
    /// An EnCodec model.
    Encodec(ModelKind),
}

/// Resolves `model`, `fast`, `preset` and `channel_format` for an input with
/// `input_channels` channels.
fn plan(options: &Options, input_channels: usize) -> Result<Plan> {
    let format = options.channel_format;
    let stereo_input = options.preset.is_some() || input_channels >= 2;
    let model = options.model.unwrap_or_else(|| {
        if !options.fast && options.preset.is_none() {
            Model::Hq
        } else if format == ChannelFormat::Stereo || (format == ChannelFormat::Auto && stereo_input)
        {
            Model::Encodec48k
        } else {
            Model::Encodec24k
        }
    });
    let plan = match (model, format) {
        (Model::Hq, ChannelFormat::Mono) => Plan::Hq {
            channels: 1,
            auto_mono: false,
        },
        (Model::Hq, ChannelFormat::Stereo) => Plan::Hq {
            channels: 2,
            auto_mono: false,
        },
        (Model::Hq, ChannelFormat::Auto) => Plan::Hq {
            channels: if input_channels >= 2 { 2 } else { 1 },
            auto_mono: true,
        },
        (Model::Encodec48k, ChannelFormat::Mono) => {
            bail!("encodec-48k is a stereo model; use encodec-24k (or --fast) for mono")
        }
        (Model::Encodec24k, ChannelFormat::Stereo) => {
            bail!("encodec-24k is a mono model; use encodec-48k (or --fast) for stereo")
        }
        (Model::Encodec48k, _) => Plan::Encodec(ModelKind::Stereo48k),
        (Model::Encodec24k, _) => Plan::Encodec(ModelKind::Mono24k),
    };
    match plan {
        Plan::Hq { .. } => {
            let encodec_only = [
                (options.preset.is_some(), "--preset"),
                (options.bandwidth.is_some(), "--bandwidth"),
                (options.stride.is_some(), "--stride"),
                (options.crossfade.is_some(), "--crossfade"),
                (options.beam.is_some(), "--beam"),
                (options.table.is_some(), "--table"),
            ];
            if let Some((_, flag)) = encodec_only.iter().find(|(set, _)| *set) {
                bail!("{flag} is for the EnCodec models (--fast); HQ takes --codebooks");
            }
        }
        Plan::Encodec(_) => {
            ensure!(
                options.side_codebooks.is_none()
                    && options.stereo.is_none()
                    && options.ps_bands.is_none()
                    && options.ps_frames.is_none()
                    && options.layer_codebooks.is_none()
                    && options.layer_side_codebooks.is_none(),
                "--stereo, --side-codebooks, --ps-bands, --ps-frames and the --layer options are \
                 for the HQ model"
            );
        }
    }
    Ok(plan)
}

/// File name of the cx2 table an EnCodec model codes with by default.
#[must_use]
pub const fn default_table(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Mono24k => "cx2_encodec_24khz_v5.ecxm",
        ModelKind::Stereo48k => "cx2_encodec_48khz_net2_v5.ecxm",
    }
}

/// Loads a self-contained EnCodec cx2 table.
///
/// # Errors
/// Fails when the file is missing or invalid.
pub fn load_table(path: &Path) -> Result<cx2::Model> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    cx2::Model::from_table(&bytes).with_context(|| format!("loading {}", path.display()))
}

/// The networks and cx2 tables encodes use, loaded from the weights and
/// tables directories on first use and kept.
#[derive(Debug)]
pub struct Models {
    weights: PathBuf,
    tables: PathBuf,
    /// Also load the EnCodec decoder, for [`Models::encodec`].
    encodec_decoder: bool,
    hq_encoder: Option<Arc<hq::Encoder>>,
    hq_decoder: Option<Arc<VocosDecoder>>,
    hq_table: Option<(PathBuf, Arc<HqModel>)>,
    encodec: Option<(ModelKind, LoadOptions, Arc<Encodec>)>,
    encodec_table: Option<(PathBuf, Arc<cx2::Model>)>,
}

impl Models {
    /// Nothing loaded yet; the files come from `weights` and `tables`.
    #[must_use]
    pub const fn new(weights: PathBuf, tables: PathBuf) -> Self {
        Self {
            weights,
            tables,
            encodec_decoder: false,
            hq_encoder: None,
            hq_decoder: None,
            hq_table: None,
            encodec: None,
            encodec_table: None,
        }
    }

    /// Also loads the decoder half of an EnCodec model, so the model
    /// [`Models::encodec`] returns can decode what it coded.
    #[must_use]
    pub const fn with_encodec_decoder(mut self) -> Self {
        self.encodec_decoder = true;
        self
    }

    /// The weights directory.
    #[must_use]
    pub fn weights(&self) -> &Path {
        &self.weights
    }

    /// The EnCodec model of the last EnCodec encode, when it was `kind`.
    #[must_use]
    pub fn encodec(&self, kind: ModelKind) -> Option<&Encodec> {
        self.encodec
            .as_ref()
            .filter(|(k, ..)| *k == kind)
            .map(|(.., model)| &**model)
    }

    /// Uses `decoder` (`dac44_vocos`, loaded elsewhere) for the HQ stereo pick.
    pub fn set_hq_decoder(&mut self, decoder: Arc<VocosDecoder>) {
        self.hq_decoder = Some(decoder);
    }

    /// The HQ decoder, when an encode has loaded it (or it was set).
    #[must_use]
    pub fn loaded_hq_decoder(&self) -> Option<Arc<VocosDecoder>> {
        self.hq_decoder.clone()
    }

    /// The HQ encoder, loaded on first use.
    ///
    /// # Errors
    /// Fails when the weights are missing or invalid.
    pub fn hq_encoder(&mut self, log: Log<'_>) -> Result<Arc<hq::Encoder>> {
        if let Some(encoder) = &self.hq_encoder {
            return Ok(Arc::clone(encoder));
        }
        let encoder = Arc::new(hq::load_encoder(&self.weights, log)?);
        self.hq_encoder = Some(Arc::clone(&encoder));
        Ok(encoder)
    }

    /// The HQ decoder, loaded on first use.
    ///
    /// # Errors
    /// Fails when the weights are missing or invalid.
    pub fn hq_decoder(&mut self, log: Log<'_>) -> Result<Arc<VocosDecoder>> {
        if let Some(decoder) = &self.hq_decoder {
            return Ok(Arc::clone(decoder));
        }
        let decoder = Arc::new(hq::load_decoder(&self.weights, log)?);
        self.hq_decoder = Some(Arc::clone(&decoder));
        Ok(decoder)
    }

    fn hq_table(&mut self, explicit: Option<&Path>) -> Result<Arc<HqModel>> {
        let path = explicit.map_or_else(|| self.tables.join(hq::TABLE_FILE), Path::to_path_buf);
        if let Some((p, table)) = &self.hq_table
            && *p == path
        {
            return Ok(Arc::clone(table));
        }
        let table = Arc::new(hq::load_table(&path)?);
        self.hq_table = Some((path, Arc::clone(&table)));
        Ok(table)
    }

    fn encodec_table(
        &mut self,
        explicit: Option<&Path>,
        kind: ModelKind,
    ) -> Result<Arc<cx2::Model>> {
        let path =
            explicit.map_or_else(|| self.tables.join(default_table(kind)), Path::to_path_buf);
        if let Some((p, table)) = &self.encodec_table
            && *p == path
        {
            return Ok(Arc::clone(table));
        }
        let table = Arc::new(load_table(&path)?);
        self.encodec_table = Some((path, Arc::clone(&table)));
        Ok(table)
    }

    /// The EnCodec model `kind` with its encoder and `codebooks` codebooks
    /// (and its decoder when kept); a loaded one is used when it holds them.
    fn encodec_model(
        &mut self,
        kind: ModelKind,
        codebooks: usize,
        log: Log<'_>,
    ) -> Result<Arc<Encodec>> {
        let decoder = self.encodec_decoder;
        if let Some((k, parts, model)) = &self.encodec
            && *k == kind
            && parts.encoder
            && (parts.decoder || !decoder)
            && parts.codebooks.is_none_or(|n| n >= codebooks)
        {
            return Ok(Arc::clone(model));
        }
        self.encodec = None;
        // Only the halves and the codebooks this run needs are read from disk.
        let parts = LoadOptions {
            encoder: true,
            decoder,
            codebooks: Some(codebooks),
        };
        let start = Instant::now();
        let model = Arc::new(kind.load_with(&self.weights, parts)?);
        log(&format!(
            "loaded {} in {:.2} s",
            kind.name(),
            start.elapsed().as_secs_f64()
        ));
        self.encodec = Some((kind, parts, Arc::clone(&model)));
        Ok(model)
    }
}

/// The header of an encoded file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Header {
    /// An HQ stream's.
    Hq(DacInfo),
    /// An EnCodec stream's.
    Encodec(Info),
}

/// An encoded file.
#[derive(Debug)]
pub struct Encoded {
    /// What the file holds.
    pub header: Header,
    /// The `.pico` file (empty with [`Options::codes_only`]).
    pub bytes: Vec<u8>,
    /// The HQ `.picol` enhancement layer, when [`Options::layer_codebooks`]
    /// asked for one.
    pub layer: Option<Vec<u8>>,
    /// With [`Options::codes_only`], each coded channel's HQ codes,
    /// `[frame][codebook]`.
    pub codes: Option<Vec<Vec<u16>>>,
    /// Playable seconds.
    pub seconds: f64,
}

/// Encodes `source` with `options`, loading what it needs into `models`.
///
/// # Errors
/// Fails on settings `picotone encode` rejects (the messages name its flags),
/// missing weights or tables, or audio that cannot be coded.
pub fn encode(
    models: &mut Models,
    source: Audio,
    options: &Options,
    log: Log<'_>,
) -> Result<Encoded> {
    for tag in &options.tags {
        check_tag(tag)?;
    }
    let kind = match plan(options, source.channels.len())? {
        Plan::Encodec(kind) => kind,
        Plan::Hq {
            channels,
            auto_mono,
        } => return encode_hq(models, source, options, channels, auto_mono, log),
    };
    let table = models.encodec_table(options.table.as_deref(), kind)?;
    let config = kind.config(&models.weights)?;
    let (params, segmenting) = encode_params(options, kind, &config)?;
    let model = models.encodec_model(kind, params.num_codebooks, log)?;
    let cfg = model.config();
    let rate = u32::try_from(cfg.sampling_rate)?;
    // Each stage's copy of the audio is released as soon as the next exists.
    let converted = source.convert(rate, cfg.audio_channels)?;
    drop(source);
    let mut audio = converted.to_seq()?;
    drop(converted);
    let mut info = Info {
        model: kind,
        table_hash: table.hash(),
        num_codebooks: params.num_codebooks,
        stride: segmenting.stride.or(cfg.segment_stride()?),
        crossfade: segmenting.crossfade,
        looped: false,
        pre_skip: 0,
        page_length: options
            .page_length
            .unwrap_or_else(|| pico::default_page_length(kind, params.num_codebooks)),
        audio_length: 0,
        end_trim: 0,
        length_in_header: !options.no_length,
        output_gain: 0,
        tags: options.tags.clone(),
    };
    if options.looped {
        let (pre, post) = layout::loop_padding(cfg.sampling_rate);
        audio = layout::wrap_loop(&audio, pre, post)?;
        info.pre_skip = pre;
        info.end_trim = post;
        info.looped = true;
    }
    info.audio_length = audio.len();
    let start = Instant::now();
    let frames = model.encode_with(&audio, &params)?;
    drop(audio);
    let bytes = pico::write(&table, &info, &frames)?;
    log(&format!(
        "encoded with {} ({} codebooks, stride {}, {} cross-fade, beam {}{}) in {:.2} s",
        model.name(),
        info.num_codebooks,
        info.stride
            .map_or_else(|| "none".to_owned(), |s| s.to_string()),
        if info.crossfade == Crossfade::EdgeAvoiding {
            "edge-avoiding"
        } else {
            "triangle"
        },
        params.beam,
        if options.looped { ", loop" } else { "" },
        start.elapsed().as_secs_f64()
    ));
    let seconds =
        crate::config::to_f64(info.playable_samples()) / crate::config::to_f64(cfg.sampling_rate);
    Ok(Encoded {
        header: Header::Encodec(info),
        bytes,
        layer: None,
        codes: None,
        seconds,
    })
}

fn encode_hq(
    models: &mut Models,
    source: Audio,
    options: &Options,
    channels: usize,
    auto_mono: bool,
    log: Log<'_>,
) -> Result<Encoded> {
    let codebooks = options.codebooks.unwrap_or(dac::DEFAULT_CODEBOOKS);
    let table = models.hq_table(options.table.as_deref())?;
    let settings = hq::Settings {
        channels,
        auto_mono,
        stereo: match options.stereo {
            Some(StereoChoice::Ms) => hq::StereoMode::MidSide,
            Some(StereoChoice::Lr) => hq::StereoMode::LeftRight,
            Some(StereoChoice::Parametric) => hq::StereoMode::Parametric,
            Some(StereoChoice::Pick) | None => hq::StereoMode::Pick,
        },
        parametric: pstereo::Config::new(
            options.ps_bands.unwrap_or(DEFAULT_PS_BANDS),
            options.ps_frames.unwrap_or(pstereo::Config::DEFAULT.frames),
        )?,
        codebooks,
        side_codebooks: options.side_codebooks.unwrap_or(codebooks),
        looped: options.looped,
        tags: &options.tags,
        page_length: options.page_length,
        no_length: options.no_length,
        table: &table,
        codes_only: options.codes_only,
        layer_codebooks: options.layer_codebooks,
        layer_side_codebooks: options.layer_side_codebooks,
    };
    let hq::Encoded {
        info,
        bytes,
        layer,
        codes,
    } = hq::encode(models, source, &settings, log)?;
    let seconds = crate::config::to_f64(info.playable_samples()) / f64::from(dac::SAMPLE_RATE);
    Ok(Encoded {
        header: Header::Hq(info),
        bytes,
        layer,
        codes,
        seconds,
    })
}

/// Resolves the preset and explicit settings into encoder settings.
fn encode_params(
    options: &Options,
    kind: ModelKind,
    config: &Config,
) -> Result<(EncodeParams, Segmenting)> {
    let preset = options.preset.map(Preset::settings);
    if preset.is_some() {
        ensure!(
            kind == ModelKind::Stereo48k,
            "the music presets are for the 48 kHz stereo model"
        );
    }
    let num_codebooks = match options.codebooks.or_else(|| preset.map(|(n, ..)| n)) {
        Some(n) => n,
        None => config.codebooks_for_bandwidth(options.bandwidth.unwrap_or(DEFAULT_BANDWIDTH))?,
    };
    let total = config.total_codebooks()?;
    ensure!(
        (1..=total).contains(&num_codebooks),
        "{num_codebooks} codebooks requested; the model has 1 to {total}"
    );
    let stride = options.stride.or_else(|| preset.map(|(_, s, ..)| s));
    let crossfade = match options.crossfade.or_else(|| preset.map(|(_, _, x, _)| x)) {
        Some(CrossfadeChoice::Edge) => Crossfade::EdgeAvoiding,
        Some(CrossfadeChoice::Triangle) | None => Crossfade::Triangle,
    };
    let beam = options
        .beam
        .or_else(|| preset.map(|(.., b)| b))
        .unwrap_or(DEFAULT_BEAM);
    ensure!(beam >= 1, "--beam must be at least 1");
    let segmenting = Segmenting { stride, crossfade };
    layout::check_segmenting(config, segmenting)?;
    Ok((
        EncodeParams {
            num_codebooks,
            segmenting,
            beam,
        },
        segmenting,
    ))
}

/// Directories to look for a data folder in: next to the executable or its
/// ancestors, the project, then the working directory.
fn data_candidates(name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().skip(1).take(4).map(|d| d.join(name)));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name));
    candidates.push(PathBuf::from(name));
    candidates
}

/// Finds the weights directory: `explicit`, `$PICOTONE_WEIGHTS`, `weights/`
/// beside the executable or its ancestors, then the project's `weights/`.
#[must_use]
pub fn weights_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Some(dir) = std::env::var_os("PICOTONE_WEIGHTS") {
        return PathBuf::from(dir);
    }
    let marker = ModelKind::Mono24k.files(Path::new("")).1;
    let found = data_candidates("weights").into_iter().find(|d| {
        d.join(hq::DECODER_FILE).is_file()
            || d.join(&marker).is_file()
            || d.join(ModelKind::Stereo48k.files(Path::new("")).1)
                .is_file()
    });
    found.unwrap_or_else(|| PathBuf::from("weights"))
}

/// Finds the cx2 tables directory: `explicit`, `$PICOTONE_TABLES`, `models/`
/// beside the executable or its ancestors, then the project's `models/`.
#[must_use]
pub fn tables_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Some(dir) = std::env::var_os("PICOTONE_TABLES") {
        return PathBuf::from(dir);
    }
    let found = data_candidates("models").into_iter().find(|d| {
        d.join(default_table(ModelKind::Stereo48k)).is_file()
            || d.join(default_table(ModelKind::Mono24k)).is_file()
    });
    found.unwrap_or_else(|| PathBuf::from("models"))
}
