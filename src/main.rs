//! `picotone`: encode audio to `.pico` files and decode them back.

mod dev;
mod hq;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use picotone::audio::{self, Audio, WavFormat};
use picotone::config::Config;
use picotone::layout::{self, Crossfade, Segmenting, Trimmer};
use picotone::model::{EncodeParams, Encodec, EncodedFrame, LoadOptions};
use picotone::nn::Seq;
use picotone::pico::{self, Codec, Info, Stream};
use picotone::{ModelKind, cx2};

#[derive(Debug, Parser)]
#[command(
    name = "picotone",
    version,
    about = "picotone: neural audio in .pico files (HQ 44.1 kHz by default; EnCodec with --fast)"
)]
struct Cli {
    /// Directory holding the model weights: `dac44_encoder.safetensors` and
    /// `dac44_student.safetensors` (HQ), `encodec_24khz.{json,safetensors}` and
    /// `encodec_48khz.{json,safetensors}`. Defaults to `$PICOTONE_WEIGHTS` (or the older
    /// `$ENCODEC_WEIGHTS`), then `weights/` next to the executable, then the project.
    #[arg(long, global = true)]
    weights: Option<PathBuf>,

    /// Directory holding the cx2 tables (`*.ecxm`). Defaults to `$PICOTONE_TABLES`,
    /// then `models/` next to the executable, then the project's.
    #[arg(long, global = true)]
    tables: Option<PathBuf>,

    /// Worker threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compress an audio file (WAV, FLAC, MP3, Ogg, M4A, ...) to a .pico file.
    Encode {
        /// Input audio file.
        input: PathBuf,
        /// Output .pico file.
        output: PathBuf,
        #[command(flatten)]
        options: EncodeOptions,
    },
    /// Decompress a .pico file to WAV.
    Decode {
        /// Input .pico file.
        input: PathBuf,
        /// Output .wav file.
        output: PathBuf,
        #[command(flatten)]
        options: WavOptions,
    },
    /// Encode to .pico in memory, then decode, writing the reconstructed audio as WAV.
    Roundtrip {
        /// Input audio file.
        input: PathBuf,
        /// Output .wav file.
        output: PathBuf,
        #[command(flatten)]
        encode: EncodeOptions,
        #[command(flatten)]
        wav: WavOptions,
    },
    /// Show what a .pico file contains.
    Info {
        /// Input .pico file.
        input: PathBuf,
    },
    /// Tools for building and checking cx2 tables and for tests.
    #[command(subcommand)]
    Dev(dev::DevCommand),
}

/// The models `--model` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ModelName {
    /// HQ (the default): DAC 44 kHz's encoder and picotone's distilled decoder; 44.1 kHz, mono
    /// or stereo, 1 to 9 codebooks per channel (0.86 kbps each).
    #[value(name = "dac44-student")]
    Dac44Student,
    /// `encodec_48khz`: 48 kHz stereo, bandwidths 3, 6, 12, 24 kbps.
    #[value(name = "encodec-48k")]
    Encodec48k,
    /// `encodec_24khz`: 24 kHz mono, bandwidths 1.5, 3, 6, 12, 24 kbps.
    #[value(name = "encodec-24k")]
    Encodec24k,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ChannelFormat {
    /// Follow the input. For HQ, a stereo input that is mono in all but name (its side over
    /// 40 dB under its mid) is coded as one channel.
    Auto,
    /// One channel (a down-mix).
    Mono,
    /// Two channels.
    Stereo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StereoChoice {
    /// Compare left/right and mid/side on sampled excerpts and keep the better; at 2 codebooks
    /// or fewer per channel, also parametric stereo with a mid of 2 x codebooks - 1.
    Pick,
    /// Mid/side: (L+R)/2 and (L-R)/2.
    Ms,
    /// Left/right.
    Lr,
    /// Parametric stereo: the mid (--codebooks) plus per-band level and coherence parameters
    /// (about 1 kbps); the decoder rebuilds the stereo. For low bitrates.
    #[value(alias = "ps")]
    Parametric,
}

/// What an encode runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    /// HQ with this many channels; `auto_mono` codes a mono-like stereo input as one.
    Hq { channels: usize, auto_mono: bool },
    /// An EnCodec model.
    Encodec(ModelKind),
}

/// Resolves `--model`, `--fast`, `--preset` and `--channel-format` for an
/// input with `input_channels` channels.
fn plan(options: &EncodeOptions, input_channels: usize) -> Result<Plan> {
    let format = options.channel_format;
    let stereo_input = options.preset.is_some() || input_channels >= 2;
    let model = options.model.unwrap_or_else(|| {
        if !options.fast && options.preset.is_none() {
            ModelName::Dac44Student
        } else if format == ChannelFormat::Stereo || (format == ChannelFormat::Auto && stereo_input)
        {
            ModelName::Encodec48k
        } else {
            ModelName::Encodec24k
        }
    });
    let plan = match (model, format) {
        (ModelName::Dac44Student, ChannelFormat::Mono) => Plan::Hq {
            channels: 1,
            auto_mono: false,
        },
        (ModelName::Dac44Student, ChannelFormat::Stereo) => Plan::Hq {
            channels: 2,
            auto_mono: false,
        },
        (ModelName::Dac44Student, ChannelFormat::Auto) => Plan::Hq {
            channels: if input_channels >= 2 { 2 } else { 1 },
            auto_mono: true,
        },
        (ModelName::Encodec48k, ChannelFormat::Mono) => {
            bail!("encodec-48k is a stereo model; use encodec-24k (or --fast) for mono")
        }
        (ModelName::Encodec24k, ChannelFormat::Stereo) => {
            bail!("encodec-24k is a mono model; use encodec-48k (or --fast) for stereo")
        }
        (ModelName::Encodec48k, _) => Plan::Encodec(ModelKind::Stereo48k),
        (ModelName::Encodec24k, _) => Plan::Encodec(ModelKind::Mono24k),
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
                    && options.ps_frames.is_none(),
                "--stereo, --side-codebooks, --ps-bands and --ps-frames are for the HQ model"
            );
        }
    }
    Ok(plan)
}

/// EnCodec bandwidth in kbps when neither `--bandwidth` nor `--codebooks` sets one.
const DEFAULT_BANDWIDTH: f64 = 6.0;

/// Beam width used when neither `--beam` nor a preset sets one.
const DEFAULT_BEAM: usize = 8;

/// Segment stride of the music presets: 140 latent frames.
const PRESET_STRIDE: usize = 44_800;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Preset {
    /// 48 kHz stereo, 15 codebooks (24.1 kbps before entropy coding), stride 44800, beam 8.
    Music24,
    /// 48 kHz stereo, 8 codebooks (12.9 kbps before entropy coding), stride 44800, beam 8.
    Music12,
    /// 48 kHz stereo, 4 codebooks (6.4 kbps before entropy coding), stride 44800, beam 8.
    Music6,
}

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

/// Beam width the presets use.
const PRESET_BEAM: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CrossfadeChoice {
    /// EnCodec's triangle-weighted overlap-add.
    Triangle,
    /// A linear ramp that avoids the damaged segment edges.
    Edge,
}

#[derive(Debug, Args)]
struct EncodeOptions {
    /// Use EnCodec instead of the HQ model: encodec-24k for mono, encodec-48k for stereo.
    #[arg(long)]
    fast: bool,
    /// The exact model to encode with; overrides `--fast` and `--preset`'s choice.
    #[arg(long, value_enum)]
    model: Option<ModelName>,
    /// Channels to code, separately from the model.
    #[arg(long, value_enum, default_value_t = ChannelFormat::Auto)]
    channel_format: ChannelFormat,
    /// HQ stereo layout (default: pick).
    #[arg(long, value_enum)]
    stereo: Option<StereoChoice>,
    /// EnCodec target bandwidth in kbps (sets the codebook count; default 6).
    #[arg(short, long)]
    bandwidth: Option<f64>,
    /// Recommended EnCodec 48 kHz music settings; implies encodec-48k. Explicit flags below
    /// override them.
    #[arg(long, value_enum)]
    preset: Option<Preset>,
    /// Codebooks per frame. HQ: per channel (the mid for mid/side), 1 to 9, default 9. EnCodec:
    /// instead of the count `--bandwidth` implies.
    #[arg(long)]
    codebooks: Option<usize>,
    /// HQ mid/side: codebooks per frame of the side (default: the same as `--codebooks`).
    #[arg(long)]
    side_codebooks: Option<usize>,
    /// HQ parametric stereo: parameter bands, 20 (default) or 10 (about half the rate).
    #[arg(long)]
    ps_bands: Option<usize>,
    /// HQ parametric stereo: frames per parameter set, 1, 2, 4, 8 (default, 10.8 sets/s) or 16.
    /// The rate is about inversely proportional.
    #[arg(long)]
    ps_frames: Option<usize>,
    /// Segment stride in samples (48 kHz model: 24000..=48000; EnCodec's own 47520).
    #[arg(long)]
    stride: Option<usize>,
    /// Cross-fade between segments.
    #[arg(long, value_enum)]
    crossfade: Option<CrossfadeChoice>,
    /// Beam width of the quantiser search (default 8; 1 = the reference's greedy search).
    #[arg(long)]
    beam: Option<usize>,
    /// Treat the input as a seamless loop: code it with its own end before it
    /// and its start after it, and mark the file so decoders trim it exactly
    /// and join its end to its start.
    #[arg(long = "loop")]
    looped: bool,
    /// A `KEY=value` comment to store (TITLE, ARTIST, LICENSE, ...); repeatable.
    #[arg(long = "tag", value_name = "KEY=VALUE")]
    tags: Vec<String>,
    /// cx2 table to code with, instead of the model's default in the tables directory.
    #[arg(long)]
    table: Option<PathBuf>,
    /// Segments per Ogg page (encodec-24k and HQ: frames); default from the bitrate (HQ: 80).
    #[arg(long, hide = true)]
    page_length: Option<usize>,
    /// Leave the length out of the header, as a streaming encoder would.
    #[arg(long, hide = true)]
    no_length: bool,
}

#[derive(Debug, Args)]
struct WavOptions {
    /// Write 32-bit float samples instead of 16-bit PCM.
    #[arg(long)]
    float: bool,
    /// Scale the output down to avoid clipping instead of hard-limiting it.
    #[arg(long)]
    rescale: bool,
    /// cx2 table to decode with, instead of the one the tables directory holds for the file.
    #[arg(long)]
    table: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("configuring the thread pool")?;
    }
    let weights = || weights_dir(cli.weights.as_deref());
    let tables = || tables_dir(cli.tables.as_deref());
    match &cli.command {
        Command::Encode {
            input,
            output,
            options,
        } => {
            let encoded = encode_file(&weights(), &tables(), input, options, false)?;
            let bytes = encoded.bytes();
            std::fs::write(output, bytes)
                .with_context(|| format!("writing {}", output.display()))?;
            eprintln!(
                "wrote {} ({} bytes, {:.2} kbps)",
                output.display(),
                bytes.len(),
                kbps(bytes.len(), encoded.seconds())
            );
        }
        Command::Decode {
            input,
            output,
            options,
        } => {
            let bytes =
                std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
            match pico::codec_of(&bytes)? {
                Codec::Dac44 => {
                    let stream = hq::parse(&bytes)?;
                    let table = hq::find_table(
                        &tables(),
                        options.table.as_deref(),
                        stream.info.table_hash,
                    )?;
                    let audio = hq::decode(&weights(), &table, stream)?;
                    write_wav(output, &audio, options)?;
                }
                Codec::Encodec(_) => {
                    let stream = Stream::parse(&bytes)?;
                    let table = find_table(&tables(), options.table.as_deref(), &stream.info)?;
                    // Only the decoder and the codebooks this file uses are read.
                    let parts = LoadOptions::decoder_only(Some(stream.info.num_codebooks));
                    let model = load(&weights(), stream.info.model, parts)?;
                    decode_to_wav(&model, &table, &stream, output, options)?;
                }
            }
        }
        Command::Roundtrip {
            input,
            output,
            encode,
            wav,
        } => match encode_file(&weights(), &tables(), input, encode, true)? {
            Encoded::Hq { bytes, .. } => {
                let stream = hq::parse(&bytes)?;
                let table =
                    hq::find_table(&tables(), wav.table.as_deref(), stream.info.table_hash)?;
                let audio = hq::decode(&weights(), &table, stream)?;
                write_wav(output, &audio, wav)?;
            }
            Encoded::Encodec { model, bytes, .. } => {
                let stream = Stream::parse(&bytes)?;
                let table = find_table(&tables(), wav.table.as_deref(), &stream.info)?;
                decode_to_wav(&model, &table, &stream, output, wav)?;
            }
        },
        Command::Info { input } => {
            let bytes =
                std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
            match pico::codec_of(&bytes)? {
                Codec::Dac44 => hq::print_info(&hq::parse(&bytes)?, bytes.len()),
                Codec::Encodec(_) => print_info(&Stream::parse(&bytes)?, bytes.len())?,
            }
        }
        Command::Dev(command) => dev::run(command, &tables())?,
    }
    Ok(())
}

fn print_info(stream: &Stream, size: usize) -> Result<()> {
    let info = &stream.info;
    let layout = info.layout()?;
    let rate = layout.sample_rate();
    let playable = info.playable_samples();
    let seconds = picotone::config::to_f64(playable) / picotone::config::to_f64(rate);
    eprintln!("model:      {}", info.model.name());
    eprintln!("samples:    {playable} ({seconds:.2} s at {rate} Hz)");
    eprintln!("codebooks:  {}", info.num_codebooks);
    eprintln!("cx2 table:  {:016x}", info.table_hash);
    if let Some(stride) = info.stride {
        let name = match info.crossfade {
            Crossfade::Triangle => "triangle",
            Crossfade::EdgeAvoiding => "edge-avoiding",
        };
        eprintln!("stride:     {stride} samples, {name} cross-fade");
    }
    eprintln!(
        "pages:      {} ({} damaged)",
        stream.pages.len(),
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
    if info.output_gain != 0 {
        eprintln!("gain:       {:.2} dB", f64::from(info.output_gain) / 256.0);
    }
    for tag in &info.tags {
        eprintln!("tag:        {tag}");
    }
    eprintln!("size:       {size} bytes ({:.2} kbps)", kbps(size, seconds));
    Ok(())
}

/// An encoded file.
enum Encoded {
    /// An HQ stream.
    Hq { info: pico::DacInfo, bytes: Vec<u8> },
    /// An EnCodec stream, with the model that coded it.
    Encodec {
        model: Box<Encodec>,
        info: Info,
        bytes: Vec<u8>,
    },
}

impl Encoded {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Hq { bytes, .. } | Self::Encodec { bytes, .. } => bytes,
        }
    }

    /// Playable seconds.
    fn seconds(&self) -> f64 {
        match self {
            Self::Hq { info, .. } => {
                picotone::config::to_f64(info.playable_samples())
                    / f64::from(picotone::dac::SAMPLE_RATE)
            }
            Self::Encodec { model, info, .. } => config_seconds(model, info.playable_samples()),
        }
    }
}

fn encode_hq(
    weights: &Path,
    tables: &Path,
    source: &Audio,
    options: &EncodeOptions,
    channels: usize,
    auto_mono: bool,
) -> Result<Encoded> {
    let codebooks = options.codebooks.unwrap_or(picotone::dac::CODEBOOKS);
    let table_path = options
        .table
        .clone()
        .unwrap_or_else(|| tables.join(hq::TABLE_FILE));
    let table = hq::load_table(&table_path)?;
    let settings = hq::Settings {
        channels,
        auto_mono,
        stereo: match options.stereo {
            Some(StereoChoice::Ms) => hq::StereoMode::MidSide,
            Some(StereoChoice::Lr) => hq::StereoMode::LeftRight,
            Some(StereoChoice::Parametric) => hq::StereoMode::Parametric,
            Some(StereoChoice::Pick) | None => hq::StereoMode::Pick,
        },
        parametric: picotone::pstereo::Config::new(
            options.ps_bands.unwrap_or(20),
            options
                .ps_frames
                .unwrap_or(picotone::pstereo::Config::DEFAULT.frames),
        )?,
        codebooks,
        side_codebooks: options.side_codebooks.unwrap_or(codebooks),
        looped: options.looped,
        tags: &options.tags,
        page_length: options.page_length,
        no_length: options.no_length,
        table: &table,
    };
    let (info, bytes) = hq::encode(weights, source, &settings)?;
    Ok(Encoded::Hq { info, bytes })
}

fn encode_file(
    weights: &Path,
    tables: &Path,
    input: &Path,
    options: &EncodeOptions,
    with_decoder: bool,
) -> Result<Encoded> {
    let source = audio::read(input)?;
    eprintln!(
        "read {}: {} Hz, {} channel(s), {:.2} s",
        input.display(),
        source.sample_rate,
        source.channels.len(),
        source.seconds()
    );
    for tag in &options.tags {
        ensure!(tag.contains('='), "--tag {tag} is not KEY=value");
    }
    let kind = match plan(options, source.channels.len())? {
        Plan::Encodec(kind) => kind,
        Plan::Hq {
            channels,
            auto_mono,
        } => return encode_hq(weights, tables, &source, options, channels, auto_mono),
    };
    let table_path = options
        .table
        .clone()
        .unwrap_or_else(|| tables.join(default_table(kind)));
    let table = load_table(&table_path)?;
    let config = kind.config(weights)?;
    let (params, segmenting) = encode_params(options, kind, &config)?;
    // Only the halves and the codebooks this run needs are read from disk.
    let parts = LoadOptions {
        encoder: true,
        decoder: with_decoder,
        codebooks: Some(params.num_codebooks),
    };
    let model = load(weights, kind, parts)?;
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
    eprintln!(
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
    );
    Ok(Encoded::Encodec {
        model: Box::new(model),
        info,
        bytes,
    })
}

/// Resolves the preset and explicit flags into encoder settings.
fn encode_params(
    options: &EncodeOptions,
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

fn decode_to_wav(
    model: &Encodec,
    table: &cx2::Model,
    stream: &Stream,
    output: &Path,
    options: &WavOptions,
) -> Result<()> {
    let start = Instant::now();
    let frames = stream.decode_all(table)?;
    let decoded = decode_frames(model, &frames, &stream.info)?;
    let profile = picotone::prof::report();
    if !profile.is_empty() {
        eprintln!("{profile}");
    }
    let rate = u32::try_from(model.config().sampling_rate)?;
    let out = Audio::from_seq(&decoded, rate);
    drop(decoded);
    eprintln!(
        "decoded {:.2} s with {} in {:.2} s",
        out.seconds(),
        model.name(),
        start.elapsed().as_secs_f64()
    );
    write_wav(output, &out, options)
}

fn write_wav(output: &Path, out: &Audio, options: &WavOptions) -> Result<()> {
    let format = if options.float {
        WavFormat::Float32
    } else {
        WavFormat::Pcm16
    };
    audio::write_wav(output, out, format, options.rescale)?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

/// Decodes a file's frames with its stride and cross-fade, then trims it.
fn decode_frames(model: &Encodec, frames: &[EncodedFrame], info: &Info) -> Result<Seq> {
    let cfg = model.config();
    let raw = model.decode_with(frames, info.audio_length, info.segmenting())?;
    let mut trimmer = Trimmer::new(
        info.pre_skip,
        info.playable_samples(),
        info.looped,
        cfg.sampling_rate,
        cfg.audio_channels,
    )?;
    if trimmer.is_identity() && trimmer.output_length() == raw.len() {
        return Ok(raw);
    }
    let out = trimmer.push(&raw)?;
    drop(raw);
    trimmer.finish()?;
    Ok(out)
}

fn load(weights: &Path, kind: ModelKind, parts: LoadOptions) -> Result<Encodec> {
    let start = Instant::now();
    let model = kind.load_with(weights, parts)?;
    eprintln!(
        "loaded {} in {:.2} s",
        kind.name(),
        start.elapsed().as_secs_f64()
    );
    Ok(model)
}

/// File name of the cx2 table a model codes with by default.
const fn default_table(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Mono24k => "cx2_encodec_24khz_v5.ecxm",
        ModelKind::Stereo48k => "cx2_encodec_48khz_net2_v5.ecxm",
    }
}

/// Loads a self-contained cx2 table.
fn load_table(path: &Path) -> Result<cx2::Model> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    cx2::Model::from_table(&bytes).with_context(|| format!("loading {}", path.display()))
}

/// The table a stream was coded with: `explicit`, or the file in `dir` with its hash.
fn find_table(dir: &Path, explicit: Option<&Path>, info: &Info) -> Result<cx2::Model> {
    let table = if let Some(path) = explicit {
        load_table(path)?
    } else {
        let mut found = None;
        let entries =
            std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))?;
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "ecxm") {
                let bytes = std::fs::read(&path)?;
                if cx2::model::self_contained_hash(&bytes) == Some(info.table_hash) {
                    found = Some(cx2::Model::from_table(&bytes)?);
                    break;
                }
            }
        }
        found.with_context(|| {
            format!(
                "no cx2 table with hash {:016x} in {} (pass --tables or --table)",
                info.table_hash,
                dir.display()
            )
        })?
    };
    ensure!(
        table.hash() == info.table_hash,
        "the file was coded with cx2 table {:016x}, not {}",
        info.table_hash,
        table.hash_hex()
    );
    Ok(table)
}

fn config_seconds(model: &Encodec, samples: usize) -> f64 {
    picotone::config::to_f64(samples) / picotone::config::to_f64(model.config().sampling_rate)
}

fn kbps(bytes: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        picotone::config::to_f64(bytes) * 8.0 / seconds / 1000.0
    } else {
        0.0
    }
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

/// Finds the weights directory: `--weights`, `$PICOTONE_WEIGHTS`, the older
/// `$ENCODEC_WEIGHTS`, `weights/` beside the executable or its ancestors, then the project's `weights/`.
fn weights_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Some(dir) =
        std::env::var_os("PICOTONE_WEIGHTS").or_else(|| std::env::var_os("ENCODEC_WEIGHTS"))
    {
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

/// Finds the cx2 tables directory: `--tables`, `$PICOTONE_TABLES`, `models/`
/// beside the executable or its ancestors, then the project's `models/`.
fn tables_dir(explicit: Option<&Path>) -> PathBuf {
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
