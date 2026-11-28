//! `picotone`: encode audio to `.pico` files and decode them back.

mod dev;
mod hq;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use picotone::audio::{self, Audio, WavFormat};
use picotone::encode::{self, Header, Models, load_table, tables_dir, weights_dir};
use picotone::layout::{Crossfade, Trimmer};
use picotone::limiter;
use picotone::model::{Encodec, EncodedFrame, LoadOptions};
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
    /// Directory holding the model weights: `dac44_stftenc.safetensors`,
    /// `dac44_quantizer.safetensors` and `dac44_vocos.safetensors` (HQ),
    /// `encodec_24khz.{json,safetensors}` and
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
    /// HQ (the default): picotone's STFT encoder with DAC 44 kHz's quantiser, and
    /// picotone's Vocos decoder (`dac44-vocos`); 44.1 kHz, mono or stereo, 1 to 16 codebooks per channel (0.86 kbps each; default 9).
    Hq,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Preset {
    /// 48 kHz stereo, 15 codebooks (24.1 kbps before entropy coding), stride 44800, beam 8.
    Music24,
    /// 48 kHz stereo, 8 codebooks (12.9 kbps before entropy coding), stride 44800, beam 8.
    Music12,
    /// 48 kHz stereo, 4 codebooks (6.4 kbps before entropy coding), stride 44800, beam 8.
    Music6,
}

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
    /// Codebooks per frame. HQ: per channel (the mid for mid/side), 1 to 16, default 9. EnCodec:
    /// instead of the count `--bandwidth` implies.
    #[arg(long)]
    codebooks: Option<usize>,
    /// HQ mid/side: codebooks per frame of the side (default: the same as `--codebooks`).
    #[arg(long)]
    side_codebooks: Option<usize>,
    /// HQ: also write an enhancement layer, OUTPUT with the extension `.picol`, holding the
    /// codebooks from `--codebooks` up to this many (per channel; the mid for mid/side), 2 to
    /// 16 but not 10. The .pico plays alone; with the layer it decodes as this many codebooks.
    /// Over parametric stereo the layer codes mid/side.
    #[arg(long)]
    layer_codebooks: Option<usize>,
    /// HQ layer of a mid/side or parametric stereo file: the side's codebooks with the layer
    /// (default: the same as `--layer-codebooks`).
    #[arg(long)]
    layer_side_codebooks: Option<usize>,
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
    /// HQ: write the codes as a code bundle (`dev codes`'s format) at this `.npy` path instead
    /// of coding a file; OUTPUT is not written. For fitting cx2 tables.
    #[arg(long, hide = true)]
    dump_codes: Option<PathBuf>,
}

#[derive(Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent command-line flags"
)]
struct WavOptions {
    /// Write 32-bit float samples instead of 16-bit PCM.
    #[arg(long)]
    float: bool,
    /// Scale the whole output down to fit instead of running the limiter.
    #[arg(long, conflicts_with = "no_limit")]
    rescale: bool,
    /// Write the decoder's output without the limiter (for measurements; with
    /// --float, samples over full scale are kept).
    #[arg(long)]
    no_limit: bool,
    /// cx2 table to decode with, instead of the one the tables directory holds for the file.
    #[arg(long)]
    table: Option<PathBuf>,
    /// HQ: decode the .pico alone, without the `.picol` layer next to it.
    #[arg(long)]
    no_layer: bool,
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
            let mut models = Models::new(weights(), tables());
            let encoded = encode_file(&mut models, input, options)?;
            if let Some(path) = &options.dump_codes {
                eprintln!("wrote the codes to {}", path.display());
                return Ok(());
            }
            write_encoded(output, &encoded)?;
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
                    let mut stream = hq::parse(&bytes)?;
                    hq::attach_layer(&mut stream, input, !options.no_layer);
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
        } => {
            let mut models = Models::new(weights(), tables()).with_encodec_decoder();
            let encoded = encode_file(&mut models, input, encode)?;
            match &encoded.header {
                Header::Hq(_) => {
                    let mut stream = hq::parse(&encoded.bytes)?;
                    if let Some(layer) = encoded.layer.as_ref().filter(|_| !wav.no_layer) {
                        stream.attach_layer(layer)?;
                    }
                    let table =
                        hq::find_table(&tables(), wav.table.as_deref(), stream.info.table_hash)?;
                    let audio = hq::decode(&weights(), &table, stream)?;
                    write_wav(output, &audio, wav)?;
                }
                Header::Encodec(info) => {
                    let model = models
                        .encodec(info.model)
                        .context("the EnCodec model is not loaded")?;
                    let stream = Stream::parse(&encoded.bytes)?;
                    let table = find_table(&tables(), wav.table.as_deref(), &stream.info)?;
                    decode_to_wav(model, &table, &stream, output, wav)?;
                }
            }
        }
        Command::Info { input } => {
            let bytes =
                std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
            match pico::codec_of(&bytes)? {
                Codec::Dac44 => {
                    let mut stream = hq::parse(&bytes)?;
                    hq::attach_layer(&mut stream, input, true);
                    let layer_size = if stream.has_layer() {
                        usize::try_from(std::fs::metadata(hq::layer_path(input))?.len())?
                    } else {
                        0
                    };
                    hq::print_info(&stream, bytes.len(), layer_size);
                }
                Codec::Encodec(_) => print_info(&Stream::parse(&bytes)?, bytes.len())?,
            }
        }
        Command::Dev(command) => dev::run(command, &tables())?,
    }
    Ok(())
}

/// Writes an encoded file, and an HQ file's layer next to it.
fn write_encoded(output: &Path, encoded: &encode::Encoded) -> Result<()> {
    let bytes = &encoded.bytes;
    std::fs::write(output, bytes).with_context(|| format!("writing {}", output.display()))?;
    eprintln!(
        "wrote {} ({} bytes, {:.2} kbps)",
        output.display(),
        bytes.len(),
        kbps(bytes.len(), encoded.seconds)
    );
    let layer_file = hq::layer_path(output);
    if let Some(layer) = &encoded.layer {
        std::fs::write(&layer_file, layer)
            .with_context(|| format!("writing {}", layer_file.display()))?;
        eprintln!(
            "wrote {} ({} bytes, {:.2} kbps; {:.2} kbps with the base)",
            layer_file.display(),
            layer.len(),
            kbps(layer.len(), encoded.seconds),
            kbps(bytes.len() + layer.len(), encoded.seconds)
        );
    } else if matches!(encoded.header, Header::Hq(_)) && layer_file.is_file() {
        eprintln!(
            "note: {} is from another encode and no longer fits; delete it",
            layer_file.display()
        );
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

impl EncodeOptions {
    /// The library's settings for these flags.
    fn to_library(&self) -> encode::Options {
        encode::Options {
            fast: self.fast,
            model: self.model.map(|m| match m {
                ModelName::Hq => encode::Model::Hq,
                ModelName::Encodec48k => encode::Model::Encodec48k,
                ModelName::Encodec24k => encode::Model::Encodec24k,
            }),
            channel_format: match self.channel_format {
                ChannelFormat::Auto => encode::ChannelFormat::Auto,
                ChannelFormat::Mono => encode::ChannelFormat::Mono,
                ChannelFormat::Stereo => encode::ChannelFormat::Stereo,
            },
            stereo: self.stereo.map(|s| match s {
                StereoChoice::Pick => encode::StereoChoice::Pick,
                StereoChoice::Ms => encode::StereoChoice::Ms,
                StereoChoice::Lr => encode::StereoChoice::Lr,
                StereoChoice::Parametric => encode::StereoChoice::Parametric,
            }),
            bandwidth: self.bandwidth,
            preset: self.preset.map(|p| match p {
                Preset::Music24 => encode::Preset::Music24,
                Preset::Music12 => encode::Preset::Music12,
                Preset::Music6 => encode::Preset::Music6,
            }),
            codebooks: self.codebooks,
            side_codebooks: self.side_codebooks,
            layer_codebooks: self.layer_codebooks,
            layer_side_codebooks: self.layer_side_codebooks,
            ps_bands: self.ps_bands,
            ps_frames: self.ps_frames,
            stride: self.stride,
            crossfade: self.crossfade.map(|c| match c {
                CrossfadeChoice::Triangle => encode::CrossfadeChoice::Triangle,
                CrossfadeChoice::Edge => encode::CrossfadeChoice::Edge,
            }),
            beam: self.beam,
            looped: self.looped,
            tags: self.tags.clone(),
            table: self.table.clone(),
            page_length: self.page_length,
            no_length: self.no_length,
            codes_only: self.dump_codes.is_some(),
        }
    }
}

/// Reads `input` and encodes it; with `--dump-codes`, writes the HQ codes.
fn encode_file(
    models: &mut Models,
    input: &Path,
    options: &EncodeOptions,
) -> Result<encode::Encoded> {
    let source = audio::read(input)?;
    eprintln!(
        "read {}: {} Hz, {} channel(s), {:.2} s",
        input.display(),
        source.sample_rate,
        source.channels.len(),
        source.seconds()
    );
    let encoded = encode::encode(models, source, &options.to_library(), &mut |m| {
        eprintln!("{m}");
    })?;
    if let (Some(path), Header::Hq(info), Some(codes)) =
        (&options.dump_codes, &encoded.header, &encoded.codes)
    {
        dev::write_dac_bundle(info, codes, path)?;
    }
    Ok(encoded)
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
    if options.rescale || options.no_limit {
        audio::write_wav(output, out, format, options.rescale, !options.no_limit)?;
    } else {
        let limited = Audio::from_seq(&limiter::limit(&out.to_seq()?)?, out.sample_rate);
        audio::write_wav(output, &limited, format, false, true)?;
    }
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

fn kbps(bytes: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        picotone::config::to_f64(bytes) * 8.0 / seconds / 1000.0
    } else {
        0.0
    }
}
