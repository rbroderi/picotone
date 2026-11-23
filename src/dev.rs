//! `picotone dev`: tools for building and checking cx2 tables, and for tests.
//!
//! The table tools read code bundles: a `.pico` file, or a `NumPy` `.npy` array
//! of codes `[frames, codebooks]` with an optional `.json` sidecar of the same
//! stem that gives the layout (written by `picotone dev codes` and by
//! `tools/codes_from_ecdc.py`). Statistics and tuning need only the codes;
//! `check` needs the layout too. A file argument starting with `@` names a
//! text file listing one path per line.

#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail, ensure};
use clap::Subcommand;
use picotone::cx2::features::Centroids;
use picotone::cx2::kernels::Simd;
use picotone::cx2::model::{EXPERT_NAMES, self_contained_hash};
use picotone::cx2::{self, Model, fit, to_f64};
use picotone::layout::Crossfade;
use picotone::model::EncodedFrame;
use picotone::pico::{self, Info, Layout, Stream};
use picotone::{ModelKind, scale};
use serde::{Deserialize, Serialize};

#[derive(Debug, Subcommand)]
pub(crate) enum DevCommand {
    /// Write a .pico file's codes as .npy (`[frames, codebooks]`, u16) plus a .json
    /// sidecar with the layout and the segment scales.
    Codes {
        /// Input .pico file.
        input: PathBuf,
        /// Output .npy file; the sidecar goes next to it with the extension .json.
        output: PathBuf,
    },
    /// Write a code bundle (.npy with its .json sidecar) as a .pico file.
    Pack {
        /// cx2 table (.ecxm).
        table: PathBuf,
        /// Input code bundle.
        input: String,
        /// Output .pico file.
        output: PathBuf,
        /// Codec weights for a version 4 table (`-`: none, the table is self-contained).
        #[arg(long, default_value = "-")]
        codec: String,
    },
    /// Code bundles with a table and check the round trip, seeking, a stream without
    /// a length field and the baseline code path. Prints one tab-separated line per
    /// file: name, 10-bit packed bytes, .pico bytes, codes, pack s, unpack s, payload bytes.
    Check {
        /// cx2 table (.ecxm).
        table: PathBuf,
        /// Segments per page (24 kHz model: frames), or `auto` for the default.
        page_length: String,
        /// Code bundles (.pico, or .npy with a .json sidecar).
        #[arg(required = true)]
        files: Vec<String>,
        /// Codec weights for a version 4 table (`-`: none, the table is self-contained).
        #[arg(long, default_value = "-")]
        codec: String,
        /// Only the round trip, for bulk size runs.
        #[arg(long)]
        quick: bool,
        /// Files checked in parallel.
        #[arg(long, default_value_t = 8)]
        jobs: usize,
    },
    /// Count the statistics the table builder fits (JSON).
    Stats {
        /// Codec weights (`encodec_*.safetensors`), for the centroids.
        codec: PathBuf,
        /// Output .json file.
        output: PathBuf,
        /// Codebooks that get the radial (geo) expert.
        geo: usize,
        /// Frames between predictor resets (0: never).
        reset: usize,
        /// Code bundles.
        #[arg(required = true)]
        files: Vec<String>,
    },
    /// Tune the expert weights of a table on sampled frames (JSON).
    Tune {
        /// cx2 table (.ecxm).
        table: PathBuf,
        /// Codec weights for a version 4 table, or `-`.
        codec: String,
        /// Output .json file.
        output: PathBuf,
        /// Frames sampled per codebook.
        frames: usize,
        /// Frames between predictor resets (0: never).
        reset: usize,
        /// Code bundles.
        #[arg(required = true)]
        files: Vec<String>,
    },
    /// Time the HQ decoder (`dac44-student`) on random codes; prints the real-time factor.
    DacBench {
        /// The decoder weights (`dac44_student.safetensors`).
        weights: PathBuf,
        /// Seconds of audio to decode, per channel.
        #[arg(long, default_value_t = 30.0)]
        seconds: f64,
        /// Codebooks per frame (the tier).
        #[arg(long, default_value_t = 9)]
        codebooks: usize,
        /// Channels to decode one after the other (2 = stereo).
        #[arg(long, default_value_t = 2)]
        channels: usize,
    },
    /// Decode a `.npy` of DAC codes (`[frames, 9]`) with the HQ decoder to raw
    /// little-endian f32 samples, for parity checks against PyTorch.
    DacDecode {
        /// The decoder weights (`dac44_student.safetensors`).
        weights: PathBuf,
        /// Codes, `[frames, codebooks]` u16.
        codes: PathBuf,
        /// Output: raw f32 samples.
        output: PathBuf,
        /// Use the first N codebooks (the tier).
        #[arg(long, default_value_t = 9)]
        codebooks: usize,
        /// Latent frames per chunk (default: the decoder's own).
        #[arg(long)]
        chunk: Option<usize>,
    },
    /// Time each expert and the whole coder on one code bundle.
    Profile {
        /// cx2 table (.ecxm).
        table: PathBuf,
        /// Code bundle.
        input: String,
        /// Codec weights for a version 4 table, or `-`.
        #[arg(long, default_value = "-")]
        codec: String,
    },
}

/// Runs a dev command; `tables` is where `.pico` inputs find their table.
pub(crate) fn run(command: &DevCommand, tables: &Path) -> Result<()> {
    match command {
        DevCommand::Codes { input, output } => {
            let bundle = read_pico(tables, &input.to_string_lossy())?;
            write_npy(output, &bundle.codes, bundle.num_codebooks)?;
            let sidecar = bundle.sidecar.context("a .pico file always has a layout")?;
            let path = output.with_extension("json");
            std::fs::write(&path, serde_json::to_vec_pretty(&sidecar)?)
                .with_context(|| format!("writing {}", path.display()))
        }
        DevCommand::Pack {
            table,
            input,
            output,
            codec,
        } => {
            let model = load(table, codec)?;
            let (info, frames) = stream_of(&read_bundle(tables, input)?, &model, None)?;
            std::fs::write(output, pico::write(&model, &info, &frames)?)
                .with_context(|| format!("writing {}", output.display()))
        }
        DevCommand::Check {
            table,
            page_length,
            files,
            codec,
            quick,
            jobs,
        } => {
            let model = load(table, codec)?;
            let mut baseline = load(table, codec)?;
            baseline.set_simd(Simd::baseline());
            let page_length = if page_length == "auto" {
                None
            } else {
                Some(
                    page_length
                        .parse()
                        .with_context(|| format!("parsing the page length {page_length}"))?,
                )
            };
            let files = expand(files)?;
            check_all(
                &model,
                (!quick).then_some(&baseline),
                page_length,
                tables,
                &files,
                *jobs,
            )
        }
        DevCommand::Stats {
            codec,
            output,
            geo,
            reset,
            files,
        } => {
            let (num_codebooks, streams) = streams(tables, files)?;
            let centroids = Centroids::from_safetensors(&read(codec)?, num_codebooks)?;
            write_json(
                output,
                &fit::histograms(&centroids, num_codebooks, *geo, *reset, &streams)?,
            )
        }
        DevCommand::Tune {
            table,
            codec,
            output,
            frames,
            reset,
            files,
        } => {
            let model = load(table, codec)?;
            let (num_codebooks, streams) = streams(tables, files)?;
            write_json(
                output,
                &fit::tune(&model, num_codebooks, &streams, *frames, *reset)?,
            )
        }
        DevCommand::Profile {
            table,
            input,
            codec,
        } => profile(&load(table, codec)?, &read_bundle(tables, input)?),
        DevCommand::DacBench {
            weights,
            seconds,
            codebooks,
            channels,
        } => dac_bench(weights, *seconds, *codebooks, *channels),
        DevCommand::DacDecode {
            weights,
            codes,
            output,
            codebooks,
            chunk,
        } => dac_decode(weights, codes, output, *codebooks, *chunk),
    }
}

/// Decodes a code array with the HQ decoder to raw f32 samples.
fn dac_decode(
    weights: &Path,
    codes: &Path,
    output: &Path,
    codebooks: usize,
    chunk: Option<usize>,
) -> Result<()> {
    let decoder = picotone::dac::DacDecoder::load(weights)?;
    let (all, columns) = read_npy(codes)?;
    ensure!(
        (1..=columns).contains(&codebooks),
        "{} has {columns} codebooks, {codebooks} asked for",
        codes.display()
    );
    let tier: Vec<u16> = all
        .chunks_exact(columns)
        .flat_map(|frame| frame[..codebooks].iter().copied())
        .collect();
    let t = Instant::now();
    let y = match chunk {
        Some(n) => decoder.decode_chunked(&tier, codebooks, n)?,
        None => decoder.decode(&tier, codebooks)?,
    };
    eprintln!(
        "decoded {} samples in {:.2} s",
        y.len(),
        t.elapsed().as_secs_f64()
    );
    let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(output, bytes).with_context(|| format!("writing {}", output.display()))
}

/// Decodes random codes with the HQ decoder and reports the real-time factor.
fn dac_bench(weights: &Path, seconds: f64, codebooks: usize, channels: usize) -> Result<()> {
    use picotone::dac::{CODEBOOK_SIZE, DacDecoder, HOP, SAMPLE_RATE};
    let start = Instant::now();
    let decoder = DacDecoder::load(weights)?;
    eprintln!(
        "loaded {} in {:.2} s",
        weights.display(),
        start.elapsed().as_secs_f64()
    );
    let frames = picotone::config::floor_to_usize(seconds * f64::from(SAMPLE_RATE) / to_f64(HOP))?;
    // A fixed linear congruential sequence: the same codes every run.
    let mut state = 0x2545_f491_u32;
    let mut codes = vec![0_u16; frames * codebooks];
    for code in &mut codes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *code = u16::try_from(usize::try_from(state >> 16)? % CODEBOOK_SIZE)?;
    }
    let audio = to_f64(frames * HOP) / f64::from(SAMPLE_RATE);
    let mut best = f64::INFINITY;
    for run in 0..3 {
        let t = Instant::now();
        for _ in 0..channels {
            let y = decoder.decode(&codes, codebooks)?;
            ensure!(y.len() == frames * HOP, "decoded length is wrong");
        }
        let took = t.elapsed().as_secs_f64();
        best = best.min(took);
        eprintln!(
            "run {run}: {audio:.1} s x {channels} channel(s) at q{codebooks} in {took:.2} s ({:.2}x real time)",
            audio / took
        );
    }
    eprintln!("best: {:.2}x real time", audio / best);
    Ok(())
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

/// Loads a table; a version 4 table needs the codec's weights.
fn load(table: &Path, weights: &str) -> Result<Model> {
    if weights == "-" {
        Model::from_table(&read(table)?)
    } else {
        Model::load(&read(table)?, &read(Path::new(weights))?)
    }
    .with_context(|| format!("loading the cx2 table {}", table.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    std::fs::write(path, serde_json::to_vec(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// Expands `@list` arguments into the paths they list.
fn expand(files: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for file in files {
        if let Some(list) = file.strip_prefix('@') {
            let text = std::fs::read_to_string(list).with_context(|| format!("reading {list}"))?;
            out.extend(
                text.lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(str::to_owned),
            );
        } else {
            out.push(file.clone());
        }
    }
    ensure!(!out.is_empty(), "no input files");
    Ok(out)
}

/// The layout of a code bundle, as the `.json` sidecar stores it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sidecar {
    /// `encodec_24khz` or `encodec_48khz`.
    model: String,
    codebooks: usize,
    /// Coded samples per channel, the trims included.
    audio_length: usize,
    /// Segment stride (48 kHz model).
    #[serde(default)]
    stride: Option<usize>,
    /// 0 triangle, 1 edge-avoiding.
    #[serde(default)]
    crossfade: u8,
    #[serde(default)]
    pre_skip: usize,
    #[serde(default)]
    end_trim: usize,
    #[serde(default)]
    looped: bool,
    /// Per-segment volume scales (48 kHz model).
    #[serde(default)]
    scales: Option<Vec<f32>>,
    /// Latent frames per segment.
    #[serde(default)]
    segment_frames: Vec<usize>,
}

/// Codes of one input, frame-major, and its layout when known.
#[derive(Debug)]
struct Bundle {
    name: String,
    num_codebooks: usize,
    codes: Vec<u16>,
    sidecar: Option<Sidecar>,
}

/// Reads the table with `hash` from `dir`.
fn table_for(dir: &Path, hash: u64) -> Result<Model> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "ecxm") {
            let bytes = read(&path)?;
            if self_contained_hash(&bytes) == Some(hash) {
                return Model::from_table(&bytes);
            }
        }
    }
    bail!("no cx2 table with hash {hash:016x} in {}", dir.display())
}

fn read_pico(tables: &Path, path: &str) -> Result<Bundle> {
    let bytes = read(Path::new(path))?;
    let stream = Stream::parse(&bytes).with_context(|| format!("parsing {path}"))?;
    let table = table_for(tables, stream.info.table_hash)?;
    let frames = stream.decode_all(&table)?;
    let info = &stream.info;
    let scales: Vec<f32> = frames.iter().filter_map(|f| f.scale).collect();
    Ok(Bundle {
        name: path.to_owned(),
        num_codebooks: info.num_codebooks,
        codes: frames
            .iter()
            .flat_map(|f| f.codes.iter().copied())
            .collect(),
        sidecar: Some(Sidecar {
            model: info.model.name().to_owned(),
            codebooks: info.num_codebooks,
            audio_length: info.audio_length,
            stride: info.stride,
            crossfade: info.crossfade.code(),
            pre_skip: info.pre_skip,
            end_trim: info.end_trim,
            looped: info.looped,
            scales: (!scales.is_empty()).then_some(scales),
            segment_frames: stream.segment_frames.clone(),
        }),
    })
}

fn read_bundle(tables: &Path, path: &str) -> Result<Bundle> {
    if Path::new(path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("pico"))
    {
        return read_pico(tables, path);
    }
    let (codes, num_codebooks) = read_npy(Path::new(path))?;
    let json = Path::new(path).with_extension("json");
    let sidecar = if json.is_file() {
        let sidecar: Sidecar = serde_json::from_slice(&read(&json)?)
            .with_context(|| format!("parsing {}", json.display()))?;
        ensure!(
            sidecar.codebooks == num_codebooks,
            "{} gives {} codebooks, the codes have {num_codebooks}",
            json.display(),
            sidecar.codebooks
        );
        Some(sidecar)
    } else {
        None
    };
    Ok(Bundle {
        name: path.to_owned(),
        num_codebooks,
        codes,
        sidecar,
    })
}

/// Reads code bundles as code streams that share one codebook count.
fn streams(tables: &Path, files: &[String]) -> Result<(usize, Vec<Vec<u16>>)> {
    let mut num_codebooks = 0;
    let mut out = Vec::new();
    for file in expand(files)? {
        let bundle = read_bundle(tables, &file)?;
        if out.is_empty() {
            num_codebooks = bundle.num_codebooks;
        }
        ensure!(
            bundle.num_codebooks == num_codebooks,
            "{file} has a different codebook count"
        );
        out.push(bundle.codes);
    }
    Ok((num_codebooks, out))
}

/// The stream a bundle describes, ready to write with `table`.
fn stream_of(
    bundle: &Bundle,
    table: &Model,
    page_length: Option<usize>,
) -> Result<(Info, Vec<EncodedFrame>)> {
    let sidecar = bundle
        .sidecar
        .as_ref()
        .with_context(|| format!("{} has no .json sidecar with its layout", bundle.name))?;
    let model = ModelKind::from_name(&sidecar.model)?;
    let layout = Layout::new(model, sidecar.stride)?;
    let nc = bundle.num_codebooks;
    let counts = layout.segment_frames(sidecar.audio_length);
    ensure!(
        counts.iter().sum::<usize>() * nc == bundle.codes.len(),
        "{}: the layout does not match the number of codes",
        bundle.name
    );
    let mut frames = Vec::with_capacity(counts.len());
    let mut pos = 0;
    for (index, &count) in counts.iter().enumerate() {
        let scale = if layout.normalizes() {
            let level = sidecar
                .scales
                .as_ref()
                .and_then(|s| s.get(index))
                .copied()
                .unwrap_or(1.0);
            // Scales from other sources are moved onto the 12-bit grid.
            Some(scale::dequantize(scale::quantize(level)))
        } else {
            None
        };
        frames.push(EncodedFrame {
            scale,
            num_codebooks: nc,
            codes: bundle.codes[pos..pos + count * nc].to_vec(),
        });
        pos += count * nc;
    }
    let info = Info {
        model,
        table_hash: table.hash(),
        num_codebooks: nc,
        stride: sidecar.stride,
        crossfade: Crossfade::from_code(sidecar.crossfade)?,
        looped: sidecar.looped,
        pre_skip: sidecar.pre_skip,
        page_length: page_length.unwrap_or_else(|| pico::default_page_length(model, nc)),
        audio_length: sidecar.audio_length,
        end_trim: sidecar.end_trim,
        length_in_header: true,
        output_gain: 0,
        tags: Vec::new(),
    };
    Ok((info, frames))
}

fn same_frames(a: &[EncodedFrame], b: &[EncodedFrame]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.codes == y.codes && x.scale.map(f32::to_bits) == y.scale.map(f32::to_bits)
        })
}

fn check(
    model: &Model,
    baseline: Option<&Model>,
    page_length: Option<usize>,
    tables: &Path,
    path: &str,
) -> Result<String> {
    let bundle = read_bundle(tables, path)?;
    let (info, frames) = stream_of(&bundle, model, page_length)?;
    let start = Instant::now();
    let packed = pico::write(model, &info, &frames)?;
    let pack_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let stream = Stream::parse(&packed)?;
    let restored = stream.decode_all(model)?;
    let unpack_seconds = start.elapsed().as_secs_f64();
    ensure!(
        same_frames(&restored, &frames) && stream.info == info,
        "round trip of {path} is not lossless"
    );
    let payload: usize = stream.pages.iter().flatten().map(Vec::len).sum();
    if let Some(baseline) = baseline {
        // Seeking: the later pages decoded alone must equal that span of the whole.
        let first = stream.pages.len() / 2;
        let mut tail = Vec::new();
        for index in first..stream.pages.len() {
            tail.extend(stream.decode_page(model, index)?);
        }
        let codes: Vec<u16> = tail.iter().flat_map(|f| f.codes.iter().copied()).collect();
        ensure!(
            bundle.codes.ends_with(&codes),
            "seek decode of {path} differs from the full decode"
        );
        ensure!(
            pico::write(baseline, &info, &frames)? == packed,
            "dispatched and baseline code paths disagree on {path}"
        );
        // A stream without the length field must carry the same audio.
        let streaming = Info {
            length_in_header: false,
            ..info
        };
        let other = Stream::parse(&pico::write(model, &streaming, &frames)?)?;
        ensure!(
            other.info == streaming && same_frames(&other.decode_all(model)?, &frames),
            "the stream of {path} without a length field does not round trip"
        );
    }
    let name = Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);
    let packed_bits = (bundle.codes.len() * 10).div_ceil(8)
        + frames.iter().filter(|f| f.scale.is_some()).count() * 4;
    Ok(format!(
        "{name}\t{packed_bits}\t{}\t{}\t{pack_seconds:.4}\t{unpack_seconds:.4}\t{payload}",
        packed.len(),
        bundle.codes.len()
    ))
}

/// Checks every file, on up to `threads` threads, and prints the result lines in order.
fn check_all(
    model: &Model,
    baseline: Option<&Model>,
    page_length: Option<usize>,
    tables: &Path,
    files: &[String],
    threads: usize,
) -> Result<()> {
    let chunk = files.len().div_ceil(threads.max(1)).max(1);
    let lines: Vec<Result<String>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for part in files.chunks(chunk) {
            handles.push(scope.spawn(move || {
                part.iter()
                    .map(|f| check(model, baseline, page_length, tables, f))
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| vec![Err(anyhow::anyhow!("a checking thread panicked"))])
            })
            .collect()
    });
    for line in lines {
        eprintln!("{}", line?);
    }
    Ok(())
}

/// Prints the mean time per code of each expert and of the other coding steps.
fn profile(model: &Model, bundle: &Bundle) -> Result<()> {
    let codes = &bundle.codes;
    let nc = bundle.num_codebooks;
    let micros = |spent: f64| spent * 1e6 / to_f64(codes.len().max(1));
    let mut scratch = vec![0_i32; 1024];
    for (index, name) in EXPERT_NAMES.iter().enumerate() {
        let mut predictor = model.predictor(nc)?;
        let mut spent = 0.0;
        for &code in codes {
            let start = Instant::now();
            predictor.add_expert(index, &mut scratch);
            spent += start.elapsed().as_secs_f64();
            predictor.update(code)?;
        }
        eprintln!("{name:<8} {:7.2} us/code", micros(spent));
    }
    let start = Instant::now();
    let chunk = cx2::encode_chunk(model, nc, &[], codes)?;
    let pack_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (_, decoded) = cx2::decode_chunk(model, nc, 0, codes.len(), &chunk)?;
    let unpack_seconds = start.elapsed().as_secs_f64();
    ensure!(&decoded == codes, "profile round trip is not lossless");
    eprintln!(
        "encode   {:7.2} us/code, decode {:7.2} us/code, {:.3} bits/code",
        micros(pack_seconds),
        micros(unpack_seconds),
        to_f64(chunk.len() * 8) / to_f64(codes.len().max(1))
    );
    Ok(())
}

/// Reads a 2-D integer `.npy` array; returns it row-major as u16 and its column count.
fn read_npy(path: &Path) -> Result<(Vec<u16>, usize)> {
    let bytes = read(path)?;
    ensure!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "{} is not a .npy file",
        path.display()
    );
    let (header_len, start) = if bytes[6] == 1 {
        (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10)
    } else {
        let b = bytes.get(8..12).context("truncated .npy header")?;
        (
            usize::try_from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))?,
            12,
        )
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + header_len)
            .context("truncated .npy header")?,
    )?;
    let field = |key: &str| -> Result<&str> {
        let at = header
            .find(&format!("'{key}'"))
            .with_context(|| format!("{} has no {key}", path.display()))?;
        Ok(header[at + key.len() + 2..].trim_start_matches([':', ' ']))
    };
    ensure!(
        field("fortran_order")?.starts_with("False"),
        "{} is in Fortran order",
        path.display()
    );
    let descr = field("descr")?;
    let descr = descr
        .get(1..descr[1..].find('\'').map_or(0, |e| e + 1))
        .unwrap_or("");
    let shape = field("shape")?;
    let dims: Vec<usize> = shape[1..shape.find(')').context("bad .npy shape")?]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    ensure!(dims.len() == 2, "{} is not a 2-D array", path.display());
    let data = &bytes[start + header_len..];
    let count = dims[0] * dims[1];
    let width = match descr {
        "|u1" | "u1" => 1,
        "<u2" | "<i2" => 2,
        "<u4" | "<i4" => 4,
        "<u8" | "<i8" => 8,
        other => bail!("{}: unsupported .npy type {other}", path.display()),
    };
    ensure!(
        data.len() >= count * width,
        "{} is truncated",
        path.display()
    );
    let codes = data[..count * width]
        .chunks_exact(width)
        .map(|b| {
            let mut wide = [0_u8; 8];
            wide[..width].copy_from_slice(b);
            u16::try_from(u64::from_le_bytes(wide)).context("code out of range")
        })
        .collect::<Result<_>>()?;
    Ok((codes, dims[1]))
}

/// Writes codes as a `.npy` array `[frames, columns]` of little-endian u16.
fn write_npy(path: &Path, codes: &[u16], columns: usize) -> Result<()> {
    let rows = codes.len() / columns.max(1);
    let mut header =
        format!("{{'descr': '<u2', 'fortran_order': False, 'shape': ({rows}, {columns}), }}");
    // The header, with its 10-byte preamble, is padded to a multiple of 64 and ends in '\n'.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&u16::try_from(header.len())?.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &code in codes {
        out.extend_from_slice(&code.to_le_bytes());
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}
