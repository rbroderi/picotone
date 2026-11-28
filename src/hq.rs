//! The HQ codec's decoding and file information in the command line; the
//! encoder is the library's [`picotone::encode::hq`].

#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use picotone::audio::Audio;
use picotone::cx2::hq::HqModel;
use picotone::dac;
use picotone::declick;
use picotone::encode::hq::{DECODER_NAME, join_books, load_decoder, load_table};
use picotone::hqstream::HqStream;
use picotone::pico::{ChannelLayout, DacStream};

/// The table a stream was coded with: `explicit`, or the HQ table in `dir` with its hash.
pub(crate) fn find_table(dir: &Path, explicit: Option<&Path>, hash: u64) -> Result<HqModel> {
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
                if picotone::cx2::hq::table_hash(&bytes) == Some(hash) {
                    found = Some(HqModel::from_table(&bytes)?);
                    break;
                }
            }
        }
        found.with_context(|| {
            format!(
                "no HQ cx2 table with hash {hash:016x} in {} (pass --tables or --table)",
                dir.display()
            )
        })?
    };
    ensure!(
        table.hash() == hash,
        "the file was coded with HQ cx2 table {hash:016x}, not {:016x}",
        table.hash()
    );
    Ok(table)
}

/// The layer file of an HQ file: the same path with the extension `.picol`.
pub(crate) fn layer_path(path: &Path) -> PathBuf {
    path.with_extension("picol")
}

/// Attaches the layer next to `input` to `stream` when there is one and
/// `use_layer`; a layer that does not fit is reported and left out.
pub(crate) fn attach_layer(stream: &mut DacStream, input: &Path, use_layer: bool) {
    let path = layer_path(input);
    if !use_layer || path == input || !path.is_file() {
        return;
    }
    match std::fs::read(&path)
        .with_context(|| format!("reading {}", path.display()))
        .and_then(|bytes| stream.attach_layer(&bytes))
    {
        Ok(()) => eprintln!(
            "layer {}: {} codebooks over the base's {}",
            path.display(),
            join_books(&stream.info.codebooks),
            join_books(stream.base_codebooks())
        ),
        Err(error) => eprintln!(
            "ignoring the layer {} ({error:#}); playing the base alone",
            path.display()
        ),
    }
}

/// Decodes an HQ stream to audio, trimmed for playback, through the same
/// [`HqStream`] as the WebAssembly decoder.
pub(crate) fn decode(weights: &Path, table: &HqModel, stream: DacStream) -> Result<Audio> {
    let decoder = load_decoder(weights, &mut |m| eprintln!("{m}"))?;
    let start = Instant::now();
    let tier = stream.info.codebooks.iter().copied().max().unwrap_or(0);
    let pre_skip = stream.info.pre_skip;
    let mut stream = HqStream::new(stream, table, &decoder, 0)?;
    stream.fetch(usize::MAX)?;
    let cx2_seconds = start.elapsed().as_secs_f64();
    let mut out = stream.decode_rest()?;
    if declick::applies(tier) {
        out = declick::declick(&out, pre_skip, pre_skip)?;
    }
    let audio = Audio::from_seq(&out, dac::SAMPLE_RATE);
    eprintln!(
        "decoded {:.2} s with {DECODER_NAME} in {:.2} s (cx2 {cx2_seconds:.2} s)",
        audio.seconds(),
        start.elapsed().as_secs_f64()
    );
    Ok(audio)
}

/// Reads an HQ stream.
pub(crate) fn parse(bytes: &[u8]) -> Result<DacStream> {
    DacStream::parse(bytes).context("reading the HQ stream")
}

/// Prints what an HQ stream holds; `size` is the base's bytes, `layer_size`
/// the layer's when one is attached.
pub(crate) fn print_info(stream: &DacStream, size: usize, layer_size: usize) {
    let info = &stream.info;
    let playable = info.playable_samples();
    let seconds = picotone::config::to_f64(playable) / f64::from(dac::SAMPLE_RATE);
    eprintln!("model:      DAC 44 kHz codes (HQ; decoder {DECODER_NAME})");
    eprintln!(
        "samples:    {playable} ({seconds:.2} s at {} Hz)",
        dac::SAMPLE_RATE
    );
    match stream.base_layout() {
        ChannelLayout::Parametric(config) => {
            eprintln!("channels:   parametric stereo ({})", config.describe());
        }
        layout => eprintln!("channels:   {}", layout.name()),
    }
    eprintln!("codebooks:  {}", join_books(stream.base_codebooks()));
    if stream.has_layer() {
        eprintln!(
            "layer:      {}, {} codebooks ({} bytes of codes)",
            info.layout.name(),
            join_books(&info.codebooks),
            stream.layer_coded_bytes().iter().sum::<usize>()
        );
    }
    eprintln!("code hash:  {:016x}", info.codebook_hash);
    eprintln!("cx2 table:  {:016x}", info.table_hash);
    let frames = stream
        .spans
        .iter()
        .zip(&stream.pages)
        .filter(|(_, p)| p.is_some())
        .map(|(s, _)| s.len())
        .sum::<usize>();
    for (c, (bytes, &n)) in stream
        .coded_bytes()
        .iter()
        .zip(stream.base_codebooks())
        .enumerate()
    {
        let codes = frames * n;
        if codes > 0 {
            let bits = picotone::config::to_f64(bytes * 8) / picotone::config::to_f64(codes);
            eprintln!(
                "channel {c}:  {bytes} bytes of codes, {bits:.3} bits per code ({:.1}% under 10)",
                100.0 * (1.0 - bits / 10.0)
            );
        }
    }
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
    if stream.has_layer() {
        let total = size + layer_size;
        let kbps = |bytes: usize| {
            if seconds > 0.0 {
                picotone::config::to_f64(bytes) * 8.0 / seconds / 1000.0
            } else {
                0.0
            }
        };
        eprintln!(
            "layer size: {layer_size} bytes ({:.2} kbps); with the base {total} bytes ({:.2} kbps)",
            kbps(layer_size),
            kbps(total)
        );
    }
}
