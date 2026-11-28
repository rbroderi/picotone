//! The C encoder against the command line: `pt_encoder_*` must give the same
//! bytes as `picotone encode` of the same WAV with the same options, and the
//! file must decode through `pt_open` and `pt_decode`.
//!
//! Needs the weights and tables (`weights/`, `models/`, or `$PICOTONE_WEIGHTS`
//! and `$PICOTONE_TABLES`) and the command-line program: `$PICOTONE_EXE`, or
//! the project's `target/release/picotone` (`cargo build --release` at the
//! root first).

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, Result, bail, ensure};
use picotone::audio::{self, Audio, WavFormat};
use picotone::encode::{tables_dir, weights_dir};
use picotone_ffi::{
    PtHeader, pt_close, pt_decode, pt_encoder_finish, pt_encoder_free, pt_encoder_new,
    pt_encoder_push, pt_encoder_set, pt_encoder_tag, pt_header, pt_init, pt_last_error, pt_open,
};

fn error() -> String {
    // SAFETY: pt_last_error never returns null, and the text is NUL-terminated.
    unsafe { CStr::from_ptr(pt_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The command-line program.
fn cli() -> Result<PathBuf> {
    let path = std::env::var_os("PICOTONE_EXE").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/release/picotone")
                .with_extension(std::env::consts::EXE_EXTENSION)
        },
        PathBuf::from,
    );
    ensure!(
        path.is_file(),
        "{} not found: run cargo build --release at the project root, or set PICOTONE_EXE",
        path.display()
    );
    Ok(path)
}

/// A deterministic test signal: tones and noise, different in each channel.
#[expect(clippy::cast_precision_loss, reason = "small sample indices")]
#[expect(clippy::suboptimal_flops, reason = "any fixed signal will do")]
fn signal(rate: u32, channels: usize, seconds: u32) -> Audio {
    let n = usize::try_from(seconds * rate).unwrap_or(0);
    let mut seed = 0x1234_5678_u32;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let channels = (0..channels)
        .map(|c| {
            (0..n)
                .map(|i| {
                    let t = i as f32 / rate as f32;
                    let f = 220.0 * (c + 1) as f32;
                    0.3 * (std::f32::consts::TAU * f * t).sin()
                        + 0.1 * (std::f32::consts::TAU * 3.0 * f * t).sin()
                        + 0.05 * noise()
                })
                .collect()
        })
        .collect();
    Audio {
        sample_rate: rate,
        channels,
    }
}

/// Encodes `audio` through the C interface with `options` (`key`, `value`)
/// and `tags`, pushing it in uneven blocks.
fn encode_c(audio: &Audio, options: &[(&str, &str)], tags: &[&str]) -> Result<(Vec<u8>, Vec<u8>)> {
    let width = audio.channels.len();
    let enc = pt_encoder_new(audio.sample_rate, u32::try_from(width)?);
    ensure!(!enc.is_null(), "pt_encoder_new: {}", error());
    let result = (|| {
        for (key, value) in options {
            let (k, v) = (CString::new(*key)?, CString::new(*value)?);
            // SAFETY: the encoder is live and the strings NUL-terminated.
            let got = unsafe { pt_encoder_set(enc, k.as_ptr(), v.as_ptr()) };
            ensure!(got == 0, "pt_encoder_set {key} {value}: {}", error());
        }
        for tag in tags {
            let t = CString::new(*tag)?;
            // SAFETY: the encoder is live and the string NUL-terminated.
            let got = unsafe { pt_encoder_tag(enc, t.as_ptr()) };
            ensure!(got == 0, "pt_encoder_tag {tag}: {}", error());
        }
        let interleaved = audio.to_seq()?.data().to_vec();
        let mut start = 0;
        for block in [1, 4_000, 100_000].into_iter().cycle() {
            let frames = block.min(audio.len() - start);
            let data = &interleaved[start * width..(start + frames) * width];
            // SAFETY: the encoder is live and `data` holds `frames` frames.
            let got = unsafe { pt_encoder_push(enc, data.as_ptr(), frames) };
            ensure!(got == 0, "pt_encoder_push: {}", error());
            start += frames;
            if start == audio.len() {
                break;
            }
        }
        let (mut pico, mut pico_len) = (std::ptr::null(), 0);
        let (mut layer, mut layer_len) = (std::ptr::null(), 0);
        // SAFETY: the encoder is live and the outputs writable.
        let got = unsafe {
            pt_encoder_finish(
                enc,
                &raw mut pico,
                &raw mut pico_len,
                &raw mut layer,
                &raw mut layer_len,
            )
        };
        ensure!(got == 0, "pt_encoder_finish: {}", error());
        // SAFETY: the outputs hold these many bytes until pt_encoder_free.
        let bytes = unsafe { std::slice::from_raw_parts(pico, pico_len) }.to_vec();
        let layer = if layer.is_null() {
            Vec::new()
        } else {
            // SAFETY: as above.
            unsafe { std::slice::from_raw_parts(layer, layer_len) }.to_vec()
        };
        // A second finish fails.
        // SAFETY: the encoder is live and the outputs writable.
        let again = unsafe {
            pt_encoder_finish(
                enc,
                &raw mut pico,
                &raw mut pico_len,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        ensure!(again == -1, "a second pt_encoder_finish succeeded");
        Ok((bytes, layer))
    })();
    // SAFETY: the encoder came from pt_encoder_new and is not used after.
    unsafe { pt_encoder_free(enc) };
    result
}

/// Decodes a whole file through the C interface; returns its header and the
/// interleaved samples.
fn decode_c(pico: &[u8], layer: &[u8]) -> Result<(PtHeader, Vec<f32>)> {
    let mut header = PtHeader::default();
    // SAFETY: the buffers are valid for their lengths; the header is writable.
    let got = unsafe {
        pt_header(
            pico.as_ptr(),
            pico.len(),
            layer.as_ptr(),
            layer.len(),
            &raw mut header,
        )
    };
    ensure!(got == 0, "pt_header: {}", error());
    // SAFETY: the buffers are valid for their lengths.
    let stream = unsafe { pt_open(pico.as_ptr(), pico.len(), layer.as_ptr(), layer.len(), 0) };
    ensure!(!stream.is_null(), "pt_open: {}", error());
    let mut out = Vec::new();
    let result = loop {
        let (mut data, mut frames) = (std::ptr::null(), 0);
        // SAFETY: the stream is live and the outputs writable.
        let got = unsafe { pt_decode(stream, 4, &raw mut data, &raw mut frames) };
        match got {
            1 => {
                let n = frames * usize::try_from(header.channels)?;
                // SAFETY: the block holds `frames` interleaved frames.
                out.extend_from_slice(unsafe { std::slice::from_raw_parts(data, n) });
            }
            0 => break Ok((header, out)),
            _ => break Err(anyhow::anyhow!("pt_decode: {}", error())),
        }
    };
    // SAFETY: the stream came from pt_open and is not used after.
    unsafe { pt_close(stream) };
    result
}

/// An encode to compare: the input, the command line's flags, the same as C
/// options, and the tags.
type Case<'a> = (
    &'a Path,
    &'a [&'a str],
    &'a [(&'a str, &'a str)],
    &'a [&'a str],
);

#[test]
fn encoder_matches_the_command_line() -> Result<()> {
    let exe = cli()?;
    let weights = weights_dir(None);
    let tables = tables_dir(None);
    // Null folders: the same default lookup as the command line's.
    // SAFETY: null is allowed for both.
    let got = unsafe { pt_init(std::ptr::null(), std::ptr::null()) };
    ensure!(got == 0, "pt_init: {}", error());
    let dir = std::env::temp_dir().join(format!("picotone-ffi-encode-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let stereo = dir.join("stereo.wav");
    let mono = dir.join("mono.wav");
    audio::write_wav(
        &stereo,
        &signal(44_100, 2, 6),
        WavFormat::Pcm16,
        false,
        true,
    )?;
    audio::write_wav(&mono, &signal(32_000, 1, 5), WavFormat::Pcm16, false, true)?;
    let cases: [Case<'_>; 5] = [
        (&stereo, &[], &[], &[]),
        (&stereo, &["--codebooks", "3"], &[("codebooks", "3")], &[]),
        (&stereo, &["--fast"], &[("fast", "1")], &[]),
        (
            &mono,
            &[
                "--codebooks",
                "6",
                "--layer-codebooks",
                "12",
                "--tag",
                "TITLE=t",
            ],
            &[("codebooks", "6"), ("layer-codebooks", "12")],
            &["TITLE=t"],
        ),
        (
            &mono,
            &["--fast", "--loop", "--beam", "2"],
            &[("fast", "1"), ("loop", "1"), ("beam", "2")],
            &[],
        ),
    ];
    for (i, (input, flags, options, tags)) in cases.iter().enumerate() {
        let out = dir.join(format!("case{i}.pico"));
        let status = Command::new(&exe)
            .arg("--weights")
            .arg(&weights)
            .arg("--tables")
            .arg(&tables)
            .arg("encode")
            .arg(input)
            .arg(&out)
            .args(*flags)
            .status()
            .context("running picotone encode")?;
        ensure!(status.success(), "picotone encode {flags:?} failed");
        let want = std::fs::read(&out)?;
        let want_layer = std::fs::read(out.with_extension("picol")).unwrap_or_default();
        let source = audio::read(input)?;
        let (bytes, layer) = encode_c(&source, options, tags)?;
        ensure!(
            bytes == want,
            "case {i} {flags:?}: the .pico differs from the command line's"
        );
        ensure!(
            layer == want_layer,
            "case {i} {flags:?}: the .picol differs from the command line's"
        );
        let (header, samples) = decode_c(&bytes, &layer)?;
        let channels = usize::try_from(header.channels)?;
        let frames = samples.len() / channels;
        ensure!(
            u64::try_from(frames)? == header.length,
            "case {i}: decoded {frames} frames of {}",
            header.length
        );
        ensure!(
            samples.iter().all(|v| v.is_finite()),
            "case {i}: non-finite samples"
        );
        eprintln!(
            "case {i} {flags:?}: {} bytes (+{} layer), identical; decoded {frames} x {channels} at {} Hz",
            bytes.len(),
            layer.len(),
            header.sample_rate
        );
    }
    drop(std::fs::remove_dir_all(&dir));
    Ok(())
}

#[test]
fn bad_options_fail_at_once() -> Result<()> {
    let enc = pt_encoder_new(44_100, 2);
    ensure!(!enc.is_null(), "pt_encoder_new: {}", error());
    let (k, v, bad) = (c"codebooks", c"many", c"nope");
    // SAFETY: the encoder is live and the strings NUL-terminated.
    let results = unsafe {
        [
            pt_encoder_set(enc, k.as_ptr(), v.as_ptr()),
            pt_encoder_set(enc, bad.as_ptr(), v.as_ptr()),
            pt_encoder_tag(enc, bad.as_ptr()),
        ]
    };
    // SAFETY: the encoder came from pt_encoder_new and is not used after.
    unsafe { pt_encoder_free(enc) };
    if results != [-1, -1, -1] {
        bail!("bad options were accepted: {results:?}");
    }
    Ok(())
}
