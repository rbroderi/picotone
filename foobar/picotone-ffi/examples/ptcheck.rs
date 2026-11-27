//! Checks the C interface on a file, as the foobar2000 component uses it:
//! the header, a whole decode (timed), and a seek, whose samples must equal
//! the whole decode's at that point.
//!
//! `cargo run --release --example ptcheck -- WEIGHTS MODELS FILE.pico [FILE.picol]`

use std::ffi::{CStr, CString};
use std::time::Instant;

use anyhow::{Context as _, Result, bail, ensure};
use picotone_ffi::{PtHeader, pt_close, pt_decode, pt_header, pt_init, pt_last_error, pt_open};

fn error() -> String {
    // SAFETY: pt_last_error never returns null, and the text is NUL-terminated.
    unsafe { CStr::from_ptr(pt_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Decodes from `start` to the end, interleaved.
fn decode(pico: &[u8], layer: &[u8], start: u64, channels: usize) -> Result<Vec<f32>> {
    // SAFETY: the buffers are valid for their lengths.
    let stream = unsafe {
        pt_open(
            pico.as_ptr(),
            pico.len(),
            layer.as_ptr(),
            layer.len(),
            start,
        )
    };
    ensure!(!stream.is_null(), "pt_open: {}", error());
    let mut out = Vec::new();
    loop {
        let mut data = std::ptr::null();
        let mut frames = 0;
        // SAFETY: the stream is live and the outputs are writable.
        let got = unsafe { pt_decode(stream, 1, &raw mut data, &raw mut frames) };
        if got < 0 {
            // SAFETY: the stream came from pt_open.
            unsafe { pt_close(stream) };
            bail!("pt_decode: {}", error());
        }
        if got == 0 {
            break;
        }
        // SAFETY: pt_decode returned `frames * channels` readable floats,
        // valid until the next call on the stream.
        let block = unsafe { std::slice::from_raw_parts(data, frames * channels) };
        out.extend_from_slice(block);
    }
    // SAFETY: the stream came from pt_open.
    unsafe { pt_close(stream) };
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 3,
        "usage: ptcheck WEIGHTS MODELS FILE.pico [FILE.picol]"
    );
    let (weights, models) = (
        CString::new(args[0].as_str())?,
        CString::new(args[1].as_str())?,
    );
    let pico = std::fs::read(&args[2]).with_context(|| args[2].clone())?;
    let layer = match args.get(3) {
        Some(path) => std::fs::read(path).with_context(|| path.clone())?,
        None => Vec::new(),
    };

    let t = Instant::now();
    // SAFETY: NUL-terminated strings.
    ensure!(
        unsafe { pt_init(weights.as_ptr(), models.as_ptr()) } == 0,
        "pt_init: {}",
        error()
    );
    let mut header = PtHeader::default();
    // SAFETY: valid buffers and a writable header.
    let ok = unsafe {
        pt_header(
            pico.as_ptr(),
            pico.len(),
            layer.as_ptr(),
            layer.len(),
            &raw mut header,
        )
    };
    ensure!(ok == 0, "pt_header: {}", error());
    println!("{header:?} (init {:.2} s)", t.elapsed().as_secs_f64());
    let channels = usize::try_from(header.channels)?;
    let length = usize::try_from(header.length)?;
    let layer = if header.layered == 1 {
        layer
    } else {
        Vec::new()
    };

    let t = Instant::now();
    let whole = decode(&pico, &layer, 0, channels)?;
    let open_and_decode = t.elapsed().as_secs_f64();
    ensure!(
        whole.len() == length * channels,
        "decoded {} samples, expected {}",
        whole.len() / channels,
        length
    );
    let t = Instant::now();
    let again = decode(&pico, &layer, 0, channels)?;
    let seconds = t.elapsed().as_secs_f64();
    ensure!(again == whole, "two decodes differ");
    let audio = f64::from(u32::try_from(header.length)?) / f64::from(header.sample_rate);
    println!(
        "decode: {seconds:.2} s for {audio:.1} s of audio, {:.1}x real time (first, with the network load: {open_and_decode:.2} s)",
        audio / seconds
    );

    for start in [length / 3, length / 2 + 12_345] {
        let part = decode(&pico, &layer, u64::try_from(start)?, channels)?;
        let expect = &whole[start * channels..];
        ensure!(
            part.len() == expect.len(),
            "seek to {start}: {} samples, expected {}",
            part.len(),
            expect.len()
        );
        let diff = part
            .iter()
            .zip(expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("seek to {start}: max difference {diff:e}");
        ensure!(diff == 0.0, "the seek differs from the whole decode");
    }
    println!("OK");
    Ok(())
}
