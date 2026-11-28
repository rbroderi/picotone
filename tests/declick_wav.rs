//! Runs the declicker on a WAV file, for listening and for comparing with the
//! prototype: `PICOTONE_DECLICK=in.wav,out.wav,start` (`start` is the frame
//! coordinate of the first sample: 0 when the file starts on a frame
//! boundary). Skipped when the variable is not set.

use std::path::Path;

use anyhow::{Context as _, Result};
use picotone::audio::{self, Audio, WavFormat};
use picotone::declick;

#[test]
fn declick_a_wav_file() -> Result<()> {
    let Ok(spec) = std::env::var("PICOTONE_DECLICK") else {
        eprintln!("skipped: PICOTONE_DECLICK is not set");
        return Ok(());
    };
    let mut parts = spec.split(',');
    let input = parts.next().context("no input")?;
    let output = parts.next().context("no output")?;
    let start: usize = parts.next().unwrap_or("0").parse()?;
    let audio = audio::read(Path::new(input))?;
    let out = declick::declick(&audio.to_seq()?, start, start)?;
    let out = Audio::from_seq(&out, audio.sample_rate);
    audio::write_wav(Path::new(output), &out, WavFormat::Float32, false, false)?;
    eprintln!("wrote {output}");
    Ok(())
}
