//! Audio file input (WAV, FLAC, MP3, Ogg Vorbis, AAC/M4A, ... via Symphonia),
//! WAV output, channel conversion and resampling.

use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};
use rayon::prelude::*;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::config::{floor_to_usize, to_f64};
use crate::nn::Seq;

/// Planar floating-point audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// One vector of samples per channel, all the same length.
    pub channels: Vec<Vec<f32>>,
}

impl Audio {
    /// Samples per channel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// Whether there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Duration in seconds.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        to_f64(self.len()) / f64::from(self.sample_rate)
    }

    /// Interleaves into a time-major sequence.
    ///
    /// # Errors
    /// Fails when there are no channels.
    pub fn to_seq(&self) -> Result<Seq> {
        let c = self.channels.len();
        ensure!(c > 0, "audio has no channels");
        let len = self.len();
        let mut data = vec![0.0; len * c];
        for (ch, samples) in self.channels.iter().enumerate() {
            for (t, &v) in samples.iter().enumerate() {
                data[t * c + ch] = v;
            }
        }
        Seq::from_vec(len, c, data)
    }

    /// De-interleaves a time-major sequence.
    #[must_use]
    pub fn from_seq(seq: &Seq, sample_rate: u32) -> Self {
        let c = seq.channels();
        let channels = (0..c)
            .map(|ch| seq.data().iter().skip(ch).step_by(c).copied().collect())
            .collect();
        Self {
            sample_rate,
            channels,
        }
    }

    /// Converts to `channels` channels and `sample_rate` Hz like EnCodec's
    /// `convert_audio`: down-mix by averaging, up-mix by duplication, then
    /// windowed-sinc resampling.
    ///
    /// # Errors
    /// Fails when the audio has no channels.
    pub fn convert(&self, sample_rate: u32, channels: usize) -> Result<Self> {
        ensure!(!self.channels.is_empty(), "audio has no channels");
        let mixed: Vec<Vec<f32>> = if self.channels.len() == channels {
            self.channels.clone()
        } else if channels == 1 {
            let n = to_f64(self.channels.len());
            vec![
                (0..self.len())
                    .map(|t| {
                        let sum: f64 = self.channels.iter().map(|c| f64::from(c[t])).sum();
                        narrow(sum / n)
                    })
                    .collect(),
            ]
        } else if self.channels.len() == 1 {
            vec![self.channels[0].clone(); channels]
        } else if self.channels.len() > channels {
            self.channels[..channels].to_vec()
        } else {
            bail!(
                "cannot convert {} channels to {channels}",
                self.channels.len()
            );
        };
        let resampled = mixed
            .par_iter()
            .map(|c| resample(c, self.sample_rate, sample_rate))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            sample_rate,
            channels: resampled,
        })
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "audio samples are well inside f32 range"
)]
const fn narrow(v: f64) -> f32 {
    v as f32
}

/// Decodes the first audio track of any format Symphonia supports.
///
/// # Errors
/// Fails when the file cannot be opened or decoded.
pub fn read(path: &Path) -> Result<Audio> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let _ = hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("{} is not a supported audio file", path.display()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("file has no audio track")?;
    let track_id = track.id;
    let mut codec = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported audio codec")?;
    let mut sample_rate = track.codec_params.sample_rate;
    let mut channel_count = track
        .codec_params
        .channels
        .map(symphonia::core::audio::Channels::count);
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e).context("reading audio packets"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match codec.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = Some(spec.rate);
                channel_count = Some(spec.channels.count());
                let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buffer.copy_interleaved_ref(decoded);
                interleaved.extend_from_slice(buffer.samples());
            }
            // Corrupt packets are skipped, as media players do.
            Err(SymphoniaError::DecodeError(_)) => {}
            Err(e) => return Err(e).context("decoding audio"),
        }
    }
    let sample_rate = sample_rate.context("unknown sample rate")?;
    let c = channel_count.context("unknown channel count")?;
    ensure!(c > 0, "audio has no channels");
    let channels = (0..c)
        .map(|ch| interleaved.iter().skip(ch).step_by(c).copied().collect())
        .collect();
    Ok(Audio {
        sample_rate,
        channels,
    })
}

/// WAV sample encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavFormat {
    /// 16-bit signed PCM.
    Pcm16,
    /// 32-bit IEEE float.
    Float32,
}

/// Writes a WAV file, scaled down to fit when `rescale` is set. Samples are
/// clipped to ±0.99 (decodes are limited to that already, so this is only a
/// guard), except float samples when `clip` is off.
///
/// # Errors
/// Fails when the file cannot be written.
pub fn write_wav(
    path: &Path,
    audio: &Audio,
    format: WavFormat,
    rescale: bool,
    clip: bool,
) -> Result<()> {
    let spec = hound::WavSpec {
        channels: u16::try_from(audio.channels.len())?,
        sample_rate: audio.sample_rate,
        bits_per_sample: match format {
            WavFormat::Pcm16 => 16,
            WavFormat::Float32 => 32,
        },
        sample_format: match format {
            WavFormat::Pcm16 => hound::SampleFormat::Int,
            WavFormat::Float32 => hound::SampleFormat::Float,
        },
    };
    let limit = 0.99_f32;
    let gain = if rescale {
        let peak = audio
            .channels
            .iter()
            .flatten()
            .fold(0.0_f32, |m, v| m.max(v.abs()));
        if peak > limit { limit / peak } else { 1.0 }
    } else {
        1.0
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("creating {}", path.display()))?;
    for t in 0..audio.len() {
        for channel in &audio.channels {
            let v = channel[t] * gain;
            match format {
                WavFormat::Pcm16 => writer.write_sample(to_i16(v.clamp(-limit, limit)))?,
                WavFormat::Float32 if clip => writer.write_sample(v.clamp(-limit, limit))?,
                WavFormat::Float32 => writer.write_sample(v)?,
            }
        }
    }
    writer.finalize()?;
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "input is clamped to ±0.99 first"
)]
fn to_i16(v: f32) -> i16 {
    (v * 32_767.0).round() as i16
}

/// Resamples one channel with the windowed-sinc filter of
/// `julius.resample_frac` (24 zero crossings, roll-off 0.945).
///
/// # Errors
/// Fails on a zero sample rate.
pub fn resample(x: &[f32], old_rate: u32, new_rate: u32) -> Result<Vec<f32>> {
    ensure!(
        old_rate > 0 && new_rate > 0,
        "sample rates must be positive"
    );
    if old_rate == new_rate || x.is_empty() {
        return Ok(x.to_vec());
    }
    let g = gcd(old_rate, new_rate);
    let old = usize::try_from(old_rate / g)?;
    let new = usize::try_from(new_rate / g)?;
    let zeros = 24.0_f64;
    let cutoff = to_f64(old.min(new)) * 0.945;
    let width = floor_to_usize((zeros * to_f64(old) / cutoff).ceil())?;
    let taps = 2 * width + old;
    let kernels: Vec<Vec<f32>> = (0..new)
        .map(|phase| {
            let raw: Vec<f64> = (0..taps)
                .map(|m| {
                    let idx = to_f64(m) - to_f64(width);
                    let t = ((-to_f64(phase) / to_f64(new) + idx / to_f64(old)) * cutoff)
                        .clamp(-zeros, zeros)
                        * PI;
                    let window = (t / zeros / 2.0).cos().powi(2);
                    let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                    sinc * window
                })
                .collect();
            let sum: f64 = raw.iter().sum();
            raw.iter().map(|&k| narrow(k / sum)).collect()
        })
        .collect();
    // Replicate-pad `width` on the left and `width + old` on the right.
    let last = x.len() - 1;
    let padded: Vec<f32> = (0..x.len() + 2 * width + old)
        .map(|i| x[i.saturating_sub(width).min(last)])
        .collect();
    let blocks = x.len() / old + 1;
    let out_len = floor_to_usize(to_f64(new) * to_f64(x.len()) / to_f64(old))?;
    let mut out = vec![0.0_f32; blocks * new];
    out.par_chunks_mut(new).enumerate().for_each(|(b, dst)| {
        let window = &padded[b * old..b * old + taps];
        for (v, kernel) in dst.iter_mut().zip(&kernels) {
            *v = kernel.iter().zip(window).map(|(k, s)| k * s).sum();
        }
    });
    out.truncate(out_len);
    Ok(out)
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[cfg(test)]
#[expect(clippy::cast_precision_loss, reason = "small test indices")]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_dc_and_length() -> Result<()> {
        let x = vec![0.5_f32; 44_100];
        let y = resample(&x, 44_100, 24_000)?;
        assert_eq!(y.len(), 24_000);
        assert!(y.iter().all(|v| (v - 0.5).abs() < 1e-4));
        Ok(())
    }

    #[test]
    fn resample_keeps_a_low_tone() -> Result<()> {
        let tone = |rate: f32, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate).sin())
                .collect()
        };
        let y = resample(&tone(48_000.0, 48_000), 48_000, 24_000)?;
        let want = tone(24_000.0, 24_000);
        let err = y[100..23_900]
            .iter()
            .zip(&want[100..23_900])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "max error {err}");
        Ok(())
    }
}
