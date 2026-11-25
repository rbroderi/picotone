//! The STFT encoder against PyTorch (a random-init fixture written by
//! `tools/stftenc_fixture.py`), its chunked and span encodes against a
//! whole-signal encode, bit for bit, and (ignored, with the real weights)
//! parity files for `tools/stftenc_parity.py` and the encode speed.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result};
use picotone::dac::{DacQuantizer, HOP};
use picotone::stftenc::StftEncoder;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

/// Little-endian int16 PCM, as `int16 / 32768`.
fn read_pcm(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f32::from(i16::from_le_bytes(b)) / 32768.0)
        .collect())
}

/// SNR of `got` against `want`, in dB.
fn snr(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut noise, mut signal) = (0.0_f64, 0.0_f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - f64::from(w);
        noise += d * d;
        signal += f64::from(w) * f64::from(w);
    }
    10.0 * (signal / noise.max(1e-300)).log10()
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn tiny() -> Result<(StftEncoder, Vec<f32>)> {
    let stem = root().join("tests/data/stftenc_tiny");
    let enc = StftEncoder::load(&stem.with_extension("safetensors"))?;
    let audio = read_pcm(&stem.with_extension("pcm"))?;
    Ok((enc, audio))
}

#[test]
fn matches_pytorch_on_the_tiny_fixture() -> Result<()> {
    let (enc, audio) = tiny()?;
    assert_eq!(enc.channels(), 4106);
    assert_eq!(enc.receptive_field(), 24);
    let stem = root().join("tests/data/stftenc_tiny");
    let want = read_f32(&stem.with_extension("z.f32"))?;
    let exact = read_f32(&stem.with_extension("z64.f32"))?;
    let z = enc.latent(&audio)?;
    assert_eq!(z.len(), audio.len() / HOP);
    let got = z.data();
    let (vs_exact, vs_torch, torch_vs_exact) =
        (snr(got, &exact), snr(got, &want), snr(&want, &exact));
    eprintln!(
        "tiny: SNR against float64 {vs_exact:.1} dB, against PyTorch float32 {vs_torch:.1} dB \
         (PyTorch float32 against float64: {torch_vs_exact:.1} dB)"
    );
    // The features are computed in double precision, so the encoder is far
    // closer to the exact (float64) model than PyTorch's float32 STFT is:
    // that STFT's rounding of the quiet bins between a tone's harmonics,
    // magnified by the compression, limits any comparison with it.
    assert!(vs_exact >= 100.0, "SNR against float64 {vs_exact:.1} dB");
    assert!(
        vs_torch >= torch_vs_exact - 3.0,
        "SNR against PyTorch {vs_torch:.1} dB"
    );
    Ok(())
}

#[test]
fn chunked_and_span_encodes_equal_the_whole_signal_bit_for_bit() -> Result<()> {
    let (enc, audio) = tiny()?;
    let frames = audio.len() / HOP;
    let whole = enc.latent_chunked(&audio, usize::MAX)?;
    assert!(whole.data().iter().all(|v| v.is_finite()));
    assert!(same_bits(whole.data(), enc.latent(&audio)?.data()));
    for chunk in [1, 5, 24, 25, 47] {
        let parts = enc.latent_chunked(&audio, chunk)?;
        assert!(same_bits(whole.data(), parts.data()), "chunk {chunk}");
    }
    // A span from just the samples it needs.
    for range in [0..1, 3..20, 30..frames, 47..48, 0..frames] {
        let needed = enc.samples_needed(range.clone(), audio.len());
        let span = enc.latent_span(
            &audio[needed.clone()],
            needed.start,
            audio.len(),
            range.clone(),
        )?;
        let latent = enc.latent_dim();
        let want = &whole.data()[range.start * latent..range.end * latent];
        assert!(same_bits(span.data(), want), "span {range:?}");
        // One sample fewer is refused.
        if needed.start > 0 {
            let short = &audio[needed.start + 1..needed.end];
            assert!(
                enc.latent_span(short, needed.start + 1, audio.len(), range)
                    .is_err()
            );
        }
    }
    // The last 300 samples are zero: a shorter signal, read as zero padded,
    // has the same latent. One sample more adds a frame, which changes the
    // last frames within the receptive field only.
    let trimmed = enc.latent(&audio[..audio.len() - 300])?;
    assert!(same_bits(whole.data(), trimmed.data()));
    let mut longer = audio;
    longer.push(0.0);
    let longer = enc.latent(&longer)?;
    assert_eq!(longer.len(), frames + 1);
    let same = (frames - enc.receptive_field()) * enc.latent_dim();
    assert!(same_bits(&whole.data()[..same], &longer.data()[..same]));
    assert_eq!(enc.latent(&[])?.len(), 0);
    Ok(())
}

/// The weights directory: `$PICOTONE_WEIGHTS` or `weights/`.
fn weights_dir() -> PathBuf {
    std::env::var_os("PICOTONE_WEIGHTS").map_or_else(|| root().join("weights"), PathBuf::from)
}

/// DAC's quantiser (`weights/dac44_quantizer.safetensors`, made by
/// `tools/export_quantizer.py`) indexes the decoder's codebooks, and codes a
/// latent the same in parallel pieces as in short calls.
#[test]
fn quantizer_matches_the_decoder_and_codes_in_pieces() -> Result<()> {
    use picotone::hqstream::HqDecoder as _;
    let quantizer = DacQuantizer::load(&weights_dir().join("dac44_quantizer.safetensors"))?;
    let decoder =
        picotone::vocos::VocosDecoder::load(&root().join("tests/data/vocos_tiny.safetensors"))?;
    assert_eq!(quantizer.codebook_hash(), decoder.codebook_hash());
    let frames = 700;
    let mut state = 7_u32;
    let values: Vec<f32> = (0..frames * picotone::dac::LATENT_DIM)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "test values")]
            let v = (state >> 8) as f32 / 16_777_216.0 - 0.5;
            4.0 * v
        })
        .collect();
    let latent = picotone::nn::Seq::from_vec(frames, picotone::dac::LATENT_DIM, values)?;
    let whole = quantizer.quantize(latent.clone())?;
    assert_eq!(whole.len(), latent.len() * picotone::dac::CODEBOOKS);
    let mut parts = Vec::new();
    for start in (0..latent.len()).step_by(100) {
        let end = (start + 100).min(latent.len());
        parts.extend(quantizer.quantize(latent.slice_rows(start, end))?);
    }
    assert_eq!(whole, parts);
    Ok(())
}

/// Real-weights parity: for each `<dir>/<clip>.in.f32` written by
/// `tools/stftenc_parity.py prepare` (default dir `work/stftenc`), writes the
/// Rust latent `<clip>.rust.f32` and prints its SNR against PyTorch's.
#[test]
#[ignore = "needs the real weights and tools/stftenc_parity.py's clips"]
fn real_weights_parity_files() -> Result<()> {
    let enc = StftEncoder::load(&weights_dir().join("dac44_stftenc.safetensors"))?;
    let dir = std::env::var_os("STFTENC_PARITY")
        .map_or_else(|| root().join("work/stftenc"), PathBuf::from);
    let mut clips: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".in.f32"))
        .collect();
    clips.sort();
    assert!(!clips.is_empty(), "no clips in {}", dir.display());
    for clip in clips {
        let name = clip
            .to_string_lossy()
            .trim_end_matches(".in.f32")
            .to_owned();
        let audio = read_f32(&clip)?;
        let z = enc.latent(&audio)?;
        write_f32(Path::new(&format!("{name}.rust.f32")), z.data())?;
        let torch = read_f32(Path::new(&format!("{name}.torch.f32")))?;
        let exact = read_f32(Path::new(&format!("{name}.torch64.f32")))?;
        eprintln!(
            "{name}: {} frames; Rust vs PyTorch fp32 {:.1} dB, Rust vs fp64 {:.1} dB, PyTorch fp32 vs fp64 \
             {:.1} dB",
            z.len(),
            snr(z.data(), &torch),
            snr(z.data(), &exact),
            snr(&torch, &exact)
        );
    }
    Ok(())
}

/// Encode speed per second of mono audio, on one thread and on all: the
/// latent, then DAC's quantiser (`dac44_quantizer.safetensors`).
#[test]
#[ignore = "a benchmark; needs the real weights"]
fn speed() -> Result<()> {
    let dir = weights_dir();
    let enc = StftEncoder::load(&dir.join("dac44_stftenc.safetensors"))?;
    let quantizer = DacQuantizer::load(&dir.join("dac44_quantizer.safetensors"))?;
    let seconds = 30;
    let mut state = 1_u32;
    let audio: Vec<f32> = (0..seconds * 44_100)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "test signal")]
            let tone = (i as f32 * 0.0627).sin() * 0.3;
            #[expect(clippy::cast_precision_loss, reason = "test signal")]
            let noise = (state >> 8) as f32 / 16_777_216.0 - 0.5;
            tone + 0.05 * noise
        })
        .collect();
    let all = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    for threads in [1, all] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        pool.install(|| -> Result<()> {
            drop(enc.latent(&audio[..44_100])?);
            let t = Instant::now();
            let latent = enc.latent(&audio)?;
            let ours = t.elapsed().as_secs_f64() / f64::from(seconds);
            let t = Instant::now();
            drop(quantizer.quantize(latent)?);
            let quantize = t.elapsed().as_secs_f64() / f64::from(seconds);
            eprintln!(
                "{threads} thread(s): STFT encoder {:.2} ms, quantiser {:.2} ms per channel-second",
                ours * 1e3,
                quantize * 1e3
            );
            Ok(())
        })?;
    }
    Ok(())
}
