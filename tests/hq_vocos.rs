//! The Vocos HQ decoder against PyTorch (random-init fixtures written by
//! `tools/vocos_fixture.py`), and its chunked and streaming decodes against
//! a whole-file decode, bit for bit.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};
use picotone::cx2::hq::HqModel;
use picotone::dac::HOP;
use picotone::hqstream::{HqDecoder, HqStream};
use picotone::pico::{self, ChannelLayout, DacInfo, DacStream};
use picotone::vocos::VocosDecoder;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A `[frames, 9]` little-endian u16 `.npy` file (format 1.0).
fn read_codes(path: &Path) -> Result<(Vec<u16>, usize)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() > 10 && &bytes[1..6] == b"NUMPY",
        "not a .npy file"
    );
    let header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header = std::str::from_utf8(&bytes[10..10 + header_len])?;
    ensure!(
        header.contains("<u2") && header.contains("False"),
        "unexpected header {header}"
    );
    let codes: Vec<u16> = bytes[10 + header_len..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
        .collect();
    ensure!(codes.len().is_multiple_of(9), "codes are not [frames, 9]");
    let frames = codes.len() / 9;
    Ok((codes, frames))
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

/// The first `n` codebooks of `[frames, 9]` codes.
fn tier(codes: &[u16], n: usize) -> Vec<u16> {
    codes
        .as_chunks::<9>()
        .0
        .iter()
        .flat_map(|f| f[..n].iter().copied())
        .collect()
}

/// Max abs error, its ratio to the reference peak, and the SNR in dB.
fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let mut err = 0.0_f64;
    let mut noise = 0.0_f64;
    let mut signal = 0.0_f64;
    let mut peak = 0.0_f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - f64::from(w);
        err = err.max(d.abs());
        noise += d * d;
        signal += f64::from(w) * f64::from(w);
        peak = peak.max(f64::from(w).abs());
    }
    (err, err / peak, 10.0 * (signal / noise.max(1e-300)).log10())
}

/// Checks `<stem>.q<n>.f32` for each tier present against a Rust decode.
fn parity(stem: &Path, tiers: &[usize]) -> Result<()> {
    let d = VocosDecoder::load(&stem.with_extension("safetensors"))?;
    let name = stem
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let (codes, frames) = read_codes(&stem.with_file_name(format!("{name}.codes.npy")))?;
    for &n in tiers {
        let want = read_f32(&stem.with_file_name(format!("{name}.q{n}.f32")))?;
        let got = d.decode(&tier(&codes, n), n)?;
        assert_eq!(got.len(), frames * HOP);
        assert_eq!(want.len(), got.len());
        let (err, rel, snr) = compare(&got, &want);
        eprintln!("{name} q{n}: max abs error {err:.3e} ({rel:.3e} of peak), SNR {snr:.1} dB");
        assert!(
            rel < 1e-4,
            "{name} q{n}: max error {err:e} is {rel:e} of peak"
        );
    }
    Ok(())
}

#[test]
fn matches_pytorch_on_the_tiny_fixture() -> Result<()> {
    parity(&root().join("tests/data/vocos_tiny"), &[9, 4, 1])
}

/// The full-size (512 wide, 8 blocks) random-init model, when it has been
/// generated into `work/vocos/full` (see `tools/vocos_fixture.py`).
#[test]
fn matches_pytorch_at_full_size() -> Result<()> {
    let stem = root().join("work/vocos/full");
    if !stem.with_extension("safetensors").exists() {
        eprintln!("skipped: work/vocos/full.safetensors is not there");
        return Ok(());
    }
    parity(&stem, &[9, 5, 1])?;
    let d = VocosDecoder::load(&stem.with_extension("safetensors"))?;
    let c = codes(200, 9, 5);
    let whole = d.decode_chunked(&c, 9, usize::MAX)?;
    for chunk in [1, 13, 64, 100] {
        assert!(
            same_bits(&whole, &d.decode_chunked(&c, 9, chunk)?),
            "chunk {chunk}"
        );
    }
    Ok(())
}

fn tiny() -> Result<VocosDecoder> {
    VocosDecoder::load(&root().join("tests/data/vocos_tiny.safetensors"))
}

/// Pseudo-random codes.
fn codes(frames: usize, n: usize, seed: u32) -> Vec<u16> {
    let mut state = seed;
    (0..frames * n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u16::try_from(state >> 22).unwrap_or(0)
        })
        .collect()
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn chunked_decodes_equal_the_whole_file_bit_for_bit() -> Result<()> {
    let d = tiny()?;
    assert_eq!(d.receptive_field(), 29);
    for (frames, n) in [(300, 9), (131, 3), (5, 2), (1, 9)] {
        let c = codes(frames, n, 11);
        let whole = d.decode_chunked(&c, n, 100_000)?;
        assert_eq!(whole.len(), frames * HOP);
        assert!(whole.iter().all(|v| v.is_finite()));
        for chunk in [1, 7, 40, 80, 256] {
            let parts = d.decode_chunked(&c, n, chunk)?;
            assert!(
                same_bits(&whole, &parts),
                "{frames} frames q{n}, chunk {chunk}"
            );
        }
    }
    Ok(())
}

fn table() -> Result<HqModel> {
    HqModel::from_table(&std::fs::read(root().join("models/cx2_dac44.ecxm"))?)
}

#[test]
fn streaming_decodes_equal_the_whole_file_bit_for_bit() -> Result<()> {
    let d = tiny()?;
    let table = table()?;
    let frames = 2100;
    let info = DacInfo {
        codebook_hash: d.codebook_hash(),
        table_hash: table.hash(),
        layout: ChannelLayout::LeftRight,
        codebooks: vec![9, 5],
        looped: false,
        pre_skip: 300,
        page_length: pico::DEFAULT_PAGE_FRAMES,
        audio_length: frames * HOP - 77,
        end_trim: 700,
        length_in_header: true,
        output_gain: 0,
        tags: Vec::new(),
    };
    let channels = vec![codes(frames, 9, 3), codes(frames, 5, 4)];
    let bytes = pico::write_dac_parametric(&info, &table, &channels, &[])?;
    let left = d.decode(&channels[0], 9)?;
    let right = d.decode(&channels[1], 5)?;
    let playable = info.playable_samples();
    let want: Vec<f32> = (300..300 + playable)
        .flat_map(|t| [left[t], right[t]])
        .collect();
    let dyn_d: &dyn HqDecoder = &d;
    for start in [0, 777, playable - 1] {
        for units in [1, 3, usize::MAX] {
            let mut s = HqStream::new(DacStream::parse(&bytes)?, &table, dyn_d, start)?;
            let mut got = Vec::new();
            while let Some((_, audio)) = s.decode_next(units)? {
                got.extend_from_slice(audio.data());
            }
            assert!(
                same_bits(&got, &want[start * 2..]),
                "start {start}, {units} units: {} of {} samples",
                got.len() / 2,
                playable - start
            );
        }
    }
    Ok(())
}
