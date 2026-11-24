//! The HQ decoder in parts: chunked network decodes, the streaming decode of
//! [`HqStream`] (every layout, part size and seek point) and its agreement
//! with a whole-file decode, bit for bit.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context as _, Result};
use picotone::cx2::hq::HqModel;
use picotone::dac::{DacDecoder, HOP};
use picotone::hqstream::{HqDecoder, HqStream};
use picotone::layout::Trimmer;
use picotone::nn::Seq;
use picotone::pico::{self, ChannelLayout, DacInfo, DacStream};
use picotone::pstereo;

fn decoder() -> Result<&'static DacDecoder> {
    static DECODER: OnceLock<Option<DacDecoder>> = OnceLock::new();
    DECODER
        .get_or_init(|| {
            DacDecoder::load(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/dac44_student.safetensors"),
            )
            .ok()
        })
        .as_ref()
        .context("weights/dac44_student.safetensors is missing or malformed")
}

fn table() -> Result<&'static HqModel> {
    static TABLE: OnceLock<Option<HqModel>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/cx2_dac44.ecxm");
            HqModel::from_table(&std::fs::read(path).ok()?).ok()
        })
        .as_ref()
        .context("models/cx2_dac44.ecxm is missing or malformed")
}

/// Smooth-ish pseudo-random codes: real codes repeat, random ones do not, but
/// the network does not mind.
fn codes(frames: usize, n: usize, seed: u32) -> Vec<u16> {
    let mut state = seed;
    (0..frames * n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u16::try_from(state >> 22).unwrap_or(0)
        })
        .collect()
}

#[test]
fn chunked_decodes_equal_the_whole_file_bit_for_bit() -> Result<()> {
    let d = decoder()?;
    for (frames, n) in [(300, 9), (131, 3)] {
        let c = codes(frames, n, 11);
        let whole = d.decode_chunked(&c, n, 100_000)?;
        assert_eq!(whole.len(), frames * HOP);
        for chunk in [1, 7, 40, 80, 256] {
            let parts = d.decode_chunked(&c, n, chunk)?;
            let bad: Vec<usize> = whole
                .iter()
                .zip(&parts)
                .enumerate()
                .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, _)| i)
                .collect();
            let max = whole
                .iter()
                .zip(&parts)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                bad.is_empty(),
                "{frames} frames q{n}, chunk {chunk}: {} samples differ, by up to {max:e}",
                bad.len()
            );
        }
    }
    Ok(())
}

/// A test stream and its expected output.
struct Case {
    name: &'static str,
    bytes: Vec<u8>,
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "test noise")]
            let v = (state >> 8) as f32 / 16_777_216.0 - 0.5;
            v * 0.5
        })
        .collect()
}

fn case(
    name: &'static str,
    layout: ChannelLayout,
    codebooks: Vec<usize>,
    frames: usize,
    tweak: impl FnOnce(&mut DacInfo),
) -> Result<Case> {
    let mut info = DacInfo {
        codebook_hash: decoder()?.codebook_hash(),
        table_hash: table()?.hash(),
        layout,
        codebooks,
        looped: false,
        pre_skip: 0,
        page_length: pico::DEFAULT_PAGE_FRAMES,
        audio_length: frames * HOP,
        end_trim: 0,
        length_in_header: true,
        output_gain: 0,
        tags: Vec::new(),
    };
    tweak(&mut info);
    let frames = info.frames();
    let channels: Vec<Vec<u16>> = info
        .codebooks
        .iter()
        .zip(3..)
        .map(|(&n, seed)| codes(frames, n, seed))
        .collect();
    let parameters = match layout {
        ChannelLayout::Parametric(config) => {
            let n = frames * HOP;
            let (a, b) = (noise(n, 1), noise(n, 2));
            let left: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.8 * x + 0.3 * y).collect();
            let right: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.4 * x - 0.3 * y).collect();
            let spans = info.pages();
            let params = pstereo::analyse(&left, &right, &spans, config);
            pstereo::write_pages(&params, &spans, config)
        }
        _ => Vec::new(),
    };
    let bytes = pico::write_dac_parametric(&info, table()?, &channels, &parameters)?;
    Ok(Case { name, bytes })
}

/// The whole-file decode: every channel's codes at once, then the output
/// channels, then the trim.
fn reference(bytes: &[u8]) -> Result<Seq> {
    let stream = DacStream::parse(bytes)?;
    let info = &stream.info;
    let d = decoder()?;
    let signals = stream
        .decode_all(table()?)?
        .iter()
        .zip(&info.codebooks)
        .map(|(c, &n)| d.decode_chunked(c, n, usize::MAX))
        .collect::<Result<Vec<_>>>()?;
    let channels = match info.layout {
        ChannelLayout::Mono | ChannelLayout::LeftRight => signals,
        ChannelLayout::MidSide => {
            let (m, s) = (&signals[0], &signals[1]);
            vec![
                m.iter().zip(s).map(|(a, b)| a + b).collect(),
                m.iter().zip(s).map(|(a, b)| a - b).collect(),
            ]
        }
        ChannelLayout::Parametric(config) => {
            let pages = (0..stream.pages.len())
                .map(|p| stream.page_parameters(p))
                .collect::<Result<Vec<_>>>()?;
            let params = pstereo::read_pages(&pages, &stream.spans, config)?;
            pstereo::upmix(&signals[0], &params, &stream.spans, config)?.to_vec()
        }
    };
    let count = channels.len();
    let length = channels[0].len();
    let mut data = vec![0.0; length * count];
    for (c, channel) in channels.iter().enumerate() {
        for (t, &v) in channel.iter().enumerate() {
            data[t * count + c] = v;
        }
    }
    let raw = Seq::from_vec(length, count, data)?;
    let mut trimmer = Trimmer::new(
        info.pre_skip,
        info.playable_samples(),
        info.looped,
        44_100,
        count,
    )?;
    let out = trimmer.push(&raw)?;
    trimmer.finish()?;
    Ok(out)
}

fn cases() -> Result<Vec<Case>> {
    Ok(vec![
        case("mono q9 trimmed", ChannelLayout::Mono, vec![9], 170, |i| {
            i.pre_skip = 300;
            i.end_trim = 1000;
            i.audio_length -= 77;
        })?,
        case(
            "lr q9/q5, short pages, length in the last packet",
            ChannelLayout::LeftRight,
            vec![9, 5],
            150,
            |i| {
                i.page_length = 30;
                i.length_in_header = false;
                i.end_trim = 600;
            },
        )?,
        case("ms q7/q3", ChannelLayout::MidSide, vec![7, 3], 130, |i| {
            i.pre_skip = 512 * 3 + 5;
        })?,
        case(
            "parametric q9",
            ChannelLayout::Parametric(pstereo::Config::DEFAULT),
            vec![9],
            200,
            |i| i.end_trim = 2000,
        )?,
        case(
            "parametric 10 bands q5, one page",
            ChannelLayout::Parametric(pstereo::Config::new(10, 4)?),
            vec![5],
            90,
            |i| i.page_length = 0,
        )?,
        case("mono loop", ChannelLayout::Mono, vec![6], 120, |i| {
            i.looped = true;
            i.pre_skip = 4000;
            i.end_trim = 1500;
        })?,
    ])
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn streaming_decodes_equal_the_whole_file_bit_for_bit() -> Result<()> {
    let d: &dyn HqDecoder = decoder()?;
    for case in cases()? {
        let want = reference(&case.bytes)?;
        let total = want.len();
        let channels = want.channels();
        for start in [0, 1, 511, 512 * 7 + 3, total / 2, total - 1] {
            for units in [1, 2, usize::MAX] {
                let stream = DacStream::parse(&case.bytes)?;
                let mut s = HqStream::new(stream, table()?, d, start)?;
                let mut got = Vec::new();
                let mut next = start;
                while let Some((offset, audio)) = s.decode_next(units)? {
                    assert_eq!(offset, next, "{}", case.name);
                    assert!(!audio.is_empty());
                    next += audio.len();
                    got.extend_from_slice(audio.data());
                }
                assert_eq!(s.units_done(), s.unit_count());
                assert!(
                    same_bits(&got, &want.data()[start * channels..]),
                    "{}: start {start}, {units} units: {} of {} samples",
                    case.name,
                    got.len() / channels,
                    total - start
                );
            }
        }
        assert!(HqStream::new(DacStream::parse(&case.bytes)?, table()?, d, total).is_err());
    }
    Ok(())
}
