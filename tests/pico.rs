//! End-to-end tests of the cx2 coder and the `.pico` format, with a synthetic
//! cx2 table that switches every expert on.

use anyhow::Result;
use picotone::ModelKind;
use picotone::cx2::features::Centroids;
use picotone::cx2::kernels::Simd;
use picotone::cx2::model::self_contained_hash;
use picotone::cx2::{Model, decode_chunk, encode_chunk, encode_codes, fit};
use picotone::layout::Crossfade;
use picotone::model::EncodedFrame;
use picotone::pico::{self, Info, Layout, Stream};
use picotone::scale;

const CARD: usize = 1024;
const DIM: usize = 128;
const CODEBOOKS: usize = 3;

/// Deterministic pseudo-random stream.
struct Lcg(u64);

impl Lcg {
    const fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A value in `-range..=range`.
    fn signed(&mut self, range: i32) -> i32 {
        let span = u64::try_from(2 * range + 1).unwrap_or(1);
        i32::try_from(self.next() % span).unwrap_or(0) - range
    }
}

/// Minimal `safetensors` file holding random centroids for `CODEBOOKS` layers.
fn weights(rng: &mut Lcg) -> Vec<u8> {
    let size = CARD * DIM * 4;
    let entries: Vec<String> = (0..CODEBOOKS)
        .map(|k| {
            format!(
                "\"quantizer.layers.{k}.codebook.embed\":{{\"dtype\":\"F32\",\
                 \"shape\":[{CARD},{DIM}],\"data_offsets\":[{},{}]}}",
                k * size,
                (k + 1) * size
            )
        })
        .collect();
    let header = format!("{{{}}}", entries.join(","));
    let mut out = u64::try_from(header.len())
        .unwrap_or(0)
        .to_le_bytes()
        .to_vec();
    out.extend_from_slice(header.as_bytes());
    for _ in 0..CODEBOOKS * CARD * DIM {
        let value = f32::from(i16::try_from(rng.signed(2000)).unwrap_or(0)) / 256.0;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn push_values(out: &mut Vec<u8>, rng: &mut Lcg, count: usize, range: i32) {
    for _ in 0..count {
        out.extend_from_slice(&rng.signed(range).to_le_bytes());
    }
}

/// Model table (version 4) with small random values in every expert, including the net.
fn table(rng: &mut Lcg) -> Vec<u8> {
    table_with(rng, None, 0)
}

/// Like [`table`], but version 5: the table stores `centroids` and its net has a
/// position input with the given period (0: none).
fn table_v5(rng: &mut Lcg, centroids: &[i16], period: u32) -> Vec<u8> {
    table_with(rng, Some(centroids), period)
}

fn table_with(rng: &mut Lcg, centroids: Option<&[i16]>, period: u32) -> Vec<u8> {
    let mut out = b"ECXM".to_vec();
    let version = if centroids.is_some() { 5 } else { 4 };
    for value in [version, u32::try_from(CODEBOOKS).unwrap_or(0), 6, 64] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for step in 0..64_u32 {
        // Roughly 2^24 * 2^(-step/64); the exact values do not matter for a round trip.
        let value = (1_u32 << 24) - step * 180_000;
        out.extend_from_slice(&value.to_le_bytes());
    }
    for &value in centroids.unwrap_or(&[]) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for _ in 0..CODEBOOKS {
        push_values(&mut out, rng, CARD, 200);
        push_values(&mut out, rng, 8 * 34, 300);
        push_values(&mut out, rng, 29 * 16, 300);
        for _ in 0..3 {
            push_values(&mut out, rng, 96, 300);
        }
        for _ in 0..3 {
            // One entry per context code.
            for offset in 0..=1024_u32 {
                out.extend_from_slice(&offset.to_le_bytes());
            }
            for context in 0..1024_u16 {
                out.extend_from_slice(&(context * 7 % 1024).to_le_bytes());
            }
            for _ in 0..CARD {
                out.extend_from_slice(&i16::try_from(rng.signed(400)).unwrap_or(0).to_le_bytes());
            }
        }
        for _ in 0..2 {
            out.extend_from_slice(&8_u32.to_le_bytes());
            out.extend_from_slice(&9_u32.to_le_bytes());
            for _ in 0..CARD * 8 {
                out.extend_from_slice(&i16::try_from(rng.signed(4000)).unwrap_or(0).to_le_bytes());
            }
            for _ in 0..CARD * 8 {
                out.extend_from_slice(&i8::try_from(rng.signed(127)).unwrap_or(0).to_le_bytes());
            }
        }
    }
    // The trained expert: tiny layers, random weights within the format's limits.
    let (h1, h2, rank) = (16_u32, 8_u32, 8_u32);
    for value in [1, h1, h2, 2, 2, 3] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    let weights = |out: &mut Vec<u8>, rng: &mut Lcg, count: u32, range: i32| {
        for _ in 0..count {
            out.extend_from_slice(&i16::try_from(rng.signed(range)).unwrap_or(0).to_le_bytes());
        }
    };
    push_values(&mut out, rng, 2 * 16, 20_000);
    if centroids.is_some() {
        // Position input: buckets 0, 1, 2 for the first frames, 3 for the rest.
        let buckets = period.min(4);
        out.extend_from_slice(&period.to_le_bytes());
        out.extend_from_slice(&buckets.to_le_bytes());
        for position in 0..period {
            out.push(u8::try_from(position.min(3)).unwrap_or(0));
        }
        push_values(
            &mut out,
            rng,
            usize::try_from(buckets * h1).unwrap_or(0),
            200_000,
        );
    }
    weights(&mut out, rng, 4 * 128 * h1, 300);
    push_values(&mut out, rng, 16, 20_000);
    weights(&mut out, rng, h1 * h2, 2000);
    push_values(&mut out, rng, 8, 200_000);
    for _ in 0..CODEBOOKS {
        out.extend_from_slice(&rank.to_le_bytes());
        weights(&mut out, rng, h2 * rank, 2000);
        weights(&mut out, rng, 128 * rank, 300);
        push_values(&mut out, rng, 8, 200_000);
        weights(&mut out, rng, rank * 1024, 2000);
        push_values(&mut out, rng, 1024, 100);
    }
    out
}

/// Codes with repeats and jumps, so the adaptive experts all see hits.
fn codes(rng: &mut Lcg, frames: usize, codebooks: usize) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(frames * codebooks);
    for _ in 0..frames * codebooks {
        let roll = rng.next() % 10;
        let code = if roll < 4 && out.len() >= codebooks {
            out[out.len() - codebooks]
        } else if roll < 7 {
            u16::try_from(rng.next() % 16).unwrap_or(0)
        } else {
            u16::try_from(rng.next() % 1024).unwrap_or(0)
        };
        out.push(code);
    }
    out
}

/// The synthetic model, from a version 4 table and random weights.
fn model(rng: &mut Lcg) -> Result<Model> {
    let table = table(rng);
    Model::load(&table, &weights(rng))
}

/// Decodes the raw chunks of [`encode_codes`] back to codes.
fn decode_raw(
    model: &Model,
    nc: usize,
    frames: usize,
    total: usize,
    chunks: &[Vec<u8>],
) -> Result<Vec<u16>> {
    let len = if frames == 0 { total } else { frames * nc };
    let mut out = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let count = len.min(total - index * len);
        out.extend(decode_chunk(model, nc, 0, count, chunk)?.1);
    }
    Ok(out)
}

#[test]
fn code_streams_round_trip_at_every_codebook_count() -> Result<()> {
    let mut rng = Lcg(11);
    let model = model(&mut rng)?;
    for codebooks in 1..=CODEBOOKS {
        let stream = codes(&mut rng, 60, codebooks);
        for reset in [0, 7, 400] {
            let chunks = encode_codes(&model, codebooks, reset, &stream)?;
            assert_eq!(
                decode_raw(&model, codebooks, reset, stream.len(), &chunks)?,
                stream
            );
        }
    }
    Ok(())
}

#[test]
fn chunks_carry_scales_before_codes() -> Result<()> {
    let mut rng = Lcg(13);
    let model = model(&mut rng)?;
    let stream = codes(&mut rng, 30, 2);
    let scales = [0_u16, 1, 4095, 3584, 1234];
    let chunk = encode_chunk(&model, 2, &scales, &stream)?;
    let (back, codes) = decode_chunk(&model, 2, scales.len(), stream.len(), &chunk)?;
    assert_eq!(back, scales);
    assert_eq!(codes, stream);
    // Twelve bits per scale, give or take the coder's final bytes.
    let bare = encode_chunk(&model, 2, &[], &stream)?;
    assert!(chunk.len() >= bare.len() + 6 && chunk.len() <= bare.len() + 10);
    assert!(encode_chunk(&model, 2, &[4096], &stream).is_err());
    Ok(())
}

#[test]
fn a_damaged_chunk_leaves_the_others_intact() -> Result<()> {
    let mut rng = Lcg(9);
    let model = model(&mut rng)?;
    let stream = codes(&mut rng, 90, 2);
    let mut chunks = encode_codes(&model, 2, 30, &stream)?;
    assert_eq!(chunks.len(), 3);
    for byte in &mut chunks[1] {
        *byte ^= 0x5a;
    }
    // The damaged chunk decodes to something or fails (never a panic); the others are exact.
    assert_eq!(decode_chunk(&model, 2, 0, 60, &chunks[0])?.1, stream[..60]);
    if let Ok((_, bad)) = decode_chunk(&model, 2, 0, 60, &chunks[1]) {
        assert_ne!(bad, stream[60..120]);
    }
    assert_eq!(decode_chunk(&model, 2, 0, 60, &chunks[2])?.1, stream[120..]);
    Ok(())
}

#[test]
fn every_dispatch_path_codes_identically() -> Result<()> {
    let mut rng = Lcg(21);
    let (table, weights) = (table(&mut rng), weights(&mut rng));
    let model = Model::load(&table, &weights)?;
    let mut baseline = Model::load(&table, &weights)?;
    baseline.set_simd(Simd::baseline());
    let stream = codes(&mut rng, 50, CODEBOOKS);
    assert_eq!(
        encode_codes(&model, CODEBOOKS, 20, &stream)?,
        encode_codes(&baseline, CODEBOOKS, 20, &stream)?
    );
    Ok(())
}

#[test]
fn self_contained_tables_with_a_position_input_round_trip() -> Result<()> {
    let mut rng = Lcg(31);
    let weights = weights(&mut rng);
    let fixed = Centroids::from_safetensors(&weights, CODEBOOKS)?
        .values()
        .to_vec();
    for period in [0, 5, 150] {
        let table = table_v5(&mut rng, &fixed, period);
        // The table alone is enough; the weights, when given, are only checked.
        let model = Model::from_table(&table)?;
        let checked = Model::load(&table, &weights)?;
        assert_eq!(model.hash_hex(), checked.hash_hex());
        assert_eq!(self_contained_hash(&table), Some(model.hash()));
        let mut baseline = Model::from_table(&table)?;
        baseline.set_simd(Simd::baseline());
        for codebooks in 1..=CODEBOOKS {
            let stream = codes(&mut rng, 40, codebooks);
            for reset in [0, 7, 15] {
                let chunks = encode_codes(&model, codebooks, reset, &stream)?;
                assert_eq!(chunks, encode_codes(&baseline, codebooks, reset, &stream)?);
                assert_eq!(chunks, encode_codes(&checked, codebooks, reset, &stream)?);
                assert_eq!(
                    decode_raw(&model, codebooks, reset, stream.len(), &chunks)?,
                    stream
                );
            }
        }
    }
    Ok(())
}

#[test]
fn a_self_contained_table_rejects_other_weights() -> Result<()> {
    let mut rng = Lcg(33);
    let weights = weights(&mut rng);
    let other = weights_with_seed(34);
    let fixed = Centroids::from_safetensors(&weights, CODEBOOKS)?
        .values()
        .to_vec();
    let table = table_v5(&mut rng, &fixed, 150);
    assert!(Model::load(&table, &other).is_err());
    assert!(Model::load(&table, &[]).is_ok());
    // A version 4 table cannot be used without weights, and has no file hash.
    let old = table_with(&mut rng, None, 0);
    assert!(Model::from_table(&old).is_err());
    assert_eq!(self_contained_hash(&old), None);
    Ok(())
}

fn weights_with_seed(seed: u64) -> Vec<u8> {
    weights(&mut Lcg(seed))
}

#[test]
fn offline_fitting_runs_on_small_streams() -> Result<()> {
    let mut rng = Lcg(3);
    let table = table(&mut rng);
    let weights = weights(&mut rng);
    let streams = vec![codes(&mut rng, 60, 2), codes(&mut rng, 40, 2)];
    let centroids = Centroids::from_safetensors(&weights, 2)?;
    let counts = fit::histograms(&centroids, 2, 1, 25, &streams)?;
    assert_eq!(counts.len(), 2);
    assert_eq!(counts[0].geo.hit.iter().sum::<u64>(), 100);
    assert_eq!(counts[1].geo.hit.iter().sum::<u64>(), 0);
    assert_eq!(counts[1].rec.all.iter().sum::<u64>(), 100 * 1024);
    let model = Model::load(&table, &weights)?;
    let tuned = fit::tune(&model, 2, &streams, 20, 25)?;
    assert_eq!(tuned.len(), 2);
    assert!(
        tuned
            .iter()
            .all(|t| t.bits_tuned.is_finite() && t.weights.len() == 11)
    );
    Ok(())
}

/// Settings of a test stream beyond its model and length.
#[derive(Clone, Copy)]
struct Shape {
    stride: usize,
    crossfade: Crossfade,
    pre_skip: usize,
    end_trim: usize,
    looped: bool,
    page_length: usize,
    length_in_header: bool,
}

const PLAIN: Shape = Shape {
    stride: 47_520,
    crossfade: Crossfade::Triangle,
    pre_skip: 0,
    end_trim: 0,
    looped: false,
    page_length: 1,
    length_in_header: true,
};

const LOOPED: Shape = Shape {
    stride: 44_800,
    crossfade: Crossfade::Triangle,
    pre_skip: 7_200,
    end_trim: 2_400,
    looped: true,
    page_length: 2,
    length_in_header: true,
};

/// A two-channel stream of `segments` whole strides plus a short tail.
fn stereo(
    rng: &mut Lcg,
    model: &Model,
    segments: usize,
    shape: Shape,
) -> Result<(Info, Vec<EncodedFrame>)> {
    let audio_length = segments * shape.stride + 480;
    let layout = Layout::new(ModelKind::Stereo48k, Some(shape.stride))?;
    let frames = layout
        .segment_frames(audio_length)
        .into_iter()
        .map(|n| EncodedFrame {
            scale: Some(scale::dequantize(
                u16::try_from(rng.next() % 4096).unwrap_or(0),
            )),
            num_codebooks: 2,
            codes: codes(rng, n, 2),
        })
        .collect();
    let info = Info {
        model: ModelKind::Stereo48k,
        table_hash: model.hash(),
        num_codebooks: 2,
        stride: Some(shape.stride),
        crossfade: shape.crossfade,
        looped: shape.looped,
        pre_skip: shape.pre_skip,
        page_length: shape.page_length,
        audio_length,
        end_trim: shape.end_trim,
        length_in_header: shape.length_in_header,
        output_gain: 0,
        tags: Vec::new(),
    };
    Ok((info, frames))
}

/// A one-channel stream of `frames` latent frames for the 24 kHz model.
fn mono(
    rng: &mut Lcg,
    model: &Model,
    frames: usize,
    page_length: usize,
) -> (Info, Vec<EncodedFrame>) {
    let info = Info {
        model: ModelKind::Mono24k,
        table_hash: model.hash(),
        num_codebooks: 3,
        stride: None,
        crossfade: Crossfade::Triangle,
        looped: false,
        pre_skip: 500,
        page_length,
        audio_length: 320 * frames - 100,
        end_trim: 0,
        length_in_header: true,
        output_gain: 0,
        tags: Vec::new(),
    };
    let frames = vec![EncodedFrame {
        scale: None,
        num_codebooks: 3,
        codes: codes(rng, frames, 3),
    }];
    (info, frames)
}

fn same(a: &[EncodedFrame], b: &[EncodedFrame]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.codes == y.codes
                && x.num_codebooks == y.num_codebooks
                && x.scale.map(f32::to_bits) == y.scale.map(f32::to_bits)
        })
}

#[test]
fn streams_round_trip_and_seek() -> Result<()> {
    let mut rng = Lcg(31);
    let model = model(&mut rng)?;
    let tags = vec!["TITLE=Test".to_owned(), "LOOPSTART=4800".to_owned()];
    for page_length in [0, 1, 2] {
        for length_in_header in [true, false] {
            let shape = Shape {
                page_length,
                length_in_header,
                ..PLAIN
            };
            let (mut info, frames) = stereo(&mut rng, &model, 5, shape)?;
            info.tags.clone_from(&tags);
            info.output_gain = -256;
            let bytes = pico::write(&model, &info, &frames)?;
            let stream = Stream::parse(&bytes)?;
            assert_eq!(stream.info, info);
            assert_eq!(pico::read_info(&bytes)?.tags, tags);
            assert_eq!(stream.playable_samples(), 5 * 47_520 + 480);
            assert!(same(&stream.decode_all(&model)?, &frames));
            // The page holding a sample is found from granule positions alone, and
            // decoding just that page gives the same codes as the full decode.
            if page_length == 1 {
                assert_eq!(stream.pages.len(), 6);
                assert_eq!(stream.page_for_sample(0), 0);
                assert_eq!(stream.page_for_sample(47_519), 0);
                assert_eq!(stream.page_for_sample(47_520), 1);
                assert_eq!(stream.page_for_sample(3 * 47_520 + 5), 3);
                assert!(same(&stream.decode_page(&model, 3)?, &frames[3..4]));
            }
        }
    }
    Ok(())
}

#[test]
fn streams_keep_stride_trims_and_loop_flag() -> Result<()> {
    let mut rng = Lcg(35);
    let model = model(&mut rng)?;
    // Once with the edge-avoiding cross-fade, once as the presets write loops.
    let edge = Shape {
        stride: 45_000,
        crossfade: Crossfade::EdgeAvoiding,
        pre_skip: 12_000,
        end_trim: 9_000,
        ..LOOPED
    };
    for base in [edge, LOOPED] {
        for (page_length, length_in_header) in [(1, true), (2, false)] {
            let shape = Shape {
                page_length,
                length_in_header,
                ..base
            };
            let (info, frames) = stereo(&mut rng, &model, 6, shape)?;
            let stream = Stream::parse(&pico::write(&model, &info, &frames)?)?;
            assert_eq!(stream.info, info);
            assert_eq!(stream.info.segmenting().stride, Some(shape.stride));
            assert_eq!(
                *stream.granules.last().unwrap_or(&0),
                u64::try_from(info.audio_length - shape.end_trim)?
            );
            assert_eq!(
                stream.playable_samples(),
                info.audio_length - shape.pre_skip - shape.end_trim
            );
            assert!(same(&stream.decode_all(&model)?, &frames));
            // Seeking counts playable samples: sample 0 lies `pre_skip` samples into the
            // first page, and pages end every `page_length` strides.
            let page = page_length * shape.stride;
            let skip = u64::try_from(shape.pre_skip)?;
            let page = u64::try_from(page)?;
            assert_eq!(stream.page_for_sample(0), 0);
            assert_eq!(stream.page_for_sample(page - skip - 1), 0);
            assert_eq!(stream.page_for_sample(page - skip), 1);
            let mut tail = Vec::new();
            for index in 1..stream.pages.len() {
                tail.extend(stream.decode_page(&model, index)?);
            }
            assert!(same(&tail, &frames[page_length..]));
        }
    }
    Ok(())
}

#[test]
fn streams_round_trip_the_unsegmented_model() -> Result<()> {
    let mut rng = Lcg(37);
    let model = model(&mut rng)?;
    for page_length in [0, 75, 300] {
        let (info, frames) = mono(&mut rng, &model, 400, page_length);
        let stream = Stream::parse(&pico::write(&model, &info, &frames)?)?;
        assert_eq!(stream.info, info);
        assert!(same(&stream.decode_all(&model)?, &frames));
        assert_eq!(stream.playable_samples(), 320 * 400 - 600);
        if page_length == 75 {
            assert_eq!(stream.pages.len(), 6);
            let part = stream.decode_page(&model, 2)?;
            assert_eq!(part[0].codes, frames[0].codes[150 * 3..225 * 3]);
        }
    }
    Ok(())
}

#[test]
fn short_sounds_have_a_small_fixed_cost() -> Result<()> {
    let mut rng = Lcg(51);
    let model = model(&mut rng)?;
    // A tenth of a second at 48 kHz: one segment, one page.
    let (mut info, mut frames) = stereo(&mut rng, &model, 0, PLAIN)?;
    info.audio_length = 4_800;
    frames[0].codes = codes(&mut rng, 15, 2);
    let bytes = pico::write(&model, &info, &frames)?;
    let starts = page_starts(&bytes);
    assert_eq!(starts.len(), 3);
    // The identification page: a 28-byte page header and a 26-byte packet
    // (6 of framing, 8 of table hash, 12 of fields).
    assert!(
        starts[1] <= 54,
        "identification page is {} bytes",
        starts[1]
    );
    let payload = Stream::parse(&bytes)?.pages[0].as_ref().map_or(0, Vec::len);
    assert_eq!(bytes.len(), starts[1] + 28 + payload);
    Ok(())
}

#[test]
fn a_different_table_is_rejected() -> Result<()> {
    let mut rng = Lcg(5);
    let shared = weights(&mut rng);
    let model = Model::load(&table(&mut rng), &shared)?;
    let other = Model::load(&table(&mut rng), &shared)?;
    let (info, frames) = mono(&mut rng, &model, 10, 0);
    let stream = Stream::parse(&pico::write(&model, &info, &frames)?)?;
    assert!(stream.decode_all(&other).is_err());
    assert!(pico::write(&other, &info, &frames).is_err());
    Ok(())
}

#[test]
fn the_writer_checks_its_input() -> Result<()> {
    let mut rng = Lcg(53);
    let model = model(&mut rng)?;
    let (info, frames) = stereo(&mut rng, &model, 2, PLAIN)?;
    // A scale off the 12-bit grid.
    let mut off = frames.clone();
    off[0].scale = Some(0.123_456_7);
    assert!(pico::write(&model, &info, &off).is_err());
    // A segment too few.
    assert!(pico::write(&model, &info, &frames[1..]).is_err());
    // Trims that leave nothing.
    let trimmed = Info {
        pre_skip: info.audio_length,
        ..info.clone()
    };
    assert!(pico::write(&model, &trimmed, &frames).is_err());
    // A stride the model cannot use.
    let wide = Info {
        stride: Some(50_000),
        ..info
    };
    assert!(pico::write(&model, &wide, &frames).is_err());
    Ok(())
}

#[test]
fn damage_is_confined_to_one_page() -> Result<()> {
    let mut rng = Lcg(33);
    let model = model(&mut rng)?;
    let (info, frames) = stereo(&mut rng, &model, 4, PLAIN)?;
    let mut bytes = pico::write(&model, &info, &frames)?;
    let clean = Stream::parse(&bytes)?;
    // Flip one byte inside an audio page in the middle of the file.
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x10;
    let damaged = Stream::parse(&bytes)?;
    assert_eq!(damaged.pages.len(), clean.pages.len());
    let lost: Vec<usize> = (0..damaged.pages.len())
        .filter(|&i| damaged.pages[i].is_none())
        .collect();
    assert_eq!(lost.len(), 1);
    for index in 0..clean.pages.len() {
        if index == lost[0] {
            assert!(damaged.decode_page(&model, index).is_err());
        } else {
            assert!(same(
                &damaged.decode_page(&model, index)?,
                &clean.decode_page(&model, index)?
            ));
        }
    }
    Ok(())
}

#[test]
fn reserved_and_unknown_fields_are_handled() -> Result<()> {
    let mut rng = Lcg(55);
    let model = model(&mut rng)?;
    let (info, frames) = mono(&mut rng, &model, 20, 0);
    let bytes = pico::write(&model, &info, &frames)?;
    let starts = page_starts(&bytes);
    // Rebuild the identification page with one extra TLV after the fixed fields.
    let with_field = |kind: u8, body: &[u8]| -> Vec<u8> {
        let lacing = usize::from(bytes[26]);
        let mut packet = bytes[27 + lacing..starts[1]].to_vec();
        packet.push(kind);
        packet.push(u8::try_from(body.len()).unwrap_or(0));
        packet.extend_from_slice(body);
        let mut page = bytes[..26].to_vec();
        page.push(1);
        page.push(u8::try_from(packet.len()).unwrap_or(0));
        page.extend_from_slice(&packet);
        page.extend_from_slice(&bytes[starts[1]..]);
        reseal(&mut page);
        page
    };
    // An unknown even (optional) type is skipped.
    let stream = Stream::parse(&with_field(8, &[1, 2, 3]))?;
    assert!(same(&stream.decode_all(&model)?, &frames));
    // Output gain is read.
    assert_eq!(
        Stream::parse(&with_field(2, &[0x00, 0x01]))?
            .info
            .output_gain,
        256
    );
    // The reserved types, and unknown odd (critical) ones, refuse the stream.
    for kind in [1, 3, 5, 7] {
        assert!(Stream::parse(&with_field(kind, &[0x11])).is_err());
    }
    Ok(())
}

/// A packet's data, its page's granule position, and its first and last flags.
type ExternalPacket = (Vec<u8>, u64, bool, bool);

/// Packets of an Ogg file as the independent `ogg` crate reads them (it verifies page
/// CRCs, lacing and sequence numbers): data, page granule position, first and last flags.
fn external_packets(bytes: &[u8]) -> Result<Vec<ExternalPacket>> {
    let mut reader = ::ogg::reading::PacketReader::new(std::io::Cursor::new(bytes));
    let mut out = Vec::new();
    let mut serial = None;
    while let Some(packet) = reader.read_packet()? {
        assert_eq!(
            *serial.get_or_insert_with(|| packet.stream_serial()),
            packet.stream_serial()
        );
        let (granule, first, last) = (
            packet.absgp_page(),
            packet.first_in_stream(),
            packet.last_in_stream(),
        );
        out.push((packet.data, granule, first, last));
    }
    Ok(out)
}

#[test]
fn an_independent_ogg_reader_sees_the_same_packets() -> Result<()> {
    let mut rng = Lcg(39);
    let model = model(&mut rng)?;
    let single = Shape {
        page_length: 0,
        ..PLAIN
    };
    let streaming = Shape {
        length_in_header: false,
        ..LOOPED
    };
    // The long file in one page is one packet of more than 65025 bytes, so it spans pages.
    for (segments, shape) in [(5, PLAIN), (6, LOOPED), (6, streaming), (254, single)] {
        let (mut info, frames) = stereo(&mut rng, &model, segments, shape)?;
        info.tags = vec!["TITLE=Test".to_owned()];
        let bytes = pico::write(&model, &info, &frames)?;
        let stream = Stream::parse(&bytes)?;
        let packets = external_packets(&bytes)?;
        assert_eq!(packets.len(), 2 + stream.pages.len());
        assert!(packets[0].0.starts_with(b"Pico\x01") && packets[0].2);
        assert!(packets[1].0.starts_with(b"PicoTags"));
        for (index, (data, granule, first, last)) in packets.iter().enumerate() {
            assert_eq!(*first, index == 0);
            assert_eq!(*last, index + 1 == packets.len());
            if index >= 2 {
                let chunk = stream.pages[index - 2].as_ref().map_or(0, Vec::len);
                // Only the last packet of a stream without a length starts with the end trim.
                let trim = if *last && !shape.length_in_header {
                    2
                } else {
                    0
                };
                assert_eq!(data.len(), trim + chunk);
                assert_eq!(*granule, stream.granules[index - 2]);
            }
        }
        // A packet longer than 255 lacing values of 255 bytes continues on further pages.
        let pages = bytes.windows(4).filter(|w| w == b"OggS").count();
        if shape.page_length == 0 {
            assert!(packets[2].0.len() > 65_025);
            assert!(pages > packets.len());
        } else {
            assert_eq!(pages, packets.len());
        }
        let end = u64::try_from(info.audio_length - info.end_trim)?;
        assert_eq!(packets.last().map(|p| p.1), Some(end));
    }
    Ok(())
}

/// Runs `ffprobe` (from `FFPROBE` or the search path) over a stream; skipped when absent.
///
/// `FFmpeg` has no decoder for these streams and reports an error for the file, but its Ogg
/// demuxer still walks every page and checks each CRC, which is what this looks at.
#[test]
fn ffprobe_finds_no_framing_errors() -> Result<()> {
    let program = std::env::var("FFPROBE").unwrap_or_else(|_| "ffprobe".to_owned());
    if std::process::Command::new(&program)
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("ffprobe not found; skipping");
        return Ok(());
    }
    let mut rng = Lcg(41);
    let model = model(&mut rng)?;
    let (mut info, frames) = stereo(&mut rng, &model, 8, PLAIN)?;
    info.tags = vec!["TITLE=Test".to_owned()];
    let bytes = pico::write(&model, &info, &frames)?;
    let pages = bytes.windows(4).filter(|w| w == b"OggS").count();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("ffprobe_check.pico");
    std::fs::write(&path, &bytes)?;
    let output = std::process::Command::new(&program)
        .args(["-v", "trace", "-f", "ogg", "-show_format"])
        .arg(&path)
        .output()?;
    let log = String::from_utf8_lossy(&output.stderr).to_lowercase();
    for complaint in [
        "crc mismatch",
        "invalid data",
        "corrupt",
        "garbage",
        "bad page",
    ] {
        assert!(!log.contains(complaint), "ffprobe reported: {complaint}");
    }
    // The whole file was read, and one page header was parsed per page we wrote.
    assert!(log.contains(&format!("{} bytes read", bytes.len())));
    let seen = log.matches("nsegs=").count();
    assert!(
        seen == 0 || seen >= pages,
        "ffprobe saw {seen} of {pages} pages"
    );
    Ok(())
}

/// Ogg's page checksum, to re-seal pages after a mutation so the damage reaches the
/// packet parsers instead of stopping at the CRC.
fn ogg_crc(bytes: &[u8]) -> u32 {
    let mut crc = 0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 == 0 {
                crc << 1
            } else {
                (crc << 1) ^ 0x04C1_1DB7
            };
        }
    }
    crc
}

/// Recomputes the checksum of every page that can still be delimited.
fn reseal(bytes: &mut [u8]) {
    let mut pos = 0;
    while pos + 27 <= bytes.len() {
        if &bytes[pos..pos + 4] != b"OggS" {
            pos += 1;
            continue;
        }
        let lacing = usize::from(bytes[pos + 26]);
        let Some(table) = bytes.get(pos + 27..pos + 27 + lacing) else {
            break;
        };
        let end = pos + 27 + lacing + table.iter().map(|&l| usize::from(l)).sum::<usize>();
        if end > bytes.len() {
            break;
        }
        bytes[pos + 22..pos + 26].fill(0);
        let crc = ogg_crc(&bytes[pos..end]);
        bytes[pos + 22..pos + 26].copy_from_slice(&crc.to_le_bytes());
        pos = end;
    }
}

/// Start offsets of the pages of an Ogg file, plus its length.
fn page_starts(bytes: &[u8]) -> Vec<usize> {
    let mut out: Vec<usize> = (0..bytes.len().saturating_sub(3))
        .filter(|&i| &bytes[i..i + 4] == b"OggS")
        .collect();
    out.push(bytes.len());
    out
}

/// One deterministic mutation of `base`, drawing on `other` for splices.
fn mutate(rng: &mut Lcg, base: &[u8], other: &[u8]) -> Vec<u8> {
    let mut out = base.to_vec();
    let pick = |rng: &mut Lcg, n: usize| usize::try_from(rng.next()).unwrap_or(0) % n.max(1);
    let rounds = 1 + pick(rng, 3);
    for _ in 0..rounds {
        let len = out.len();
        match rng.next() % 9 {
            0 => {
                for _ in 0..=pick(rng, 8) {
                    let at = pick(rng, len);
                    if let Some(byte) = out.get_mut(at) {
                        *byte ^= 1 << (rng.next() % 8);
                    }
                }
            }
            1 => out.truncate(pick(rng, len + 1)),
            2 => {
                // Tamper with a length-like field: 1, 2, 4 or 8 bytes set to an extreme.
                let width = [1, 2, 4, 8][pick(rng, 4)];
                let at = pick(rng, len);
                let value = [0, 1, 0x7f, 0x80, 0xff, 0xfe][pick(rng, 6)];
                for byte in out.iter_mut().skip(at).take(width) {
                    *byte = value;
                }
            }
            3 => {
                // The same, aimed at the first 96 bytes, where the headers live.
                let width = [1, 2, 4, 8][pick(rng, 4)];
                let at = pick(rng, len.min(96));
                let value = u8::try_from(rng.next() % 256).unwrap_or(0);
                for byte in out.iter_mut().skip(at).take(width) {
                    *byte = value;
                }
            }
            4 => {
                // Splice: pages (or byte ranges) of two files in a new order.
                let (a, b) = (page_starts(&out), page_starts(other));
                let mut joined = Vec::new();
                for _ in 0..=pick(rng, 6) {
                    let (source, starts) = if rng.next().is_multiple_of(2) {
                        (out.as_slice(), &a)
                    } else {
                        (other, &b)
                    };
                    if starts.len() >= 2 {
                        let page = pick(rng, starts.len() - 1);
                        joined.extend_from_slice(&source[starts[page]..starts[page + 1]]);
                    } else {
                        let from = pick(rng, source.len());
                        joined.extend_from_slice(&source[from..(from + 64).min(source.len())]);
                    }
                }
                out = joined;
            }
            5 => {
                let from = pick(rng, len);
                let count = pick(rng, 64).min(len - from);
                drop(out.drain(from..from + count));
            }
            6 => {
                let at = pick(rng, len + 1);
                let count = pick(rng, 64);
                let extra: Vec<u8> = (0..count)
                    .map(|_| u8::try_from(rng.next() % 256).unwrap_or(0))
                    .collect();
                drop(out.splice(at..at, extra));
            }
            7 => {
                let from = pick(rng, len);
                for byte in out.iter_mut().skip(from).take(1 + pick(rng, 32)) {
                    *byte = u8::try_from(rng.next() % 256).unwrap_or(0);
                }
            }
            _ => {
                out = (0..pick(rng, 400))
                    .map(|_| u8::try_from(rng.next() % 256).unwrap_or(0))
                    .collect();
            }
        }
    }
    if !rng.next().is_multiple_of(4) {
        reseal(&mut out);
    }
    out
}

/// Runs one reader and reports it when it is slow, so a hang can be traced to its step.
fn timed<T>(step: &str, run: impl FnOnce() -> T) -> T {
    if std::env::var_os("FUZZ_TRACE").is_some() {
        eprintln!("begin {step}");
    }
    let began = std::time::Instant::now();
    let out = run();
    if began.elapsed().as_secs() >= 2 {
        eprintln!("SLOW: {step} took {} s", began.elapsed().as_secs());
    }
    out
}

/// Feeds one input to every reader. Errors are fine; the caller catches panics.
fn exercise(model: &Model, input: &[u8], probe: u64) {
    drop(timed("read_info", || pico::read_info(input)));
    if let Ok(stream) = timed("parse", || Stream::parse(input)) {
        let _: usize = stream.playable_samples();
        let _: usize = stream.page_for_sample(probe);
        for index in 0..stream.pages.len().min(8) {
            drop(timed("decode_page", || stream.decode_page(model, index)));
        }
        drop(timed("decode_all", || stream.decode_all(model)));
    }
}

/// Files the fuzzer starts from: streams of both models, with and without a
/// length field, comments and loops.
fn fuzz_seeds(rng: &mut Lcg, model: &Model) -> Result<Vec<Vec<u8>>> {
    let streaming = Shape {
        length_in_header: false,
        ..LOOPED
    };
    let mut seeds = Vec::new();
    for (segments, shape, tags) in [
        (3, PLAIN, true),
        (2, LOOPED, true),
        (3, streaming, false),
        (
            2,
            Shape {
                page_length: 0,
                ..PLAIN
            },
            false,
        ),
    ] {
        let (mut info, frames) = stereo(rng, model, segments, shape)?;
        if tags {
            info.tags = vec!["TITLE=Test".to_owned(), "LOOPSTART=0".to_owned()];
        }
        seeds.push(pico::write(model, &info, &frames)?);
    }
    for page_length in [0, 75] {
        let (info, frames) = mono(rng, model, 120, page_length);
        seeds.push(pico::write(model, &info, &frames)?);
    }
    // Real files, when given: FUZZ_FILES is a list of paths separated by `;`.
    if let Ok(list) = std::env::var("FUZZ_FILES") {
        for path in list.split(';').filter(|p| !p.is_empty()) {
            seeds.push(std::fs::read(path)?);
        }
    }
    Ok(seeds)
}

/// The model the fuzzer decodes with: a real self-contained table from `FUZZ_TABLE`,
/// or the synthetic one. Real files in `FUZZ_FILES` need the real table.
fn fuzz_model(rng: &mut Lcg) -> Result<(Model, Vec<u8>, Vec<u8>)> {
    if let Ok(path) = std::env::var("FUZZ_TABLE") {
        let table = std::fs::read(path)?;
        return Ok((Model::from_table(&table)?, table, Vec::new()));
    }
    let (table, weights) = (table(rng), weights(rng));
    Ok((Model::load(&table, &weights)?, table, weights))
}

/// Loads a damaged table (every fourth time, damaged weights) and codes with the result.
fn exercise_table(rng: &mut Lcg, table: &[u8], weights: &[u8]) {
    let loaded = if rng.next().is_multiple_of(4) {
        Model::load(table, &mutate(rng, weights, table))
    } else {
        Model::load(&mutate(rng, table, weights), weights)
    };
    if let Ok(model) = loaded {
        let stream = codes(rng, 40, 2);
        if let Ok(chunk) = encode_chunk(&model, 2, &[7, 4000], &stream) {
            drop(decode_chunk(&model, 2, 2, 80, &chunk));
        }
    }
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Runs case `case` of the fuzzer; returns false if a reader panicked.
fn fuzz_case(model: &Model, tables: (&[u8], &[u8]), seeds: &[Vec<u8>], case: u64) -> bool {
    let mut rng = Lcg(case.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5DEE_CE66);
    if case % 64 == 63 {
        return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            exercise_table(&mut rng, tables.0, tables.1);
        }))
        .is_ok();
    }
    let count = seeds.len().max(1);
    let base = &seeds[usize::try_from(rng.next()).unwrap_or(0) % count];
    let other = &seeds[usize::try_from(rng.next()).unwrap_or(0) % count];
    let input = mutate(&mut rng, base, other);
    let probe = rng.next() << (rng.next() % 32);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        exercise(model, &input, probe);
    }))
    .is_ok()
}

/// Deterministic mutation fuzzer for every reader of untrusted bytes.
///
/// `cargo test --release --test pico fuzz -- --ignored --nocapture`, with `FUZZ_SECONDS`
/// (default 20), `FUZZ_THREADS` (default 4), `FUZZ_START` (first case number) and
/// `FUZZ_CASE_SECONDS` (default 20: a case running longer counts as a hang and ends the
/// process). A case is fully determined by its number, which is what a failure reports;
/// `FUZZ_CASE=<n>` replays one case.
#[test]
#[ignore = "runs for FUZZ_SECONDS; start it by hand"]
fn fuzz_readers_never_panic_or_hang() -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut rng = Lcg(43);
    let (model, table, weights) = fuzz_model(&mut rng)?;
    let tables = (table.as_slice(), weights.as_slice());
    let seeds = fuzz_seeds(&mut rng, &model)?;
    if let Ok(case) = std::env::var("FUZZ_CASE") {
        assert!(
            fuzz_case(&model, tables, &seeds, case.parse()?),
            "case {case} panicked"
        );
        return Ok(());
    }
    let seconds = env_number("FUZZ_SECONDS", 20);
    let threads = env_number("FUZZ_THREADS", 4).clamp(1, 64);
    let start = env_number("FUZZ_START", 0);
    let limit = env_number("FUZZ_CASE_SECONDS", 20);
    let began = std::time::Instant::now();
    // Per thread: the case in progress and when it started (milliseconds since `began`).
    let progress: Vec<(AtomicU64, AtomicU64)> = (0..threads)
        .map(|_| (AtomicU64::new(0), AtomicU64::new(0)))
        .collect();
    let done = AtomicU64::new(0);
    let finished = AtomicU64::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for (index, slot) in progress.iter().enumerate() {
            let (model, seeds, done, finished, failures) =
                (&model, &seeds, &done, &finished, &failures);
            let _handle = scope.spawn(move || {
                let mut case = start + u64::try_from(index).unwrap_or(0);
                while began.elapsed().as_secs() < seconds {
                    slot.0.store(case, Ordering::Relaxed);
                    let now = u64::try_from(began.elapsed().as_millis()).unwrap_or(u64::MAX);
                    slot.1.store(now, Ordering::Relaxed);
                    if !fuzz_case(model, tables, seeds, case)
                        && let Ok(mut list) = failures.lock()
                    {
                        list.push(case);
                    }
                    let _ = done.fetch_add(1, Ordering::Relaxed);
                    case += threads;
                }
                let _ = finished.fetch_add(1, Ordering::Relaxed);
            });
        }
        // Watchdog: a case that runs too long is a hang; name it and stop the process.
        while finished.load(Ordering::Relaxed) < threads {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let now = u64::try_from(began.elapsed().as_millis()).unwrap_or(u64::MAX);
            for slot in &progress {
                let started = slot.1.load(Ordering::Relaxed);
                if now.saturating_sub(started) > limit * 1000
                    && finished.load(Ordering::Relaxed) < threads
                {
                    eprintln!(
                        "HANG: case {} ran over {limit} s",
                        slot.0.load(Ordering::Relaxed)
                    );
                    std::process::exit(3);
                }
            }
        }
    });
    let failed = failures.lock().map(|list| list.clone()).unwrap_or_default();
    let listed = failed
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "fuzz: {} cases from {start} on {threads} threads in {} s; panics in cases {}",
        done.load(Ordering::Relaxed),
        began.elapsed().as_secs(),
        listed
    );
    assert!(failed.is_empty(), "readers panicked in cases {listed}");
    Ok(())
}

/// A damaged length must not make the reader allocate or decode without bound.
#[test]
fn a_damaged_length_is_an_error_not_a_long_decode() -> Result<()> {
    let mut rng = Lcg(45);
    let model = model(&mut rng)?;
    for length_in_header in [false, true] {
        let shape = Shape {
            length_in_header,
            ..PLAIN
        };
        let (info, frames) = stereo(&mut rng, &model, 3, shape)?;
        let mut bytes = pico::write(&model, &info, &frames)?;
        // Without a length field the last page's granule position gives the length.
        let last = page_starts(&bytes)[page_starts(&bytes).len() - 2];
        for granule in [1_u64 << 39, u64::MAX - 1, 1] {
            bytes[last + 6..last + 14].copy_from_slice(&granule.to_le_bytes());
            reseal(&mut bytes);
            let began = std::time::Instant::now();
            assert!(Stream::parse(&bytes).is_err());
            assert!(began.elapsed().as_secs() < 2);
        }
    }
    Ok(())
}

/// Found by the fuzzer: a damaged page sequence number made the reader record billions of
/// lost pages, never returning and allocating without bound.
#[test]
fn a_damaged_sequence_number_cannot_inflate_the_stream() -> Result<()> {
    let mut rng = Lcg(47);
    let model = model(&mut rng)?;
    let (info, frames) = stereo(&mut rng, &model, 3, PLAIN)?;
    let mut bytes = pico::write(&model, &info, &frames)?;
    let page = page_starts(&bytes)[2];
    bytes[page + 18..page + 22].copy_from_slice(&0xFFFF_FFF0_u32.to_le_bytes());
    reseal(&mut bytes);
    let began = std::time::Instant::now();
    // The stream may or may not be usable, but reading it ends promptly and stays small.
    if let Ok(stream) = Stream::parse(&bytes) {
        assert!(stream.pages.len() <= 2 * bytes.len() / 27);
        drop(stream.decode_page(&model, 0));
    }
    assert!(began.elapsed().as_secs() < 2);
    Ok(())
}
