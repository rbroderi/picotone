//! The HQ cx2 model against the integer emulation in `train/hq_net.py`, and HQ page round trips.

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use picotone::cx2::hq::{self, HqModel, Role};
use picotone::cx2::kernels::Simd;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model() -> Result<HqModel> {
    let path = root().join("models").join("cx2_dac44.ecxm");
    HqModel::from_table(
        &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
    )
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn role(value: u64) -> Result<Role> {
    Ok(match value {
        0 => Role::LeftRight,
        1 => Role::Mid,
        2 => Role::Side,
        other => anyhow::bail!("unknown role {other}"),
    })
}

struct Case {
    role: Role,
    codebooks: usize,
    codes: Vec<u16>,
    hashes: Vec<String>,
    bits: f64,
}

fn fixture() -> Result<(String, usize, usize, Vec<Case>)> {
    let path = root()
        .join("tests")
        .join("data")
        .join("cx2_dac44_fixture.json");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let get = |key: &str| json.get(key).context("fixture field missing");
    let frames = usize::try_from(get("frames")?.as_u64().context("frames")?)?;
    let page = usize::try_from(get("page")?.as_u64().context("page")?)?;
    let hash = get("table_hash")?
        .as_str()
        .context("table_hash")?
        .to_owned();
    let mut cases = Vec::new();
    for case in get("cases")?.as_array().context("cases")? {
        let field = |key: &str| case.get(key).context("case field missing");
        cases.push(Case {
            role: role(field("role")?.as_u64().context("role")?)?,
            codebooks: usize::try_from(field("codebooks")?.as_u64().context("codebooks")?)?,
            codes: field("codes")?
                .as_array()
                .context("codes")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|c| u16::try_from(c).ok())
                        .context("code")
                })
                .collect::<Result<_>>()?,
            hashes: field("logit_hashes")?
                .as_array()
                .context("hashes")?
                .iter()
                .map(|v| v.as_str().map(str::to_owned).context("hash"))
                .collect::<Result<_>>()?,
            bits: field("coded_bits")?.as_f64().context("coded_bits")?,
        });
    }
    Ok((hash, frames, page, cases))
}

#[test]
fn logits_match_the_python_emulation_on_every_dispatch_path() -> Result<()> {
    let (hash, frames, page, cases) = fixture()?;
    for simd in [Simd::new(), Simd::baseline()] {
        let mut model = model()?;
        ensure!(
            format!("{:016x}", model.hash()) == hash,
            "the fixture is for another table"
        );
        model.set_simd(simd);
        for case in &cases {
            let n = case.codebooks;
            let mut predictor = model.predictor(case.role, n)?;
            let mut at = 0;
            for t in 0..frames {
                if t % page == 0 {
                    predictor.reset();
                }
                for _ in 0..n {
                    let logits: Vec<u8> = predictor
                        .logits()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    assert_eq!(
                        format!("{:016x}", fnv1a(&logits)),
                        case.hashes[at],
                        "frame {t}, codebook {}",
                        predictor.codebook()
                    );
                    predictor.update(case.codes[at])?;
                    at += 1;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn pages_round_trip_at_the_fixture_size() -> Result<()> {
    let (_, frames, page, cases) = fixture()?;
    let model = model()?;
    for case in &cases {
        let n = case.codebooks;
        let mut bytes = 0;
        for start in (0..frames).step_by(page) {
            let end = (start + page).min(frames);
            let codes = &case.codes[start * n..end * n];
            let packed = hq::encode_page(&model, case.role, n, codes)?;
            bytes += packed.len();
            assert_eq!(
                hq::decode_page(&model, case.role, n, codes.len(), &packed)?,
                codes
            );
        }
        // The range coder lands within a few bytes per page of the model's bits
        // (its flush can trim the last bytes, so a little under is possible too).
        let bits = cast(bytes) * 8.0;
        let pages = cast(frames.div_ceil(page));
        assert!(
            bits >= case.bits - 16.0 * pages && bits <= case.bits + 64.0 * pages,
            "{bits} vs {}",
            case.bits
        );
    }
    Ok(())
}

#[test]
fn truncated_pages_are_refused() -> Result<()> {
    let (_, _, page, cases) = fixture()?;
    let model = model()?;
    let case = &cases[0];
    let n = case.codebooks;
    let codes = &case.codes[..page * n];
    let packed = hq::encode_page(&model, case.role, n, codes)?;
    let short = &packed[..packed.len() / 2];
    assert!(hq::decode_page(&model, case.role, n, codes.len(), short).is_err());
    Ok(())
}

/// Where an HQ decode's time goes: `cargo test --release --test hq_cx2 -- --ignored --nocapture`.
/// Every figure is the best of several runs, so other load on the machine matters less.
#[test]
#[ignore = "a timing report, not a check"]
fn profile_decode() -> Result<()> {
    use picotone::cx2::kernels::Matrix;
    use std::time::Instant;
    let best = |runs: usize, f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        let mut out = f64::INFINITY;
        for _ in 0..runs {
            let t = Instant::now();
            f()?;
            out = out.min(t.elapsed().as_secs_f64());
        }
        Ok(out)
    };
    let (_, frames, _, cases) = fixture()?;
    let model = model()?;
    for case in &cases {
        let n = case.codebooks;
        let page = &case.codes[..80.min(frames) * n];
        let packed = hq::encode_page(&model, case.role, n, page)?;
        let page_frames = cast(page.len() / n);
        let decode = best(9, &mut || {
            drop(hq::decode_page(&model, case.role, n, page.len(), &packed)?);
            Ok(())
        })? / page_frames;
        // The predictor without the range coder or the frequency step.
        let mut predictor = model.predictor(case.role, n)?;
        let logits = best(9, &mut || {
            predictor.reset();
            for &code in page {
                let _ = std::hint::black_box(predictor.logits());
                predictor.update(code)?;
            }
            Ok(())
        })? / page_frames;
        eprintln!(
            "q{n}: decode {:.1} us per frame ({:.4} s per channel-second); logits alone {:.1} us",
            decode * 1e6,
            decode * 86.13,
            logits * 1e6
        );
    }
    // Kernel shapes, with half the inputs zero as after a ReLU.
    for (rows, columns, per_frame) in [
        (512, 1152, 1),
        (256, 512, 1),
        (128, 256 + 64, 9),
        (1024, 128, 9),
    ] {
        let weights: Vec<i16> = (0..rows * columns)
            .map(|i| i16::try_from(i * 7919 % 601).unwrap_or(0) - 300)
            .collect();
        let matrix = Matrix::from_columns(&weights, rows).context("shape")?;
        let input: Vec<i16> = (0..columns)
            .map(|i| {
                if i % 2 == 0 {
                    0
                } else {
                    i16::try_from(i % 1000).unwrap_or(0)
                }
            })
            .collect();
        let mut acc = vec![0_i32; rows];
        for simd in [Simd::new(), Simd::baseline()] {
            let calls = 500;
            let us = best(7, &mut || {
                for _ in 0..calls {
                    simd.multiply_add(&mut acc, &matrix, &input);
                }
                Ok(())
            })? / cast(calls)
                * 1e6;
            eprintln!(
                "{rows:5} x {columns:4} ({}): {us:7.2} us per call, x{per_frame} per q9 frame, {:.1} GMAC/s",
                if simd.is_baseline() {
                    "baseline"
                } else {
                    "dispatch"
                },
                cast(rows * columns) / us / 1e3
            );
        }
        let _ = std::hint::black_box(&acc);
    }
    Ok(())
}

#[expect(clippy::cast_precision_loss, reason = "byte counts are small")]
const fn cast(value: usize) -> f64 {
    value as f64
}
