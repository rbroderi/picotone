//! The 12-bit volume scale of a segment.
//!
//! The 48 kHz model codes each segment at unit loudness and stores the
//! segment's RMS level beside its codes. picotone stores that level as a
//! 12-bit index on a logarithmic grid with 128 steps per octave (a step is
//! about 0.047 dB, at most 0.27 % off the true level), covering `2^-28`
//! (below the `1e-8` floor of silence) to just under `2^4`.
//!
//! The encoder quantises the level before it normalises the segment and
//! divides by the quantised value, so the decoder's multiplication undoes
//! exactly what the encoder did. [`dequantize`] is a table lookup times an
//! exact power of two, so every platform turns an index into the same `f32`.

/// Number of scale indices.
pub const LEVELS: u16 = 1 << BITS;
/// Bits per stored scale.
pub const BITS: u32 = 12;
/// Grid steps per octave.
const STEPS: i32 = 128;
/// Index of the level 1.0.
const UNITY: i32 = 28 * STEPS;

/// `2^(i / 128)` for `i` in `0..128`, as `f32` bit patterns (correctly rounded).
const FRACTIONS: [u32; 128] = [
    0x3f80_0000,
    0x3f80_b1ed,
    0x3f81_64d2,
    0x3f82_18af,
    0x3f82_cd87,
    0x3f83_8359,
    0x3f84_3a29,
    0x3f84_f1f6,
    0x3f85_aac3,
    0x3f86_6491,
    0x3f87_1f62,
    0x3f87_db35,
    0x3f88_980f,
    0x3f89_55ee,
    0x3f8a_14d5,
    0x3f8a_d4c6,
    0x3f8b_95c2,
    0x3f8c_57ca,
    0x3f8d_1adf,
    0x3f8d_df04,
    0x3f8e_a43a,
    0x3f8f_6a81,
    0x3f90_31dc,
    0x3f90_fa4d,
    0x3f91_c3d3,
    0x3f92_8e72,
    0x3f93_5a2b,
    0x3f94_26ff,
    0x3f94_f4f0,
    0x3f95_c3ff,
    0x3f96_942d,
    0x3f97_657d,
    0x3f98_37f0,
    0x3f99_0b88,
    0x3f99_e046,
    0x3f9a_b62b,
    0x3f9b_8d3a,
    0x3f9c_6573,
    0x3f9d_3eda,
    0x3f9e_196e,
    0x3f9e_f532,
    0x3f9f_d228,
    0x3fa0_b051,
    0x3fa1_8faf,
    0x3fa2_7043,
    0x3fa3_520f,
    0x3fa4_3516,
    0x3fa5_1958,
    0x3fa5_fed7,
    0x3fa6_e595,
    0x3fa7_cd94,
    0x3fa8_b6d5,
    0x3fa9_a15b,
    0x3faa_8d26,
    0x3fab_7a3a,
    0x3fac_6897,
    0x3fad_583f,
    0x3fae_4934,
    0x3faf_3b79,
    0x3fb0_2f0e,
    0x3fb1_23f6,
    0x3fb2_1a32,
    0x3fb3_11c4,
    0x3fb4_0aaf,
    0x3fb5_04f3,
    0x3fb6_0094,
    0x3fb6_fd92,
    0x3fb7_fbf0,
    0x3fb8_fbaf,
    0x3fb9_fcd2,
    0x3fba_ff5b,
    0x3fbc_034a,
    0x3fbd_08a4,
    0x3fbe_0f68,
    0x3fbf_179a,
    0x3fc0_213b,
    0x3fc1_2c4d,
    0x3fc2_38d2,
    0x3fc3_46cd,
    0x3fc4_563f,
    0x3fc5_672a,
    0x3fc6_7991,
    0x3fc7_8d75,
    0x3fc8_a2d8,
    0x3fc9_b9be,
    0x3fca_d226,
    0x3fcb_ec15,
    0x3fcd_078c,
    0x3fce_248c,
    0x3fcf_4319,
    0x3fd0_6334,
    0x3fd1_84df,
    0x3fd2_a81e,
    0x3fd3_ccf1,
    0x3fd4_f35b,
    0x3fd6_1b5e,
    0x3fd7_44fd,
    0x3fd8_7039,
    0x3fd9_9d16,
    0x3fda_cb94,
    0x3fdb_fbb8,
    0x3fdd_2d82,
    0x3fde_60f5,
    0x3fdf_9613,
    0x3fe0_ccdf,
    0x3fe2_055b,
    0x3fe3_3f89,
    0x3fe4_7b6d,
    0x3fe5_b907,
    0x3fe6_f85b,
    0x3fe8_396a,
    0x3fe9_7c38,
    0x3fea_c0c7,
    0x3fec_0719,
    0x3fed_4f30,
    0x3fee_9910,
    0x3fef_e4ba,
    0x3ff1_3231,
    0x3ff2_8177,
    0x3ff3_d290,
    0x3ff5_257d,
    0x3ff6_7a41,
    0x3ff7_d0df,
    0x3ff9_295a,
    0x3ffa_83b3,
    0x3ffb_dfed,
    0x3ffd_3e0c,
    0x3ffe_9e11,
];

/// The level a stored index stands for.
///
/// Indices past the grid are clamped to its top.
#[must_use]
pub fn dequantize(index: u16) -> f32 {
    let step = i32::from(index.min(LEVELS - 1)) - UNITY;
    let octave = step.div_euclid(STEPS);
    let fraction = FRACTIONS[usize::try_from(step.rem_euclid(STEPS)).unwrap_or(0)];
    // 2^octave, exactly: octave is in -28..=3, far from the subnormals.
    let power = u32::try_from(127 + octave).unwrap_or(127) << 23;
    f32::from_bits(fraction) * f32::from_bits(power)
}

/// The index of the grid level nearest to `level` (in octaves), clamped to the grid.
#[must_use]
pub fn quantize(level: f32) -> u16 {
    if level.is_nan() || level <= 0.0 {
        return 0;
    }
    let position = f64::from(STEPS).mul_add(f64::from(level).log2(), f64::from(UNITY));
    let top = f64::from(LEVELS - 1);
    // The clamp keeps the value in 0..=4095, so the conversion is exact.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to 0..=4095 and already rounded"
    )]
    let index = position.round().clamp(0.0, top) as u16;
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_survives_a_round_trip() {
        for index in 0..LEVELS {
            assert_eq!(quantize(dequantize(index)), index);
        }
    }

    #[test]
    fn the_grid_is_logarithmic() {
        assert!((dequantize(3584) - 1.0).abs() < f32::EPSILON);
        assert!((dequantize(3584 + 128) - 2.0).abs() < f32::EPSILON);
        assert!((dequantize(3584 - 128 * 10) - 1.0 / 1024.0).abs() < 1e-12);
        assert!(dequantize(0) < 1e-8);
        assert!(dequantize(LEVELS - 1) > 15.0);
        for level in [1e-8_f32, 3e-4, 0.1, 0.5, 0.999, 1.0, 3.7] {
            let back = dequantize(quantize(level));
            assert!((back / level - 1.0).abs() < 0.003, "{level} -> {back}");
        }
        assert_eq!(quantize(0.0), 0);
        assert_eq!(quantize(f32::NAN), 0);
        assert_eq!(quantize(1e9), LEVELS - 1);
    }
}
