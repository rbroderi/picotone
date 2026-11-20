//! Model hyper-parameters, read from the Hugging Face `config.json` that ships
//! with each checkpoint.

use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;

/// How convolution weights are normalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    /// Weight normalisation (`weight_g` / `weight_v`), used by the 24 kHz model.
    WeightNorm,
    /// Plain convolution followed by a one-group `GroupNorm`, used by the 48 kHz model.
    TimeGroupNorm,
}

/// Padding mode applied before each convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PadMode {
    /// Mirror the signal around its edges, excluding the edge sample.
    Reflect,
    /// Pad with zeros.
    #[serde(alias = "zero")]
    Constant,
    /// Repeat the edge sample.
    Replicate,
}

/// EnCodec hyper-parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    /// Number of audio channels the model consumes and produces.
    pub audio_channels: usize,
    /// Length of each independently coded segment, in seconds.
    pub chunk_length_s: Option<f64>,
    /// Codebook vector size; defaults to `hidden_size`.
    pub codebook_dim: Option<usize>,
    /// Number of entries in every codebook.
    pub codebook_size: usize,
    /// Channel reduction inside residual blocks.
    pub compress: usize,
    /// Dilation growth across residual layers.
    pub dilation_growth_rate: usize,
    /// Latent dimension.
    pub hidden_size: usize,
    /// Kernel size of the first convolution.
    pub kernel_size: usize,
    /// Kernel size of the last convolution.
    pub last_kernel_size: usize,
    /// Convolution normalisation.
    pub norm_type: NormType,
    /// Whether each segment is volume-normalised before encoding.
    pub normalize: bool,
    /// Base channel count.
    pub num_filters: usize,
    /// LSTM layer count.
    pub num_lstm_layers: usize,
    /// Residual blocks per resolution.
    pub num_residual_layers: usize,
    /// Fractional overlap between segments.
    pub overlap: Option<f64>,
    /// Convolution padding mode.
    pub pad_mode: PadMode,
    /// Kernel size inside residual blocks.
    pub residual_kernel_size: usize,
    /// Sample rate in Hz.
    pub sampling_rate: usize,
    /// Supported bandwidths in kbps.
    pub target_bandwidths: Vec<f64>,
    /// Share of transposed-convolution padding trimmed from the right (causal models).
    pub trim_right_ratio: f64,
    /// Decoder up-sampling ratios (the encoder uses them reversed).
    pub upsampling_ratios: Vec<usize>,
    /// Whether convolutions are causal.
    pub use_causal_conv: bool,
    /// Whether residual blocks use a 1x1 convolution shortcut.
    #[serde(default = "default_true")]
    pub use_conv_shortcut: bool,
}

const fn default_true() -> bool {
    true
}

impl Config {
    /// Reads and validates a Hugging Face `config.json`.
    ///
    /// # Errors
    /// Fails when the file cannot be read, parsed or describes an unsupported model.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Parses and validates the text of a Hugging Face `config.json`.
    ///
    /// # Errors
    /// Fails when the text cannot be parsed or describes an unsupported model.
    pub fn from_json(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text).context("parsing the model configuration")?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.audio_channels > 0, "audio_channels must be positive");
        ensure!(self.sampling_rate > 0, "sampling_rate must be positive");
        ensure!(
            !self.upsampling_ratios.is_empty(),
            "upsampling_ratios is empty"
        );
        ensure!(
            self.upsampling_ratios.iter().all(|&r| r > 0),
            "upsampling ratios must be positive"
        );
        ensure!(self.compress > 0, "compress must be positive");
        ensure!(
            self.codebook_size.is_power_of_two() && self.codebook_size <= 1 << 16,
            "codebook_size must be a power of two no larger than 65536"
        );
        ensure!(
            !self.target_bandwidths.is_empty(),
            "target_bandwidths is empty"
        );
        if let Some(overlap) = self.overlap {
            ensure!((0.0..1.0).contains(&overlap), "overlap must be in [0, 1)");
        }
        if let Some(chunk) = self.chunk_length_s {
            ensure!(chunk > 0.0, "chunk_length_s must be positive");
        }
        Ok(())
    }

    /// Size of a codebook vector.
    #[must_use]
    pub fn codebook_dim(&self) -> usize {
        self.codebook_dim.unwrap_or(self.hidden_size)
    }

    /// Samples per latent frame.
    #[must_use]
    pub fn hop_length(&self) -> usize {
        self.upsampling_ratios.iter().product()
    }

    /// Latent frames per second.
    #[must_use]
    pub fn frame_rate(&self) -> usize {
        self.sampling_rate.div_ceil(self.hop_length())
    }

    /// Bits needed to store one code.
    #[must_use]
    pub const fn bits_per_codebook(&self) -> u32 {
        self.codebook_size.trailing_zeros()
    }

    /// Number of codebooks the full model holds.
    ///
    /// # Errors
    /// Fails when the bandwidth list is empty.
    pub fn total_codebooks(&self) -> Result<usize> {
        let max = self
            .target_bandwidths
            .iter()
            .copied()
            .reduce(f64::max)
            .context("target_bandwidths is empty")?;
        self.codebooks_for_bandwidth(max)
    }

    /// Number of codebooks used at `bandwidth_kbps`.
    ///
    /// # Errors
    /// Fails when the bandwidth is not one the model supports.
    pub fn codebooks_for_bandwidth(&self, bandwidth_kbps: f64) -> Result<usize> {
        if !self
            .target_bandwidths
            .iter()
            .any(|&b| (b - bandwidth_kbps).abs() < 1e-9)
        {
            bail!(
                "bandwidth {bandwidth_kbps} kbps is not supported; choose one of {}",
                self.bandwidth_list()
            );
        }
        let bits_per_second_per_codebook =
            u64::from(self.bits_per_codebook()) * to_u64(self.frame_rate());
        let bits_per_second = round_to_u64(bandwidth_kbps * 1000.0)?;
        Ok(usize::try_from(bits_per_second / bits_per_second_per_codebook)?.max(1))
    }

    /// Bandwidths supported by this model, as display text.
    #[must_use]
    pub fn bandwidth_list(&self) -> String {
        self.target_bandwidths
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Length of each coded segment in samples, or `None` for whole-file coding.
    ///
    /// # Errors
    /// Fails when the segment length does not fit in `usize`.
    pub fn segment_length(&self) -> Result<Option<usize>> {
        self.chunk_length_s
            .map(|seconds| floor_to_usize(seconds * to_f64(self.sampling_rate)))
            .transpose()
    }

    /// Distance between consecutive segment starts, or `None` for whole-file coding.
    ///
    /// # Errors
    /// Fails when the stride does not fit in `usize`.
    pub fn segment_stride(&self) -> Result<Option<usize>> {
        let Some(length) = self.segment_length()? else {
            return Ok(None);
        };
        let overlap = self.overlap.unwrap_or(0.0);
        Ok(Some(
            floor_to_usize((1.0 - overlap) * to_f64(length))?.max(1),
        ))
    }
}

/// Converts a count to `f64`; counts here stay far below 2^52.
#[expect(
    clippy::cast_precision_loss,
    reason = "audio sample counts are far below 2^52"
)]
#[must_use]
pub const fn to_f64(value: usize) -> f64 {
    value as f64
}

fn to_u64(value: usize) -> u64 {
    // usize is at most 64 bits on every supported target.
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Floors a non-negative finite `f64` into `usize`.
///
/// # Errors
/// Fails when the value is negative, not finite or too large.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "range is checked before the cast"
)]
pub fn floor_to_usize(value: f64) -> Result<usize> {
    ensure!(
        (0.0..9.0e15).contains(&value),
        "value {value} is out of range"
    );
    Ok(value.floor() as usize)
}

/// Rounds a non-negative finite `f64` to `u64`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "range is checked before the cast"
)]
fn round_to_u64(value: f64) -> Result<u64> {
    ensure!(
        (0.0..9.0e15).contains(&value),
        "value {value} is out of range"
    );
    Ok(value.round() as u64)
}
