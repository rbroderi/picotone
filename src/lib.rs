//! picotone: the EnCodec neural audio codec with the cx2 entropy coder, in pure Rust.
//!
//! Runs Meta's two published EnCodec checkpoints from their Hugging Face
//! `safetensors` weights: `encodec_24khz` (mono, causal) and `encodec_48khz`
//! (stereo, segmented with overlap-add). Compressed audio is stored as
//! `.pico` files: the codes, entropy coded by [`cx2`], in Ogg pages (see
//! [`pico`]).
//!
//! The model itself needs neither a filesystem nor threads: build it with
//! [`model::Encodec::from_bytes`] (or [`model::Encodec::decoder_from_bytes`]
//! for a decoder-only subset of the weights) and decode incrementally with
//! [`model::Encodec::stream_decoder`] or [`model::Encodec::decode_segment`]
//! plus [`model::OverlapAdd`]. Audio file input and WAV output live in the
//! `audio` module behind the default-on `audio-io` feature; the command-line
//! program is behind `cli`.

#[cfg(feature = "audio-io")]
pub mod audio;
pub mod config;
pub mod cx2;
pub mod dac;
pub mod declick;
mod fft;
pub mod hqstream;
mod kernels;
pub mod layout;
pub mod limiter;
pub mod model;
pub mod nn;
pub mod pico;
pub mod player;
mod pool;
pub mod prof;
pub mod pstereo;
pub mod scale;
pub mod stereo;
pub mod stftenc;
pub mod vocos;
pub mod weights;

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// One of the two published EnCodec models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// `encodec_24khz`: 24 kHz mono, 1.5 to 24 kbps.
    Mono24k,
    /// `encodec_48khz`: 48 kHz stereo, 3 to 24 kbps.
    Stereo48k,
}

impl ModelKind {
    /// The model's name, also used for weight file names.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mono24k => "encodec_24khz",
            Self::Stereo48k => "encodec_48khz",
        }
    }

    /// Looks a model up by its name.
    ///
    /// # Errors
    /// Fails for unknown names.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "encodec_24khz" => Ok(Self::Mono24k),
            "encodec_48khz" => Ok(Self::Stereo48k),
            other => bail!("unknown EnCodec model {other}"),
        }
    }

    /// Paths of the configuration and weights inside `dir`.
    #[must_use]
    pub fn files(self, dir: &Path) -> (PathBuf, PathBuf) {
        (
            dir.join(format!("{}.json", self.name())),
            dir.join(format!("{}.safetensors", self.name())),
        )
    }

    /// Loads the model from `dir`.
    ///
    /// # Errors
    /// Fails when the files are missing or invalid.
    pub fn load(self, dir: &Path) -> Result<model::Encodec> {
        self.load_with(dir, model::LoadOptions::ALL)
    }

    /// Reads the model's configuration from `dir`.
    ///
    /// # Errors
    /// Fails when the file is missing or invalid.
    pub fn config(self, dir: &Path) -> Result<config::Config> {
        config::Config::from_file(&self.files(dir).0)
    }

    /// Loads the parts of the model that `options` selects from `dir`.
    ///
    /// # Errors
    /// Fails when the files are missing or invalid.
    pub fn load_with(self, dir: &Path, options: model::LoadOptions) -> Result<model::Encodec> {
        let (config, weights) = self.files(dir);
        if !weights.is_file() || !config.is_file() {
            bail!(
                "weights for {} not found in {} (expected {} and {})",
                self.name(),
                dir.display(),
                config.display(),
                weights.display()
            );
        }
        model::Encodec::load_with(self.name(), &config, &weights, options)
    }
}
