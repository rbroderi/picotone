//! The EnCodec model: SEANet encoder and decoder plus residual vector quantiser,
//! with the segmenting, volume normalisation and overlap-add of the reference
//! implementation.
//!
//! Each half of the model is a convolution stack, an LSTM and another
//! convolution stack. The LSTM runs once over all segments as a batch. Where
//! the convolutions are causal and carry no normalisation (the 24 kHz model),
//! a long signal goes through them in chunks with a few frames of left
//! context, which gives the same values as one pass but keeps the activations
//! small.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use rayon::prelude::*;

use crate::config::{Config, NormType, to_f64};
use crate::layout::{Crossfade, EdgeRamp, Segmenting, check_segmenting};
use crate::nn::{
    Conv1d, ConvGeometry, ConvKernel, ConvTranspose1d, GroupNorm1, Input, Lazy, Lstm, LstmParams,
    LstmState, Seq, matmul_into,
};
use crate::weights::Weights;

/// Latent frames per chunk when a causal stack is run in chunks.
const CHUNK_FRAMES: usize = 16;
/// Latent frames quantised per task.
const QUANTIZER_BLOCK: usize = 128;

/// Codes and scale for one coded segment.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedFrame {
    /// Volume scale applied before encoding, for normalising models.
    pub scale: Option<f32>,
    /// Codebooks per latent frame.
    pub num_codebooks: usize,
    /// Codes stored `[latent frame][codebook]`.
    pub codes: Vec<u16>,
}

impl EncodedFrame {
    /// Number of latent frames.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.codes
            .len()
            .checked_div(self.num_codebooks)
            .unwrap_or(0)
    }
}

/// A coded segment's position in the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// First sample.
    pub offset: usize,
    /// Number of samples.
    pub length: usize,
    /// Number of latent frames.
    pub frames: usize,
}

/// Splits `audio_length` samples into the model's coded segments.
///
/// # Errors
/// Fails when the configuration's segment length overflows.
pub fn segments(config: &Config, audio_length: usize) -> Result<Vec<Segment>> {
    segments_with(config, audio_length, Segmenting::STOCK)
}

/// Splits `audio_length` samples into coded segments `segmenting.stride`
/// apart (the model's stride when that is `None`).
///
/// # Errors
/// Fails when the configuration's segment length overflows or the stride
/// does not suit the model.
pub fn segments_with(
    config: &Config,
    audio_length: usize,
    segmenting: Segmenting,
) -> Result<Vec<Segment>> {
    let length = config.segment_length()?.unwrap_or(audio_length);
    let stride = stride_for(config, segmenting)?
        .unwrap_or(audio_length)
        .max(1);
    Ok((0..audio_length)
        .step_by(stride)
        .map(|offset| {
            let len = (audio_length - offset).min(length);
            Segment {
                offset,
                length: len,
                frames: (len * config.frame_rate()).div_ceil(config.sampling_rate),
            }
        })
        .collect())
}

/// The stride `segmenting` selects for this model, `None` for unsegmented models.
fn stride_for(config: &Config, segmenting: Segmenting) -> Result<Option<usize>> {
    check_segmenting(config, segmenting)?;
    Ok(match segmenting.stride {
        Some(stride) => Some(stride),
        None => config.segment_stride()?,
    })
}

/// Encoder settings beyond the audio itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeParams {
    /// Codebooks per latent frame.
    pub num_codebooks: usize,
    /// Segment stride (the cross-fade only matters to the decoder).
    pub segmenting: Segmenting,
    /// Beam width of the residual vector quantiser search; 1 is the
    /// reference's greedy search.
    pub beam: usize,
}

impl EncodeParams {
    /// The reference encoder with `num_codebooks` codebooks.
    #[must_use]
    pub const fn stock(num_codebooks: usize) -> Self {
        Self {
            num_codebooks,
            segmenting: Segmenting::STOCK,
            beam: 1,
        }
    }
}

/// Which parts of a checkpoint to build. Tensors of the parts left out are
/// neither read nor required to exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// Build the encoder, needed by [`Encodec::encode`].
    pub encoder: bool,
    /// Build the decoder, needed by [`Encodec::decode`].
    pub decoder: bool,
    /// Load only the first this-many codebooks; `None` loads all of them.
    pub codebooks: Option<usize>,
}

impl LoadOptions {
    /// Everything: encoder, decoder and all codebooks.
    pub const ALL: Self = Self {
        encoder: true,
        decoder: true,
        codebooks: None,
    };

    /// The encoder and the first `codebooks` codebooks (`None` for all).
    #[must_use]
    pub const fn encoder_only(codebooks: Option<usize>) -> Self {
        Self {
            encoder: true,
            decoder: false,
            codebooks,
        }
    }

    /// The decoder and the first `codebooks` codebooks (`None` for all).
    #[must_use]
    pub const fn decoder_only(codebooks: Option<usize>) -> Self {
        Self {
            encoder: false,
            decoder: true,
            codebooks,
        }
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::ALL
    }
}

/// A loaded EnCodec model.
#[derive(Debug)]
pub struct Encodec {
    name: String,
    config: Config,
    encoder: Option<Stack>,
    decoder: Option<Stack>,
    quantizer: Quantizer,
}

impl Encodec {
    /// Loads the configuration and `.safetensors` weights of model `name`.
    ///
    /// # Errors
    /// Fails when either file is missing or does not describe a valid model.
    pub fn load(name: &str, config_path: &Path, weights_path: &Path) -> Result<Self> {
        Self::load_with(name, config_path, weights_path, LoadOptions::ALL)
    }

    /// Loads the parts of model `name` that `options` selects; only their
    /// tensors are read from the weights file.
    ///
    /// # Errors
    /// Fails when either file is missing or does not describe a valid model.
    pub fn load_with(
        name: &str,
        config_path: &Path,
        weights_path: &Path,
        options: LoadOptions,
    ) -> Result<Self> {
        let config = Config::from_file(config_path)?;
        let weights = Weights::load(weights_path)?;
        Self::build(name, config, &weights, options)
    }

    /// Builds model `name` from the text of its `config.json` and the bytes of
    /// its `.safetensors` file, without touching the filesystem.
    ///
    /// # Errors
    /// Fails when the inputs do not describe a valid model.
    pub fn from_bytes(name: &str, config_json: &str, safetensors: &[u8]) -> Result<Self> {
        Self::from_bytes_with(name, config_json, safetensors, LoadOptions::ALL)
    }

    /// Like [`Encodec::from_bytes`], building only what `options` selects.
    ///
    /// # Errors
    /// Fails when the inputs do not describe a valid model.
    pub fn from_bytes_with(
        name: &str,
        config_json: &str,
        safetensors: &[u8],
        options: LoadOptions,
    ) -> Result<Self> {
        let config = Config::from_json(config_json)?;
        let weights = Weights::from_bytes(safetensors)?;
        Self::build(name, config, &weights, options)
    }

    /// Builds only the decoder and the codebooks of model `name` from bytes.
    /// The safetensors data may leave out every `encoder.*` tensor.
    /// [`Encodec::encode`] returns an error on the resulting model.
    ///
    /// # Errors
    /// Fails when the inputs do not describe a valid decoder.
    pub fn decoder_from_bytes(name: &str, config_json: &str, safetensors: &[u8]) -> Result<Self> {
        Self::from_bytes_with(
            name,
            config_json,
            safetensors,
            LoadOptions::decoder_only(None),
        )
    }

    fn build(name: &str, config: Config, weights: &Weights, options: LoadOptions) -> Result<Self> {
        let builder = Builder {
            w: weights,
            cfg: &config,
        };
        let encoder = options
            .encoder
            .then(|| builder.encoder().context("building the encoder"))
            .transpose()?;
        let decoder = options
            .decoder
            .then(|| builder.decoder().context("building the decoder"))
            .transpose()?;
        let quantizer = Quantizer::new(weights, &config, options.codebooks, options.encoder)
            .context("building the quantizer")?;
        Ok(Self {
            name: name.to_owned(),
            config,
            encoder,
            decoder,
            quantizer,
        })
    }

    /// Model name, e.g. `encodec_24khz`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Model hyper-parameters.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Whether the encoder was loaded.
    #[must_use]
    pub const fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// Whether the decoder was loaded.
    #[must_use]
    pub const fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    /// Encodes time-major audio `[samples, channels]` at the model's sample rate.
    ///
    /// # Errors
    /// Fails on an unsupported bandwidth, a channel-count mismatch, or when
    /// the model was loaded without its encoder.
    pub fn encode(&self, audio: &Seq, bandwidth_kbps: f64) -> Result<Vec<EncodedFrame>> {
        let num_codebooks = self.config.codebooks_for_bandwidth(bandwidth_kbps)?;
        self.encode_with(audio, &EncodeParams::stock(num_codebooks))
    }

    /// Encodes time-major audio with explicit codebook count, segment stride
    /// and quantiser beam width.
    ///
    /// # Errors
    /// Fails on a codebook count that is not loaded, a stride the model cannot
    /// use, a channel-count mismatch, or when the model was loaded without
    /// its encoder.
    pub fn encode_with(&self, audio: &Seq, params: &EncodeParams) -> Result<Vec<EncodedFrame>> {
        let encoder = self
            .encoder
            .as_ref()
            .context("this model was loaded without its encoder")?;
        ensure!(params.beam >= 1, "beam width must be at least 1");
        ensure!(
            audio.channels() == self.config.audio_channels,
            "{} expects {} channel(s), got {}",
            self.name,
            self.config.audio_channels,
            audio.channels()
        );
        ensure!(!audio.is_empty(), "cannot encode empty audio");
        let num_codebooks = params.num_codebooks;
        let segs = segments_with(&self.config, audio.len(), params.segmenting)?;

        // Convolutions before the LSTM, a few segments at a time.
        let mut scales = Vec::with_capacity(segs.len());
        let mut features = Vec::with_capacity(segs.len());
        for wave in segs.chunks(segments_in_flight()) {
            let done = wave
                .par_iter()
                .map(|seg| {
                    let mut chunk = audio.slice_rows(seg.offset, seg.offset + seg.length);
                    let scale = self.normalize(&mut chunk);
                    Ok((scale, encoder.run_pre(chunk)?))
                })
                .collect::<Result<Vec<_>>>()?;
            for (scale, feature) in done {
                scales.push(scale);
                features.push(feature);
            }
        }

        let features = encoder.lstm.forward_batch(&features)?;

        features
            .into_par_iter()
            .zip(scales)
            .zip(segs.par_iter())
            .map(|((feature, scale), seg)| {
                let embeddings = run(&encoder.post, feature)?;
                let codes = if params.beam > 1 {
                    self.quantizer
                        .encode_beam(&embeddings, num_codebooks, params.beam)?
                } else {
                    self.quantizer.encode(&embeddings, num_codebooks)?
                };
                let frame = EncodedFrame {
                    scale,
                    num_codebooks,
                    codes,
                };
                ensure!(
                    frame.frames() == seg.frames,
                    "encoder produced {} frames, expected {}",
                    frame.frames(),
                    seg.frames
                );
                Ok(frame)
            })
            .collect()
    }

    /// Scales a segment to unit RMS when the model normalises its input.
    fn normalize(&self, x: &mut Seq) -> Option<f32> {
        if !self.config.normalize {
            return None;
        }
        let channels = x.channels();
        let mean_square = x
            .data()
            .chunks_exact(channels)
            .map(|frame| {
                let mono = frame.iter().map(|&v| f64::from(v)).sum::<f64>() / to_f64(channels);
                mono * mono
            })
            .sum::<f64>()
            / to_f64(x.len());
        // The stored scale is the 12-bit level; dividing by exactly that value
        // lets the decoder's multiplication undo this step.
        let scale =
            crate::scale::dequantize(crate::scale::quantize(narrow(mean_square.sqrt()) + 1e-8));
        for v in x.data_mut() {
            *v /= scale;
        }
        Some(scale)
    }

    /// Decodes frames back to time-major audio `[audio_length, channels]`.
    ///
    /// # Errors
    /// Fails when the frames do not match the model, or when the model was
    /// loaded without its decoder.
    pub fn decode(&self, frames: &[EncodedFrame], audio_length: usize) -> Result<Seq> {
        self.decode_with(frames, audio_length, Segmenting::STOCK)
    }

    /// Decodes frames coded with `segmenting`'s stride, blending segments
    /// with its cross-fade. Returns all `audio_length` coded samples; trimming
    /// is the caller's ([`crate::layout::Trimmer`]).
    ///
    /// # Errors
    /// Fails when the frames do not match the model or the layout, or when
    /// the model was loaded without its decoder.
    pub fn decode_with(
        &self,
        frames: &[EncodedFrame],
        audio_length: usize,
        segmenting: Segmenting,
    ) -> Result<Seq> {
        let segs = segments_with(&self.config, audio_length, segmenting)?;
        ensure!(
            segs.len() == frames.len(),
            "expected {} frames for {audio_length} samples, got {}",
            segs.len(),
            frames.len()
        );
        let mut audio_parts = self.decode_segments(frames)?;

        let stride = stride_for(&self.config, segmenting)?.unwrap_or(audio_length);
        let mut audio = if audio_parts.len() == 1 {
            audio_parts.pop().context("no decoded frame")?
        } else {
            let ramp = self.edge_ramp(segmenting, stride)?;
            linear_overlap_add(&audio_parts, stride, ramp)?
        };
        ensure!(
            audio.len() >= audio_length,
            "decoded {} samples, expected at least {audio_length}",
            audio.len()
        );
        audio.trim_rows(0, audio.len() - audio_length);
        Ok(audio)
    }
}

impl Encodec {
    /// Decodes one coded segment on its own: the raw decoder output for that
    /// segment, scaled by its volume, before any overlap-add. For segmented
    /// models, feed the results to [`OverlapAdd`] in order to get the audio
    /// [`Encodec::decode`] returns.
    ///
    /// # Errors
    /// Fails when the frame does not match the model, or when the model was
    /// loaded without its decoder.
    pub fn decode_segment(&self, frame: &EncodedFrame) -> Result<Seq> {
        let decoder = self
            .decoder
            .as_ref()
            .context("this model was loaded without its decoder")?;
        let features = run(&decoder.pre, self.quantizer.decode(frame)?)?;
        let features = decoder.lstm.forward(&features)?;
        let mut audio = decoder.run_post(features)?;
        if let Some(scale) = frame.scale {
            for v in audio.data_mut() {
                *v *= scale;
            }
        }
        Ok(audio)
    }

    /// Decodes several coded segments, each as [`Encodec::decode_segment`]
    /// does and to the same values, but together: one pass over the LSTM's
    /// weights per time step serves all of them, and the worker threads
    /// share the convolutions of a few segments at a time.
    ///
    /// # Errors
    /// Fails when a frame does not match the model, or when the model was
    /// loaded without its decoder.
    pub fn decode_segments(&self, frames: &[EncodedFrame]) -> Result<Vec<Seq>> {
        let decoder = self
            .decoder
            .as_ref()
            .context("this model was loaded without its decoder")?;
        let features = frames
            .par_iter()
            .map(|frame| run(&decoder.pre, self.quantizer.decode(frame)?))
            .collect::<Result<Vec<_>>>()?;
        let mut features = decoder.lstm.forward_batch(&features)?;

        // Convolutions after the LSTM, a few segments at a time.
        let mut audio_parts = Vec::with_capacity(frames.len());
        let wave = segments_in_flight();
        for frames_wave in frames.chunks(wave) {
            let features_wave: Vec<Seq> = features.drain(..frames_wave.len()).collect();
            let done = features_wave
                .into_par_iter()
                .zip(frames_wave)
                .map(|(feature, frame)| {
                    let mut audio = decoder.run_post(feature)?;
                    if let Some(scale) = frame.scale {
                        for v in audio.data_mut() {
                            *v *= scale;
                        }
                    }
                    Ok(audio)
                })
                .collect::<Result<Vec<_>>>()?;
            audio_parts.extend(done);
        }
        Ok(audio_parts)
    }

    /// An overlap-add accumulator set up for this model's segment stride.
    ///
    /// # Errors
    /// Fails when the model codes the whole file as one segment; use
    /// [`Encodec::stream_decoder`] for those.
    pub fn overlap_add(&self) -> Result<OverlapAdd> {
        self.overlap_add_with(Segmenting::STOCK)
    }

    /// An overlap-add accumulator for files coded with `segmenting`.
    ///
    /// # Errors
    /// Fails when the model codes the whole file as one segment or the
    /// stride does not suit the model.
    pub fn overlap_add_with(&self, segmenting: Segmenting) -> Result<OverlapAdd> {
        let stride = stride_for(&self.config, segmenting)?
            .with_context(|| format!("{} does not code audio in segments", self.name))?;
        let mut acc = OverlapAdd::new(stride)?;
        acc.edge = self.edge_ramp(segmenting, stride)?;
        Ok(acc)
    }

    /// The edge-avoiding ramp when `segmenting` asks for it.
    fn edge_ramp(&self, segmenting: Segmenting, stride: usize) -> Result<Option<EdgeRamp>> {
        Ok(match segmenting.crossfade {
            Crossfade::Triangle => None,
            Crossfade::EdgeAvoiding => {
                let length = self
                    .config
                    .segment_length()?
                    .context("unsegmented models have no cross-fade")?;
                Some(EdgeRamp::new(length, stride))
            }
        })
    }

    /// Starts decoding a causal, unsegmented model (`encodec_24khz`)
    /// incrementally: codes go in a few frames at a time and audio comes out
    /// as soon as it is final. `num_codebooks` is the number of codes per
    /// latent frame.
    ///
    /// # Errors
    /// Fails for models that are not causal or that code audio in normalised
    /// segments (`encodec_48khz`; use [`Encodec::decode_segment`] there), for
    /// an unsupported codebook count, or when the decoder was not loaded.
    pub fn stream_decoder(&self, num_codebooks: usize) -> Result<StreamDecoder<'_>> {
        let decoder = self
            .decoder
            .as_ref()
            .context("this model was loaded without its decoder")?;
        let (Some(pre), Some(post)) = (decoder.pre_chunking, decoder.post_chunking) else {
            anyhow::bail!(
                "{} is not causal, so it cannot be decoded as a stream",
                self.name
            );
        };
        ensure!(
            pre.hop == 1
                && post.up
                && self.config.chunk_length_s.is_none()
                && !self.config.normalize,
            "{} codes audio in segments, so it cannot be decoded as a stream",
            self.name
        );
        ensure!(
            (1..=self.quantizer.codebooks.len()).contains(&num_codebooks),
            "{num_codebooks} codebooks requested, {} are loaded",
            self.quantizer.codebooks.len()
        );
        let channels = self.config.audio_channels;
        Ok(StreamDecoder {
            model: self,
            decoder,
            pre_halo: pre.halo,
            post_halo: post.halo,
            hop: post.hop,
            num_codebooks,
            lstm: decoder.lstm.initial_state(),
            waiting: Vec::new(),
            started: false,
            embed_tail: Seq::zeros(0, self.quantizer.dim),
            feature_tail: None,
            held: Seq::zeros(0, channels),
            emitted: 0,
        })
    }
}

/// Incremental decoder for causal models, created by
/// [`Encodec::stream_decoder`].
///
/// It carries everything the decoder needs from earlier calls: the latent
/// frames the first convolution still looks back at, the hidden and cell
/// state of every LSTM layer, and the LSTM output frames that the
/// up-sampling stack (transposed convolutions and causal convolutions) still
/// looks back at. The audio it produces is the audio [`Encodec::decode`]
/// produces, and its memory use depends on the size of the pushes, not on the
/// length of the stream.
#[derive(Debug)]
pub struct StreamDecoder<'a> {
    model: &'a Encodec,
    decoder: &'a Stack,
    /// Latent frames of history the convolution before the LSTM needs.
    pre_halo: usize,
    /// LSTM output frames of history the stack after the LSTM needs.
    post_halo: usize,
    /// Audio samples per latent frame.
    hop: usize,
    num_codebooks: usize,
    lstm: LstmState,
    /// Codes not decoded yet. The start of the signal is padded by
    /// reflection, so the first frames are only final once enough frames
    /// have arrived for that reflection.
    waiting: Vec<u16>,
    started: bool,
    /// The last `pre_halo` latent frames.
    embed_tail: Seq,
    /// The last `post_halo` LSTM output frames.
    feature_tail: Option<Seq>,
    /// Decoded samples not returned yet: the stream's last latent frame may
    /// reach past the end of the audio, so one frame is held back.
    held: Seq,
    /// Samples returned so far.
    emitted: usize,
}

impl StreamDecoder<'_> {
    /// Feeds the codes of the next latent frames, frame-major
    /// (`[frames][codebooks]`), and returns the audio that became final.
    /// The result may be empty, in particular for the first few frames.
    ///
    /// # Errors
    /// Fails when `codes` is not a whole number of frames or holds a code
    /// that is out of range.
    pub fn push(&mut self, codes: &[u16]) -> Result<Seq> {
        ensure!(
            codes.len().is_multiple_of(self.num_codebooks),
            "{} codes are not a whole number of frames of {} codebooks",
            codes.len(),
            self.num_codebooks
        );
        self.waiting.extend_from_slice(codes);
        let first_chunk = self.pre_halo.max(self.post_halo) + 1;
        if !self.started && self.waiting.len() / self.num_codebooks < first_chunk {
            return Ok(Seq::zeros(0, self.held.channels()));
        }
        let mut audio = self.advance()?;
        let keep = audio.len().min(self.hop);
        self.held = audio.slice_rows(audio.len() - keep, audio.len());
        audio.trim_rows(0, keep);
        self.emitted += audio.len();
        Ok(audio)
    }

    /// Decodes whatever is still pending and returns the last samples,
    /// trimmed so that the whole stream is `audio_length` samples long.
    ///
    /// # Errors
    /// Fails when `audio_length` does not match the frames that were pushed.
    pub fn finish(mut self, audio_length: usize) -> Result<Seq> {
        let mut audio = self.advance()?;
        let total = self.emitted + audio.len();
        ensure!(
            (self.emitted..=total).contains(&audio_length),
            "the stream decoded {total} samples, which does not fit an audio length of {audio_length}"
        );
        audio.trim_rows(0, total - audio_length);
        Ok(audio)
    }

    /// Decodes the waiting codes and returns the held samples followed by
    /// the new ones.
    fn advance(&mut self) -> Result<Seq> {
        let codes = std::mem::take(&mut self.waiting);
        let held = std::mem::replace(&mut self.held, Seq::zeros(0, 0));
        if codes.is_empty() {
            return Ok(held);
        }
        self.started = true;
        let decoder = self.decoder;

        // Convolution before the LSTM, with its history in front.
        let embeddings = self.model.quantizer.embed(&codes, self.num_codebooks)?;
        let history = self.embed_tail.len();
        let tail = std::mem::replace(&mut self.embed_tail, Seq::zeros(0, 0));
        let input = Seq::concat(vec![tail, embeddings])?;
        self.embed_tail = input.slice_rows(input.len().saturating_sub(self.pre_halo), input.len());
        let mut features = run(&decoder.pre, input)?;
        features.trim_rows(history, 0);

        let features = decoder.lstm.forward_stream(&mut self.lstm, &features)?;

        // Up-sampling stack, again with its history in front.
        let (history, input) = match self.feature_tail.take() {
            Some(tail) => (tail.len(), Seq::concat(vec![tail, features])?),
            None => (0, features),
        };
        self.feature_tail =
            Some(input.slice_rows(input.len().saturating_sub(self.post_halo), input.len()));
        let mut audio = decoder.run_post(input)?;
        audio.trim_rows(history * self.hop, 0);
        Seq::concat(vec![held, audio])
    }
}

/// Incremental triangle-weighted overlap-add of decoded segments, the
/// streaming form of what [`Encodec::decode`] does for segmented models.
///
/// Segments are pushed in order; each push returns the samples no later
/// segment can change. Only the overlapping tail is kept in memory.
#[derive(Debug, Clone)]
pub struct OverlapAdd {
    stride: usize,
    /// Length of the first segment, which fixes the weights.
    frame_length: usize,
    channels: usize,
    /// Segments pushed so far.
    segments: usize,
    /// Samples returned so far.
    emitted: usize,
    /// Set once a segment shorter than the stride has been pushed.
    closed: bool,
    /// Weighted sums for the samples from `emitted` on, `[sample][channel]`.
    sums: Vec<f64>,
    /// Sum of weights for the samples from `emitted` on.
    weights: Vec<f64>,
    /// The edge-avoiding cross-fade, or `None` for Meta's triangle.
    edge: Option<EdgeRamp>,
    /// Edge-avoiding only: unweighted sums and segment counts, for samples
    /// whose only segment has weight zero.
    plain: Vec<f64>,
    counts: Vec<u32>,
}

impl OverlapAdd {
    /// An accumulator for segments that start `stride` samples apart,
    /// blended with Meta's triangle weights.
    ///
    /// # Errors
    /// Fails when `stride` is zero.
    pub fn new(stride: usize) -> Result<Self> {
        ensure!(stride > 0, "overlap-add stride is zero");
        Ok(Self {
            stride,
            frame_length: 0,
            channels: 0,
            segments: 0,
            emitted: 0,
            closed: false,
            sums: Vec::new(),
            weights: Vec::new(),
            edge: None,
            plain: Vec::new(),
            counts: Vec::new(),
        })
    }

    /// An accumulator for segments of `length` samples (the model's segment
    /// length) that start `stride` samples apart, blended with `crossfade`.
    ///
    /// # Errors
    /// Fails when `stride` is zero, or for the edge-avoiding cross-fade when
    /// more than two segments would overlap.
    pub fn with_crossfade(length: usize, stride: usize, crossfade: Crossfade) -> Result<Self> {
        let mut acc = Self::new(stride)?;
        if crossfade == Crossfade::EdgeAvoiding {
            ensure!(
                2 * stride >= length,
                "the edge-avoiding cross-fade needs a stride of at least half a segment"
            );
            acc.edge = Some(EdgeRamp::new(length, stride));
        }
        Ok(acc)
    }

    /// Adds the next segment and returns the samples that are now final:
    /// those before the start of the following segment.
    ///
    /// # Errors
    /// Fails when the segment does not fit the ones before it: a different
    /// channel count, longer than the first segment, or pushed after a
    /// segment shorter than the stride (which can only be the last one).
    pub fn push(&mut self, segment: &Seq) -> Result<Seq> {
        if self.segments == 0 {
            self.frame_length = segment.len();
            self.channels = segment.channels();
        }
        let channels = self.channels;
        ensure!(segment.channels() == channels, "segment channel mismatch");
        ensure!(
            segment.len() <= self.frame_length,
            "segment longer than the first"
        );
        ensure!(!self.closed, "a segment was pushed after the last one");
        let offset = self.segments * self.stride - self.emitted;
        let covered = offset + segment.len();
        if self.weights.len() < covered {
            self.weights.resize(covered, 0.0);
            self.sums.resize(covered * channels, 0.0);
            if self.edge.is_some() {
                self.plain.resize(covered * channels, 0.0);
                self.counts.resize(covered, 0);
            }
        }
        // torch.linspace(0, 1, frame_length + 2)[1:-1]
        let steps = to_f64(self.frame_length + 1);
        let first = self.segments == 0;
        for (t, row) in segment.data().chunks_exact(channels.max(1)).enumerate() {
            let w = match &self.edge {
                None => 0.5 - (to_f64(t + 1) / steps - 0.5).abs(),
                Some(ramp) => ramp.segment_weight(t, self.stride, first),
            };
            self.weights[offset + t] += w;
            let span = (offset + t) * channels..(offset + t + 1) * channels;
            for (sum, &v) in self.sums[span.clone()].iter_mut().zip(row) {
                *sum += w * f64::from(v);
            }
            if self.edge.is_some() {
                self.counts[offset + t] += 1;
                for (sum, &v) in self.plain[span].iter_mut().zip(row) {
                    *sum += f64::from(v);
                }
            }
        }
        self.segments += 1;
        if segment.len() < self.stride {
            self.closed = true;
            return Ok(Seq::zeros(0, channels));
        }
        self.emit(self.segments * self.stride - self.emitted)
    }

    /// Returns the remaining samples, trimmed so that everything returned
    /// adds up to `audio_length` samples.
    ///
    /// # Errors
    /// Fails when `audio_length` does not match the segments that were pushed.
    pub fn finish(mut self, audio_length: usize) -> Result<Seq> {
        let total = self.emitted + self.weights.len();
        ensure!(
            (self.emitted..=total).contains(&audio_length),
            "the segments cover {total} samples, which does not fit an audio length of {audio_length}"
        );
        self.emit(audio_length - self.emitted)
    }

    /// Removes the first `count` pending samples and returns them.
    fn emit(&mut self, count: usize) -> Result<Seq> {
        let channels = self.channels;
        let data = if self.edge.is_none() {
            ensure!(
                self.weights[..count].iter().all(|&w| w > 0.0),
                "overlap-add left a gap"
            );
            self.sums[..count * channels]
                .chunks_exact(channels.max(1))
                .zip(&self.weights)
                .flat_map(|(row, &w)| row.iter().map(move |&sum| narrow(sum / w)))
                .collect()
        } else {
            ensure!(
                self.counts[..count].iter().all(|&n| n > 0),
                "overlap-add left a gap"
            );
            let mut data = Vec::with_capacity(count * channels);
            for (t, (&w, &n)) in self.weights[..count].iter().zip(&self.counts).enumerate() {
                let span = t * channels..(t + 1) * channels;
                if w > 0.0 {
                    data.extend(self.sums[span].iter().map(|&sum| narrow(sum / w)));
                } else {
                    let n = f64::from(n);
                    data.extend(self.plain[span].iter().map(|&sum| narrow(sum / n)));
                }
            }
            drop(self.plain.drain(..count * channels));
            drop(self.counts.drain(..count));
            data
        };
        drop(self.sums.drain(..count * channels));
        drop(self.weights.drain(..count));
        self.emitted += count;
        Seq::from_vec(count, channels, data)
    }
}

/// How many segments go through the convolution stacks at once. Each one in
/// flight holds its own activations, and the convolutions inside a segment are
/// parallel too, so a small number keeps every core busy at a fraction of the
/// memory.
fn segments_in_flight() -> usize {
    rayon::current_num_threads().div_ceil(4).max(1)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RMS of audio is well inside f32 range"
)]
const fn narrow(v: f64) -> f32 {
    v as f32
}

/// Triangle-weighted overlap-add of decoded segments (`_linear_overlap_add`).
///
/// Every output sample is computed on its own from the one or few segments
/// that cover it, so no full-length `f64` accumulators are needed.
fn linear_overlap_add(frames: &[Seq], stride: usize, edge: Option<EdgeRamp>) -> Result<Seq> {
    if let Some(ramp) = edge {
        return edge_overlap_add(frames, stride, ramp);
    }
    let first = frames.first().context("no frames to overlap")?;
    let last = frames.last().context("no frames to overlap")?;
    ensure!(stride > 0, "overlap-add stride is zero");
    let channels = first.channels();
    let total = stride * (frames.len() - 1) + last.len();
    let frame_length = first.len();
    for frame in frames {
        ensure!(frame.channels() == channels, "frame channel mismatch");
        ensure!(frame.len() <= frame_length, "frame longer than the first");
    }
    // torch.linspace(0, 1, frame_length + 2)[1:-1]
    let steps = to_f64(frame_length + 1);
    let weight = |t: usize| 0.5 - (to_f64(t + 1) / steps - 0.5).abs();
    let _prof = crate::prof::scope("overlap-add", channels, 0, total);
    let mut out = Seq::zeros(total, channels);
    out.data_mut()
        .par_chunks_mut(channels)
        .with_min_len(OVERLAP_BLOCK)
        .enumerate()
        .try_for_each(|(p, row)| -> Result<()> {
            // Segments `lo..=hi` start at or before sample `p` and may reach it.
            let hi = (p / stride).min(frames.len() - 1);
            let lo = (p + 1).saturating_sub(frame_length).div_ceil(stride);
            let mut sum_weight = 0.0_f64;
            let mut acc = [0.0_f64; MAX_OVERLAP_CHANNELS];
            let acc = acc
                .get_mut(..channels)
                .context("too many channels to overlap-add")?;
            for (index, frame) in frames.iter().enumerate().take(hi + 1).skip(lo) {
                let t = p - index * stride;
                if t < frame.len() {
                    let w = weight(t);
                    sum_weight += w;
                    for (a, &src) in acc.iter_mut().zip(frame.row(t)) {
                        *a += w * f64::from(src);
                    }
                }
            }
            ensure!(sum_weight > 0.0, "overlap-add left a gap");
            for (dst, &a) in row.iter_mut().zip(acc.iter()) {
                *dst = narrow(a / sum_weight);
            }
            Ok(())
        })?;
    Ok(out)
}

/// Overlap-add with the edge-avoiding cross-fade: the same sums as
/// [`OverlapAdd`] forms, sample by sample. A sample whose only covering
/// segment has weight zero takes that segment's value.
fn edge_overlap_add(frames: &[Seq], stride: usize, ramp: EdgeRamp) -> Result<Seq> {
    let first = frames.first().context("no frames to overlap")?;
    let last = frames.last().context("no frames to overlap")?;
    ensure!(stride > 0, "overlap-add stride is zero");
    let channels = first.channels();
    let total = stride * (frames.len() - 1) + last.len();
    let frame_length = first.len();
    for frame in frames {
        ensure!(frame.channels() == channels, "frame channel mismatch");
        ensure!(frame.len() <= frame_length, "frame longer than the first");
    }
    let _prof = crate::prof::scope("overlap-add", channels, 1, total);
    let mut out = Seq::zeros(total, channels);
    out.data_mut()
        .par_chunks_mut(channels)
        .with_min_len(OVERLAP_BLOCK)
        .enumerate()
        .try_for_each(|(p, row)| -> Result<()> {
            let hi = (p / stride).min(frames.len() - 1);
            let lo = (p + 1).saturating_sub(frame_length).div_ceil(stride);
            let mut sum_weight = 0.0_f64;
            let mut count = 0_u32;
            let mut acc = [0.0_f64; MAX_OVERLAP_CHANNELS];
            let mut plain = [0.0_f64; MAX_OVERLAP_CHANNELS];
            let acc = acc
                .get_mut(..channels)
                .context("too many channels to overlap-add")?;
            let plain = plain
                .get_mut(..channels)
                .context("too many channels to overlap-add")?;
            for (index, frame) in frames.iter().enumerate().take(hi + 1).skip(lo) {
                let t = p - index * stride;
                if t < frame.len() {
                    let w = ramp.segment_weight(t, stride, index == 0);
                    sum_weight += w;
                    count += 1;
                    for ((a, q), &src) in acc.iter_mut().zip(plain.iter_mut()).zip(frame.row(t)) {
                        *a += w * f64::from(src);
                        *q += f64::from(src);
                    }
                }
            }
            ensure!(count > 0, "overlap-add left a gap");
            if sum_weight > 0.0 {
                for (dst, &a) in row.iter_mut().zip(acc.iter()) {
                    *dst = narrow(a / sum_weight);
                }
            } else {
                let n = f64::from(count);
                for (dst, &q) in row.iter_mut().zip(plain.iter()) {
                    *dst = narrow(q / n);
                }
            }
            Ok(())
        })?;
    Ok(out)
}

/// Most audio channels the overlap-add supports.
const MAX_OVERLAP_CHANNELS: usize = 8;
/// Samples the overlap-add hands to a worker thread at a time, at least.
const OVERLAP_BLOCK: usize = 4096;

#[derive(Debug, Clone)]
enum Block {
    Conv(Box<Conv1d>),
    ConvTranspose(Box<ConvTranspose1d>),
    Resnet(Box<Resnet>),
    Elu,
}

/// A residual block: `conv1x1(elu(conv(elu(x)))) + shortcut(x)`.
#[derive(Debug, Clone)]
struct Resnet {
    conv: Box<Conv1d>,
    tail: Tail,
}

/// The second half of a residual block.
#[derive(Debug, Clone)]
enum Tail {
    /// The 1x1 convolution and the 1x1 shortcut convolution rewritten as one
    /// convolution over `[hidden | x]` with summed biases. Exact, and only
    /// possible when neither is followed by a normalisation.
    Fused(ConvKernel),
    /// Separate 1x1 convolution and optional shortcut convolution.
    Separate {
        conv: Box<Conv1d>,
        shortcut: Option<Box<Conv1d>>,
    },
}

impl Resnet {
    /// Applies the block. The normalisations, the ELU inside the block and
    /// the final sum are left to the convolutions that read them.
    fn forward(&self, x: Lazy) -> Result<Lazy> {
        // Both branches read `x`, one of them through an ELU.
        let stored = !matches!(self.tail, Tail::Fused(_)) || x.is_stored();
        let x = if x.has_elu() || !stored {
            Lazy::new(x.settle())
        } else {
            x
        };
        let hidden = self.conv.forward_lazy(&x.input().with_elu()?, true)?;
        match &self.tail {
            Tail::Fused(kernel) => {
                let hidden = hidden.settle();
                let y = kernel.apply_input(&Input::concat(&hidden, &x.input())?)?;
                hidden.recycle();
                x.recycle();
                Ok(Lazy::new(y))
            }
            Tail::Separate {
                conv,
                shortcut: Some(shortcut),
            } => {
                let skip = shortcut.forward_lazy(&x.input(), false)?;
                x.recycle();
                let y = conv.forward_lazy(&hidden.input(), false)?;
                hidden.recycle();
                y.plus(skip)
            }
            Tail::Separate {
                conv,
                shortcut: None,
            } => {
                let y = conv.forward_lazy(&hidden.input(), false)?;
                hidden.recycle();
                y.plus(x)
            }
        }
    }

    /// Earlier frames an output frame depends on, when the block is causal
    /// and free of normalisation.
    fn left_context(&self) -> Option<usize> {
        let (k, s) = self.conv.causal_window()?;
        let tail = match &self.tail {
            Tail::Fused(_) => 0,
            Tail::Separate { conv, shortcut } => {
                let (kc, sc) = conv.causal_window()?;
                let (ks, ss) = match shortcut {
                    Some(shortcut) => shortcut.causal_window()?,
                    None => (1, 1),
                };
                if sc != 1 || ss != 1 {
                    return None;
                }
                (kc - 1).max(ks - 1)
            }
        };
        (s == 1).then_some(k - 1 + tail)
    }
}

fn run(blocks: &[Block], x: Seq) -> Result<Seq> {
    let mut x = Lazy::new(x);
    for block in blocks {
        x = match block {
            Block::Conv(conv) => {
                let y = conv.forward_lazy(&x.input(), false)?;
                x.recycle();
                y
            }
            Block::ConvTranspose(conv) => {
                let y = conv.forward_lazy(&x.input(), false)?;
                x.recycle();
                y
            }
            // Applied by the layer that reads the result.
            Block::Elu => x.then_elu(),
            Block::Resnet(resnet) => resnet.forward(x)?,
        };
    }
    Ok(x.settle())
}

/// How a causal stack can be run over a long input in chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunking {
    /// Latent frames of left context after which a chunk's output equals the
    /// output of an unchunked run.
    halo: usize,
    /// Samples per latent frame.
    hop: usize,
    /// Whether the stack up-samples latent frames (decoder) or down-samples
    /// audio (encoder).
    up: bool,
}

/// Works out whether `blocks` can run in chunks, and with how much context.
///
/// A chunk that does not start at the true start of the signal sees padding
/// instead of real history, so its first outputs are wrong. Walking the
/// layers, `dirty` counts those wrong leading frames at the current rate;
/// everything after them matches the unchunked run exactly, because causal
/// layers never look ahead.
fn chunking(blocks: &[Block]) -> Option<Chunking> {
    let (mut dirty, mut down, mut up) = (0_usize, 1_usize, 1_usize);
    for block in blocks {
        match block {
            Block::Elu => {}
            Block::Conv(conv) => {
                let (k, s) = conv.causal_window()?;
                dirty = (dirty + k - s).div_ceil(s);
                down *= s;
            }
            Block::ConvTranspose(conv) => {
                let (taps, s) = conv.causal_window()?;
                dirty = (dirty + taps - 1) * s;
                up *= s;
            }
            Block::Resnet(resnet) => dirty += resnet.left_context()?,
        }
    }
    match (down, up) {
        (hop, 1) => Some(Chunking {
            halo: dirty,
            hop,
            up: false,
        }),
        (1, hop) => Some(Chunking {
            halo: dirty.div_ceil(hop),
            hop,
            up: true,
        }),
        _ => None,
    }
}

/// Runs `blocks` over `x` in parallel chunks of [`CHUNK_FRAMES`] latent frames.
///
/// Chunks go through in waves of one per worker thread, which bounds how many
/// sets of activations exist at once.
fn run_chunked(blocks: &[Block], x: &Seq, plan: Chunking) -> Result<Seq> {
    let (in_hop, out_hop) = if plan.up {
        (1, plan.hop)
    } else {
        (plan.hop, 1)
    };
    let frames = x.len().div_ceil(in_hop);
    let starts: Vec<usize> = (0..frames).step_by(CHUNK_FRAMES).collect();
    let mut parts = Vec::with_capacity(starts.len());
    for wave in starts.chunks(rayon::current_num_threads().max(1)) {
        let done = wave
            .par_iter()
            .map(|&first| {
                let end = (first + CHUNK_FRAMES).min(frames);
                let start = first.saturating_sub(plan.halo);
                let input = x.slice_rows(start * in_hop, (end * in_hop).min(x.len()));
                let mut y = run(blocks, input)?;
                ensure!(
                    y.len() == (end - start) * out_hop,
                    "chunk produced {} frames, expected {}",
                    y.len(),
                    (end - start) * out_hop
                );
                y.trim_rows((first - start) * out_hop, 0);
                Ok(y)
            })
            .collect::<Result<Vec<_>>>()?;
        parts.extend(done);
    }
    Seq::concat(parts)
}

/// One half of the model: convolutions, the LSTM, more convolutions.
#[derive(Debug)]
struct Stack {
    pre: Vec<Block>,
    lstm: Lstm,
    post: Vec<Block>,
    /// Chunked execution of `pre`, when it is exact.
    pre_chunking: Option<Chunking>,
    /// Chunked execution of `post`, when it is exact.
    post_chunking: Option<Chunking>,
}

impl Stack {
    fn new(pre: Vec<Block>, lstm: Lstm, post: Vec<Block>) -> Self {
        Self {
            pre_chunking: chunking(&pre),
            post_chunking: chunking(&post),
            pre,
            lstm,
            post,
        }
    }

    fn run_pre(&self, x: Seq) -> Result<Seq> {
        match self.pre_chunking {
            Some(plan) if x.len() > CHUNK_FRAMES * plan.hop => run_chunked(&self.pre, &x, plan),
            _ => run(&self.pre, x),
        }
    }

    fn run_post(&self, x: Seq) -> Result<Seq> {
        match self.post_chunking {
            Some(plan) if x.len() > CHUNK_FRAMES => run_chunked(&self.post, &x, plan),
            _ => run(&self.post, x),
        }
    }
}

/// Builds layers in the Hugging Face module order so tensor names line up.
struct Builder<'a> {
    w: &'a Weights,
    cfg: &'a Config,
}

impl Builder<'_> {
    fn encoder(&self) -> Result<Stack> {
        let cfg = self.cfg;
        let mut layers = Layers::new("encoder.layers");
        let mut blocks = vec![Block::Conv(self.conv(
            &layers.next(),
            geometry(cfg.audio_channels, cfg.num_filters, cfg.kernel_size, 1, 1),
        )?)];
        let mut scaling = 1;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let scale = scaling * cfg.num_filters;
            for j in 0..cfg.num_residual_layers {
                let dilation = dilation(cfg, j)?;
                blocks.push(Block::Resnet(self.resnet(
                    &layers.next(),
                    scale,
                    dilation,
                )?));
            }
            layers.skip();
            blocks.push(Block::Elu);
            blocks.push(Block::Conv(self.conv(
                &layers.next(),
                geometry(scale, scale * 2, ratio * 2, ratio, 1),
            )?));
            scaling *= 2;
        }
        let dim = scaling * cfg.num_filters;
        let lstm = self.lstm(&layers.next(), dim)?;
        layers.skip();
        let post = vec![
            Block::Elu,
            Block::Conv(self.conv(
                &layers.next(),
                geometry(dim, cfg.hidden_size, cfg.last_kernel_size, 1, 1),
            )?),
        ];
        Ok(Stack::new(blocks, lstm, post))
    }

    fn decoder(&self) -> Result<Stack> {
        let cfg = self.cfg;
        let mut layers = Layers::new("decoder.layers");
        let mut scaling = 1_usize << cfg.upsampling_ratios.len();
        let dim = scaling * cfg.num_filters;
        let pre = vec![Block::Conv(self.conv(
            &layers.next(),
            geometry(cfg.hidden_size, dim, cfg.last_kernel_size, 1, 1),
        )?)];
        let lstm = self.lstm(&layers.next(), dim)?;
        let mut blocks = Vec::new();
        for &ratio in &cfg.upsampling_ratios {
            let scale = scaling * cfg.num_filters;
            layers.skip();
            blocks.push(Block::Elu);
            blocks.push(Block::ConvTranspose(self.conv_transpose(
                &layers.next(),
                geometry(scale, scale / 2, ratio * 2, ratio, 1),
            )?));
            for j in 0..cfg.num_residual_layers {
                let dilation = dilation(cfg, j)?;
                blocks.push(Block::Resnet(self.resnet(
                    &layers.next(),
                    scale / 2,
                    dilation,
                )?));
            }
            scaling /= 2;
        }
        layers.skip();
        blocks.push(Block::Elu);
        blocks.push(Block::Conv(self.conv(
            &layers.next(),
            geometry(
                cfg.num_filters,
                cfg.audio_channels,
                cfg.last_kernel_size,
                1,
                1,
            ),
        )?));
        Ok(Stack::new(pre, lstm, blocks))
    }

    fn resnet(&self, prefix: &str, dim: usize, dilation: usize) -> Result<Box<Resnet>> {
        let cfg = self.cfg;
        let hidden = dim / cfg.compress;
        let mut layers = Layers::new(&format!("{prefix}.block"));
        layers.skip();
        let conv = self.conv(
            &layers.next(),
            geometry(dim, hidden, cfg.residual_kernel_size, 1, dilation),
        )?;
        layers.skip();
        let last = self.conv(&layers.next(), geometry(hidden, dim, 1, 1, 1))?;
        let shortcut = if cfg.use_conv_shortcut {
            Some(self.conv(&format!("{prefix}.shortcut"), geometry(dim, dim, 1, 1, 1))?)
        } else {
            None
        };
        // With weight normalisation nothing sits between the two 1x1
        // convolutions and their sum, so they collapse into one.
        let tail = match shortcut {
            Some(shortcut) if cfg.norm_type == NormType::WeightNorm => {
                Tail::Fused(ConvKernel::fuse_sum(last.kernel(), shortcut.kernel())?)
            }
            shortcut => Tail::Separate {
                conv: last,
                shortcut,
            },
        };
        Ok(Box::new(Resnet { conv, tail }))
    }

    /// Conv weights, materialising weight normalisation when used.
    fn weight(&self, prefix: &str, shape: [usize; 3]) -> Result<Vec<f32>> {
        match self.cfg.norm_type {
            NormType::TimeGroupNorm => Ok(self
                .w
                .get(&format!("{prefix}.weight"), &shape)?
                .into_owned()),
            NormType::WeightNorm => {
                let g = self
                    .w
                    .get(&format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
                let mut out = self
                    .w
                    .get(&format!("{prefix}.weight_v"), &shape)?
                    .into_owned();
                let row = shape[1] * shape[2];
                for (chunk, &gain) in out.chunks_exact_mut(row).zip(g.iter()) {
                    let norm = chunk
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        .sqrt();
                    let factor = narrow(f64::from(gain) / norm);
                    for x in chunk.iter_mut() {
                        *x *= factor;
                    }
                }
                Ok(out)
            }
        }
    }

    fn norm(&self, prefix: &str, channels: usize) -> Result<Option<GroupNorm1>> {
        match self.cfg.norm_type {
            NormType::WeightNorm => Ok(None),
            NormType::TimeGroupNorm => Ok(Some(GroupNorm1::new(
                self.w
                    .get(&format!("{prefix}.norm.weight"), &[channels])?
                    .into_owned(),
                self.w
                    .get(&format!("{prefix}.norm.bias"), &[channels])?
                    .into_owned(),
            )?)),
        }
    }

    fn conv(&self, prefix: &str, g: ConvGeometry) -> Result<Box<Conv1d>> {
        let shape = [g.out_channels, g.in_channels, g.kernel];
        let weight = self.weight(&format!("{prefix}.conv"), shape)?;
        let bias = self
            .w
            .get(&format!("{prefix}.conv.bias"), &[g.out_channels])?;
        let kernel = ConvKernel::from_torch(
            &weight,
            &bias,
            (g.out_channels, g.in_channels, g.kernel),
            g.stride,
            g.dilation,
        )?;
        Ok(Box::new(Conv1d::new(
            kernel,
            self.cfg.use_causal_conv,
            self.cfg.pad_mode,
            self.norm(prefix, g.out_channels)?,
        )))
    }

    fn conv_transpose(&self, prefix: &str, g: ConvGeometry) -> Result<Box<ConvTranspose1d>> {
        let shape = [g.in_channels, g.out_channels, g.kernel];
        let weight = self.weight(&format!("{prefix}.conv"), shape)?;
        let bias = self
            .w
            .get(&format!("{prefix}.conv.bias"), &[g.out_channels])?;
        ConvTranspose1d::new(
            &weight,
            &bias,
            g,
            self.cfg.use_causal_conv,
            self.cfg.trim_right_ratio,
            self.norm(prefix, g.out_channels)?,
        )
        .map(Box::new)
    }

    fn lstm(&self, prefix: &str, dim: usize) -> Result<Lstm> {
        let mut lstm = Lstm::new(dim, &[])?;
        for l in 0..self.cfg.num_lstm_layers {
            let name = |kind: &str| format!("{prefix}.lstm.{kind}_l{l}");
            lstm.push_layer(&LstmParams {
                w_ih: &self.w.get(&name("weight_ih"), &[4 * dim, dim])?,
                w_hh: &self.w.get(&name("weight_hh"), &[4 * dim, dim])?,
                b_ih: &self.w.get(&name("bias_ih"), &[4 * dim])?,
                b_hh: &self.w.get(&name("bias_hh"), &[4 * dim])?,
            })?;
        }
        Ok(lstm)
    }
}

const fn geometry(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> ConvGeometry {
    ConvGeometry {
        in_channels,
        out_channels,
        kernel,
        stride,
        dilation,
    }
}

fn dilation(cfg: &Config, layer: usize) -> Result<usize> {
    let exponent = u32::try_from(layer)?;
    cfg.dilation_growth_rate
        .checked_pow(exponent)
        .context("dilation overflows")
}

/// Hands out `prefix.0`, `prefix.1`, ... matching `nn.Sequential` indices.
struct Layers {
    prefix: String,
    index: usize,
}

impl Layers {
    fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_owned(),
            index: 0,
        }
    }

    fn next(&mut self) -> String {
        let name = format!("{}.{}", self.prefix, self.index);
        self.index += 1;
        name
    }

    /// Skips a parameter-free layer such as ELU.
    const fn skip(&mut self) {
        self.index += 1;
    }
}

#[derive(Debug)]
struct Codebook {
    /// `[size, dim]`.
    embed: Vec<f32>,
    /// `[dim, size]`, only present when the model can encode.
    embed_t: Option<Vec<f32>>,
    /// Squared norm of every entry.
    norms: Vec<f32>,
}

#[derive(Debug)]
struct Quantizer {
    codebooks: Vec<Codebook>,
    dim: usize,
    size: usize,
}

impl Quantizer {
    fn new(w: &Weights, cfg: &Config, limit: Option<usize>, for_encoding: bool) -> Result<Self> {
        let dim = cfg.codebook_dim();
        let size = cfg.codebook_size;
        let total = cfg.total_codebooks()?;
        ensure!(
            !w.contains(&format!("quantizer.layers.{total}.codebook.embed")),
            "checkpoint has more codebooks than the configuration allows"
        );
        let count = limit.map_or(total, |n| n.min(total));
        let codebooks = (0..count)
            .map(|q| {
                let embed = w
                    .get(
                        &format!("quantizer.layers.{q}.codebook.embed"),
                        &[size, dim],
                    )?
                    .into_owned();
                let embed_t = for_encoding.then(|| {
                    let mut embed_t = vec![0.0; size * dim];
                    for n in 0..size {
                        for d in 0..dim {
                            embed_t[d * size + n] = embed[n * dim + d];
                        }
                    }
                    embed_t
                });
                let norms = embed
                    .chunks_exact(dim)
                    .map(|e| e.iter().map(|v| v * v).sum())
                    .collect();
                Ok(Codebook {
                    embed,
                    embed_t,
                    norms,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            codebooks,
            dim,
            size,
        })
    }

    /// Residual vector quantisation, in blocks of frames so the score matrix
    /// of a block stays in cache.
    fn encode(&self, embeddings: &Seq, num_codebooks: usize) -> Result<Vec<u16>> {
        ensure!(embeddings.channels() == self.dim, "latent width mismatch");
        ensure!(
            (1..=self.codebooks.len()).contains(&num_codebooks),
            "{} codebooks are loaded, {num_codebooks} requested",
            self.codebooks.len()
        );
        let (dim, size) = (self.dim, self.size);
        let mut residual = embeddings.data().to_vec();
        let mut codes = vec![0_u16; embeddings.len() * num_codebooks];
        residual
            .par_chunks_mut(QUANTIZER_BLOCK * dim)
            .zip(codes.par_chunks_mut(QUANTIZER_BLOCK * num_codebooks))
            .try_for_each(|(residual, codes)| -> Result<()> {
                let rows = residual.len() / dim;
                let mut dots = vec![0.0_f32; rows * size];
                for (q, book) in self.codebooks.iter().take(num_codebooks).enumerate() {
                    let embed_t = book
                        .embed_t
                        .as_deref()
                        .context("codebooks were loaded for decoding only")?;
                    matmul_into(residual, embed_t, (rows, dim, size), &mut dots)?;
                    for ((x, dot), code) in residual
                        .chunks_exact_mut(dim)
                        .zip(dots.chunks_exact(size))
                        .zip(codes.chunks_exact_mut(num_codebooks))
                    {
                        let best = nearest(x, dot, &book.norms);
                        code[q] = u16::try_from(best)?;
                        let entry = &book.embed[best * dim..(best + 1) * dim];
                        for (r, e) in x.iter_mut().zip(entry) {
                            *r -= e;
                        }
                    }
                }
                Ok(())
            })?;
        Ok(codes)
    }

    /// Beam-search residual vector quantisation: every frame keeps the
    /// `beam` partial code sequences with the smallest residual after each
    /// stage, and the one with the smallest final residual wins. The codes
    /// are ordinary RVQ codes, so any decoder reads them.
    fn encode_beam(&self, embeddings: &Seq, num_codebooks: usize, beam: usize) -> Result<Vec<u16>> {
        ensure!(embeddings.channels() == self.dim, "latent width mismatch");
        ensure!(
            (1..=self.codebooks.len()).contains(&num_codebooks),
            "{} codebooks are loaded, {num_codebooks} requested",
            self.codebooks.len()
        );
        ensure!(beam >= 1, "beam width must be at least 1");
        let (dim, nq) = (self.dim, num_codebooks);
        // Fewer frames per task as the beam grows, so the scores stay small.
        let block = (QUANTIZER_BLOCK / beam).max(4);
        let mut codes = vec![0_u16; embeddings.len() * nq];
        embeddings
            .data()
            .par_chunks(block * dim)
            .zip(codes.par_chunks_mut(block * nq))
            .try_for_each(|(x, codes)| self.beam_block(x, codes, nq, beam))?;
        Ok(codes)
    }

    /// [`Quantizer::encode_beam`] for one block of frames.
    #[expect(
        clippy::suboptimal_flops,
        reason = "the distance is formed as the greedy search forms it"
    )]
    fn beam_block(&self, x: &[f32], out: &mut [u16], nq: usize, beam: usize) -> Result<()> {
        let (dim, size) = (self.dim, self.size);
        let rows = x.len() / dim;
        // Hypotheses per frame, all frames alike: 1 before the first stage.
        let mut width = 1;
        let mut residual = x.to_vec();
        let mut hyp_codes = vec![0_u16; rows * nq];
        let mut dots = Vec::new();
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);
        for (q, book) in self.codebooks.iter().take(nq).enumerate() {
            let embed_t = book
                .embed_t
                .as_deref()
                .context("codebooks were loaded for decoding only")?;
            let hyps = rows * width;
            dots.resize(hyps * size, 0.0);
            matmul_into(&residual, embed_t, (hyps, dim, size), &mut dots)?;
            let next_width = beam.min(width * size);
            let mut next_residual = vec![0.0_f32; rows * next_width * dim];
            let mut next_codes = vec![0_u16; rows * next_width * nq];
            for row in 0..rows {
                best.clear();
                for h in 0..width {
                    let hyp = row * width + h;
                    let r = &residual[hyp * dim..(hyp + 1) * dim];
                    let x2: f32 = r.iter().map(|v| v * v).sum();
                    let scores = &dots[hyp * size..(hyp + 1) * size];
                    for (n, (&d, &e2)) in scores.iter().zip(&book.norms).enumerate() {
                        let distance = (x2 - 2.0 * d) + e2;
                        if best.len() < next_width
                            || best.last().is_some_and(|&(worst, _)| distance < worst)
                        {
                            // Keep `best` sorted; equal scores keep the earlier candidate first.
                            let at = best.partition_point(|&(s, _)| s <= distance);
                            best.insert(at, (distance, h * size + n));
                            best.truncate(next_width);
                        }
                    }
                }
                for (k, &(_, candidate)) in best.iter().enumerate() {
                    let (h, n) = (candidate / size, candidate % size);
                    let parent = row * width + h;
                    let child = row * next_width + k;
                    let entry = &book.embed[n * dim..(n + 1) * dim];
                    let src = &residual[parent * dim..(parent + 1) * dim];
                    for ((dst, &r), &e) in next_residual[child * dim..(child + 1) * dim]
                        .iter_mut()
                        .zip(src)
                        .zip(entry)
                    {
                        *dst = r - e;
                    }
                    let (from, to) = (parent * nq, child * nq);
                    next_codes[to..to + nq].copy_from_slice(&hyp_codes[from..from + nq]);
                    next_codes[to + q] = u16::try_from(n)?;
                }
            }
            residual = next_residual;
            hyp_codes = next_codes;
            width = next_width;
        }
        for (row, dst) in out.chunks_exact_mut(nq).enumerate() {
            let mut winner = 0;
            let mut smallest = f32::INFINITY;
            for h in 0..width {
                let hyp = row * width + h;
                let e: f32 = residual[hyp * dim..(hyp + 1) * dim]
                    .iter()
                    .map(|v| v * v)
                    .sum();
                if e < smallest {
                    smallest = e;
                    winner = hyp;
                }
            }
            dst.copy_from_slice(&hyp_codes[winner * nq..(winner + 1) * nq]);
        }
        Ok(())
    }

    fn decode(&self, frame: &EncodedFrame) -> Result<Seq> {
        self.embed(&frame.codes, frame.num_codebooks)
    }

    /// Sums the codebook entries of frame-major codes, `k` per frame.
    fn embed(&self, codes: &[u16], k: usize) -> Result<Seq> {
        ensure!(
            (1..=self.codebooks.len()).contains(&k),
            "frame uses {k} codebooks, {} are loaded",
            self.codebooks.len()
        );
        let mut out = Seq::zeros(codes.len() / k, self.dim);
        for (row, codes) in out
            .data_mut()
            .chunks_exact_mut(self.dim)
            .zip(codes.chunks_exact(k))
        {
            for (book, &code) in self.codebooks.iter().zip(codes) {
                let code = usize::from(code);
                ensure!(code < self.size, "code {code} is out of range");
                let entry = &book.embed[code * self.dim..(code + 1) * self.dim];
                for (r, e) in row.iter_mut().zip(entry) {
                    *r += e;
                }
            }
        }
        Ok(out)
    }
}

/// Index of the codebook entry nearest to `x`, given `x · e` for every entry
/// and the squared norms of the entries. Ties go to the lowest index, and the
/// distance is formed exactly as the reference does, `(|x|² - 2 x·e) + |e|²`.
#[expect(
    clippy::suboptimal_flops,
    reason = "mul_add is a libm call without the FMA target feature"
)]
fn nearest(x: &[f32], dots: &[f32], norms: &[f32]) -> usize {
    let x2: f32 = x.iter().map(|v| v * v).sum();
    let mut best = 0;
    let mut best_distance = f32::INFINITY;
    for (n, (&d, &e2)) in dots.iter().zip(norms).enumerate() {
        let distance = (x2 - 2.0 * d) + e2;
        if distance < best_distance {
            best_distance = distance;
            best = n;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(len: usize, channels: usize, f: impl Fn(usize) -> f32) -> Result<Seq> {
        Seq::from_vec(len, channels, (0..len * channels).map(f).collect())
    }

    #[test]
    fn overlap_add_blends_linearly() -> Result<()> {
        // Constant segments must reproduce the constant; lengths follow the stride.
        let frames = vec![
            seq(10, 2, |_| 1.0)?,
            seq(10, 2, |_| 1.0)?,
            seq(4, 2, |_| 1.0)?,
        ];
        let out = linear_overlap_add(&frames, 8, None)?;
        assert_eq!(out.len(), 20);
        assert!(out.data().iter().all(|v| (v - 1.0).abs() < 1e-6));
        // Outside the overlap a sample is its segment's value, bit for bit.
        let a = seq(10, 1, |i| 0.1 * f32::from(u8::try_from(i).unwrap_or(0)))?;
        let b = seq(10, 1, |_| -3.0)?;
        let mixed = linear_overlap_add(&[a.clone(), b], 8, None)?;
        assert_eq!(&mixed.data()[..8], &a.data()[..8]);
        assert!(mixed.data()[8] < a.data()[8]);
        assert!(
            mixed.data()[10..]
                .iter()
                .all(|v| v.to_bits() == (-3.0_f32).to_bits())
        );
        Ok(())
    }

    #[test]
    fn nearest_prefers_the_first_of_equal_entries() {
        // Two identical entries: the lower index wins, as with torch's argmax.
        assert_eq!(nearest(&[1.0, 0.0], &[0.0, 1.0, 1.0], &[1.0, 1.0, 1.0]), 1);
    }

    /// Reads one of the real checkpoints, or `None` when the weights are not
    /// checked out next to the crate.
    fn real_model(name: &str, options: LoadOptions) -> Result<Option<Encodec>> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
        let (config, weights) = (
            dir.join(format!("{name}.json")),
            dir.join(format!("{name}.safetensors")),
        );
        if !(config.is_file() && weights.is_file()) {
            return Ok(None);
        }
        Encodec::load_with(name, &config, &weights, options).map(Some)
    }

    fn test_audio(len: usize, channels: usize) -> Result<Seq> {
        seq(len, channels, |i| {
            let t = f32::from(u16::try_from(i % 4801).unwrap_or(0));
            0.3_f32.mul_add((t * 0.05).sin(), 0.1 * (t * 0.61).sin())
        })
    }

    #[test]
    fn chunked_causal_stacks_match_one_pass() -> Result<()> {
        let Some(model) = real_model("encodec_24khz", LoadOptions::ALL)? else {
            return Ok(());
        };
        let (Some(encoder), Some(decoder)) = (&model.encoder, &model.decoder) else {
            anyhow::bail!("model was loaded without a half");
        };
        let (Some(pre), Some(post)) = (encoder.pre_chunking, decoder.post_chunking) else {
            anyhow::bail!("the 24 kHz model must be chunkable");
        };
        assert_eq!(
            (pre.hop, pre.up, post.hop, post.up),
            (320, false, 320, true)
        );
        // Not a multiple of the hop, and several chunks long.
        let audio = test_audio(320 * (3 * CHUNK_FRAMES + 5) + 123, 1)?;
        let whole = run(&encoder.pre, audio.clone())?;
        let chunked = run_chunked(&encoder.pre, &audio, pre)?;
        assert_eq!(whole.len(), chunked.len());
        assert!(whole == chunked, "chunked encoder differs from one pass");
        // One frame less context must not be enough, or the halo is not tight.
        let short = Chunking {
            halo: pre.halo - 1,
            ..pre
        };
        assert!(run_chunked(&encoder.pre, &audio, short)? != whole);

        let features = encoder.lstm.forward(&whole)?;
        let whole = run(&decoder.post, features.clone())?;
        let chunked = run_chunked(&decoder.post, &features, post)?;
        assert!(whole == chunked, "chunked decoder differs from one pass");
        Ok(())
    }

    #[test]
    fn partial_models_work_and_refuse_the_other_direction() -> Result<()> {
        for (name, channels, bandwidth) in [("encodec_24khz", 1, 6.0), ("encodec_48khz", 2, 6.0)] {
            let Some(full) = real_model(name, LoadOptions::ALL)? else {
                return Ok(());
            };
            let count = full.config().codebooks_for_bandwidth(bandwidth)?;
            let audio = test_audio(full.config().sampling_rate * 5 / 4, channels)?;
            let frames = full.encode(&audio, bandwidth)?;
            let reference = full.decode(&frames, audio.len())?;
            assert_eq!(reference.len(), audio.len());

            let encoder = real_model(name, LoadOptions::encoder_only(Some(count)))?
                .context("weights disappeared")?;
            assert!(encoder.encode(&audio, bandwidth)? == frames);
            assert!(encoder.decode(&frames, audio.len()).is_err());

            let decoder = real_model(name, LoadOptions::decoder_only(Some(count)))?
                .context("weights disappeared")?;
            assert!(decoder.decode(&frames, audio.len())? == reference);
            assert!(decoder.encode(&audio, bandwidth).is_err());
        }
        Ok(())
    }

    /// Rewrites a safetensors file without the tensors whose name starts with `prefix`.
    fn without_prefix(bytes: &[u8], prefix: &str) -> Result<Vec<u8>> {
        let length: [u8; 8] = bytes[..8].try_into()?;
        let data_start = 8 + usize::try_from(u64::from_le_bytes(length))?;
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..data_start])?;
        header.retain(|name, _| !name.starts_with(prefix));
        let mut data = Vec::new();
        for (name, value) in &mut header {
            if name == "__metadata__" {
                continue;
            }
            let (start, end): (usize, usize) =
                serde_json::from_value(value["data_offsets"].clone())?;
            value["data_offsets"] = serde_json::json!([data.len(), data.len() + end - start]);
            data.extend_from_slice(&bytes[data_start + start..data_start + end]);
        }
        let header = serde_json::to_vec(&header)?;
        let mut out = u64::try_from(header.len())?.to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        Ok(out)
    }

    #[test]
    fn decoder_builds_from_bytes_without_encoder_tensors() -> Result<()> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
        let (config, weights) = (
            dir.join("encodec_24khz.json"),
            dir.join("encodec_24khz.safetensors"),
        );
        if !(config.is_file() && weights.is_file()) {
            return Ok(());
        }
        let config = std::fs::read_to_string(config)?;
        let subset = without_prefix(&std::fs::read(weights)?, "encoder.")?;
        let decoder = Encodec::decoder_from_bytes("encodec_24khz", &config, &subset)?;
        assert!(decoder.has_decoder() && !decoder.has_encoder());
        assert!(Encodec::from_bytes("encodec_24khz", &config, &subset).is_err());
        let frame = EncodedFrame {
            scale: None,
            num_codebooks: 2,
            codes: (0..40_u16).map(|i| (i * 37) % 1024).collect(),
        };
        let audio = decoder.decode(&[frame], 20 * 320 - 7)?;
        assert_eq!(audio.len(), 20 * 320 - 7);
        assert!(audio.data().iter().all(|v| v.is_finite()));
        assert!(decoder.encode(&audio, 6.0).is_err());
        Ok(())
    }

    fn pseudo_codes(frames: usize, codebooks: usize) -> Vec<u16> {
        let mut state = 0x2545_f491_u32;
        (0..frames * codebooks)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                u16::try_from((state >> 16) % 1024).unwrap_or(0)
            })
            .collect()
    }

    /// Signal-to-noise ratio of `got` against `want`, in dB.
    fn snr_db(want: &Seq, got: &Seq) -> f64 {
        let signal: f64 = want.data().iter().map(|&v| f64::from(v).powi(2)).sum();
        let noise: f64 = want
            .data()
            .iter()
            .zip(got.data())
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        10.0 * (signal / noise.max(1e-300)).log10()
    }

    #[test]
    fn stream_decoder_matches_whole_file_decode() -> Result<()> {
        let codebooks = 8;
        let Some(model) = real_model("encodec_24khz", LoadOptions::decoder_only(Some(codebooks)))?
        else {
            return Ok(());
        };
        // 10 seconds of latent frames; the audio ends inside the last frame.
        let frames = 750;
        let codes = pseudo_codes(frames, codebooks);
        let audio_length = frames * 320 - 77;
        let whole = model.decode(
            &[EncodedFrame {
                scale: None,
                num_codebooks: codebooks,
                codes: codes.clone(),
            }],
            audio_length,
        )?;
        let mut random = 12_345_usize;
        let mut next_random = move || {
            random = (random * 1_103_515_245 + 12_345) % 2_147_483_648;
            1 + random % 97
        };
        let mut patterns: Vec<Box<dyn FnMut() -> usize>> = vec![
            Box::new(|| 1),
            Box::new(|| 7),
            Box::new(|| 375),
            Box::new(&mut next_random),
        ];
        for chunk_size in &mut patterns {
            let mut decoder = model.stream_decoder(codebooks)?;
            let mut parts = Vec::new();
            let mut start = 0;
            let mut largest = 0;
            while start < frames {
                let end = (start + chunk_size()).min(frames);
                let part = decoder.push(&codes[start * codebooks..end * codebooks])?;
                // Output is bounded by what was pushed, plus the frames held at the start.
                largest = largest.max(part.len());
                assert!(part.len() <= (end - start + 8) * 320);
                parts.push(part);
                start = end;
            }
            parts.push(decoder.finish(audio_length)?);
            let streamed = Seq::concat(parts)?;
            assert_eq!(streamed.len(), whole.len());
            let snr = snr_db(&whole, &streamed);
            assert!(
                snr >= 100.0,
                "streamed decode is only {snr:.1} dB from one pass"
            );
            assert!(streamed == whole, "streamed decode is not bit-exact");
            assert!(largest > 0);
        }
        Ok(())
    }

    #[test]
    fn stream_decoder_handles_short_streams_and_bad_input() -> Result<()> {
        let codebooks = 2;
        let Some(model) = real_model("encodec_24khz", LoadOptions::decoder_only(Some(codebooks)))?
        else {
            return Ok(());
        };
        // Shorter than the start-up buffering, exactly at it, and just past it.
        for frames in [1_usize, 3, 7, 8, 20] {
            let codes = pseudo_codes(frames, codebooks);
            let audio_length = frames * 320 - 5;
            let frame = EncodedFrame {
                scale: None,
                num_codebooks: codebooks,
                codes: codes.clone(),
            };
            let whole = model.decode(&[frame], audio_length)?;
            let mut decoder = model.stream_decoder(codebooks)?;
            let mut parts = Vec::new();
            for one in codes.chunks(codebooks) {
                parts.push(decoder.push(one)?);
            }
            parts.push(decoder.finish(audio_length)?);
            assert!(Seq::concat(parts)? == whole, "{frames} frames differ");
        }
        let mut decoder = model.stream_decoder(codebooks)?;
        assert!(decoder.push(&[1, 2, 3]).is_err(), "not a whole frame");
        assert!(
            decoder.push(&[1, 5000]).is_ok(),
            "codes are checked when decoded"
        );
        assert!(decoder.finish(320).is_err(), "code out of range");
        let decoder = model.stream_decoder(codebooks)?;
        assert!(decoder.finish(1).is_err(), "no frames for one sample");
        assert!(model.stream_decoder(codebooks + 1).is_err());
        assert!(
            model.overlap_add().is_err(),
            "24 kHz model is not segmented"
        );
        Ok(())
    }

    #[test]
    fn segment_decode_with_overlap_add_matches_decode() -> Result<()> {
        let Some(model) = real_model("encodec_48khz", LoadOptions::ALL)? else {
            return Ok(());
        };
        assert!(
            model.stream_decoder(4).is_err(),
            "48 kHz model is not causal"
        );
        // Three full segments and a short fourth one.
        let audio = test_audio(48_000 * 3 + 9_000, 2)?;
        let frames = model.encode(&audio, 6.0)?;
        assert_eq!(frames.len(), 4);
        let whole = model.decode(&frames, audio.len())?;
        let mut overlap = model.overlap_add()?;
        let mut parts = Vec::new();
        for frame in &frames {
            parts.push(overlap.push(&model.decode_segment(frame)?)?);
        }
        assert_eq!(parts[0].len(), 47_520);
        parts.push(overlap.finish(audio.len())?);
        let joined = Seq::concat(parts)?;
        assert_eq!(joined.len(), whole.len());
        let snr = snr_db(&whole, &joined);
        assert!(
            snr >= 100.0,
            "segment-wise decode is only {snr:.1} dB from decode"
        );
        assert!(joined == whole, "segment-wise decode is not bit-exact");
        Ok(())
    }

    #[test]
    fn incremental_overlap_add_matches_batch() -> Result<()> {
        let segments = vec![
            seq(10, 2, |i| f32::from(u8::try_from(i % 7).unwrap_or(0)) - 3.0)?,
            seq(10, 2, |i| 0.5 * f32::from(u8::try_from(i % 5).unwrap_or(0)))?,
            seq(10, 2, |i| -f32::from(u8::try_from(i % 3).unwrap_or(0)))?,
            seq(3, 2, |_| 4.0)?,
        ];
        let batch = linear_overlap_add(&segments, 8, None)?;
        let mut overlap = OverlapAdd::new(8)?;
        let mut parts = Vec::new();
        for segment in &segments {
            parts.push(overlap.push(segment)?);
        }
        assert!(
            overlap.clone().push(&segments[3]).is_err(),
            "nothing follows a short segment"
        );
        assert!(overlap.clone().finish(batch.len() + 1).is_err());
        parts.push(overlap.finish(batch.len() - 1)?);
        let joined = Seq::concat(parts)?;
        assert_eq!(joined.data(), &batch.data()[..(batch.len() - 1) * 2]);
        // A single segment passes through unchanged.
        let mut single = OverlapAdd::new(8)?;
        assert_eq!(single.push(&segments[0])?.len(), 8);
        assert_eq!(single.finish(10)?.data(), &segments[0].data()[16..]);
        assert!(OverlapAdd::new(0).is_err());
        Ok(())
    }

    #[test]
    fn edge_overlap_add_follows_the_ramp_and_streams_exactly() -> Result<()> {
        // Segments of 40 samples, stride 30: overlap 10, hi = 10, lo = 0.
        let segments = vec![
            seq(40, 1, |_| 1.0)?,
            seq(40, 1, |_| 3.0)?,
            seq(40, 1, |i| f32::from(u8::try_from(i % 9).unwrap_or(0)))?,
            seq(5, 1, |_| -2.0)?,
        ];
        let ramp = EdgeRamp::new(40, 30);
        let batch = linear_overlap_add(&segments, 30, Some(ramp))?;
        assert_eq!(batch.len(), 95);
        assert!(batch.data()[..30].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        for t in 0..10_u8 {
            let a = f64::from(t + 1) / 11.0;
            let want = 3.0f64.mul_add(a, 1.0 - a);
            let got = f64::from(batch.row(30 + usize::from(t))[0]);
            assert!((got - want).abs() < 1e-6, "t {t}: {got} vs {want}");
        }
        let mut overlap = OverlapAdd::with_crossfade(40, 30, Crossfade::EdgeAvoiding)?;
        let mut parts = Vec::new();
        for segment in &segments {
            parts.push(overlap.push(segment)?);
        }
        parts.push(overlap.finish(batch.len())?);
        let joined = Seq::concat(parts)?;
        assert!(joined == batch, "streamed edge overlap-add differs");
        assert!(OverlapAdd::with_crossfade(40, 19, Crossfade::EdgeAvoiding).is_err());
        Ok(())
    }

    #[test]
    fn strided_edge_decode_streams_exactly() -> Result<()> {
        let Some(model) = real_model("encodec_48khz", LoadOptions::ALL)? else {
            return Ok(());
        };
        let audio = test_audio(48_000 * 3 + 2_000, 2)?;
        let segmenting = Segmenting {
            stride: Some(45_000),
            crossfade: Crossfade::EdgeAvoiding,
        };
        let params = EncodeParams {
            num_codebooks: 15,
            segmenting,
            beam: 1,
        };
        let frames = model.encode_with(&audio, &params)?;
        assert_eq!(frames.len(), 4);
        assert!(frames.iter().all(|f| f.num_codebooks == 15));
        let whole = model.decode_with(&frames, audio.len(), segmenting)?;
        let mut overlap = model.overlap_add_with(segmenting)?;
        let mut parts = Vec::new();
        for frame in &frames {
            parts.push(overlap.push(&model.decode_segment(frame)?)?);
        }
        assert_eq!(parts[0].len(), 45_000);
        parts.push(overlap.finish(audio.len())?);
        let joined = Seq::concat(parts)?;
        assert!(
            joined == whole,
            "segment-wise strided decode is not bit-exact"
        );
        let snr = snr_db(&audio, &whole);
        assert!(
            snr > 3.0,
            "strided decode is only {snr:.1} dB from the input"
        );
        assert!(
            model
                .encode_with(
                    &audio,
                    &EncodeParams {
                        segmenting: Segmenting {
                            stride: Some(20_000),
                            crossfade: Crossfade::Triangle,
                        },
                        ..params
                    },
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn beam_search_never_does_worse_than_greedy() -> Result<()> {
        let Some(model) = real_model("encodec_48khz", LoadOptions::ALL)? else {
            return Ok(());
        };
        let audio = test_audio(30_000, 2)?;
        let greedy = model.encode_with(&audio, &EncodeParams::stock(4))?;
        let one = model.encode_with(
            &audio,
            &EncodeParams {
                beam: 1,
                ..EncodeParams::stock(4)
            },
        )?;
        assert_eq!(greedy, one);
        // Beam 1 through the beam code path is the greedy search.
        let encoder = model.encoder.as_ref().context("encoder")?;
        let mut chunk = audio;
        let _ = model.normalize(&mut chunk);
        let z = run(
            &encoder.post,
            encoder.lstm.forward(&encoder.run_pre(chunk)?)?,
        )?;
        let codes_greedy = model.quantizer.encode(&z, 4)?;
        assert_eq!(model.quantizer.encode_beam(&z, 4, 1)?, codes_greedy);
        let residual = |codes: &[u16]| -> Result<f64> {
            let q = model.quantizer.embed(codes, 4)?;
            Ok(z.data()
                .iter()
                .zip(q.data())
                .map(|(&a, &b)| f64::from(a - b).powi(2))
                .sum())
        };
        let e1 = residual(&codes_greedy)?;
        let e8 = residual(&model.quantizer.encode_beam(&z, 4, 8)?)?;
        assert!(e8 <= e1, "beam 8 residual {e8} > greedy {e1}");
        assert!(e8 < 0.999 * e1, "beam 8 should help: {e8} vs {e1}");
        Ok(())
    }
}
