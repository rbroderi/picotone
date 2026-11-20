/** EnCodec hyper-parameters (the subset of the Hugging Face config the decoder needs). */
export interface ModelConfig {
  readonly audio_channels: number;
  readonly chunk_length_s: number | null;
  readonly codebook_dim?: number | null;
  readonly codebook_size: number;
  readonly compress: number;
  readonly dilation_growth_rate: number;
  readonly hidden_size: number;
  readonly kernel_size: number;
  readonly last_kernel_size: number;
  readonly norm_type: 'weight_norm' | 'time_group_norm';
  readonly normalize: boolean;
  readonly num_filters: number;
  readonly num_lstm_layers: number;
  readonly num_residual_layers: number;
  readonly overlap: number | null;
  readonly pad_mode: 'reflect' | 'constant' | 'zero' | 'replicate';
  readonly residual_kernel_size: number;
  readonly sampling_rate: number;
  readonly target_bandwidths: readonly number[];
  readonly trim_right_ratio: number;
  readonly upsampling_ratios: readonly number[];
  readonly use_causal_conv: boolean;
  readonly use_conv_shortcut: boolean;
}

/** A coded segment's position in the audio (mirrors `model::Segment`). */
export interface Segment {
  readonly offset: number;
  readonly length: number;
  readonly frames: number;
}

export function codebookDim(cfg: ModelConfig): number {
  return cfg.codebook_dim ?? cfg.hidden_size;
}

/** Samples per latent frame. */
export function hopLength(cfg: Pick<ModelConfig, 'upsampling_ratios'>): number {
  return cfg.upsampling_ratios.reduce((a, b) => a * b, 1);
}

/** Latent frames per second. */
export function frameRate(cfg: Pick<ModelConfig, 'upsampling_ratios' | 'sampling_rate'>): number {
  return Math.ceil(cfg.sampling_rate / hopLength(cfg));
}

export function bitsPerCodebook(cfg: Pick<ModelConfig, 'codebook_size'>): number {
  return Math.round(Math.log2(cfg.codebook_size));
}

/** Segment length in samples, or null for whole-file coding. */
export function segmentLength(cfg: ModelConfig): number | null {
  return cfg.chunk_length_s === null ? null : Math.floor(cfg.chunk_length_s * cfg.sampling_rate);
}

/** Distance between segment starts, or null for whole-file coding. */
export function segmentStride(cfg: ModelConfig): number | null {
  const length = segmentLength(cfg);
  if (length === null) return null;
  return Math.max(1, Math.floor((1 - (cfg.overlap ?? 0)) * length));
}

/**
 * Splits `audioLength` samples into coded segments (`model::segments_with`),
 * `stride` apart (default: the model's stride; a `.pico` ID header sets it).
 */
export function segments(cfg: ModelConfig, audioLength: number, stride?: number | null): Segment[] {
  const length = segmentLength(cfg) ?? audioLength;
  const step = Math.max(1, (segmentLength(cfg) === null ? null : stride) ?? segmentStride(cfg) ?? audioLength);
  const rate = frameRate(cfg);
  const out: Segment[] = [];
  for (let offset = 0; offset < audioLength; offset += step) {
    const len = Math.min(audioLength - offset, length);
    out.push({ offset, length: len, frames: Math.ceil((len * rate) / cfg.sampling_rate) });
  }
  return out;
}

/** Approximate bandwidth in kbps for a given number of codebooks. */
export function bandwidthKbps(
  cfg: Pick<ModelConfig, 'upsampling_ratios' | 'sampling_rate' | 'codebook_size'>,
  numCodebooks: number,
): number {
  return (numCodebooks * bitsPerCodebook(cfg) * frameRate(cfg)) / 1000;
}
