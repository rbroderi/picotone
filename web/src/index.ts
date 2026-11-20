/// <reference types="@webgpu/types" />
/**
 * picotone `.pico` decoding on WebGPU: the cx2 codes are range-decoded in
 * WebAssembly (bit-exact with the command line) and the EnCodec network runs on the GPU.
 *
 * ```ts
 * import { isWebGpuAvailable, PicotoneWebGpu, modelNameOf } from './picotone-webgpu/picotone-webgpu.js';
 * const gpu = isWebGpuAvailable();
 * if (!gpu.available) useFallback(gpu.reason);
 * const model = await PicotoneWebGpu.load({ model: modelNameOf(bytes), baseUrl: '/static/picotone/models/', dtype: 'f16' });
 * const { sampleRate, channels } = await model.decode(bytes, { onChunk: (c) => play(c) });
 * ```
 */

export {
  type DecodeOptions,
  type DecodedAudio,
  type DecodedChunk,
  type LoadOptions,
  PicotoneWebGpu,
  modelNameOf,
} from './decoder.js';
export {
  type CodedSegment,
  type PicoCodes,
  type PicoHeader,
  Cx2Reader,
  PicoError,
  parseHeader,
} from './pico.js';
export {
  type ModelConfig,
  type Segment,
  bandwidthKbps,
  frameRate,
  hopLength,
  segments,
} from './config.js';
export { type WebGpuAvailability, WebGpuUnavailableError, checkWebGpu, isWebGpuAvailable, requestDevice } from './gpu.js';
export { type PlayOptions, play } from './play.js';
export { Trimmer } from './trim.js';
export { type WeightDtype } from './weights.js';
export { type PlanarAudio, decodeWav, encodeWav } from './wav.js';
