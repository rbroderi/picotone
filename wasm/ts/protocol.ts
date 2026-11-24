/**
 * Messages between {@link PicotoneDecoder} (any thread) and its dedicated
 * decoder worker (`worker.js`). Shared by both sides; no runtime code.
 */

/**
 * The code spaces a `.pico` file can use, as its header names them: the two
 * published EnCodec models, and `dac44`, the HQ codec's DAC 44.1 kHz codes
 * (decoded by the `dac44_student` or `dac44_vocos` network).
 */
export type ModelName = "encodec_24khz" | "encodec_48khz" | "dac44";

/**
 * The networks that decode `dac44` codes. An HQ file names its code space,
 * not a decoder, so either plays any HQ file: `dac44_student` (the default,
 * a distilled DAC decoder) or `dac44_vocos` (Vocos, an iSTFT decoder with
 * about a fifth of the work).
 */
export type HqDecoderName = "dac44_student" | "dac44_vocos";

/** Which WebAssembly build to run. `auto` picks `threaded` when possible. */
export type BuildPreference = "auto" | "threaded" | "single";

/** The build the worker chose, and why. */
export interface BuildInfo {
  /** `threaded`: rayon over a Web Worker pool. `single`: one thread, SIMD only. */
  readonly kind: "threaded" | "single";
  /**
   * Whether the build uses relaxed SIMD (fused multiply-add). Only the
   * single-threaded build has such a variant; it is about a third faster.
   */
  readonly relaxedSimd: boolean;
  /** Rayon threads (1 for `single`). */
  readonly threads: number;
  /** `self.crossOriginIsolated` as seen by the worker. */
  readonly crossOriginIsolated: boolean;
  /** `self.isSecureContext` as seen by the worker. */
  readonly isSecureContext: boolean;
  /** Human-readable reason for the choice (why threads are unavailable, if so). */
  readonly reason: string;
}

/** A loaded decoder model. */
export interface ModelInfo {
  readonly name: ModelName;
  /** The network that decodes it: the model name, or for `dac44` an {@link HqDecoderName}. */
  readonly network: string;
  readonly sampleRate: number;
  readonly channels: number;
  /** Bytes of weights fetched (0 when served from Cache Storage). */
  readonly fetchedBytes: number;
  /** Whether the weights came from Cache Storage. */
  readonly fromCache: boolean;
  /** Hashes (16 hex digits) of the cx2 tables loaded for the model. */
  readonly tables: readonly string[];
  /** Time to fetch and build the model, in milliseconds. */
  readonly loadMs: number;
  /**
   * WebAssembly memory size after the load, tables included. It never
   * shrinks, so this is the peak so far; it stays the worker's footprint.
   */
  readonly memoryBytes: number;
  /**
   * WebAssembly memory size at each phase of building the network, in
   * order: `start` (before), the network's own phases, `built` (after).
   * Empty when the network was already built.
   */
  readonly loadMemory: Readonly<Record<string, number>>;
}

/** Timing and memory of one decode. */
export interface DecodeStats {
  /** Wall time of the whole decode in the worker, in milliseconds. */
  readonly decodeMs: number;
  /** Wall time until the first chunk was ready, in milliseconds. */
  readonly firstChunkMs: number;
  /** Decoded audio duration in seconds. */
  readonly audioSeconds: number;
  /** Audio seconds per wall-clock second (× realtime). */
  readonly speed: number;
  /** Of `decodeMs`, the cx2 (entropy) decode of the codes, in milliseconds. */
  readonly cx2Ms: number;
  /** Of `decodeMs`, the network, stereo and trim, in milliseconds. */
  readonly networkMs: number;
  /** WebAssembly memory size after the first chunk. */
  readonly firstChunkMemoryBytes: number;
  /** WebAssembly memory size after the decode (its peak: it never shrinks). */
  readonly memoryBytes: number;
}

export interface InitRequest {
  readonly type: "init";
  readonly id: number;
  /** URL of the directory holding `pkg-st/`, `pkg-st-fma/` and `pkg-mt/`. */
  readonly baseUrl: string;
  readonly prefer: BuildPreference;
  /** Whether the single-threaded build may be the relaxed SIMD one. */
  readonly relaxedSimd: boolean;
  /** Thread pool size for the threaded build; 0 means `navigator.hardwareConcurrency`. */
  readonly threads: number;
}

export interface LoadRequest {
  readonly type: "load";
  readonly id: number;
  readonly model: ModelName;
  /**
   * URL of the directory holding `<model>.json`, `<model>.safetensors`,
   * `cx2-tables.json` and the `.ecxm` tables it lists.
   */
  readonly modelsUrl: string;
  /** Cache Storage bucket for model files, or null to rely on the HTTP cache. */
  readonly cacheName: string | null;
  /** For `dac44`, the network to decode with (default `dac44_student`). */
  readonly hqDecoder?: HqDecoderName;
}

export interface DecodeRequest {
  readonly type: "decode";
  readonly id: number;
  readonly bytes: ArrayBuffer;
  /** First playable sample to decode (a seek; 0 for the whole file). */
  readonly startSample: number;
  /** Segments in the first chunk (smaller starts playback sooner). */
  readonly firstChunkSegments: number;
  /** Largest number of segments per later chunk. */
  readonly chunkSegments: number;
}

export interface CancelRequest {
  readonly type: "cancel";
  readonly id: number;
}

export type Request = InitRequest | LoadRequest | DecodeRequest | CancelRequest;

export interface ReadyResponse {
  readonly type: "ready";
  readonly id: number;
  readonly build: BuildInfo;
}

export interface LoadProgressResponse {
  readonly type: "loadProgress";
  readonly id: number;
  readonly loaded: number;
  /** Total bytes, or 0 when the server sent no length. */
  readonly total: number;
}

export interface LoadedResponse {
  readonly type: "loaded";
  readonly id: number;
  readonly model: ModelInfo;
}

export interface ChunkResponse {
  readonly type: "chunk";
  readonly id: number;
  readonly sampleRate: number;
  /** First sample of the chunk, per channel, from the start of the file. */
  readonly offset: number;
  readonly length: number;
  /** Total samples per channel in the file. */
  readonly audioLength: number;
  /** One planar buffer per channel (transferred). */
  readonly channels: Float32Array<ArrayBuffer>[];
  readonly segmentsDone: number;
  readonly segmentCount: number;
  /** Milliseconds since the decode started. */
  readonly elapsedMs: number;
}

export interface DoneResponse {
  readonly type: "done";
  readonly id: number;
  readonly stats: DecodeStats;
}

export interface ErrorResponse {
  readonly type: "error";
  readonly id: number;
  readonly message: string;
}

export type Response =
  | ReadyResponse
  | LoadProgressResponse
  | LoadedResponse
  | ChunkResponse
  | DoneResponse
  | ErrorResponse;
