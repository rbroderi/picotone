/**
 * Browser picotone decoder: `.pico` bytes in, planar `Float32Array`s out.
 *
 * Decoding runs in a dedicated module worker (`worker.js`), never on the
 * calling thread. The worker picks the threaded WebAssembly build when the page
 * is cross-origin isolated and the single-threaded SIMD build otherwise; see
 * {@link PicotoneDecoder.build}.
 *
 * ```ts
 * const decoder = await PicotoneDecoder.create({ baseUrl: "/static/picotone/" });
 * await decoder.loadModel("encodec_48khz");
 * const playback = decoder.play(audioContext, picoBytes, { loop: true });
 * ```
 */

import type {
  BuildInfo,
  BuildPreference,
  ChunkResponse,
  DecodeStats,
  ModelInfo,
  ModelName,
  Request,
  Response,
} from "./protocol.js";

export type { BuildInfo, BuildPreference, DecodeStats, ModelInfo, ModelName } from "./protocol.js";
export { type PicoInfo, readPicoHeader } from "./pico-header.js";
import { readPicoHeader } from "./pico-header.js";

/** Options of {@link PicotoneDecoder.create}. */
export interface PicotoneOptions {
  /**
   * Directory holding `worker.js`, `pkg-st/`, `pkg-st-fma/` and `pkg-mt/`. Relative URLs
   * resolve against the document. Defaults to the directory of this module.
   */
  readonly baseUrl?: string | URL;
  /**
   * Directory holding `<model>.json`, `<model>.safetensors`, `cx2-tables.json` and the
   * `.ecxm` tables it lists. Defaults to `<baseUrl>models/`.
   */
  readonly modelsUrl?: string | URL;
  /** Worker script URL. Defaults to `<baseUrl>worker.js`. */
  readonly workerUrl?: string | URL;
  /** Which build to run; `auto` (default) uses threads when the page allows. */
  readonly build?: BuildPreference;
  /** Thread pool size of the threaded build; default `navigator.hardwareConcurrency`. */
  readonly threads?: number;
  /**
   * Whether the single-threaded build may be the relaxed SIMD one (default
   * true). It is used where the browser has relaxed SIMD and is about a third
   * faster; its output differs from the plain build's by float rounding only.
   */
  readonly relaxedSimd?: boolean;
  /**
   * Cache Storage bucket for model files (default `picotone-models-v1`), or
   * null to rely on the HTTP cache only. Cache Storage needs a secure context;
   * elsewhere the HTTP cache is used regardless.
   */
  readonly cacheName?: string | null;
}

/** Decoding progress. */
export interface DecodeProgress {
  readonly segmentsDone: number;
  readonly segmentCount: number;
  /** Final samples per channel emitted so far. */
  readonly samplesDone: number;
  readonly audioLength: number;
}

/** Final samples `[offset, offset + length)` of a decode, planar. */
export interface DecodedChunk {
  readonly sampleRate: number;
  readonly offset: number;
  readonly length: number;
  readonly audioLength: number;
  readonly channels: readonly Float32Array<ArrayBuffer>[];
  /** Milliseconds since the worker started decoding. */
  readonly elapsedMs: number;
}

/** Options of {@link PicotoneDecoder.decode}. */
export interface DecodeOptions {
  readonly onChunk?: (chunk: DecodedChunk) => void;
  readonly onProgress?: (progress: DecodeProgress) => void;
  readonly signal?: AbortSignal;
  /** Segments in the first chunk (default 1, so playback can start early). */
  readonly firstChunkSegments?: number;
  /**
   * Most segments per later chunk (default 16 threaded, 8 single). Chunk
   * sizes double up to this. A segment is ~1 s of audio (for the causal
   * 24 kHz model, 1 s of latent frames). Larger chunks cost a little less
   * per call and hold more memory while decoding; smaller ones arrive sooner.
   */
  readonly chunkSegments?: number;
  /** Transfer (detach) the caller's buffer instead of copying it. */
  readonly transfer?: boolean;
  /**
   * First playable sample to decode (default 0): a seek. Chunk offsets stay
   * counted from the start of the file; the result holds the samples from here
   * on. The output equals the same samples of a whole-file decode.
   */
  readonly startSample?: number;
}

/** A whole decoded file (from `startSample` on, after a seek). */
export interface DecodedAudio {
  readonly sampleRate: number;
  /** Samples per channel. */
  readonly length: number;
  /** First playable sample held (the decode's `startSample`). */
  readonly startSample: number;
  readonly channels: readonly Float32Array<ArrayBuffer>[];
  readonly stats: DecodeStats;
}

/** Options of {@link PicotoneDecoder.play}. */
export interface PlayOptions extends DecodeOptions {
  /** Where to connect the sources (default `context.destination`). */
  readonly destination?: AudioNode;
  /**
   * When to start: `auto` (default) as soon as the buffered audio should
   * outlast the rest of the decode at the measured speed (immediately after
   * the first chunk when decoding is faster than realtime); a number starts
   * once that many seconds are decoded.
   */
  readonly prebuffer?: "auto" | number;
  /** Seconds between "ready" and the first sample, to absorb scheduling jitter (default 0.05). */
  readonly startDelay?: number;
  /**
   * Loop gaplessly once the file is decoded (AudioBufferSourceNode `loop`).
   * A file made with `picotone encode --loop` joins its end to its start
   * without a click, so no loop region is needed; other files click slightly
   * at the seam because EnCodec's first and last few ms are not faithful.
   */
  readonly loop?: boolean;
  /** Loop region in seconds (default: the whole file). */
  readonly loopStart?: number;
  readonly loopEnd?: number;
  /** Called when the first sample is scheduled, with ms since `play()` was called. */
  readonly onStart?: (timeToFirstAudioMs: number) => void;
  /** Called when a chunk arrived too late and playback paused for `gapSeconds`. */
  readonly onUnderrun?: (gapSeconds: number) => void;
}

/** A running {@link PicotoneDecoder.play}. */
export interface Playback {
  /** Resolves with the whole decoded file once decoding finishes. */
  readonly decoded: Promise<DecodedAudio>;
  /** Resolves with the time to first audio in ms once playback is scheduled. */
  readonly started: Promise<number>;
  /** Stops playback and cancels the decode. */
  stop(): void;
  /** Number of underruns so far. */
  readonly underruns: number;
}

type Listener = (message: Response) => void;

/** The decoder worker and its loaded models. */
export class PicotoneDecoder {
  /** The WebAssembly build in use and why it was chosen. */
  readonly build: BuildInfo;
  readonly #worker: Worker;
  readonly #modelsUrl: string;
  readonly #cacheName: string | null;
  readonly #listeners = new Map<number, Listener>();
  #nextId = 1;

  private constructor(worker: Worker, build: BuildInfo, modelsUrl: string, cacheName: string | null) {
    this.#worker = worker;
    this.build = build;
    this.#modelsUrl = modelsUrl;
    this.#cacheName = cacheName;
    worker.addEventListener("message", (event: MessageEvent<Response>) => {
      this.#listeners.get(event.data.id)?.(event.data);
    });
  }

  /** Starts the worker and its WebAssembly build. */
  static async create(options: PicotoneOptions = {}): Promise<PicotoneDecoder> {
    const documentUrl = typeof location === "undefined" ? import.meta.url : location.href;
    const baseUrl = new URL(options.baseUrl ?? new URL(".", import.meta.url), documentUrl).href;
    // Explicit URLs resolve against the document; defaults live under baseUrl.
    const modelsUrl =
      options.modelsUrl === undefined
        ? new URL("models/", baseUrl).href
        : new URL(options.modelsUrl, documentUrl).href;
    const workerUrl =
      options.workerUrl === undefined
        ? new URL("worker.js", baseUrl)
        : new URL(options.workerUrl, documentUrl);
    const worker = new Worker(workerUrl, { type: "module", name: "picotone" });
    const build = await new Promise<BuildInfo>((resolve, reject) => {
      const onMessage = (event: MessageEvent<Response>): void => {
        const message = event.data;
        if (message.id !== 0) {
          return;
        }
        worker.removeEventListener("message", onMessage);
        if (message.type === "ready") {
          resolve(message.build);
        } else if (message.type === "error") {
          reject(new Error(message.message));
        }
      };
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", (event) => {
        reject(new Error(`decoder worker failed to start: ${event.message}`));
      });
      const init: Request = {
        type: "init",
        id: 0,
        baseUrl,
        prefer: options.build ?? "auto",
        threads: options.threads ?? 0,
        relaxedSimd: options.relaxedSimd ?? true,
      };
      worker.postMessage(init);
    });
    const cacheName = options.cacheName === undefined ? "picotone-models-v1" : options.cacheName;
    return new PicotoneDecoder(worker, build, modelsUrl, cacheName);
  }

  #send(request: Request, listener: Listener, transfer: Transferable[] = []): void {
    this.#listeners.set(request.id, listener);
    this.#worker.postMessage(request, transfer);
  }

  /**
   * Fetches (once; later calls hit the cache) and builds a decoder-only model,
   * with the cx2 tables `cx2-tables.json` lists for it. Both models can be
   * loaded at the same time; `decode` picks the one the file names and the
   * table by the hash the file stores.
   */
  loadModel(
    name: ModelName,
    options: { readonly onProgress?: (loaded: number, total: number) => void } = {},
  ): Promise<ModelInfo> {
    const id = this.#nextId++;
    return new Promise<ModelInfo>((resolve, reject) => {
      this.#send(
        { type: "load", id, model: name, modelsUrl: this.#modelsUrl, cacheName: this.#cacheName },
        (message) => {
          if (message.type === "loadProgress") {
            options.onProgress?.(message.loaded, message.total);
          } else if (message.type === "loaded") {
            this.#listeners.delete(id);
            resolve(message.model);
          } else if (message.type === "error") {
            this.#listeners.delete(id);
            reject(new Error(message.message));
          }
        },
      );
    });
  }

  /** Decodes a whole `.pico` file, streaming final chunks to `onChunk` as they finish. */
  decode(bytes: ArrayBuffer | ArrayBufferView, options: DecodeOptions = {}): Promise<DecodedAudio> {
    const info = readPicoHeader(bytes);
    const startSample = options.startSample ?? 0;
    if (!Number.isInteger(startSample) || startSample < 0 || startSample >= info.audioLength) {
      return Promise.reject(new Error(`startSample ${String(startSample)} is outside the file`));
    }
    const buffer = toTransferable(bytes, options.transfer === true);
    const id = this.#nextId++;
    const channels = Array.from(
      { length: info.channels },
      () => new Float32Array(info.audioLength - startSample),
    );
    return new Promise<DecodedAudio>((resolve, reject) => {
      const abort = (): void => {
        this.#worker.postMessage({ type: "cancel", id } satisfies Request);
      };
      if (options.signal?.aborted === true) {
        reject(new Error("decode aborted"));
        return;
      }
      options.signal?.addEventListener("abort", abort, { once: true });
      const finish = (): void => {
        this.#listeners.delete(id);
        options.signal?.removeEventListener("abort", abort);
      };
      this.#send(
        {
          type: "decode",
          id,
          bytes: buffer,
          startSample,
          firstChunkSegments: options.firstChunkSegments ?? 1,
          chunkSegments: options.chunkSegments ?? (this.build.kind === "threaded" ? 16 : 8),
        },
        (message) => {
          if (message.type === "chunk") {
            const chunk = toChunk(message);
            chunk.channels.forEach((data, c) => channels[c]?.set(data, chunk.offset - startSample));
            options.onChunk?.(chunk);
            options.onProgress?.({
              segmentsDone: message.segmentsDone,
              segmentCount: message.segmentCount,
              samplesDone: message.offset + message.length,
              audioLength: message.audioLength,
            });
          } else if (message.type === "done") {
            finish();
            resolve({
              sampleRate: info.sampleRate,
              length: info.audioLength - startSample,
              startSample,
              channels,
              stats: message.stats,
            });
          } else if (message.type === "error") {
            finish();
            reject(new Error(message.message));
          }
        },
        [buffer],
      );
    });
  }

  /** Decodes a whole file into an `AudioBuffer`, e.g. to cache a sound effect or loop. */
  async decodeToAudioBuffer(
    context: BaseAudioContext,
    bytes: ArrayBuffer | ArrayBufferView,
    options: DecodeOptions = {},
  ): Promise<AudioBuffer> {
    return toAudioBuffer(context, await this.decode(bytes, options));
  }

  /**
   * Streams a file into `context`: playback starts once the first chunk (or
   * enough of the file, see {@link PlayOptions.prebuffer}) is decoded. With
   * `startSample`, playback starts there (a seek); a loop still wraps to its
   * `loopStart`.
   */
  play(context: AudioContext, bytes: ArrayBuffer | ArrayBufferView, options: PlayOptions = {}): Playback {
    const t0 = performance.now();
    const info = readPicoHeader(bytes);
    const startSample = options.startSample ?? 0;
    const startSeconds = startSample / info.sampleRate;
    const destination = options.destination ?? context.destination;
    const startDelay = options.startDelay ?? 0.05;
    const prebuffer = options.prebuffer ?? "auto";
    const controller = new AbortController();
    options.signal?.addEventListener("abort", () => controller.abort(), { once: true });
    const sources: AudioBufferSourceNode[] = [];
    const pending: DecodedChunk[] = [];
    let startTime = -1;
    let underruns = 0;
    const duration = info.audioLength / info.sampleRate;
    const loopStart = Math.min(Math.max(0, options.loopStart ?? 0), duration);
    const loopEnd = Math.min(Math.max(loopStart, options.loopEnd ?? duration), duration);
    // Seconds into the file where the streamed copy hands over to the loop.
    const handover = options.loop === true ? loopEnd : null;
    let lastChunkAt = t0;
    let lastChunkWall = 0;
    let resolveStarted: (ms: number) => void = () => {};
    const started = new Promise<number>((resolve) => {
      resolveStarted = resolve;
    });

    const schedule = (chunk: DecodedChunk): void => {
      const margin = 0.01;
      let when = startTime + (chunk.offset - startSample) / chunk.sampleRate;
      if (when < context.currentTime + margin) {
        const gap = context.currentTime + margin - when;
        startTime += gap;
        when += gap;
        underruns += 1;
        options.onUnderrun?.(gap);
      }
      const source = context.createBufferSource();
      source.buffer = toAudioBuffer(context, chunk);
      source.connect(destination);
      source.start(when);
      if (handover !== null) {
        source.stop(Math.max(when, startTime + handover - startSeconds));
      }
      sources.push(source);
    };

    const readyToStart = (chunk: DecodedChunk): boolean => {
      const decodedSeconds = (chunk.offset + chunk.length - startSample) / chunk.sampleRate;
      const totalSeconds = (chunk.audioLength - startSample) / chunk.sampleRate;
      if (decodedSeconds >= totalSeconds) {
        return true;
      }
      if (typeof prebuffer === "number") {
        return decodedSeconds >= prebuffer;
      }
      // Decode speed so far (× realtime) and the buffer needed to outlast the rest.
      const speed = decodedSeconds / Math.max(1e-3, chunk.elapsedMs / 1000);
      const remaining = totalSeconds - decodedSeconds;
      const needed = speed >= 1 ? 0 : (remaining * (1 - speed)) / speed;
      // Chunks grow, so the next one may take about twice as long as the last.
      return decodedSeconds >= Math.max(needed * 1.1, 2 * lastChunkWall);
    };

    const decoded = this.decode(bytes, {
      ...options,
      signal: controller.signal,
      onChunk: (chunk) => {
        const now = performance.now();
        lastChunkWall = (now - lastChunkAt) / 1000;
        lastChunkAt = now;
        options.onChunk?.(chunk);
        if (controller.signal.aborted) {
          return;
        }
        if (startTime >= 0) {
          schedule(chunk);
          return;
        }
        pending.push(chunk);
        if (readyToStart(chunk)) {
          startTime = context.currentTime + startDelay;
          for (const queued of pending.splice(0)) {
            schedule(queued);
          }
          const ms = performance.now() - t0 + startDelay * 1000;
          options.onStart?.(ms);
          resolveStarted(ms);
        }
      },
    });

    if (options.loop === true) {
      void decoded.then(async (audio) => {
        if (controller.signal.aborted) {
          return;
        }
        // After a seek the decode lacks the file's start; the loop needs all of it.
        if (audio.startSample > 0) {
          audio = await this.decode(bytes, { signal: controller.signal });
        }
        // Hand over to one looping source exactly where the streamed copy stops
        // (loopEnd), continuing from loopStart.
        const source = context.createBufferSource();
        source.buffer = toAudioBuffer(context, audio);
        source.loop = true;
        source.loopStart = loopStart;
        source.loopEnd = loopEnd;
        source.connect(destination);
        const at = startTime + loopEnd - startSeconds;
        if (at >= context.currentTime) {
          source.start(at, loopStart);
        } else {
          // Decoding finished after the streamed copy ran out (underrun): join the loop in phase.
          const late = context.currentTime - at;
          const span = Math.max(1e-3, loopEnd - loopStart);
          source.start(context.currentTime, loopStart + (late % span));
        }
        sources.push(source);
      });
    }

    return {
      decoded,
      started,
      stop: () => {
        controller.abort();
        for (const source of sources) {
          try {
            source.stop();
          } catch {
            // Not started yet or already stopped.
          }
          source.disconnect();
        }
      },
      get underruns() {
        return underruns;
      },
    };
  }

  /** Terminates the worker. */
  dispose(): void {
    this.#worker.terminate();
    this.#listeners.clear();
  }
}

function toChunk(message: ChunkResponse): DecodedChunk {
  return {
    sampleRate: message.sampleRate,
    offset: message.offset,
    length: message.length,
    audioLength: message.audioLength,
    channels: message.channels,
    elapsedMs: message.elapsedMs,
  };
}

function toTransferable(bytes: ArrayBuffer | ArrayBufferView, transfer: boolean): ArrayBuffer {
  if (bytes instanceof ArrayBuffer) {
    return transfer ? bytes : bytes.slice(0);
  }
  const whole =
    bytes.buffer instanceof ArrayBuffer &&
    bytes.byteOffset === 0 &&
    bytes.byteLength === bytes.buffer.byteLength;
  if (transfer && whole) {
    return bytes.buffer as ArrayBuffer;
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return copy.buffer;
}

/** Copies planar audio into an `AudioBuffer` of its own sample rate. */
export function toAudioBuffer(
  context: BaseAudioContext,
  audio: { readonly sampleRate: number; readonly channels: readonly Float32Array<ArrayBuffer>[] },
): AudioBuffer {
  const length = audio.channels[0]?.length ?? 0;
  const buffer = context.createBuffer(audio.channels.length, Math.max(1, length), audio.sampleRate);
  audio.channels.forEach((data, c) => buffer.copyToChannel(data, c));
  return buffer;
}

/** Encodes planar audio as a WAV file (32-bit float by default, or 16-bit PCM). */
export function encodeWav(
  audio: { readonly sampleRate: number; readonly channels: readonly Float32Array[] },
  format: "float32" | "int16" = "float32",
): ArrayBuffer {
  const channels = audio.channels.length;
  const length = audio.channels[0]?.length ?? 0;
  const bytesPerSample = format === "float32" ? 4 : 2;
  const dataBytes = length * channels * bytesPerSample;
  const out = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(out);
  const text = (at: number, value: string): void => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(at + i, value.charCodeAt(i));
    }
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, format === "float32" ? 3 : 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  text(36, "data");
  view.setUint32(40, dataBytes, true);
  let at = 44;
  for (let t = 0; t < length; t += 1) {
    for (const data of audio.channels) {
      const v = data[t] ?? 0;
      if (format === "float32") {
        view.setFloat32(at, v, true);
      } else {
        const clamped = Math.max(-1, Math.min(1, v));
        view.setInt16(at, Math.round(clamped * 32767), true);
      }
      at += bytesPerSample;
    }
  }
  return out;
}
