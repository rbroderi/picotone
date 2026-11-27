/**
 * EnCodec decoder on WebGPU.
 *
 * Every op runs on the GPU: codebook lookup, padded convolutions, polyphase
 * transposed convolutions, group norm, ELU (fused into convolution loads),
 * the LSTM (input projection as a GEMM, then one dispatch per timestep for the
 * whole batch, all inside the chunk's single command buffer) and, for the
 * 48 kHz model, scaling plus triangle overlap-add.
 *
 * Bounded GPU memory:
 * - Whole-file models (24 kHz, causal, weight norm) are decoded as a stream of
 *   chunks of at most `chunkFrames` latent frames. Because every layer is
 *   causal and has no global normalisation, the decoder carries the exact
 *   state between chunks: the last `kernel - 1` input frames of every
 *   convolution (`taps - 1` for transposed convolutions) and the LSTM h/c
 *   state. The first chunk uses the reflect padding of the full decode. The
 *   result equals a whole-file decode while activation buffers stay sized by
 *   `chunkFrames`, independent of file length.
 * - Segmented models (48 kHz) decode `batchSize` independent 1 s segments per
 *   command buffer. Overlap-add happens on the GPU per batch, carrying the
 *   overlapping tail of the batch's last segment to the next batch.
 * - Activations live in a few pooled buffers reused across levels (two
 *   ping-pong buffers, one residual hidden buffer, two pre-norm buffers), and
 *   output is read back per chunk through a small ring of staging buffers, so
 *   only codes and the final PCM ever scale with the file length (on the CPU).
 */

import {
  type ModelConfig,
  codebookDim,
  hopLength,
  segmentLength,
  segmentStride,
} from './config.js';
import { type PicoCodes, Cx2Reader, PicoError, parseHeader } from './pico.js';
import { Limiter } from './limit.js';
import { Trimmer } from './trim.js';
import { type Kernel, makeKernel, requestDevice } from './gpu.js';
import {
  ADD_WGSL,
  CONV_SMALL_WGSL,
  CONV_WGSL,
  GN_APPLY_WGSL,
  GN_PARTS,
  GN_STATS_WGSL,
  lstmStepWgsl,
  OLA_WGSL,
  QUANTIZER_WGSL,
  UNIFORM_SLOT,
} from './shaders.js';
import { type FetchProgress, type LoadedWeights, type WeightDtype, fetchWeights, floatToHalf } from './weights.js';

export interface LoadOptions {
  /** Model name, `encodec_24khz` or `encodec_48khz`. */
  readonly model: string;
  /**
   * URL of the folder holding `<model>/manifest.json`, `cx2-tables.json`, the
   * `.ecxm` tables it lists and `cx2/` (the WebAssembly cx2 decoder). Default `models/`.
   */
  readonly baseUrl?: string;
  /** Folder of `picotone.js` and `picotone_bg.wasm`, if not `<baseUrl>/cx2/`. */
  readonly cx2Url?: string;
  /** Weight precision to download (computation is always f32). Default `f32`. */
  readonly dtype?: WeightDtype;
  /** Existing device to share; otherwise one is requested. */
  readonly device?: GPUDevice;
  readonly onProgress?: FetchProgress;
  /** Segments per command buffer for segmented models (1-8). Default 4. */
  readonly batchSize?: number;
  /** Latent frames per chunk for whole-file models. Default 375 (5 s at 24 kHz). */
  readonly chunkFrames?: number;
  /** Latent frames of the first chunk, for a quick first audio. Default 75. */
  readonly firstChunkFrames?: number;
}

/** A piece of decoded audio, planar. */
export interface DecodedChunk {
  /** First sample (per channel) of this chunk in the file. */
  readonly offset: number;
  readonly sampleRate: number;
  readonly channels: readonly Float32Array[];
}

export interface DecodeOptions {
  /** Called with each finished piece of audio, in order. */
  readonly onChunk?: (chunk: DecodedChunk) => void;
  /** Called with the decoded fraction in [0, 1]. */
  readonly onProgress?: (fraction: number) => void;
  readonly signal?: AbortSignal;
  /** Keep the whole output (default true). With false, `channels` are empty. */
  readonly collect?: boolean;
  /**
   * Debug: time every dispatch with GPU timestamps (needs the
   * `timestamp-query` feature; serialises submissions, so slower overall).
   */
  readonly profile?: boolean;
}

export interface DecodedAudio {
  readonly sampleRate: number;
  /** Samples per channel. */
  readonly length: number;
  /** Planar audio, one Float32Array per channel. */
  readonly channels: readonly Float32Array[];
  /**
   * Whether the file was encoded as a gapless loop (`encodec-rs encode
   * --loop`): its last sample runs into its first, so loop the whole buffer
   * (`AudioBufferSourceNode.loop = true`) with no loop points.
   */
  readonly loop: boolean;
  /** GPU milliseconds per `kernel:layer`, when profiling was requested. */
  readonly profileMs?: Readonly<Record<string, number>>;
}

interface Tensor {
  readonly buf: GPUBuffer;
  /** Row offset of frame 0 within each batch item. */
  readonly start: number;
  /** Rows per batch item. */
  readonly item: number;
}

interface Norm {
  readonly gamma: GPUBuffer;
  readonly beta: GPUBuffer;
}

interface ConvLayer {
  readonly name: string;
  readonly cin: number;
  readonly cout: number;
  readonly kernel: number;
  readonly dilation: number;
  readonly w: GPUBuffer;
  readonly b: GPUBuffer;
  readonly norm: Norm | null;
  /** History rows carried between chunks (streaming only). */
  hist: GPUBuffer | null;
}

interface ConvTLayer {
  readonly name: string;
  readonly cin: number;
  readonly cout: number;
  readonly kernel: number;
  readonly stride: number;
  readonly taps: number;
  readonly w: GPUBuffer;
  readonly b: GPUBuffer;
  readonly norm: Norm | null;
  hist: GPUBuffer | null;
}

interface ResLayer {
  readonly conv1: ConvLayer;
  readonly conv2: ConvLayer;
  readonly shortcut: ConvLayer;
}

interface LstmLayer {
  readonly wih: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly whh: GPUBuffer;
}

interface Stage {
  readonly up: ConvTLayer;
  readonly res: readonly ResLayer[];
}

type Op =
  | {
      readonly kind: 'dispatch';
      readonly kernel: Kernel;
      readonly group: GPUBindGroup;
      readonly wg: readonly [number, number, number];
      readonly dyn?: number;
      readonly label: string;
    }
  | { readonly kind: 'copy'; readonly src: GPUBuffer; readonly srcOff: number; readonly dst: GPUBuffer; readonly dstOff: number; readonly size: number }
  | { readonly kind: 'clear'; readonly buf: GPUBuffer };

const MAX_SLOTS = 2048;
const MAX_BATCH = 8;
/** Hidden units per LSTM-step workgroup. */
const LSTM_UNITS = 4;

/** Records one command buffer's worth of work, with per-op uniforms in an arena. */
class Recorder {
  readonly ops: Op[] = [];
  /** Label attached to the following dispatches (for profiling). */
  tag = '';
  private slots = 0;
  private readonly data = new ArrayBuffer(MAX_SLOTS * UNIFORM_SLOT);
  private readonly u32 = new Uint32Array(this.data);
  private readonly f32 = new Float32Array(this.data);

  constructor(
    private readonly device: GPUDevice,
    private readonly arena: GPUBuffer,
  ) {}

  /** Stores uniforms; entries given as `{ f: value }` are written as f32. */
  uniform(values: readonly (number | { readonly f: number })[]): GPUBufferBinding {
    if (this.slots >= MAX_SLOTS) throw new Error('uniform arena exhausted');
    const base = (this.slots * UNIFORM_SLOT) / 4;
    values.forEach((v, i) => {
      if (typeof v === 'number') this.u32[base + i] = v;
      else this.f32[base + i] = v.f;
    });
    const binding = { buffer: this.arena, offset: this.slots * UNIFORM_SLOT, size: UNIFORM_SLOT };
    this.slots++;
    return binding;
  }

  group(kernel: Kernel, resources: readonly GPUBufferBinding[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: kernel.layout,
      entries: resources.map((resource, binding) => ({ binding, resource })),
    });
  }

  dispatch(
    kernel: Kernel,
    params: readonly (number | { readonly f: number })[],
    buffers: readonly GPUBuffer[],
    wg: readonly [number, number, number],
  ): void {
    const group = this.group(kernel, [this.uniform(params), ...buffers.map((buffer) => ({ buffer }))]);
    this.ops.push({ kind: 'dispatch', kernel, group, wg, label: `${kernel.label}:${this.tag}` });
  }

  dispatchDynamic(kernel: Kernel, group: GPUBindGroup, dyn: number, wg: readonly [number, number, number]): void {
    this.ops.push({ kind: 'dispatch', kernel, group, wg, dyn, label: `${kernel.label}:${this.tag}` });
  }

  copy(src: GPUBuffer, srcOff: number, dst: GPUBuffer, dstOff: number, size: number): void {
    if (size > 0) this.ops.push({ kind: 'copy', src, srcOff, dst, dstOff, size });
  }

  clear(buf: GPUBuffer): void {
    this.ops.push({ kind: 'clear', buf });
  }

  /**
   * Encodes the recorded work. With `timestamps`, every dispatch gets its own
   * pass with begin/end timestamps; the returned labels name each pair.
   */
  encode(encoder: GPUCommandEncoder, timestamps?: GPUQuerySet): string[] {
    this.device.queue.writeBuffer(this.arena, 0, this.data, 0, Math.max(1, this.slots) * UNIFORM_SLOT);
    let pass: GPUComputePassEncoder | null = null;
    let current: GPUComputePipeline | null = null;
    const labels: string[] = [];
    for (const op of this.ops) {
      if (op.kind === 'dispatch') {
        if (timestamps && labels.length * 2 + 2 <= timestamps.count) {
          if (pass) pass.end();
          pass = encoder.beginComputePass({
            timestampWrites: {
              querySet: timestamps,
              beginningOfPassWriteIndex: labels.length * 2,
              endOfPassWriteIndex: labels.length * 2 + 1,
            },
          });
          labels.push(op.label);
          current = null;
        } else if (!pass) {
          pass = encoder.beginComputePass();
          current = null;
        }
        if (current !== op.kernel.pipeline) {
          pass.setPipeline(op.kernel.pipeline);
          current = op.kernel.pipeline;
        }
        if (op.dyn === undefined) pass.setBindGroup(0, op.group);
        else pass.setBindGroup(0, op.group, [op.dyn]);
        pass.dispatchWorkgroups(op.wg[0], op.wg[1], op.wg[2]);
      } else {
        if (pass) {
          pass.end();
          pass = null;
        }
        if (op.kind === 'copy') encoder.copyBufferToBuffer(op.src, op.srcOff, op.dst, op.dstOff, op.size);
        else encoder.clearBuffer(op.buf);
      }
    }
    if (pass) pass.end();
    return labels;
  }
}

function gridGroups(total: number): number {
  return Math.max(1, Math.min(4096, Math.ceil(total / 256)));
}

interface Kernels {
  readonly conv: Kernel;
  readonly convSmall: Kernel;
  readonly lstm: Kernel;
  readonly quant: Kernel;
  readonly add: Kernel;
  readonly gnStats: Kernel;
  readonly gnApply: Kernel;
  readonly ola: Kernel;
}

/** Per-command-buffer context. */
interface Pass {
  readonly rec: Recorder;
  /** Batch items. */
  readonly batch: number;
  /** Whole-file streaming: carry state between chunks. */
  readonly streaming: boolean;
  readonly first: boolean;
  readonly last: boolean;
}

interface Staging {
  readonly buf: GPUBuffer;
}

/** A loaded EnCodec decoder bound to a GPU device. */
export class PicotoneWebGpu {
  readonly name: string;
  readonly config: ModelConfig;
  readonly sampleRate: number;
  readonly channels: number;
  readonly dtype: WeightDtype;
  /** Whole-file causal model decoded as a stream of chunks. */
  readonly streaming: boolean;
  /** The cx2 range decoder (WebAssembly) and its tables. */
  readonly cx2: Cx2Reader;

  private readonly k: Kernels;
  private readonly owned: GPUBuffer[] = [];
  private weightBytes = 0;
  private activationBytes = 0;

  private readonly conv0: ConvLayer;
  private readonly lstm: readonly LstmLayer[];
  private readonly stages: readonly Stage[];
  private readonly convOut: ConvLayer;
  private readonly codebooks: GPUBuffer;
  private readonly totalCodebooks: number;
  private readonly dim: number;
  private readonly latentDim: number;
  private readonly hop: number;
  /** Max rows of history before frame 0 in pooled input buffers. */
  private readonly prefix: number;

  /** Max latent frames per batch item. */
  private readonly maxFrames: number;
  private readonly firstFrames: number;
  private readonly maxBatch: number;

  // Activation buffers (allocated once).
  private readonly arena: GPUBuffer;
  private readonly stepUniforms: GPUBuffer;
  private readonly codes: GPUBuffer;
  private readonly emb: GPUBuffer;
  private readonly a0: GPUBuffer;
  private readonly gx: GPUBuffer;
  private readonly h: readonly GPUBuffer[];
  private readonly c: readonly GPUBuffer[];
  private readonly hState: readonly GPUBuffer[];
  private readonly pool: readonly [GPUBuffer, GPUBuffer];
  private readonly hidden: GPUBuffer;
  private readonly tmp: readonly [GPUBuffer, GPUBuffer];
  private readonly parts: readonly [GPUBuffer, GPUBuffer];
  private readonly out: GPUBuffer;
  private readonly scales: GPUBuffer;
  private readonly tail: GPUBuffer;
  private readonly olaOut: GPUBuffer;
  private readonly readbackBytes: number;
  private readonly staging: Staging[] = [];

  private constructor(
    readonly device: GPUDevice,
    weights: LoadedWeights,
    opts: LoadOptions,
    cx2: Cx2Reader,
  ) {
    this.cx2 = cx2;
    const cfg = weights.manifest.config;
    this.name = weights.manifest.name;
    this.config = cfg;
    this.dtype = weights.dtype;
    this.sampleRate = cfg.sampling_rate;
    this.channels = cfg.audio_channels;
    this.totalCodebooks = weights.manifest.total_codebooks;
    this.hop = hopLength(cfg);
    this.latentDim = codebookDim(cfg);
    this.dim = (1 << cfg.upsampling_ratios.length) * cfg.num_filters;
    if (cfg.pad_mode === 'replicate') throw new Error('replicate padding is not supported');
    if (!cfg.use_conv_shortcut) throw new Error('models without a conv shortcut are not supported');

    this.k = {
      conv: makeKernel(device, 'conv', CONV_WGSL, ['uniform', 'read', 'read', 'read', 'rw']),
      convSmall: makeKernel(device, 'conv-small', CONV_SMALL_WGSL, ['uniform', 'read', 'read', 'read', 'rw']),
      lstm: makeKernel(device, 'lstm', lstmStepWgsl(LSTM_UNITS, this.dim, weights.dtype === 'f16'), ['uniform', 'uniform-dynamic', 'read', 'read', 'rw', 'rw']),
      quant: makeKernel(device, 'quantizer', QUANTIZER_WGSL, ['uniform', 'read', 'read', 'rw']),
      add: makeKernel(device, 'add', ADD_WGSL, ['uniform', 'read', 'read', 'rw']),
      gnStats: makeKernel(device, 'gn-stats', GN_STATS_WGSL, ['uniform', 'read', 'rw']),
      gnApply: makeKernel(device, 'gn-apply', GN_APPLY_WGSL, ['uniform', 'read', 'read', 'read', 'read', 'rw']),
      ola: makeKernel(device, 'ola', OLA_WGSL, ['uniform', 'read', 'read', 'rw', 'rw']),
    };

    // ---- weights, built in Hugging Face layer order (Builder::decoder) ----
    const t = weights.tensors;
    const upload = (name: string): GPUBuffer => {
      const tensor = t.get(name);
      if (!tensor) throw new Error(`weights are missing ${name}`);
      const buf = this.createBuffer(tensor.data.length * 4, GPUBufferUsage.STORAGE, true, name);
      new Float32Array(buf.getMappedRange()).set(tensor.data);
      buf.unmap();
      this.weightBytes += buf.size;
      return buf;
    };
    const packedHalf = weights.dtype === 'f16';
    const uploadHalf = (name: string): GPUBuffer => {
      const tensor = t.get(name);
      if (!tensor) throw new Error(`weights are missing ${name}`);
      const buf = this.createBuffer(tensor.data.length * 2, GPUBufferUsage.STORAGE, true, name);
      const out = new Uint16Array(buf.getMappedRange());
      tensor.data.forEach((v, i) => (out[i] = floatToHalf(v)));
      buf.unmap();
      this.weightBytes += buf.size;
      return buf;
    };
    const norm = (prefix: string): Norm | null =>
      cfg.norm_type === 'time_group_norm' ? { gamma: upload(`${prefix}.gamma`), beta: upload(`${prefix}.beta`) } : null;
    const conv = (name: string, cin: number, cout: number, kernel: number, dilation: number): ConvLayer => {
      if (cin % 16 !== 0) throw new Error(`${name}: input channels must be a multiple of 16`);
      return { name, cin, cout, kernel, dilation, w: upload(`${name}.w`), b: upload(`${name}.b`), norm: norm(name), hist: null };
    };
    let idx = 0;
    const next = (): string => `decoder.layers.${idx++}`;
    let scaling = 1 << cfg.upsampling_ratios.length;
    this.conv0 = conv(next(), cfg.hidden_size, this.dim, cfg.last_kernel_size, 1);
    const lstmName = next();
    this.lstm = Array.from({ length: cfg.num_lstm_layers }, (_, l) => ({
      wih: upload(`${lstmName}.w_ih_l${l}`),
      bias: upload(`${lstmName}.b_l${l}`),
      whh: packedHalf ? uploadHalf(`${lstmName}.w_hh_l${l}`) : upload(`${lstmName}.w_hh_l${l}`),
    }));
    const stages: Stage[] = [];
    for (const ratio of cfg.upsampling_ratios) {
      const scale = scaling * cfg.num_filters;
      idx++; // ELU
      const name = next();
      const cin = scale;
      if (cin % 16 !== 0) throw new Error(`${name}: input channels must be a multiple of 16`);
      const kernel = ratio * 2;
      const up: ConvTLayer = {
        name, cin, cout: scale / 2, kernel, stride: ratio, taps: Math.ceil(kernel / ratio),
        w: upload(`${name}.w`), b: upload(`${name}.b`), norm: norm(name), hist: null,
      };
      const res: ResLayer[] = [];
      for (let j = 0; j < cfg.num_residual_layers; j++) {
        const prefix = next();
        const d = scale / 2;
        const hid = d / cfg.compress;
        res.push({
          conv1: conv(`${prefix}.block.1`, d, hid, cfg.residual_kernel_size, cfg.dilation_growth_rate ** j),
          conv2: conv(`${prefix}.block.3`, hid, d, 1, 1),
          shortcut: conv(`${prefix}.shortcut`, d, d, 1, 1),
        });
      }
      stages.push({ up, res });
      scaling /= 2;
    }
    this.stages = stages;
    idx++; // ELU
    this.convOut = conv(next(), cfg.num_filters, cfg.audio_channels, cfg.last_kernel_size, 1);
    this.codebooks = upload('quantizer.embed');

    // ---- geometry ----
    const segLen = segmentLength(cfg);
    const trimLeftZero = this.stages.every((s) => {
      const pt = s.up.kernel - s.up.stride;
      return pt - Math.ceil(pt * cfg.trim_right_ratio) === 0;
    });
    this.streaming = segLen === null && cfg.use_causal_conv && cfg.norm_type === 'weight_norm' && trimLeftZero;
    if (segLen === null && !this.streaming) {
      throw new Error('whole-file models must be causal without group norm to be decoded in bounded memory');
    }
    if (this.streaming) {
      this.maxBatch = 1;
      this.maxFrames = Math.max(8, Math.floor(opts.chunkFrames ?? 375));
      this.firstFrames = Math.min(this.maxFrames, Math.max(8, Math.floor(opts.firstChunkFrames ?? 75)));
    } else {
      this.maxBatch = Math.min(MAX_BATCH, Math.max(1, Math.floor(opts.batchSize ?? 4)));
      this.maxFrames = Math.ceil(((segLen ?? 0) * Math.ceil(cfg.sampling_rate / this.hop)) / cfg.sampling_rate);
      this.firstFrames = this.maxFrames;
    }
    const convs = this.allConvs();
    this.prefix = Math.max(
      1,
      ...convs.map((c) => (c.kernel - 1) * c.dilation),
      ...this.stages.map((s) => s.up.taps - 1),
    );

    // ---- activations ----
    const B = this.maxBatch;
    const N = this.maxFrames;
    const P = this.prefix;
    const act = (floats: number, label: string): GPUBuffer => {
      const buf = this.createBuffer(floats * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, false, label);
      this.activationBytes += buf.size;
      return buf;
    };
    let poolFloats = (P + N) * this.dim;
    let hiddenFloats = 0;
    let tmpFloats = N * this.dim;
    let rows = N;
    for (const s of this.stages) {
      tmpFloats = Math.max(tmpFloats, ((rows - 1) * s.up.stride + s.up.kernel) * s.up.cout);
      rows *= s.up.stride;
      poolFloats = Math.max(poolFloats, (P + rows) * s.up.cout);
      for (const r of s.res) hiddenFloats = Math.max(hiddenFloats, rows * r.conv1.cout);
      tmpFloats = Math.max(tmpFloats, rows * s.up.cout);
    }
    const hasNorm = cfg.norm_type === 'time_group_norm';
    const outRows = N * this.hop;
    this.arena = this.createBuffer(MAX_SLOTS * UNIFORM_SLOT, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, false, 'uniforms');
    this.stepUniforms = this.createBuffer(N * UNIFORM_SLOT, GPUBufferUsage.UNIFORM, true, 'lstm-steps');
    {
      const u = new Uint32Array(this.stepUniforms.getMappedRange());
      for (let s = 0; s < N; s++) u[(s * UNIFORM_SLOT) / 4] = s;
      this.stepUniforms.unmap();
    }
    this.codes = act(B * N * this.totalCodebooks, 'codes');
    this.emb = act(B * (P + N) * this.latentDim, 'emb');
    this.a0 = act(B * N * this.dim, 'lstm-in');
    this.gx = act(B * N * 4 * this.dim, 'lstm-gx');
    this.h = this.lstm.map((_, l) => act(B * (N + 1) * this.dim, `lstm-h${l}`));
    this.c = this.lstm.map((_, l) => act(B * this.dim, `lstm-c${l}`));
    this.hState = this.streaming ? this.lstm.map((_, l) => act(this.dim, `lstm-state${l}`)) : [];
    this.pool = [act(B * poolFloats, 'pool-a'), act(B * poolFloats, 'pool-b')];
    this.hidden = act(B * hiddenFloats, 'res-hidden');
    this.tmp = hasNorm ? [act(B * tmpFloats, 'prenorm-a'), act(B * tmpFloats, 'prenorm-b')] : [this.hidden, this.hidden];
    this.parts = [act(B * 2 * GN_PARTS, 'gn-part-a'), act(B * 2 * GN_PARTS, 'gn-part-b')];
    this.out = act(B * outRows * this.channels, 'out');
    // Files may set any stride from half a segment to a whole one (the ID header's stride).
    const segLength = segmentLength(cfg) ?? 0;
    const minStride = Math.ceil(segLength / 2);
    const maxStride = segLength;
    this.scales = act(B, 'scales');
    this.tail = act(Math.max(1, outRows - minStride) * this.channels, 'ola-tail');
    this.olaOut = this.streaming ? this.out : act((B * maxStride + outRows) * this.channels, 'ola-out');
    this.readbackBytes = (this.streaming ? outRows : B * maxStride + outRows) * this.channels * 4;
    if (this.streaming) {
      for (const c of convs) {
        const hist = (c.kernel - 1) * c.dilation;
        if (hist > 0) c.hist = act(hist * c.cin, `${c.name}.hist`);
      }
      for (const s of this.stages) if (s.up.taps > 1) s.up.hist = act((s.up.taps - 1) * s.up.cin, `${s.up.name}.hist`);
    }
  }

  /** Fetches weights and builds the decoder. */
  static async load(opts: LoadOptions): Promise<PicotoneWebGpu> {
    const device = opts.device ?? (await requestDevice());
    const base = opts.baseUrl ?? 'models/';
    const url = `${base.endsWith('/') ? base : `${base}/`}${opts.model}/`;
    const [weights, cx2] = await Promise.all([
      fetchWeights(url, opts.dtype ?? 'f32', opts.onProgress),
      Cx2Reader.load(base, opts.cx2Url),
    ]);
    return new PicotoneWebGpu(device, weights, opts, cx2);
  }

  /** GPU memory held, in bytes. */
  get gpuBytes(): { readonly weights: number; readonly activations: number } {
    return { weights: this.weightBytes, activations: this.activationBytes };
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
    for (const s of this.staging) s.buf.destroy();
    this.owned.length = 0;
    this.staging.length = 0;
  }

  private createBuffer(bytes: number, usage: number, mapped: boolean, label: string): GPUBuffer {
    const size = Math.max(16, Math.ceil(bytes / 4) * 4);
    const buf = this.device.createBuffer({ label, size, usage, mappedAtCreation: mapped });
    this.owned.push(buf);
    return buf;
  }

  private allConvs(): ConvLayer[] {
    const out: ConvLayer[] = [this.conv0];
    for (const s of this.stages) for (const r of s.res) out.push(r.conv1, r.conv2, r.shortcut);
    out.push(this.convOut);
    return out;
  }

  // ---------------------------------------------------------------- ops ----

  private gnStats(p: Pass, src: Tensor, rows: number, ch: number, part: GPUBuffer): void {
    for (const mode of [0, 1]) {
      p.rec.dispatch(this.k.gnStats, [rows, ch, src.start, src.item, mode], [src.buf, part], [GN_PARTS, p.batch, 1]);
    }
  }

  private gnApply(
    p: Pass, src: Tensor, rows: number, statRows: number, ch: number, part: GPUBuffer, norm: Norm, dst: Tensor, accumulate: boolean,
  ): void {
    p.rec.dispatch(
      this.k.gnApply,
      [rows, ch, statRows, src.start, src.item, dst.start, dst.item, accumulate ? 1 : 0, { f: 1e-5 }],
      [src.buf, part, norm.gamma, norm.beta, dst.buf],
      [Math.max(1, Math.min(1024, Math.ceil((rows * ch) / 256))), p.batch, 1],
    );
  }

  private convDispatch(
    p: Pass,
    w: GPUBuffer,
    b: GPUBuffer,
    g: {
      rowsOut: number; ncols: number; cin: number; taps: number; dilation: number; padLeft: number;
      inLen: number; inStart: number; reflect: boolean; maxPad: number; elu: boolean;
      phases: number; cout: number; trimLeft: number; outLen: number; accumulate: boolean;
    },
    input: Tensor,
    out: Tensor,
  ): void {
    const small = g.ncols <= 4 && g.taps * g.cin <= 256 && (64 + (g.taps - 1) * g.dilation) * g.cin <= 2560;
    p.rec.dispatch(
      small ? this.k.convSmall : this.k.conv,
      [
        g.rowsOut, g.ncols, g.cin, g.taps, 1, g.dilation, g.padLeft, g.inLen,
        g.inStart, input.item, g.reflect ? 1 : 0, g.maxPad, g.elu ? 1 : 0, g.phases, g.cout, g.trimLeft,
        g.outLen, out.start, out.item, g.accumulate ? 1 : 0,
      ],
      [input.buf, w, b, out.buf],
      small ? [Math.ceil(g.rowsOut / 64), 1, p.batch] : [Math.ceil(g.ncols / 64), Math.ceil(g.rowsOut / 64), p.batch],
    );
  }

  /** Copies history rows in before, and out after, a streaming layer (batch 1). */
  private history(p: Pass, input: Tensor, len: number, hist: GPUBuffer | null, rows: number, ch: number): void {
    if (!p.streaming || !hist || rows === 0) return;
    if (!p.first) p.rec.copy(hist, 0, input.buf, (input.start - rows) * ch * 4, rows * ch * 4);
    if (!p.last) p.rec.copy(input.buf, (input.start + len - rows) * ch * 4, hist, 0, rows * ch * 4);
  }

  /**
   * Stride-1 padded convolution of `len` frames. With group norm the result
   * goes through `tmp` and is normalised into `dst`; the returned function
   * performs that final write so callers can order two of them.
   */
  private conv(
    p: Pass, layer: ConvLayer, input: Tensor, len: number, elu: boolean, dst: Tensor, accumulate: boolean, slot: 0 | 1 = 0,
  ): () => void {
    p.rec.tag = layer.name;
    const H = (layer.kernel - 1) * layer.dilation;
    let inStart = input.start;
    let inLen = len;
    let padLeft: number;
    let maxPad: number;
    if (p.streaming) {
      this.history(p, input, len, layer.hist, H, layer.cin);
      if (p.first) {
        padLeft = H;
        maxPad = H;
      } else {
        inStart = input.start - H;
        inLen = len + H;
        padLeft = 0;
        maxPad = 0;
      }
    } else if (this.config.use_causal_conv) {
      padLeft = H;
      maxPad = H;
    } else {
      const right = Math.floor(H / 2);
      padLeft = H - right;
      maxPad = Math.max(padLeft, right);
    }
    const target: Tensor = layer.norm ? { buf: this.tmp[slot], start: 0, item: len } : dst;
    this.convDispatch(
      p, layer.w, layer.b,
      {
        rowsOut: len, ncols: layer.cout, cin: layer.cin, taps: layer.kernel, dilation: layer.dilation, padLeft,
        inLen, inStart, reflect: this.config.pad_mode === 'reflect', maxPad, elu, phases: 1, cout: layer.cout,
        trimLeft: 0, outLen: len, accumulate: layer.norm ? false : accumulate,
      },
      input, target,
    );
    const norm = layer.norm;
    if (!norm) return () => undefined;
    const part = this.parts[slot];
    this.gnStats(p, target, len, layer.cout, part);
    return () => this.gnApply(p, target, len, len, layer.cout, part, norm, dst, accumulate);
  }

  private convTranspose(p: Pass, layer: ConvTLayer, input: Tensor, len: number, dst: Tensor): void {
    p.rec.tag = layer.name;
    const s = layer.stride;
    const hist = layer.taps - 1;
    const pt = layer.kernel - s;
    const outLen = len * s;
    if (p.streaming) {
      this.history(p, input, len, layer.hist, hist, layer.cin);
      this.convDispatch(
        p, layer.w, layer.b,
        {
          rowsOut: len, ncols: s * layer.cout, cin: layer.cin, taps: layer.taps, dilation: 1,
          padLeft: p.first ? hist : 0, inLen: p.first ? len : len + hist, inStart: p.first ? input.start : input.start - hist,
          reflect: false, maxPad: 0, elu: true, phases: s, cout: layer.cout, trimLeft: 0, outLen, accumulate: false,
        },
        input, dst,
      );
      return;
    }
    const right = this.config.use_causal_conv ? Math.ceil(pt * this.config.trim_right_ratio) : Math.floor(pt / 2);
    const left = pt - right;
    const fullLen = (len - 1) * s + layer.kernel;
    const geom = {
      rowsOut: len + hist, ncols: s * layer.cout, cin: layer.cin, taps: layer.taps, dilation: 1, padLeft: hist,
      inLen: len, inStart: input.start, reflect: false, maxPad: 0, elu: true, phases: s, cout: layer.cout,
    };
    if (!layer.norm) {
      this.convDispatch(p, layer.w, layer.b, { ...geom, trimLeft: left, outLen, accumulate: false }, input, dst);
      return;
    }
    // Group norm sees the untrimmed output, as in EnCodec.
    const tmp: Tensor = { buf: this.tmp[0], start: 0, item: fullLen };
    this.convDispatch(p, layer.w, layer.b, { ...geom, trimLeft: 0, outLen: fullLen, accumulate: false }, input, tmp);
    this.gnStats(p, tmp, fullLen, layer.cout, this.parts[0]);
    this.gnApply(p, { buf: tmp.buf, start: left, item: fullLen }, outLen, fullLen, layer.cout, this.parts[0], layer.norm, dst, false);
  }

  private lstmForward(p: Pass, input: Tensor, len: number, dst: Tensor): void {
    const N = this.maxFrames;
    const D = this.dim;
    let src = input;
    p.rec.tag = 'lstm';
    this.lstm.forEach((layer, l) => {
      const h = this.h[l];
      const c = this.c[l];
      if (!h || !c) throw new Error('missing LSTM state');
      const hs = this.hState[l];
      if (!p.streaming || p.first) {
        p.rec.clear(h);
        p.rec.clear(c);
      } else if (hs) {
        p.rec.copy(hs, 0, h, 0, D * 4);
      }
      this.convDispatch(
        p, layer.wih, layer.bias,
        {
          rowsOut: len, ncols: 4 * D, cin: D, taps: 1, dilation: 1, padLeft: 0, inLen: len, inStart: src.start,
          reflect: false, maxPad: 0, elu: false, phases: 1, cout: 4 * D, trimLeft: 0, outLen: len, accumulate: false,
        },
        src, { buf: this.gx, start: 0, item: N },
      );
      const group = p.rec.group(this.k.lstm, [
        p.rec.uniform([D, N, N + 1, p.batch]),
        { buffer: this.stepUniforms, offset: 0, size: UNIFORM_SLOT },
        { buffer: layer.whh },
        { buffer: this.gx },
        { buffer: h },
        { buffer: c },
      ]);
      for (let t = 0; t < len; t++) p.rec.dispatchDynamic(this.k.lstm, group, t * UNIFORM_SLOT, [D / LSTM_UNITS, 1, 1]);
      if (p.streaming && !p.last && hs) p.rec.copy(h, len * D * 4, hs, 0, D * 4);
      src = { buf: h, start: 1, item: N + 1 };
    });
    const total = p.batch * len * D;
    p.rec.dispatch(
      this.k.add,
      [len, D, src.start, src.item, input.start, input.item, dst.start, dst.item, p.batch],
      [src.buf, input.buf, dst.buf],
      [gridGroups(total), 1, 1],
    );
  }

  /** Records the decoder for `len` latent frames per batch item; output in `this.out`. */
  private forward(p: Pass, len: number, numCodebooks: number): void {
    const N = this.maxFrames;
    const P = this.prefix;
    const emb: Tensor = { buf: this.emb, start: P, item: P + N };
    p.rec.tag = 'codes';
    p.rec.dispatch(
      this.k.quant,
      [len, this.latentDim, numCodebooks, this.config.codebook_size, emb.start, emb.item, p.batch],
      [this.codes, this.codebooks, this.emb],
      [gridGroups(p.batch * len * this.latentDim), 1, 1],
    );
    const a0: Tensor = { buf: this.a0, start: 0, item: N };
    this.conv(p, this.conv0, emb, len, false, a0, false)();
    let rowsMax = N;
    let rows = len;
    let cur: Tensor = { buf: this.pool[0], start: P, item: P + rowsMax };
    this.lstmForward(p, a0, len, cur);
    let which = 0;
    for (const stage of this.stages) {
      const s = stage.up.stride;
      rowsMax *= s;
      const up: Tensor = { buf: this.pool[1 - which] ?? this.pool[0], start: P, item: P + rowsMax };
      this.convTranspose(p, stage.up, cur, rows, up);
      rows *= s;
      cur = up;
      which = 1 - which;
      for (const res of stage.res) {
        const hid: Tensor = { buf: this.hidden, start: 0, item: rowsMax };
        const outT: Tensor = { buf: this.pool[1 - which] ?? this.pool[0], start: P, item: P + rowsMax };
        this.conv(p, res.conv1, cur, rows, true, hid, false)();
        const applyMain = this.conv(p, res.conv2, hid, rows, true, outT, false, 0);
        if (res.conv2.norm) {
          const applyShort = this.conv(p, res.shortcut, cur, rows, false, outT, true, 1);
          applyMain();
          applyShort();
        } else {
          // Main path writes first, then the shortcut accumulates.
          this.conv(p, res.shortcut, cur, rows, false, outT, true, 1)();
        }
        cur = outT;
        which = 1 - which;
      }
    }
    this.conv(p, this.convOut, cur, rows, true, { buf: this.out, start: 0, item: N * this.hop }, false)();
  }

  // ------------------------------------------------------------- decode ----

  private takeStaging(): Staging {
    const s = this.staging.pop();
    if (s) return s;
    return {
      buf: this.device.createBuffer({
        label: 'readback',
        size: Math.max(16, this.readbackBytes),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  /**
   * Decodes a `.pico` file to planar float audio. The cx2 codes are
   * range-decoded in WebAssembly a page at a time; the network runs on the GPU.
   */
  async decode(file: Uint8Array, opts: DecodeOptions = {}): Promise<DecodedAudio> {
    const head = parseHeader(file);
    if (head.model !== this.name) {
      throw new PicoError(`file was encoded with ${head.model}, but this decoder holds ${this.name}`);
    }
    if (head.numCodebooks > this.totalCodebooks) {
      throw new PicoError(`file uses ${head.numCodebooks} codebooks, the model has ${this.totalCodebooks}`);
    }
    const header = head;
    const C = this.channels;
    // Coded samples (segment layout) and the trimmed output.
    const total = header.codedLength;
    const length = header.audioLength;
    const collect = opts.collect ?? true;
    const outChannels = Array.from({ length: C }, () => new Float32Array(collect ? length : 0));
    const result: DecodedAudio = { sampleRate: this.sampleRate, length, channels: outChannels, loop: header.loop };
    if (length === 0) return result;
    const trimmer = new Trimmer(header.preSkip, length, header.loop, this.sampleRate, C);
    const limiter = new Limiter(C);
    let emitted = 0;
    const emit = (planar: Float32Array[]): void => {
      const offset = emitted;
      const count = planar[0]?.length ?? 0;
      emitted += count;
      if (count > 0) {
        if (collect) planar.forEach((ch, c) => (outChannels[c] as Float32Array).set(ch, offset));
        opts.onChunk?.({ offset, sampleRate: this.sampleRate, channels: planar });
      }
    };

    this.device.pushErrorScope('validation');
    this.device.pushErrorScope('out-of-memory');
    const pending: { promise: Promise<void>; staging: Staging; offset: number; count: number }[] = [];
    const deliver = async (item: (typeof pending)[number]): Promise<void> => {
      await item.promise;
      const data = new Float32Array(item.staging.buf.getMappedRange(0, item.count * C * 4).slice(0));
      item.staging.buf.unmap();
      this.staging.push(item.staging);
      const raw = Array.from({ length: C }, () => new Float32Array(item.count));
      for (let i = 0; i < item.count; i++) {
        for (let c = 0; c < C; c++) (raw[c] as Float32Array)[i] = data[i * C + c] ?? 0;
      }
      emit(limiter.push(trimmer.identity ? raw : trimmer.push(raw)));
      opts.onProgress?.(Math.min(1, (item.offset + item.count) / total));
    };
    const submit = async (rec: Recorder, src: GPUBuffer, offset: number, count: number): Promise<void> => {
      while (pending.length >= 2) {
        const first = pending.shift();
        if (first) await deliver(first);
      }
      const staging = this.takeStaging();
      const encoder = this.device.createCommandEncoder();
      const labels = rec.encode(encoder, prof?.query);
      encoder.copyBufferToBuffer(src, 0, staging.buf, 0, Math.ceil((count * C * 4) / 4) * 4);
      if (prof && labels.length > 0) {
        encoder.resolveQuerySet(prof.query, 0, labels.length * 2, prof.resolve, 0);
        encoder.copyBufferToBuffer(prof.resolve, 0, prof.read, 0, labels.length * 16);
      }
      this.device.queue.submit([encoder.finish()]);
      pending.push({ promise: staging.buf.mapAsync(GPUMapMode.READ, 0, count * C * 4), staging, offset, count });
      if (prof && labels.length > 0) {
        await prof.read.mapAsync(GPUMapMode.READ);
        const ts = new BigUint64Array(prof.read.getMappedRange(0, labels.length * 16));
        labels.forEach((label, i) => {
          const ns = Number((ts[2 * i + 1] ?? 0n) - (ts[2 * i] ?? 0n));
          prof.ms.set(label, (prof.ms.get(label) ?? 0) + ns / 1e6);
        });
        prof.read.unmap();
      }
    };
    const prof =
      opts.profile && this.device.features.has('timestamp-query')
        ? {
            query: this.device.createQuerySet({ type: 'timestamp', count: 4096 }),
            resolve: this.device.createBuffer({ size: 4096 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
            read: this.device.createBuffer({ size: 4096 * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
            ms: new Map<string, number>(),
          }
        : null;

    const codes = this.cx2.open(file, this.config);
    try {
      if (this.streaming) await this.decodeStream(codes, total, opts, submit);
      else await this.decodeSegments(codes, total, opts, submit);
      while (pending.length > 0) {
        const first = pending.shift();
        if (first) await deliver(first);
      }
      if (!trimmer.identity) trimmer.finish();
      emit(limiter.finish());
      if (emitted < length) throw new PicoError(`decoded ${emitted} samples, expected ${length}`);
    } finally {
      codes.free();
      // Drain anything still in flight after an error.
      for (const p of pending) {
        await p.promise.catch(() => undefined);
        if (p.staging.buf.mapState === 'mapped') p.staging.buf.unmap();
        this.staging.push(p.staging);
      }
      const oom = await this.device.popErrorScope();
      const invalid = await this.device.popErrorScope();
      if (oom) throw new Error(`GPU out of memory: ${oom.message}`);
      if (invalid) throw new Error(`WebGPU validation error: ${invalid.message}`);
    }
    if (prof) {
      prof.query.destroy();
      prof.resolve.destroy();
      prof.read.destroy();
      return { ...result, profileMs: Object.fromEntries(prof.ms) };
    }
    return result;
  }

  private async decodeStream(
    source: PicoCodes,
    total: number,
    opts: DecodeOptions,
    submit: (rec: Recorder, src: GPUBuffer, offset: number, count: number) => Promise<void>,
  ): Promise<void> {
    const nq = source.header.numCodebooks;
    const frames = source.segmentFrames[0] ?? 0;
    let f0 = 0;
    while (f0 < frames) {
      opts.signal?.throwIfAborted();
      const want = f0 === 0 ? this.firstFrames : this.maxFrames;
      // The first chunk is at least 8 frames (more than any layer's history) or the
      // whole file; later chunks may be any length because history is restored first.
      const len = Math.min(want, frames - f0);
      const last = f0 + len >= frames;
      const codes = new Uint32Array(len * nq);
      codes.set(source.frames(f0, len));
      this.device.queue.writeBuffer(this.codes, 0, codes);
      const rec = new Recorder(this.device, this.arena);
      this.forward({ rec, batch: 1, streaming: true, first: f0 === 0, last }, len, nq);
      const offset = f0 * this.hop;
      const count = Math.min(len * this.hop, total - offset);
      if (count > 0) await submit(rec, this.out, offset, count);
      f0 += len;
    }
  }

  private async decodeSegments(
    source: PicoCodes,
    total: number,
    opts: DecodeOptions,
    submit: (rec: Recorder, src: GPUBuffer, offset: number, count: number) => Promise<void>,
  ): Promise<void> {
    const { header, segmentFrames } = source;
    const nq = header.numCodebooks;
    const n = segmentFrames.length;
    const length = segmentLength(this.config) ?? total;
    const stride = (header.stride > 0 ? header.stride : null) ?? segmentStride(this.config) ?? total;
    if (stride < Math.ceil(length / 2) || stride > length) {
      throw new PicoError(`stride ${stride} is outside ${Math.ceil(length / 2)}..=${length}`);
    }
    // layout::EdgeRamp: the ramp avoids the last 10 ms of the old segment and
    // the first 30 ms of the new one where the overlap allows.
    const ov = length - stride;
    const hi = ov > 960 ? ov - 480 : ov;
    const lo = Math.min(1440, Math.max(0, hi - 240));
    const xf = header.crossfade === 1 ? 1 : 0;
    const firstFrames = segmentFrames[0] ?? 0;
    const frameLen = firstFrames * this.hop;
    const lastLen = (segmentFrames[n - 1] ?? 0) * this.hop;
    const fullLen = stride * (n - 1) + lastLen;
    if (fullLen < total) throw new PicoError(`decoded ${fullLen} samples, expected at least ${total}`);
    let k0 = 0;
    let batchIndex = 0;
    while (k0 < n) {
      opts.signal?.throwIfAborted();
      // First batch holds one segment so audio starts quickly.
      const want = batchIndex === 0 ? 1 : this.maxBatch;
      const frames = segmentFrames[k0] ?? 0;
      let B = 0;
      while (B < want && k0 + B < n && segmentFrames[k0 + B] === frames) B++;
      const finalBatch = k0 + B >= n;
      const codes = new Uint32Array(B * frames * nq);
      const scales = new Float32Array(B);
      for (let b = 0; b < B; b++) {
        const seg = source.segment(k0 + b);
        codes.set(seg.codes, b * frames * nq);
        scales[b] = seg.scale ?? 1;
      }
      this.device.queue.writeBuffer(this.codes, 0, codes);
      this.device.queue.writeBuffer(this.scales, 0, scales);
      const rec = new Recorder(this.device, this.arena);
      const pass: Pass = { rec, batch: B, streaming: false, first: true, last: true };
      this.forward(pass, frames, nq);
      const t0 = k0 * stride;
      const end = Math.min(total, finalBatch ? fullLen : (k0 + B) * stride);
      const count = end - t0;
      const segLen = frames * this.hop;
      const segItem = this.maxFrames * this.hop;
      rec.tag = 'ola';
      const ola = (mode: number): void =>
        rec.dispatch(
          this.k.ola,
          [
            Math.max(0, count), this.channels, t0, k0, B, segLen, segItem, stride, frameLen, k0 > 0 ? 1 : 0, frameLen, mode,
            xf, lo, hi, ov,
          ],
          [this.out, this.scales, this.tail, this.olaOut],
          [gridGroups(Math.max(count, segLen - stride) * this.channels), 1, 1],
        );
      // With a single segment the weights cancel and this is just the scale.
      ola(0);
      if (!finalBatch) ola(1);
      if (count > 0) await submit(rec, this.olaOut, t0, count);
      k0 += B;
      batchIndex++;
    }
  }
}

/** Reads the model name from a `.pico` file's header. */
export function modelNameOf(bytes: Uint8Array): string {
  return parseHeader(bytes).model;
}
