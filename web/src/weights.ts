/** Fetching the exported decoder weights (see tools/export-weights.mjs). */

import type { ModelConfig } from './config.js';

export type WeightDtype = 'f32' | 'f16';

export interface TensorEntry {
  readonly offset: number;
  readonly length: number;
  readonly shape: readonly number[];
}

export interface WeightFile {
  readonly file: string;
  readonly bytes: number;
  readonly tensors: Readonly<Record<string, TensorEntry>>;
}

export interface Manifest {
  readonly format: 'picotone-webgpu-weights';
  readonly version: number;
  readonly name: string;
  readonly config: ModelConfig;
  readonly total_codebooks: number;
  readonly files: Readonly<Partial<Record<WeightDtype, WeightFile>>>;
}

export interface LoadedWeights {
  readonly manifest: Manifest;
  readonly dtype: WeightDtype;
  /** Tensors converted to f32. */
  readonly tensors: ReadonlyMap<string, { readonly shape: readonly number[]; readonly data: Float32Array }>;
}

export type FetchProgress = (loadedBytes: number, totalBytes: number) => void;

function joinUrl(base: string, path: string): string {
  return base.endsWith('/') ? base + path : `${base}/${path}`;
}

async function fetchBytes(url: string, expected: number, onProgress?: FetchProgress): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetching ${url} failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? expected) || expected;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(buf.length, buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const out = new Uint8Array(Math.max(total, expected));
  let loaded = 0;
  let grown: Uint8Array = out;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > grown.length) {
      const bigger = new Uint8Array(Math.max(grown.length * 2, loaded + value.length));
      bigger.set(grown.subarray(0, loaded));
      grown = bigger;
    }
    grown.set(value, loaded);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  return grown.subarray(0, loaded);
}

const convF32 = new Float32Array(1);
const convU32 = new Uint32Array(convF32.buffer);

/** float32 -> IEEE half bits, round to nearest even. */
export function floatToHalf(v: number): number {
  convF32[0] = v;
  const x = convU32[0] ?? 0;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && half & 1)) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half++;
  return sign | half;
}

let halfTable: Float32Array | null = null;
function halfToFloatTable(): Float32Array {
  if (halfTable) return halfTable;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) t[h] = s * m * 2 ** -24;
    else if (e === 31) t[h] = m ? NaN : s * Infinity;
    else t[h] = s * (1 + m / 1024) * 2 ** (e - 15);
  }
  halfTable = t;
  return t;
}

/** Fetches `<modelBaseUrl>/manifest.json` and the chosen weight blob. */
export async function fetchWeights(
  modelBaseUrl: string,
  dtype: WeightDtype,
  onProgress?: FetchProgress,
): Promise<LoadedWeights> {
  const manifestUrl = joinUrl(modelBaseUrl, 'manifest.json');
  const res = await fetch(manifestUrl);
  if (!res.ok) throw new Error(`fetching ${manifestUrl} failed: HTTP ${res.status}`);
  const manifest = (await res.json()) as Manifest;
  if (manifest.format !== 'picotone-webgpu-weights') {
    throw new Error(`${manifestUrl} is not a picotone WebGPU weight manifest (format ${String(manifest.format)}); re-export the models with tools/export-weights.mjs`);
  }
  const file = manifest.files[dtype];
  if (!file) throw new Error(`${manifestUrl} has no ${dtype} weights`);
  const bytes = await fetchBytes(joinUrl(modelBaseUrl, file.file), file.bytes, onProgress);
  if (bytes.length < file.bytes) throw new Error(`weight file is truncated (${bytes.length} of ${file.bytes} bytes)`);
  const tensors = new Map<string, { shape: readonly number[]; data: Float32Array }>();
  const table = dtype === 'f16' ? halfToFloatTable() : null;
  for (const [name, entry] of Object.entries(file.tensors)) {
    let data: Float32Array;
    if (table) {
      const half = new Uint16Array(bytes.buffer, bytes.byteOffset + entry.offset, entry.length);
      data = new Float32Array(entry.length);
      for (let i = 0; i < entry.length; i++) data[i] = table[half[i] ?? 0] ?? 0;
    } else {
      data = new Float32Array(bytes.buffer.slice(bytes.byteOffset + entry.offset, bytes.byteOffset + entry.offset + entry.length * 4));
    }
    tensors.set(name, { shape: entry.shape, data });
  }
  return { manifest, dtype, tensors };
}
