// Exports decoder-only EnCodec weights for the WebGPU decoder.
//
// Reads ../weights/encodec_<sr>.safetensors and .json (read-only) and writes
//   public/models/<name>/weights-f32.bin, weights-f16.bin and manifest.json,
//   public/models/<name>/EnCodec-MIT.txt and public/models/EnCodec-MIT.txt (the weights' licence),
//   public/models/<table>.ecxm (each model's default cx2 table, from ../models) and
//   public/models/cx2-tables.json ({ tables: [{ file, model, hash }] }).
// The cx2 WebAssembly module goes to public/models/cx2/ at build time (tools/build.mjs).
//
// Weight norm is folded at export (g * v / ||v|| over dim 0) and every tensor
// is stored in the layout the WGSL kernels read directly:
//   conv   "<prefix>.w"     [kernel * cin, cout]      row (j * cin + i), column o
//   convT  "<prefix>.w"     [taps * cin, stride * cout] polyphase matrix (see nn.rs)
//   conv   "<prefix>.b"     [cout]
//   norm   "<prefix>.gamma" / ".beta" [cout]          (time_group_norm models)
//   lstm   "<prefix>.w_ih_l<n>" [dim, 4 dim]           (transposed weight_ih)
//          "<prefix>.b_l<n>"    [4 dim]                (bias_ih + bias_hh)
//          "<prefix>.w_hh_l<n>" [4 dim, dim]           (PyTorch layout)
//   "quantizer.embed" [codebooks, codebook_size, codebook_dim]
//
// Usage: node tools/export-weights.mjs [--weights <dir>] [--out <dir>] [name...]

import { readFileSync, writeFileSync, mkdirSync, statSync, copyFileSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
let weightsDir = resolve(here, '..', '..', 'weights');
let outDir = resolve(here, '..', 'public', 'models');
const names = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--weights') weightsDir = resolve(args[++i]);
  else if (args[i] === '--out') outDir = resolve(args[++i]);
  else names.push(args[i]);
}
if (names.length === 0) names.push('encodec_24khz', 'encodec_48khz');

const tablesDir = resolve(here, '..', '..', 'models');
const licence = resolve(here, '..', '..', 'LICENSES', 'EnCodec-MIT.txt');
// The table each model codes with by default (as the picotone command line).
const TABLES = { encodec_24khz: 'cx2_encodec_24khz_v5.ecxm', encodec_48khz: 'cx2_encodec_48khz_net2_v5.ecxm' };
const tableList = [];

/** The hash a .pico stream stores for a self-contained table: FNV-1a 64 of its bytes, as 16 hex digits. */
function tableHash(bytes) {
  if (bytes.subarray(0, 4).toString('latin1') !== 'ECXM' || bytes.readUInt32LE(4) !== 5) {
    throw new Error('not a self-contained (version 5) cx2 table');
  }
  let h = 0xcbf29ce484222325n;
  for (const b of bytes) h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
  return h.toString(16).padStart(16, '0');
}

function readSafetensors(path) {
  const buf = readFileSync(path);
  const len = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + len).toString('utf8'));
  const base = 8 + len;
  const tensors = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (info.dtype !== 'F32') throw new Error(`${name}: dtype ${info.dtype} is not F32`);
    const [start, end] = info.data_offsets;
    const bytes = buf.subarray(base + start, base + end);
    // Copy so the data is 4-byte aligned.
    const data = new Float32Array(new Uint8Array(bytes).buffer);
    tensors.set(name, { shape: info.shape, data });
  }
  return tensors;
}

function sameShape(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

class Exporter {
  constructor(tensors, cfg) {
    this.t = tensors;
    this.cfg = cfg;
    this.out = []; // { name, shape, data }
    this.used = new Set();
  }

  get(name, shape) {
    const t = this.t.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    if (!sameShape(t.shape, shape)) {
      throw new Error(`${name}: shape ${JSON.stringify(t.shape)}, expected ${JSON.stringify(shape)}`);
    }
    this.used.add(name);
    return t.data;
  }

  emit(name, shape, data) {
    const n = shape.reduce((a, b) => a * b, 1);
    if (data.length !== n) throw new Error(`${name}: ${data.length} values for shape ${shape}`);
    this.out.push({ name, shape, data });
  }

  /** Conv weight, folding weight norm; torch layout `shape`, normalised over dim 0. */
  weight(prefix, shape) {
    if (this.cfg.norm_type === 'time_group_norm') {
      return this.get(`${prefix}.weight`, shape);
    }
    const g = this.get(`${prefix}.weight_g`, [shape[0], 1, 1]);
    const v = this.get(`${prefix}.weight_v`, shape);
    const row = shape[1] * shape[2];
    const out = new Float32Array(v.length);
    for (let r = 0; r < shape[0]; r++) {
      let sq = 0;
      for (let k = 0; k < row; k++) sq += v[r * row + k] * v[r * row + k];
      const factor = Math.fround(g[r] / Math.sqrt(sq));
      for (let k = 0; k < row; k++) out[r * row + k] = Math.fround(v[r * row + k] * factor);
    }
    return out;
  }

  norm(prefix, channels) {
    if (this.cfg.norm_type !== 'time_group_norm') return;
    this.emit(`${prefix}.gamma`, [channels], this.get(`${prefix}.norm.weight`, [channels]));
    this.emit(`${prefix}.beta`, [channels], this.get(`${prefix}.norm.bias`, [channels]));
  }

  conv(prefix, cin, cout, k) {
    const w = this.weight(`${prefix}.conv`, [cout, cin, k]);
    const m = new Float32Array(k * cin * cout);
    for (let o = 0; o < cout; o++)
      for (let i = 0; i < cin; i++)
        for (let j = 0; j < k; j++) m[(j * cin + i) * cout + o] = w[(o * cin + i) * k + j];
    this.emit(`${prefix}.w`, [k * cin, cout], m);
    this.emit(`${prefix}.b`, [cout], this.get(`${prefix}.conv.bias`, [cout]));
    this.norm(prefix, cout);
  }

  convTranspose(prefix, cin, cout, k, s) {
    const w = this.weight(`${prefix}.conv`, [cin, cout, k]);
    const taps = Math.ceil(k / s);
    const cols = s * cout;
    const m = new Float32Array(taps * cin * cols);
    for (let tap = 0; tap < taps; tap++)
      for (let p = 0; p < s; p++) {
        const j = p + (taps - 1 - tap) * s;
        if (j >= k) continue;
        for (let i = 0; i < cin; i++)
          for (let o = 0; o < cout; o++) m[(tap * cin + i) * cols + p * cout + o] = w[(i * cout + o) * k + j];
      }
    this.emit(`${prefix}.w`, [taps * cin, cols], m);
    this.emit(`${prefix}.b`, [cout], this.get(`${prefix}.conv.bias`, [cout]));
    this.norm(prefix, cout);
  }

  lstm(prefix, dim) {
    for (let l = 0; l < this.cfg.num_lstm_layers; l++) {
      const wih = this.get(`${prefix}.lstm.weight_ih_l${l}`, [4 * dim, dim]);
      const whh = this.get(`${prefix}.lstm.weight_hh_l${l}`, [4 * dim, dim]);
      const bih = this.get(`${prefix}.lstm.bias_ih_l${l}`, [4 * dim]);
      const bhh = this.get(`${prefix}.lstm.bias_hh_l${l}`, [4 * dim]);
      const t = new Float32Array(dim * 4 * dim);
      for (let r = 0; r < 4 * dim; r++) for (let c = 0; c < dim; c++) t[c * 4 * dim + r] = wih[r * dim + c];
      const b = new Float32Array(4 * dim);
      for (let r = 0; r < 4 * dim; r++) b[r] = Math.fround(bih[r] + bhh[r]);
      this.emit(`${prefix}.w_ih_l${l}`, [dim, 4 * dim], t);
      this.emit(`${prefix}.b_l${l}`, [4 * dim], b);
      this.emit(`${prefix}.w_hh_l${l}`, [4 * dim, dim], whh);
    }
  }

  resnet(prefix, dim, dilation) {
    const cfg = this.cfg;
    const hidden = dim / cfg.compress;
    // block.0 ELU, block.1 conv, block.2 ELU, block.3 conv (nn.Sequential indices).
    this.conv(`${prefix}.block.1`, dim, hidden, cfg.residual_kernel_size, dilation);
    this.conv(`${prefix}.block.3`, hidden, dim, 1);
    if (cfg.use_conv_shortcut !== false) this.conv(`${prefix}.shortcut`, dim, dim, 1);
  }

  /** Mirrors Builder::decoder in src/model.rs. */
  decoder() {
    const cfg = this.cfg;
    let idx = 0;
    const next = () => `decoder.layers.${idx++}`;
    let scaling = 1 << cfg.upsampling_ratios.length;
    const dim = scaling * cfg.num_filters;
    this.conv(next(), cfg.hidden_size, dim, cfg.last_kernel_size);
    this.lstm(next(), dim);
    for (const ratio of cfg.upsampling_ratios) {
      const scale = scaling * cfg.num_filters;
      idx++; // ELU
      this.convTranspose(next(), scale, scale / 2, ratio * 2, ratio);
      for (let j = 0; j < cfg.num_residual_layers; j++) {
        this.resnet(next(), scale / 2, cfg.dilation_growth_rate ** j);
      }
      scaling /= 2;
    }
    idx++; // ELU
    this.conv(next(), cfg.num_filters, cfg.audio_channels, cfg.last_kernel_size);
  }

  quantizer(count) {
    const dim = this.cfg.codebook_dim ?? this.cfg.hidden_size;
    const size = this.cfg.codebook_size;
    const all = new Float32Array(count * size * dim);
    for (let q = 0; q < count; q++) all.set(this.get(`quantizer.layers.${q}.codebook.embed`, [size, dim]), q * size * dim);
    if (this.t.has(`quantizer.layers.${count}.codebook.embed`)) throw new Error('more codebooks than the config allows');
    this.emit('quantizer.embed', [count, size, dim], all);
  }
}

function totalCodebooks(cfg) {
  const hop = cfg.upsampling_ratios.reduce((a, b) => a * b, 1);
  const frameRate = Math.ceil(cfg.sampling_rate / hop);
  const bits = Math.log2(cfg.codebook_size);
  const max = Math.max(...cfg.target_bandwidths);
  return Math.max(1, Math.floor(Math.round(max * 1000) / (bits * frameRate)));
}

// float32 -> IEEE half, round to nearest even.
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && (half & 1))) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
  return sign | half;
}

const ALIGN = 256;
function writeBlob(tensors, dtype) {
  const bpe = dtype === 'f16' ? 2 : 4;
  let offset = 0;
  const entries = {};
  for (const t of tensors) {
    offset = Math.ceil(offset / ALIGN) * ALIGN;
    entries[t.name] = { offset, length: t.data.length, shape: t.shape };
    offset += t.data.length * bpe;
  }
  const bytes = new Uint8Array(Math.ceil(offset / ALIGN) * ALIGN);
  const view = new DataView(bytes.buffer);
  for (const t of tensors) {
    const base = entries[t.name].offset;
    if (dtype === 'f16') for (let i = 0; i < t.data.length; i++) view.setUint16(base + 2 * i, toHalf(t.data[i]), true);
    else for (let i = 0; i < t.data.length; i++) view.setFloat32(base + 4 * i, t.data[i], true);
  }
  return { bytes, entries };
}

const CONFIG_KEYS = [
  'audio_channels', 'chunk_length_s', 'codebook_dim', 'codebook_size', 'compress', 'dilation_growth_rate',
  'hidden_size', 'kernel_size', 'last_kernel_size', 'norm_type', 'normalize', 'num_filters', 'num_lstm_layers',
  'num_residual_layers', 'overlap', 'pad_mode', 'residual_kernel_size', 'sampling_rate', 'target_bandwidths',
  'trim_right_ratio', 'upsampling_ratios', 'use_causal_conv', 'use_conv_shortcut',
];

for (const name of names) {
  const cfgFull = JSON.parse(readFileSync(join(weightsDir, `${name}.json`), 'utf8'));
  const cfg = {};
  for (const k of CONFIG_KEYS) if (k in cfgFull) cfg[k] = cfgFull[k];
  if (!('use_conv_shortcut' in cfg)) cfg.use_conv_shortcut = true;
  const tensors = readSafetensors(join(weightsDir, `${name}.safetensors`));
  const ex = new Exporter(tensors, cfg);
  ex.decoder();
  const nq = totalCodebooks(cfg);
  ex.quantizer(nq);

  const dir = join(outDir, name);
  mkdirSync(dir, { recursive: true });
  const manifest = { format: 'picotone-webgpu-weights', version: 1, name, config: cfg, total_codebooks: nq, files: {} };
  const params = ex.out.reduce((a, t) => a + t.data.length, 0);
  for (const dtype of ['f32', 'f16']) {
    const { bytes, entries } = writeBlob(ex.out, dtype);
    const file = `weights-${dtype}.bin`;
    writeFileSync(join(dir, file), bytes);
    manifest.files[dtype] = { file, bytes: bytes.length, tensors: entries };
  }
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 1));
  const mb = (f) => (statSync(join(dir, f)).size / 1e6).toFixed(2);
  console.log(
    `${name}: ${ex.out.length} tensors, ${(params / 1e6).toFixed(2)} M params, ` +
      `f32 ${mb('weights-f32.bin')} MB, f16 ${mb('weights-f16.bin')} MB -> ${dir}`,
  );
  copyFileSync(licence, join(dir, 'EnCodec-MIT.txt'));
  const table = TABLES[name];
  const bytes = readFileSync(join(tablesDir, table));
  copyFileSync(join(tablesDir, table), join(outDir, table));
  tableList.push({ file: table, model: name, hash: tableHash(bytes) });
  console.log(`${name}: table ${table} (${(bytes.length / 1e6).toFixed(1)} MB, hash ${tableList.at(-1).hash})`);
}
copyFileSync(licence, join(outDir, 'EnCodec-MIT.txt'));
// Keep tables of models not exported this time.
let previous = [];
try {
  previous = JSON.parse(readFileSync(join(outDir, 'cx2-tables.json'), 'utf8')).tables ?? [];
} catch {
  previous = [];
}
const merged = [...previous.filter((e) => !tableList.some((n) => n.model === e.model)), ...tableList];
writeFileSync(join(outDir, 'cx2-tables.json'), JSON.stringify({ tables: merged }, null, 2) + '\n');
