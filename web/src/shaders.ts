/**
 * WGSL compute kernels. All activations are time-major f32: value `c` of frame
 * `t` in batch item `b` lives at `((b * item_rows) + start + t) * channels + c`.
 */

/** Uniform slot stride; every kernel's parameters fit in one slot. */
export const UNIFORM_SLOT = 256;

/** Shared parameters, bindings and padding/output helpers of the conv kernels. */
const CONV_HEADER = /* wgsl */ `
struct P {
  rows_out: u32, ncols: u32, cin: u32, taps: u32,
  stride: u32, dilation: u32, pad_left: u32, in_len: u32,
  in_start: u32, in_item: u32, pad_mode: u32, max_pad: u32,
  act: u32, phases: u32, cout: u32, trim_left: u32,
  out_len: u32, out_start: u32, out_item: u32, accumulate: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

// Virtual padded coordinate -> source frame, or -1 for a zero.
fn src_row(v: i32) -> i32 {
  let n = i32(p.in_len);
  if (p.pad_mode == 1u) {
    // Reflect; inputs no longer than the pad are zero-extended first (EnCodec).
    let base = max(n, i32(p.max_pad) + 1);
    var u = v;
    if (u < 0) { u = -u; }
    if (u >= base) { u = 2 * (base - 1) - u; }
    if (u < 0 || u >= n) { return -1; }
    return u;
  }
  if (v < 0 || v >= n) { return -1; }
  return v;
}

fn load_x(b: u32, v: i32, i: u32) -> f32 {
  let s = src_row(v);
  if (s < 0) { return 0.0; }
  let val = x[(b * p.in_item + p.in_start + u32(s)) * p.cin + i];
  if (p.act == 1u && val < 0.0) { return exp(val) - 1.0; }
  return val;
}

fn store_y(b: u32, t: u32, n: u32, acc: f32) {
  let ph = n / p.cout;
  let o = n % p.cout;
  let f = i32(t * p.phases + ph) - i32(p.trim_left);
  if (f < 0 || f >= i32(p.out_len)) { return; }
  let idx = (b * p.out_item + p.out_start + u32(f)) * p.cout + o;
  var v = acc + bias[o];
  if (p.accumulate == 1u) { v += y[idx]; }
  y[idx] = v;
}
`;

/**
 * Direct convolution for very few output columns (the final 32 -> 1 or 2
 * channel layer), where a 64-wide GEMM tile would be mostly idle: one thread
 * per output frame, up to 4 columns.
 */
export const CONV_SMALL_WGSL = CONV_HEADER + /* wgsl */ `
// Requires stride 1, taps * cin <= 256 and (64 + span - 1) * cin <= 2560.
const SM_ROWS: u32 = 64u;
var<workgroup> ws: array<vec4f, 256>;  // weights padded to 4 columns
var<workgroup> xs: array<f32, 2560>;   // input tile, then partial sums

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let b = wg.z;
  let row0 = wg.x * SM_ROWS;
  let K = p.taps * p.cin;
  for (var k = li; k < K; k += 256u) {
    var wv = vec4f(0.0);
    for (var n = 0u; n < 4u; n++) { if (n < p.ncols) { wv[n] = w[k * p.ncols + n]; } }
    ws[k] = wv;
  }
  let tile_rows = SM_ROWS + (p.taps - 1u) * p.dilation;
  for (var e = li; e < tile_rows * p.cin; e += 256u) {
    let r = e / p.cin;
    xs[e] = load_x(b, i32(row0 + r) - i32(p.pad_left), e % p.cin);
  }
  workgroupBarrier();
  // Thread (r, q): output row r, input channels i = q, q + 4, ...
  let r = li / 4u;
  let q = li % 4u;
  var acc = vec4f(0.0);
  for (var j = 0u; j < p.taps; j++) {
    let base = (r + j * p.dilation) * p.cin;
    for (var i = q; i < p.cin; i += 4u) { acc = fma(vec4f(xs[base + i]), ws[j * p.cin + i], acc); }
  }
  workgroupBarrier();
  for (var n = 0u; n < 4u; n++) { xs[li * 4u + n] = acc[n]; }
  workgroupBarrier();
  let t = row0 + r;
  if (q == 0u && t < p.rows_out) {
    for (var n = 0u; n < 4u; n++) {
      if (n < p.ncols) {
        let s = xs[li * 4u + n] + xs[(li + 1u) * 4u + n] + xs[(li + 2u) * 4u + n] + xs[(li + 3u) * 4u + n];
        store_y(b, t, n, s);
      }
    }
  }
}
`;

/**
 * Padded convolution as a tiled GEMM (64x64 output tile, K step 16).
 * Also runs polyphase transposed convolutions (`phases` > 1) and the LSTM
 * input projection (taps 1). Padding is virtual: frame `v` of the padded
 * input maps to source frame `src_row(v)` (reflect or zero padding).
 */
export const CONV_WGSL = CONV_HEADER + /* wgsl */ `
const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let b = wg.z;
  let row0 = wg.y * TM;
  let col0 = wg.x * TN;
  let K = p.taps * p.cin;
  let xbase = b * p.in_item;
  var acc: array<array<f32, 4>, 4>;
  for (var r = 0u; r < 4u; r++) { for (var c = 0u; c < 4u; c++) { acc[r][c] = 0.0; } }

  for (var k0 = 0u; k0 < K; k0 += TK) {
    let j = k0 / p.cin;
    let i0 = k0 % p.cin;
    for (var q = 0u; q < 4u; q++) {
      let e = li + q * 256u;
      let r = e / TK;
      let kk = e % TK;
      let t = row0 + r;
      var val = 0.0;
      if (t < p.rows_out) {
        let s = src_row(i32(t * p.stride + j * p.dilation) - i32(p.pad_left));
        if (s >= 0) {
          val = x[(xbase + p.in_start + u32(s)) * p.cin + i0 + kk];
          if (p.act == 1u && val < 0.0) { val = exp(val) - 1.0; }
        }
      }
      As[kk * TM + r] = val;
    }
    for (var q = 0u; q < 4u; q++) {
      let e = li + q * 256u;
      let kk = e / TN;
      let n = e % TN;
      let col = col0 + n;
      var val = 0.0;
      if (col < p.ncols) { val = w[(k0 + kk) * p.ncols + col]; }
      Bs[kk * TN + n] = val;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < TK; kk++) {
      var a: array<f32, 4>;
      var bb: array<f32, 4>;
      for (var r = 0u; r < 4u; r++) { a[r] = As[kk * TM + lid.y + 16u * r]; }
      for (var c = 0u; c < 4u; c++) { bb[c] = Bs[kk * TN + lid.x + 16u * c]; }
      for (var r = 0u; r < 4u; r++) {
        for (var c = 0u; c < 4u; c++) { acc[r][c] = fma(a[r], bb[c], acc[r][c]); }
      }
    }
    workgroupBarrier();
  }

  for (var r = 0u; r < 4u; r++) {
    let t = row0 + lid.y + 16u * r;
    if (t >= p.rows_out) { continue; }
    for (var c = 0u; c < 4u; c++) {
      let n = col0 + lid.x + 16u * c;
      if (n >= p.ncols) { continue; }
      let ph = n / p.cout;
      let o = n % p.cout;
      let f = i32(t * p.phases + ph) - i32(p.trim_left);
      if (f < 0 || f >= i32(p.out_len)) { continue; }
      let idx = (b * p.out_item + p.out_start + u32(f)) * p.cout + o;
      var v = acc[r][c] + bias[o];
      if (p.accumulate == 1u) { v += y[idx]; }
      y[idx] = v;
    }
  }
}
`;

/**
 * One LSTM timestep for every batch item: gates = gx[t] + W_hh h[t-1], then
 * the cell update (PyTorch gate order i, f, g, o). Each workgroup owns `units`
 * hidden units (4 * units gate rows, 64 / units threads per row) and first
 * stages h[t-1] of every batch item in workgroup memory. `h` holds row 0 =
 * initial state and row t+1 = output at step t, so step t reads row t and
 * writes row t+1.
 */
export function lstmStepWgsl(units: number, hidden: number, packedHalf: boolean): string {
  const tpr = 64 / units;
  // The step is bound by reading W_hh (4 MB per layer in f32), so with f16
  // weights it is stored as packed halves (2 per u32), which is lossless there.
  const whhDecl = packedHalf ? 'array<vec2u>' : 'array<vec4f>';
  const load = packedHalf
    ? 'let q = whh[row * H4 + lane + m * TPR]; wv[m] = vec4f(unpack2x16float(q.x), unpack2x16float(q.y));'
    : 'wv[m] = whh[row * H4 + lane + m * TPR];';
  return /* wgsl */ `
struct P { hidden: u32, gx_item: u32, h_item: u32, batch: u32 }
struct S { t: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> s: S;
@group(0) @binding(2) var<storage, read> whh: ${whhDecl};
@group(0) @binding(3) var<storage, read> gx: array<f32>;
@group(0) @binding(4) var<storage, read_write> h: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;

const H: u32 = ${hidden}u;
const H4: u32 = ${hidden / 4}u;
const KPT: u32 = ${hidden / 4 / tpr}u; // vec4 weight loads per thread
const U: u32 = ${units}u;
const TPR: u32 = ${tpr}u;
const MAXB: u32 = 8u;
// h[t-1] as vec4 ([MAXB][hidden / 4], hidden <= 512); then reused for the partial
// sums (floats [0, MAXB * 256)) and the gate sums (floats from 3072).
var<workgroup> sh: array<vec4f, 1024>;

fn gsum(i: u32) -> f32 { return sh[i / 4u][i % 4u]; }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
fn tanh_safe(v: f32) -> f32 {
  let e = exp(-2.0 * abs(v));
  return sign(v) * (1.0 - e) / (1.0 + e);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = s.t;
  for (var i = li; i < p.batch * H4; i += 256u) {
    let b = i / H4;
    let base = (b * p.h_item + t) * H + 4u * (i % H4);
    sh[i] = vec4f(h[base], h[base + 1u], h[base + 2u], h[base + 3u]);
  }
  workgroupBarrier();
  let row_local = li / TPR;
  let lane = li % TPR;
  let gate = row_local / U;
  let u = row_local % U;
  let row = gate * H + wg.x * U + u;
  var acc: array<f32, MAXB>;
  for (var b = 0u; b < MAXB; b++) { acc[b] = 0.0; }
  // Issue every weight load before using any (constant trip counts unroll).
  var wv: array<vec4f, KPT>;
  for (var m = 0u; m < KPT; m++) { ${load} }
  for (var m = 0u; m < KPT; m++) {
    let k4 = lane + m * TPR;
    for (var b = 0u; b < p.batch; b++) { acc[b] += dot(wv[m], sh[b * H4 + k4]); }
  }
  workgroupBarrier();
  for (var b = 0u; b < p.batch; b++) {
    let r = b * 256u + li;
    sh[r / 4u][r % 4u] = acc[b];
  }
  workgroupBarrier();
  if (lane == 0u) {
    for (var b = 0u; b < p.batch; b++) {
      var total = 0.0;
      for (var l = 0u; l < TPR; l++) {
        let r = b * 256u + li + l;
        total += sh[r / 4u][r % 4u];
      }
      let g = 3072u + b * 4u * U + row_local;
      sh[g / 4u][g % 4u] = total;
    }
  }
  workgroupBarrier();
  if (li < U * p.batch) {
    let uu = li % U;
    let b = li / U;
    let j = wg.x * U + uu;
    let g0 = (b * p.gx_item + t) * 4u * H;
    let sb = 3072u + b * 4u * U + uu;
    let ig = sigm(gx[g0 + j] + gsum(sb));
    let fg = sigm(gx[g0 + H + j] + gsum(sb + U));
    let gg = tanh_safe(gx[g0 + 2u * H + j] + gsum(sb + 2u * U));
    let og = sigm(gx[g0 + 3u * H + j] + gsum(sb + 3u * U));
    let ci = b * H + j;
    let cn = fg * c[ci] + ig * gg;
    c[ci] = cn;
    h[(b * p.h_item + t + 1u) * H + j] = og * tanh_safe(cn);
  }
}
`;
}

/** Sum of codebook embeddings for each frame. */
export const QUANTIZER_WGSL = /* wgsl */ `
struct P { frames: u32, dim: u32, nq: u32, cb_size: u32, out_start: u32, out_item: u32, batch: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> cb: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let total = p.batch * p.frames * p.dim;
  for (var i = gid.x; i < total; i += nw.x * 256u) {
    let d = i % p.dim;
    let bt = i / p.dim;
    let t = bt % p.frames;
    let b = bt / p.frames;
    var acc = 0.0;
    for (var q = 0u; q < p.nq; q++) {
      let code = codes[bt * p.nq + q];
      acc += cb[(q * p.cb_size + code) * p.dim + d];
    }
    y[(b * p.out_item + p.out_start + t) * p.dim + d] = acc;
  }
}
`;

/** dst = a + b over matching row ranges (the LSTM skip connection). */
export const ADD_WGSL = /* wgsl */ `
struct P { rows: u32, ch: u32, a_start: u32, a_item: u32, b_start: u32, b_item: u32, d_start: u32, d_item: u32, batch: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> bsrc: array<f32>;
@group(0) @binding(3) var<storage, read_write> d: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let per = p.rows * p.ch;
  let total = p.batch * per;
  for (var i = gid.x; i < total; i += nw.x * 256u) {
    let b = i / per;
    let k = i % per;
    d[(b * p.d_item + p.d_start) * p.ch + k] =
      a[(b * p.a_item + p.a_start) * p.ch + k] + bsrc[(b * p.b_item + p.b_start) * p.ch + k];
  }
}
`;

/** Number of partial sums per batch item in the group-norm reductions. */
export const GN_PARTS = 64;

/**
 * Group-norm statistics (one group) in two passes for accuracy:
 * mode 0 writes partial sums, mode 1 partial sums of squared deviations
 * from the mean of pass 0. Dispatch (GN_PARTS, batch).
 */
export const GN_STATS_WGSL = /* wgsl */ `
struct P { rows: u32, ch: u32, start: u32, item: u32, mode: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>; // [batch][2][64]
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let b = wg.y;
  let n = p.rows * p.ch;
  let base = (b * p.item + p.start) * p.ch;
  var mean = 0.0;
  if (p.mode == 1u) {
    sh[li] = select(0.0, part[b * 128u + li], li < 64u);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s / 2u) {
      if (li < s) { sh[li] += sh[li + s]; }
      workgroupBarrier();
    }
    mean = sh[0] / f32(n);
    workgroupBarrier();
  }
  var acc = 0.0;
  for (var i = wg.x * 256u + li; i < n; i += 64u * 256u) {
    let v = x[base + i];
    if (p.mode == 0u) { acc += v; } else { let dv = v - mean; acc += dv * dv; }
  }
  sh[li] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s / 2u) {
    if (li < s) { sh[li] += sh[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { part[b * 128u + p.mode * 64u + wg.x] = sh[0]; }
}
`;

/** Applies group norm from the partial sums: dst (+)= (x - mean) / std * gamma + beta. */
export const GN_APPLY_WGSL = /* wgsl */ `
struct P {
  rows: u32, ch: u32, stat_rows: u32, src_start: u32,
  src_item: u32, dst_start: u32, dst_item: u32, accumulate: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> part: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32,
        @builtin(num_workgroups) nw: vec3u) {
  let b = wg.y;
  sh[li] = select(0.0, part[b * 128u + li], li < 128u);
  workgroupBarrier();
  // Reduce the two halves separately: sums in [0, 64), squared deviations in [64, 128).
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (li < s || (li >= 64u && li < 64u + s)) { sh[li] += sh[li + s]; }
    workgroupBarrier();
  }
  let count = f32(p.stat_rows * p.ch);
  let mean = sh[0] / count;
  let var_ = sh[64] / count;
  let inv = 1.0 / sqrt(var_ + p.eps);
  let n = p.rows * p.ch;
  let sbase = (b * p.src_item + p.src_start) * p.ch;
  let dbase = (b * p.dst_item + p.dst_start) * p.ch;
  for (var i = wg.x * 256u + li; i < n; i += nw.x * 256u) {
    let c = i % p.ch;
    var v = (x[sbase + i] - mean) * inv * gamma[c] + beta[c];
    if (p.accumulate == 1u) { v += y[dbase + i]; }
    y[dbase + i] = v;
  }
}
`;

/**
 * Scale and triangle-weighted overlap-add (`_linear_overlap_add`) of a batch
 * of decoded segments, plus the tail of the previous batch's last segment.
 * Writes `count` finished samples starting at global sample `t0`.
 */
export const OLA_WGSL = /* wgsl */ `
struct P {
  count: u32, ch: u32, t0: u32, k0: u32,
  nseg: u32, seg_len: u32, seg_item: u32, stride: u32,
  frame_len: u32, has_prev: u32, prev_len: u32, mode: u32,
  // Cross-fade: 0 Meta's triangle, 1 edge-avoiding (the ID header's edge flag); lo/hi bound
  // the edge ramp inside the overlap of ov samples (layout::EdgeRamp).
  xf: u32, lo: u32, hi: u32, ov: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> seg: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> tail: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn ramp(t: u32) -> f32 {
  if (t < p.lo) { return 0.0; }
  if (t < p.hi) { return f32(t - p.lo + 1u) / f32(p.hi - p.lo + 1u); }
  return 1.0;
}

// Weight of segment k at local samples from its start.
fn weight(local: u32, k: u32) -> f32 {
  if (p.xf == 0u) {
    return 0.5 - abs(f32(local + 1u) / f32(p.frame_len + 1u) - 0.5);
  }
  var w = 1.0;
  if (k > 0u && local < p.ov) { w = ramp(local); }
  if (local >= p.stride) { w *= 1.0 - ramp(local - p.stride); }
  return w;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
  if (p.mode == 1u) {
    // Save the overlap tail of the batch's last segment (scaled) for the next batch.
    let rows = p.seg_len - p.stride;
    let b = p.nseg - 1u;
    for (var i = gid.x; i < rows * p.ch; i += nw.x * 256u) {
      tail[i] = seg[(b * p.seg_item + p.stride) * p.ch + i] * scales[b];
    }
    return;
  }
  for (var i = gid.x; i < p.count * p.ch; i += nw.x * 256u) {
    let c = i % p.ch;
    let t = p.t0 + i / p.ch;
    let kc = t / p.stride;
    var acc = 0.0;
    var wsum = 0.0;
    var plain = 0.0;
    var n = 0.0;
    for (var d = 0u; d < 2u; d++) {
      if (kc < d) { break; }
      let k = kc - d;
      let local = t - k * p.stride;
      var v = 0.0;
      var hit = false;
      if (k >= p.k0 && k < p.k0 + p.nseg) {
        if (local < p.seg_len) {
          let b = k - p.k0;
          v = seg[(b * p.seg_item + local) * p.ch + c] * scales[b];
          hit = true;
        }
      } else if (p.has_prev == 1u && k + 1u == p.k0 && local < p.prev_len) {
        v = tail[(local - p.stride) * p.ch + c];
        hit = true;
      }
      if (hit) {
        let wt = weight(local, k);
        acc += wt * v;
        wsum += wt;
        plain += v;
        n += 1.0;
      }
    }
    // Edge weights can all be zero (never the triangle's): fall back to the mean.
    y[i] = select(plain / max(n, 1.0), acc / wsum, wsum > 0.0);
  }
}
`;
