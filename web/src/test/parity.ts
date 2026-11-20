/** Parity harness run in headless Chrome by tools/test-parity.mjs. */

import { PicotoneWebGpu, type WeightDtype, checkWebGpu, decodeWav, parseHeader } from '../index.js';

export interface ParityCase {
  readonly name: string;
  readonly pico: string;
  readonly ref: string;
}

export interface ParityResult {
  readonly name: string;
  readonly model: string;
  readonly dtype: WeightDtype;
  readonly codebooks: number;
  readonly seconds: number;
  readonly snrDb: number;
  readonly maxAbsErr: number;
  readonly coldMs: number;
  readonly warmMs: number;
  readonly realtime: number;
  readonly firstAudioColdMs: number;
  readonly firstAudioWarmMs: number;
  /** Whether every chunk arrived before playback (started at first audio) would need it. */
  readonly streamAhead: boolean;
  readonly minLeadMs: number;
}

export interface ParityReport {
  readonly adapter: string;
  readonly loads: { readonly model: string; readonly dtype: WeightDtype; readonly ms: number; readonly gpuMB: number }[];
  readonly results: ParityResult[];
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

interface Timed {
  readonly ms: number;
  readonly firstMs: number;
  readonly channels: readonly Float32Array[];
  readonly ahead: boolean;
  readonly minLeadMs: number;
}

async function timedDecode(model: PicotoneWebGpu, bytes: Uint8Array): Promise<Timed> {
  const t0 = performance.now();
  let firstMs = -1;
  let minLead = Infinity;
  const result = await model.decode(bytes, {
    onChunk: (c) => {
      const now = performance.now() - t0;
      if (firstMs < 0) {
        firstMs = now;
        return;
      }
      // Playback starts at first audio; this chunk is needed at firstMs + offset / rate.
      const due = firstMs + (c.offset / c.sampleRate) * 1000;
      minLead = Math.min(minLead, due - now);
    },
  });
  return { ms: performance.now() - t0, firstMs, channels: result.channels, ahead: minLead >= 0, minLeadMs: minLead };
}

async function run(cases: readonly ParityCase[], dtypes: readonly WeightDtype[]): Promise<ParityReport> {
  const gpu = await checkWebGpu();
  if (!gpu.available) throw new Error(`WebGPU unavailable: ${gpu.reason}`);
  const adapter = await navigator.gpu.requestAdapter();
  const info = adapter?.info;
  const report: ParityReport = { adapter: info ? `${info.vendor} ${info.architecture} ${info.description}`.trim() : '?', loads: [], results: [] };
  const models = new Map<string, PicotoneWebGpu>();
  for (const dtype of dtypes) {
    for (const c of cases) {
      const bytes = await fetchBytes(c.pico);
      const header = parseHeader(bytes);
      const key = `${header.model}/${dtype}`;
      let model = models.get(key);
      if (!model) {
        const t0 = performance.now();
        model = await PicotoneWebGpu.load({ model: header.model, baseUrl: '../public/models/', dtype });
        const g = model.gpuBytes;
        report.loads.push({ model: header.model, dtype, ms: performance.now() - t0, gpuMB: (g.weights + g.activations) / 1e6 });
        models.set(key, model);
      }
      const ref = decodeWav(await fetchBytes(c.ref));
      const cold = await timedDecode(model, bytes);
      const warm = await timedDecode(model, bytes);
      let sig = 0;
      let err = 0;
      let maxAbs = 0;
      ref.channels.forEach((r, ch) => {
        const o = warm.channels[ch];
        if (!o || o.length !== r.length) throw new Error(`${c.name}: output length ${o?.length} vs reference ${r.length}`);
        for (let i = 0; i < r.length; i++) {
          const rv = r[i] ?? 0;
          // The CLI's WAV writer hard-limits to +-0.99 (as Meta's does); the decoder returns raw samples.
          const d = Math.max(-0.99, Math.min(0.99, o[i] ?? 0)) - rv;
          sig += rv * rv;
          err += d * d;
          maxAbs = Math.max(maxAbs, Math.abs(d));
        }
      });
      const seconds = header.audioLength / model.sampleRate;
      report.results.push({
        name: c.name,
        model: header.model,
        dtype,
        codebooks: header.numCodebooks,
        seconds,
        snrDb: 10 * Math.log10(sig / Math.max(err, 1e-30)),
        maxAbsErr: maxAbs,
        coldMs: cold.ms,
        warmMs: warm.ms,
        realtime: (seconds * 1000) / warm.ms,
        firstAudioColdMs: cold.firstMs,
        firstAudioWarmMs: warm.firstMs,
        streamAhead: warm.ahead,
        minLeadMs: warm.minLeadMs,
      });
    }
  }
  for (const m of models.values()) m.destroy();
  return report;
}

/** Per-kernel GPU time (ms) for one file, largest first. */
async function profile(pico: string): Promise<[string, number][]> {
  const bytes = await fetchBytes(pico);
  const model = await PicotoneWebGpu.load({ model: parseHeader(bytes).model, baseUrl: '../public/models/' });
  await model.decode(bytes);
  const out = await model.decode(bytes, { profile: true });
  model.destroy();
  return Object.entries(out.profileMs ?? {}).sort((a, b) => b[1] - a[1]);
}

declare global {
  interface Window {
    runParity?: (cases: readonly ParityCase[], dtypes: readonly WeightDtype[]) => Promise<ParityReport>;
    runProfile?: (pico: string) => Promise<[string, number][]>;
  }
}
window.runParity = run;
window.runProfile = profile;
