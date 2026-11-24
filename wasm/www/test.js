// Harness for test/run.mjs: decodes /test-data/<name>.pico in the browser and
// compares it with the native reference /test-data/<name>.ref.wav.
import { PicotoneDecoder, readPicoHeader } from "./picotone/picotone.js";

function readWavFloat(buffer) {
  const view = new DataView(buffer);
  let at = 12;
  let channels = 0;
  let rate = 0;
  let format = 0;
  let bits = 0;
  while (at + 8 <= view.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buffer, at, 4));
    const size = view.getUint32(at + 4, true);
    if (id === "fmt ") {
      format = view.getUint16(at + 8, true);
      channels = view.getUint16(at + 10, true);
      rate = view.getUint32(at + 12, true);
      bits = view.getUint16(at + 22, true);
      if (format === 0xfffe) format = view.getUint16(at + 32, true);
    } else if (id === "data") {
      if (format !== 3 || bits !== 32) throw new Error(`reference must be float32 WAV (format ${format}, ${bits} bits)`);
      const frames = size / 4 / channels;
      const out = Array.from({ length: channels }, () => new Float32Array(frames));
      for (let t = 0; t < frames; t++) {
        for (let c = 0; c < channels; c++) out[c][t] = view.getFloat32(at + 8 + (t * channels + c) * 4, true);
      }
      return { rate, channels: out };
    }
    at += 8 + size + (size & 1);
  }
  throw new Error("no data chunk");
}

function compare(ref, got) {
  let signal = 0;
  let noise = 0;
  let maxDiff = 0;
  let exact = 0;
  let total = 0;
  for (let c = 0; c < ref.length; c++) {
    const a = ref[c];
    const b = got[c];
    if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
    for (let t = 0; t < a.length; t++) {
      // The CLI's WAV writer hard-limits to +-0.99 (as Meta's does); the decoder returns raw samples.
      const d = a[t] - Math.max(-0.99, Math.min(0.99, b[t]));
      signal += a[t] * a[t];
      noise += d * d;
      maxDiff = Math.max(maxDiff, Math.abs(d));
      if (d === 0) exact++;
      total++;
    }
  }
  return {
    snrDb: noise === 0 ? Infinity : 10 * Math.log10(signal / noise),
    maxAbsDiff: maxDiff,
    exactFraction: exact / total,
  };
}

/** Jump across the loop seam [start, end) relative to the typical sample-to-sample step. */
function seam(channels, start = 0, end = channels[0].length) {
  const steps = [];
  let jump = 0;
  for (const x of channels) {
    for (let t = start + 1; t < end; t++) steps.push(Math.abs(x[t] - x[t - 1]));
    jump = Math.max(jump, Math.abs(x[start] - x[end - 1]));
  }
  steps.sort((p, q) => p - q);
  const median = steps[Math.floor(steps.length / 2)] ?? 0;
  const p999 = steps[Math.floor(steps.length * 0.999)] ?? 0;
  return { jump, medianStep: median, p999Step: p999 };
}

let decoder = null;
const loaded = new Map();

window.picotoneTest = {
  async init(build, threads = 0, relaxedSimd = true) {
    decoder?.dispose();
    loaded.clear();
    decoder = await PicotoneDecoder.create({ baseUrl: "./picotone/", modelsUrl: "./models/", build, threads, relaxedSimd, cacheName: null });
    return { build: decoder.build, isSecureContext: self.isSecureContext, crossOriginIsolated: self.crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency };
  },

  async runCase(name, { stream = true, chunkSegments = undefined, firstChunkSegments = undefined, seamCheck = false, loopRegion = null, startSample = 0, hqDecoder = undefined } = {}) {
    const bytes = new Uint8Array(await (await fetch(`/test-data/${name}.pico`)).arrayBuffer());
    const info = readPicoHeader(bytes);
    // HQ files play with either network; switching reloads (the worker keeps one per model).
    const key = `${info.model}/${hqDecoder ?? ""}`;
    if (!loaded.has(key)) {
      for (const k of [...loaded.keys()]) if (k.startsWith(`${info.model}/`)) loaded.delete(k);
      loaded.set(key, await decoder.loadModel(info.model, { hqDecoder }));
    }
    const ref = readWavFloat(await (await fetch(`/test-data/${name}.ref.wav`)).arrayBuffer());
    let result;
    let startedMs = null;
    let underruns = 0;
    const chunkSizes = [];
    if (stream) {
      const context = new AudioContext({ sampleRate: 48000 });
      await context.resume();
      const silent = context.createGain();
      silent.gain.value = 0;
      silent.connect(context.destination);
      const playback = decoder.play(context, bytes, {
        destination: silent,
        chunkSegments,
        firstChunkSegments,
        startSample,
        onChunk: (chunk) => chunkSizes.push(chunk.length),
      });
      result = await playback.decoded;
      startedMs = await playback.started;
      underruns = playback.underruns;
      playback.stop();
      await context.close();
    } else {
      result = await decoder.decode(bytes, { chunkSegments, firstChunkSegments, startSample, onChunk: (chunk) => chunkSizes.push(chunk.length) });
    }
    return {
      name,
      model: info.model,
      seconds: info.durationSeconds,
      stats: result.stats,
      startedMs,
      underruns,
      chunks: chunkSizes.length,
      network: loaded.get(key).network,
      modelLoadMs: loaded.get(key).loadMs,
      startSample,
      parity: compare(ref.channels.map((c) => c.subarray(startSample)), result.channels),
      seam: seamCheck
        ? { decoded: loopRegion ? seam(result.channels, loopRegion[0], loopRegion[1]) : seam(result.channels), reference: loopRegion ? seam(ref.channels, loopRegion[0], loopRegion[1]) : seam(ref.channels) }
        : null,
    };
  },
};
/**
 * Decodes one file in `workers` single-threaded decoders at once (one Web Worker each), as a
 * player that decodes each coded channel in its own worker would: the wall time of all, and
 * each decode's own stats.
 */
window.picotoneConcurrent = async (name, workers, hqDecoder = undefined) => {
  const bytes = new Uint8Array(await (await fetch(`/test-data/${name}.pico`)).arrayBuffer());
  const info = readPicoHeader(bytes);
  const decoders = await Promise.all(
    Array.from({ length: workers }, () =>
      PicotoneDecoder.create({ baseUrl: "./picotone/", modelsUrl: "./models/", build: "single", relaxedSimd: true, cacheName: null }),
    ),
  );
  try {
    await Promise.all(decoders.map((d) => d.loadModel(info.model, { hqDecoder })));
    const t0 = performance.now();
    const results = await Promise.all(decoders.map((d) => d.decode(bytes.slice())));
    return { wallMs: performance.now() - t0, seconds: info.durationSeconds, stats: results.map((r) => r.stats) };
  } finally {
    for (const d of decoders) d.dispose();
  }
};
window.picotoneInfo = (buffer) => readPicoHeader(buffer).audioLength;
window.picotoneReady = true;
