/** Demo page: pick a .pico file, decode it on the GPU, play it, save it as WAV. */

import {
  type DecodedAudio,
  type PicoHeader,
  type WeightDtype,
  PicotoneWebGpu,
  bandwidthKbps,
  checkWebGpu,
  encodeWav,
  frameRate,
  parseHeader,
} from '../index.js';

const MODEL_BASE_URL = 'public/models/';
const SAMPLE_RATES: Record<string, number> = { encodec_24khz: 24000, encodec_48khz: 48000 };
const CHANNELS: Record<string, number> = { encodec_24khz: 1, encodec_48khz: 2 };

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`missing #${id}`);
  return e as T;
}

const ui = {
  gpu: el<HTMLDivElement>('gpu'),
  drop: el<HTMLDivElement>('drop'),
  file: el<HTMLInputElement>('file'),
  metaCard: el<HTMLDivElement>('meta-card'),
  meta: el<HTMLDListElement>('meta'),
  dtype: el<HTMLSelectElement>('dtype'),
  stream: el<HTMLInputElement>('stream'),
  decode: el<HTMLButtonElement>('decode'),
  play: el<HTMLButtonElement>('play'),
  stop: el<HTMLButtonElement>('stop'),
  download: el<HTMLButtonElement>('download'),
  progress: el<HTMLProgressElement>('progress'),
  status: el<HTMLDivElement>('status'),
};

let gpuOk = false;
let current: { name: string; bytes: Uint8Array; header: PicoHeader } | null = null;
let decoded: DecodedAudio | null = null;
const models = new Map<string, Promise<PicotoneWebGpu>>();
let ctx: AudioContext | null = null;
let sources: AudioBufferSourceNode[] = [];

function status(text: string, error = false): void {
  ui.status.textContent = text;
  ui.status.classList.toggle('err', error);
}

function audioContext(rate: number): AudioContext {
  if (!ctx || ctx.sampleRate !== rate) {
    void ctx?.close();
    ctx = new AudioContext({ sampleRate: rate });
  }
  return ctx;
}

function stopPlayback(): void {
  for (const s of sources) {
    try {
      s.stop();
    } catch {
      // already stopped
    }
  }
  sources = [];
  ui.stop.disabled = true;
}

function schedule(ac: AudioContext, channels: readonly Float32Array[], when: number): void {
  const n = channels[0]?.length ?? 0;
  if (n === 0) return;
  const buf = ac.createBuffer(channels.length, n, ac.sampleRate);
  channels.forEach((ch, c) => buf.copyToChannel(new Float32Array(ch), c));
  const src = ac.createBufferSource();
  src.buffer = buf;
  src.connect(ac.destination);
  src.start(Math.max(when, ac.currentTime));
  sources.push(src);
  ui.stop.disabled = false;
  src.onended = () => {
    sources = sources.filter((s) => s !== src);
    if (sources.length === 0) ui.stop.disabled = true;
  };
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`;
}

function showMeta(name: string, size: number, h: PicoHeader): void {
  const rate = SAMPLE_RATES[h.model];
  const rows: [string, string][] = [
    ['File', name],
    ['Size', `${(size / 1024).toFixed(1)} KiB`],
    ['Model', h.model],
    ['Coding', `cx2, table ${h.tableHash}`],
    ['Codebooks', String(h.numCodebooks)],
  ];
  if (rate) {
    const cfgLike = { upsampling_ratios: [8, 5, 4, 2], sampling_rate: rate, codebook_size: 1024 };
    const seconds = h.audioLength / rate;
    rows.push(
      ['Duration', `${fmtTime(seconds)} (${h.audioLength.toLocaleString()} samples)`],
      ['Sample rate', `${rate} Hz, ${CHANNELS[h.model] === 2 ? 'stereo' : 'mono'}`],
      ['Bandwidth', `${bandwidthKbps(cfgLike, h.numCodebooks)} kbps nominal, ${((size * 8) / seconds / 1000).toFixed(2)} kbps actual`],
      ['Frame rate', `${frameRate(cfgLike)} Hz`],
    );
    if (h.stride > 0) rows.push(['Segment stride', `${h.stride} samples`]);
    if (h.crossfade === 1) rows.push(['Cross-fade', 'edge-avoiding']);
    if (h.preSkip > 0 || h.endTrim > 0) rows.push(['Trim', `${h.preSkip} + ${h.endTrim} samples`]);
    if (h.loop) rows.push(['Loop', 'gapless (end runs into start)']);
  }
  ui.meta.replaceChildren(
    ...rows.flatMap(([k, v]) => {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = v;
      return [dt, dd];
    }),
  );
  ui.metaCard.hidden = false;
}

async function openFile(file: File): Promise<void> {
  stopPlayback();
  decoded = null;
  ui.play.disabled = true;
  ui.download.disabled = true;
  ui.progress.value = 0;
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const header = parseHeader(bytes);
    current = { name: file.name, bytes, header };
    showMeta(file.name, bytes.length, header);
    if (!SAMPLE_RATES[header.model]) throw new Error(`unknown model ${header.model}`);
    ui.decode.disabled = !gpuOk;
    status(gpuOk ? 'Ready to decode.' : 'WebGPU is unavailable, so this file cannot be decoded here.', !gpuOk);
  } catch (e) {
    current = null;
    ui.decode.disabled = true;
    status(e instanceof Error ? e.message : String(e), true);
  }
}

function getModel(name: string, dtype: WeightDtype): Promise<PicotoneWebGpu> {
  const key = `${name}/${dtype}`;
  let p = models.get(key);
  if (!p) {
    p = PicotoneWebGpu.load({
      model: name,
      baseUrl: MODEL_BASE_URL,
      dtype,
      onProgress: (loaded, total) => {
        ui.progress.value = total ? loaded / total : 0;
        status(`Downloading ${name} weights (${dtype}): ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
      },
    });
    p.catch(() => models.delete(key));
    models.set(key, p);
  }
  return p;
}

async function decode(): Promise<void> {
  if (!current) return;
  const { bytes, header, name } = current;
  stopPlayback();
  ui.decode.disabled = true;
  ui.play.disabled = true;
  ui.download.disabled = true;
  try {
    const dtype = ui.dtype.value === 'f32' ? 'f32' : 'f16';
    const model = await getModel(header.model, dtype);
    const streaming = ui.stream.checked;
    const ac = streaming ? audioContext(model.sampleRate) : null;
    if (ac) await ac.resume();
    let startAt = 0;
    let firstMs = -1;
    const t0 = performance.now();
    status('Decoding...');
    ui.progress.value = 0;
    decoded = await model.decode(bytes, {
      onProgress: (f) => {
        ui.progress.value = f;
        status(`Decoding... ${(f * 100).toFixed(0)}%`);
      },
      onChunk: (chunk) => {
        if (firstMs < 0) {
          firstMs = performance.now() - t0;
          if (ac) startAt = ac.currentTime + 0.05;
        }
        if (ac) schedule(ac, chunk.channels, startAt + chunk.offset / chunk.sampleRate);
      },
    });
    const ms = performance.now() - t0;
    const seconds = decoded.length / decoded.sampleRate;
    status(
      `Decoded ${fmtTime(seconds)} of ${name} in ${(ms / 1000).toFixed(2)} s ` +
        `(${(seconds / (ms / 1000)).toFixed(1)}x realtime, first audio after ${firstMs.toFixed(0)} ms).`,
    );
    ui.play.disabled = false;
    ui.download.disabled = false;
  } catch (e) {
    status(`Decoding failed: ${e instanceof Error ? e.message : String(e)}`, true);
  } finally {
    ui.decode.disabled = !current || !gpuOk;
  }
}

function play(): void {
  if (!decoded) return;
  stopPlayback();
  const ac = audioContext(decoded.sampleRate);
  void ac.resume().then(() => {
    if (decoded) schedule(ac, decoded.channels, ac.currentTime + 0.02);
  });
}

function download(): void {
  if (!decoded || !current) return;
  const wav = encodeWav(decoded, 'pcm16');
  const url = URL.createObjectURL(new Blob([wav.slice().buffer], { type: 'audio/wav' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = current.name.replace(/\.pico$/i, '') + '.wav';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

ui.drop.addEventListener('click', () => ui.file.click());
ui.drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') ui.file.click();
});
ui.file.addEventListener('change', () => {
  const f = ui.file.files?.[0];
  if (f) void openFile(f);
});
ui.drop.addEventListener('dragover', (e) => {
  e.preventDefault();
  ui.drop.classList.add('over');
});
ui.drop.addEventListener('dragleave', () => ui.drop.classList.remove('over'));
ui.drop.addEventListener('drop', (e) => {
  e.preventDefault();
  ui.drop.classList.remove('over');
  const f = e.dataTransfer?.files[0];
  if (f) void openFile(f);
});
ui.decode.addEventListener('click', () => void decode());
ui.play.addEventListener('click', play);
ui.stop.addEventListener('click', stopPlayback);
ui.download.addEventListener('click', download);

void checkWebGpu().then((r) => {
  gpuOk = r.available;
  if (!r.available) {
    ui.gpu.textContent = `WebGPU unavailable: ${r.reason}`;
    ui.gpu.classList.add('err');
  }
  if (current) ui.decode.disabled = !gpuOk;
});
