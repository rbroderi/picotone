// Generates synthetic test audio and reference decodes with the Rust CLI.
//
// Usage: node tools/make-refs.mjs [--dir <scratch dir>] [--exe <picotone.exe>] [--weights <dir>] [--force]
// Defaults: $PICOTONE_TEST_DIR (else <os tmp>/picotone-web-test), $PICOTONE_EXE (else
// ../target/release/picotone.exe), ../weights.
// Writes <dir>/<case>.pico and <dir>/<case>.ref.wav plus cases.json. Existing outputs are reused.

import { copyFileSync, writeFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_TEST_DIR } from './paths.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const dir = resolve(opt('--dir', DEFAULT_TEST_DIR));
const exe = resolve(opt('--exe', process.env.PICOTONE_EXE ?? resolve(here, '..', '..', 'target', 'release', 'picotone.exe')));
const weights = resolve(opt('--weights', resolve(here, '..', '..', 'weights')));
const force = args.includes('--force');
mkdirSync(dir, { recursive: true });

// Deterministic PRNG.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Synthetic "music": plucked notes with harmonics, a bass line, hi-hat noise bursts. */
function synth(rate, seconds, channels, seed) {
  const n = Math.round(rate * seconds);
  const out = Array.from({ length: channels }, () => new Float32Array(n));
  const rand = rng(seed);
  const scale = [0, 2, 4, 7, 9, 12, 14, 16];
  const beat = 0.25;
  for (let start = 0; start < seconds; start += beat) {
    const note = scale[Math.floor(rand() * scale.length)];
    const f = 220 * 2 ** (note / 12);
    const pan = rand();
    const amp = 0.15 + 0.1 * rand();
    const s0 = Math.floor(start * rate);
    const len = Math.min(n - s0, Math.floor(rate * 0.6));
    for (let i = 0; i < len; i++) {
      const t = i / rate;
      const env = Math.exp(-t * 6) * Math.min(1, t * 200);
      let v = 0;
      for (let h = 1; h <= 5; h++) v += Math.sin(2 * Math.PI * f * h * t) / (h * h);
      v *= amp * env;
      for (let c = 0; c < channels; c++) out[c][s0 + i] += v * (channels === 1 ? 1 : c === 0 ? 1 - pan : pan) * 1.4;
    }
    // Hi-hat burst on every other beat.
    if (Math.round(start / beat) % 2 === 1) {
      const hl = Math.min(n - s0, Math.floor(rate * 0.05));
      for (let i = 0; i < hl; i++) {
        const v = (rand() * 2 - 1) * 0.08 * Math.exp(-i / (rate * 0.01));
        for (let c = 0; c < channels; c++) out[c][s0 + i] += v;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const bass = 0.12 * Math.sin(2 * Math.PI * (55 * (1 + Math.floor(t / 2) % 2 * 0.5)) * t);
    for (let c = 0; c < channels; c++) out[c][i] += bass;
  }
  return out;
}

function writeWav16(path, rate, chans) {
  const C = chans.length;
  const n = chans[0].length;
  const buf = Buffer.alloc(44 + n * C * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * C * 2, 4);
  buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(C, 22);
  buf.writeUInt32LE(rate, 24);
  buf.writeUInt32LE(rate * C * 2, 28);
  buf.writeUInt16LE(C * 2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * C * 2, 40);
  let o = 44;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < C; c++) {
      buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, chans[c][i])) * 32767), o);
      o += 2;
    }
  writeFileSync(path, buf);
}

// name, model, rate, channels, seconds, bandwidth, seed[, encode args replacing -b, [source wav]]
const CASES = [
  ['mono24_10s_1.5k', 'mono', 24000, 1, 10, 1.5, 1],
  ['mono24_10s_24k', 'mono', 24000, 1, 10, 24, 1],
  ['mono24_60s_6k', 'mono', 24000, 1, 60, 6, 2],
  ['mono24_fx_0.5s_6k', 'mono', 24000, 1, 0.5, 6, 3],
  ['stereo48_3.5s_3k', 'stereo', 48000, 2, 3.5, 3, 4],
  ['stereo48_3.5s_24k', 'stereo', 48000, 2, 3.5, 24, 4],
  ['stereo48_75s_6k', 'stereo', 48000, 2, 75, 6, 5],
  ['stereo48_fx_0.4s_12k', 'stereo', 48000, 2, 0.4, 12, 6],
  // Configurable stride and cross-fade; the presets at beam 1 keep the refs quick to make.
  ['stereo48_8s_music24', 'stereo', 48000, 2, 8, 0, 7, ['--preset', 'music24', '--beam', '1']],
  ['stereo48_8s_music12', 'stereo', 48000, 2, 8, 0, 8, ['--preset', 'music12', '--beam', '1']],
  ['stereo48_8s_music6', 'stereo', 48000, 2, 8, 0, 9, ['--preset', 'music6', '--beam', '1']],
  ['stereo48_3s_edge_ss24000', 'stereo', 48000, 2, 3, 0, 10, ['--codebooks', '8', '--stride', '24000', '--crossfade', 'edge']],
  // Gapless loops (pre-skip, end trim, loop flag): the decode is trimmed and its end folded onto its start.
  ['stereo48_5s_loop_music6', 'stereo', 48000, 2, 5, 0, 11, ['--preset', 'music6', '--beam', '1', '--loop']],
  ['mono24_3s_loop_6k', 'mono', 24000, 1, 3, 6, 12, ['-b', '6', '--loop']],
  // A stream without its length in the header, on short pages.
  ['mono24_6s_streamed_6k', 'mono', 24000, 1, 6, 6, 13, ['-b', '6', '--no-length', '--page-length', '30']],
  // The listening clips at music12, when PICOTONE_LISTEN_DIR points at them.
  ...(process.env.PICOTONE_LISTEN_DIR
    ? ['orchestral', 'ambient-electronic']
        .map((clip) => [clip, join(process.env.PICOTONE_LISTEN_DIR, `${clip}-0-original.wav`)])
        .filter(([, src]) => existsSync(src))
        .map(([clip, src]) => [`stereo48_${clip}_music12`, 'stereo', 48000, 2, 120, 0, 0, ['--preset', 'music12'], src])
    : []),
];

const cases = [];
for (const [name, model, rate, channels, seconds, bw, seed, extra, source] of CASES) {
  const wav = join(dir, `${name}.in.wav`);
  const pico = join(dir, `${name}.pico`);
  const ref = join(dir, `${name}.ref.wav`);
  const encodeArgs = extra ?? ['-b', String(bw)];
  const stamp = join(dir, `${name}.args`);
  const stale = !existsSync(stamp) || readFileSync(stamp, 'utf8') !== encodeArgs.join(' ');
  if (force || stale || !existsSync(ref) || !existsSync(pico)) {
    if (source) copyFileSync(source, wav);
    else writeWav16(wav, rate, synth(rate, seconds, channels, seed));
    execFileSync(exe, ['--weights', weights, 'encode', wav, pico, '--model', model === 'mono' ? 'encodec-24k' : 'encodec-48k', ...encodeArgs], { stdio: 'inherit' });
    writeFileSync(stamp, encodeArgs.join(' '));
    execFileSync(exe, ['--weights', weights, 'decode', pico, ref, '--float'], { stdio: 'inherit' });
    console.log(`made ${name}`);
  }
  cases.push({ name, pico: `/test-data/${name}.pico`, ref: `/test-data/${name}.ref.wav` });
}
writeFileSync(join(dir, 'cases.json'), JSON.stringify(cases, null, 1));
console.log(`${cases.length} reference cases in ${dir}`);
