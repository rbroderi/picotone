// Runs the WebGPU decoder in headless Chrome against the Rust reference decodes.
//
// Usage: node tools/test-parity.mjs [--dir <test dir>] [--chrome <path>] [--headful] [--only <substring>]
// Needs the outputs of tools/make-refs.mjs and a build (npm run build).
// Exits non-zero if any f32 case is below 60 dB SNR.

import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';
import { DEFAULT_TEST_DIR as DEFAULT_DIR } from './paths.mjs';

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const dir = resolve(opt('--dir', DEFAULT_DIR));
const chrome = opt('--chrome', process.env.CHROME_PATH ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe');
const only = opt('--only', '');
const THRESHOLD_DB = 60;

let cases = JSON.parse(readFileSync(join(dir, 'cases.json'), 'utf8'));
if (only) cases = cases.filter((c) => c.name.includes(only));
const port = 8000 + Math.floor(Math.random() * 1000);
const server = await startServer({ port, dataDir: dir });
const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: !args.includes('--headful'),
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--ignore-gpu-blocklist', '--no-first-run'],
  protocolTimeout: 30 * 60 * 1000,
});
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`[page] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`[page error] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/test/parity.html`);
  await page.waitForFunction(() => typeof window.runParity === 'function', { timeout: 30000 });
  const report = await page.evaluate((c) => window.runParity(c, ['f32', 'f16']), cases);
  console.log(`adapter: ${report.adapter}`);
  for (const l of report.loads) console.log(`load ${l.model} ${l.dtype}: ${l.ms.toFixed(0)} ms, GPU ${l.gpuMB.toFixed(1)} MB`);
  const rows = report.results.map((r) => ({
    case: r.name,
    w: r.dtype,
    nq: r.codebooks,
    's': r.seconds,
    'SNR dB': r.snrDb.toFixed(1),
    'max err': r.maxAbsErr.toExponential(1),
    'cold ms': r.coldMs.toFixed(0),
    'warm ms': r.warmMs.toFixed(0),
    'x RT': r.realtime.toFixed(1),
    '1st audio cold': r.firstAudioColdMs.toFixed(0),
    '1st audio warm': r.firstAudioWarmMs.toFixed(0),
    // Smallest margin by which a later chunk beat its playback deadline (null: one chunk).
    'min lead ms': r.minLeadMs === null ? '1 chunk' : r.minLeadMs.toFixed(0),
  }));
  console.table(rows);
  writeFileSync(join(dir, 'parity-report.json'), JSON.stringify(report, null, 1));
  for (const r of report.results) {
    if (r.dtype === 'f32' && !(r.snrDb >= THRESHOLD_DB)) {
      console.error(`FAIL ${r.name}: ${r.snrDb.toFixed(1)} dB < ${THRESHOLD_DB} dB`);
      failed = true;
    }
  }
  console.log(failed ? 'parity: FAILED' : `parity: all f32 cases >= ${THRESHOLD_DB} dB`);
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
