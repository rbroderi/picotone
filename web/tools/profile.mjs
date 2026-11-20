// Prints per-kernel GPU time for one test file (debug aid).
// Usage: node tools/profile.mjs <case name> [--dir <test dir>]
import { resolve } from 'node:path';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';
import { DEFAULT_TEST_DIR } from './paths.mjs';

const args = process.argv.slice(2);
const i = args.indexOf('--dir');
const dir = resolve(i >= 0 ? args[i + 1] : DEFAULT_TEST_DIR);
const name = args.find((a, k) => !a.startsWith('--') && args[k - 1] !== '--dir') ?? 'stereo48_3.5s_3k';
const port = 9000 + Math.floor(Math.random() * 1000);
const server = await startServer({ port, dataDir: dir });
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-dawn-features=allow_unsafe_apis', '--ignore-gpu-blocklist'],
  protocolTimeout: 10 * 60 * 1000,
});
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log(`[page error] ${e.message}`));
  await page.goto(`http://127.0.0.1:${port}/test/parity.html`);
  await page.waitForFunction(() => typeof window.runProfile === 'function');
  const rows = await page.evaluate((u) => window.runProfile(u), `/test-data/${name}.pico`);
  const total = rows.reduce((a, r) => a + r[1], 0);
  console.log(`${name}: ${total.toFixed(1)} ms GPU total`);
  for (const [label, ms] of rows.slice(0, 40)) console.log(`${ms.toFixed(2).padStart(9)} ms ${((100 * ms) / total).toFixed(1).padStart(5)}%  ${label}`);
} finally {
  await browser.close();
  server.close();
}
