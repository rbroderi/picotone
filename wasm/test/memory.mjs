// Peak WebAssembly memory of each decoder network in each build, by phase.
//
//   node test/memory.mjs [--only name,name] [--builds mt,st-fma,st] [--threads n]
//
// Each measurement runs in a fresh decoder worker (a fresh wasm instance), so
// nothing else is in its memory. wasm memory never shrinks, so each figure is
// the peak up to that phase:
//   start     before the network is built (the empty instance, plus the thread pool)
//   ...       the network's own load phases (wasm/src/lib.rs, Decoder.loadMemory)
//   built     after the network is built
//   loaded    after the cx2 tables are added too
//   first     after the first decoded chunk
//   peak      after the whole decode (the steady state: later decodes stay here)
//
// It decodes files test/run.mjs made (PICOTONE_TEST_DIR, default <tmp>/picotone-wasm-test);
// run that once first. Env: CHROME as for run.mjs.

import { spawn } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..");
const dir = process.env.PICOTONE_TEST_DIR ?? join(tmpdir(), "picotone-wasm-test");
const chrome = process.env.CHROME ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const option = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
};
const only = option("--only", null)?.split(",") ?? null;
const threads = Number(option("--threads", "0"));

const CASES = [
  { name: "student_stereo_q9_30s", file: "hq_stereo_q9_30s" },
  { name: "vocos_stereo_q9_30s", file: "hq_stereo_q9_30s", hqDecoder: "dac44_vocos" },
  { name: "student_mono_q9_30s", file: "hq_mono_q9_30s" },
  { name: "vocos_mono_q9_30s", file: "hq_mono_q9_30s", hqDecoder: "dac44_vocos" },
  { name: "encodec48_stereo_8s", file: "stereo_8s" },
  { name: "encodec24_mono_6s", file: "mono_6s" },
].filter((c) => only === null || only.includes(c.name));
for (const c of CASES) {
  if (!existsSync(join(dir, `${c.file}.pico`))) throw new Error(`${c.file}.pico is missing in ${dir}: run test/run.mjs first`);
}

const builds = option("--builds", null)?.split(",") ?? null;
const VARIANTS = [
  { label: "mt", serverArgs: [], build: "auto", relaxed: true },
  { label: "st-fma", serverArgs: ["--no-isolation"], build: "single", relaxed: true },
  { label: "st", serverArgs: ["--no-isolation"], build: "single", relaxed: false },
].filter((v) => builds === null || builds.includes(v.label));

function startServer(extraArgs) {
  return new Promise((resolveServer, reject) => {
    const child = spawn(process.execPath, [join(wasmDir, "serve.mjs"), "--port", "0", "--extra", dir, ...extraArgs], {
      stdio: ["ignore", "pipe", "inherit"],
    });
    child.stdout.on("data", (data) => {
      const m = /http:\/\/[^ ]+:(\d+)\//.exec(String(data));
      if (m) resolveServer({ child, url: `http://127.0.0.1:${m[1]}/` });
    });
    child.on("exit", (code) => reject(new Error(`server exited ${code}`)));
  });
}

const MiB = (bytes) => (bytes / 2 ** 20).toFixed(0).padStart(4);
const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--no-first-run", "--mute-audio"],
  protocolTimeout: 600_000,
});
const results = [];
try {
  for (const variant of VARIANTS) {
    const server = await startServer(variant.serverArgs);
    try {
      const page = await browser.newPage();
      page.on("pageerror", (e) => console.log(`  [page error] ${e.message}`));
      await page.goto(`${server.url}test.html`);
      await page.waitForFunction("window.picotoneReady === true", { timeout: 30_000 });
      console.log(`\n== ${variant.label}`);
      for (const c of CASES) {
        const r = await page.evaluate(
          (file, options) => window.picotoneMemory(file, options),
          c.file,
          { build: variant.build, threads, relaxedSimd: variant.relaxed, hqDecoder: c.hqDecoder },
        );
        const phases = { ...r.model.loadMemory, loaded: r.model.memoryBytes, first: r.stats.firstChunkMemoryBytes, peak: r.stats.memoryBytes };
        results.push({ variant: variant.label, case: c.name, threads: r.build.threads, speed: r.stats.speed, phases });
        console.log(
          `   ${c.name.padEnd(22)} ${r.build.kind} ${String(r.build.threads).padStart(2)} thr  ` +
            Object.entries(phases).map(([k, v]) => `${k} ${MiB(v)}`).join("  ") +
            `  MiB  (${r.stats.speed.toFixed(2)}x)`,
        );
      }
      await page.close();
    } finally {
      server.child.kill();
    }
  }
} finally {
  await browser.close();
}
const out = join(dir, "memory.json");
writeFileSync(out, JSON.stringify(results, null, 2));
console.log(`\nresults: ${out}`);
