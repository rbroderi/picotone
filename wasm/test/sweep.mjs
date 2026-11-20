// Tuning sweep: thread count and chunk sizes for one test file, threaded build.
//   node test/sweep.mjs [case=stereo_8s]   (run test/run.mjs first to create the files)
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const dir = process.env.PICOTONE_TEST_DIR ?? join(tmpdir(), "picotone-wasm-test");
const chrome = process.env.CHROME ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const name = process.argv[2] ?? "stereo_8s";

const server = await new Promise((ok, fail) => {
  const child = spawn(process.execPath, [resolve(here, "..", "serve.mjs"), "--port", "0", "--extra", dir], { stdio: ["ignore", "pipe", "inherit"] });
  child.stdout.on("data", (d) => {
    const m = /:(\d+)\//.exec(String(d));
    if (m) ok({ child, url: `http://127.0.0.1:${m[1]}/` });
  });
  child.on("exit", (c) => fail(new Error(`server exited ${c}`)));
});
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--autoplay-policy=no-user-gesture-required", "--mute-audio"], protocolTimeout: 600_000 });
try {
  const page = await browser.newPage();
  await page.goto(`${server.url}test.html`);
  await page.waitForFunction("window.picotoneReady === true");
  for (const threads of (process.env.THREADS ?? "16,8,4").split(",").map(Number)) {
    const env = await page.evaluate((t) => window.picotoneTest.init("threaded", t), threads);
    for (const [first, chunk, stream] of [[1, 4, false], [1, 4, true], [1, 4, false], [1, 16, false], [2, 16, false], [100, 100, false], [1, 16, true]]) {
      const r = await page.evaluate(
        (n, f, c, s) => window.picotoneTest.runCase(n, { stream: s, firstChunkSegments: f, chunkSegments: c }),
        name, first, chunk, stream,
      );
      console.log(
        `${env.build.kind} threads=${String(env.build.threads).padStart(2)} first=${String(first).padStart(3)} chunk=${String(chunk).padStart(3)}${stream ? " play()" : "       "}: ` +
          `${r.stats.speed.toFixed(2)}x, first chunk ${r.stats.firstChunkMs.toFixed(0)} ms`,
      );
    }
  }
} finally {
  await browser.close();
  server.child.kill();
}
