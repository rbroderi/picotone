// End-to-end test: synthetic audio -> native encode -> native float reference
// decode -> browser decode with every WebAssembly build in headless Chrome ->
// parity (SNR) and speed report.
//
//   node test/run.mjs [--quick] [--keep-refs] [--only name,name]
//
// Env: PICOTONE_TEST_DIR (scratch; default <tmp>/picotone-wasm-test),
//      PICOTONE_EXE (default ../target/release/picotone.exe),
//      PICOTONE_TEST_THREADS (threaded build pool size; default 0 = hardwareConcurrency),
//      CHROME (default C:\Program Files\Google\Chrome\Application\chrome.exe).
// Exit code 1 when any case falls below 100 dB SNR or fails.

import { spawn, spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..");
const repo = resolve(wasmDir, "..");
const quick = process.argv.includes("--quick");
const onlyAt = process.argv.indexOf("--only");
const only = onlyAt >= 0 ? (process.argv[onlyAt + 1] ?? "").split(",") : null;
const dir = process.env.PICOTONE_TEST_DIR ?? join(tmpdir(), "picotone-wasm-test");
const exeSrc = process.env.PICOTONE_EXE ?? join(repo, "target", "release", "picotone.exe");
const chrome = process.env.CHROME ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const MIN_SNR_DB = 100;
const THREADS = Number(process.env.PICOTONE_TEST_THREADS ?? 0);

mkdirSync(dir, { recursive: true });

// ---------- synthetic audio ----------

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function writeWav16(path, rate, channels) {
  const n = channels[0].length;
  const c = channels.length;
  const buf = Buffer.alloc(44 + n * c * 2);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + n * c * 2, 4);
  buf.write("WAVEfmt ", 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(c, 22);
  buf.writeUInt32LE(rate, 24);
  buf.writeUInt32LE(rate * c * 2, 28);
  buf.writeUInt16LE(c * 2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(n * c * 2, 40);
  let at = 44;
  for (let t = 0; t < n; t++) {
    for (let ch = 0; ch < c; ch++) {
      const v = Math.max(-1, Math.min(1, channels[ch][t]));
      buf.writeInt16LE(Math.round(v * 32767), at);
      at += 2;
    }
  }
  writeFileSync(path, buf);
}

/** Music-like stereo: chord stabs with harmonics, kick and hi-hat, panned. */
function music(rate, seconds, seed) {
  const n = Math.round(rate * seconds);
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  const rand = rng(seed);
  const roots = [110, 146.83, 164.81, 130.81];
  const beat = 0.5;
  for (let b = 0; b * beat < seconds; b++) {
    const t0 = Math.round(b * beat * rate);
    const root = roots[Math.floor(b / 4) % roots.length];
    const chord = [1, 1.25, 1.5, 2].map((r) => root * r);
    const pan = 0.5 + 0.4 * Math.sin(b * 0.7);
    const len = Math.min(n - t0, Math.round(rate * 0.9));
    for (let i = 0; i < len; i++) {
      const t = i / rate;
      const env = Math.exp(-3 * t) * Math.min(1, t * 200);
      let v = 0;
      for (const f of chord) for (let h = 1; h <= 4; h++) v += Math.sin(2 * Math.PI * f * h * t) / (h * h);
      v *= 0.06 * env;
      L[t0 + i] += v * (1 - pan);
      R[t0 + i] += v * pan;
    }
    // kick on every other beat, hat on every beat
    if (b % 2 === 0) {
      for (let i = 0; i < Math.min(n - t0, rate * 0.25); i++) {
        const t = i / rate;
        const v = 0.35 * Math.sin(2 * Math.PI * (50 + 100 * Math.exp(-30 * t)) * t) * Math.exp(-12 * t);
        L[t0 + i] += v;
        R[t0 + i] += v;
      }
    }
    const h0 = t0 + Math.round(rate * beat * 0.5);
    let prev = 0;
    for (let i = 0; i < Math.min(n - h0, rate * 0.06); i++) {
      const white = rand() * 2 - 1;
      const v = (white - prev) * 0.08 * Math.exp((-60 * i) / rate);
      prev = white;
      L[h0 + i] += v * 0.7;
      R[h0 + i] += v;
    }
  }
  return [L, R];
}

/** Mono effect-like signal: sweep, tones and a noise burst. */
function effect(rate, seconds, seed) {
  const n = Math.round(rate * seconds);
  const x = new Float32Array(n);
  const rand = rng(seed);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const f = 200 + 1800 * (t / seconds);
    phase += (2 * Math.PI * f) / rate;
    x[i] = 0.3 * Math.sin(phase) * Math.min(1, t * 50) + 0.1 * Math.sin(2 * Math.PI * 440 * t);
    if (t > seconds * 0.6 && t < seconds * 0.7) x[i] += 0.2 * (rand() * 2 - 1);
  }
  return [x];
}

/** A seamless stereo loop: every component has a whole number of cycles. */
function loop(rate, seconds) {
  const n = Math.round(rate * seconds);
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const am = 0.6 + 0.4 * Math.sin((2 * Math.PI * 2 * t) / seconds);
    L[i] = am * (0.25 * Math.sin(2 * Math.PI * 110 * t) + 0.1 * Math.sin(2 * Math.PI * 330 * t));
    R[i] = am * (0.25 * Math.sin(2 * Math.PI * 165 * t) + 0.1 * Math.sin(2 * Math.PI * 440 * t));
  }
  return [L, R];
}

// The listening clips (120 s, 48 kHz stereo) from PICOTONE_LISTEN_DIR, when set (full runs only).
const listenDir = process.env.PICOTONE_LISTEN_DIR ?? null;
const listenCases =
  listenDir === null
    ? []
    : ["orchestral", "ambient-electronic"]
        .map((clip) => ({ clip, src: join(listenDir, `${clip}-0-original.wav`) }))
        .filter(({ src }) => existsSync(src))
        .map(({ clip, src }) => ({ name: `${clip}_music12`, mode: "stereo", rate: 48000, src, args: ["--preset", "music12"] }));

const B6 = ["-b", "6"];
const SEEK_CASES = [
  "mono_6s",
  "stereo_8s_music24",
  "stereo_8s_music6",
  "stereo_8s_edge",
  "stereo_music_6s_gapless",
  "hq_lr_q5_4s",
  "hq_ms_q9_streamed",
  "hq_ps_q3_4s",
];
const HQ = "hq";
const CASES = [
  { name: "mono_sfx_1s", mode: "mono", rate: 24000, make: () => effect(24000, 1.2, 7) },
  { name: "mono_6s_streamed", mode: "mono", rate: 24000, make: () => effect(24000, 6, 11), args: [...["-b", "6"], "--no-length", "--page-length", "30"] },
  { name: "mono_6s", mode: "mono", rate: 24000, make: () => effect(24000, 6, 3) },
  { name: "stereo_8s", mode: "stereo", rate: 48000, make: () => music(48000, 8, 1) },
  // Configurable stride and cross-fade.
  { name: "stereo_8s_music24", mode: "stereo", rate: 48000, make: () => music(48000, 8, 1), args: ["--preset", "music24", "--beam", "1"] },
  { name: "stereo_8s_music12", mode: "stereo", rate: 48000, make: () => music(48000, 8, 5), args: ["--preset", "music12", "--beam", "1"] },
  { name: "stereo_8s_music6", mode: "stereo", rate: 48000, make: () => music(48000, 8, 6), args: ["--preset", "music6", "--beam", "1"] },
  { name: "stereo_8s_edge", mode: "stereo", rate: 48000, make: () => music(48000, 8, 9), args: ["--codebooks", "4", "--stride", "46560", "--crossfade", "edge"] },
  // A plain whole-file loop clicks a little at the seam; --loop (pre-skip, end trim, loop flag) must not.
  // The music presets write 1, 2 or 4 segments per page; mono_6s_streamed has no length in its header.
  { name: "stereo_loop_4s", mode: "stereo", rate: 48000, make: () => loop(48000, 4), seam: true },
  { name: "stereo_loop_4s_gapless", mode: "stereo", rate: 48000, make: () => loop(48000, 4), args: [...B6, "--loop"], seam: true, maxSeamRatio: 3 },
  { name: "stereo_music_6s_gapless", mode: "stereo", rate: 48000, make: () => music(48000, 6, 8), args: ["--preset", "music6", "--beam", "1", "--loop"], seam: true, maxSeamRatio: 3 },
  { name: "mono_loop_3s_gapless", mode: "mono", rate: 24000, make: () => [loop(24000, 3)[0]], args: [...B6, "--loop"], seam: true, maxSeamRatio: 3 },
  // HQ (DAC 44.1 kHz codes, the Vocos decoder): every layout, several codebook counts.
  { name: "hq_mono_q9_3s", model: HQ, rate: 44100, make: () => effect(44100, 3, 21), args: ["--codebooks", "9", "--channel-format", "mono"] },
  { name: "hq_lr_q5_4s", model: HQ, rate: 44100, make: () => music(44100, 4, 31), args: ["--codebooks", "5", "--stereo", "lr"] },
  {
    name: "hq_ms_q9_streamed",
    model: HQ,
    rate: 44100,
    make: () => music(44100, 4, 32),
    args: ["--codebooks", "9", "--side-codebooks", "4", "--stereo", "ms", "--no-length", "--page-length", "30"],
  },
  { name: "hq_ps_q3_4s", model: HQ, rate: 44100, make: () => music(44100, 4, 33), args: ["--codebooks", "3", "--stereo", "parametric"] },
  { name: "hq_ps10_q1_3s", model: HQ, rate: 44100, make: () => music(44100, 3, 34), args: ["--codebooks", "1", "--stereo", "parametric", "--ps-bands", "10", "--ps-frames", "4"] },
  { name: "hq_loop_q6_3s", model: HQ, rate: 44100, make: () => loop(44100, 3), args: ["--codebooks", "6", "--stereo", "lr", "--loop"], seam: true },
  ...(quick
    ? []
    : [
        // Speed: 30 s of stereo at q5 and q9.
        { name: "hq_stereo_q5_30s", model: HQ, rate: 44100, make: () => music(44100, 30, 35), args: ["--codebooks", "5", "--stereo", "lr"] },
        { name: "hq_stereo_q9_30s", model: HQ, rate: 44100, make: () => music(44100, 30, 35), args: ["--codebooks", "9", "--stereo", "lr"] },
        // One coded channel of the above, for the one-worker-per-channel measurement.
        { name: "hq_mono_q9_30s", model: HQ, rate: 44100, make: () => [music(44100, 30, 35)[0]], args: ["--codebooks", "9", "--channel-format", "mono"] },
        { name: "mono_music_75s", mode: "mono", rate: 24000, make: () => [music(24000, 75, 4)[0]] },
        { name: "stereo_music_75s", mode: "stereo", rate: 48000, make: () => music(48000, 75, 2) },
        ...listenCases,
      ]),
].filter((c) => only === null || only.includes(c.name));

// ---------- native references ----------

// Run a private copy of the exe so a rebuild of target/ is never blocked by a lock.
const exe = join(dir, "picotone.exe");
if (!existsSync(exe) || statSync(exeSrc).mtimeMs > statSync(exe).mtimeMs) copyFileSync(exeSrc, exe);
const weights = join(repo, "weights");
// Size and modification time of the exe and of every weights and table file, so that new weights or a new
// build regenerate the native references (they were once kept from a random-init stand-in decoder).
const inputsStamp = [exeSrc, ...[weights, join(repo, "models")].flatMap((d) =>
  readdirSync(d).filter((f) => /\.(safetensors|ecxm|json)$/.test(f)).sort().map((f) => join(d, f)))]
  .map((p) => `${p}:${statSync(p).size}:${Math.round(statSync(p).mtimeMs)}`)
  .join("|");

function run(args) {
  const r = spawnSync(exe, ["--weights", weights, ...args], { encoding: "utf8" });
  if (r.status !== 0) throw new Error(`picotone ${args.join(" ")} failed:\n${r.stderr}`);
  return r.stderr;
}

for (const c of CASES) {
  const wav = join(dir, `${c.name}.wav`);
  const pico = join(dir, `${c.name}.pico`);
  const ref = join(dir, `${c.name}.ref.wav`);
  const stamp = join(dir, `${c.name}.args`);
  const args = c.args ?? B6;
  // The references depend on the exe, the weights and the tables too, not only the arguments.
  const stampText = [...args, inputsStamp].join(" ");
  const stale = !existsSync(stamp) || readFileSync(stamp, "utf8") !== stampText;
  if (stale || !existsSync(ref) || !existsSync(pico) || process.argv.includes("--fresh")) {
    if (c.src) copyFileSync(c.src, wav);
    else writeWav16(wav, c.rate, c.make());
    run(["encode", wav, pico, "--model", c.model ?? (c.mode === "mono" ? "encodec-24k" : "encodec-48k"), ...args]);
    writeFileSync(stamp, stampText);
    run(["decode", pico, ref, "--float"]);
    console.log(`made ${c.name}: ${statSync(pico).size} bytes .pico`);
  }
}

// Native speed for comparison (single thread and all threads).
const nativeSpeed = {};
const NATIVE_TIMED = ["stereo_8s", "mono_6s", "hq_stereo_q5_30s", "hq_stereo_q9_30s"];
for (const c of CASES.filter((k) => NATIVE_TIMED.includes(k.name))) {
  for (const threads of [1, 0]) {
    const args = threads ? ["--threads", "1"] : [];
    const t0 = performance.now();
    const r = spawnSync(exe, ["--weights", weights, ...args, "decode", join(dir, `${c.name}.pico`), join(dir, "native-tmp.wav")]);
    if (r.status !== 0) throw new Error(`native decode failed`);
    const secs = (performance.now() - t0) / 1000;
    nativeSpeed[`${c.name}/${threads ? 1 : "all"}`] = secs;
  }
}

// ---------- browser ----------

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

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required", "--no-first-run", "--mute-audio"],
  protocolTimeout: 600_000,
});

const results = [];
let failed = false;
try {
  for (const variant of [
    { label: "threaded", serverArgs: [], build: "auto", relaxed: true, expect: "threaded" },
    { label: "single", serverArgs: ["--no-isolation"], build: "auto", relaxed: true, expect: "single" },
    { label: "single, plain SIMD", serverArgs: ["--no-isolation"], build: "auto", relaxed: false, expect: "single" },
  ]) {
    const server = await startServer(variant.serverArgs);
    try {
      const page = await browser.newPage();
      page.on("console", (msg) => {
        if (msg.type() === "error" || msg.type() === "warn") console.log(`  [page ${msg.type()}] ${msg.text()}`);
      });
      page.on("pageerror", (e) => console.log(`  [page error] ${e.message}`));
      await page.goto(`${server.url}test.html`);
      await page.waitForFunction("window.picotoneReady === true", { timeout: 30_000 });
      const env = await page.evaluate((b, relaxed, threads) => window.picotoneTest.init(b, threads, relaxed), variant.build, variant.relaxed, THREADS);
      console.log(`\n== ${variant.label}: ${env.build.kind} (${env.build.threads} threads); ${env.build.reason}`);
      console.log(`   isSecureContext=${env.isSecureContext} crossOriginIsolated=${env.crossOriginIsolated} hardwareConcurrency=${env.hardwareConcurrency}`);
      if (env.build.kind !== variant.expect) {
        failed = true;
        console.log(`   FAIL: expected the ${variant.expect} build`);
      }
      // Chrome has relaxed SIMD, so the single-threaded build must be that variant when allowed.
      if (env.build.relaxedSimd !== (variant.expect === "single" && variant.relaxed)) {
        failed = true;
        console.log(`   FAIL: relaxed SIMD is ${env.build.relaxedSimd}`);
      }
      for (const c of CASES) {
        const r = await page.evaluate(
          (name, seamCheck) => window.picotoneTest.runCase(name, { stream: true, seamCheck }),
          c.name,
          c.seam === true,
        );
        r.label = c.name;
        r.build = env.build.kind;
        r.relaxedSimd = env.build.relaxedSimd;
        r.threads = env.build.threads;
        results.push(r);
        // Infinity (bit-exact) arrives as null through page.evaluate.
        const ok = r.parity.snrDb === null || r.parity.snrDb >= MIN_SNR_DB;
        if (!ok) failed = true;
        console.log(
          `   ${c.name.padEnd(18)} ${r.seconds.toFixed(1).padStart(5)} s  ` +
            `decode ${(r.stats.decodeMs / 1000).toFixed(2).padStart(6)} s = ${r.stats.speed.toFixed(2).padStart(5)}x  ` +
            `first audio ${r.startedMs.toFixed(0).padStart(5)} ms  underruns ${r.underruns}  chunks ${r.chunks}  ` +
            `SNR ${r.parity.snrDb === null ? "inf" : r.parity.snrDb.toFixed(1)} dB  max|d| ${r.parity.maxAbsDiff.toExponential(2)}  ` +
            `exact ${(100 * r.parity.exactFraction).toFixed(1)}%  mem ${(r.stats.memoryBytes / 2 ** 20).toFixed(0)} MiB  ` +
            `cx2 ${r.stats.cx2Ms.toFixed(0)} ms  net ${r.stats.networkMs.toFixed(0)} ms  ` +
            `${ok ? "ok" : "FAIL"}`,
        );
        if (r.seam) {
          const s = r.seam.decoded;
          // A gapless loop's wrap-around step must look like any other step.
          const seamOk = c.maxSeamRatio === undefined || s.jump <= c.maxSeamRatio * s.p999Step;
          if (!seamOk) failed = true;
          console.log(
            `     loop seam (whole file): jump ${s.jump.toExponential(2)} ` +
              `vs median step ${s.medianStep.toExponential(2)}, 99.9th pct step ${s.p999Step.toExponential(2)}` +
              `${c.maxSeamRatio === undefined ? "" : seamOk ? "  ok" : "  FAIL"}`,
          );
        }
      }
      // Seeks: decoding from a later sample must give the same samples as the whole decode.
      for (const c of CASES.filter((k) => SEEK_CASES.includes(k.name))) {
        for (const fraction of [0.37, 0.81]) {
          const r = await page.evaluate(
            (name, f) =>
              fetch(`/test-data/${name}.pico`)
                .then((res) => res.arrayBuffer())
                .then((b) => window.picotoneTest.runCase(name, { stream: false, startSample: Math.floor(f * window.picotoneInfo(b)) })),
            c.name,
            fraction,
          );
          const ok = r.parity.snrDb === null || r.parity.snrDb >= MIN_SNR_DB;
          if (!ok) failed = true;
          console.log(
            `   seek ${c.name} from sample ${r.startSample}: SNR ${r.parity.snrDb === null ? "inf" : r.parity.snrDb.toFixed(1)} dB  ` +
              `exact ${(100 * r.parity.exactFraction).toFixed(1)}%  ${ok ? "ok" : "FAIL"}`,
          );
        }
      }
      // One single-threaded worker per coded channel: n workers decoding a mono file at once.
      for (const c of CASES.filter((k) => k.name === "hq_mono_q9_30s")) {
        if (quick || env.build.kind !== "single" || !env.build.relaxedSimd) break;
        for (const workers of [1, 2]) {
          const r = await page.evaluate((name, n) => window.picotoneConcurrent(name, n), c.name, workers);
          const channelSeconds = r.seconds * workers;
          console.log(
            `   ${workers} worker(s) x ${c.name} at once: ${(r.wallMs / 1000).toFixed(2)} s wall = ` +
              `${(channelSeconds / (r.wallMs / 1000)).toFixed(2)} channel-seconds/s ` +
              `(cx2 ${r.stats.map((s) => s.cx2Ms.toFixed(0)).join("/")} ms, net ${r.stats.map((s) => s.networkMs.toFixed(0)).join("/")} ms)`,
          );
          results.push({ label: `${c.name}_workers_${workers}`, concurrent: r });
        }
      }
      console.log(`   model load: ${results.filter((r) => r.build === env.build.kind).map((r) => `${r.model} ${r.modelLoadMs.toFixed(0)} ms`).filter((v, i, a) => a.indexOf(v) === i).join(", ")}`);
      await page.close();
    } finally {
      server.child.kill();
    }
  }
} finally {
  await browser.close();
}

console.log("\nnative (wall time incl. model load):");
for (const [k, v] of Object.entries(nativeSpeed)) console.log(`   ${k.padEnd(22)} ${v.toFixed(2)} s`);
writeFileSync(join(dir, "results.json"), JSON.stringify({ results, nativeSpeed }, null, 2));
console.log(`\nresults: ${join(dir, "results.json")}`);
console.log(failed ? "FAILED" : `PASSED (all cases >= ${MIN_SNR_DB} dB SNR)`);
process.exit(failed ? 1 : 0);
