// Demo page logic. Uses the compiled API in ./picotone/ (see ts/picotone.ts).
import { PicotoneDecoder, readPicoHeader, encodeWav } from "./picotone/picotone.js";

const $ = (id) => document.getElementById(id);
const log = (text) => {
  $("log").textContent += `${text}\n`;
};

const params = new URLSearchParams(location.search);
const decoder = await PicotoneDecoder.create({
  baseUrl: "./picotone/",
  modelsUrl: "./models/",
  build: params.get("build") ?? "auto",
});
const { build } = decoder;
$("build").textContent = `${build.kind === "threaded" ? `threaded (${build.threads} threads)` : "single-threaded"} — ${build.reason}`;
if (build.kind !== "threaded") $("build").classList.add("warn");
$("context").textContent = `isSecureContext=${self.isSecureContext}, crossOriginIsolated=${self.crossOriginIsolated}`;

let bytes = null;
let name = "";
let decoded = null;
let playback = null;
let context = null;
const loaded = new Map();

async function ensureModel(model) {
  if (!loaded.has(model)) {
    $("m-load").textContent = "fetching…";
    const progress = $("progress");
    progress.hidden = false;
    loaded.set(
      model,
      decoder
        .loadModel(model, {
          onProgress: (done, total) => {
            if (total > 0) progress.value = done / total;
          },
        })
        .then((info) => {
          $("m-load").textContent =
            `${(info.loadMs / 1000).toFixed(2)} s ` +
            (info.fromCache ? "(from Cache Storage)" : `(${(info.fetchedBytes / 1e6).toFixed(1)} MB fetched)`);
          return info;
        }),
    );
  }
  return loaded.get(model);
}

async function choose(file) {
  bytes = new Uint8Array(await file.arrayBuffer());
  name = file.name;
  decoded = null;
  $("wav").disabled = true;
  try {
    const info = readPicoHeader(bytes);
    $("meta").hidden = false;
    $("m-file").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KiB)`;
    $("m-model").textContent = `${info.model} — ${info.sampleRate} Hz, ${info.channels === 1 ? "mono" : "stereo"}`;
    $("m-duration").textContent = `${info.durationSeconds.toFixed(2)} s (${info.audioLength} samples)`;
    $("m-bw").textContent = `${info.bandwidthKbps} kbps (${info.numCodebooks} codebooks)${info.useLm ? " — LM coded, unsupported" : ""}`;
    $("m-stats").textContent = "";
    $("play").disabled = info.useLm;
    void ensureModel(info.model).catch((e) => log(`model load failed: ${e.message}`));
  } catch (e) {
    log(`${file.name}: ${e.message}`);
    $("play").disabled = true;
  }
}

async function play() {
  const info = readPicoHeader(bytes);
  await ensureModel(info.model);
  playback?.stop();
  context ??= new AudioContext({ sampleRate: 48000 });
  await context.resume();
  $("progress").value = 0;
  $("stop").disabled = false;
  $("play").disabled = true;
  playback = decoder.play(context, bytes, {
    loop: $("loop").checked,
    onProgress: (p) => {
      $("progress").value = p.samplesDone / p.audioLength;
    },
    onStart: (ms) => log(`first audio after ${ms.toFixed(0)} ms`),
    onUnderrun: (gap) => log(`underrun: paused ${(gap * 1000).toFixed(0)} ms (decoding slower than realtime)`),
  });
  try {
    decoded = await playback.decoded;
    const s = decoded.stats;
    $("m-stats").textContent =
      `${(s.decodeMs / 1000).toFixed(2)} s, ${s.speed.toFixed(2)}× realtime, first chunk ${s.firstChunkMs.toFixed(0)} ms, ` +
      `wasm memory ${(s.memoryBytes / 2 ** 20).toFixed(0)} MiB`;
    $("wav").disabled = false;
  } catch (e) {
    log(`decode: ${e.message}`);
  } finally {
    $("play").disabled = false;
  }
}

$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", () => {
  const file = $("file").files?.[0];
  if (file) void choose(file);
});
$("drop").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("drop").classList.add("over");
});
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  const file = e.dataTransfer?.files?.[0];
  if (file) void choose(file);
});
$("play").addEventListener("click", () => void play());
$("stop").addEventListener("click", () => {
  playback?.stop();
  $("stop").disabled = true;
});
$("wav").addEventListener("click", () => {
  if (!decoded) return;
  const blob = new Blob([encodeWav(decoded)], { type: "audio/wav" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name.replace(/\.pico$/i, "") + ".wav";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
});
