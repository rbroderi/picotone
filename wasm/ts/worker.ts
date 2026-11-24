/**
 * Dedicated decoder worker. Loads the threaded WebAssembly build when the page
 * is cross-origin isolated (its rayon pool is a set of nested workers) and the
 * single-threaded SIMD build otherwise. Never runs on the main thread.
 *
 * Started by {@link PicotoneDecoder} as a module worker: `new Worker(url, { type: "module" })`.
 */

import type {
  BuildInfo,
  BuildPreference,
  DecodeRequest,
  InitRequest,
  LoadRequest,
  ModelName,
  Request,
  Response,
} from "./protocol.js";

// Minimal typings of the wasm-bindgen output in pkg-st/, pkg-st-fma/ and pkg-mt/.
interface WasmChunk {
  readonly offset: number;
  readonly length: number;
  readonly channels: number;
  channel(index: number): Float32Array<ArrayBuffer>;
  free(): void;
}

interface WasmStream {
  readonly audioLength: number;
  readonly segmentCount: number;
  readonly segmentsDone: number;
  readonly cx2Ms: number;
  readonly networkMs: number;
  decodeNext(decoder: WasmDecoder, maxSegments: number): WasmChunk | undefined;
  free(): void;
}

interface WasmTables {
  add(table: Uint8Array): string;
  has(hash: string): boolean;
}

interface WasmDecoder {
  readonly name: string;
  readonly sampleRate: number;
  readonly channels: number;
  openAt(pico: Uint8Array, tables: WasmTables, startSample: number): WasmStream;
  free(): void;
}

interface WasmPkg {
  default(init: { module_or_path: string }): Promise<{ memory: WebAssembly.Memory }>;
  initThreadPool?: (threads: number) => Promise<unknown>;
  isThreadedBuild(): boolean;
  threadCount(): number;
  readHeader(pico: Uint8Array): string;
  Decoder: new (name: string, configJson: string, safetensors: Uint8Array) => WasmDecoder;
  Tables: new () => WasmTables;
}

/** `cx2-tables.json`: the tables the exporter copied, per model. */
interface TableManifest {
  readonly tables: readonly { readonly file: string; readonly model: string; readonly hash: string }[];
}

// The DOM lib types `self` as a Window; describe the worker scope we use.
interface WorkerScope {
  readonly crossOriginIsolated?: boolean;
  readonly isSecureContext?: boolean;
  readonly caches?: CacheStorage;
  readonly navigator: { readonly hardwareConcurrency?: number };
  postMessage(message: Response, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<Request>) => void): void;
}

const scope = self as unknown as WorkerScope;

interface Runtime {
  readonly pkg: WasmPkg;
  readonly memory: WebAssembly.Memory;
  readonly build: BuildInfo;
}

let runtime: Promise<Runtime> | null = null;
const models = new Map<string, { decoder: WasmDecoder; tables: string[] }>();
let tables: WasmTables | null = null;
const cancelled = new Set<number>();

function post(message: Response, transfer: Transferable[] = []): void {
  scope.postMessage(message, transfer);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Why threads cannot be used here, or null when they can. */
function threadBlocker(prefer: BuildPreference): string | null {
  if (prefer === "single") {
    return "the single-threaded build was requested";
  }
  if (scope.isSecureContext !== true) {
    return (
      "this page is not a secure context (plain http on an address other than localhost), " +
      "so browsers ignore COOP/COEP and threads are unavailable; serve it over HTTPS " +
      "(see INTEGRATION-gaim.md) or open it via localhost"
    );
  }
  if (scope.crossOriginIsolated !== true) {
    return (
      "this page is not cross-origin isolated; serve it with " +
      "Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp"
    );
  }
  if (typeof SharedArrayBuffer === "undefined") {
    return "SharedArrayBuffer is unavailable in this browser";
  }
  return null;
}

/**
 * A module whose only function calls `f32x4.relaxed_madd`: it validates
 * exactly where the browser has relaxed SIMD.
 */
function relaxedSimdSupported(): boolean {
  const zero = new Array<number>(16).fill(0);
  const body = [
    0x00, // no locals
    ...[0, 1, 2].flatMap(() => [0xfd, 0x0c, ...zero]), // three times v128.const 0
    0xfd, 0x85, 0x02, // f32x4.relaxed_madd
    0x1a, // drop
    0x0b, // end
  ];
  const module = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic, version
    0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type section: () -> ()
    0x03, 0x02, 0x01, 0x00, // function section
    0x0a, body.length + 2, 0x01, body.length, ...body, // code section
  ]);
  try {
    return WebAssembly.validate(module);
  } catch {
    return false;
  }
}

async function loadPkg(baseUrl: string, dir: "pkg-mt" | "pkg-st" | "pkg-st-fma"): Promise<{
  pkg: WasmPkg;
  memory: WebAssembly.Memory;
}> {
  const pkg = (await import(new URL(`${dir}/picotone.js`, baseUrl).href)) as WasmPkg;
  const { memory } = await pkg.default({
    module_or_path: new URL(`${dir}/picotone_bg.wasm`, baseUrl).href,
  });
  return { pkg, memory };
}

async function start(request: InitRequest): Promise<Runtime> {
  const isolated = scope.crossOriginIsolated === true;
  const secure = scope.isSecureContext === true;
  let reason = threadBlocker(request.prefer);
  if (reason === null) {
    try {
      const { pkg, memory } = await loadPkg(request.baseUrl, "pkg-mt");
      const threads =
        request.threads > 0 ? request.threads : (scope.navigator.hardwareConcurrency ?? 4);
      if (pkg.initThreadPool === undefined) {
        throw new Error("pkg-mt has no initThreadPool");
      }
      await pkg.initThreadPool(threads);
      return {
        pkg,
        memory,
        build: {
          kind: "threaded",
          relaxedSimd: false,
          threads: pkg.threadCount(),
          crossOriginIsolated: isolated,
          isSecureContext: secure,
          reason: `cross-origin isolated: running ${pkg.threadCount()} rayon threads`,
        },
      };
    } catch (error) {
      reason = `the threaded build failed to start (${errorText(error)})`;
    }
  }
  let simd = "plain SIMD";
  if (!request.relaxedSimd) {
    simd = "plain SIMD was requested";
  } else if (!relaxedSimdSupported()) {
    simd = "this browser has no relaxed SIMD";
  } else {
    try {
      const { pkg, memory } = await loadPkg(request.baseUrl, "pkg-st-fma");
      return {
        pkg,
        memory,
        build: {
          kind: "single",
          relaxedSimd: true,
          threads: pkg.threadCount(),
          crossOriginIsolated: isolated,
          isSecureContext: secure,
          reason: `single-threaded, relaxed SIMD: ${reason}`,
        },
      };
    } catch (error) {
      simd = `the relaxed SIMD build failed to start (${errorText(error)})`;
    }
  }
  const { pkg, memory } = await loadPkg(request.baseUrl, "pkg-st");
  return {
    pkg,
    memory,
    build: {
      kind: "single",
      relaxedSimd: false,
      threads: pkg.threadCount(),
      crossOriginIsolated: isolated,
      isSecureContext: secure,
      reason: `single-threaded, ${simd}: ${reason}`,
    },
  };
}

async function ready(): Promise<Runtime> {
  if (runtime === null) {
    throw new Error("the worker was not initialised");
  }
  return runtime;
}

/** Fetches `url`, reporting progress, through Cache Storage when available. */
async function fetchBytes(
  url: string,
  cache: Cache | null,
  onProgress: (loaded: number, total: number) => void,
): Promise<{ bytes: Uint8Array; fetched: number; fromCache: boolean }> {
  const cached = cache === null ? undefined : await cache.match(url);
  const fromCache = cached !== undefined;
  const response = cached ?? (await fetch(url));
  if (!response.ok) {
    throw new Error(`fetching ${url} failed: HTTP ${response.status}`);
  }
  if (!fromCache && cache !== null) {
    await cache.put(url, response.clone());
  }
  const total = Number(response.headers.get("content-length") ?? 0);
  const body = response.body;
  if (body === null) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress(bytes.length, bytes.length);
    return { bytes, fetched: fromCache ? 0 : bytes.length, fromCache };
  }
  const reader = body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    parts.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const part of parts) {
    bytes.set(part, at);
    at += part.length;
  }
  return { bytes, fetched: fromCache ? 0 : loaded, fromCache };
}

async function openCache(name: string | null): Promise<Cache | null> {
  // Cache Storage exists only in secure contexts.
  if (name === null || scope.caches === undefined) {
    return null;
  }
  try {
    return await scope.caches.open(name);
  } catch {
    return null;
  }
}

async function load(request: LoadRequest): Promise<void> {
  const { pkg } = await ready();
  const t0 = performance.now();
  const existing = models.get(request.model);
  if (existing !== undefined) {
    // Weights stay in wasm memory for the worker's lifetime; reuse them.
    post({
      type: "loaded",
      id: request.id,
      model: {
        name: request.model,
        sampleRate: existing.decoder.sampleRate,
        channels: existing.decoder.channels,
        fetchedBytes: 0,
        fromCache: true,
        tables: existing.tables,
        loadMs: performance.now() - t0,
      },
    });
    return;
  }
  const cache = await openCache(request.cacheName);
  const base = request.modelsUrl;
  // HQ files (`dac44` codes) decode with the student network; another
  // decoder of the same codes would be another network name here.
  const network = request.model === "dac44" ? "dac44_student" : request.model;
  const config = await fetchBytes(new URL(`${network}.json`, base).href, cache, () => {});
  const weights = await fetchBytes(
    new URL(`${network}.safetensors`, base).href,
    cache,
    (loaded, total) => post({ type: "loadProgress", id: request.id, loaded, total }),
  );
  const configJson = new TextDecoder().decode(config.bytes);
  const decoder = new pkg.Decoder(network, configJson, weights.bytes);
  // The cx2 tables of this model; a file finds its table by hash.
  const manifest = await fetchBytes(new URL("cx2-tables.json", base).href, cache, () => {});
  const listed = (JSON.parse(new TextDecoder().decode(manifest.bytes)) as TableManifest).tables.filter(
    (t) => t.model === request.model,
  );
  if (listed.length === 0) {
    throw new Error(`cx2-tables.json lists no table for ${request.model}`);
  }
  tables ??= new pkg.Tables();
  const loadedTables: string[] = [];
  let tableBytes = 0;
  for (const entry of listed) {
    const table = await fetchBytes(new URL(entry.file, base).href, cache, () => {});
    const hash = tables.add(table.bytes);
    if (hash !== entry.hash) {
      throw new Error(`${entry.file} has hash ${hash}, but cx2-tables.json says ${entry.hash}`);
    }
    loadedTables.push(hash);
    tableBytes += table.fetched;
  }
  models.set(request.model, { decoder, tables: loadedTables });
  post({
    type: "loaded",
    id: request.id,
    model: {
      name: request.model,
      sampleRate: decoder.sampleRate,
      channels: decoder.channels,
      fetchedBytes: config.fetched + weights.fetched + manifest.fetched + tableBytes,
      fromCache: config.fromCache && weights.fromCache,
      tables: loadedTables,
      loadMs: performance.now() - t0,
    },
  });
}

function yieldToEvents(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function decode(request: DecodeRequest): Promise<void> {
  const { pkg, memory } = await ready();
  const bytes = new Uint8Array(request.bytes);
  const header = JSON.parse(pkg.readHeader(bytes)) as {
    model: ModelName;
    tableHash: string;
    sampleRate: number;
  };
  const decoder = models.get(header.model)?.decoder;
  if (decoder === undefined || tables === null) {
    throw new Error(`model ${header.model} is not loaded; call loadModel("${header.model}") first`);
  }
  if (!tables.has(header.tableHash)) {
    throw new Error(
      `this file was coded with cx2 table ${header.tableHash}, which cx2-tables.json does not list; ` +
        "export that table next to the models (wasm/tools/export-models.mjs)",
    );
  }
  const t0 = performance.now();
  const stream = decoder.openAt(bytes, tables, request.startSample);
  let firstChunkMs = -1;
  let maxSegments = Math.max(1, request.firstChunkSegments);
  try {
    for (;;) {
      if (cancelled.delete(request.id)) {
        throw new Error("decode cancelled");
      }
      const chunk = stream.decodeNext(decoder, maxSegments);
      if (chunk === undefined) {
        break;
      }
      const channels: Float32Array<ArrayBuffer>[] = [];
      for (let c = 0; c < chunk.channels; c += 1) {
        channels.push(chunk.channel(c));
      }
      const elapsedMs = performance.now() - t0;
      if (firstChunkMs < 0) {
        firstChunkMs = elapsedMs;
      }
      post(
        {
          type: "chunk",
          id: request.id,
          sampleRate: header.sampleRate,
          offset: chunk.offset,
          length: chunk.length,
          audioLength: stream.audioLength,
          channels,
          segmentsDone: stream.segmentsDone,
          segmentCount: stream.segmentCount,
          elapsedMs,
        },
        channels.map((channel) => channel.buffer),
      );
      chunk.free();
      maxSegments = Math.min(Math.max(1, request.chunkSegments), maxSegments * 2);
      // Let cancel requests and other jobs in between chunks.
      await yieldToEvents();
    }
    const decodeMs = performance.now() - t0;
    const audioSeconds = (stream.audioLength - request.startSample) / header.sampleRate;
    post({
      type: "done",
      id: request.id,
      stats: {
        decodeMs,
        firstChunkMs,
        audioSeconds,
        speed: audioSeconds / (decodeMs / 1000),
        cx2Ms: stream.cx2Ms,
        networkMs: stream.networkMs,
        memoryBytes: memory.buffer.byteLength,
      },
    });
  } finally {
    // After a trap the stream stays "borrowed" and free() throws; keep the
    // original error rather than masking it.
    try {
      stream.free();
    } catch {
      // The instance may be unusable now; nothing more to release.
    }
  }
}

async function handle(request: Request): Promise<void> {
  switch (request.type) {
    case "init": {
      runtime ??= start(request);
      const { build } = await runtime;
      post({ type: "ready", id: request.id, build });
      return;
    }
    case "load":
      return load(request);
    case "decode":
      return decode(request);
    case "cancel":
      cancelled.add(request.id);
      return;
  }
}

scope.addEventListener("message", (event) => {
  const request = event.data;
  handle(request).catch((error: unknown) => {
    post({ type: "error", id: request.id, message: errorText(error) });
  });
});
