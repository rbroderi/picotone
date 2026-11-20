/**
 * Reading `.pico` files (Ogg pages of cx2-coded EnCodec codes) for the WebGPU decoder.
 *
 * The header is parsed in TypeScript ({@link parseHeader}); the codes are
 * range-decoded by the picotone WebAssembly module (`readCodes`), the same
 * Rust code as the command line, so they are bit-exact. Only the neural
 * decoder runs on the GPU. The range coder is not ported to WGSL.
 *
 * Files to serve (the exporters copy them):
 *
 * ```text
 * <models>/cx2-tables.json        { tables: [{ file, model, hash }] }
 * <models>/<table>.ecxm           the cx2 tables it lists
 * <models>/cx2/picotone.js        wasm-bindgen glue (single-threaded build)
 * <models>/cx2/picotone_bg.wasm
 * ```
 */

import { type PicoInfo, readPicoHeader } from '../../wasm/ts/pico-header.js';
import { type ModelConfig, segments } from './config.js';

export class PicoError extends Error {
  override name = 'PicoError';
}

/** A `.pico` file's headers. */
export type PicoHeader = PicoInfo;

/** Codes and scale for one coded segment. */
export interface CodedSegment {
  /** Volume scale to apply after decoding (normalising models), else null. */
  readonly scale: number | null;
  /** Latent frames. */
  readonly frames: number;
  /** Codes stored `[frame][codebook]`. */
  readonly codes: Uint16Array;
}

/**
 * A file's codes, range-decoded one Ogg page at a time as they are asked for
 * (so the GPU can start on the first page while later ones are still coded).
 * Ask in order: segments in increasing index, frames in increasing start.
 */
export interface PicoCodes {
  readonly header: PicoHeader;
  /** Latent frames per segment (one entry for the causal 24 kHz model). */
  readonly segmentFrames: readonly number[];
  /** Segment `index` of a segmented (48 kHz) file. */
  segment(index: number): CodedSegment;
  /** Codes of latent frames `start..start + count` of a causal (24 kHz) file, `[frame][codebook]`. */
  frames(start: number, count: number): Uint16Array;
  /** Releases the WebAssembly reader (also done after the last page). */
  free(): void;
}

/** Parses the headers only (no tables or WebAssembly needed). */
export function parseHeader(bytes: Uint8Array): PicoHeader {
  try {
    return readPicoHeader(bytes);
  } catch (e) {
    throw new PicoError(e instanceof Error ? e.message : String(e));
  }
}

interface WasmTables {
  add(table: Uint8Array): string;
  has(hash: string): boolean;
}

interface WasmCodes {
  codes(): Uint16Array;
  scales(): Float32Array;
  free(): void;
}

interface WasmCodePages {
  readonly info: string;
  readonly pageCount: number;
  nextPage(): WasmCodes | undefined;
  free(): void;
}

interface WasmPkg {
  default(init: { module_or_path: string }): Promise<unknown>;
  CodePages: new (bytes: Uint8Array, tables: WasmTables) => WasmCodePages;
  Tables: new () => WasmTables;
}

interface TableManifest {
  readonly tables: readonly { readonly file: string; readonly model: string; readonly hash: string }[];
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new PicoError(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** The cx2 decoder (WebAssembly) and the tables of one models folder. */
export class Cx2Reader {
  private constructor(
    private readonly pkg: WasmPkg,
    private readonly tables: WasmTables,
    /** Hashes of the loaded tables. */
    readonly hashes: readonly string[],
  ) {}

  private static readonly cache = new Map<string, Promise<Cx2Reader>>();

  /**
   * Loads the module from `wasmUrl` (default `<modelsUrl>/cx2/`) and every
   * table `<modelsUrl>/cx2-tables.json` lists. Shared per URL pair.
   */
  static load(modelsUrl: string, wasmUrl?: string): Promise<Cx2Reader> {
    const base = new URL(modelsUrl.endsWith('/') ? modelsUrl : `${modelsUrl}/`, location.href).href;
    const wasm = new URL(wasmUrl ?? 'cx2/', base).href;
    const dir = wasm.endsWith('/') ? wasm : `${wasm}/`;
    const key = `${base} ${dir}`;
    let reader = Cx2Reader.cache.get(key);
    if (!reader) {
      reader = Cx2Reader.open(base, dir);
      reader.catch(() => Cx2Reader.cache.delete(key));
      Cx2Reader.cache.set(key, reader);
    }
    return reader;
  }

  private static async open(base: string, dir: string): Promise<Cx2Reader> {
    const pkg = (await import(/* webpackIgnore: true */ `${dir}picotone.js`)) as WasmPkg;
    await pkg.default({ module_or_path: `${dir}picotone_bg.wasm` });
    const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes(new URL('cx2-tables.json', base).href))) as TableManifest;
    const tables = new pkg.Tables();
    const hashes: string[] = [];
    for (const entry of manifest.tables) {
      const hash = tables.add(await fetchBytes(new URL(entry.file, base).href));
      if (hash !== entry.hash) throw new PicoError(`${entry.file} has hash ${hash}, but cx2-tables.json says ${entry.hash}`);
      hashes.push(hash);
    }
    return new Cx2Reader(pkg, tables, hashes);
  }

  /** Opens a file's codes; pages are range-decoded as {@link PicoCodes} asks for them. */
  open(bytes: Uint8Array, config: ModelConfig): PicoCodes {
    const header = parseHeader(bytes);
    if (!this.tables.has(header.tableHash)) {
      throw new PicoError(
        `this file was coded with cx2 table ${header.tableHash}, which cx2-tables.json does not list; ` +
          'export that table next to the models (web/tools/export-weights.mjs)',
      );
    }
    let reader: WasmCodePages | null;
    try {
      reader = new this.pkg.CodePages(bytes, this.tables);
    } catch (e) {
      throw new PicoError(e instanceof Error ? e.message : String(e));
    }
    const segmentFrames = (JSON.parse(reader.info) as { segmentFrames: number[] }).segmentFrames;
    const stride = header.stride > 0 ? header.stride : null;
    const geometry = segments(config, header.codedLength, stride);
    if (geometry.length !== segmentFrames.length || geometry.some((g, i) => g.frames !== segmentFrames[i])) {
      reader.free();
      throw new PicoError(`the file's segments do not match its length (${header.codedLength} samples)`);
    }
    const nq = header.numCodebooks;
    // Decoded but not yet used: codes, frame-major, from `first` (a segment or frame index).
    const queue: CodedSegment[] = [];
    let queued = new Uint16Array(0);
    let first = 0;
    const free = (): void => {
      reader?.free();
      reader = null;
    };
    /** Decodes the next page; false after the last. */
    const pull = (): boolean => {
      if (reader === null) return false;
      let page: WasmCodes | undefined;
      try {
        page = reader.nextPage();
      } catch (e) {
        free();
        throw new PicoError(e instanceof Error ? e.message : String(e));
      }
      if (page === undefined) {
        free();
        return false;
      }
      try {
        const codes = page.codes();
        const scales = page.scales();
        for (const code of codes) {
          if (code >= config.codebook_size) throw new PicoError(`code ${code} is out of range`);
        }
        if (stride === null) {
          const merged = new Uint16Array(queued.length + codes.length);
          merged.set(queued);
          merged.set(codes, queued.length);
          queued = merged;
        } else {
          let at = 0;
          for (let s = 0; at < codes.length; s++) {
            const frames = segmentFrames[first + queue.length] ?? 0;
            if (frames === 0) throw new PicoError('page holds more segments than the file');
            queue.push({ scale: config.normalize ? (scales[s] ?? 1) : null, frames, codes: codes.subarray(at, at + frames * nq) });
            at += frames * nq;
          }
        }
      } finally {
        page.free();
      }
      return true;
    };
    return {
      header,
      segmentFrames,
      segment(index) {
        if (index < first) throw new PicoError(`segment ${index} was already read`);
        while (first + queue.length <= index) {
          if (!pull()) throw new PicoError(`segment ${index} is missing`);
        }
        queue.splice(0, index - first);
        first = index;
        const seg = queue.shift() as CodedSegment;
        first++;
        return seg;
      },
      frames(start, count) {
        if (start < first) throw new PicoError(`frame ${start} was already read`);
        while ((first + queued.length / nq) < start + count) {
          if (!pull()) throw new PicoError(`frames ${start}..${start + count} are missing`);
        }
        const out = queued.slice((start - first) * nq, (start - first + count) * nq);
        queued = queued.subarray((start + count - first) * nq);
        first = start + count;
        return out;
      },
      free,
    };
  }
}
