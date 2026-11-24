/**
 * The `.pico` header reader, without the worker or a model (no DOM use).
 * Shared by the WebAssembly decoder's API and the WebGPU decoder.
 */

import type { ModelName } from "./protocol.js";

/** A `.pico` file's headers plus what follows from the model. */
export interface PicoInfo {
  readonly model: ModelName;
  /** Playable samples per channel (after the pre-skip and end trim). */
  readonly audioLength: number;
  readonly numCodebooks: number;
  readonly sampleRate: number;
  readonly channels: number;
  readonly durationSeconds: number;
  /** Nominal bandwidth: codebooks times the model's 10-bit frame rate. */
  readonly bandwidthKbps: number;
  /** Actual size of the file per second of playable audio. */
  readonly bitrateKbps: number;
  /** Coded samples per channel; `audioLength` is this minus `preSkip` and `endTrim`. */
  readonly codedLength: number;
  /** Segment stride in samples; 0 for the causal 24 kHz model. */
  readonly stride: number;
  /** Cross-fade between segments: 0 triangle, 1 edge-avoiding. */
  readonly crossfade: number;
  /** Samples dropped at the start. */
  readonly preSkip: number;
  /** Samples dropped at the end. */
  readonly endTrim: number;
  /** Whether the file was encoded as a seamless loop. */
  readonly loop: boolean;
  /** Hash of the cx2 table the file was coded with, 16 hex digits. */
  readonly tableHash: string;
  /** Segments per Ogg page (24 kHz model: latent frames); 0 for a single page. */
  readonly pageLength: number;
  /** Comment tags, `KEY=value`. */
  readonly tags: readonly string[];
  /** Playback gain in dB (the stream's Q7.8 field; 0 when absent). */
  readonly outputGainDb: number;
  /**
   * HQ files (`dac44`): how the coded channels make the output, `mono`,
   * `lr` (left/right), `ms` (mid/side) or `parametric` (a mid plus stereo
   * parameters). `null` for EnCodec files.
   */
  readonly layout: HqLayout | null;
  /** Codebooks per frame of each coded channel (EnCodec: one entry). */
  readonly codebooks: readonly number[];
  /** HQ files: hash of the DAC codebooks the codes index, 16 hex digits. */
  readonly codebookHash: string | null;
}

/** How an HQ file's coded channels make its output. */
export type HqLayout = "mono" | "lr" | "ms" | "parametric";

const HQ_LAYOUTS: readonly HqLayout[] = ["mono", "lr", "ms", "parametric"];

const MODELS: Readonly<
  Record<ModelName, { sampleRate: number; channels: number; frameRate: number; segmentLength: number }>
> = {
  encodec_24khz: { sampleRate: 24_000, channels: 1, frameRate: 75, segmentLength: 0 },
  encodec_48khz: { sampleRate: 48_000, channels: 2, frameRate: 150, segmentLength: 48_000 },
  dac44: { sampleRate: 44_100, channels: 2, frameRate: 44_100 / 512, segmentLength: 0 },
};

function hex64(bytes: Uint8Array): string {
  return Array.from(bytes)
    .reverse()
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function toBytes(bytes: ArrayBuffer | ArrayBufferView): Uint8Array {
  return bytes instanceof ArrayBuffer
    ? new Uint8Array(bytes)
    : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

interface OggPage {
  readonly flags: number;
  readonly granule: bigint;
  readonly lacing: Uint8Array;
  readonly body: Uint8Array;
}

/** Splits a stream into Ogg pages, checking only the framing (not the CRC; the decoder does). */
function oggPages(view: Uint8Array): OggPage[] {
  const pages: OggPage[] = [];
  let at = 0;
  while (at < view.length) {
    if (view.length - at < 27 || String.fromCharCode(...view.subarray(at, at + 4)) !== "OggS") {
      throw new Error("not a .pico file (bad Ogg page)");
    }
    const data = new DataView(view.buffer, view.byteOffset + at);
    const count = view[at + 26] ?? 0;
    const lacing = view.subarray(at + 27, at + 27 + count);
    const size = lacing.reduce((sum, v) => sum + v, 0);
    const bodyAt = at + 27 + count;
    if (lacing.length !== count || bodyAt + size > view.length) {
      throw new Error(".pico file is truncated");
    }
    pages.push({
      flags: view[at + 5] ?? 0,
      granule: data.getBigUint64(6, true),
      lacing,
      body: view.subarray(bodyAt, bodyAt + size),
    });
    at = bodyAt + size;
  }
  return pages;
}

/** The packets of a stream, in order (a packet may span pages). */
function oggPackets(pages: readonly OggPage[]): Uint8Array[] {
  const packets: Uint8Array[] = [];
  let parts: Uint8Array[] = [];
  for (const page of pages) {
    let at = 0;
    for (const lace of page.lacing) {
      parts.push(page.body.subarray(at, at + lace));
      at += lace;
      if (lace < 255) {
        const size = parts.reduce((sum, p) => sum + p.length, 0);
        const packet = new Uint8Array(size);
        let to = 0;
        for (const p of parts) {
          packet.set(p, to);
          to += p.length;
        }
        packets.push(packet);
        parts = [];
      }
    }
  }
  return packets;
}

class Fields {
  #at = 0;
  constructor(private readonly bytes: Uint8Array) {}

  get remaining(): number {
    return this.bytes.length - this.#at;
  }

  take(n: number): Uint8Array {
    if (n > this.remaining) {
      throw new Error(".pico header is truncated");
    }
    const out = this.bytes.subarray(this.#at, this.#at + n);
    this.#at += n;
    return out;
  }

  u8(): number {
    return this.take(1)[0] ?? 0;
  }

  /** An unsigned LEB128 varint, at most 2^53 (a JavaScript integer). */
  varint(): number {
    let value = 0;
    for (let shift = 0; shift < 56; shift += 7) {
      const byte = this.u8();
      value += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) {
        if (!Number.isSafeInteger(value)) {
          throw new Error(".pico header value is out of range");
        }
        return value;
      }
    }
    throw new Error(".pico header varint is too long");
  }
}

const HEAD_LOOP = 1;
const HEAD_COMMENTS = 2;
const HEAD_LENGTH = 4;
const HEAD_EDGE = 8;

/**
 * Parses a `.pico` file's headers without the worker or a model. Reads the
 * identification header, the comment header if any, and (when the header
 * holds no length) the last page's granule position.
 *
 * @throws When the bytes are not a `.pico` file of a known model and version.
 */
export function readPicoHeader(bytes: ArrayBuffer | ArrayBufferView): PicoInfo {
  const view = toBytes(bytes);
  const pages = oggPages(view);
  const packets = oggPackets(pages);
  const head = packets[0];
  if (head === undefined) {
    throw new Error("not a .pico file (no header)");
  }
  const f = new Fields(head);
  if (String.fromCharCode(...f.take(4)) !== "Pico") {
    throw new Error("not a .pico file (bad magic)");
  }
  const version = f.u8();
  if (version !== 1) {
    throw new Error(`unsupported .pico version ${String(version)} (this decoder reads version 1)`);
  }
  const h = new Fields(f.take(f.varint()));
  const modelCode = h.u8();
  const name: ModelName | null =
    modelCode === 0 ? "encodec_24khz" : modelCode === 1 ? "encodec_48khz" : modelCode === 2 ? "dac44" : null;
  if (name === null) {
    throw new Error(`unknown model ${String(modelCode)}`);
  }
  const hq = name === "dac44";
  const model = MODELS[name];
  const scheme = h.u8();
  if (scheme !== 2) {
    throw new Error(`unknown entropy coder ${String(scheme)}`);
  }
  // HQ: the codebook hash comes before the table hash.
  const codebookHash = hq ? hex64(h.take(8)) : null;
  const tableHash = hex64(h.take(8));
  let layout: HqLayout | null = null;
  let codebooks: number[];
  if (hq) {
    const code = h.u8();
    layout = HQ_LAYOUTS[code] ?? null;
    if (layout === null) {
      throw new Error(`unknown HQ channel layout ${String(code)}`);
    }
    if (layout === "parametric") {
      h.u8(); // The parametric stereo configuration.
    }
    const coded = layout === "lr" || layout === "ms" ? 2 : 1;
    codebooks = Array.from({ length: coded }, () => h.u8());
    if (codebooks.some((n) => n < 1 || n > 9)) {
      throw new Error("implausible HQ codebook count");
    }
  } else {
    codebooks = [h.u8()];
  }
  const numCodebooks = Math.max(...codebooks);
  const flags = h.u8();
  const known = HEAD_LOOP | HEAD_COMMENTS | HEAD_LENGTH | (hq ? 0 : HEAD_EDGE);
  if ((flags & ~known) !== 0) {
    throw new Error(`unknown .pico header flags ${String(flags)}`);
  }
  const preSkip = h.varint();
  const stride = model.segmentLength === 0 ? 0 : h.varint();
  const pageLength = h.varint();
  let audioLength: number;
  let endTrim: number;
  if ((flags & HEAD_LENGTH) !== 0) {
    audioLength = h.varint();
    endTrim = h.varint();
  } else {
    // The last page's granule is the playable end; its packet starts with the end trim.
    const last = pages[pages.length - 1];
    const lastPacket = packets[packets.length - 1];
    if (last === undefined || lastPacket === undefined || packets.length < 2) {
      throw new Error(".pico file has no audio");
    }
    audioLength = Number(last.granule) - preSkip;
    endTrim = new Fields(lastPacket).varint();
  }
  let outputGainDb = 0;
  while (f.remaining > 0) {
    const kind = f.varint();
    const body = f.take(f.varint());
    if (kind === 2 && body.length === 2) {
      outputGainDb = new DataView(body.buffer, body.byteOffset, 2).getInt16(0, true) / 256;
    } else if (kind % 2 === 1) {
      throw new Error(`.pico stream uses critical field ${String(kind)}, which this decoder does not read`);
    }
  }
  const tags: string[] = [];
  if ((flags & HEAD_COMMENTS) !== 0) {
    const packet = packets[1];
    if (packet === undefined) {
      throw new Error(".pico comment header is missing");
    }
    const c = new Fields(packet);
    if (String.fromCharCode(...c.take(8)) !== "PicoTags") {
      throw new Error(".pico comment header is missing");
    }
    c.take(c.varint());
    const text = new TextDecoder();
    for (let n = c.varint(); n > 0; n -= 1) {
      tags.push(text.decode(c.take(c.varint())));
    }
  }
  if (!(audioLength > 0)) {
    throw new Error(".pico file has no audio");
  }
  const allCodebooks = codebooks.reduce((sum, n) => sum + n, 0);
  const bandwidthKbps = (allCodebooks * model.frameRate * 10) / 1000;
  const durationSeconds = audioLength / model.sampleRate;
  return {
    model: name,
    audioLength,
    numCodebooks,
    sampleRate: model.sampleRate,
    channels: layout === null ? model.channels : layout === "mono" ? 1 : 2,
    durationSeconds,
    bandwidthKbps,
    bitrateKbps: (view.length * 8) / durationSeconds / 1000,
    codedLength: preSkip + audioLength + endTrim,
    stride,
    crossfade: (flags & HEAD_EDGE) !== 0 ? 1 : 0,
    preSkip,
    endTrim,
    loop: (flags & HEAD_LOOP) !== 0,
    tableHash,
    pageLength,
    tags,
    outputGainDb,
    layout,
    codebooks,
    codebookHash,
  };
}
