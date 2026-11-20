// Writes decoder-only checkpoints and the cx2 tables for the browser:
//   www/models/<name>.safetensors  (decoder.* and quantizer.* tensors only)
//   www/models/<name>.json         (the model config, copied unchanged)
//   www/models/EnCodec-MIT.txt     (the weights' licence, beside every model file)
//   www/models/<table>.ecxm        (the default cx2 table of each exported model)
//   www/models/cx2-tables.json     ({ tables: [{ file, model, hash }] }; the worker reads it)
// Reads ../weights/<name>.{json,safetensors} and ../models/*.ecxm; never modifies them.
//
// Usage: node tools/export-models.mjs [name ...]   (default: both models)

import { mkdirSync, openSync, readSync, closeSync, writeFileSync, copyFileSync, statSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const weightsDir = resolve(here, "..", "..", "weights");
const outDir = resolve(here, "..", "www", "models");
const tablesDir = resolve(here, "..", "..", "models");
const licence = resolve(here, "..", "..", "LICENSES", "EnCodec-MIT.txt");
// The table each model codes with by default (as the picotone command line).
const TABLES = { encodec_24khz: "cx2_encodec_24khz_v5.ecxm", encodec_48khz: "cx2_encodec_48khz_net2_v5.ecxm" };

/** The hash a .pico stream stores for a self-contained table: FNV-1a 64 of its bytes, as 16 hex digits. */
function tableHash(bytes) {
  if (bytes.subarray(0, 4).toString("latin1") !== "ECXM" || bytes.readUInt32LE(4) !== 5) {
    throw new Error("not a self-contained (version 5) cx2 table");
  }
  let h = 0xcbf29ce484222325n;
  for (const b of bytes) h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
  return h.toString(16).padStart(16, "0");
}
const KEEP = ["decoder.", "quantizer."];

function readRange(fd, position, length) {
  const buf = Buffer.alloc(length);
  let done = 0;
  while (done < length) {
    const n = readSync(fd, buf, done, length - done, position + done);
    if (n === 0) throw new Error("unexpected end of file");
    done += n;
  }
  return buf;
}

function exportModel(name) {
  const src = join(weightsDir, `${name}.safetensors`);
  const fd = openSync(src, "r");
  try {
    const headerLen = Number(readRange(fd, 0, 8).readBigUInt64LE(0));
    const header = JSON.parse(readRange(fd, 8, headerLen).toString("utf8"));
    const dataStart = 8 + headerLen;
    const names = Object.keys(header)
      .filter((k) => k !== "__metadata__" && KEEP.some((p) => k.startsWith(p)))
      .sort((a, b) => header[a].data_offsets[0] - header[b].data_offsets[0]);
    const outHeader = {};
    if (header.__metadata__) outHeader.__metadata__ = header.__metadata__;
    let offset = 0;
    for (const k of names) {
      const [s, e] = header[k].data_offsets;
      outHeader[k] = { dtype: header[k].dtype, shape: header[k].shape, data_offsets: [offset, offset + (e - s)] };
      offset += e - s;
    }
    let json = Buffer.from(JSON.stringify(outHeader), "utf8");
    // Pad the header with spaces to an 8-byte boundary, as safetensors does.
    const pad = (8 - (json.length % 8)) % 8;
    json = Buffer.concat([json, Buffer.alloc(pad, 0x20)]);
    const out = Buffer.alloc(8 + json.length + offset);
    out.writeBigUInt64LE(BigInt(json.length), 0);
    json.copy(out, 8);
    let at = 8 + json.length;
    for (const k of names) {
      const [s, e] = header[k].data_offsets;
      readRange(fd, dataStart + s, e - s).copy(out, at);
      at += e - s;
    }
    mkdirSync(outDir, { recursive: true });
    const dst = join(outDir, `${name}.safetensors`);
    writeFileSync(dst, out);
    copyFileSync(join(weightsDir, `${name}.json`), join(outDir, `${name}.json`));
    const total = Object.keys(header).filter((k) => k !== "__metadata__").length;
    console.log(
      `${name}: kept ${names.length}/${total} tensors, ` +
        `${(statSync(src).size / 1e6).toFixed(1)} MB -> ${(out.length / 1e6).toFixed(1)} MB (${dst})`,
    );
  } finally {
    closeSync(fd);
  }
}

const requested = process.argv.slice(2);
const names = requested.length > 0 ? requested : ["encodec_24khz", "encodec_48khz"];
const manifest = [];
for (const name of names) {
  exportModel(name);
  const file = TABLES[name];
  const bytes = readFileSync(join(tablesDir, file));
  copyFileSync(join(tablesDir, file), join(outDir, file));
  manifest.push({ file, model: name, hash: tableHash(bytes) });
  console.log(`${name}: table ${file} (${(bytes.length / 1e6).toFixed(1)} MB, hash ${manifest.at(-1).hash})`);
}
copyFileSync(licence, join(outDir, "EnCodec-MIT.txt"));
writeFileSync(join(outDir, "cx2-tables.json"), JSON.stringify({ tables: manifest }, null, 2) + "\n");
console.log(`wrote ${join(outDir, "cx2-tables.json")} and EnCodec-MIT.txt`);
