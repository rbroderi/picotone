// Writes decoder-only checkpoints and the cx2 tables for the browser:
//   www/models/<name>.safetensors  (decoder.* and quantizer.* tensors only)
//   www/models/<name>.json         (the model config, copied unchanged)
//   www/models/EnCodec-MIT.txt     (the weights' licence, beside every model file)
//   www/models/dac44_student.*     (the HQ decoder, for model dac44: its weights, config,
//                                   NOTICE and training credits, copied unchanged)
//   www/models/dac44_vocos.*       (the Vocos HQ decoder, when ../weights has it; same files)
//   www/models/DAC-MIT.txt         (the licence of DAC, which the HQ decoder derives from)
//   www/models/<table>.ecxm        (the default cx2 table of each exported model)
//   www/models/cx2-tables.json     ({ tables: [{ file, model, hash }] }; the worker reads it)
// Reads ../weights/* and ../models/*.ecxm; never modifies them.
//
// Usage: node tools/export-models.mjs [name ...]
//   names: encodec_24khz, encodec_48khz, dac44 (HQ); default: all three

import {
  closeSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const weightsDir = resolve(here, "..", "..", "weights");
const outDir = resolve(here, "..", "www", "models");
const tablesDir = resolve(here, "..", "..", "models");
const licence = resolve(here, "..", "..", "LICENSES", "EnCodec-MIT.txt");
// The table each model codes with by default (as the picotone command line).
const TABLES = {
  encodec_24khz: "cx2_encodec_24khz_v5.ecxm",
  encodec_48khz: "cx2_encodec_48khz_net2_v5.ecxm",
  dac44: "cx2_dac44.ecxm",
};
// The HQ decoder networks of the dac44 code space, and the files shipped with each. The
// student is required; the Vocos decoder (tools/export_vocos.py) is copied when present.
const HQ_DECODER = "dac44_student";
const HQ_VOCOS = "dac44_vocos";
const hqFiles = (network) => [
  `${network}.safetensors`,
  `${network}.json`,
  `${network}.NOTICE.txt`,
  `${network}.credits.jsonl`,
];
const HQ_FILES = [...hqFiles(HQ_DECODER), "DAC-MIT.txt"];

/**
 * The hash a .pico stream stores for its table: FNV-1a 64 of the file's bytes, as 16 hex
 * digits. Self-contained (version 5) EnCodec tables and HQ tables (ECXH) only.
 */
function tableHash(bytes) {
  const magic = bytes.subarray(0, 4).toString("latin1");
  if (magic !== "ECXH" && (magic !== "ECXM" || bytes.readUInt32LE(4) !== 5)) {
    throw new Error("not a self-contained (version 5) cx2 table or an HQ table");
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
const names = requested.length > 0 ? requested : ["encodec_24khz", "encodec_48khz", "dac44"];
const manifest = [];
let encodec = false;
for (const name of names) {
  if (!(name in TABLES)) throw new Error(`unknown model ${name}`);
  if (name === "dac44") {
    mkdirSync(outDir, { recursive: true });
    for (const file of HQ_FILES) {
      copyFileSync(join(weightsDir, file), join(outDir, file));
    }
    console.log(`dac44: ${HQ_FILES.join(", ")} copied unchanged`);
    // The weights and config make the network; a trained export also has its NOTICE and
    // credits (a random-init stand-in has none).
    const vocos = hqFiles(HQ_VOCOS).filter((file) => existsSync(join(weightsDir, file)));
    if (vocos.includes(`${HQ_VOCOS}.safetensors`) && vocos.includes(`${HQ_VOCOS}.json`)) {
      for (const file of vocos) copyFileSync(join(weightsDir, file), join(outDir, file));
      const missing = hqFiles(HQ_VOCOS).filter((file) => !vocos.includes(file));
      console.log(
        `dac44: ${vocos.join(", ")} copied unchanged` +
          (missing.length > 0 ? ` (no ${missing.join(", ")}: not a trained export?)` : ""),
      );
    } else {
      console.log(`dac44: no ${HQ_VOCOS}.safetensors/.json in ${weightsDir}; the Vocos decoder is not exported`);
    }
  } else {
    exportModel(name);
    encodec = true;
  }
  const file = TABLES[name];
  const bytes = readFileSync(join(tablesDir, file));
  copyFileSync(join(tablesDir, file), join(outDir, file));
  manifest.push({ file, model: name, hash: tableHash(bytes) });
  console.log(`${name}: table ${file} (${(bytes.length / 1e6).toFixed(1)} MB, hash ${manifest.at(-1).hash})`);
}
if (encodec) copyFileSync(licence, join(outDir, "EnCodec-MIT.txt"));
writeFileSync(join(outDir, "cx2-tables.json"), JSON.stringify({ tables: manifest }, null, 2) + "\n");
console.log(`wrote ${join(outDir, "cx2-tables.json")}${encodec ? " and EnCodec-MIT.txt" : ""}`);
