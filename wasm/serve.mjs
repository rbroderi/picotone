// Tiny static server for wasm/www with cross-origin isolation headers.
//
//   node serve.mjs [--port 8080] [--host 127.0.0.1] [--no-isolation] [--root www] [--extra dir]
//   (--port 0 picks a free port and prints it)
//
// By default every response carries
//   Cross-Origin-Opener-Policy: same-origin
//   Cross-Origin-Embedder-Policy: require-corp
// so the page is cross-origin isolated and the threaded build runs.
// --no-isolation leaves them out to exercise the single-threaded fallback.

import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const isolation = !args.includes("--no-isolation");
const port = Number(option("--port", "8080"));
const host = option("--host", "127.0.0.1");
const root = resolve(here, option("--root", "www"));
// --extra <dir> serves <dir> at /test-data/ (used by test/run.mjs).
const extraDir = option("--extra", null);
const extra = extraDir === null ? null : resolve(extraDir);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".wav": "audio/wav",
  ".pico": "audio/ogg",
  ".ecxm": "application/octet-stream",
  ".txt": "text/plain; charset=utf-8",
  ".safetensors": "application/octet-stream",
  ".map": "application/json",
  ".ts": "text/plain; charset=utf-8",
};

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  let path = decodeURIComponent(url.pathname);
  if (path.endsWith("/")) path += "index.html";
  let base = root;
  if (extra !== null && path.startsWith("/test-data/")) {
    base = extra;
    path = path.slice("/test-data".length);
  }
  const file = normalize(join(base, path));
  if (file !== base && !file.startsWith(base + sep)) {
    res.writeHead(403).end("forbidden");
    return;
  }
  let stat;
  try {
    stat = statSync(file);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("not found");
    return;
  }
  if (!stat.isFile()) {
    res.writeHead(404).end("not found");
    return;
  }
  const headers = {
    "Content-Type": TYPES[extname(file).toLowerCase()] ?? "application/octet-stream",
    "Content-Length": String(stat.size),
    // Revalidate everything: a rebuild must never be served stale.
    "Cache-Control": "no-cache",
    "Last-Modified": stat.mtime.toUTCString(),
  };
  if (isolation) {
    headers["Cross-Origin-Opener-Policy"] = "same-origin";
    headers["Cross-Origin-Embedder-Policy"] = "require-corp";
  }
  const since = req.headers["if-modified-since"];
  if (since && Date.parse(since) >= Math.floor(stat.mtimeMs / 1000) * 1000) {
    res.writeHead(304, headers).end();
    return;
  }
  res.writeHead(200, headers);
  if (req.method === "HEAD") {
    res.end();
    return;
  }
  createReadStream(file).pipe(res);
});

server.listen(port, host, () => {
  const address = server.address();
  const actual = typeof address === "object" && address ? address.port : port;
  console.log(
    `serving ${root} at http://${host}:${actual}/ ` +
      (isolation ? "(cross-origin isolated: COOP/COEP on)" : "(--no-isolation: COOP/COEP off)"),
  );
});
