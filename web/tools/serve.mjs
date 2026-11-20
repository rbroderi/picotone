// Minimal static file server for the demo.
// Usage: node tools/serve.mjs [--port 8080] [--host 127.0.0.1] [--data <dir>]
// Serves web/ at /, and optionally <dir> at /test-data/.

import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { resolve, join, extname, sep, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.map': 'application/json',
  '.bin': 'application/octet-stream',
  '.pico': 'audio/ogg',
  '.ecxm': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.txt': 'text/plain; charset=utf-8',
  '.wav': 'audio/wav',
  '.svg': 'image/svg+xml',
};

export function startServer({ port = 8080, host = '127.0.0.1', root, dataDir } = {}) {
  const webRoot = resolve(root ?? join(dirname(fileURLToPath(import.meta.url)), '..'));
  const mounts = [['/test-data/', dataDir ? resolve(dataDir) : null], ['/', webRoot]];
  const server = createServer((req, res) => {
    let path;
    try {
      path = decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname);
    } catch {
      res.writeHead(400).end();
      return;
    }
    for (const [prefix, dir] of mounts) {
      if (!dir || !path.startsWith(prefix)) continue;
      let file = resolve(dir, '.' + sep + path.slice(prefix.length));
      if (file !== dir && !file.startsWith(dir + sep)) break;
      try {
        let st = statSync(file);
        if (st.isDirectory()) {
          file = join(file, 'index.html');
          st = statSync(file);
        }
        res.writeHead(200, {
          'Content-Type': MIME[extname(file).toLowerCase()] ?? 'application/octet-stream',
          'Content-Length': st.size,
          'Cache-Control': 'no-cache',
        });
        if (req.method === 'HEAD') res.end();
        else createReadStream(file).pipe(res);
        return;
      } catch {
        break;
      }
    }
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
  });
  return new Promise((ok) => server.listen(port, host, () => ok(server)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const opt = (name, dflt) => {
    const i = args.indexOf(name);
    return i >= 0 ? args[i + 1] : dflt;
  };
  const port = Number(opt('--port', '8080'));
  const host = opt('--host', '127.0.0.1');
  await startServer({ port, host, dataDir: opt('--data', undefined) });
  console.log(`serving web/ at http://${host}:${port}/  (WebGPU needs localhost or HTTPS)`);
}
