// Bundles the demo app, the parity test page and an ESM library build, and copies the
// single-threaded picotone WebAssembly module (the cx2 range decoder) to public/models/cx2/.
import { build } from 'esbuild';
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const common = { bundle: true, format: 'esm', target: 'es2022', sourcemap: true, logLevel: 'info', absWorkingDir: root };

await build({
  ...common,
  entryPoints: { app: 'src/app/main.ts', parity: 'src/test/parity.ts' },
  outdir: 'public/build',
  minify: true,
});
await build({ ...common, entryPoints: { 'picotone-webgpu': 'src/index.ts' }, outdir: 'dist', minify: false });

const pkg = join(root, '..', 'wasm', 'www', 'picotone', 'pkg-st');
const cx2 = join(root, 'public', 'models', 'cx2');
const files = ['picotone.js', 'picotone_bg.wasm'];
if (!files.every((f) => existsSync(join(pkg, f)))) {
  throw new Error(`${pkg} is missing; build it with: powershell -File ../wasm/build.ps1 -Only st -SkipModels`);
}
mkdirSync(cx2, { recursive: true });
for (const f of files) copyFileSync(join(pkg, f), join(cx2, f));
console.log(`copied the cx2 decoder to ${cx2}`);
