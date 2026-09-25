// Headless compile using the same in-memory path as the browser.
// Usage: node tools/compile-node.mjs <file.zig> [out.wasm]
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { ZigCompiler } from '../src/zigc.js';

const root = resolve(import.meta.dirname, '..');
const src = process.argv[2] ?? join(root, 'seed/agent.zig');
const out = process.argv[3] ?? join(root, 'out/agent.wasm');
const z = join(root, 'zig-out');
const t0 = performance.now();
const zc = await ZigCompiler.load({
  zigWasm: await readFile(join(z, 'bin/zig.wasm')),
  zigTar: new Uint8Array(await readFile(join(z, 'zig.tar.gz'))),
  compilerRt: new Uint8Array(await readFile(join(z, 'libcompiler_rt.a'))),
});
console.error(`compiler loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
const r = await zc.compile(await readFile(src, 'utf8'), { extraArgs: process.argv.slice(4) });
process.stderr.write(r.stderr);
console.error(`zig exited ${r.code} in ${(r.ms / 1000).toFixed(1)}s`);
if (r.ok) { await mkdir(dirname(out), { recursive: true }); await writeFile(out, r.wasm); console.error(`wrote ${out} (${r.wasm.length} bytes)`); }
process.exit(r.ok ? 0 : 1);
