// zigc: run Zig's self-hosted compiler (zig.wasm, wasm32-wasi) fully in memory.
// Shared by the browser compiler worker and the Node test harness.
import { WASI, WASIProcExit, File, Directory, OpenFile, PreopenDirectory, ConsoleStdout, wasi as defs } from '../vendor/wasi-shim/index.js';

/** Minimal ustar reader: returns [{name, data}] for regular files. */
export function untar(bytes) {
  const out = [], dec = new TextDecoder();
  let off = 0, longName = null;
  while (off + 512 <= bytes.length) {
    const h = bytes.subarray(off, off + 512);
    if (h[0] === 0) break;
    const str = (a, b) => dec.decode(h.subarray(a, b)).replace(/\0.*$/s, '');
    let name = str(0, 100);
    const prefix = str(345, 500);
    if (prefix) name = prefix + '/' + name;
    const size = parseInt(str(124, 136).trim() || '0', 8);
    const type = String.fromCharCode(h[156] || 48);
    const data = bytes.subarray(off + 512, off + 512 + size);
    off += 512 + Math.ceil(size / 512) * 512;
    if (type === 'L') { longName = dec.decode(data).replace(/\0.*$/s, ''); continue; }
    if (type === 'x' || type === 'g') {
      const m = dec.decode(data).match(/\d+ path=([^\n]*)\n/);
      if (m && type === 'x') longName = m[1];
      continue;
    }
    if (longName) { name = longName; longName = null; }
    if (type === '0' || type === '\0' || type === '7') out.push({ name, data: data.slice() });
  }
  return out;
}

async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Build the in-memory /lib tree from zig.tar(.gz). */
export async function buildLib(tarBytes) {
  const root = new Map();
  for (const { name, data } of untar(await gunzip(tarBytes))) {
    if (!name.startsWith('lib/')) continue;
    const parts = name.slice(4).split('/').filter(Boolean);
    if (!parts.length) continue;
    let dir = root;
    for (const seg of parts.slice(0, -1)) { if (!dir.has(seg)) dir.set(seg, new Map()); dir = dir.get(seg); }
    dir.set(parts.at(-1), data);
  }
  const convert = node => new Directory([...node].map(([k, v]) => [k, v instanceof Uint8Array ? new File(v, { readonly: true }) : convert(v)]));
  return convert(root);
}

export class ZigCompiler {
  /** @param {{module: WebAssembly.Module, lib: Directory, compilerRt: Uint8Array}} parts */
  constructor({ module, lib, compilerRt }) { Object.assign(this, { module, lib, compilerRt }); }

  static async load({ zigWasm, zigTar, compilerRt }) {
    const [module, lib] = await Promise.all([WebAssembly.compile(zigWasm), buildLib(zigTar)]);
    return new ZigCompiler({ module, lib, compilerRt });
  }

  /**
   * Compile one Zig source file to a freestanding-style wasm32-wasi module.
   * @returns {Promise<{ok: boolean, code: number, wasm?: Uint8Array, stderr: string, ms: number}>}
   */
  async compile(source, { extraArgs = [], onStderr } = {}) {
    let stderr = '';
    const dec = new TextDecoder();
    const sink = () => {
      const c = new ConsoleStdout(buf => { const t = dec.decode(buf, { stream: true }); stderr += t; onStderr?.(t); });
      c.fd_pwrite = () => ({ ret: defs.ERRNO_SPIPE, nwritten: 0 });
      return c;
    };
    const cwd = new PreopenDirectory('.', new Map([
      ['main.zig', new File(new TextEncoder().encode(source))],
      ['libcompiler_rt.a', new File(this.compilerRt, { readonly: true })],
    ]));
    const args = ['zig.wasm', 'build-exe', 'main.zig', 'libcompiler_rt.a', '-fno-compiler-rt', '-fno-entry', '-rdynamic', ...extraArgs];
    const wasi = new WASI(args, [], [
      new OpenFile(new File([])), sink(), sink(), cwd,
      new PreopenDirectory('/lib', this.lib.contents),
      new PreopenDirectory('/cache', new Map()),
    ], { debug: false });
    const instance = await WebAssembly.instantiate(this.module, { wasi_snapshot_preview1: wasi.wasiImport });
    const t0 = performance.now();
    let code;
    try { code = wasi.start(instance); }
    catch (e) {
      if (e instanceof WASIProcExit) code = e.code;
      else return { ok: false, code: -1, stderr: stderr + '\n' + String(e?.stack || e), ms: performance.now() - t0 };
    }
    const ms = performance.now() - t0;
    const out = cwd.dir.contents.get('main.wasm');
    if (code === 0 && out) return { ok: true, code, wasm: out.data.slice(), stderr, ms };
    return { ok: false, code, stderr: stderr || `zig exited ${code} without main.wasm`, ms };
  }
}
