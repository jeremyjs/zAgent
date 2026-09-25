// Compiler service: loads zig.wasm + stdlib + compiler_rt once, then compiles on request.
import { ZigCompiler } from './zigc.js';

let compiler = null, loading = null;
const status = text => postMessage({ type: 'status', text });

async function fetchBytes(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const out = new Uint8Array(total), reader = r.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length;
    postMessage({ type: 'progress', label, got, total });
  }
  return out;
}

function load(base) {
  loading ??= (async () => {
    const t0 = performance.now();
    status('downloading compiler');
    const [zigWasm, zigTar, compilerRt] = await Promise.all([
      fetchBytes(base + 'bin/zig.wasm', 'zig.wasm'),
      fetchBytes(base + 'zig.tar.gz', 'stdlib'),
      fetchBytes(base + 'libcompiler_rt.a', 'compiler_rt'),
    ]);
    status('preparing compiler');
    compiler = await ZigCompiler.load({ zigWasm, zigTar, compilerRt });
    postMessage({ type: 'ready', ms: performance.now() - t0 });
  })().catch(err => { loading = null; throw err; });
  return loading;
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load') await load(m.base);
    if (m.type === 'compile') {
      await load(m.base);
      status('compiling');
      const r = await compiler.compile(m.source);
      status('ready');
      if (r.ok) postMessage({ type: 'result', id: m.id, ok: true, wasm: r.wasm, ms: r.ms, stderr: r.stderr }, [r.wasm.buffer]);
      else postMessage({ type: 'result', id: m.id, ok: false, error: r.stderr, ms: r.ms });
    }
  } catch (err) {
    const error = String(err?.message || err);
    if (m.type === 'compile') postMessage({ type: 'result', id: m.id, ok: false, error });
    else postMessage({ type: 'error', error });
  }
};
