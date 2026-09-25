// Host side of ABI v1 (see ABI.md). Environment-agnostic: used by the runtime
// worker, the candidate sandbox, and the Node tests.

export const EV = Object.freeze({ USER: 1, LLM_REPLY: 2, BUILD_FAILED: 3, TIMER: 4, SELFTEST: 5, LLM_ERROR: 6 });
export const REQUIRED_EXPORTS = ['memory', 'alloc', 'free', 'boot', 'on_event'];
export const HOST_IMPORTS = ['host_log', 'host_say', 'host_llm', 'host_propose', 'host_save', 'host_timer', 'host_now'];

const enc = new TextEncoder();
const dec = new TextDecoder();

/** Minimal WASI preview1 stubs so std.debug output / panics from a Debug build still work. */
function wasiStubs(onLog) {
  let mem = null;
  const ENOSYS = 52;
  const view = () => new DataView(mem.buffer);
  const stubs = {
    fd_write(fd, iovs, iovsLen, nwritten) {
      let total = 0, text = '';
      for (let i = 0; i < iovsLen; i++) {
        const p = view().getUint32(iovs + i * 8, true), l = view().getUint32(iovs + i * 8 + 4, true);
        text += dec.decode(new Uint8Array(mem.buffer, p, l));
        total += l;
      }
      if (fd === 1 || fd === 2) onLog(text.replace(/\n$/, ''));
      view().setUint32(nwritten, total, true);
      return 0;
    },
    proc_exit(code) { throw new Error(`agent called proc_exit(${code})`); },
    random_get(buf, len) { crypto.getRandomValues(new Uint8Array(mem.buffer, buf, len)); return 0; },
    clock_time_get(_id, _prec, out) { view().setBigUint64(out, BigInt(Date.now()) * 1_000_000n, true); return 0; },
    environ_sizes_get(a, b) { view().setUint32(a, 0, true); view().setUint32(b, 0, true); return 0; },
    args_sizes_get(a, b) { view().setUint32(a, 0, true); view().setUint32(b, 0, true); return 0; },
    environ_get() { return 0; },
    args_get() { return 0; },
  };
  return { bind: m => { mem = m; }, get: name => stubs[name] ?? (() => ENOSYS) };
}

/**
 * Instantiate an agent generation.
 * @param {BufferSource} bytes
 * @param {{log(s:string):void, say(s:string):void, llm(prompt:string):number, propose(src:string):void, save(bytes:Uint8Array):void, timer(ms:number):void}} host
 */
export async function instantiateAgent(bytes, host) {
  const module = await WebAssembly.compile(bytes);
  const exportNames = WebAssembly.Module.exports(module).map(e => e.name);
  const missing = REQUIRED_EXPORTS.filter(n => !exportNames.includes(n));
  if (missing.length) throw new Error(`missing exports: ${missing.join(', ')}`);

  let memory = null;
  const str = (p, l) => dec.decode(new Uint8Array(memory.buffer, p, l).slice());
  const env = {
    host_log: (p, l) => host.log(str(p, l)),
    host_say: (p, l) => host.say(str(p, l)),
    host_llm: (p, l) => host.llm(str(p, l)) >>> 0,
    host_propose: (p, l) => host.propose(str(p, l)),
    host_save: (p, l) => host.save(new Uint8Array(memory.buffer, p, l).slice()),
    host_timer: ms => host.timer(ms >>> 0),
    host_now: () => Date.now(),
  };
  const wasi = wasiStubs(host.log);
  const imports = { env: {}, wasi_snapshot_preview1: {} };
  for (const imp of WebAssembly.Module.imports(module)) {
    if (imp.module === 'env' && imp.kind === 'function') {
      if (!env[imp.name]) throw new Error(`imports unknown host function env.${imp.name}`);
      imports.env[imp.name] = env[imp.name];
    } else if (imp.module === 'wasi_snapshot_preview1') {
      imports.wasi_snapshot_preview1[imp.name] = wasi.get(imp.name);
    } else {
      throw new Error(`unsupported import ${imp.module}.${imp.name}`);
    }
  }
  const instance = await WebAssembly.instantiate(module, imports);
  const x = instance.exports;
  memory = x.memory;
  wasi.bind(memory);

  /** Copy bytes into agent memory, run fn(ptr,len), free. */
  const withBytes = (data, fn) => {
    const b = typeof data === 'string' ? enc.encode(data) : (data ?? new Uint8Array());
    if (b.length === 0) return fn(0, 0);
    const p = x.alloc(b.length) >>> 0;
    if (p === 0) throw new Error('agent alloc() returned null');
    new Uint8Array(memory.buffer, p, b.length).set(b);
    try { return fn(p, b.length); } finally { x.free(p, b.length); }
  };

  return {
    exports: exportNames,
    boot: state => withBytes(state, (p, l) => x.boot(p, l)),
    event: (kind, text) => withBytes(text, (p, l) => x.on_event(kind, p, l)),
  };
}
