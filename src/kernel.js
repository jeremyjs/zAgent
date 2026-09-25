// zAgent kernel: the immutable part. Owns storage, the compiler, the LLM, the
// generation lifecycle, and every guardrail. The agent (Zig -> wasm) owns behaviour.
import { EV } from './host.js';
import { PROVIDERS } from './providers/index.js';

const ZIG_BASE = new URL('../zig-out/', import.meta.url).href;
const RUNTIME_URL = new URL('./runtime-worker.js', import.meta.url);
const COMPILER_URL = new URL('./compiler-worker.js', import.meta.url);
const EVENT_BUDGET_MS = 5000;
const SANDBOX_BUDGET_MS = 5000;
const MIN_TIMER_MS = 1000;
const MAX_ERROR_CHARS = 6000;

// ---------------------------------------------------------------- storage (OPFS)

class Store {
  async mount() {
    if (!navigator.storage?.getDirectory) throw new Error('OPFS unavailable (serve over http://localhost or https)');
    this.root = await (await navigator.storage.getDirectory()).getDirectoryHandle('zagent', { create: true });
    await navigator.storage.persist?.().catch(() => {});
  }
  async #dir(parts, create) { let d = this.root; for (const p of parts) d = await d.getDirectoryHandle(p, { create }); return d; }
  async write(path, data) {
    const parts = path.split('/').filter(Boolean), name = parts.pop();
    const h = await (await this.#dir(parts, true)).getFileHandle(name, { create: true });
    const w = await h.createWritable(); await w.write(data); await w.close();
  }
  async read(path, as = 'text') {
    const parts = path.split('/').filter(Boolean), name = parts.pop();
    const f = await (await (await this.#dir(parts, false)).getFileHandle(name)).getFile();
    return as === 'bytes' ? new Uint8Array(await f.arrayBuffer()) : f.text();
  }
  async readJSON(path, fallback) { try { return JSON.parse(await this.read(path)); } catch { return fallback; } }
  writeJSON(path, v) { return this.write(path, JSON.stringify(v, null, 2)); }
  async wipe() { for await (const name of this.root.keys()) await this.root.removeEntry(name, { recursive: true }); }
}

// ---------------------------------------------------------------- settings

const SETTINGS_KEY = 'zagent.settings';
const KEY_KEY = 'zagent.apiKey';
function loadSettings() {
  let s = {};
  try { s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'); } catch {}
  let apiKey = '';
  try { apiKey = sessionStorage.getItem(KEY_KEY) || localStorage.getItem(KEY_KEY) || ''; } catch {}
  return { provider: 'local', localModel: 'smollm2-135m', model: '', endpoint: '', effort: 'high', rememberKey: false, minGapMs: 3000, maxCallsPerHour: 60, ...s, apiKey };
}
function storeSettings(s) {
  const { apiKey, ...rest } = s;
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(rest));
    sessionStorage.setItem(KEY_KEY, apiKey);
    if (s.rememberKey) localStorage.setItem(KEY_KEY, apiKey); else localStorage.removeItem(KEY_KEY);
  } catch {}
}

// ---------------------------------------------------------------- kernel

export class Kernel extends EventTarget {
  constructor({ recovery = false } = {}) {
    super();
    this.recovery = recovery;
    this.store = new Store();
    this.settings = loadSettings();
    this.manifest = { nextId: 0, current: null, generations: [] };
    this.state = new Uint8Array();
    this.active = null;        // { id, source, worker }
    this.paused = false;
    this.building = false;
    this.seq = 0;
    this.watchdogs = new Map();
    this.timer = null;
    this.timerHeld = false;
    this.llmLog = [];          // timestamps of LLM calls, for rate limiting
    this.llmChain = Promise.resolve();
    this.stats = { llmCalls: 0, builds: 0, buildFailures: 0, rollbacks: 0 };
    this.compiler = null;
    this.compileSeq = 0;
    this.compileWaiters = new Map();
    this.compilerState = 'not loaded';
  }

  // ---- UI plumbing
  emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }
  line(text, cls = 'sys') { this.emit('line', { text, cls }); }
  changed() { this.emit('change'); }

  updateSettings(patch) { Object.assign(this.settings, patch); storeSettings(this.settings); this.changed(); }

  // ---- boot

  async boot() {
    await this.store.mount();
    this.abi = await (await fetch(new URL('../ABI.md', import.meta.url))).text();
    this.manifest = await this.store.readJSON('manifest.json', this.manifest);
    try { this.state = await this.store.read('state.bin', 'bytes'); } catch {}
    this.changed();

    const cur = !this.recovery && this.gen(this.manifest.current);
    if (cur) {
      try {
        this.line(`Restoring generation ${cur.id} from OPFS`);
        await this.activate(cur.id);
        return;
      } catch (e) {
        this.line(`Generation ${cur.id} failed to boot: ${e.message}`, 'error');
      }
    }
    await this.bootstrap();
  }

  /** Self-bootstrap: compile the seed source with the in-browser Zig compiler. */
  async bootstrap() {
    this.line(this.recovery ? 'Recovery mode: rebuilding from seed' : 'No generation yet. Self-bootstrapping: compiling seed/agent.zig with Zig-in-WASM');
    const source = await (await fetch(new URL('../seed/agent.zig', import.meta.url), { cache: 'no-store' })).text();
    const r = await this.build(source, { parent: null, origin: 'seed' });
    if (!r.ok) throw new Error('seed failed to build: ' + r.error);
  }

  gen(id) { return this.manifest.generations.find(g => g.id === id) || null; }

  // ---- compiler

  ensureCompiler() {
    if (this.compiler) return;
    this.compiler = new Worker(COMPILER_URL, { type: 'module' });
    this.compiler.onmessage = ({ data: m }) => {
      if (m.type === 'status') { this.compilerState = m.text; this.changed(); }
      else if (m.type === 'progress') this.emit('progress', m);
      else if (m.type === 'ready') { this.compilerState = 'ready'; this.line(`Zig compiler ready (${(m.ms / 1000).toFixed(1)}s)`, 'tool'); this.changed(); }
      else if (m.type === 'error') { this.compilerState = 'failed'; this.line('Compiler failed to load: ' + m.error, 'error'); this.changed(); }
      else if (m.type === 'result') { this.compileWaiters.get(m.id)?.(m); this.compileWaiters.delete(m.id); }
    };
    this.compiler.postMessage({ type: 'load', base: ZIG_BASE });
  }

  compile(source) {
    this.ensureCompiler();
    const id = ++this.compileSeq;
    return new Promise(res => {
      this.compileWaiters.set(id, res);
      this.compiler.postMessage({ type: 'compile', id, source, base: ZIG_BASE });
    });
  }

  // ---- build pipeline: compile -> sandbox self-test -> persist -> activate

  async build(source, { parent, origin }) {
    if (this.building) return { ok: false, error: 'another build is already in progress' };
    this.building = true; this.changed();
    try {
      this.stats.builds++;
      this.line(`Building candidate (${origin}, ${source.length} bytes)…`, 'tool');
      const c = await this.compile(source);
      if (!c.ok) return this.#buildFail(`compile error:\n${c.error}`);
      this.line(`Compiled in ${(c.ms / 1000).toFixed(1)}s → ${c.wasm.length} bytes. Self-testing in sandbox…`, 'tool');
      const t = await this.sandboxTest(c.wasm);
      if (!t.ok) return this.#buildFail(`self-test failed: ${t.error}`);

      const id = 'g' + this.manifest.nextId++;
      await this.store.write(`gens/${id}.wasm`, c.wasm);
      await this.store.write(`gens/${id}.zig`, source);
      this.manifest.generations.push({ id, parent, origin, createdAt: new Date().toISOString(), size: c.wasm.length, status: 'ok' });
      await this.store.writeJSON('manifest.json', this.manifest);
      try {
        await this.activate(id, { bytes: c.wasm, source });
      } catch (e) {
        this.gen(id).status = 'crashed';
        await this.store.writeJSON('manifest.json', this.manifest);
        if (parent) await this.activate(parent).catch(() => {});
        return this.#buildFail(`boot failed after passing sandbox: ${e.message}`);
      }
      this.line(`✓ Generation ${id} active`, 'tool');
      return { ok: true, id };
    } finally {
      this.building = false; this.changed();
    }
  }

  #buildFail(error) {
    this.stats.buildFailures++;
    const trimmed = error.length > MAX_ERROR_CHARS ? error.slice(0, MAX_ERROR_CHARS) + '\n…(truncated)' : error;
    this.line('✗ ' + trimmed, 'error');
    return { ok: false, error: trimmed };
  }

  /** Boot the candidate in a throwaway worker with the real state; it must survive selftest + /status. */
  sandboxTest(wasm) {
    return new Promise(resolve => {
      const w = new Worker(RUNTIME_URL, { type: 'module' });
      const finish = r => { clearTimeout(t); w.terminate(); resolve(r); };
      const t = setTimeout(() => finish({ ok: false, error: `no response within ${SANDBOX_BUDGET_MS} ms (infinite loop?)` }), SANDBOX_BUDGET_MS);
      let step = 0;
      w.onerror = e => finish({ ok: false, error: e.message || 'worker error' });
      w.onmessage = ({ data: m }) => {
        if (m.type === 'trap') return finish({ ok: false, error: m.error.split('\n').slice(0, 6).join('\n') });
        if (m.type !== 'done') return; // host calls are ignored in the sandbox
        step++;
        if (step === 1) w.postMessage({ type: 'event', seq: 2, kind: EV.SELFTEST, text: '' });
        else if (step === 2) w.postMessage({ type: 'event', seq: 3, kind: EV.USER, text: '/status' });
        else finish({ ok: true });
      };
      const bytes = wasm.slice();
      w.postMessage({ type: 'boot', seq: 1, bytes, state: this.state.slice() }, [bytes.buffer]);
    });
  }

  // ---- activation / runtime

  async activate(id, preloaded) {
    const bytes = preloaded?.bytes ?? await this.store.read(`gens/${id}.wasm`, 'bytes');
    const source = preloaded?.source ?? await this.store.read(`gens/${id}.zig`);
    this.#stopRuntime();
    const worker = new Worker(RUNTIME_URL, { type: 'module' });
    this.active = { id, source, worker };
    worker.onmessage = ({ data: m }) => this.#onRuntime(worker, m);
    worker.onerror = e => this.#onRuntime(worker, { type: 'trap', error: e.message || 'worker error' });
    this.manifest.current = id;
    await this.store.writeJSON('manifest.json', this.manifest);
    this.changed();
    const copy = bytes.slice();
    await new Promise((resolve, reject) => {
      const seq = ++this.seq;
      this.#watch(seq, 'boot');
      this.bootWaiter = { seq, resolve, reject };
      worker.postMessage({ type: 'boot', seq, bytes: copy, state: this.state.slice() }, [copy.buffer]);
    });
  }

  #failBoot(error) {
    const w = this.bootWaiter; this.bootWaiter = null;
    this.#stopRuntime(); this.changed();
    w.reject(new Error(error));
  }

  #stopRuntime() {
    clearTimeout(this.timer); this.timer = null; this.timerHeld = false;
    for (const t of this.watchdogs.values()) clearTimeout(t);
    this.watchdogs.clear();
    this.active?.worker.terminate();
    this.active = null;
  }

  #watch(seq, what) {
    this.watchdogs.set(seq, setTimeout(() => {
      this.watchdogs.delete(seq);
      const msg = `${what} did not return within ${EVENT_BUDGET_MS} ms`;
      if (this.bootWaiter?.seq === seq) return this.#failBoot(msg);
      this.#onTrap(msg);
    }, EVENT_BUDGET_MS));
  }

  send(kind, text = '') {
    if (!this.active) return;
    const seq = ++this.seq;
    this.#watch(seq, `event ${kind}`);
    this.active.worker.postMessage({ type: 'event', seq, kind, text });
  }

  #onRuntime(worker, m) {
    if (worker !== this.active?.worker) return; // stale generation
    if (m.type === 'done' || m.type === 'trap') {
      clearTimeout(this.watchdogs.get(m.seq)); this.watchdogs.delete(m.seq);
      if (this.bootWaiter && (this.bootWaiter.seq === m.seq || m.seq === undefined)) {
        if (m.type === 'trap') return this.#failBoot(m.error);
        const w = this.bootWaiter; this.bootWaiter = null;
        this.line(`Runtime ${this.active.id} up · exports: ${m.exports.join(', ')}`);
        return w.resolve();
      }
      if (m.type === 'trap') this.#onTrap(m.error);
      return;
    }
    if (m.type !== 'call') return;
    const genId = this.active.id;
    switch (m.name) {
      case 'log': this.line(`[${genId}] ${m.arg}`, 'log'); break;
      case 'say': this.emit('say', { text: m.arg, gen: genId }); break;
      case 'save':
        this.state = m.arg;
        this.store.write('state.bin', m.arg).catch(e => this.line('state save failed: ' + e.message, 'error'));
        break;
      case 'timer':
        clearTimeout(this.timer);
        this.timer = setTimeout(() => {
          this.timer = null;
          if (this.paused) this.timerHeld = true; else this.send(EV.TIMER);
        }, Math.max(MIN_TIMER_MS, m.arg));
        break;
      case 'llm': this.#llm(genId, m.arg); break;
      case 'propose': this.#propose(genId, m.arg); break;
    }
  }

  async #onTrap(error) {
    const cur = this.active && this.gen(this.active.id);
    this.line(`Runtime trap in ${cur?.id ?? '?'}: ${error.split('\n')[0]}`, 'error');
    this.#stopRuntime();
    if (cur) { cur.status = 'crashed'; await this.store.writeJSON('manifest.json', this.manifest); }
    const parent = cur?.parent && this.gen(cur.parent);
    if (parent) {
      this.stats.rollbacks++;
      this.line(`Rolling back to ${parent.id}`, 'tool');
      await this.activate(parent.id).catch(e => this.line(`Rollback failed: ${e.message}`, 'error'));
    } else {
      this.line('No parent generation to roll back to. Use Compile, Recovery (?recovery) or Reset.', 'error');
    }
    this.changed();
  }

  async #propose(genId, source) {
    const r = await this.build(source, { parent: genId, origin: 'agent' });
    // On success the proposer has been replaced. On failure, tell it (if it is still active).
    if (!r.ok && this.active?.id === genId) this.send(EV.BUILD_FAILED, r.error);
  }

  // ---- LLM with kernel-enforced rate limits and pause

  systemPrompt(genId, source) {
    return `You are the reasoning core of zAgent, a self-modifying autonomous agent. Its body is one Zig 0.16 source file
compiled to WebAssembly inside the user's browser. The running program sends you prompts through host_llm and
parses your reply itself.

Replying:
- Plain text is passed to the program, which normally shows it to the user.
- To change the program, include exactly ONE \`\`\`zig fenced block containing the COMPLETE new agent.zig. The kernel
  compiles it, self-tests it, and hot-swaps it in. Diffs, partial files, or placeholders like "..." will fail.
- Preserve the ABI exactly, keep state loading backward compatible, keep existing commands working unless asked,
  and keep each change small enough that it compiles on the first try.

${this.abi}

CURRENT agent.zig (generation ${genId}):
\`\`\`zig
${source}\`\`\``;
  }

  #llm(genId, prompt) {
    const source = this.active.source;
    this.llmChain = this.llmChain.then(async () => {
      const deliver = (kind, text) => { if (this.active?.id === genId) this.send(kind, text); };
      if (this.paused) return deliver(EV.LLM_ERROR, 'kernel is paused');
      const now = Date.now();
      this.llmLog = this.llmLog.filter(t => now - t < 3600_000);
      if (this.llmLog.length >= this.settings.maxCallsPerHour) return deliver(EV.LLM_ERROR, `rate limit: ${this.settings.maxCallsPerHour} calls/hour reached`);
      const wait = (this.llmLog.at(-1) ?? 0) + this.settings.minGapMs - now;
      if (wait > 0) await new Promise(r => setTimeout(r, wait));
      this.llmLog.push(Date.now());
      this.stats.llmCalls++;
      const provider = PROVIDERS[this.settings.provider] ?? PROVIDERS.demo;
      this.llmBusy = true; this.changed();
      this.line(`→ ${provider.label}: ${prompt.split('\n')[0].slice(0, 100)}`, 'llm');
      try {
        this.emit('stream', { start: true, gen: genId });
        const onText = t => this.emit('stream', { text: t, gen: genId });
        const text = await provider.complete({ system: this.systemPrompt(genId, source), prompt, source, settings: this.settings, onText });
        this.line(`← ${text.length} chars`, 'llm');
        deliver(EV.LLM_REPLY, text);
      } catch (e) {
        deliver(EV.LLM_ERROR, String(e?.message || e));
      } finally { this.llmBusy = false; this.emit('stream', { end: true }); this.changed(); }
    });
  }

  // ---- human controls

  userMessage(text) { this.send(EV.USER, text); }

  setPaused(p) {
    this.paused = p;
    this.line(p ? 'Kernel PAUSED: timers held, LLM calls refused' : 'Kernel resumed', 'tool');
    if (!p && this.timerHeld) { this.timerHeld = false; this.send(EV.TIMER); }
    this.changed();
  }

  async rollback() {
    const parent = this.active && this.gen(this.gen(this.active.id)?.parent);
    if (!parent) return this.line('Nothing to roll back to', 'error');
    this.stats.rollbacks++;
    await this.activate(parent.id);
  }

  async compileHuman(source) {
    const r = await this.build(source, { parent: this.active?.id ?? null, origin: 'human' });
    if (!r.ok) this.line('Your edit did not build; the running generation is unchanged.', 'error');
  }

  async reset() {
    this.#stopRuntime();
    await this.store.wipe();
    try { localStorage.removeItem(SETTINGS_KEY); } catch {}
  }
}
