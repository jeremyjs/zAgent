// Bundled local brain: the Zig micro inference engine running SmolLM2 in a worker.
import { MODELS, DEFAULT_MODEL } from '../llm/catalog.js';

const SYSTEM = `You are the on-device brain of zAgent, a small agent written in Zig that runs inside the user's web browser.
You are a tiny language model. Be brief, friendly, and honest.
You cannot write or edit Zig code reliably. If asked to change the agent's code, say in one sentence that code changes need a larger model, selected in the Brain panel.`;

class LocalEngine extends EventTarget {
  constructor() {
    super();
    this.state = 'not loaded';
    this.seq = 0;
    this.pending = new Map();
    this.worker = null;
  }
  #ensure() {
    if (this.worker) return;
    this.worker = new Worker(new URL('../llm/worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = ({ data: m }) => {
      if (m.type === 'status') this.#set(m.text);
      else if (m.type === 'progress') this.dispatchEvent(new CustomEvent('progress', { detail: m }));
      else if (m.type === 'ready') { this.loaded = m.id; this.#set(`ready · ${(m.bytes / 1e6).toFixed(0)} MB int8${m.cached ? ' · from OPFS' : ''}`); this.dispatchEvent(new CustomEvent('ready', { detail: m })); }
      else if (m.type === 'error') this.#set('failed: ' + m.error);
      else if (m.type === 'token') this.pending.get(m.id)?.onText?.(m.text, m.n);
      else if (m.type === 'result') {
        const p = this.pending.get(m.id); this.pending.delete(m.id);
        if (!p) return;
        if (m.error) p.reject(new Error(m.error));
        else {
          this.speed = `${(m.newTokens / m.genMs * 1000).toFixed(1)} tok/s · prefill ${m.promptTokens - m.reused} tok in ${(m.prefillMs / 1000).toFixed(1)}s`;
          this.dispatchEvent(new CustomEvent('change'));
          p.resolve(m);
        }
      }
    };
  }
  #set(s) { this.state = s; this.dispatchEvent(new CustomEvent('change')); }
  load(model = DEFAULT_MODEL) { this.#ensure(); this.#set('loading'); this.worker.postMessage({ type: 'load', model }); }
  generate(model, messages, options, onText) {
    this.#ensure();
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onText });
      this.worker.postMessage({ type: 'generate', id, model, messages, options });
    });
  }
}

export const engine = new LocalEngine();

export const local = {
  id: 'local',
  label: 'Local · Zig micro engine',
  compact: true,
  async complete({ prompt, settings, onText }) {
    // Prompt adapter: the agent writes prompts for code-capable models. A 135M model can't
    // produce compiling Zig, so self-modification requests are declined here instead of
    // burning CPU on doomed builds, and chat gets only the user's words.
    if (/^AUTONOMOUS STEP/m.test(prompt)) return 'WAIT';
    if (/FAILED TO BUILD/.test(prompt)) return 'The local model cannot repair Zig code. Switch the brain to Claude or an OpenAI-compatible endpoint for self-modification.';
    const user = prompt.match(/USER MESSAGE:\n([\s\S]*?)\n\n(?:Answer|Reply)/)?.[1] ?? prompt;
    const model = MODELS[settings.localModel] ? settings.localModel : DEFAULT_MODEL;
    const r = await engine.generate(model, [{ role: 'system', content: SYSTEM }, { role: 'user', content: user.slice(-6000) }],
      { maxTokens: 256, temperature: 0.2, topP: 0.85 }, onText);
    return r.text.trim();
  },
};
