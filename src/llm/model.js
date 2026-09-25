// LocalModel: drives engine.wasm (Zig) with a BPE tokenizer. Environment-agnostic
// (browser worker or Node). Keeps the KV cache between calls and reuses the longest
// common token prefix, so a stable system prompt is only prefilled once.
import { Tokenizer, chatML } from './tokenizer.js';

const TENSORS = { embed: 0, final_norm: 1, att_norm: 2, wq: 3, wk: 4, wv: 5, wo: 6, ffn_norm: 7, w_gate: 8, w_down: 9, w_up: 10 };
const HF_NAMES = [
  [/^model\.embed_tokens\.weight$/, 'embed'],
  [/^model\.norm\.weight$/, 'final_norm'],
  [/^model\.layers\.(\d+)\.input_layernorm\.weight$/, 'att_norm'],
  [/^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$/, 'wq'],
  [/^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$/, 'wk'],
  [/^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$/, 'wv'],
  [/^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$/, 'wo'],
  [/^model\.layers\.(\d+)\.post_attention_layernorm\.weight$/, 'ffn_norm'],
  [/^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$/, 'w_gate'],
  [/^model\.layers\.(\d+)\.mlp\.down_proj\.weight$/, 'w_down'],
  [/^model\.layers\.(\d+)\.mlp\.up_proj\.weight$/, 'w_up'],
];

export function parseSafetensorsHeader(bytes) {
  const n = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  return { header, dataStart: 8 + n };
}

export class LocalModel {
  static async create(engineBytes, { config, tokenizer, seqLen = 2048 }) {
    const { instance } = await WebAssembly.instantiate(engineBytes, {});
    const m = new LocalModel(instance.exports, config, new Tokenizer(tokenizer), seqLen);
    const rc = m.x.configure(config.hidden_size, config.intermediate_size, config.num_hidden_layers,
      config.num_attention_heads, config.num_key_value_heads, config.vocab_size, seqLen, config.rope_theta, config.rms_norm_eps);
    if (rc !== 0) throw new Error(`engine configure failed (${rc})`);
    return m;
  }

  constructor(exports, config, tok, seqLen) {
    Object.assign(this, { x: exports, config, tok, seqLen });
    this.cached = [];          // tokens currently in the KV cache
    this.eos = new Set([tok.token('<|im_end|>'), tok.token('<|endoftext|>')]);
  }

  get mem() { return this.x.memory.buffer; }

  /** Quantize bf16 safetensors (Uint8Array of the whole file) into the engine. */
  loadSafetensors(file, onProgress) {
    const { header, dataStart } = parseSafetensorsHeader(file);
    const entries = Object.entries(header).filter(([k]) => k !== '__metadata__');
    let done = 0;
    for (const [name, info] of entries) {
      const hit = HF_NAMES.map(([re, kind]) => [name.match(re), kind]).find(([mm]) => mm);
      if (!hit) continue;
      if (info.dtype !== 'BF16') throw new Error(`${name}: dtype ${info.dtype} unsupported (need BF16)`);
      const [mm, kind] = hit;
      const [a, b] = info.data_offsets;
      const src = file.subarray(dataStart + a, dataStart + b);
      const p = this.x.alloc(src.length) >>> 0;
      if (!p) throw new Error('engine out of memory');
      new Uint8Array(this.mem, p, src.length).set(src);
      const rc = this.x.load_tensor(TENSORS[kind], mm[1] ? Number(mm[1]) : 0, p, src.length / 2);
      this.x.free(p, src.length);
      if (rc !== 0) throw new Error(`load_tensor ${name} failed (${rc})`);
      onProgress?.(++done / entries.length);
    }
  }

  snapshot() { return new Uint8Array(this.mem, this.x.weights_ptr() >>> 0, this.x.weights_len() >>> 0).slice(); }

  restore(bytes) {
    if (bytes.length !== this.x.weights_len() >>> 0) throw new Error('snapshot size mismatch');
    new Uint8Array(this.mem, this.x.weights_ptr() >>> 0, bytes.length).set(bytes);
  }

  /** Raw logits after feeding `ids` from scratch (for tests). */
  logitsFor(ids) {
    ids.forEach((t, i) => this.x.forward(t, i));
    this.cached = [];
    return new Float32Array(this.mem, this.x.logits_ptr() >>> 0, this.config.vocab_size).slice();
  }

  /**
   * Chat completion. Streams text through onText; returns { text, promptTokens, newTokens, prefillMs, genMs }.
   */
  async generate(messages, { maxTokens = 512, temperature = 0.3, topP = 0.9, seed = Date.now() >>> 0, onText, signal, yieldEvery = 8 } = {}) {
    let ids = this.tok.encode(chatML(messages));
    const budget = this.seqLen - maxTokens;
    if (ids.length > budget) ids = [...ids.slice(0, 16), ...ids.slice(ids.length - budget + 16)]; // keep head (system start) + tail

    // Reuse the KV cache for the shared prefix.
    let keep = 0;
    while (keep < ids.length - 1 && keep < this.cached.length && this.cached[keep] === ids[keep]) keep++;
    const t0 = performance.now();
    for (let i = keep; i < ids.length; i++) {
      this.x.forward(ids[i], i);
      if ((i - keep) % 32 === 31) { await pause(); if (signal?.aborted) throw new Error('aborted'); }
    }
    const prefillMs = performance.now() - t0;
    this.cached = ids.slice();

    this.x.set_seed(seed);
    const out = [];
    let emitted = 0, text = '';
    const dec = new TextDecoder();
    const t1 = performance.now();
    for (let n = 0; n < maxTokens; n++) {
      const next = this.x.sample(temperature, topP);
      if (this.eos.has(next)) break;
      out.push(next);
      const pos = ids.length + out.length - 1;
      if (pos >= this.seqLen) break;
      this.x.forward(next, pos);
      this.cached.push(next);
      const chunk = dec.decode(this.tok.decodeBytes(out.slice(emitted)), { stream: true });
      emitted = out.length;
      if (chunk) { text += chunk; onText?.(chunk, out.length); }
      if (n % yieldEvery === yieldEvery - 1) { await pause(); if (signal?.aborted) break; }
    }
    text += dec.decode();
    return { text, promptTokens: ids.length, reused: keep, newTokens: out.length, prefillMs, genMs: performance.now() - t1 };
  }
}

const pause = () => new Promise(r => setTimeout(r, 0));
