// Byte-level BPE tokenizer (GPT-2 style) for HF tokenizer.json files, as used by SmolLM2.
// Pre-tokenization: Digits(individual) -> ByteLevel regex. Special tokens are matched verbatim.

const GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function byteMaps() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const enc = new Array(256), dec = new Map();
  bs.forEach((b, i) => { enc[b] = String.fromCodePoint(cs[i]); dec.set(String.fromCodePoint(cs[i]), b); });
  return { enc, dec };
}

export class Tokenizer {
  constructor(json) {
    const m = json.model;
    this.vocab = new Map(Object.entries(m.vocab));
    this.idToToken = [];
    for (const [t, id] of this.vocab) this.idToToken[id] = t;
    this.ranks = new Map(m.merges.map((mg, i) => [Array.isArray(mg) ? mg.join(' ') : mg, i]));
    this.special = new Map(json.added_tokens.filter(t => t.special).map(t => [t.content, t.id]));
    for (const [c, id] of this.special) this.idToToken[id] = c;
    this.specialRe = new RegExp([...this.special.keys()].map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
    const { enc, dec } = byteMaps();
    this.byteEnc = enc; this.byteDec = dec;
    this.cache = new Map();
    this.utf8 = new TextEncoder();
  }

  token(s) { return this.special.get(s) ?? this.vocab.get(s); }

  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = [...word];
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
    }
    const ids = parts.map(p => this.vocab.get(p));
    if (this.cache.size < 50000) this.cache.set(word, ids);
    return ids;
  }

  encodeOrdinary(text) {
    const out = [];
    // Digits(individual_digits) isolates every digit before the byte-level regex runs.
    for (const piece of text.split(/(\p{N})/u)) {
      if (!piece) continue;
      for (const m of piece.matchAll(GPT2_SPLIT)) {
        const mapped = Array.from(this.utf8.encode(m[0]), b => this.byteEnc[b]).join('');
        for (const id of this.bpe(mapped)) if (id !== undefined) out.push(id);
      }
    }
    return out;
  }

  /** Encode with special tokens (e.g. <|im_start|>) recognised verbatim. */
  encode(text) {
    const out = [];
    let last = 0;
    for (const m of text.matchAll(this.specialRe)) {
      out.push(...this.encodeOrdinary(text.slice(last, m.index)), this.special.get(m[0]));
      last = m.index + m[0].length;
    }
    out.push(...this.encodeOrdinary(text.slice(last)));
    return out;
  }

  /** Decode ids to raw bytes (so callers can stream UTF-8 safely). */
  decodeBytes(ids) {
    const bytes = [];
    for (const id of ids) {
      const t = this.idToToken[id] ?? '';
      if (this.special.has(t)) { bytes.push(...this.utf8.encode(t)); continue; }
      for (const ch of t) { const b = this.byteDec.get(ch); if (b !== undefined) bytes.push(b); }
    }
    return new Uint8Array(bytes);
  }

  decode(ids) { return new TextDecoder().decode(this.decodeBytes(ids)); }
}

/** ChatML, as in SmolLM2's chat_template. */
export function chatML(messages) {
  let s = '';
  if (messages[0]?.role !== 'system') s += '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n';
  for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  return s + '<|im_start|>assistant\n';
}
