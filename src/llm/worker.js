// Local inference worker: engine.wasm (Zig) + BPE tokenizer.
// Weight sources, in order: OPFS int8 snapshot -> same-origin ./models/<name>/ -> Hugging Face CDN.
import { LocalModel } from './model.js';
import { MODELS } from './catalog.js';

const ENGINE_URL = new URL('../../engine/engine.wasm', import.meta.url);
const SEQ_LEN = 2048;
let model = null, modelId = null, loading = null;
const post = (type, extra) => postMessage({ type, ...extra });

async function opfsDir(id, create) {
  const root = await navigator.storage.getDirectory();
  const d = await root.getDirectoryHandle('zagent-models', { create });
  return d.getDirectoryHandle(id, { create });
}
async function opfsRead(id, name) {
  try { return new Uint8Array(await (await (await (await opfsDir(id, false)).getFileHandle(name)).getFile()).arrayBuffer()); }
  catch { return null; }
}
async function opfsWrite(id, name, bytes) {
  const h = await (await opfsDir(id, true)).getFileHandle(name, { create: true });
  const w = await h.createWritable(); await w.write(bytes); await w.close();
}

async function fetchBytes(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  const reader = r.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    post('progress', { label, got, total });
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function pickBase(repo) {
  const local = new URL(`../../models/${repo.split('/')[1]}/`, import.meta.url).href;
  try { if ((await fetch(local + 'config.json', { method: 'HEAD' })).ok) return local; } catch {}
  return `https://huggingface.co/${repo}/resolve/main/`;
}

async function load(id) {
  if (model && modelId === id) return;
  const spec = MODELS[id];
  if (!spec) throw new Error('unknown model ' + id);
  const t0 = performance.now();
  const engine = await (await fetch(ENGINE_URL)).arrayBuffer();
  const dec = new TextDecoder();
  let config = await opfsRead(id, 'config.json'), tokenizer = await opfsRead(id, 'tokenizer.json');
  const snapshot = config && tokenizer ? await opfsRead(id, 'weights.q8') : null;
  let base = null;
  if (!snapshot) {
    base = await pickBase(spec.repo);
    post('status', { text: `downloading from ${base.includes('huggingface.co') ? 'Hugging Face' : 'this server'}` });
    config = await fetchBytes(base + 'config.json', 'config');
    tokenizer = await fetchBytes(base + 'tokenizer.json', 'tokenizer');
  }
  const m = await LocalModel.create(engine, { config: JSON.parse(dec.decode(config)), tokenizer: JSON.parse(dec.decode(tokenizer)), seqLen: SEQ_LEN });
  if (snapshot) {
    post('status', { text: 'restoring int8 weights from OPFS' });
    m.restore(snapshot);
  } else {
    const st = await fetchBytes(base + 'model.safetensors', 'weights');
    post('status', { text: 'quantizing to int8' });
    m.loadSafetensors(st);
    post('status', { text: 'caching int8 weights in OPFS' });
    try {
      await opfsWrite(id, 'weights.q8', m.snapshot());
      await opfsWrite(id, 'config.json', config);
      await opfsWrite(id, 'tokenizer.json', tokenizer);
    } catch (e) { post('status', { text: 'OPFS cache failed: ' + e.message }); }
  }
  model = m; modelId = id;
  post('ready', { id, ms: performance.now() - t0, cached: !!snapshot, bytes: m.x.weights_len() >>> 0 });
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load') {
      loading = load(m.model).finally(() => { loading = null; });
      await loading;
    } else if (m.type === 'generate') {
      if (loading) await loading;
      await load(m.model);
      const r = await model.generate(m.messages, { ...m.options, onText: (t, n) => post('token', { id: m.id, text: t, n }) });
      post('result', { id: m.id, ...r });
    }
  } catch (e) {
    post(m.type === 'generate' ? 'result' : 'error', { id: m.id, error: String(e?.message || e) });
  }
};
