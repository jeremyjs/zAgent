import { Kernel } from './kernel.js';
import { PROVIDERS } from './providers/index.js';
import { engine } from './providers/local.js';
import { MODELS } from './llm/catalog.js';

const $ = id => document.getElementById(id);
const logEl = $('log');
const kernel = new Kernel({ recovery: new URLSearchParams(location.search).has('recovery') });
window.zagent = kernel; // for poking from devtools

// ---- log

function line(text, cls = 'sys', gen) {
  const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 40;
  const d = document.createElement('div');
  d.className = cls;
  if (gen) d.dataset.gen = gen;
  d.textContent = text;
  logEl.appendChild(d);
  if (stick) logEl.scrollTop = logEl.scrollHeight;
  return d;
}
kernel.addEventListener('line', e => line(e.detail.text, e.detail.cls));
kernel.addEventListener('say', e => { streamEl?.remove(); streamEl = null; line(e.detail.text, 'agent', e.detail.gen); });

let streamEl = null;
kernel.addEventListener('stream', e => {
  const d = e.detail;
  if (d.start) { streamEl = line('', 'llm'); streamEl.textContent = '… '; }
  else if (d.text && streamEl) { streamEl.textContent += d.text; logEl.scrollTop = logEl.scrollHeight; }
  else if (d.end && streamEl) { streamEl.remove(); streamEl = null; }
});

// ---- header + kernel card

function render() {
  const s = kernel.settings, a = kernel.active, g = a && kernel.gen(a.id);
  const busy = kernel.building || kernel.llmBusy;
  $('dot').className = 'dot ' + (!a ? 'bad' : busy ? 'busy' : 'ok');
  $('statusText').textContent = !a ? 'no active generation' : kernel.paused ? 'paused' : kernel.building ? 'building…' : kernel.llmBusy ? 'thinking…' : 'running';
  $('genBadge').textContent = a?.id ?? '–';
  $('modelBadge').textContent = s.provider === 'local' ? (MODELS[s.localModel]?.label ?? s.localModel) : s.provider === 'demo' ? 'scripted' : (s.model || PROVIDERS[s.provider]?.defaultModel || '?');
  $('kGen').textContent = a ? `${a.id} (${g?.origin ?? '?'})` : '–';
  const lineage = []; for (let x = g; x; x = kernel.gen(x.parent)) lineage.push(x.id);
  $('kLineage').textContent = lineage.join(' ← ') || '–';
  $('kCompiler').textContent = kernel.compilerState;
  $('kBuilds').textContent = `${kernel.stats.builds} (${kernel.stats.buildFailures} failed)`;
  $('kCalls').textContent = kernel.stats.llmCalls;
  $('kRollbacks').textContent = kernel.stats.rollbacks;
  $('pauseBtn').textContent = kernel.paused ? 'Resume' : 'Pause';
  $('pauseBtn').classList.toggle('on', kernel.paused);

  const gens = $('gens'); gens.textContent = '';
  for (const x of [...kernel.manifest.generations].reverse()) {
    const row = document.createElement('div');
    row.className = 'gen' + (x.id === a?.id ? ' active' : '') + (x.status === 'crashed' ? ' crashed' : '');
    row.innerHTML = `<b></b><span class="st"></span>`;
    row.querySelector('b').textContent = x.id;
    row.querySelector('.st').textContent = `${x.origin}${x.parent ? ' ← ' + x.parent : ''} · ${(x.size / 1024).toFixed(0)} KB${x.status === 'crashed' ? ' · crashed' : ''}`;
    if (x.id !== a?.id) {
      const b = document.createElement('button'); b.textContent = 'activate';
      b.onclick = () => kernel.activate(x.id).catch(err => line('Activate failed: ' + err.message, 'error'));
      row.appendChild(b);
    }
    gens.appendChild(row);
  }
  if (a && a.id !== shownSourceGen) { $('source').value = a.source; shownSourceGen = a.id; }
  renderBrain();
}
let shownSourceGen = null;
kernel.addEventListener('change', render);

// ---- compiler download progress

const dl = $('dl'), dlText = $('dlText');
const progress = new Map();
function showProgress(key, got, total) {
  progress.set(key, { got, total });
  let g = 0, t = 0; for (const p of progress.values()) { g += p.got; t += p.total; }
  dl.hidden = dlText.hidden = g >= t;
  dl.max = t; dl.value = g;
  dlText.textContent = `downloading ${(g / 1e6).toFixed(1)} / ${(t / 1e6).toFixed(1)} MB`;
}
kernel.addEventListener('progress', e => showProgress('c:' + e.detail.label, e.detail.got, e.detail.total));
engine.addEventListener('progress', e => showProgress('m:' + e.detail.label, e.detail.got, e.detail.total));

// ---- brain settings

const provSel = $('provider');
for (const [id, p] of Object.entries(PROVIDERS)) provSel.add(new Option(p.label, id));
for (const [id, m] of Object.entries(MODELS)) $('localModel').add(new Option(`${m.label} · ${m.download}`, id));

function renderBrain() {
  const s = kernel.settings;
  provSel.value = s.provider;
  $('localModel').value = s.localModel;
  $('localBox').hidden = s.provider !== 'local';
  $('remoteBox').hidden = !['anthropic', 'openai'].includes(s.provider);
  $('endpoint').hidden = s.provider !== 'openai';
  $('effort').hidden = s.provider !== 'anthropic';
  $('engState').textContent = engine.state;
  $('engSpeed').textContent = engine.speed ?? '–';
  $('loadEngine').disabled = engine.state === 'loading' || engine.loaded === s.localModel;
  for (const [id, key] of [['model', 'model'], ['endpoint', 'endpoint'], ['apiKey', 'apiKey'], ['effort', 'effort']]) {
    if (document.activeElement !== $(id)) $(id).value = s[key] ?? '';
  }
  $('model').placeholder = PROVIDERS[s.provider]?.defaultModel || 'model name (e.g. qwen2.5-coder:7b)';
  $('rememberKey').checked = s.rememberKey;
  $('brainNote').textContent = {
    local: 'Runs fully on-device: Zig engine (27 KB wasm, int8 SIMD) + SmolLM2. Weights come from ./models/ or Hugging Face once, then OPFS. Chats offline; rewriting its own Zig needs a code-capable model.',
    anthropic: 'Calls the Anthropic API directly from this page. The key never leaves this browser except to api.anthropic.com.',
    openai: 'Any OpenAI-compatible /chat/completions endpoint, e.g. Ollama at http://localhost:11434/v1/chat/completions.',
    demo: 'Not a model: deterministic replies that really rewrite and rebuild the agent. For testing the self-modification loop.',
  }[s.provider] ?? '';
}
engine.addEventListener('change', renderBrain);
engine.addEventListener('ready', e => line(`Local model ready in ${(e.detail.ms / 1000).toFixed(1)}s${e.detail.cached ? ' (OPFS cache)' : ''}`, 'tool'));

provSel.onchange = () => kernel.updateSettings({ provider: provSel.value });
$('localModel').onchange = () => kernel.updateSettings({ localModel: $('localModel').value });
$('loadEngine').onclick = () => engine.load(kernel.settings.localModel);
for (const id of ['model', 'endpoint', 'apiKey', 'effort']) $(id).onchange = () => kernel.updateSettings({ [id]: $(id).value.trim() });
$('rememberKey').onchange = () => kernel.updateSettings({ rememberKey: $('rememberKey').checked });

// ---- controls

$('composer').onsubmit = e => {
  e.preventDefault();
  const text = $('msg').value.trim();
  if (!text) return;
  $('msg').value = '';
  line('you> ' + text, 'you');
  if (!kernel.active) return line('No active generation.', 'error');
  kernel.userMessage(text);
};
$('pauseBtn').onclick = () => kernel.setPaused(!kernel.paused);
$('rollbackBtn').onclick = () => kernel.rollback().catch(err => line('Rollback failed: ' + err.message, 'error'));
$('compileBtn').onclick = () => kernel.compileHuman($('source').value);
$('revertBtn').onclick = () => { if (kernel.active) $('source').value = kernel.active.source; };
$('resetBtn').onclick = async () => {
  if (!confirm('Delete all generations, agent state and settings for this site? (Cached model weights are kept.)')) return;
  await kernel.reset();
  location.reload();
};

// ---- boot

render();
kernel.boot()
  .then(() => { if (kernel.settings.provider === 'local') engine.load(kernel.settings.localModel); })
  .catch(err => { line('BOOT ERROR: ' + (err?.stack || err), 'error'); render(); });
