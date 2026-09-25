// Run the local engine in Node against models/<name>. Usage: node tools/llm-node.mjs [prompt] [--logits]
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { LocalModel } from '../src/llm/model.js';

const root = resolve(import.meta.dirname, '..');
const dir = join(root, 'models', process.env.MODEL ?? 'SmolLM2-135M-Instruct');
const args = process.argv.slice(2);
const t0 = performance.now();
const model = await LocalModel.create(await readFile(join(root, 'engine/engine.wasm')), {
  config: JSON.parse(await readFile(join(dir, 'config.json'), 'utf8')),
  tokenizer: JSON.parse(await readFile(join(dir, 'tokenizer.json'), 'utf8')),
  seqLen: 1024,
});
model.loadSafetensors(new Uint8Array(await readFile(join(dir, 'model.safetensors'))));
console.error(`loaded + quantized in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

if (args.includes('--logits')) {
  const ids = model.tok.encode('The capital of France is');
  console.log(JSON.stringify({ ids, logits: Array.from(model.logitsFor(ids).slice(0, 8)), top: Array.from(model.logitsFor(ids)).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5) }));
  process.exit(0);
}
const prompt = args.find(a => !a.startsWith('--')) ?? 'Write one sentence about the ocean.';
const r = await model.generate([{ role: 'user', content: prompt }], { maxTokens: 120, temperature: 0, onText: t => process.stdout.write(t) });
console.error(`\n\nprompt ${r.promptTokens} tok in ${(r.prefillMs / 1000).toFixed(2)}s (${(r.promptTokens / r.prefillMs * 1000).toFixed(1)} tok/s) · gen ${r.newTokens} tok at ${(r.newTokens / r.genMs * 1000).toFixed(1)} tok/s`);
