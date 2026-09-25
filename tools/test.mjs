// End-to-end headless test of the self-modification loop, with a scripted LLM.
// Exercises the real compiler (zig.wasm), the real host ABI, and the seed agent.
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { ZigCompiler } from '../src/zigc.js';
import { instantiateAgent, EV } from '../src/host.js';
import { mutateSource } from '../src/providers/demo.js';

const root = resolve(import.meta.dirname, '..');
const z = join(root, 'zig-out');
const zc = await ZigCompiler.load({
  zigWasm: await readFile(join(z, 'bin/zig.wasm')),
  zigTar: new Uint8Array(await readFile(join(z, 'zig.tar.gz'))),
  compilerRt: new Uint8Array(await readFile(join(z, 'libcompiler_rt.a'))),
});

const said = [], llmCalls = [], proposals = [], timers = [];
let state = new Uint8Array();
const host = {
  log: s => console.log('  [log]', s),
  say: s => { said.push(s); console.log('  [say]', s.split('\n')[0]); },
  llm: p => { llmCalls.push(p); return llmCalls.length; },
  propose: src => proposals.push(src),
  save: b => { state = b; },
  timer: ms => timers.push(ms),
};

async function build(src) {
  const r = await zc.compile(src);
  if (!r.ok) return { ok: false, err: r.stderr };
  try {
    const sandbox = await instantiateAgent(r.wasm, { ...host, say() {}, llm: () => 0, propose() {}, save() {}, timer() {} });
    sandbox.boot(state);
    sandbox.event(EV.SELFTEST, '');
    sandbox.event(EV.USER, '/status');
    return { ok: true, wasm: r.wasm };
  } catch (e) { return { ok: false, err: 'self-test: ' + e.message }; }
}

let step = 0;
const check = (name, fn) => fn().then(() => console.log(`ok ${++step} - ${name}`));

const seedSrc = await readFile(join(root, 'seed/agent.zig'), 'utf8');
let agent, currentSrc = seedSrc;

await check('seed compiles and boots', async () => {
  const b = await build(seedSrc);
  assert.ok(b.ok, b.err);
  agent = await instantiateAgent(b.wasm, host);
  agent.boot(state);
  assert.match(said.at(-1), /mutation 0/);
});

await check('commands persist state', async () => {
  agent.event(EV.USER, '/goal learn to count');
  assert.match(new TextDecoder().decode(state), /goal=learn to count/);
  agent.event(EV.USER, '/status');
  assert.match(said.at(-1), /learn to count/);
});

await check('chat goes to the LLM and reply is shown', async () => {
  agent.event(EV.USER, 'hello');
  assert.match(llmCalls.at(-1), /USER MESSAGE:\nhello/);
  agent.event(EV.LLM_REPLY, 'Hi there.');
  assert.equal(said.at(-1), 'Hi there.');
});

await check('self-modification: proposal compiles, swaps, keeps state', async () => {
  agent.event(EV.USER, 'change yourself');
  const reply = 'Bumped mutation.\n```zig\n' + mutateSource(currentSrc) + '```\n';
  agent.event(EV.LLM_REPLY, reply);
  assert.equal(proposals.length, 1);
  const b = await build(proposals[0]);
  assert.ok(b.ok, b.err);
  currentSrc = proposals[0];
  agent = await instantiateAgent(b.wasm, host);
  agent.boot(state);
  assert.match(said.at(-1), /mutation 1/);
  agent.event(EV.USER, '/status');
  assert.match(said.at(-1), /learn to count/, 'goal survived the generation swap');
});

await check('broken proposal is reported back and repair is requested', async () => {
  agent.event(EV.USER, 'break');
  agent.event(EV.LLM_REPLY, '```zig\nconst x = ;\n```');
  const b = await build(proposals.at(-1));
  assert.equal(b.ok, false);
  agent.event(EV.BUILD_FAILED, b.err);
  assert.match(llmCalls.at(-1), /FAILED TO BUILD/);
  assert.match(llmCalls.at(-1), /const x = ;/);
});

await check('autonomy: timer triggers an autonomous step', async () => {
  agent.event(EV.LLM_REPLY, 'WAIT'); // clear busy
  agent.event(EV.USER, '/auto on');
  agent.event(EV.TIMER, '');
  assert.match(llmCalls.at(-1), /AUTONOMOUS STEP 1/);
});

console.log(`\nall ${step} checks passed`);
