// Hosts one agent generation. Every host_* call is forwarded to the kernel.
// Also used as the throwaway sandbox for candidate self-tests.
import { instantiateAgent } from './host.js';

let agent = null, llmSeq = 0;
const call = (name, arg, extra) => postMessage({ type: 'call', name, arg, ...extra });
const host = {
  log: s => call('log', s),
  say: s => call('say', s),
  llm: p => { const id = ++llmSeq; call('llm', p, { id }); return id; },
  propose: s => call('propose', s),
  save: b => call('save', b),
  timer: ms => call('timer', ms),
};
const fail = (seq, err) => postMessage({ type: 'trap', seq, error: String(err?.stack || err) });

onmessage = async ({ data: m }) => {
  if (m.type === 'boot') {
    try {
      agent = await instantiateAgent(m.bytes, host);
      agent.boot(m.state);
      postMessage({ type: 'done', seq: m.seq, exports: agent.exports });
    } catch (e) { fail(m.seq, e); }
  } else if (m.type === 'event') {
    if (!agent) return fail(m.seq, 'no agent booted');
    try { agent.event(m.kind, m.text); postMessage({ type: 'done', seq: m.seq }); }
    catch (e) { fail(m.seq, e); }
  }
};
