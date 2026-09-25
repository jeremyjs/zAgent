// Scripted offline "LLM" so the full self-modification loop runs with no API key.
// It performs real, compilable mutations of the agent source (bumps `mutation`).

export function mutateSource(src) {
  const re = /const mutation: u32 = (\d+);/;
  const m = src.match(re);
  if (!m) return src;
  const n = Number(m[1]) + 1;
  return src.replace(re, `const mutation: u32 = ${n};`);
}

const fence = src => '```zig\n' + (src.endsWith('\n') ? src : src + '\n') + '```';

export const demo = {
  id: 'demo',
  label: 'Scripted mutator (test harness)',
  async complete({ prompt, source }) {
    await new Promise(r => setTimeout(r, 400));
    if (/FAILED TO BUILD/.test(prompt)) return `Reverting to a known-good body.\n${fence(mutateSource(source))}`;
    if (/AUTONOMOUS STEP/.test(prompt)) return `Demo step: bumped mutation counter.\n${fence(mutateSource(source))}`;
    const user = (prompt.match(/USER MESSAGE:\n([\s\S]*?)\n\n/) || [])[1] ?? prompt;
    if (/\bbreak\b/i.test(user)) return `Deliberately broken build (to test repair).\n\`\`\`zig\nconst std = @import("std");\nexport fn boot( void {}\n\`\`\``;
    if (/\b(change|modify|evolve|improve|rewrite|mutate)\b/i.test(user)) return `Rewrote myself: mutation +1.\n${fence(mutateSource(source))}`;
    return `(demo model) You said: "${user.trim()}". Pick a real model in Settings for real answers. Try "evolve", "break", or /auto on.`;
  },
};
