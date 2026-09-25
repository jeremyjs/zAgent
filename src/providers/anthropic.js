// Claude via the official SDK, loaded on first use. The key stays in this browser.
const SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.128.0/+esm';
const FALLBACK_MODELS = new Set(['claude-opus-5', 'claude-fable-5-1']);
let Anthropic = null;

export const anthropic = {
  id: 'anthropic',
  label: 'Claude (Anthropic API)',
  defaultModel: 'claude-opus-5',
  async complete({ system, prompt, settings, signal }) {
    if (!settings.apiKey) throw new Error('No Anthropic API key set (Settings)');
    Anthropic ??= (await import(SDK_URL)).default;
    const client = new Anthropic({ apiKey: settings.apiKey, dangerouslyAllowBrowser: true });
    const model = settings.model || this.defaultModel;
    const params = {
      model,
      max_tokens: 32000,
      thinking: { type: 'adaptive' },
      output_config: { effort: settings.effort || 'high' },
      system,
      messages: [{ role: 'user', content: prompt }],
    };
    // Server-side refusal fallback, on the models that support it.
    const stream = FALLBACK_MODELS.has(model)
      ? client.beta.messages.stream({ ...params, betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' }, { signal })
      : client.messages.stream(params, { signal });
    const msg = await stream.finalMessage();
    if (msg.stop_reason === 'refusal') throw new Error(`model declined (${msg.stop_details?.category ?? 'refusal'})`);
    const text = msg.content.filter(b => b.type === 'text').map(b => b.text).join('');
    if (msg.stop_reason === 'max_tokens') throw new Error('reply truncated at max_tokens');
    return text;
  },
};
