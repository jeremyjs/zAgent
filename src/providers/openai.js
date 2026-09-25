// Any OpenAI-compatible /chat/completions endpoint: Ollama, LM Studio, llama.cpp, OpenRouter, vLLM...
export const openaiCompatible = {
  id: 'openai',
  label: 'OpenAI-compatible endpoint',
  defaultModel: '',
  async complete({ system, prompt, settings, signal }) {
    const endpoint = settings.endpoint || 'http://localhost:11434/v1/chat/completions';
    if (!settings.model) throw new Error('Set a model name for the OpenAI-compatible endpoint (Settings)');
    const r = await fetch(endpoint, {
      method: 'POST',
      signal,
      headers: { 'content-type': 'application/json', ...(settings.apiKey ? { authorization: `Bearer ${settings.apiKey}` } : {}) },
      body: JSON.stringify({ model: settings.model, messages: [{ role: 'system', content: system }, { role: 'user', content: prompt }] }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
    const j = await r.json();
    return j?.choices?.[0]?.message?.content ?? '';
  },
};
