// Models the bundled engine can run (llama architecture, BF16 safetensors).
export const MODELS = {
  'smollm2-135m': { repo: 'HuggingFaceTB/SmolLM2-135M-Instruct', label: 'SmolLM2 135M Instruct', download: '269 MB', ram: '~250 MB' },
  'smollm2-360m': { repo: 'HuggingFaceTB/SmolLM2-360M-Instruct', label: 'SmolLM2 360M Instruct', download: '724 MB', ram: '~520 MB' },
};
export const DEFAULT_MODEL = 'smollm2-135m';
