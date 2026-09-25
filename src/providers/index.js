import { demo } from './demo.js';
import { anthropic } from './anthropic.js';
import { openaiCompatible } from './openai.js';
import { local } from './local.js';

export const PROVIDERS = { local, anthropic, openai: openaiCompatible, demo };
