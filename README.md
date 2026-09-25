# zAgent

An experimental self-bootstrapping, self-modifying agent that lives entirely in a browser tab.
Its body is one Zig file. It compiles itself with Zig running as WebAssembly, hot-swaps into
each new build, and thinks with a small LLM engine, also written in Zig, that runs on-device.

```
./run            # build what changed, serve http://127.0.0.1:8787, open the browser
./run --tunnel   # same, plus a public https://….trycloudflare.com URL via cloudflared
```

Live: https://zagent.raspy-thunder-20d2.workers.dev (deploy with `npm run deploy`: a static-assets-only
Worker, see `wrangler.jsonc` / `.assetsignore`; model weights come from Hugging Face).

Needs Zig 0.16 (`mise install` picks it up from `mise.toml`). Node is only for tests.

## How it works

```
index.html + src/          immutable kernel (JS): storage, compiler, LLM, guardrails, UI
 ├─ zig-out/               Zig's wasm32-wasi self-hosted compiler + stdlib (≈7 MB)
 ├─ seed/agent.zig         generation 0 of the agent
 ├─ engine/engine.zig      micro inference engine → engine/engine.wasm (27 KB)
 └─ OPFS                   generations g0, g1, …, agent state, cached int8 weights
```

1. **Self-bootstrap.** On first load there is no agent binary. The kernel fetches
   `seed/agent.zig`, compiles it in a worker with `zig.wasm` (about 1 s), self-tests it in a
   sandbox worker, stores it as `g0` in OPFS, and boots it.
2. **The agent drives.** The Zig program receives events (user message, LLM reply, timer,
   build failure) and acts only through the host ABI: `host_say`, `host_llm`, `host_propose`,
   `host_save`, `host_timer`, `host_log`. See [ABI.md](ABI.md). It builds its own prompts and
   parses its own replies.
3. **Self-modification.** When the LLM returns a new `agent.zig`, the agent calls
   `host_propose`. The kernel compiles it, boots it in a throwaway sandbox with the real state,
   sends it `selftest` and `/status`, and only then hot-swaps it in. Compile errors go back to
   the agent, which asks the LLM for a fix (up to 3 repairs).
4. **Autonomy.** `/auto on` makes the agent take a self-improvement step on a timer, toward its
   `/goal`.
5. **Guardrails the agent cannot touch:** a sandbox gate, a 5 s watchdog per call, automatic
   rollback to the parent generation on any trap, a rate-limited LLM, a Pause kill switch, and
   manual rollback / activate for any generation. `?recovery` rebuilds from the seed.

## Brains

| Provider | What it is |
|---|---|
| **Local · Zig micro engine** (default) | `engine/engine.zig`: llama-architecture transformer, int8 weights and activations, SIMD128, GQA, RoPE, SwiGLU, KV-cache prefix reuse, top-k/top-p sampling. Runs SmolLM2 135M (≈60 tok/s) or 360M. Weights load from `./models/` if present, else Hugging Face, get quantized in-engine, then cached in OPFS (reloads in ~0.3 s). |
| Claude (Anthropic API) | Official SDK from the browser, default `claude-opus-5`. Needed for real self-modification. |
| OpenAI-compatible | Ollama, LM Studio, llama.cpp, OpenRouter… |
| Scripted mutator | Not a model. Deterministic replies that really rewrite and rebuild the agent, for exercising the loop. |

The local model is for offline chat. At 135M parameters it can't write compiling Zig, so the
local provider declines autonomous steps and repairs rather than burning CPU on doomed builds.

Engine output was checked against PyTorch/transformers: identical token IDs, identical top-5
next tokens, logits within int8 quantization noise.

## Dev

```
npm test                          # e2e: real compiler + ABI + seed, scripted LLM, generation swaps
node tools/llm-node.mjs "prompt"  # run the engine in Node (needs tools/fetch-model.sh)
tools/fetch-model.sh [360M]       # cache SmolLM2 weights under models/ (gitignored)
tools/fetch-compiler.sh           # refresh zig-out/ from the zigtools playground build
```

`tools/serve.zig` is the dev server (Zig 0.16 `std.Io` + `std.http`, streams files, correct
wasm MIME type, no path traversal).
