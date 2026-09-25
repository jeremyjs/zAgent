# AGENTS.md — zAgent

Experimental in-browser, self-modifying Zig agent. Read README.md for the architecture and
ABI.md for the kernel ↔ agent contract.

## Contract

- The **kernel** (`index.html`, `src/`) is immutable to the agent. Guardrails live there:
  sandbox self-test, watchdog, rollback, rate limits, pause. Don't move a guardrail into
  agent-controlled code.
- The **agent** (`seed/agent.zig` and every generation after it) may only talk to the world
  through the imports in ABI.md. Changing the ABI means changing ABI.md (it is fed to the LLM
  verbatim), `src/host.js`, and the seed together.
- The seed must compile with the in-browser compiler (Zig 0.16, self-hosted wasm backend,
  Debug). `npm test` proves it does.
- `engine/engine.wasm` is a committed build artifact. Rebuild it (`./run` does this when
  `engine.zig` is newer) and commit both files together.

## Verify before claiming done

- `npm test`: seed compiles, boots, persists state, self-modifies, repairs, runs autonomy.
- Engine changes: `node tools/llm-node.mjs --logits` and compare against transformers
  (top-5 tokens must match), then `node tools/llm-node.mjs "Name three colors."`.
- Kernel/UI changes: `./run`, then check the browser shows self-bootstrap → g0 active, and a
  `Scripted mutator` "evolve" produces g1.

## Style

Plain ES modules, no build step, no framework. Strict about errors at boundaries (worker
messages, wasm calls, fetches). Zig: 0.16 idioms, `std.heap.wasm_allocator` in wasm.
