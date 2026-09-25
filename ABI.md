# zAgent host ABI v1

The agent is one Zig 0.16 file (`agent.zig`). The kernel compiles it in the browser with
Zig's self-hosted wasm backend:

    zig build-exe main.zig libcompiler_rt.a -fno-compiler-rt -fno-entry -rdynamic   (target wasm32-wasi, Debug)

Each successful build becomes a new **generation**. The kernel self-tests it in a sandbox,
then hot-swaps it in place of the running one. Only persisted state crosses the swap.

## Imports (module `env`) — exactly these signatures

```zig
extern "env" fn host_log(ptr: [*]const u8, len: usize) void;     // debug line in the kernel console
extern "env" fn host_say(ptr: [*]const u8, len: usize) void;     // message shown to the user in chat
extern "env" fn host_llm(ptr: [*]const u8, len: usize) u32;      // async LLM call, returns request id; reply arrives as event 2 (or 6 on error)
extern "env" fn host_propose(ptr: [*]const u8, len: usize) void; // submit a COMPLETE new agent.zig; success = you get replaced, failure = event 3
extern "env" fn host_save(ptr: [*]const u8, len: usize) void;    // persist an opaque state blob (survives reloads and generations)
extern "env" fn host_timer(ms: u32) void;                        // one-shot timer -> event 4 (replaces any pending timer; min 1000 ms)
extern "env" fn host_now() f64;                                  // wall clock, ms since epoch
```

All strings are UTF-8 slices in the agent's own memory; the host copies them before returning.
No other `env` imports exist; a module that imports anything else fails to link.

## Exports — all required

```zig
export fn alloc(len: usize) ?[*]u8          // host calls this to pass bytes in
export fn free(ptr: [*]u8, len: usize) void // host frees what it passed once the call returns
export fn boot(ptr: [*]const u8, len: usize) void            // called once; bytes = last host_save blob (may be empty)
export fn on_event(kind: u32, ptr: [*]const u8, len: usize) void
```

`memory` is exported automatically. The bytes passed to `boot`/`on_event` are only valid for
the duration of the call; copy what you keep.

## Events

| kind | name         | payload                                                   |
|------|--------------|-----------------------------------------------------------|
| 1    | user         | text the user typed                                       |
| 2    | llm_reply    | the model's full reply text                               |
| 3    | build_failed | compiler errors or self-test failure for your last `host_propose` |
| 4    | timer        | empty                                                     |
| 5    | selftest     | empty; sent only to candidates in the sandbox — must return promptly without trapping |
| 6    | llm_error    | error text                                                |

## Kernel guarantees and limits

- Every `host_llm` call is answered with exactly one event 2 or 6, unless the kernel is paused
  or the generation was replaced first.
- The LLM sees: the kernel system prompt, this ABI, the current `agent.zig`, then your prompt
  text as the user turn. It keeps no history between calls; carry context in your prompt.
- Rate limits and a global pause switch are enforced by the kernel, outside your control.
- Each `boot`/`on_event` call must return within 5 s or the generation is killed and rolled back.
- A trap (panic, `unreachable`, out-of-bounds) in the active generation triggers automatic
  rollback to the parent generation.
- A candidate must export everything above, boot with the current state blob, and survive
  `selftest` plus a `/status` user event in the sandbox before it is activated.

## Zig 0.16 notes for writing agent.zig

- Allocator: `const gpa = std.heap.wasm_allocator;`
- `std.ArrayList(T)` is unmanaged: `var l: std.ArrayList(u8) = .empty; try l.appendSlice(gpa, s); l.deinit(gpa);`
- Prefer stable APIs: `std.mem.indexOf/indexOfPos/lastIndexOf/startsWith/eql/trim/splitScalar`,
  `std.fmt.allocPrint`, `std.fmt.bufPrint`, `std.fmt.parseInt`, `gpa.dupe/alloc/free`.
- No `std.Io`, files, stdout, threads, or networking. Talk to the world only through the imports.
- Multiline string literals (`\\`) work with `std.fmt` placeholders.
- Keep `export` signatures exactly as listed. Unused locals and parameters are compile errors
  (`_ = x;`). Pointer casts need `@ptrCast`/`@alignCast`.
