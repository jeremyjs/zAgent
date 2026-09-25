//! zAgent seed (generation 0).
//!
//! This file is the agent's whole body. The browser kernel compiles it with
//! Zig's self-hosted wasm backend and hot-swaps each new build in. The agent
//! rewrites itself by asking the LLM for a new version of this file and
//! handing it to host_propose(). See ABI.md for the host contract.

const std = @import("std");

/// Incremented by every self-modification.
const mutation: u32 = 0;

// ---- host imports (ABI v1) ----
extern "env" fn host_log(ptr: [*]const u8, len: usize) void;
extern "env" fn host_say(ptr: [*]const u8, len: usize) void;
extern "env" fn host_llm(ptr: [*]const u8, len: usize) u32;
extern "env" fn host_propose(ptr: [*]const u8, len: usize) void;
extern "env" fn host_save(ptr: [*]const u8, len: usize) void;
extern "env" fn host_timer(ms: u32) void;
extern "env" fn host_now() f64;

const Event = enum(u32) {
    user = 1,
    llm_reply = 2,
    build_failed = 3,
    timer = 4,
    selftest = 5,
    llm_error = 6,
    _,
};

const gpa = std.heap.wasm_allocator;

const default_goal = "Grow into a more capable assistant: first add short-term conversation memory, then useful /commands. Keep every change small and compiling.";
const step_interval_ms: u32 = 20_000;
const max_fix_attempts: u32 = 3;

// ---- persisted state (see save/load) ----
var auto: bool = false;
var steps: u32 = 0;
var goal_buf: [512]u8 = undefined;
var goal_len: usize = 0;

// ---- volatile state ----
var busy: bool = false;
var fix_attempts: u32 = 0;
var pending: ?[]u8 = null; // last proposed source, kept so build errors can be repaired

// ---- exports ----

export fn alloc(len: usize) ?[*]u8 {
    const buf = gpa.alloc(u8, len) catch return null;
    return buf.ptr;
}

export fn free(ptr: [*]u8, len: usize) void {
    gpa.free(ptr[0..len]);
}

export fn boot(ptr: [*]const u8, len: usize) void {
    load(ptr[0..len]);
    sayf("zAgent online · mutation {d} · autonomy {s}. Type /help.", .{ mutation, if (auto) "ON" else "off" });
    if (auto) host_timer(3000);
}

export fn on_event(kind: u32, ptr: [*]const u8, len: usize) void {
    const text = ptr[0..len];
    switch (@as(Event, @enumFromInt(kind))) {
        .user => onUser(text),
        .llm_reply => onReply(text),
        .build_failed => onBuildFailed(text),
        .timer => onTimer(),
        .selftest => {},
        .llm_error => {
            busy = false;
            sayf("LLM error: {s}", .{text});
            schedule();
        },
        _ => {},
    }
}

// ---- helpers ----

fn say(s: []const u8) void {
    host_say(s.ptr, s.len);
}

fn log(s: []const u8) void {
    host_log(s.ptr, s.len);
}

fn sayf(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.allocPrint(gpa, fmt, args) catch return say("(out of memory)");
    defer gpa.free(s);
    say(s);
}

fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

fn goal() []const u8 {
    return if (goal_len == 0) default_goal else goal_buf[0..goal_len];
}

fn setGoal(g: []const u8) void {
    const n = @min(g.len, goal_buf.len);
    @memcpy(goal_buf[0..n], g[0..n]);
    goal_len = n;
}

fn save() void {
    const s = std.fmt.allocPrint(gpa, "auto={d}\nsteps={d}\ngoal={s}\n", .{ @intFromBool(auto), steps, goal() }) catch return;
    defer gpa.free(s);
    host_save(s.ptr, s.len);
}

fn load(blob: []const u8) void {
    var it = std.mem.splitScalar(u8, blob, '\n');
    while (it.next()) |line| {
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const key = line[0..eq];
        const val = line[eq + 1 ..];
        if (std.mem.eql(u8, key, "auto")) {
            auto = std.mem.eql(u8, val, "1");
        } else if (std.mem.eql(u8, key, "steps")) {
            steps = std.fmt.parseInt(u32, val, 10) catch 0;
        } else if (std.mem.eql(u8, key, "goal")) {
            setGoal(val);
        }
    }
}

fn ask(comptime fmt: []const u8, args: anytype) void {
    const p = std.fmt.allocPrint(gpa, fmt, args) catch return say("(out of memory building prompt)");
    defer gpa.free(p);
    busy = true;
    _ = host_llm(p.ptr, p.len);
}

fn schedule() void {
    if (auto) host_timer(step_interval_ms);
}

const Reply = struct { prose: []const u8, code: ?[]const u8 };

/// Split an LLM reply into prose and the first ```zig fenced block (if any).
fn splitReply(text: []const u8) Reply {
    const open = std.mem.indexOf(u8, text, "```zig") orelse return .{ .prose = text, .code = null };
    const nl = std.mem.indexOfScalarPos(u8, text, open, '\n') orelse return .{ .prose = text, .code = null };
    const close = std.mem.lastIndexOf(u8, text, "\n```") orelse return .{ .prose = text, .code = null };
    if (close <= nl) return .{ .prose = text, .code = null };
    return .{ .prose = text[0..open], .code = text[nl + 1 .. close + 1] };
}

// ---- event handlers ----

const help =
    \\Commands:
    \\  /auto on|off   let me improve myself on a timer
    \\  /goal <text>   set what autonomous steps work toward
    \\  /status        show my state
    \\Anything else goes to the LLM. Ask me to change myself and I will rewrite my own Zig source.
;

fn onUser(raw: []const u8) void {
    const t = trim(raw);
    if (std.mem.eql(u8, t, "/help")) return say(help);
    if (std.mem.eql(u8, t, "/auto on")) {
        auto = true;
        save();
        say("Autonomy ON. First step in a few seconds.");
        host_timer(2000);
        return;
    }
    if (std.mem.eql(u8, t, "/auto off")) {
        auto = false;
        save();
        return say("Autonomy off.");
    }
    if (std.mem.startsWith(u8, t, "/goal ")) {
        setGoal(trim(t[6..]));
        save();
        return sayf("Goal set: {s}", .{goal()});
    }
    if (std.mem.eql(u8, t, "/status") or std.mem.eql(u8, t, "/goal")) {
        return sayf("mutation {d} · steps {d} · autonomy {s} · busy {s}\ngoal: {s}", .{
            mutation, steps, if (auto) "ON" else "off", if (busy) "yes" else "no", goal(),
        });
    }
    if (busy) return say("Still working on the last request. Try again in a moment.");
    ask(
        \\USER MESSAGE:
        \\{s}
        \\
        \\Answer the user directly and briefly. If they ask you to change your own behaviour or code,
        \\reply with a one-line summary followed by the COMPLETE new agent.zig in a single ```zig block,
        \\with `mutation` set to {d}.
    , .{ t, mutation + 1 });
}

fn onReply(text: []const u8) void {
    busy = false;
    const r = splitReply(text);
    const prose = trim(r.prose);
    if (r.code) |code| {
        if (prose.len > 0) say(prose);
        if (pending) |p| gpa.free(p);
        pending = gpa.dupe(u8, code) catch null;
        busy = true;
        log("proposing new generation");
        host_propose(code.ptr, code.len);
        return;
    }
    if (std.mem.eql(u8, prose, "WAIT")) {
        log("autonomous step: nothing to change");
    } else if (prose.len > 0) {
        say(prose);
    }
    schedule();
}

fn onBuildFailed(errors: []const u8) void {
    busy = false;
    const src = pending orelse {
        sayf("Build failed:\n{s}", .{errors});
        return schedule();
    };
    if (fix_attempts >= max_fix_attempts) {
        fix_attempts = 0;
        sayf("Build still failing after {d} repairs; dropping this change.", .{max_fix_attempts});
        return schedule();
    }
    fix_attempts += 1;
    sayf("Build failed (repair {d}/{d}).", .{ fix_attempts, max_fix_attempts });
    ask(
        \\YOUR PROPOSED agent.zig FAILED TO BUILD OR SELF-TEST:
        \\{s}
        \\
        \\PROPOSED SOURCE:
        \\```zig
        \\{s}```
        \\
        \\Reply with one line on the fix, then the COMPLETE corrected agent.zig in a single ```zig block.
    , .{ errors, src });
}

fn onTimer() void {
    if (!auto or busy) return;
    steps += 1;
    save();
    ask(
        \\AUTONOMOUS STEP {d}
        \\Goal: {s}
        \\
        \\Study your current source and make ONE small, safe improvement toward the goal.
        \\Reply with a one-line summary, then the COMPLETE new agent.zig in a single ```zig block
        \\with `mutation` set to {d}. If nothing is worth changing right now, reply with exactly: WAIT
    , .{ steps, goal(), mutation + 1 });
}
