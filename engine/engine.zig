//! zAgent micro inference engine: llama-architecture transformer (SmolLM2 family)
//! in ~300 lines of Zig, compiled to wasm32 + SIMD128.
//!
//! - int8 weights and activations, symmetric per-group (64) scales, i32 dot products
//! - grouped-query attention, rotate-half RoPE, RMSNorm, SwiGLU, tied embeddings
//! - f32 KV cache, temperature / top-k / top-p sampling
//! - quantizes bf16 tensors on load; the quantized weight slab is one contiguous
//!   block so the host can snapshot it (OPFS) and restore it without re-quantizing.
//!
//! Host protocol: configure() -> load_tensor() for every tensor (or write a snapshot
//! into weights_ptr()) -> forward(token, pos) -> sample().

const std = @import("std");
const gpa = std.heap.wasm_allocator;

const GS = 64; // quantization group size
const VL = 16; // f32 SIMD width we ask LLVM for

const Q8 = struct {
    q: []i8,
    s: []f32,
    rows: usize,
    cols: usize,
};

const Layer = struct {
    att_norm: []f32,
    wq: Q8,
    wk: Q8,
    wv: Q8,
    wo: Q8,
    ffn_norm: []f32,
    w_gate: Q8,
    w_down: Q8,
    w_up: Q8,
};

const Config = struct {
    dim: usize,
    hidden: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv: usize,
    vocab: usize,
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    eps: f32,
};

var cfg: Config = undefined;
var slab: []align(64) u8 = &.{};
var slab_used: usize = 0;
var embed: Q8 = undefined;
var final_norm: []f32 = &.{};
var layers: []Layer = &.{};

// run state
var x: []f32 = &.{};
var xb: []f32 = &.{};
var xb2: []f32 = &.{};
var hb: []f32 = &.{};
var hb2: []f32 = &.{};
var q: []f32 = &.{};
var att: []f32 = &.{};
var logits: []f32 = &.{};
var xq: []i8 = &.{};
var xs: []f32 = &.{};
var key_cache: []f32 = &.{};
var val_cache: []f32 = &.{};
var rng: u64 = 0x9E3779B97F4A7C15;

// ---------------------------------------------------------------- setup

fn carve(comptime T: type, n: usize, dry: bool) []T {
    const a = std.mem.alignForward(usize, slab_used, 64);
    slab_used = a + n * @sizeOf(T);
    if (dry) return &.{};
    const bytes = slab[a..slab_used];
    return @as([*]T, @ptrCast(@alignCast(bytes.ptr)))[0..n];
}

fn carveQ8(rows: usize, cols: usize, dry: bool) Q8 {
    return .{ .q = carve(i8, rows * cols, dry), .s = carve(f32, rows * cols / GS, dry), .rows = rows, .cols = cols };
}

fn layout(dry: bool) void {
    slab_used = 0;
    const kv_dim = cfg.n_kv * cfg.head_dim;
    const q_dim = cfg.n_heads * cfg.head_dim;
    embed = carveQ8(cfg.vocab, cfg.dim, dry);
    final_norm = carve(f32, cfg.dim, dry);
    for (layers) |*l| {
        l.* = .{
            .att_norm = carve(f32, cfg.dim, dry),
            .wq = carveQ8(q_dim, cfg.dim, dry),
            .wk = carveQ8(kv_dim, cfg.dim, dry),
            .wv = carveQ8(kv_dim, cfg.dim, dry),
            .wo = carveQ8(cfg.dim, q_dim, dry),
            .ffn_norm = carve(f32, cfg.dim, dry),
            .w_gate = carveQ8(cfg.hidden, cfg.dim, dry),
            .w_down = carveQ8(cfg.dim, cfg.hidden, dry),
            .w_up = carveQ8(cfg.hidden, cfg.dim, dry),
        };
    }
}

fn allocF(n: usize) ![]f32 {
    const s = try gpa.alloc(f32, n);
    @memset(s, 0);
    return s;
}

/// Returns 0 on success, negative on error.
export fn configure(dim: u32, hidden: u32, n_layers: u32, n_heads: u32, n_kv: u32, vocab: u32, seq_len: u32, theta: f32, eps: f32) i32 {
    cfg = .{
        .dim = dim, .hidden = hidden, .n_layers = n_layers, .n_heads = n_heads, .n_kv = n_kv,
        .vocab = vocab, .seq_len = seq_len, .head_dim = dim / n_heads, .theta = theta, .eps = eps,
    };
    if (dim % GS != 0 or hidden % GS != 0 or dim % n_heads != 0 or n_heads % n_kv != 0) return -1;
    configureAlloc() catch return -2;
    return 0;
}

fn configureAlloc() !void {
    layers = try gpa.alloc(Layer, cfg.n_layers);
    layout(true);
    slab = try gpa.alignedAlloc(u8, .@"64", slab_used);
    layout(false);
    const kv_dim = cfg.n_kv * cfg.head_dim;
    const big = @max(cfg.dim, cfg.hidden);
    x = try allocF(cfg.dim);
    xb = try allocF(cfg.dim);
    xb2 = try allocF(cfg.dim);
    hb = try allocF(cfg.hidden);
    hb2 = try allocF(cfg.hidden);
    q = try allocF(cfg.n_heads * cfg.head_dim);
    att = try allocF(cfg.n_heads * cfg.seq_len);
    logits = try allocF(cfg.vocab);
    xq = try gpa.alloc(i8, big);
    xs = try allocF(big / GS);
    key_cache = try allocF(cfg.n_layers * cfg.seq_len * kv_dim);
    val_cache = try allocF(cfg.n_layers * cfg.seq_len * kv_dim);
}

export fn weights_ptr() [*]u8 {
    return slab.ptr;
}
export fn weights_len() usize {
    return slab.len;
}
export fn logits_ptr() [*]f32 {
    return logits.ptr;
}

export fn alloc(n: usize) ?[*]u8 {
    const s = gpa.alloc(u8, n) catch return null;
    return s.ptr;
}
export fn free(p: [*]u8, n: usize) void {
    gpa.free(p[0..n]);
}

// ---------------------------------------------------------------- loading

inline fn bf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn quantizeRow(src: []const u16, dq: []i8, ds: []f32) void {
    var g: usize = 0;
    while (g < src.len / GS) : (g += 1) {
        const chunk = src[g * GS ..][0..GS];
        var amax: f32 = 0;
        for (chunk) |v| amax = @max(amax, @abs(bf16(v)));
        const scale = amax / 127.0;
        ds[g] = scale;
        const inv: f32 = if (scale == 0) 0 else 1.0 / scale;
        for (chunk, 0..) |v, j| dq[g * GS + j] = @intFromFloat(@round(bf16(v) * inv));
    }
}

const Tensor = enum(u32) { embed, final_norm, att_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_down, w_up };

/// Quantize one bf16 tensor into its slot. Returns 0 ok, -1 bad id/layer, -2 size mismatch.
export fn load_tensor(id: u32, layer: u32, src: [*]const u16, n: usize) i32 {
    const t = std.enums.fromInt(Tensor, id) orelse return -1;
    if (t != .embed and t != .final_norm and layer >= cfg.n_layers) return -1;
    const l = if (t == .embed or t == .final_norm) &layers[0] else &layers[layer];
    const f: ?[]f32 = switch (t) {
        .final_norm => final_norm,
        .att_norm => l.att_norm,
        .ffn_norm => l.ffn_norm,
        else => null,
    };
    if (f) |dst| {
        if (dst.len != n) return -2;
        for (dst, 0..) |*d, i| d.* = bf16(src[i]);
        return 0;
    }
    const w: *Q8 = switch (t) {
        .embed => &embed,
        .wq => &l.wq,
        .wk => &l.wk,
        .wv => &l.wv,
        .wo => &l.wo,
        .w_gate => &l.w_gate,
        .w_down => &l.w_down,
        .w_up => &l.w_up,
        else => unreachable,
    };
    if (w.rows * w.cols != n) return -2;
    const gpr = w.cols / GS;
    for (0..w.rows) |r| quantizeRow(src[r * w.cols ..][0..w.cols], w.q[r * w.cols ..][0..w.cols], w.s[r * gpr ..][0..gpr]);
    return 0;
}

// ---------------------------------------------------------------- math

fn rmsnorm(out: []f32, in: []const f32, w: []const f32) void {
    var ss: f32 = 0;
    for (in) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(in.len)) + cfg.eps);
    for (out, in, w) |*o, v, g| o.* = v * inv * g;
}

fn quantizeAct(in: []const f32) void {
    for (0..in.len / GS) |g| {
        const chunk = in[g * GS ..][0..GS];
        var amax: f32 = 0;
        for (chunk) |v| amax = @max(amax, @abs(v));
        const scale = amax / 127.0;
        xs[g] = scale;
        const inv: f32 = if (scale == 0) 0 else 1.0 / scale;
        for (chunk, 0..) |v, j| xq[g * GS + j] = @intFromFloat(@round(v * inv));
    }
}

inline fn dotGroup(a: *const [GS]i8, b: *const [GS]i8) i32 {
    const va: @Vector(GS, i32) = @as(@Vector(GS, i8), a.*);
    const vb: @Vector(GS, i32) = @as(@Vector(GS, i8), b.*);
    return @reduce(.Add, va * vb);
}

/// out[d] = W[d, n] · in[n], with `in` already quantized into xq/xs.
fn matmul(out: []f32, w: Q8) void {
    const n = w.cols;
    const gpr = n / GS;
    for (0..w.rows) |i| {
        const row = w.q[i * n ..];
        const rs = w.s[i * gpr ..];
        var sum: f32 = 0;
        for (0..gpr) |g| {
            const d = dotGroup(xq[g * GS ..][0..GS], row[g * GS ..][0..GS]);
            sum += @as(f32, @floatFromInt(d)) * rs[g] * xs[g];
        }
        out[i] = sum;
    }
}

fn dotF(a: []const f32, b: []const f32) f32 {
    var acc: @Vector(VL, f32) = @splat(0);
    var i: usize = 0;
    while (i + VL <= a.len) : (i += VL) acc += @as(@Vector(VL, f32), a[i..][0..VL].*) * @as(@Vector(VL, f32), b[i..][0..VL].*);
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

fn rope(v: []f32, pos: usize) void {
    const hd = cfg.head_dim;
    const half = hd / 2;
    var h: usize = 0;
    while (h < v.len) : (h += hd) {
        for (0..half) |i| {
            const freq = 1.0 / std.math.pow(f32, cfg.theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(hd)));
            const ang = @as(f32, @floatFromInt(pos)) * freq;
            const c = @cos(ang);
            const s = @sin(ang);
            const x0 = v[h + i];
            const x1 = v[h + i + half];
            v[h + i] = x0 * c - x1 * s;
            v[h + i + half] = x0 * s + x1 * c;
        }
    }
}

fn softmax(v: []f32) void {
    var m: f32 = v[0];
    for (v) |e| m = @max(m, e);
    var sum: f32 = 0;
    for (v) |*e| {
        e.* = @exp(e.* - m);
        sum += e.*;
    }
    for (v) |*e| e.* /= sum;
}

// ---------------------------------------------------------------- forward

/// Run one token at `pos` through the network; logits land in logits_ptr().
export fn forward(token: u32, pos: u32) i32 {
    if (token >= cfg.vocab or pos >= cfg.seq_len) return -1;
    const dim = cfg.dim;
    const hd = cfg.head_dim;
    const kv_dim = cfg.n_kv * hd;
    const group = cfg.n_heads / cfg.n_kv;
    const p: usize = pos;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    const gpr = dim / GS;
    for (0..dim) |j| x[j] = @as(f32, @floatFromInt(embed.q[token * dim + j])) * embed.s[token * gpr + j / GS];

    for (layers, 0..) |l, li| {
        const kc = key_cache[(li * cfg.seq_len + p) * kv_dim ..][0..kv_dim];
        const vc = val_cache[(li * cfg.seq_len + p) * kv_dim ..][0..kv_dim];

        rmsnorm(xb, x, l.att_norm);
        quantizeAct(xb);
        matmul(q, l.wq);
        matmul(kc, l.wk);
        matmul(vc, l.wv);
        rope(q, p);
        rope(kc, p);

        for (0..cfg.n_heads) |h| {
            const qh = q[h * hd ..][0..hd];
            const a = att[h * cfg.seq_len ..][0 .. p + 1];
            const kvh = (h / group) * hd;
            for (0..p + 1) |t| a[t] = dotF(qh, key_cache[(li * cfg.seq_len + t) * kv_dim + kvh ..][0..hd]) * scale;
            softmax(a);
            const out = xb[h * hd ..][0..hd];
            @memset(out, 0);
            for (0..p + 1) |t| {
                const vt = val_cache[(li * cfg.seq_len + t) * kv_dim + kvh ..][0..hd];
                const w = a[t];
                for (out, vt) |*o, v| o.* += w * v;
            }
        }
        quantizeAct(xb[0 .. cfg.n_heads * hd]);
        matmul(xb2, l.wo);
        for (x, xb2) |*a, b| a.* += b;

        rmsnorm(xb, x, l.ffn_norm);
        quantizeAct(xb);
        matmul(hb, l.w_gate);
        matmul(hb2, l.w_up);
        for (hb, hb2) |*g, u| g.* = g.* / (1.0 + @exp(-g.*)) * u;
        quantizeAct(hb);
        matmul(xb, l.w_down);
        for (x, xb) |*a, b| a.* += b;
    }
    rmsnorm(x, x, final_norm);
    quantizeAct(x);
    matmul(logits, embed);
    return 0;
}

// ---------------------------------------------------------------- sampling

export fn set_seed(seed: u32) void {
    rng = @as(u64, seed) *% 0x9E3779B97F4A7C15 | 1;
}

fn randf() f32 {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return @as(f32, @floatFromInt((rng *% 0x2545F4914F6CDD1D) >> 40)) / 16777216.0;
}

const TOPK = 40;

/// temperature <= 0 means greedy. Logits for tokens in `ban` never win (host sets via logit_bias).
export fn sample(temperature: f32, top_p: f32) u32 {
    var best: u32 = 0;
    for (logits, 0..) |v, i| {
        if (v > logits[best]) best = @intCast(i);
    }
    if (temperature <= 0) return best;

    // top-k by insertion into a small sorted list
    var ids: [TOPK]u32 = undefined;
    var vals: [TOPK]f32 = undefined;
    var k: usize = 0;
    for (logits, 0..) |v, i| {
        if (k == TOPK and v <= vals[TOPK - 1]) continue;
        var j = if (k < TOPK) k else TOPK - 1;
        if (k < TOPK) k += 1;
        while (j > 0 and vals[j - 1] < v) : (j -= 1) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
        }
        vals[j] = v;
        ids[j] = @intCast(i);
    }
    var sum: f32 = 0;
    for (vals[0..k]) |*v| {
        v.* = @exp((v.* - vals[0]) / temperature);
        sum += v.*;
    }
    var cum: f32 = 0;
    var cut = k;
    for (vals[0..k], 0..) |v, i| {
        cum += v / sum;
        if (cum >= top_p) {
            cut = i + 1;
            break;
        }
    }
    var mass: f32 = 0;
    for (vals[0..cut]) |v| mass += v;
    const r = randf() * mass;
    var acc: f32 = 0;
    for (vals[0..cut], 0..) |v, i| {
        acc += v;
        if (r < acc) return ids[i];
    }
    return ids[cut - 1];
}

/// Force a logit (e.g. -inf to ban a token) after forward(), before sample().
export fn logit_bias(token: u32, value: f32) void {
    if (token < logits.len) logits[token] = value;
}
