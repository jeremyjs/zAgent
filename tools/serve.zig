//! Tiny static file server for zAgent, written against Zig 0.16 std.Io.
//!   zig run tools/serve.zig -- [--port 8787] [--host 127.0.0.1] [--root .]
//! Streams files (model weights are large), sets correct MIME types
//! (application/wasm matters for streaming compile), and refuses path traversal.

const std = @import("std");
const Io = std.Io;
const mem = std.mem;

const Context = struct { io: Io, root: Io.Dir };

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    var port: u16 = 8787;
    var host: []const u8 = "127.0.0.1";
    var root_path: []const u8 = ".";
    var args = try init.minimal.args.iterateAllocator(arena);
    defer args.deinit();
    _ = args.skip();
    while (args.next()) |arg| {
        if (mem.eql(u8, arg, "--port")) {
            port = try std.fmt.parseInt(u16, args.next() orelse return error.MissingPort, 10);
        } else if (mem.eql(u8, arg, "--host")) {
            host = args.next() orelse return error.MissingHost;
        } else if (mem.eql(u8, arg, "--root")) {
            root_path = args.next() orelse return error.MissingRoot;
        } else {
            std.log.err("unknown argument: {s}", .{arg});
            return error.BadArgs;
        }
    }

    var root = try Io.Dir.cwd().openDir(io, root_path, .{});
    defer root.close(io);

    const address = try Io.net.IpAddress.parse(host, port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    std.log.info("zAgent serving {s} at http://{s}:{d}/", .{ root_path, host, server.socket.address.getPort() });

    var ctx: Context = .{ .io = io, .root = root };
    var group: Io.Group = .init;
    defer group.cancel(io);
    while (true) {
        const stream = try server.accept(io);
        group.async(io, handleConnection, .{ &ctx, stream });
    }
}

fn handleConnection(ctx: *Context, stream: Io.net.Stream) void {
    const io = ctx.io;
    defer stream.close(io);
    var recv_buf: [8192]u8 = undefined;
    var send_buf: [8192]u8 = undefined;
    var conn_reader = stream.reader(io, &recv_buf);
    var conn_writer = stream.writer(io, &send_buf);
    var http = std.http.Server.init(&conn_reader.interface, &conn_writer.interface);
    while (http.reader.state == .ready) {
        var req = http.receiveHead() catch return;
        serve(ctx, &req) catch |err| {
            std.log.warn("{s} {s}: {t}", .{ @tagName(req.head.method), req.head.target, err });
            return;
        };
    }
}

fn serve(ctx: *Context, req: *std.http.Server.Request) !void {
    const io = ctx.io;
    var target = req.head.target;
    if (mem.indexOfAny(u8, target, "?#")) |i| target = target[0..i];
    var rel = mem.trimStart(u8, target, "/");
    if (rel.len == 0 or rel[rel.len - 1] == '/') rel = if (rel.len == 0) "index.html" else rel;
    if (mem.indexOf(u8, rel, "..") != null or mem.indexOfScalar(u8, rel, '\\') != null) {
        return notFound(req, rel);
    }

    var path_buf: [1024]u8 = undefined;
    const path = if (rel[rel.len - 1] == '/')
        try std.fmt.bufPrint(&path_buf, "{s}index.html", .{rel})
    else
        rel;

    const file = ctx.root.openFile(io, path, .{}) catch return notFound(req, path);
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind == .directory) return notFound(req, path);

    std.log.info("{s} /{s} ({d} bytes)", .{ @tagName(req.head.method), path, stat.size });
    var body_buf: [64 * 1024]u8 = undefined;
    var body = try req.respondStreaming(&body_buf, .{
        .content_length = stat.size,
        .respond_options = .{ .extra_headers = &.{
            .{ .name = "content-type", .value = mimeType(path) },
            .{ .name = "cache-control", .value = "no-cache" },
            .{ .name = "access-control-allow-origin", .value = "*" },
        } },
    });
    // For HEAD, std.http elides the body writer, but end() still expects content_length bytes.
    var read_buf: [64 * 1024]u8 = undefined;
    var fr = file.reader(io, &read_buf);
    _ = try fr.interface.streamRemaining(&body.writer);
    try body.end();
}

fn notFound(req: *std.http.Server.Request, path: []const u8) !void {
    std.log.info("404 /{s}", .{path});
    try req.respond("not found\n", .{
        .status = .not_found,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
    });
}

fn mimeType(path: []const u8) []const u8 {
    const types = [_]struct { []const u8, []const u8 }{
        .{ ".html", "text/html; charset=utf-8" },
        .{ ".js", "text/javascript; charset=utf-8" },
        .{ ".mjs", "text/javascript; charset=utf-8" },
        .{ ".css", "text/css; charset=utf-8" },
        .{ ".json", "application/json" },
        .{ ".wasm", "application/wasm" },
        .{ ".zig", "text/plain; charset=utf-8" },
        .{ ".md", "text/markdown; charset=utf-8" },
        .{ ".svg", "image/svg+xml" },
        .{ ".png", "image/png" },
        .{ ".gz", "application/gzip" },
    };
    for (types) |t| if (mem.endsWith(u8, path, t[0])) return t[1];
    return "application/octet-stream";
}
