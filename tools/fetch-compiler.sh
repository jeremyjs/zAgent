#!/usr/bin/env bash
# Fetch Zig's wasm32-wasi compiler build (self-hosted backend), stdlib, and compiler_rt
# into zig-out/. These are the assets the zigtools playground serves.
set -euo pipefail
cd "$(dirname "$0")/.."
B=https://playground.zigtools.org/assets
mkdir -p zig-out/bin
curl -fL "$B/zig-DZb1EcLK.wasm" -o zig-out/bin/zig.wasm
curl -fL "$B/libcompiler_rt-BdDvk7bF.a" -o zig-out/libcompiler_rt.a
curl -fL "$B/zig.tar-BoS5LufH.gz" -o zig-out/zig.tar.gz
ls -la zig-out zig-out/bin
