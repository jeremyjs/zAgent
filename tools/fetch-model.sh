#!/usr/bin/env bash
# Download SmolLM2 weights into models/ so the page loads them same-origin (offline-capable).
#   tools/fetch-model.sh            # 135M (269 MB)
#   tools/fetch-model.sh 360M       # 360M (724 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${1:-135M}"
REPO="HuggingFaceTB/SmolLM2-${SIZE}-Instruct"
DIR="models/SmolLM2-${SIZE}-Instruct"
mkdir -p "$DIR"
for f in config.json tokenizer.json model.safetensors; do
  [ -s "$DIR/$f" ] && { echo "have $DIR/$f"; continue; }
  curl -fL "https://huggingface.co/$REPO/resolve/main/$f" -o "$DIR/$f"
done
ls -la "$DIR"
