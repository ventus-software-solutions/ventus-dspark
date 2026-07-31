#!/usr/bin/env bash
# Resumable download + verify of the 0731 weights on ONE node.
# Run on both nodes (or download once and copy). ~155 GB.
#
#   ./scripts/fetch-weights.sh                 # -> ~/models/v4-flash-0731
#   ./scripts/fetch-weights.sh /mnt/weights/v4-flash-0731
set -euo pipefail

DIR="${1:-$HOME/models/v4-flash-0731}"
REPO=deepseek-ai/DeepSeek-V4-Flash-0731
REV=9e165c30e2704aec5d9d593cce3eebd58bbef1cb
EXPECTED_BYTES=166886535336   # 48 safetensors shards, measured from the HF blob listing

if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

echo "downloading $REPO @ ${REV:0:8} -> $DIR (resumable)"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
"$HF" download "$REPO" --revision "$REV" --local-dir "$DIR" --local-dir-use-symlinks False

python3 - "$DIR" "$EXPECTED_BYTES" <<'PY'
import os, sys
root, expected = sys.argv[1], int(sys.argv[2])
total = 0
n = 0
for cur, _, files in os.walk(root):
    for f in files:
        total += os.path.getsize(os.path.join(cur, f))
        n += 1
ok = os.path.isfile(os.path.join(root, "config.json")) and total >= expected * 0.99
print(f"files={n} bytes={total:,} expected={expected:,} {'OK' if ok else 'INCOMPLETE — rerun (resumes)'}")
sys.exit(0 if ok else 1)
PY
