#!/usr/bin/env bash
# Resumable download + verify of model weights on ONE node.
# Run on both nodes, or download once and copy over the fabric link.
#
#   ./scripts/fetch-weights.sh                                    # 0731 defaults
#   ./scripts/fetch-weights.sh --dir /mnt/weights/v4-flash-0731
#   ./scripts/fetch-weights.sh --repo org/Some-Model --revision <sha> --dir ~/models/some-model
#
# Verification: with --expected-bytes (or the 0731 default) the byte total is
# checked; otherwise only config.json presence and the absence of partial
# downloads. Interrupted runs resume — just rerun.
set -euo pipefail

REPO=deepseek-ai/DeepSeek-V4-Flash-0731
REV=9e165c30e2704aec5d9d593cce3eebd58bbef1cb
DIR=""
EXPECTED_BYTES=""
DEFAULT_EXPECTED=166886535336   # 48 safetensors shards, measured from the HF blob listing

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift ;;
    --revision) REV="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --expected-bytes) EXPECTED_BYTES="$2"; shift ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) # positional dir, kept for backward compatibility
       DIR="$1" ;;
  esac
  shift
done
[ -n "$DIR" ] || DIR="$HOME/models/$(basename "$REPO" | tr '[:upper:]' '[:lower:]')"
# Only the stock 0731 pin has a known byte total.
[ -n "$EXPECTED_BYTES" ] || { [ "$REPO" = "deepseek-ai/DeepSeek-V4-Flash-0731" ] && EXPECTED_BYTES=$DEFAULT_EXPECTED || EXPECTED_BYTES=0; }

say()  { printf '\033[36m[fetch-weights]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[fetch-weights FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else die "no hf/huggingface-cli on PATH — pip install -U huggingface_hub hf_transfer"
fi

# Disk guard: refuse to start a download the disk cannot hold.
if [ "$EXPECTED_BYTES" -gt 0 ]; then
  mkdir -p "$DIR"
  have=$(du -sb "$DIR" 2>/dev/null | cut -f1 || echo 0)
  free=$(df -B1 --output=avail "$DIR" | tail -1 | tr -d ' ')
  need=$(( EXPECTED_BYTES - have ))
  [ "$need" -lt "$free" ] || die "need ~$((need/1024/1024/1024)) GiB more, only $((free/1024/1024/1024)) GiB free at $DIR"
fi

say "model    $REPO @ ${REV:0:8}"
say "target   $DIR"
[ "$EXPECTED_BYTES" -gt 0 ] && say "size     ~$((EXPECTED_BYTES/1024/1024/1024)) GiB (resumable — rerun after any interruption)"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
"$HF" download "$REPO" --revision "$REV" --local-dir "$DIR" --local-dir-use-symlinks False

python3 - "$DIR" "$EXPECTED_BYTES" <<'PY'
import os, sys
root, expected = sys.argv[1], int(sys.argv[2])
total = files = partial = 0
for cur, _, names in os.walk(root):
    for f in names:
        files += 1
        total += os.path.getsize(os.path.join(cur, f))
        if f.endswith(".incomplete"):
            partial += 1
ok = os.path.isfile(os.path.join(root, "config.json")) and partial == 0
if expected:
    ok = ok and total >= expected * 0.99
    print(f"files={files} bytes={total:,} expected={expected:,} partial={partial} "
          f"{'OK' if ok else 'INCOMPLETE — rerun (resumes)'}")
else:
    print(f"files={files} bytes={total:,} partial={partial} "
          f"{'OK (no byte total to verify against)' if ok else 'INCOMPLETE — rerun (resumes)'}")
sys.exit(0 if ok else 1)
PY
say "done: $DIR"
say "copy to the worker over the fabric link (much faster than a second download):"
say "  rsync -a --partial --info=progress2 $DIR/ <worker>:$DIR/"
