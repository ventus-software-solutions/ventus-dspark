#!/usr/bin/env bash
# Per-node hardware telemetry, one key=value line per field.
#
# Runs on the node itself (locally on the head, over ssh on workers). Kept as
# its own file rather than an inline ssh string so it can be read, tested and
# shipped to the node without quoting games.
#
# GB10 note: nvidia-smi reports GPU memory as [N/A] because host and GPU share
# one pool — host memory percentage IS the GPU memory signal on this hardware,
# and it is the number that precedes a unified-memory wedge.
set -uo pipefail

emit() { printf '%s=%s\n' "$1" "$2"; }

if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -r gtemp gutil gpower <<<"$(nvidia-smi \
    --query-gpu=temperature.gpu,utilization.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  [ -n "${gtemp:-}" ] && emit gpu_temp_c "$gtemp"
  [ -n "${gutil:-}" ] && emit gpu_util_pct "$gutil"
  [ -n "${gpower:-}" ] && emit gpu_power_w "$gpower"
fi

free -b 2>/dev/null | awk '/^Mem:/ {
  printf "mem_used_pct=%.1f\nmem_total_gib=%.0f\n", 100*($3/$2), $2/1073741824 }'

awk '{ printf "load1=%s\n", $1 }' /proc/loadavg 2>/dev/null
df -P / 2>/dev/null | awk 'NR==2 { gsub("%","",$5); printf "disk_used_pct=%s\n", $5 }'
awk '{ printf "uptime_h=%.0f\n", $1/3600 }' /proc/uptime 2>/dev/null

# NVRM allocation failures since boot. On GB10 these accumulate for hours
# before the host wedges hard enough to drop ssh (lived 2026-08-05 on the
# worker), so a rising count is the earliest warning we get.
nvrm=$(journalctl -k -b 2>/dev/null | grep -c 'NVRM.*NV_ERR_NO_MEMORY' || true)
emit nvrm_oom "${nvrm:-0}"
