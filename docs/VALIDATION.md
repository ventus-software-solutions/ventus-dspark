# Validation — measured on the Ventus 2× DGX Spark fleet

Date: 2026-07-31 evening. Hardware: 2× DGX Spark (GB10, SM121, 128 GB
unified), TP=2 over a 200 Gb ConnectX RoCE v2 link.
Runtime: the exact image chain this repo builds (base `ghcr.io/bjk110/
vllm-spark:unholy-fusion-prod-ready` → overlay → nvfp4 stages), engine
vLLM `0.21.1rc1.dev339+g1967a5627bc3`.

## Profile (validated, do not change without re-measuring)

```
kv_cache_dtype          nvfp4_ds_mla
block_size              256
max_model_len           1048576 (1M)
max_num_seqs            6
max_num_batched_tokens  8192
gpu_memory_utilization  0.78
spec tokens (k)         5  (checkpoint dspark_block_size)
moe backend             B12X (VLLM_USE_B12X_MOE=1)
flashinfer sampler      on; autotune cache warm
prefix caching          on; chunked prefill on
distributed             mp executor, 2 nodes, worker-first
NCCL                    single HCA, GID 3 (RoCE v2)
```

## Numbers

| check | result |
|---|---|
| /health | 200 |
| /v1/models | `/models/v4-flash-0731`, `max_model_len` 1,048,576 |
| math | `17*19` → `"323"` (0.2–0.3 s warm) |
| prefill 22.5K tokens | 12.7–12.9 s → **1,747–1,770 tok/s** |
| tool calling | `get_weather({"city":"Berlin"})`, `finish_reason=tool_calls` |
| decode (200 tok, count-300) | 3.5 s → 56.9 tok/s; peak run **78.6 tok/s** |
| KV pool @ 1M ctx | 1,468,303 tokens → 1.40× concurrency (10.25 GiB) |
| boot | cold ~13 min (systemd path), warm ~7 min |

Upstream reference (tonyd2wild, same hardware/profile): 32.7 → **55.4 tok/s
mean** and acceptance 25.7% → **60.2%** with Patch 4; peak-finder prompt
78.4 tok/s at 98.9% acceptance. Our 56.9/78.6 matches within run variance.

## Engine A/B — 0.21 lane vs 0.25 lane (2026-08-04)

Same hardware, same weights, same profile (gmu 0.80 on 0.25), full 20-cell
compat sweep (`scripts/benchmark.py --profile compat`: prompts 256→131K,
concurrency 1–6, temperature 0.6, single trial). Raw JSON:
`results/compat.json` (0.21) and `results/compat-025.json` (0.25).

| metric | 0.21 lane | 0.25 lane |
|---|---|---|
| decode, mean delta across 20 cells | baseline | **+8.4%** |
| single-stream range | 57–70 tok/s | 72–81 tok/s |
| KV pool @ 1M ctx | 1,468,303 tokens | 1,691,551 (+15%) |
| draft acceptance | lower in 16/20 cells | higher (drafter improvement) |
| warm boot | ~7 min | ~5 min |

Caveats: compat profile is single-trial at temperature 0.6, so per-cell
deltas carry acceptance variance — the +30% best cell drew 0.878 acceptance.
Acceptance-corrected engine gain is ~+6–9% single-stream; the rest of the
improvement is 0.25's drafter accepting more, which is real but
workload-dependent. Decode speed on any engine tracks draft acceptance
almost linearly (~`20 + 65·α` tok/s on 0.21): free prose ≈ 0.33 → ~38 tok/s,
code ≈ 0.56 → ~56 tok/s, repetitive text ≈ 0.74+ → 70+. Quote workload
numbers, not peak numbers.

## Long-context depth curve (2026-08-12)

The published decode numbers were all ≤131K. Benchmarking deeper exposed a
kernel-dispatch bug: `nvfp4_ds_mla` KV was routed to the slow bf16 path
(`use_fp8_cache` only matched the fp8 name), collapsing decode at depth.
One-line overlay fix; quality gate byte-identical; raw JSON in
`results/depth-{before,after}.json`.

| context | before | after |
|---|---|---|
| 131,072 | 38.4 tok/s | 46.5 tok/s |
| 262,144 | 2.65 tok/s | **72.5 tok/s (27x)** |

The depth penalty is gone, not reduced — 262K decodes in the same band as
short context. Real agent traffic here has a p99 prompt of ~238K tokens, so
the deepest quarter-percent of sessions were running ~25x slower than
necessary until this fix. Lesson recorded: benchmark at the workload's p99
depth, not the benchmark suite's habitual depths.

## What was tested vs not

Tested: dark-window smoke suite (above), full systemd reboot path (cold boot,
health, last-known-good update), fit gate (4.6× concurrency @ 1M),
verify-source (weights match `deepseek-ai/DeepSeek-V4-Flash-0731` @
`9e165c30…`), two independent smoke passes on the production instance.

Not yet tested: multi-day stability under sustained agent traffic, true 1M
single prompts (we tested 22.5K), memory growth over weeks, behavior after
upstream vLLM merges SM12x support. Treat this as a strong first validation,
not a lifetime warranty.

## Why these knobs are the ones that matter

- **gmu 0.78, not 0.80/0.85:** spec-decode buffers allocate on the *first real
  request*, not at boot — 0.80 boots clean then dies under traffic.
- **k=5, not 7 (0.21 lane only):** the card's `num_speculative_tokens: 7`
  crashes the 0.21 engine on first generation ("size of tensor a (7) must
  match tensor b (5)"). The 0.25 engine has no such constraint and k=7 is
  DeepSeek's own recommendation — untested here, queued in
  [EXPERIMENTS.md](EXPERIMENTS.md). k=6 wastes a
  dead slot. Valid: k ≤ 5 or a multiple of 5.
- **Single HCA:** the second IB port on many Sparks has no RoCEv2 GID;
  NCCL hits it and fails the QP handshake with no readable exception.
- **NVFP4 KV:** 7,495 B/token vs 31,593 on the fp8-era profile — the entire
  reason 1M context fits.
