# ventus-dspark

**One command to serve DeepSeek-V4-Flash-0731 on two NVIDIA DGX Spark (GB10)
machines — 1M context, DSpark speculative decoding, OpenAI + Anthropic
compatible APIs. No .env editing, no kernel patches, no build on your side.**

```bash
# on both nodes: get the image (or build it — see below)
docker pull ventus/dspark-vllm:0731-0.1.0

# on the head node:
./ventus-dspark up --worker 10.10.10.2 --model ~/models/v4-flash-0731
```

That's it. The launcher probes the IB fabric, validates the drafter geometry
against the checkpoint, generates the full validated profile, starts the
worker first, waits for health, and prints your endpoint:

```
[ventus-dspark] IB: rocep1s0f1 gid=3 eth=enp1s0f1np1
[ventus-dspark] dspark k=5 context=1048576 (from checkpoint config)
[ventus-dspark] serving: http://gx10-1:8000 — model /models/v4-flash-0731
```

## Requirements

- 2× DGX Spark (GB10, SM121), 128 GB unified memory each, joined by a
  ConnectX/InfiniBand link (RoCE v2) — the 156 GB checkpoint cannot run on one.
- Docker with compose v2 on both nodes, ssh from head to worker.
- The 0731 weights on **both** nodes (download once, copy over):
  `~/models/v4-flash-0731` (155 GB, `deepseek-ai/DeepSeek-V4-Flash-0731`).

## What it does for you

- **Auto-probes the IB HCA/GID.** Many Sparks have a second IB port with no
  usable RoCEv2 GID; NCCL round-robins into it and dies with a cryptic
  `ibv_modify_qp` error. The launcher picks the live port on both nodes.
- **Validates k from the checkpoint.** The model card says
  `num_speculative_tokens: 7`; the real `dspark_block_size` is 5, and k=7
  crashes on first generation. The launcher reads the checkpoint and refuses
  bad values with a clear message.
- **Ships the validated profile.** k=5, gmu 0.78 (0.80 dies on the first real
  request), 6 sequences, 8192 batched tokens, NVFP4 KV, B12X MoE, FlashInfer
  sampler — every knob that turned a booting model into a fast one.
- **Both APIs.** OpenAI-compatible (`/v1/chat/completions`, AIDE/Codex) and
  Anthropic-compatible (`/v1/messages`, Claude Code), tool calling and
  reasoning content included.

## Measured on a 2× DGX Spark fleet (2026-07-31)

| metric | value |
|---|---|
| context window | 1,048,576 (1M), KV pool 1,468,303 tokens → 1.4× concurrency at 1M |
| prefill | ~1,750 tok/s at 22.5K-token prompts |
| decode | 56.9 tok/s mean, 78.6 peak (Patch 4, ~60% draft acceptance) |
| boot | ~12–15 min cold, ~7 min warm (cached autotune) |
| APIs | health 200, tool calls, reasoning content, math, 1M `/v1/models` |

Full methodology and config: [docs/VALIDATION.md](docs/VALIDATION.md).

## Building the image yourself

No compiled code of our own — the image is a pinned public base plus a pure
Python overlay, so you can build (and audit) every layer:

```bash
./scripts/build.sh          # on each node, or ARM64 CI
```

Pins: base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`, engine
vLLM `0.21.1rc1.dev339`, overlay + Patch 4 vendored under `docker/` (MIT,
see [CREDITS.md](CREDITS.md)). The final image verifies itself with
`import vllm` before reporting success.

## How safe is it?

Honest answer: the *overlay and launcher* are fully auditable — our code plus
MIT-licensed patches, no hidden binaries. The *base image* is a community
build from the same lineage as MiaAI-Lab/Anemll/r0b0tlab (see CREDITS); we pin
it by tag today and will move to digest pins + cosign signatures + SBOM in CI
as the next step. The runtime runs offline (`HF_HUB_OFFLINE=1`) and binds only
local ports. We did not write any kernel code — see
[docs/VALIDATION.md](docs/VALIDATION.md) for what is and isn't proven.

## Layout

```
ventus-dspark.sh            the one command (up/down, --dry-run)
compose/ventus-dspark.yml   compose service (worker/head roles)
scripts/build.sh            reproducible image build
docker/                     vendored overlay + stage Dockerfiles (MIT)
docs/VALIDATION.md          measured numbers + methodology
```

## Roadmap

- digest-pinned base + SBOM + cosign signature
- ARM64 GitHub Actions build so `docker pull` is fully reproducible
- automatic weight bootstrap (`hf download`, resumable)
- upstream tracking: when vLLM merges SM12x support (PR #41834), swap the
  base for mainline and re-measure

## License

MIT (Ventus Works) for this repo's code; upstream work under its own licenses
— see [CREDITS.md](CREDITS.md). Model weights under DeepSeek's license.
