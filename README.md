# ventus-dspark

**One command to serve DeepSeek-V4-Flash-0731 on two NVIDIA DGX Spark (GB10)
machines — 1M context, DSpark speculative decoding, OpenAI + Anthropic
compatible APIs. No .env editing, no kernel patches, no build on your side.**

```bash
# on both nodes: get the image (or build it — see below)
docker pull ghcr.io/ventus-software-solutions/dspark-vllm:0731-0.1.0

# on the head node:
./ventus-dspark up --worker 192.168.1.2 --model ~/models/v4-flash-0731
```

That's it. The launcher probes the IB fabric, validates the drafter geometry
against the checkpoint, generates the full validated profile, starts the
worker first, waits for health, and prints your endpoint:

```
[ventus-dspark] IB: rocep1s0f1 gid=3 eth=enp1s0f1np1
[ventus-dspark] dspark k=5 context=1048576 (from checkpoint config)
[ventus-dspark] serving: http://head-node:8000 — model /models/v4-flash-0731
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
- **Reasoning that actually arrives.** 0731 ships no Jinja chat template —
  reasoning is driven by the checkpoint's own encoder via
  `chat_template_kwargs`. Anthropic clients send a `thinking` block instead,
  so the overlay translates it; without that, Claude Code asks for extended
  thinking and silently gets none. See [Reasoning effort](#reasoning-effort).

## Measured on a 2× DGX Spark fleet (2026-07-31)

| metric | value |
|---|---|
| context window | 1,048,576 (1M), KV pool 1,691,551 tokens (0.25 engine) |
| prefill | ~1,750–1,990 tok/s single-stream |
| decode | 72–81 tok/s single-stream; ~194 tok/s aggregate at 6 streams |
| draft acceptance | ~0.56 real code, ~0.70–0.88 predictable text — decode tracks this |
| boot | ~12–15 min cold, ~5 min warm (cached autotune) |
| APIs | health 200, tool calls, reasoning content, math, 1M `/v1/models` |

Full methodology and config: [docs/VALIDATION.md](docs/VALIDATION.md).

Reproduce on your own fleet:

```bash
./scripts/benchmark.py --output results/ventus-dspark-0731.json
```

Two profiles: `--profile strict` (default — temperature 0, three trials,
median, one discarded warmup) and `--profile compat`, which reproduces the
sampling and single-trial conditions other published DGX Spark tables use so
the numbers are comparable. `--corpus code` swaps the repeated filler prompt
for real source text; the gap between the two corpora is how much DSpark
acceptance the filler was inflating. Draft acceptance is read from
`/metrics` per case, because acceptance is what explains a decode number.

## Reasoning effort

The 0731 checkpoint has no Jinja `chat_template`. `--tokenizer-mode
deepseek_v4` calls the checkpoint's `encoding/encoding_dsv4.py`, which takes
`off`, `low`, `high` or `max` — `low` is the base mode (opens `<think>` with
no effort prefix).

Set the server default at launch:

```bash
./ventus-dspark up --worker 192.168.1.2 --thinking high
```

Default is `off`. Request-level settings always win over it.

- **OpenAI path** — nest it, don't use the top-level `reasoning_effort`
  field; vLLM does not route that into the encoder:
  `"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}`
- **Anthropic path** — send the normal Anthropic block, and the overlay
  translates it: `"thinking": {"type": "enabled", "budget_tokens": 8192}`.
  Budget maps to effort at 4096 (`high`) and 16384 (`max`); below that,
  `low`. Setting `chat_template_kwargs` explicitly overrides the mapping.

## Tests

```bash
pip install pytest pydantic && python -m pytest tests/ -q
```

Overlay unit tests are pure pydantic — no GPU, no cluster, no image.

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

## Layout

```
ventus-dspark               the one command (up/down, --dry-run)
compose/ventus-dspark.yml   compose service (worker/head roles)
scripts/build.sh            reproducible image build
scripts/benchmark.py        prefill/decode/acceptance sweep (stdlib only)
docker/                     vendored overlay + stage Dockerfiles (MIT)
tests/                      overlay unit tests (no GPU required)
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

---

Maintained by [Ventus Software Solutions](https://ventus.works)
