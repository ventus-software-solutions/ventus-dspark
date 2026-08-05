# Experiment backlog — speculative decoding (2026-08-04/05)

## Status of the first A/B attempt (2026-08-05 evening, paused)

Run with `VENTUS_K=7` +
`VENTUS_SPEC_EXTRA=',"rejection_sample_method":"block","num_speculative_tokens_per_batch_size":[[1,2,7],[3,8,5]]'`.

- k5 baseline banked (strict, temp 0, quality probes in
  `results/quality-k5.json`, cells in `results/ab-k5-{a,b}.json`):
  p2048c1 51.6 / p8192c1 67.0 / p2048c4 agg 112.7 / p8192c4 agg 79.1 /
  p256c6 agg 194.3 tok/s.
- Full combo (k7+block+schedule) FAILED the quality gate: temp-0 outputs
  nondeterministic even on a quiet endpoint (three identical requests,
  three different answers). Greedy output must be batch-shape-stable within
  a config; something in the combo breaks that. Prime suspect:
  `num_speculative_tokens_per_batch_size` (docs say untested with dspark;
  the DSpark speculator preallocates fixed-k buffers).
- Isolation ladder for next session: (1) k7+block, no schedule → probe
  determinism; (2) k7 alone; (3) if k7 alone fails, reject and keep k5.
  Cluster reverted to k5 defaults meanwhile.

Operational note from the same day: worker gx10-2 hard-hung after ~20h at
gmu 0.80 (NVRM out-of-memory kernel checks, then the GB10 full-host wedge);
the 025 lane default is 0.78 now. gx10-2's fabric port also has a chronic
DHCP-flap misconfig (a second NM profile, ~3.5k failures since Jul 31) —
fix pending. Add engine liveness monitoring: the head engine died with exit
code 0 and a oneshot systemd unit noticed nothing for 90 minutes.

Findings from a research + code-review + local-lab pass on raising DSpark
decode throughput. Nothing here is deployed; the fleet A/B below is ready to
run whenever the cluster can take one restart + ~40 min of benchmark cells.

## Ready to try (one fleet A/B session)

All three are config-only on the 025 lane, lossless by construction, and
revert by env. Keep whatever the benchmark confirms; the decision rule is
"no c4/c6 aggregate regression larger than the single-stream gain."

1. **k=7** — `num_speculative_tokens: 7`. DeepSeek's own model-card
   recommendation for this checkpoint. The launcher's k<=5 guard is a 0.21-era
   rule: that engine crashed on k>5 ("size of tensor a (7) must match tensor
   b (5)"); the 0.25 engine has no divisibility constraint — the parallel
   drafter simply extends its block (paper: +0.2–1.3% draft cost for up to
   +26% accepted length on code). Expected +10–15% single-stream. Remove the
   guard for the 025 lane only.
2. **Block verification** — add `"rejection_sample_method": "block"` to the
   speculative config. Implemented in our image (Sun et al. 2403.10444),
   provably never worse than token-level rejection, active only at temp>0.
   Local lab measured +3% on 2 of 3 lanes; literature says +5–8%.
3. **Batch-size→k schedule** —
   `"num_speculative_tokens_per_batch_size": [[1,2,7],[3,8,5]]` so k=7's
   single-stream win doesn't cost batched throughput (k>5 measured -12%-class
   effects at high concurrency elsewhere). Officially untested with dspark;
   that's what the A/B is for.

## Measured and buried

- **Draft-temperature decoupling** (our idea): the verifier reads q from the
  same processed logits the drafter samples from, so draft temperature is a
  free, lossless knob — proven empirically (TV ≈ 5e-5 across scales 0.5–1.5).
  But on a real drafter (Qwen3-4B + dspark_qwen3_4b_block7, RTX 3070 lab),
  ±30% scale moved acceptance less than run-to-run noise across all three
  workload lanes. Null result. Patch preserved at
  `scratchpad/patch_draft_temp.py` (session archive) if a future drafter
  wants retesting; do not spend fleet time on it.
- **KV offload / LMCache on unified memory**: measured regressions upstream
  (12x TTFT convoy #44294, block_size=256+specdec assert #48919, 0.25.1
  crash #50454) and no upside on GB10 — the "CPU tier" is the same DRAM.
  Spend memory on the GPU pool instead.
- **NCCL tuning for 2x Spark TP=2**: nobody has found wins; decode is
  compute/acceptance-bound at this scale. Baked config is already right.
- **Tree drafting / lossy acceptance / ngram+model hybrid**: not available
  in vLLM 0.25/0.26; dead ends for now.

- **Fusing the sequential Markov sampling loop**: profiled on the lab pair
  (CUDA events around backbone vs `_sample_sequential`) — 46% of draft time
  in eager mode, but that share is launch-overhead artifact: the same server
  drops 180→25 tok/s without CUDA graphs, and production graphs amortize the
  ~28 per-round kernel launches to ~100µs of on-GPU overhead against a much
  heavier DSV4F backbone. Estimated production gain ~1%; not worth building.
  Useful corollary: the draft BACKBONE forward is the production bottleneck —
  the fused draft-MLA kernel below is the lever that matters.

## Watch (act when upstream ships)

- **vLLM 0.27 fused draft-attention** (#50911, merged): 2.3–7.3x per-layer
  draft kernel speedup — the next engine-lane bump when a packaged SM121
  build exists (r0b0tlab's 0.26 lane is the likely vehicle; note #48137
  costs ~10.6% acceptance in 0.26 and needs cherry-care).
- **Confidence-scheduled verification** (#47808, open): the checkpoint ships
  a confidence head vLLM doesn't wire; upstream PR exists. Revisit on merge.
- **Suffix decoding** (`method=suffix`) for echo-heavy agent lanes
  (2.8–4x literature claims on repetitive agentic work): would be a second
  serving lane, not a replacement. Benchmark against the tool-call profile
  before committing.

## Correctness fixes queued separately (quality, not speed)

- 0731 encoder corrupts dict-shaped tool-call `arguments` on re-render
  (MiaAI recipe repo issue #21, 4-line fix) — hits agent traffic directly.
- `reasoning_content` silently dropped for thinks ≳100 tokens on the Anemll
  image (their issue #1) — undermines our /v1/messages thinking passthrough;
  verify on our endpoint, then patch.

## Local lab (reusable)

WSL2 lab on the operator PC (RTX 3070): vLLM 0.25.1 + Qwen3-4B-AWQ +
`deepseek-ai/dspark_qwen3_4b_block7` — same `method=dspark` code path as the
fleet. `~/run-cell.sh <tag> [spec-extra-json] [draft-temp-scale]` boots,
warms, measures acceptance/tok-s across three workload lanes (~6 min/cell).
Setup gotchas that cost an evening, so they're recorded: WSL kills backgrounded
processes with the session (keep the launcher process alive from the Windows
side); vLLM's `is_pin_memory_available()` blanket-blocks WSL though pinned
memory works (lab venv overrides it); the CUDA toolchain must be version-
consistent incl. torch's bundled nvrtc (nvcc/ptxas 13.3 wheels) and needs
`libcuda`/`libcudart` link shims from `/usr/lib/wsl/lib` and the venv.
Lanes reproduce the fleet's acceptance ordering (prose < code < predictable),
which is what makes relative effects transferable.
