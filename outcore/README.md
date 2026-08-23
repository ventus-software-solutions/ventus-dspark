# Ventus Outcore

Experimental C++/CUDA runtime for exact inference with models larger than RAM and VRAM.

The first target is DeepSeek-V4-Flash on Windows with an RTX 3070 (8 GB), 64 GB RAM,
and SSD-backed weight paging. GPU kernels perform model math. The CPU schedules I/O,
caches, and transfers.

## Phase 0

Build and test the dependency-free safetensors reader in WSL2:

```sh
cd /mnt/c/git/ventus-products/ventus-dspark/.worktrees/outcore-windows-phase0/outcore
make test
make inspect MODEL=/mnt/e/model-archive/deepseek-v4-flash
```

The source checkpoint on `E:` remains read-only. Repacked hot bundles will live on `D:`.

Repack one layer, then resume the complete model conversion:

```sh
make build/outcore-repack
./build/outcore-repack \
  --model /mnt/e/model-archive/deepseek-v4-flash \
  --output /mnt/d/ventus-outcore-cache/deepseek-v4-flash-v1 \
  --layer 0
./build/outcore-repack \
  --model /mnt/e/model-archive/deepseek-v4-flash \
  --output /mnt/d/ventus-outcore-cache/deepseek-v4-flash-v1 \
  --all
```

Completed segments are checksum-verified and atomically renamed. A resumed run skips them.

Compile and run the CUDA baseline in the pinned development container:

```powershell
docker run --rm --gpus all `
  -v "${PWD}:/workspace" -w /workspace/outcore `
  nvidia/cuda:13.0.2-devel-ubuntu24.04 make cuda-bench
```

## Inference contract

- Preserve checkpoint values; no new quantization in the baseline.
- Run attention, routing, experts, normalization, and sampling on the GPU.
- Use CPU only for file I/O, scheduling, caching, and transfers.
- Missing prefetches may stall but never change selected experts.
- Establish correctness before throughput optimization.

## Memory hierarchy

- `D:` stores the complete expert-contiguous model.
- RAM uses a byte-budgeted LRU, targeting 40–48 GiB only when host headroom permits.
- Two small pinned buffers stage asynchronous GPU transfers.
- VRAM stores packed experts; weight tiles are expanded only for computation.
- Initial VRAM budget is 6 GiB, leaving WDDM and CUDA headroom.
