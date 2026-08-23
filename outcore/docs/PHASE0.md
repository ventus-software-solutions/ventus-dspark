# Phase 0 Results

Measured 2026-08-23 on Windows 11, WSL2, and Docker Desktop.

## Host

- GPU: RTX 3070, 8 GiB, compute capability 8.6
- RAM: 64 GiB
- `C:` Samsung 980 PRO NVMe, 57 GiB free
- `D:` SanDisk SATA SSD, 575 GiB free
- `E:` 8 TB HDD, checkpoint archive
- CUDA container: `nvidia/cuda:13.0.2-devel-ubuntu24.04`

## Checkpoint

`E:\model-archive\deepseek-v4-flash` was read without modification.

- 46 shards
- 69,187 tensors
- 159,609,485,896 tensor bytes (148.648 GiB)
- 11,264 routed-expert bundles
- 13,369,344 bytes per routed expert
- 140.250 GiB routed experts

The generated manifest is
`D:\ventus-outcore-cache\manifests\deepseek-v4-flash.v1.jsonl`.

## Baselines

| Measurement | Result |
|---|---:|
| Native Windows direct read, `D:` SSD | 0.396 GiB/s |
| Native Windows direct read, `E:` HDD | 0.094 GiB/s |
| WSL2 direct read, `D:` SSD | 0.221 GiB/s |
| Docker SSD to pinned RAM to VRAM | 0.226 GiB/s |
| Pinned RAM to VRAM | 10.951 GiB/s |
| VRAM to pinned RAM | 11.290 GiB/s |
| FP16 2048x128x4096 GEMM | 29.781 TFLOP/s |

The Docker pager averaged 55.215 ms for one 12.75 MiB expert bundle. File access,
not PCIe transfer, is the current bottleneck.

## Correctness gates

- Synthetic safetensors offset, shape, and dtype test: pass
- Full checkpoint tensor and bundle manifest validation: pass
- Expert 0 repack and full byte comparison: pass
- FP4 E2M1 and E8M0 CPU oracle: pass
- GPU unpack of `layers.0.ffn.experts.0.w1`: 8,388,608 FP16 values, zero bit mismatches
- GPU FP4 unpack time for that matrix: 1.272 ms
