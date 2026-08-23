# Phase 1 Results

Measured 2026-08-23 on the Phase 0 host.

## Repacked store

- Format: expert-contiguous, 4 KiB-aligned safetensors segments
- Segments: global, dense layer, routed-expert layer, and MTP
- Resume rule: only checksum-verified, atomically renamed segments are skipped
- Layer 0 expert payload: 3,422,552,064 bytes
- Layer 0 expert tensors: 1,536
- Layer 0 expert bundles: 256 contiguous bundles of 13,369,344 bytes
- Layer 0 copy: 77.349 seconds
- Layer 0 destination verification: 17.712 seconds

## RAM cache

Expert `layer:0:expert:17`:

- Span: 13,369,344 bytes across six tensors
- First WSL filesystem load: 98.758 ms
- Three subsequent LRU hits: below 0.001 ms each
- Cache result: one miss, three hits

## Exact quantization

- NVIDIA E4M3 decode comparison: 256 of 256 bit patterns match
- NVIDIA E4M3 encode samples: all match
- DeepSeek per-128 BF16 to FP8 values: zero bit mismatches
- DeepSeek E8M0 activation scales: zero bit mismatches
- RTX 3070 activation quantization: 0.0082 ms for 4,096 values

## Complete expert

`layers.0.ffn.experts.0`, one decode token:

- Packed weights and scales: 13,369,344 bytes
- Cold allocation, upload, and weight preparation: 22.544 ms
- Resident packed-weight dequantization: 3.977 ms
- Resident `w1 + w3 + SwiGLU + w2` forward: 0.160 ms
- CPU-oracle maximum absolute error: 0.000977
- Outputs outside tolerance: 0 of 4,096

## Packed VRAM cache

- Cache unit: one 13,369,344-byte packed expert
- Initial budget: 4.5 GiB, approximately 361 experts
- Test budget: 32 MiB with two 24 MiB entries
- LRU eviction: pass
- Borrowed buffer remains valid after eviction: pass
- CUDA event protects upload and last recorded stream use: pass
