# Credits

The runtime inside this container is assembled from public community work.
Please credit the upstream authors when reusing it.

- **tonyd2wild / Tony Deangelo** — Stage C runtime overlay + Patch 4
  (DSpark shared-expert loader fix), validated 0731 numbers.
  https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark (MIT)
- **drowzeys / Keys** — DSpark concurrency patch.
  https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- **Roady001 + Fable** — DSpark cold-start garble root-cause fix (Patch 3).
- **jasl** — SM12x enablement for DeepSeek V4 Flash in vLLM (PR #41834, unmerged upstream).
  https://github.com/vllm-project/vllm/pull/41834
- **MiaAI-Lab** — native DeepSeek-V4-Flash-Vision-Exp support for the 025 lane
  (vision_exp modules, vision + empty-encoder-output hotfixes; fetched @ 7963d432).
  https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- **MiaAI-Lab, Anemll, r0b0tlab, rafaelcaricio, bjk110** — the prebuilt runtime
  lineage this container builds on (base image
  `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`).
- **The vLLM project** — the serving engine itself (Apache-2.0).
- **DeepSeek** — the model weights (see their license).

The launcher, build wiring, IB probe, k-validation, and this documentation are
Ventus Works' own additions (MIT, see LICENSE).
