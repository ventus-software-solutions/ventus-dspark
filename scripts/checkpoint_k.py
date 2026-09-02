#!/usr/bin/env python3
"""Drafter geometry from a checkpoint's config.json.

    python3 scripts/checkpoint_k.py <config.json> <engine> [requested_k]

Prints "block nextn k" on success; on an illegal k prints the reason to
stderr and exits 2, so the launcher fails before the engine does.

Rules (measured; docs/VALIDATION.md, docs/EXPERIMENTS.md):
- block = dspark_block_size, the drafter geometry. The model card's
  num_speculative_tokens (7) is NOT it.
- nextn = num_nextn_predict_layers. Above 1 the MTP head is a chain and the
  engine rejects any k that is not a multiple of it at boot
  ("num_speculative_tokens: 7 must be divisible by n_predict=3", Vision-Exp,
  2026-08-31). k defaults to the first multiple of nextn at or above block:
  6 for Vision-Exp, the validated value.
- nextn == 1: k = 7 on the 025 family when block is 5 (+8.8% decode with
  byte-identical greedy output, 2026-08-06); 021 keeps block (crashes above 5).
"""
import json
import sys


def geometry(cfg: dict, engine: str, requested=None):
    block = int(cfg.get("dspark_block_size", 5))
    nextn = int(cfg.get("num_nextn_predict_layers", 1))
    if requested is None:
        if nextn > 1:
            k = -(-block // nextn) * nextn
        else:
            k = 7 if engine != "021" and block == 5 else block
    else:
        k = int(requested)
        if engine == "021" and k > 5:
            raise ValueError(f"k={k} unsupported on the 021 lane (crashes >5)")
        if k < block:
            raise ValueError(f"k={k} < drafter block {block}")
        if nextn > 1 and k % nextn:
            legal = ", ".join(str(m) for m in range(nextn, 4 * nextn + 1, nextn) if m >= block)
            raise ValueError(
                f"k={k} is not a multiple of num_nextn_predict_layers={nextn} "
                f"(the engine rejects it at boot); legal: {legal}")
    return block, nextn, k


def main(argv):
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    cfg = json.load(open(argv[1]))
    try:
        block, nextn, k = geometry(cfg, argv[2], argv[3] if len(argv) == 4 else None)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(block, nextn, k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
