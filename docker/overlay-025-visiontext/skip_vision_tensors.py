#!/usr/bin/env python3
"""Serve DeepSeek-V4-Flash-Vision-Exp TEXT-ONLY on the 0.25 lane.

REPRODUCIBLE FAILURE (2026-08-31, k=6 boot): the Vision-Exp checkpoint
declares plain DeepseekV4ForCausalLM but its shards carry 267 vision tensors
(vision.*, aligner.*, image_start/end/newline/pad embeds). The 0.25 engine
builds the text model and dies loading them:

    ValueError: There is no module or parameter named 'aligner' in
    DeepseekV4ForCausalLM

FIX: the model's own load_weights already skips spec-head tensors via
AutoWeightsLoader(skip_substrs=["mtp."]). Extend that list with the vision
prefixes. The text tower and MTP head load exactly as before; the model
serves as the text-equal sibling of 0731 that DeepSeek's card describes.

HONEST SCOPE: this is text-only BY CONSTRUCTION — image input still needs an
engine that builds the vision encoder and aligner. This patch exists so the
fleet can evaluate the text claims today; it must never be mistaken for
vision support.

Fails closed on drift, ambiguity, and double-patching.
"""
import argparse
import py_compile
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py")

OLD = 'AutoWeightsLoader(self, skip_substrs=["mtp."])'
# ".ffn.gate.bias" is a SUBSTRING match: it covers both the new per-layer
# router biases (.ffn.gate.bias) and their vision-conditioned twins
# (.ffn.gate.bias_vl), and cannot match the pre-existing
# .ffn.gate.e_score_correction_bias. CAVEAT, recorded deliberately: these 92
# tensors are MoE ROUTER BIASES added by the vision training — skipping them
# changes expert routing relative to how the checkpoint was trained. The
# quality probe against the 0731 baseline is the judge; if text quality is
# degraded, this lane cannot serve Vision-Exp faithfully and the verdict is
# "wait for real engine support", not "ship it".
NEW = ('AutoWeightsLoader(self, skip_substrs=["mtp.", "vision.", "aligner.", '
       '"image_start", "image_end", "image_newline", "image_pad", '
       '".ffn.gate.bias"])')


GUARD_OLD = """                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr("""

GUARD_NEW = """                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict and name.endswith(
                        (".gate.e_score_correction_bias", ".gate.bias_vl")
                    ):
                        # Vision-Exp ships a correction bias for every MoE
                        # layer, but the first num_hash_layers are hash-MoE
                        # and are built WITHOUT that parameter (the mapper
                        # renames the checkpoint's .gate.bias to this name).
                        # Dropping it here matches the module structure;
                        # non-hash layers still load theirs normally.
                        # bias_vl is the vision-conditioned router bias: no
                        # layer in this text-only engine has that parameter,
                        # and an inner loader re-iterates without the skip
                        # list, so it must be tolerated here as well.
                        continue
                    param = params_dict[name]
                    weight_loader = getattr("""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=TARGET)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    text = args.target.read_text(encoding="utf-8")
    if '"aligner."' in text or "hash-MoE" in text:
        raise SystemExit("REFUSED: already patched")
    n = text.count(OLD)
    if n != 1:
        raise SystemExit(f"REFUSED: expected exactly 1 loader site, found {n}")
    if not args.apply:
        print(f"DRY-RUN: would patch {args.target}")
        return
    n = text.count(GUARD_OLD)
    if n != 1:
        raise SystemExit(f"REFUSED: expected exactly 1 fallthrough loader site, found {n}")
    text = text.replace(OLD, NEW, 1).replace(GUARD_OLD, GUARD_NEW, 1)
    args.target.write_text(text, encoding="utf-8")
    py_compile.compile(str(args.target), doraise=True)
    print(f"PATCHED {args.target}: vision tensors skipped at load (text-only lane)")


if __name__ == "__main__":
    main()
