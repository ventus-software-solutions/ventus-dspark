"""Drafter geometry + engine-lane resolution, offline.

Why this exists: the 2026-08-31 Vision-Exp boot crashed with
"num_speculative_tokens: 7 must be divisible by n_predict=3" — the launcher
derived k from dspark_block_size alone and never looked at the checkpoint's
chained MTP head. These tests pin the rule so no lane can regress it.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "scripts" / "checkpoint_k.py"
LAUNCHER = REPO / "ventus-dspark"

spec = importlib.util.spec_from_file_location("checkpoint_k", MODULE)
ck = importlib.util.module_from_spec(spec)
sys.modules["checkpoint_k"] = ck
spec.loader.exec_module(ck)

FLASH_0731 = {"dspark_block_size": 5, "num_nextn_predict_layers": 1}
VISION_EXP = {"dspark_block_size": 5, "num_nextn_predict_layers": 3}


def test_0731_keeps_the_measured_025_default_of_7():
    assert ck.geometry(FLASH_0731, "025") == (5, 1, 7)
    assert ck.geometry(FLASH_0731, "025-vision") == (5, 1, 7)


def test_021_lane_stays_at_the_checkpoint_block():
    assert ck.geometry(FLASH_0731, "021") == (5, 1, 5)


def test_chained_mtp_head_takes_first_multiple_of_nextn_at_or_above_block():
    # Vision-Exp: block 5, nextn 3 -> 6 (validated 2026-08-31), on any lane.
    assert ck.geometry(VISION_EXP, "025-vision") == (5, 3, 6)
    assert ck.geometry(VISION_EXP, "025") == (5, 3, 6)


def test_override_not_a_multiple_of_nextn_is_rejected_before_boot():
    with pytest.raises(ValueError, match="multiple of num_nextn_predict_layers=3"):
        ck.geometry(VISION_EXP, "025-vision", requested=7)


def test_legal_overrides_pass_through():
    assert ck.geometry(VISION_EXP, "025-vision", requested=9)[2] == 9
    assert ck.geometry(FLASH_0731, "025", requested=5)[2] == 5


def test_override_below_block_is_rejected():
    with pytest.raises(ValueError, match="< drafter block"):
        ck.geometry(VISION_EXP, "025-vision", requested=3)


def test_021_rejects_k_above_5():
    with pytest.raises(ValueError, match="021 lane"):
        ck.geometry(FLASH_0731, "021", requested=7)


def test_cli_prints_block_nextn_k_and_fails_loud(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(VISION_EXP))
    ok = subprocess.run([sys.executable, str(MODULE), str(cfg), "025-vision"],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.split() == ["5", "3", "6"]
    bad = subprocess.run([sys.executable, str(MODULE), str(cfg), "025-vision", "7"],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "multiple of" in bad.stderr


def _resolve_engine(engine):
    """Evaluate the launcher's resolve_engine() in isolation."""
    script = (
        'fatal() { echo "FATAL: $*"; exit 1; }; '
        f'SCRIPT_DIR="{REPO.as_posix()}"; IMAGE=""; GPU_MEMORY_UTILIZATION=""; ENGINE="{engine}"; '
        f'eval "$(sed -n \'/^resolve_engine()/,/^}}/p\' "{LAUNCHER.as_posix()}")"; '
        'resolve_engine && echo "$IMAGE|$COMPOSE|$GPU_MEMORY_UTILIZATION"'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_vision_lane_defaults_to_the_vision_image_on_the_025_compose():
    r = _resolve_engine("025-vision")
    assert r.returncode == 0, r.stdout + r.stderr
    image, compose, gmu = r.stdout.strip().split("|")
    assert image == "ghcr.io/ventus-software-solutions/dspark-vllm:0731-025-vision-0.1.0"
    assert compose.endswith("compose/ventus-dspark-025.yml")
    assert gmu == "0.78"


def test_unknown_lane_fails_loud():
    r = _resolve_engine("026")
    assert r.returncode != 0 and "FATAL" in r.stdout
