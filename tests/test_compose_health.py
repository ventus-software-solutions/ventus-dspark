"""The compose healthcheck must probe the port the engine actually serves, and
must not condemn the headless worker rank.

Lived 2026-09-02: both ranks showed "unhealthy" (failing streak 10) while the
brain served fine — the probe read VLLM_PORT from the container env, which
compose never populates, and fell back to :8888. The worker has no API at all.
Rendered through `docker compose config`, exactly as the launcher does.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COMPOSE_FILES = ["compose/ventus-dspark-025.yml", "compose/ventus-dspark.yml"]

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
    reason="docker compose not available",
)


def render(compose_file, **env):
    full = dict(os.environ, DSPARK_VLLM_IMAGE="img:test", VLLM_PORT="8000", **env)
    r = subprocess.run(["docker", "compose", "-f", str(REPO / compose_file), "config", "--format", "json"],
                       capture_output=True, text=True, env=full, cwd=REPO)
    assert r.returncode == 0, r.stderr
    service = next(iter(json.loads(r.stdout)["services"].values()))
    return service["healthcheck"]["test"]


@pytest.mark.parametrize("compose_file", COMPOSE_FILES)
def test_head_probe_targets_the_served_port(compose_file):
    test = render(compose_file)
    probe = " ".join(test)
    assert "127.0.0.1:8000/v1/models" in probe, probe
    assert "8888" not in probe, probe
    assert "exit 0" not in probe, "head must actually be probed"


@pytest.mark.parametrize("compose_file", COMPOSE_FILES)
def test_headless_worker_rank_passes_by_construction(compose_file):
    test = render(compose_file, HEADLESS="1")
    assert test[0] == "CMD-SHELL", test
    assert test[1].lstrip().startswith("exit 0"), test
