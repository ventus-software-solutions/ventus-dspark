"""Load the vendored vLLM overlay modules by path.

The overlay under docker/overlay/ is copied into the image at build time and
is not an importable package from the repo root, so tests load it directly.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

OVERLAY = Path(__file__).resolve().parent.parent / "docker" / "overlay"


def load_overlay_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, OVERLAY / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Both engine lanes carry the same thinking-translation patch; every
# protocol test runs against each so a port can't silently drift.
OVERLAY_ROOT = Path(__file__).resolve().parent.parent / "docker"
PROTOCOL_LANES = {
    "v021": "overlay/vllm/entrypoints/anthropic/protocol.py",
    "v025": "overlay-025/vllm/entrypoints/anthropic/protocol.py",
}


@pytest.fixture(scope="session", params=sorted(PROTOCOL_LANES))
def anthropic_protocol(request):
    relative = PROTOCOL_LANES[request.param]
    spec = importlib.util.spec_from_file_location(
        f"ventus_anthropic_protocol_{request.param}", OVERLAY_ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
