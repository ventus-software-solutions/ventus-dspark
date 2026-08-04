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


@pytest.fixture(scope="session")
def anthropic_protocol():
    return load_overlay_module(
        "vllm/entrypoints/anthropic/protocol.py", "ventus_anthropic_protocol"
    )
