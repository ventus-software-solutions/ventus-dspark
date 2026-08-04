"""Prometheus scraping in scripts/benchmark.py.

vLLM exposes three counter families per spec-decode quantity:

    spec_decode_num_accepted_tokens_total          the cumulative count
    spec_decode_num_accepted_tokens_created        a unix timestamp
    spec_decode_num_accepted_tokens_per_pos_total  per draft position

Only the first is the number we want. Matching on substring pulls in the
other two, which turns acceptance into epoch-seconds plus a double count —
a number that looks like a number and is not one.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Captured from a live 2x DGX Spark head node, trimmed to the relevant lines.
LIVE_METRICS = """\
# HELP vllm:spec_decode_num_drafts_total Number of drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="/models/v4-flash-0731"} 2.04363e+06
vllm:spec_decode_num_drafts_created{engine="0",model_name="/models/v4-flash-0731"} 1.785538744138251e+09
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="/models/v4-flash-0731"} 1.0218118e+07
vllm:spec_decode_num_draft_tokens_created{engine="0",model_name="/models/v4-flash-0731"} 1.7855387441389015e+09
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="/models/v4-flash-0731"} 7.268642e+06
vllm:spec_decode_num_accepted_tokens_created{engine="0",model_name="/models/v4-flash-0731"} 1.785538744138932e+09
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="/models/v4-flash-0731",position="0"} 1.883866e+06
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="/models/v4-flash-0731",position="1"} 1.732372e+06
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="/models/v4-flash-0731",position="2"} 1.489894e+06
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="/models/v4-flash-0731",position="3"} 1.204113e+06
"""

ACCEPTED = 7.268642e06
DRAFTED = 1.0218118e07


@pytest.fixture(scope="session")
def benchmark():
    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("ventus_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ventus_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_running(benchmark):
    """asyncio.run() at module scope would fire the whole sweep on import."""
    assert callable(benchmark.parse_spec_decode)


def test_picks_only_the_cumulative_totals(benchmark):
    assert benchmark.parse_spec_decode(LIVE_METRICS) == (ACCEPTED, DRAFTED)


def test_created_timestamps_are_not_counted(benchmark):
    accepted, drafted = benchmark.parse_spec_decode(LIVE_METRICS)
    # A _created epoch value would swamp the real count by two orders.
    assert accepted < 1e8
    assert drafted < 1e8


def test_per_position_breakdown_is_not_double_counted(benchmark):
    accepted, _ = benchmark.parse_spec_decode(LIVE_METRICS)
    per_pos = 1.883866e06 + 1.732372e06 + 1.489894e06 + 1.204113e06
    assert accepted != pytest.approx(ACCEPTED + per_pos)
    assert accepted == pytest.approx(ACCEPTED)


def test_multiple_engines_are_summed(benchmark):
    text = LIVE_METRICS + (
        'vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="m"} 1000.0\n'
        'vllm:spec_decode_num_draft_tokens_total{engine="1",model_name="m"} 2000.0\n'
    )
    assert benchmark.parse_spec_decode(text) == (ACCEPTED + 1000.0, DRAFTED + 2000.0)


def test_missing_counters_report_absent_not_zero(benchmark):
    assert benchmark.parse_spec_decode("# nothing here\n") is None


def test_acceptance_ratio(benchmark):
    before = (1000.0, 5000.0)
    after = (1600.0, 6000.0)
    assert benchmark.acceptance_between(before, after) == pytest.approx(0.6)


def test_acceptance_is_none_when_no_drafts_ran(benchmark):
    assert benchmark.acceptance_between((10.0, 20.0), (10.0, 20.0)) is None
