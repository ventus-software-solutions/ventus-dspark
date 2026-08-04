"""Anthropic /v1/messages reasoning passthrough.

DeepSeek-V4-Flash-0731 ships no Jinja chat_template. Reasoning is controlled
by the checkpoint's encoding_dsv4.py encoder, which reads `thinking` and
`reasoning_effort` out of `chat_template_kwargs`.

Anthropic clients (Claude Code) express the same intent as a top-level
`thinking: {"type": "enabled", "budget_tokens": N}` block. Nothing bridged
the two, and pydantic silently drops unmodelled fields — so Claude Code ran
with reasoning off no matter what it asked for, with no error.
"""

import pytest


def build(protocol, **overrides):
    payload = {
        "model": "/models/v4-flash-0731",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    }
    payload.update(overrides)
    return protocol.AnthropicMessagesRequest(**payload)


def test_thinking_block_is_modelled(anthropic_protocol):
    """The regression itself: an unmodelled field is silently discarded."""
    request = build(
        anthropic_protocol, thinking={"type": "enabled", "budget_tokens": 4096}
    )
    assert request.thinking is not None, "thinking block dropped on the floor"
    assert request.thinking.type == "enabled"
    assert request.thinking.budget_tokens == 4096


def test_enabled_thinking_reaches_the_encoder(anthropic_protocol):
    request = build(
        anthropic_protocol, thinking={"type": "enabled", "budget_tokens": 4096}
    )
    assert request.chat_template_kwargs == {"thinking": True, "reasoning_effort": "high"}


def test_disabled_thinking_reaches_the_encoder(anthropic_protocol):
    request = build(anthropic_protocol, thinking={"type": "disabled"})
    assert request.chat_template_kwargs == {"thinking": False}


@pytest.mark.parametrize(
    ("budget", "effort"),
    [
        (1024, "low"),
        (4095, "low"),
        (4096, "high"),
        (16383, "high"),
        (16384, "max"),
        (64000, "max"),
    ],
)
def test_budget_maps_to_effort(anthropic_protocol, budget, effort):
    request = build(
        anthropic_protocol, thinking={"type": "enabled", "budget_tokens": budget}
    )
    assert request.chat_template_kwargs["reasoning_effort"] == effort


def test_enabled_without_budget_uses_base_mode(anthropic_protocol):
    """`low` is the checkpoint's base reasoning mode: <think> with no prefix."""
    request = build(anthropic_protocol, thinking={"type": "enabled"})
    assert request.chat_template_kwargs == {"thinking": True, "reasoning_effort": "low"}


def test_explicit_chat_template_kwargs_wins(anthropic_protocol):
    """A caller who speaks vLLM directly is not second-guessed."""
    request = build(
        anthropic_protocol,
        thinking={"type": "enabled", "budget_tokens": 32000},
        chat_template_kwargs={"thinking": True, "reasoning_effort": "low"},
    )
    assert request.chat_template_kwargs == {"thinking": True, "reasoning_effort": "low"}


def test_partial_chat_template_kwargs_is_merged_not_replaced(anthropic_protocol):
    """Unrelated keys survive; only reasoning keys are filled in."""
    request = build(
        anthropic_protocol,
        thinking={"type": "enabled", "budget_tokens": 4096},
        chat_template_kwargs={"some_other_flag": 1},
    )
    assert request.chat_template_kwargs == {
        "some_other_flag": 1,
        "thinking": True,
        "reasoning_effort": "high",
    }


def test_no_thinking_block_leaves_kwargs_untouched(anthropic_protocol):
    """No thinking block means the server default applies, not an override."""
    request = build(anthropic_protocol)
    assert request.chat_template_kwargs is None
