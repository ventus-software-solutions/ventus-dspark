# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "tool_reference",
        "thinking",
        "redacted_thinking",
    ]
    text: str | None = None
    # For image content
    source: dict[str, Any] | None = None
    # For tool use/result
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
    # For tool_reference content
    tool_name: str | None = None
    # For thinking content
    thinking: str | None = None
    signature: str | None = None
    # For redacted thinking content (safety-filtered by the API)
    data: str | None = None


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    defer_loading: bool | None = None

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicThinkingConfig(BaseModel):
    """Anthropic extended-thinking block.

    Clients (Claude Code) ask for reasoning with this. DeepSeek-V4-Flash-0731
    has no Jinja chat_template — its encoding_dsv4.py encoder reads
    `thinking` / `reasoning_effort` from chat_template_kwargs instead — so
    this has to be translated. See translate_thinking below.
    """

    type: Literal["enabled", "disabled"]
    budget_tokens: int | None = None


# budget_tokens is a continuous budget; the encoder takes three discrete
# levels. These thresholds are a judgement call, not an upstream mapping:
# Anthropic's floor is 1024, and Claude Code's common budgets cluster around
# 4k (normal) and 16k+ (deep). "low" is the encoder's base mode — it opens
# <think> with no effort prefix.
THINKING_BUDGET_THRESHOLDS = ((16384, "max"), (4096, "high"), (0, "low"))


def budget_to_reasoning_effort(budget_tokens: int | None) -> str:
    if budget_tokens is None:
        return "low"
    for floor, effort in THINKING_BUDGET_THRESHOLDS:
        if budget_tokens >= floor:
            return effort
    return "low"


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    thinking: AnthropicThinkingConfig | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @model_validator(mode="after")
    def translate_thinking(self) -> "AnthropicMessagesRequest":
        """Fold the Anthropic thinking block into chat_template_kwargs.

        Without this the block is silently dropped and the server-side
        --default-chat-template-kwargs wins, so a client asking for extended
        thinking gets none and sees no error.

        A caller who sets a key in chat_template_kwargs directly is speaking
        vLLM's own dialect and is not second-guessed; only absent keys are
        filled in.
        """
        if self.thinking is None:
            return self

        kwargs = dict(self.chat_template_kwargs or {})
        if self.thinking.type == "disabled":
            kwargs.setdefault("thinking", False)
        else:
            kwargs.setdefault("thinking", True)
            kwargs.setdefault(
                "reasoning_effort",
                budget_to_reasoning_effort(self.thinking.budget_tokens),
            )
        self.chat_template_kwargs = kwargs
        return self


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    signature: str | None = None

    # Message delta
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: "AnthropicMessagesResponse | None" = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicContextManagement(BaseModel):
    """Context management information for token counting."""

    original_input_tokens: int


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic messages.count_tokens request"""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None

    # vLLM-specific fields that are not in Anthropic spec
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic messages.count_tokens response"""

    input_tokens: int
    context_management: AnthropicContextManagement | None = None
