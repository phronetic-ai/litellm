"""Regression tests for Bedrock Converse ``toolSpec.strict`` forwarding.

Bedrock Converse routes Claude Opus 4.7/4.8 and Claude Sonnet 4 through an
Anthropic-compatible validator that rejects ``toolSpec.strict`` even though
Anthropic's native API accepts ``strict`` as a top-level tool field. See
BerriAI/litellm#31582.

That per-model gate only covers models whose cost-map entry carries the flag, so a
``strict: false`` that litellm itself synthesized still broke unflagged models. Since
``strict: false`` is the Chat Completions default, it is now dropped for every model
rather than forwarded as a no-op the provider can reject. See BerriAI/litellm#33193.

Forwarding ``strict: true`` is opt-in via ``litellm.bedrock_forward_strict_tools``:
Bedrock compiles a grammar per strict toolSpec and 400s with "Compiled grammar size
(...) exceeds maximum allowed size (300MB)" for schemas as small as a few optional
string/int fields. The per-model gate below only applies once that opt-in is set.
"""

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt
from litellm.llms.bedrock.common_utils import bedrock_converse_supports_strict_tools


@pytest.fixture
def forward_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "bedrock_forward_strict_tools", True)

_STRICT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "strict": True,
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }
]

_NON_STRICT_TOOL = [
    {
        "type": "function",
        "function": {
            **_STRICT_TOOL[0]["function"],
            "strict": False,
        },
    }
]


@pytest.mark.parametrize(
    "model_id",
    [
        "bedrock/us.anthropic.claude-opus-4-7",
        "bedrock/us.anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7-v1:0",
        "bedrock/eu.anthropic.claude-opus-4-8-v1:0",
        "bedrock/global.anthropic.claude-opus-4-7",
        # Sonnet 4 also rejects toolSpec.strict on Bedrock Converse
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/apac.anthropic.claude-sonnet-4-20250514-v1:0",
        # Sonnet 5 rejects it too, verified live against Bedrock in us-east-1
        "anthropic.claude-sonnet-5",
        "bedrock/us.anthropic.claude-sonnet-5",
        "bedrock/eu.anthropic.claude-sonnet-5",
        "bedrock/jp.anthropic.claude-sonnet-5",
    ],
)
@pytest.mark.usefixtures("forward_strict")
def test_bedrock_tools_pt_strict_dropped_for_strict_unsupported_models(
    model_id: str,
) -> None:
    """Opus 4.7/4.8, Sonnet 4 and Sonnet 5 reject toolSpec.strict and additionalProperties."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    tool_spec = result[0]["toolSpec"]
    assert (
        "strict" not in tool_spec
    ), f"strict leaked into toolSpec for {model_id}: {tool_spec}"
    assert (
        "additionalProperties" not in tool_spec["inputSchema"]["json"]
    ), f"additionalProperties leaked into toolSpec for {model_id}: {tool_spec}"


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-6",
        "bedrock/us.anthropic.claude-opus-4-6",
        "bedrock/us.anthropic.claude-opus-4-5",
    ],
)
@pytest.mark.usefixtures("forward_strict")
def test_bedrock_tools_pt_strict_kept_for_other_anthropic(model_id: str) -> None:
    """Sonnet 4.5/4.6 and Opus <=4.6 accept toolSpec.strict — forward it once opted in."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    assert (
        result[0]["toolSpec"]["strict"] is True
    ), f"strict missing for {model_id}: {result[0]['toolSpec']}"


_INCIDENT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "strict": True,
            "description": "Search tools",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "skip": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 50},
                },
                "additionalProperties": False,
            },
        },
    }
]


@pytest.mark.parametrize(
    "model_id",
    [
        "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-6",
        "bedrock/us.anthropic.claude-opus-4-6",
        "bedrock/global.anthropic.claude-sonnet-5",
    ],
)
def test_bedrock_tools_pt_strict_dropped_by_default_for_every_model(model_id: str) -> None:
    """Regression: with no opt-in, ``strict`` and ``additionalProperties`` never reach a
    Bedrock toolSpec, even for Anthropic models whose cost-map entry allows strict.
    Forwarding this three-optional-field tool with ``strict: true`` made Bedrock 400
    with a 332MB compiled grammar."""
    assert litellm.bedrock_forward_strict_tools is False
    assert bedrock_converse_supports_strict_tools(model_id) is False
    tool_spec = _bedrock_tools_pt(_INCIDENT_TOOL, model=model_id)[0]["toolSpec"]
    assert "strict" not in tool_spec, f"strict leaked into toolSpec for {model_id}: {tool_spec}"
    assert "additionalProperties" not in tool_spec["inputSchema"]["json"]
    assert tool_spec["inputSchema"]["json"]["properties"] == _INCIDENT_TOOL[0]["function"]["parameters"]["properties"]


@pytest.mark.usefixtures("forward_strict")
def test_bedrock_tools_pt_opt_in_restores_strict_forwarding() -> None:
    """The opt-in flips the default back to the per-model gate."""
    model_id = "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
    assert bedrock_converse_supports_strict_tools(model_id) is True
    tool_spec = _bedrock_tools_pt(_INCIDENT_TOOL, model=model_id)[0]["toolSpec"]
    assert tool_spec["strict"] is True
    assert tool_spec["inputSchema"]["json"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "model_id",
    [
        "bedrock/us.anthropic.claude-sonnet-5",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-6",
        "bedrock/us.anthropic.claude-opus-4-8",
    ],
)
def test_bedrock_tools_pt_falsy_strict_always_dropped(model_id: str) -> None:
    """``strict: false`` is the Chat Completions default, so forwarding it says nothing
    the provider does not already assume. Bedrock Converse rejects the key's presence
    for a growing set of Claude models, so it is dropped for every model, including the
    ones whose cost-map entry still allows ``strict: true`` through."""
    result = _bedrock_tools_pt(_NON_STRICT_TOOL, model=model_id)
    tool_spec = result[0]["toolSpec"]
    assert (
        "strict" not in tool_spec
    ), f"no-op strict: false leaked into toolSpec for {model_id}: {tool_spec}"


def test_responses_bridge_function_tool_does_not_reach_bedrock_with_strict() -> None:
    """The Responses-to-Chat-Completions bridge stamps ``strict: false`` onto every
    function tool even when the caller never sent one, which is how Codex CLI requests
    acquired the key. Assert the fabricated value does not survive to toolSpec."""
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    responses_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    chat_tools, _ = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            [responses_tool]
        )
    )
    result = _bedrock_tools_pt(chat_tools, model="bedrock/us.anthropic.claude-sonnet-5")
    assert "strict" not in result[0]["toolSpec"]


@pytest.mark.parametrize(
    "model_id",
    [
        "us.amazon.nova-micro-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
    ],
)
def test_bedrock_tools_pt_strict_dropped_for_non_anthropic(model_id: str) -> None:
    """Non-Anthropic Bedrock families reject toolSpec.strict — must be dropped."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    assert "strict" not in result[0]["toolSpec"]


@pytest.mark.usefixtures("forward_strict")
def test_bedrock_converse_supports_strict_tools_helper() -> None:
    """Direct check for the per-model gate helper used by factory.py, with the opt-in set."""
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-opus-4-7")
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-opus-4-8")
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools(
            "anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        is True
    )
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-opus-4-6")
        is True
    )
    assert bedrock_converse_supports_strict_tools("us.amazon.nova-micro-v1:0") is False
    assert bedrock_converse_supports_strict_tools("") is False
    # Sonnet 4 also rejects strict on Bedrock Converse
    assert (
        bedrock_converse_supports_strict_tools(
            "anthropic.claude-sonnet-4-20250514-v1:0"
        )
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools(
            "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        is False
    )
    assert bedrock_converse_supports_strict_tools("anthropic.claude-sonnet-5") is False
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-sonnet-5")
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
        is True
    )


@pytest.mark.parametrize(
    "cost_map_key",
    [
        "anthropic.claude-opus-4-7",
        "us.anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "eu.anthropic.claude-sonnet-4-20250514-v1:0",
        "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-sonnet-5",
        "au.anthropic.claude-sonnet-5",
        "jp.anthropic.claude-sonnet-5",
    ],
)
def test_strict_tools_flag_set_in_model_cost_map(cost_map_key: str) -> None:
    """The gate is driven by ``bedrock_converse_supports_strict_tools: false`` in
    ``model_prices_and_context_window.json``, not hardcoded model patterns."""
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    cost_map = GetModelCostMap.load_local_model_cost_map()
    assert cost_map[cost_map_key]["bedrock_converse_supports_strict_tools"] is False
