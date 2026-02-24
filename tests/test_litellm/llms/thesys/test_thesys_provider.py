"""
Tests for Thesys provider
"""

import pytest
import respx

import litellm
from litellm import completion


@pytest.fixture
def thesys_response():
    """Mock response from Thesys API"""
    return {
        "id": "chatcmpl-thesys-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "c1-nightly",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_thesys():
    """Test that get_llm_provider correctly identifies thesys provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("thesys/c1-nightly")
    assert model == "c1-nightly"
    assert provider == "thesys"
    assert api_base == "https://api.thesys.dev/v1/embed"


def test_thesys_in_provider_lists():
    """Test that thesys is registered in all necessary provider lists"""
    assert "thesys" in litellm.openai_compatible_providers
    assert "thesys" in litellm.provider_list


def test_thesys_config_provider_name():
    """Test that ThesysChatConfig returns correct provider name"""
    cfg = litellm.ThesysChatConfig()
    assert cfg.custom_llm_provider == "thesys"


def test_thesys_api_base_from_env(monkeypatch):
    """Test that THESYS_API_BASE env var overrides default"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("THESYS_API_BASE", "https://custom.thesys.dev/v1")
    _, _, _, api_base = get_llm_provider("thesys/c1-nightly")
    assert api_base == "https://custom.thesys.dev/v1"


def test_thesys_sync_completion(respx_mock, thesys_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("THESYS_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.thesys.dev/v1/embed/chat/completions").respond(
        json=thesys_response
    )

    response = completion(
        model="thesys/c1-nightly",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert str(request.url) == "https://api.thesys.dev/v1/embed/chat/completions"
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_thesys_async_completion(respx_mock, thesys_response, monkeypatch):
    """Test async completion call"""
    monkeypatch.setenv("THESYS_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.thesys.dev/v1/embed/chat/completions").respond(
        json=thesys_response
    )

    response = await litellm.acompletion(
        model="thesys/c1-nightly",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert str(request.url) == "https://api.thesys.dev/v1/embed/chat/completions"
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"
