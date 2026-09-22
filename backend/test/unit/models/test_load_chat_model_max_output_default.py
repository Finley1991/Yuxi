"""load_chat_model 为未显式配置的模型下发 64K max_output 下限。

DeepSeek/GLM 等 provider 默认仅 4K 输出，生成 write_file 等 tool_call
arguments 时会被截断成 invalid_tool_calls，用户侧表现为
"arguments were malformed or truncated"。这里验证 load_chat_model
在调用方未传 max_output 时为三个 provider 分支分别下发正确参数名。
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from langchain_core.messages import HumanMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yuxi.models.chat import load_chat_model
from yuxi.models.providers.cache import ModelInfo


def _make_info(provider_type: str, provider_id: str = "test-provider") -> ModelInfo:
    return ModelInfo(
        provider_id=provider_id,
        model_id="test-model",
        model_type="chat",
        display_name="Test",
        api_key="test-key",
        base_url="https://example.com/v1",
        provider_type=provider_type,
    )


def _intercept_request():
    """返回 (transport, bodies) 捕获实际下发的请求 body。"""
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(respond)
    return transport, bodies


def test_load_chat_model_sets_default_max_completion_tokens_for_openai(monkeypatch):
    info = _make_info("openai")
    monkeypatch.setattr("yuxi.models.chat.model_cache.get_model_info", lambda _: info)
    transport, bodies = _intercept_request()

    model = load_chat_model(
        info.spec,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    model.invoke([HumanMessage("hi")])

    assert bodies, "no request captured"
    assert bodies[0]["max_completion_tokens"] == 65_536


def test_load_chat_model_respects_explicit_max_completion_tokens(monkeypatch):
    info = _make_info("openai")
    monkeypatch.setattr("yuxi.models.chat.model_cache.get_model_info", lambda _: info)
    transport, bodies = _intercept_request()

    model = load_chat_model(
        info.spec,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
        max_completion_tokens=1_024,
    )
    model.invoke([HumanMessage("hi")])

    assert bodies[0]["max_completion_tokens"] == 1_024


def test_load_chat_model_sets_default_max_tokens_for_anthropic(monkeypatch):
    info = _make_info("anthropic")
    monkeypatch.setattr("yuxi.models.chat.model_cache.get_model_info", lambda _: info)
    monkeypatch.setattr(
        "yuxi.models.chat.get_docker_safe_url",
        lambda url: "https://api.anthropic.com",
    )

    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", FakeChatAnthropic)
    load_chat_model(info.spec, max_retries=0)

    assert captured.get("max_tokens") == 65_536
    assert "max_completion_tokens" not in captured
    assert "max_output_tokens" not in captured


def test_load_chat_model_sets_default_max_output_tokens_for_gemini(monkeypatch):
    info = _make_info("gemini")
    monkeypatch.setattr("yuxi.models.chat.model_cache.get_model_info", lambda _: info)

    captured = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        FakeChatGoogleGenerativeAI,
    )
    load_chat_model(info.spec, max_retries=0)

    assert captured.get("max_output_tokens") == 65_536
    assert "max_completion_tokens" not in captured
    assert "max_tokens" not in captured
