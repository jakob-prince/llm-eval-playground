"""Tests for model adapters: FakeAdapter, build_adapter routing, and BYO-key adapters."""
from __future__ import annotations

from llm_eval.adapters import (
    AnthropicAdapter,
    FakeAdapter,
    ModelConfig,
    OpenAIAdapter,
    build_adapter,
)

# ---------------------------------------------------------------------------
# FakeAdapter
# ---------------------------------------------------------------------------


def test_fake_adapter_returns_completion_without_error() -> None:
    adapter = FakeAdapter()
    completion = adapter.complete(prompt="hello", config=ModelConfig("fake", "alpha"))
    assert completion.error is None
    assert completion.text != ""


def test_fake_adapter_text_contains_display_label() -> None:
    config = ModelConfig("fake", "my-model", label="custom-label")
    completion = FakeAdapter().complete(prompt="test", config=config)
    assert "custom-label" in completion.text


def test_fake_adapter_truncates_long_prompt() -> None:
    # FakeAdapter embeds " ".join(prompt.split())[:160] in the output.
    long_prompt = "word " * 80  # ~400 chars, well over the 160-char snippet limit
    completion = FakeAdapter().complete(prompt=long_prompt, config=ModelConfig("fake", "m"))
    marker = "You asked: "
    idx = completion.text.index(marker) + len(marker)
    snippet = completion.text[idx:]
    assert len(snippet) <= 160


# ---------------------------------------------------------------------------
# build_adapter routing
# ---------------------------------------------------------------------------


def test_build_adapter_missing_key_falls_back_to_fake() -> None:
    assert isinstance(build_adapter("openai", None), FakeAdapter)


def test_build_adapter_empty_string_key_falls_back_to_fake() -> None:
    assert isinstance(build_adapter("anthropic", ""), FakeAdapter)


def test_build_adapter_unknown_provider_falls_back_to_fake() -> None:
    assert isinstance(build_adapter("cohere", "any-key"), FakeAdapter)


def test_build_adapter_openai_with_key_returns_openai_adapter() -> None:
    assert isinstance(build_adapter("openai", "sk-test"), OpenAIAdapter)


def test_build_adapter_anthropic_with_key_returns_anthropic_adapter() -> None:
    assert isinstance(build_adapter("anthropic", "sk-ant-test"), AnthropicAdapter)


# ---------------------------------------------------------------------------
# OpenAIAdapter: request shaping and error capture (httpx stubbed via monkeypatch)
# ---------------------------------------------------------------------------


class _OpenAIResponse:
    """Minimal httpx.Response stub for the OpenAI chat/completions endpoint."""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:  # type: ignore[type-arg]
        return {"choices": [{"message": {"content": "openai reply"}}]}


def test_openai_adapter_shapes_request_correctly(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict = {}  # type: ignore[type-arg]

    def fake_post(url: str, **kwargs) -> _OpenAIResponse:  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _OpenAIResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    config = ModelConfig("openai", "gpt-4o-mini", temperature=0.3)
    completion = OpenAIAdapter("sk-test").complete(prompt="hi", config=config)

    assert "chat/completions" in captured["url"]
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer sk-test"
    body = captured["kwargs"]["json"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.3
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert completion.text == "openai reply"
    assert completion.error is None


def test_openai_adapter_captures_network_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def boom(url: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise OSError("connection refused")

    monkeypatch.setattr("httpx.post", boom)
    completion = OpenAIAdapter("sk-test").complete(
        prompt="hi", config=ModelConfig("openai", "gpt-4o-mini")
    )
    assert completion.text == ""
    assert completion.error is not None
    assert "connection refused" in completion.error


# ---------------------------------------------------------------------------
# AnthropicAdapter: request shaping and multi-block text assembly
# ---------------------------------------------------------------------------


class _AnthropicResponse:
    """Minimal httpx.Response stub for the Anthropic messages endpoint."""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:  # type: ignore[type-arg]
        return {
            "content": [
                {"type": "text", "text": "part one "},
                {"type": "text", "text": "part two"},
            ]
        }


def test_anthropic_adapter_shapes_request_and_joins_text_blocks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict = {}  # type: ignore[type-arg]

    def fake_post(url: str, **kwargs) -> _AnthropicResponse:  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _AnthropicResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    config = ModelConfig("anthropic", "claude-3-5-haiku-latest", temperature=0.0)
    completion = AnthropicAdapter("sk-ant-test").complete(prompt="hello", config=config)

    assert "/messages" in captured["url"]
    headers = captured["kwargs"]["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    # Multiple text content blocks must be joined
    assert completion.text == "part one part two"
    assert completion.error is None


def test_anthropic_adapter_captures_network_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def boom(url: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise OSError("timeout")

    monkeypatch.setattr("httpx.post", boom)
    completion = AnthropicAdapter("sk-ant-test").complete(
        prompt="hi", config=ModelConfig("anthropic", "claude-3-5-haiku-latest")
    )
    assert completion.text == ""
    assert completion.error is not None
    assert "timeout" in completion.error
