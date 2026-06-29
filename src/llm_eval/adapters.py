"""Model adapters: one uniform interface over different LLM providers.

Every provider (OpenAI, Anthropic, ...) is wrapped as a `ModelAdapter`. A
`FakeAdapter` returns deterministic canned text so the app and the whole test
suite run with **no API key and no network spend**. Real adapters read a key
supplied at construction time (bring-your-own-key) — nothing is baked in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelConfig:
    """A model + sampling settings to run a prompt under."""

    provider: str  # "openai" | "anthropic" | "fake"
    model: str  # e.g. "gpt-4o-mini", "claude-3-5-sonnet-latest"
    temperature: float = 0.0
    label: str | None = None  # optional friendly column name

    def display(self) -> str:
        return self.label or f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class Completion:
    """The result of running one prompt under one `ModelConfig`."""

    config: ModelConfig
    text: str
    error: str | None = None  # set (and text empty) if the call failed


class ModelAdapter(Protocol):
    """Runs a single prompt under a single config, never raising network errors.

    Implementations must capture transport/API failures into `Completion.error`
    rather than raising, so one bad model doesn't sink a whole grid run.
    """

    def complete(self, *, prompt: str, config: ModelConfig) -> Completion:
        ...


class FakeAdapter:
    """Deterministic, offline adapter — the default so the app runs with no key."""

    def complete(self, *, prompt: str, config: ModelConfig) -> Completion:
        snippet = " ".join(prompt.split())[:160]
        text = (
            f"[{config.display()}] Here is a clear, correct answer. "
            f"You asked: {snippet}"
        )
        return Completion(config=config, text=text)


class OpenAIAdapter:
    """OpenAI Chat Completions adapter. Key is supplied by the caller (BYOK)."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def complete(self, *, prompt: str, config: ModelConfig) -> Completion:
        import httpx

        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": config.model,
                    "temperature": config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return Completion(config=config, text=text)
        except Exception as exc:  # noqa: BLE001 — failures become data, not exceptions
            return Completion(config=config, text="", error=str(exc))


class AnthropicAdapter:
    """Anthropic Messages adapter. Key is supplied by the caller (BYOK)."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def complete(self, *, prompt: str, config: ModelConfig) -> Completion:
        import httpx

        try:
            resp = httpx.post(
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": config.model,
                    "max_tokens": 1024,
                    "temperature": config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            return Completion(config=config, text=text)
        except Exception as exc:  # noqa: BLE001
            return Completion(config=config, text="", error=str(exc))


def build_adapter(provider: str, api_key: str | None) -> ModelAdapter:
    """Pick an adapter by provider name. Missing key or 'fake' -> FakeAdapter."""
    if provider == "openai" and api_key:
        return OpenAIAdapter(api_key)
    if provider == "anthropic" and api_key:
        return AnthropicAdapter(api_key)
    return FakeAdapter()
