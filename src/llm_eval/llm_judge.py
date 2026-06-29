"""Judges: an LLM-as-judge for production and a deterministic fake for offline/CI.

Both satisfy the `Judge` protocol from rubric.py, so they're interchangeable in
`run_eval`.
"""
from __future__ import annotations

import json

from llm_eval.adapters import ModelAdapter, ModelConfig
from llm_eval.rubric import Criterion, CriterionScore

_JUDGE_TEMPLATE = """You are grading an AI response against a single criterion.

Criterion: {name}
What it means: {description}

The user's prompt was:
{prompt}

The response to grade:
{response}

Return ONLY compact JSON: {{"score": <integer 0-10>, "rationale": "<one sentence>"}}.
10 = perfectly satisfies the criterion, 0 = not at all."""


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _parse(text: str) -> tuple[float, str]:
    """Parse the judge's JSON. On any malformed output, score 0 with a note."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        raw = float(data["score"])
        return _clamp01(raw / 10.0), str(data.get("rationale", ""))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 0.0, "could not parse judge output"


class LLMJudge:
    """Asks a model to score (prompt, response) against one criterion, 0..1."""

    def __init__(self, adapter: ModelAdapter, config: ModelConfig) -> None:
        self._adapter = adapter
        self._config = config

    def score(self, *, prompt: str, response: str, criterion: Criterion) -> CriterionScore:
        judge_prompt = _JUDGE_TEMPLATE.format(
            name=criterion.name,
            description=criterion.description,
            prompt=prompt,
            response=response,
        )
        completion = self._adapter.complete(prompt=judge_prompt, config=self._config)
        if completion.error:
            return CriterionScore(
                name=criterion.name, score=0.0, rationale=f"judge error: {completion.error}"
            )
        score, rationale = _parse(completion.text)
        return CriterionScore(name=criterion.name, score=score, rationale=rationale)


class FakeJudge:
    """Deterministic offline judge: stable pseudo-score from the response text.

    Not meaningful grading — just makes the offline app produce varied, stable
    numbers so the grid is demonstrable with no key.
    """

    def score(self, *, prompt: str, response: str, criterion: Criterion) -> CriterionScore:
        h = abs(hash((criterion.name, response))) % 11  # 0..10, stable per run
        return CriterionScore(
            name=criterion.name, score=h / 10.0, rationale="deterministic fake score"
        )
