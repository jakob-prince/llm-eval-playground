"""Rubric-based scoring of LLM outputs (LLM-as-judge), with a swappable judge.

This is the framework-agnostic core of llm-eval-playground: given a prompt, a
candidate response, and a weighted rubric, produce per-criterion scores and a
single weighted total. The actual judging is behind the `Judge` protocol so the
real implementation (an LLM call) and a deterministic fake (for tests/CI) are
interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Criterion:
    """One thing we grade a response on. `weight` is relative (need not sum to 1)."""

    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class CriterionScore:
    name: str
    score: float  # normalized to 0.0..1.0
    rationale: str


@dataclass(frozen=True)
class RubricResult:
    scores: list[CriterionScore]
    weighted_total: float  # 0.0..1.0


class Judge(Protocol):
    """Scores a single (prompt, response) pair against a single criterion."""

    def score(self, *, prompt: str, response: str, criterion: Criterion) -> CriterionScore:
        ...


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def weighted_total(scores: list[CriterionScore], criteria: list[Criterion]) -> float:
    """Weight-average per-criterion scores. Non-positive total weight -> 0.0.

    Weights come from `criteria`; scores are matched by name so ordering is
    irrelevant. A criterion with no matching score is treated as absent.
    """
    weights = {c.name: c.weight for c in criteria}
    num = 0.0
    den = 0.0
    for s in scores:
        w = weights.get(s.name, 0.0)
        if w <= 0.0:
            continue
        num += w * _clamp01(s.score)
        den += w
    return _clamp01(num / den) if den > 0.0 else 0.0


def grade(
    *,
    prompt: str,
    response: str,
    criteria: list[Criterion],
    judge: Judge,
) -> RubricResult:
    """Grade `response` against every criterion using `judge`, then aggregate."""
    scores = [
        CriterionScore(
            name=cs.name,
            score=_clamp01(cs.score),
            rationale=cs.rationale,
        )
        for cs in (
            judge.score(prompt=prompt, response=response, criterion=c) for c in criteria
        )
    ]
    return RubricResult(scores=scores, weighted_total=weighted_total(scores, criteria))
