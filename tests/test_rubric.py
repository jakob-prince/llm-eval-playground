"""Tests for the rubric scorer. Uses a deterministic fake judge (no LLM, CI-safe)."""
from __future__ import annotations

from llm_eval.rubric import Criterion, CriterionScore, Judge, grade, weighted_total


class KeywordJudge:
    """Deterministic fake: score = fraction of the criterion's name-words present."""

    def score(self, *, prompt: str, response: str, criterion: Criterion) -> CriterionScore:
        words = criterion.name.lower().split()
        hits = sum(1 for w in words if w in response.lower())
        frac = hits / len(words) if words else 0.0
        return CriterionScore(name=criterion.name, score=frac, rationale=f"{hits}/{len(words)} hit")


def _criteria() -> list[Criterion]:
    return [
        Criterion("clear answer", "Is the answer clear?", weight=2.0),
        Criterion("correct facts", "Are facts correct?", weight=1.0),
    ]


def test_weighted_total_is_weight_average() -> None:
    criteria = _criteria()
    scores = [CriterionScore("clear answer", 1.0, ""), CriterionScore("correct facts", 0.0, "")]
    # (2*1.0 + 1*0.0) / (2+1) == 0.666...
    assert abs(weighted_total(scores, criteria) - (2 / 3)) < 1e-9


def test_weights_match_by_name_not_order() -> None:
    criteria = _criteria()
    in_order = [CriterionScore("clear answer", 1.0, ""), CriterionScore("correct facts", 0.0, "")]
    reversed_ = list(reversed(in_order))
    assert weighted_total(in_order, criteria) == weighted_total(reversed_, criteria)


def test_out_of_range_scores_are_clamped() -> None:
    criteria = [Criterion("a", "", weight=1.0)]
    assert weighted_total([CriterionScore("a", 5.0, "")], criteria) == 1.0
    assert weighted_total([CriterionScore("a", -3.0, "")], criteria) == 0.0


def test_zero_total_weight_is_zero_not_error() -> None:
    criteria = [Criterion("a", "", weight=0.0)]
    assert weighted_total([CriterionScore("a", 1.0, "")], criteria) == 0.0


def test_grade_end_to_end_with_fake_judge() -> None:
    criteria = _criteria()
    judge: Judge = KeywordJudge()
    res = grade(
        prompt="q",
        response="this is a clear answer with correct facts",
        criteria=criteria,
        judge=judge,
    )
    assert {s.name for s in res.scores} == {"clear answer", "correct facts"}
    # both criteria's words appear -> perfect score
    assert res.weighted_total == 1.0
