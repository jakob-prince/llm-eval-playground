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


def test_empty_criteria_yields_zero_total() -> None:
    """grade() with no criteria returns empty scores and zero weighted_total."""
    result = grade(prompt="q", response="r", criteria=[], judge=KeywordJudge())
    assert result.scores == []
    assert result.weighted_total == 0.0


def test_grade_clamps_judge_score_above_1() -> None:
    """grade() clamps a criterion score above 1.0 returned by the judge."""

    class HighJudge:
        def score(self, *, prompt: str, response: str, criterion: Criterion) -> CriterionScore:
            return CriterionScore(name=criterion.name, score=2.5, rationale="")

    result = grade(prompt="q", response="r", criteria=[Criterion("x", "")], judge=HighJudge())
    assert result.scores[0].score == 1.0
    assert result.weighted_total == 1.0


def test_grade_clamps_judge_score_below_0() -> None:
    """grade() clamps a criterion score below 0.0 returned by the judge."""

    class LowJudge:
        def score(self, *, prompt: str, response: str, criterion: Criterion) -> CriterionScore:
            return CriterionScore(name=criterion.name, score=-0.5, rationale="")

    result = grade(prompt="q", response="r", criteria=[Criterion("x", "")], judge=LowJudge())
    assert result.scores[0].score == 0.0
    assert result.weighted_total == 0.0


def test_weighted_total_score_for_unknown_criterion_ignored() -> None:
    """A score whose name is not in criteria is silently skipped."""
    criteria = [Criterion("known", "", weight=1.0)]
    scores = [
        CriterionScore("known", 1.0, ""),
        CriterionScore("not_in_criteria", 0.0, ""),
    ]
    assert weighted_total(scores, criteria) == 1.0


def test_weighted_total_negative_weight_skipped() -> None:
    """Criteria with weight <= 0 are excluded from the weighted average."""
    criteria = [Criterion("neg", "", weight=-1.0), Criterion("pos", "", weight=2.0)]
    scores = [CriterionScore("neg", 0.0, ""), CriterionScore("pos", 1.0, "")]
    # Only the positive-weight criterion contributes -> total == 1.0
    assert weighted_total(scores, criteria) == 1.0
