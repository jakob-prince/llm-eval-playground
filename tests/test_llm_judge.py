"""Tests for LLMJudge, FakeJudge, and the JSON-parsing helper."""
from __future__ import annotations

from llm_eval.adapters import Completion, ModelConfig
from llm_eval.llm_judge import FakeJudge, LLMJudge, _parse
from llm_eval.rubric import Criterion

# ---------------------------------------------------------------------------
# _parse: JSON extraction and score normalization
# ---------------------------------------------------------------------------


def test_parse_clean_json() -> None:
    score, rationale = _parse('{"score": 8, "rationale": "clear and correct"}')
    assert abs(score - 0.8) < 1e-9
    assert rationale == "clear and correct"


def test_parse_json_surrounded_by_prose() -> None:
    text = 'Here is my assessment:\n{"score": 6, "rationale": "adequate"}\n—end'
    score, rationale = _parse(text)
    assert abs(score - 0.6) < 1e-9
    assert rationale == "adequate"


def test_parse_float_score_normalized() -> None:
    score, _ = _parse('{"score": 7.5, "rationale": "halfway"}')
    assert abs(score - 0.75) < 1e-9


def test_parse_score_above_10_clamped_to_1() -> None:
    score, _ = _parse('{"score": 15, "rationale": "x"}')
    assert score == 1.0


def test_parse_score_negative_clamped_to_0() -> None:
    score, _ = _parse('{"score": -3, "rationale": "x"}')
    assert score == 0.0


def test_parse_missing_score_key_falls_back() -> None:
    score, rationale = _parse('{"rationale": "forgot the score field"}')
    assert score == 0.0
    assert "parse" in rationale


def test_parse_nonnumeric_score_falls_back() -> None:
    score, rationale = _parse('{"score": "excellent", "rationale": "x"}')
    assert score == 0.0
    assert "parse" in rationale


def test_parse_no_json_at_all_falls_back() -> None:
    score, rationale = _parse("The response was decent overall.")
    assert score == 0.0
    assert "parse" in rationale


def test_parse_empty_string_falls_back() -> None:
    score, rationale = _parse("")
    assert score == 0.0
    assert "parse" in rationale


# ---------------------------------------------------------------------------
# LLMJudge: adapter-backed judge
# ---------------------------------------------------------------------------


class _JsonAdapter:
    """Returns a canned string as if a real judge model responded."""

    def __init__(self, text: str, error: str | None = None) -> None:
        self._text = text
        self._error = error

    def complete(self, *, prompt: str, config: ModelConfig) -> Completion:
        if self._error:
            return Completion(config=config, text="", error=self._error)
        return Completion(config=config, text=self._text)


def test_llm_judge_valid_output_parsed_correctly() -> None:
    adapter = _JsonAdapter('{"score": 9, "rationale": "excellent"}')
    judge = LLMJudge(adapter, ModelConfig("fake", "judge"))
    cs = judge.score(
        prompt="explain caching",
        response="A cache stores results ...",
        criterion=Criterion("clarity", "Is the response clear?"),
    )
    assert abs(cs.score - 0.9) < 1e-9
    assert cs.rationale == "excellent"
    assert cs.name == "clarity"


def test_llm_judge_adapter_error_yields_zero_with_error_rationale() -> None:
    adapter = _JsonAdapter("", error="connection refused")
    judge = LLMJudge(adapter, ModelConfig("fake", "judge"))
    cs = judge.score(
        prompt="q",
        response="a",
        criterion=Criterion("accuracy", "Is it accurate?"),
    )
    assert cs.score == 0.0
    assert "judge error" in cs.rationale
    assert "connection refused" in cs.rationale


def test_llm_judge_malformed_output_yields_zero() -> None:
    """A judge response that is not parseable JSON falls back to score=0."""
    adapter = _JsonAdapter("I think it was pretty good, maybe a 7 out of 10.")
    judge = LLMJudge(adapter, ModelConfig("fake", "judge"))
    cs = judge.score(
        prompt="q",
        response="a",
        criterion=Criterion("conciseness", "Is it concise?"),
    )
    assert cs.score == 0.0


# ---------------------------------------------------------------------------
# FakeJudge: deterministic offline judge
# ---------------------------------------------------------------------------


def test_fake_judge_score_always_in_unit_interval() -> None:
    judge = FakeJudge()
    criterion = Criterion("clarity", "")
    for response in ["", "hello world", "x" * 300, "abc123!@#"]:
        cs = judge.score(prompt="p", response=response, criterion=criterion)
        assert 0.0 <= cs.score <= 1.0, f"out of range for response={response!r}"
