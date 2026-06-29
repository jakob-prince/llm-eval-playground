"""Tests for run_eval orchestration. All on fakes — no network, no keys."""
from __future__ import annotations

from llm_eval.adapters import Completion, FakeAdapter, ModelConfig
from llm_eval.llm_judge import FakeJudge
from llm_eval.rubric import Criterion
from llm_eval.runner import run_eval


class BrokenAdapter:
    """Adapter that always reports an error (no result expected)."""

    def complete(self, *, prompt: str, config: ModelConfig) -> Completion:
        return Completion(config=config, text="", error="boom")


def _criteria() -> list[Criterion]:
    return [
        Criterion("clarity", "Is it clear?", 1.0),
        Criterion("accuracy", "Is it accurate?", 1.0),
    ]


def test_run_eval_grades_every_model() -> None:
    configs = [
        ModelConfig("fake", "alpha"),
        ModelConfig("fake", "beta"),
    ]
    grid = run_eval(
        prompt="explain X",
        configs=configs,
        adapters={"fake": FakeAdapter()},
        criteria=_criteria(),
        judge=FakeJudge(),
    )
    assert len(grid.cells) == 2
    assert all(c.result is not None for c in grid.cells)


def test_errored_model_yields_no_result_but_does_not_abort() -> None:
    configs = [ModelConfig("fake", "good"), ModelConfig("broken", "bad")]
    grid = run_eval(
        prompt="q",
        configs=configs,
        adapters={"fake": FakeAdapter(), "broken": BrokenAdapter()},
        criteria=_criteria(),
        judge=FakeJudge(),
    )
    by_model = {c.completion.config.model: c for c in grid.cells}
    assert by_model["good"].result is not None
    assert by_model["bad"].result is None
    assert by_model["bad"].completion.error == "boom"


def test_ranked_puts_errored_cells_last() -> None:
    configs = [ModelConfig("broken", "bad"), ModelConfig("fake", "good")]
    grid = run_eval(
        prompt="q",
        configs=configs,
        adapters={"fake": FakeAdapter(), "broken": BrokenAdapter()},
        criteria=_criteria(),
        judge=FakeJudge(),
    )
    ranked = grid.ranked()
    assert ranked[-1].result is None  # the broken one is last


def test_fake_judge_is_deterministic() -> None:
    j = FakeJudge()
    c = Criterion("clarity", "")
    a = j.score(prompt="p", response="same text", criterion=c)
    b = j.score(prompt="p", response="same text", criterion=c)
    assert a.score == b.score
