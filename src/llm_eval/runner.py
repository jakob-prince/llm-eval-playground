"""Orchestration: fan one prompt across many models, grade each output.

This is the pure-Python heart that the Streamlit UI renders. It has no UI and
no I/O of its own beyond the adapter/judge it's handed, so it's fully testable
on fakes.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_eval.adapters import Completion, ModelAdapter, ModelConfig
from llm_eval.rubric import Criterion, Judge, RubricResult, grade


@dataclass(frozen=True)
class EvalCell:
    """One model's output plus its rubric result (None if the model errored)."""

    completion: Completion
    result: RubricResult | None


@dataclass(frozen=True)
class EvalGrid:
    """The full side-by-side comparison for one prompt."""

    prompt: str
    criteria: list[Criterion]
    cells: list[EvalCell]

    def ranked(self) -> list[EvalCell]:
        """Cells best-first. Errored cells (no result) sort to the bottom."""
        return sorted(
            self.cells,
            key=lambda c: c.result.weighted_total if c.result is not None else -1.0,
            reverse=True,
        )


def run_eval(
    *,
    prompt: str,
    configs: list[ModelConfig],
    adapters: dict[str, ModelAdapter],
    criteria: list[Criterion],
    judge: Judge,
) -> EvalGrid:
    """Run `prompt` under each config, grade the output, and assemble the grid.

    `adapters` maps provider name -> adapter, so several configs can share one
    provider client. A model that errors yields a cell with `result=None` rather
    than aborting the run.
    """
    cells: list[EvalCell] = []
    for config in configs:
        adapter = adapters[config.provider]
        completion: Completion = adapter.complete(prompt=prompt, config=config)
        result: RubricResult | None = (
            None
            if completion.error
            else grade(
                prompt=prompt,
                response=completion.text,
                criteria=criteria,
                judge=judge,
            )
        )
        cells.append(EvalCell(completion=completion, result=result))
    return EvalGrid(prompt=prompt, criteria=criteria, cells=cells)
