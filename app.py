"""Streamlit UI for llm-eval-playground.

Thin shell over the pure-Python core: collect a prompt + model picks + rubric in
the sidebar, run the eval, and render a side-by-side grid of outputs and scores.
Boots on fake models so it runs with no API key; paste your own key (session-only)
to use real providers. NOTE: not runnable in the PromptQL sandbox — run locally
with `uv run streamlit run app.py`.
"""
from __future__ import annotations

import streamlit as st

from llm_eval.adapters import ModelConfig, build_adapter
from llm_eval.llm_judge import FakeJudge, LLMJudge
from llm_eval.rubric import Criterion
from llm_eval.runner import EvalGrid, run_eval

DEFAULT_CRITERIA = [
    Criterion("clarity", "Is the answer clear and well-structured?", 1.0),
    Criterion("accuracy", "Is the answer factually correct?", 2.0),
    Criterion("conciseness", "Is it appropriately concise?", 1.0),
]

CATALOG = {
    "fake": ["alpha", "beta"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
}


def _sidebar() -> tuple[list[ModelConfig], dict[str, str | None]]:
    st.sidebar.header("Models")
    keys: dict[str, str | None] = {
        "openai": st.sidebar.text_input("OpenAI API key", type="password") or None,
        "anthropic": st.sidebar.text_input("Anthropic API key", type="password") or None,
        "fake": None,
    }
    st.sidebar.caption("Keys are session-only — never stored or logged.")
    configs: list[ModelConfig] = []
    for provider, models in CATALOG.items():
        picked = st.sidebar.multiselect(f"{provider} models", models, key=f"sel_{provider}")
        for m in picked:
            configs.append(ModelConfig(provider=provider, model=m))
    if not configs:  # always have something to run
        configs = [ModelConfig("fake", "alpha"), ModelConfig("fake", "beta")]
        st.sidebar.caption("No models selected — running fake:alpha and fake:beta by default.")
    return configs, keys


def _render(grid: EvalGrid) -> None:
    ranked = grid.ranked()
    cols = st.columns(len(ranked))
    for col, cell in zip(cols, ranked, strict=True):
        with col:
            st.subheader(cell.completion.config.display())
            if cell.result is None:
                st.error(cell.completion.error or "errored")
                continue
            st.metric("Weighted score", f"{cell.result.weighted_total:.2f}")
            st.write(cell.completion.text)
            for s in cell.result.scores:
                st.progress(s.score, text=f"{s.name}: {s.score:.2f}")


def main() -> None:
    st.set_page_config(page_title="LLM Eval Playground", layout="wide")
    st.title("LLM Eval Playground")
    st.write("Run one prompt across multiple models and score each against a rubric.")

    configs, keys = _sidebar()
    prompt = st.text_area("Prompt", "Explain what a token bucket rate limiter is.")
    use_real_judge = st.checkbox("Use LLM-as-judge (needs OpenAI key)", value=False)

    if st.button("Run eval", type="primary"):
        adapters = {p: build_adapter(p, keys.get(p)) for p in CATALOG}
        if use_real_judge and keys.get("openai"):
            judge = LLMJudge(adapters["openai"], ModelConfig("openai", "gpt-4o-mini"))
        else:
            judge = FakeJudge()
        grid = run_eval(
            prompt=prompt,
            configs=configs,
            adapters=adapters,
            criteria=DEFAULT_CRITERIA,
            judge=judge,
        )
        _render(grid)


if __name__ == "__main__":
    main()
