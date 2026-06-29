.PHONY: check test lint typecheck demo

check: lint typecheck test

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

demo:
	uv run streamlit run app.py
