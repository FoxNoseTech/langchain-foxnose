.PHONY: install test lint format docs clean

install:
	pip install -e ".[test,lint,docs]"

test:
	pytest tests/ --cov=langchain_foxnose --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ site/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
