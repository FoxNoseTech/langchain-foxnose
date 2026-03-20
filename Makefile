.PHONY: install test tests integration_test integration_tests lint format type docs docs-serve clean

install:
	pip install -e ".[test,lint,docs]"

test tests:
	pytest tests/unit_tests/ --disable-socket --allow-unix-socket --cov=langchain_foxnose --cov-report=term-missing --cov-report=xml

integration_test integration_tests:
	pytest tests/integration_tests/

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

type:
	mypy src/

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ site/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
