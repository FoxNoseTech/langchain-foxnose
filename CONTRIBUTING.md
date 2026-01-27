# Contributing to langchain-foxnose

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/FoxNoseTech/langchain-foxnose.git
cd langchain-foxnose
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,lint,docs]"
```

## Running Tests

```bash
# All tests
pytest tests/ --cov=langchain_foxnose

# Unit tests only
pytest tests/ --ignore=tests/integration

# Specific test file
pytest tests/test_search_builder.py
```

## Linting & Formatting

```bash
# Check
ruff check src/ tests/
ruff format --check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
ruff format src/ tests/
```

## Building Documentation

```bash
mkdocs serve   # Local preview at http://127.0.0.1:8000
mkdocs build   # Build static site
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Submit a pull request with a clear description of your changes.

## Code Style

- Follow existing patterns in the codebase.
- Use type annotations for all public APIs.
- Write Google-style docstrings for public classes and functions.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
