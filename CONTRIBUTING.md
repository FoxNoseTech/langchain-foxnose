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
# Unit tests (with coverage)
make test

# Specific test file
pytest tests/unit_tests/test_search_builder.py

# Integration tests (requires FoxNose credentials via env vars)
make integration_test
```

Integration tests run against a real FoxNose environment and skip themselves
unless it is configured. The full variable table, and the shape the collections
have to have, are in the README under "Running integration tests" — point them
at a throwaway environment with synthetic documents, never at production or
customer data.

## Linting & Formatting

```bash
# Check (ruff + mypy)
make lint

# Auto-fix
make format

# Type checking only
make type
```

## Building Documentation

```bash
mkdocs serve   # Local preview; mkdocs prints the address it binds to
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
