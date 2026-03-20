"""Skip all integration tests when required environment variables are missing."""

from __future__ import annotations

import os

import pytest

REQUIRED_ENV_VARS = [
    "FOXNOSE_BASE_URL",
    "FOXNOSE_API_PREFIX",
    "FOXNOSE_PUBLIC_KEY",
    "FOXNOSE_SECRET_KEY",
    "FOXNOSE_FOLDER_PATH",
]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        skip = pytest.mark.skip(reason=f"Missing env vars: {', '.join(missing)}")
        for item in items:
            item.add_marker(skip)
