"""Shared fixtures for langchain-foxnose tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_search_response(
    results: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a realistic FoxNose _search response."""
    if results is None:
        results = SAMPLE_RESULTS
    resp: dict[str, Any] = {
        "limit": len(results),
        "next": None,
        "previous": None,
        "results": results,
    }
    if metadata is not None:
        resp["metadata"] = metadata
    return resp


SAMPLE_RESULTS = [
    {
        "_sys": {
            "key": "abc123",
            "created_at": "2024-06-01T10:00:00Z",
            "updated_at": "2024-06-15T12:00:00Z",
            "folder": "articles",
        },
        "data": {
            "title": "Getting Started with FoxNose",
            "body": "FoxNose is a serverless knowledge platform...",
            "category": "tech",
            "status": "published",
        },
    },
    {
        "_sys": {
            "key": "def456",
            "created_at": "2024-07-01T08:00:00Z",
            "updated_at": "2024-07-10T09:00:00Z",
            "folder": "articles",
        },
        "data": {
            "title": "Vector Search Guide",
            "body": "Learn how to use vector search in FoxNose...",
            "category": "tutorial",
            "status": "published",
        },
    },
    {
        "_sys": {
            "key": "ghi789",
            "created_at": "2024-08-01T14:00:00Z",
            "updated_at": "2024-08-05T16:00:00Z",
            "folder": "articles",
        },
        "data": {
            "title": "Hybrid Search Best Practices",
            "body": "Combine text and vector search for best results...",
            "category": "guide",
            "status": "draft",
        },
    },
]

SAMPLE_SEARCH_RESPONSE = _make_search_response(
    metadata={
        "search_mode": "hybrid",
        "vector_search_enabled": True,
        "tokens_used": 256,
    }
)


@pytest.fixture()
def sample_results() -> list[dict[str, Any]]:
    """Return a copy of sample FoxNose results."""
    import copy

    return copy.deepcopy(SAMPLE_RESULTS)


@pytest.fixture()
def sample_response() -> dict[str, Any]:
    """Return a copy of a complete FoxNose search response."""
    import copy

    return copy.deepcopy(SAMPLE_SEARCH_RESPONSE)


@pytest.fixture()
def mock_flux_client(sample_response: dict[str, Any]) -> MagicMock:
    """Return a mocked FluxClient with a pre-configured search response."""
    client = MagicMock()
    client.search.return_value = sample_response
    return client


@pytest.fixture()
def mock_async_flux_client(sample_response: dict[str, Any]) -> AsyncMock:
    """Return a mocked AsyncFluxClient with a pre-configured search response."""
    client = AsyncMock()
    client.search.return_value = sample_response
    return client
