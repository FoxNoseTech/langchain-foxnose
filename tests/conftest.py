"""Shared fixtures for langchain-foxnose tests."""

from __future__ import annotations

import itertools
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
    """Return a mocked FluxClient with pre-configured responses for all search methods."""
    client = MagicMock()
    client.search.return_value = sample_response
    client.vector_search.return_value = sample_response
    client.vector_field_search.return_value = sample_response
    client.hybrid_search.return_value = sample_response
    client.boosted_search.return_value = sample_response
    return client


@pytest.fixture()
def mock_async_flux_client(sample_response: dict[str, Any]) -> AsyncMock:
    """Return a mocked AsyncFluxClient with pre-configured responses for all search methods."""
    client = AsyncMock()
    client.search.return_value = sample_response
    client.vector_search.return_value = sample_response
    client.vector_field_search.return_value = sample_response
    client.hybrid_search.return_value = sample_response
    client.boosted_search.return_value = sample_response
    return client


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _make_list_response(
    results: list[dict[str, Any]] | None = None,
    *,
    count: int | None = None,
    next_cursor: str | None = None,
    previous_cursor: str | None = None,
) -> dict[str, Any]:
    """Build a realistic FoxNose ``list_resources`` response."""
    if results is None:
        results = SAMPLE_RESULTS
    return {
        "count": count if count is not None else len(results),
        "next": next_cursor,
        "previous": previous_cursor,
        "results": results,
    }


@pytest.fixture()
def mock_flux_client_with_list(sample_response: dict[str, Any]) -> MagicMock:
    """Return a mocked FluxClient with pre-configured search and list_resources."""
    client = MagicMock()
    client.search.return_value = sample_response
    client.vector_search.return_value = sample_response
    client.vector_field_search.return_value = sample_response
    client.hybrid_search.return_value = sample_response
    client.boosted_search.return_value = sample_response
    client.list_resources.return_value = _make_list_response()
    return client


@pytest.fixture()
def mock_async_flux_client_with_list(sample_response: dict[str, Any]) -> AsyncMock:
    """Return a mocked AsyncFluxClient with pre-configured search and list_resources."""
    client = AsyncMock()
    client.search.return_value = sample_response
    client.vector_search.return_value = sample_response
    client.vector_field_search.return_value = sample_response
    client.hybrid_search.return_value = sample_response
    client.boosted_search.return_value = sample_response
    client.list_resources.return_value = _make_list_response()
    return client


# ---------------------------------------------------------------------------
# Writer helpers
# ---------------------------------------------------------------------------


def _make_write_response(
    resource_key: str = "res_1",
    revision_key: str = "rev_1",
) -> dict[str, Any]:
    """Build a realistic FoxNose Flux write response."""
    return {
        "resource_key": resource_key,
        "revision_key": revision_key,
        "write_units": 1,
        "published": True,
    }


@pytest.fixture()
def mock_write_client() -> MagicMock:
    """Return a mocked FluxClient whose writes return distinct resource keys."""
    client = MagicMock()
    counter = itertools.count(1)
    client.create_resource.side_effect = lambda *a, **kw: _make_write_response(
        resource_key=f"res_{next(counter)}", revision_key="rev_new"
    )
    client.update_resource.return_value = _make_write_response(
        resource_key="res_1", revision_key="rev_2"
    )
    return client


@pytest.fixture()
def mock_async_write_client() -> AsyncMock:
    """Return a mocked AsyncFluxClient whose writes return distinct resource keys."""
    client = AsyncMock()
    counter = itertools.count(1)

    async def _create(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _make_write_response(resource_key=f"res_{next(counter)}", revision_key="rev_new")

    client.create_resource.side_effect = _create
    client.update_resource.return_value = _make_write_response(
        resource_key="res_1", revision_key="rev_2"
    )
    return client


# ---------------------------------------------------------------------------
# Wire-level helpers
# ---------------------------------------------------------------------------
#
# Mock-based tests assert on the keyword arguments we hand the SDK, so they
# would keep passing if the SDK renamed a parameter or changed its request
# shape -- a MagicMock accepts anything. The wire tests drive a real
# FluxClient over an httpx.MockTransport and assert on the outgoing HTTP
# request instead. No sockets are opened, and WIRE_BASE_URL is deliberately
# unresolvable so a transport-injection slip fails instead of reaching the
# network.

WIRE_BASE_URL = "https://invalid.invalid"


def make_wire_recorder(
    response: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Return a ``(recorded requests, MockTransport)`` pair.

    Each recorded entry holds the request's ``method``, ``path``, query
    ``params`` and decoded JSON ``body``.
    """
    import json

    import httpx

    recorded: list[dict[str, Any]] = []
    payload = (
        response
        if response is not None
        else {"results": [], "limit": 0, "next": None, "previous": None}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(
            {
                "method": request.method,
                "path": request.url.path,
                "params": dict(request.url.params),
                "body": json.loads(request.content) if request.content else None,
            }
        )
        return httpx.Response(200, json=payload)

    return recorded, httpx.MockTransport(handler)


def inject_wire_transport(client: Any, transport: Any, *, is_async: bool = False) -> None:
    """Swap an SDK client's internal httpx client for one using *transport*.

    ``FluxClient`` builds its own httpx client and exposes no injection point,
    so this reaches into ``_transport``. Fail loudly rather than cryptically if
    the SDK ever renames those internals.
    """
    import httpx

    attr = "_async_client" if is_async else "_client"
    sdk_transport = getattr(client, "_transport", None)
    if sdk_transport is None or not hasattr(sdk_transport, attr):
        pytest.fail(
            f"foxnose-sdk no longer exposes _transport.{attr}; update the wire "
            f"tests' transport injection."
        )
    factory = httpx.AsyncClient if is_async else httpx.Client
    setattr(sdk_transport, attr, factory(base_url=WIRE_BASE_URL, transport=transport))
