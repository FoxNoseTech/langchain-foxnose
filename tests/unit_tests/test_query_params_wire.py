"""Wire-level checks that query-string parameters actually reach the request URL.

The mock-based tests in ``test_query_params.py`` assert on the keyword arguments
the retriever passes to the SDK.  They would keep passing if the SDK renamed or
dropped ``params`` / ``query_params``, because a ``MagicMock`` accepts anything.
These tests drive a real :class:`~foxnose_sdk.flux.FluxClient` over an
``httpx.MockTransport`` and assert on the query string of the outgoing request,
so an SDK signature change fails here.

No sockets are opened: ``MockTransport`` answers in-process, and the base URL is
deliberately unresolvable so that a transport-injection slip cannot reach the
network instead of failing.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import AsyncFluxClient, FluxClient

from langchain_foxnose import FoxNoseRetriever

BASE_URL = "https://invalid.invalid"

# (search_mode, extra retriever kwargs, label)
SEARCH_MODES = [
    ("text", {}, "text"),
    ("vector", {}, "vector"),
    ("hybrid", {}, "hybrid"),
    ("vector_boosted", {}, "vector_boosted"),
    (
        "vector",
        {"vector_field": "embedding", "query_vector": [0.1, 0.2]},
        "vector_field",
    ),
]


def _recorder() -> tuple[list[dict[str, str]], httpx.MockTransport]:
    """Return a (recorded query params, transport) pair."""
    recorded: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(dict(request.url.params))
        return httpx.Response(200, json={"results": [], "limit": 0, "next": None, "previous": None})

    return recorded, httpx.MockTransport(handler)


def _inject(client: Any, attr: str, transport: httpx.MockTransport) -> None:
    """Swap the SDK's internal httpx client for one using *transport*.

    ``FluxClient`` builds its own httpx client and exposes no injection point,
    so this reaches into ``_transport``.  Fail loudly rather than cryptically if
    the SDK ever renames those internals.
    """
    sdk_transport = getattr(client, "_transport", None)
    if sdk_transport is None or not hasattr(sdk_transport, attr):
        pytest.fail(
            f"foxnose-sdk no longer exposes _transport.{attr}; update this "
            f"test's transport injection."
        )
    factory = httpx.AsyncClient if attr == "_async_client" else httpx.Client
    setattr(sdk_transport, attr, factory(base_url=BASE_URL, transport=transport))


@pytest.mark.parametrize(("search_mode", "extra", "label"), SEARCH_MODES)
def test_query_params_reach_the_url(search_mode: str, extra: dict[str, Any], label: str) -> None:
    recorded, transport = _recorder()
    client = FluxClient(base_url=BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk"))
    _inject(client, "_client", transport)

    FoxNoseRetriever(
        client=client,
        collection_path="articles",
        page_content_field="body",
        search_mode=search_mode,
        truncate_text=99,
        query_params={"locale": "en"},
        **extra,
    ).invoke("q")

    assert recorded == [{"locale": "en", "truncate_text": "99"}], label


@pytest.mark.parametrize(("search_mode", "extra", "label"), SEARCH_MODES)
async def test_query_params_reach_the_url_async(
    search_mode: str, extra: dict[str, Any], label: str
) -> None:
    recorded, transport = _recorder()
    client = AsyncFluxClient(base_url=BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk"))
    _inject(client, "_async_client", transport)

    await FoxNoseRetriever(
        async_client=client,
        collection_path="articles",
        page_content_field="body",
        search_mode=search_mode,
        truncate_text=99,
        query_params={"locale": "en"},
        **extra,
    ).ainvoke("q")

    assert recorded == [{"locale": "en", "truncate_text": "99"}], label


def test_no_query_string_when_unset() -> None:
    """Without the parameters, no stray query string is sent."""
    recorded, transport = _recorder()
    client = FluxClient(base_url=BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk"))
    _inject(client, "_client", transport)

    FoxNoseRetriever(
        client=client,
        collection_path="articles",
        page_content_field="body",
        search_mode="text",
    ).invoke("q")

    assert recorded == [{}]
