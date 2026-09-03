"""Wire-level checks that query-string parameters actually reach the request URL.

The mock-based tests in ``test_query_params.py`` assert on the keyword arguments
the retriever passes to the SDK.  They would keep passing if the SDK renamed or
dropped ``params`` / ``query_params``, because a ``MagicMock`` accepts anything.
These tests drive a real :class:`~foxnose_sdk.flux.FluxClient` over an
``httpx.MockTransport`` and assert on the query string of the outgoing request,
so an SDK signature change fails here.
"""

from __future__ import annotations

from typing import Any

import pytest
from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import AsyncFluxClient, FluxClient

from langchain_foxnose import FoxNoseRetriever
from tests.conftest import WIRE_BASE_URL, inject_wire_transport, make_wire_recorder

# (search_mode, extra retriever kwargs, label)
SEARCH_MODES = [
    ("text", {}, "text"),
    ("vector", {}, "vector"),
    ("hybrid", {}, "hybrid"),
    ("vector_boosted", {}, "vector_boosted"),
    ("vector", {"vector_field": "embedding", "query_vector": [0.1, 0.2]}, "vector_field"),
]


@pytest.mark.parametrize(("search_mode", "extra", "label"), SEARCH_MODES)
def test_query_params_reach_the_url(search_mode: str, extra: dict[str, Any], label: str) -> None:
    recorded, transport = make_wire_recorder()
    client = FluxClient(base_url=WIRE_BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk"))
    inject_wire_transport(client, transport)

    FoxNoseRetriever(
        client=client,
        collection_path="articles",
        page_content_field="body",
        search_mode=search_mode,
        truncate_text=99,
        query_params={"locale": "en"},
        **extra,
    ).invoke("q")

    assert len(recorded) == 1, label
    assert recorded[0]["params"] == {"locale": "en", "truncate_text": "99"}, label
    assert recorded[0]["path"] == "/api/articles/_search", label


@pytest.mark.parametrize(("search_mode", "extra", "label"), SEARCH_MODES)
async def test_query_params_reach_the_url_async(
    search_mode: str, extra: dict[str, Any], label: str
) -> None:
    recorded, transport = make_wire_recorder()
    client = AsyncFluxClient(
        base_url=WIRE_BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk")
    )
    inject_wire_transport(client, transport, is_async=True)

    await FoxNoseRetriever(
        async_client=client,
        collection_path="articles",
        page_content_field="body",
        search_mode=search_mode,
        truncate_text=99,
        query_params={"locale": "en"},
        **extra,
    ).ainvoke("q")

    assert len(recorded) == 1, label
    assert recorded[0]["params"] == {"locale": "en", "truncate_text": "99"}, label


def test_no_query_string_when_unset() -> None:
    """Without the parameters, no stray query string is sent."""
    recorded, transport = make_wire_recorder()
    client = FluxClient(base_url=WIRE_BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk"))
    inject_wire_transport(client, transport)

    FoxNoseRetriever(
        client=client,
        collection_path="articles",
        page_content_field="body",
        search_mode="text",
    ).invoke("q")

    assert recorded[0]["params"] == {}
