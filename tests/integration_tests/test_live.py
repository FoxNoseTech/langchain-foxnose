"""Live integration tests for FoxNoseRetriever.

These tests require real FoxNose credentials and are gated behind
environment variables. They are skipped in CI unless explicitly enabled.

Required environment variables:
    FOXNOSE_BASE_URL: e.g. "https://<env_key>.fxns.io"
    FOXNOSE_API_PREFIX: e.g. "my_api"
    FOXNOSE_PUBLIC_KEY: Flux API public key
    FOXNOSE_SECRET_KEY: Flux API secret key
    FOXNOSE_FOLDER_PATH: Folder path containing test content
    FOXNOSE_CONTENT_FIELD: Name of the field to use as page_content
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FOXNOSE_BASE_URL"),
    reason="Live tests require FOXNOSE_BASE_URL environment variable",
)


@pytest.fixture()
def live_retriever():
    """Create a retriever connected to a live FoxNose environment."""
    from foxnose_sdk.auth import SimpleKeyAuth
    from foxnose_sdk.flux import FluxClient

    from langchain_foxnose import FoxNoseRetriever

    client = FluxClient(
        base_url=os.environ["FOXNOSE_BASE_URL"],
        api_prefix=os.environ["FOXNOSE_API_PREFIX"],
        auth=SimpleKeyAuth(
            os.environ["FOXNOSE_PUBLIC_KEY"],
            os.environ["FOXNOSE_SECRET_KEY"],
        ),
    )
    return FoxNoseRetriever(
        client=client,
        folder_path=os.environ["FOXNOSE_FOLDER_PATH"],
        page_content_field=os.environ.get("FOXNOSE_CONTENT_FIELD", "body"),
        search_mode="hybrid",
        top_k=3,
    )


def test_live_hybrid_search(live_retriever) -> None:
    """Smoke test: invoke a hybrid search against a live environment."""
    docs = live_retriever.invoke("test query")
    assert isinstance(docs, list)
    for doc in docs:
        assert doc.page_content
        assert "key" in doc.metadata


def test_live_text_search(live_retriever) -> None:
    """Smoke test: text-only search."""
    live_retriever.search_mode = "text"
    docs = live_retriever.invoke("test")
    assert isinstance(docs, list)


def test_live_vector_search(live_retriever) -> None:
    """Smoke test: vector-only search."""
    live_retriever.search_mode = "vector"
    docs = live_retriever.invoke("knowledge management")
    assert isinstance(docs, list)
