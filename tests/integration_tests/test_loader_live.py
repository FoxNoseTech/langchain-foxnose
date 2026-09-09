"""Live tests for FoxNoseLoader against a configured FoxNose environment."""

from __future__ import annotations

from typing import Any

from langchain_foxnose import FoxNoseLoader
from tests.integration_tests.conftest import skip_unconfigured


def _loader(client: Any, path: str, field: str, **kwargs: Any) -> FoxNoseLoader:
    kwargs.setdefault("page_content_field", field)
    return FoxNoseLoader(client=client, collection_path=path, **kwargs)


def test_load_returns_documents(flux_client: Any, collection_path: str, content_field: str) -> None:
    docs = _loader(flux_client, collection_path, content_field).load()
    assert docs
    assert all(doc.page_content for doc in docs)
    assert all("key" in doc.metadata for doc in docs)


def test_lazy_load_paginates(flux_client: Any, collection_path: str, content_field: str) -> None:
    """A batch_size below the corpus size forces the cursor loop to run.

    This is the regression test for the infinite-pagination bug: FoxNose
    returns `next` as a full URL rather than the token the parameter takes, so
    feeding it straight back re-fetched page one forever.
    """
    eager = _loader(flux_client, collection_path, content_field).load()
    if len(eager) <= 2:
        skip_unconfigured(
            f"the read collection has only {len(eager)} documents, too few to exercise pagination"
        )
    lazy = list(_loader(flux_client, collection_path, content_field, batch_size=2).lazy_load())
    assert len(lazy) == len(eager)
    assert {doc.metadata["key"] for doc in lazy} == {doc.metadata["key"] for doc in eager}


async def test_alazy_load(async_flux_client: Any, collection_path: str, content_field: str) -> None:
    loader = FoxNoseLoader(
        async_client=async_flux_client,
        collection_path=collection_path,
        page_content_field=content_field,
        batch_size=2,
    )
    docs = [doc async for doc in loader.alazy_load()]
    assert docs
    assert all(doc.page_content for doc in docs)


def test_truncate_text(flux_client: Any, collection_path: str, content_field: str) -> None:
    """Two-sided: the cap holds AND the uncapped load really is longer."""
    untruncated = _loader(flux_client, collection_path, content_field).load()
    if not any(len(doc.page_content) > 40 for doc in untruncated):
        skip_unconfigured(
            "no document is longer than 40 characters, so the truncation assertion would be vacuous"
        )
    truncated = _loader(flux_client, collection_path, content_field, truncate_text=40).load()
    assert truncated
    assert all(len(doc.page_content) <= 40 for doc in truncated)


def test_truncate_text_via_raw_params(
    flux_client: Any, collection_path: str, content_field: str
) -> None:
    """`params` is a documented raw query-string passthrough; keep it working.

    This is the loader's contract and it differs from the retriever's, whose
    search_kwargs is a request-BODY passthrough and rejects the key.
    """
    docs = _loader(flux_client, collection_path, content_field, params={"truncate_text": 40}).load()
    assert docs
    assert all(len(doc.page_content) <= 40 for doc in docs)


def test_include_sys_metadata_false(
    flux_client: Any, collection_path: str, content_field: str
) -> None:
    docs = _loader(flux_client, collection_path, content_field, include_sys_metadata=False).load()
    assert docs
    assert all("key" not in doc.metadata for doc in docs)
