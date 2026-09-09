"""Live tests for FoxNoseRetriever against a configured FoxNose environment."""

from __future__ import annotations

import os
from typing import Any

from foxnose_sdk.errors import FoxnoseAPIError

from langchain_foxnose import FoxNoseRetriever
from tests.integration_tests.conftest import (
    skip_if_vector_unavailable,
    skip_unconfigured,
)


def _retriever(client: Any, path: str, field: str, **kwargs: Any) -> FoxNoseRetriever:
    kwargs.setdefault("page_content_field", field)
    return FoxNoseRetriever(client=client, collection_path=path, **kwargs)


class TestSearchModes:
    def test_text_search_returns_documents(self, live_retriever: Any, query: str) -> None:
        docs = live_retriever.invoke(query)
        assert docs
        assert all(doc.page_content for doc in docs)
        assert all("key" in doc.metadata for doc in docs)

    def test_hybrid_search(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        retriever = _retriever(
            flux_client, collection_path, content_field, search_mode="hybrid", top_k=3
        )
        try:
            docs = retriever.invoke(query)
        except FoxnoseAPIError as exc:
            skip_if_vector_unavailable(exc)
        assert isinstance(docs, list)

    def test_vector_search(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        retriever = _retriever(
            flux_client, collection_path, content_field, search_mode="vector", top_k=3
        )
        try:
            docs = retriever.invoke(query)
        except FoxnoseAPIError as exc:
            skip_if_vector_unavailable(exc)
        assert isinstance(docs, list)

    def test_vector_boosted_search(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        retriever = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="vector_boosted",
            top_k=3,
        )
        try:
            docs = retriever.invoke(query)
        except FoxnoseAPIError as exc:
            skip_if_vector_unavailable(exc)
        assert isinstance(docs, list)


class TestFilter:
    def test_where_filter_excludes_non_matching(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        filter_predicate: tuple[str, str],
        read_corpus_is_sufficient: None,
    ) -> None:
        """Three-sided, because a malformed `where` is SILENTLY ignored.

        The backend only walks all_of / any_of, so a wrong filter shape is
        dropped rather than rejected -- without the control query, a broken
        filter would look like a pass.
        """
        field, value = filter_predicate
        unfiltered = _retriever(
            flux_client, collection_path, content_field, search_mode="text", top_k=10
        ).invoke(query)
        if not any(doc.metadata.get(field) != value for doc in unfiltered):
            skip_unconfigured(
                f"every document matching FOXNOSE_QUERY already has "
                f"{field}={value!r}, so the filter assertion would be vacuous"
            )

        filtered = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="text",
            top_k=10,
            where={"$": {"all_of": [{f"{field}__eq": value}]}},
        ).invoke(query)
        assert filtered, "the filter matched nothing at all"
        assert all(doc.metadata.get(field) == value for doc in filtered)


class TestResultCount:
    def test_top_k_constructor(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = _retriever(
            flux_client, collection_path, content_field, search_mode="text", top_k=2
        ).invoke(query)
        assert len(docs) <= 2

    def test_top_k_runtime_override(self, live_retriever: Any, query: str) -> None:
        assert len(live_retriever.invoke(query, top_k=1)) <= 1

    def test_k_runtime_alias(self, live_retriever: Any, query: str) -> None:
        assert len(live_retriever.invoke(query, k=1)) <= 1

    def test_search_kwargs_limit_and_offset(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="text",
            top_k=10,
            search_kwargs={"limit": 2, "offset": 1},
        ).invoke(query)
        assert len(docs) <= 2


class TestQueryStringParameters:
    def test_truncate_text(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        """Two-sided: the cap holds AND the uncapped query really is longer."""
        untruncated = _retriever(
            flux_client, collection_path, content_field, search_mode="text", top_k=10
        ).invoke(query)
        if not any(len(doc.page_content) > 40 for doc in untruncated):
            skip_unconfigured(
                "no document matching FOXNOSE_QUERY is longer than 40 characters, "
                "so the truncation assertion would be vacuous"
            )

        truncated = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="text",
            top_k=10,
            truncate_text=40,
        ).invoke(query)
        assert truncated
        assert all(len(doc.page_content) <= 40 for doc in truncated)

    def test_query_params_passthrough(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="text",
            top_k=10,
            query_params={"truncate_text": 40},
        ).invoke(query)
        assert docs
        assert all(len(doc.page_content) <= 40 for doc in docs)


class TestMapping:
    def test_metadata_whitelist(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        metadata_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="text",
            metadata_fields=[metadata_field],
        ).invoke(query)
        assert docs
        sys_keys = {"key", "folder", "created_at", "updated_at"}
        for doc in docs:
            assert set(doc.metadata) <= sys_keys | {metadata_field}

    def test_include_sys_metadata_false(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = _retriever(
            flux_client,
            collection_path,
            content_field,
            search_mode="text",
            include_sys_metadata=False,
        ).invoke(query)
        assert docs
        assert all("key" not in doc.metadata for doc in docs)

    def test_page_content_fields_are_concatenated(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        metadata_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = FoxNoseRetriever(
            client=flux_client,
            collection_path=collection_path,
            page_content_fields=[metadata_field, content_field],
            search_mode="text",
        ).invoke(query)
        assert docs
        assert all("\n\n" in doc.page_content for doc in docs)

    def test_page_content_mapper(
        self,
        flux_client: Any,
        collection_path: str,
        metadata_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        docs = FoxNoseRetriever(
            client=flux_client,
            collection_path=collection_path,
            page_content_mapper=lambda result: str(result["data"][metadata_field]).upper(),
            search_mode="text",
        ).invoke(query)
        assert docs
        assert all(doc.page_content.isupper() for doc in docs)


class TestAsync:
    async def test_native_async(
        self,
        async_flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        retriever = FoxNoseRetriever(
            async_client=async_flux_client,
            collection_path=collection_path,
            page_content_field=content_field,
            search_mode="text",
            top_k=3,
        )
        docs = await retriever.ainvoke(query)
        assert docs

    async def test_async_falls_back_to_the_sync_client_and_keeps_kwargs(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        query: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        """With no async client, ainvoke runs the sync path in an executor.

        The runtime top_k must survive that hop.
        """
        retriever = _retriever(
            flux_client, collection_path, content_field, search_mode="text", top_k=5
        )
        assert retriever.async_client is None
        assert len(await retriever.ainvoke(query, top_k=1)) <= 1


def test_from_client_params(
    collection_path: str, content_field: str, query: str, read_corpus_is_sufficient: None
) -> None:
    from foxnose_sdk.auth import SimpleKeyAuth

    retriever = FoxNoseRetriever.from_client_params(
        base_url=os.environ["FOXNOSE_BASE_URL"],
        api_prefix=os.environ["FOXNOSE_API_PREFIX"],
        auth=SimpleKeyAuth(os.environ["FOXNOSE_PUBLIC_KEY"], os.environ["FOXNOSE_SECRET_KEY"]),
        collection_path=collection_path,
        page_content_field=content_field,
        search_mode="text",
    )
    assert retriever.invoke(query)
