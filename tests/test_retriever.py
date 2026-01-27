"""Tests for FoxNoseRetriever (synchronous)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_foxnose import FoxNoseRetriever


class TestRetrieverValidation:
    """Pydantic model validation."""

    def test_requires_at_least_one_client(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)at least one of"):
            FoxNoseRetriever(
                folder_path="kb",
                page_content_field="body",
            )

    def test_requires_content_mapping(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)content mapping strategy"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
            )

    def test_rejects_multiple_content_strategies(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)only one"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                page_content_fields=["title", "body"],
            )

    def test_rejects_invalid_search_mode(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)invalid search_mode"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="invalid",
            )

    def test_rejects_top_k_zero(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)top_k must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                top_k=0,
            )

    def test_rejects_negative_top_k(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)top_k must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                top_k=-5,
            )

    def test_rejects_both_metadata_field_options(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)mutually exclusive"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                metadata_fields=["title"],
                exclude_metadata_fields=["status"],
            )

    def test_rejects_empty_page_content_fields(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)must not be empty"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_fields=[],
            )

    def test_rejects_text_threshold_out_of_range(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)text_threshold must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                text_threshold=1.5,
            )

    def test_rejects_negative_text_threshold(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)text_threshold must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                text_threshold=-0.1,
            )

    def test_rejects_similarity_threshold_out_of_range(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)similarity_threshold must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                similarity_threshold=2.0,
            )

    def test_rejects_search_mode_in_search_kwargs(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)do not override.*search_mode"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_kwargs={"search_mode": "text"},
            )

    def test_accepts_valid_config(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
            search_mode="hybrid",
        )
        assert retriever.folder_path == "kb"
        assert retriever.search_mode == "hybrid"

    def test_accepts_async_client_only(self) -> None:
        from unittest.mock import AsyncMock

        retriever = FoxNoseRetriever(
            async_client=AsyncMock(),
            folder_path="kb",
            page_content_field="body",
        )
        assert retriever.client is None
        assert retriever.async_client is not None

    def test_accepts_page_content_mapper(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="kb",
            page_content_mapper=lambda r: r["data"]["body"],
        )
        assert retriever.page_content_mapper is not None


class TestRetrieverSync:
    """Synchronous retrieval."""

    def test_invoke_returns_documents(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
        )
        docs = retriever.invoke("test query")
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."
        assert docs[0].metadata["key"] == "abc123"

    def test_invoke_calls_client_search(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
        )
        retriever.invoke("test query")
        mock_flux_client.search.assert_called_once()
        call_args = mock_flux_client.search.call_args
        assert call_args[0][0] == "articles"
        body = call_args[1]["body"]
        assert body["search_mode"] == "hybrid"
        assert body["limit"] == 5
        assert body["find_text"]["query"] == "test query"
        assert body["vector_search"]["query"] == "test query"

    def test_text_mode_search_body(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            top_k=10,
        )
        retriever.invoke("keyword search")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["search_mode"] == "text"
        assert "find_text" in body
        assert "vector_search" not in body

    def test_vector_mode_search_body(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            top_k=20,
            similarity_threshold=0.8,
        )
        retriever.invoke("semantic search")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["search_mode"] == "vector"
        assert "vector_search" in body
        assert "find_text" not in body
        assert body["vector_search"]["similarity_threshold"] == 0.8

    def test_where_filter_passed(self, mock_flux_client: MagicMock) -> None:
        where_filter = {"$": {"all_of": [{"status__eq": "published"}]}}
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            where=where_filter,
        )
        retriever.invoke("query")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["where"] == where_filter

    def test_search_kwargs_override(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_kwargs={"ignore_unknown_fields": True, "limit": 50},
        )
        retriever.invoke("query")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["ignore_unknown_fields"] is True
        assert body["limit"] == 50

    def test_multi_field_content(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_fields=["title", "body"],
            page_content_separator=" | ",
        )
        docs = retriever.invoke("query")
        assert " | " in docs[0].page_content
        assert "Getting Started with FoxNose" in docs[0].page_content

    def test_custom_mapper(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_mapper=lambda r: f"# {r['data']['title']}",
        )
        docs = retriever.invoke("query")
        assert docs[0].page_content == "# Getting Started with FoxNose"

    def test_metadata_whitelist(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            metadata_fields=["title"],
        )
        docs = retriever.invoke("query")
        assert "title" in docs[0].metadata
        assert "category" not in docs[0].metadata

    def test_metadata_blacklist(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            exclude_metadata_fields=["status"],
        )
        docs = retriever.invoke("query")
        assert "status" not in docs[0].metadata
        assert "title" in docs[0].metadata

    def test_empty_results(self, mock_flux_client: MagicMock) -> None:
        mock_flux_client.search.return_value = {"results": [], "limit": 5}
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = retriever.invoke("query")
        assert docs == []

    def test_error_propagation(self, mock_flux_client: MagicMock) -> None:
        mock_flux_client.search.side_effect = RuntimeError("API error")
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(RuntimeError, match="API error"):
            retriever.invoke("query")

    def test_sync_invoke_without_sync_client_raises(self) -> None:
        from unittest.mock import AsyncMock

        retriever = FoxNoseRetriever(
            async_client=AsyncMock(),
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Synchronous retrieval requires"):
            retriever.invoke("query")
