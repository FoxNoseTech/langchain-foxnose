"""Tests for FoxNoseRetriever (async)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.embeddings import Embeddings

from langchain_foxnose import FoxNoseRetriever


class _AsyncMockEmbeddings(Embeddings):
    """Embeddings mock that tracks async calls."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [0.1, 0.2, 0.3]
        self.aembed_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector

    async def aembed_query(self, text: str) -> list[float]:
        self.aembed_query_calls.append(text)
        return self._vector


class TestRetrieverAsync:
    """Async retrieval with AsyncFluxClient."""

    @pytest.mark.asyncio
    async def test_ainvoke_with_async_client(self, mock_async_flux_client: AsyncMock) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
        )
        docs = await retriever.ainvoke("test query")
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."
        assert docs[0].metadata["key"] == "abc123"

    @pytest.mark.asyncio
    async def test_ainvoke_hybrid_calls_hybrid_search(
        self, mock_async_flux_client: AsyncMock
    ) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
        )
        await retriever.ainvoke("test query")
        mock_async_flux_client.hybrid_search.assert_called_once()
        kwargs = mock_async_flux_client.hybrid_search.call_args[1]
        assert kwargs["query"] == "test query"

    @pytest.mark.asyncio
    async def test_ainvoke_vector_calls_vector_search(
        self, mock_async_flux_client: AsyncMock
    ) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            top_k=20,
        )
        await retriever.ainvoke("semantic query")
        mock_async_flux_client.vector_search.assert_called_once()
        kwargs = mock_async_flux_client.vector_search.call_args[1]
        assert kwargs["query"] == "semantic query"
        assert kwargs["top_k"] == 20

    @pytest.mark.asyncio
    async def test_ainvoke_text_calls_search(self, mock_async_flux_client: AsyncMock) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
        )
        await retriever.ainvoke("text query")
        mock_async_flux_client.search.assert_called_once()
        body = mock_async_flux_client.search.call_args[1]["body"]
        assert body["search_mode"] == "text"

    @pytest.mark.asyncio
    async def test_ainvoke_with_both_clients_uses_async(
        self,
        mock_flux_client: MagicMock,
        mock_async_flux_client: AsyncMock,
    ) -> None:
        """When both clients are provided, async path uses async_client."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = await retriever.ainvoke("query")
        # Async client should be called, not sync
        mock_async_flux_client.hybrid_search.assert_called_once()
        mock_flux_client.hybrid_search.assert_not_called()
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_ainvoke_empty_results(self, mock_async_flux_client: AsyncMock) -> None:
        mock_async_flux_client.hybrid_search.return_value = {"results": [], "limit": 5}
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = await retriever.ainvoke("query")
        assert docs == []

    @pytest.mark.asyncio
    async def test_ainvoke_error_propagation(self, mock_async_flux_client: AsyncMock) -> None:
        mock_async_flux_client.hybrid_search.side_effect = RuntimeError("Async API error")
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(RuntimeError, match="Async API error"):
            await retriever.ainvoke("query")

    @pytest.mark.asyncio
    async def test_ainvoke_with_where_filter(self, mock_async_flux_client: AsyncMock) -> None:
        where_filter = {"$": {"all_of": [{"status__eq": "published"}]}}
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            where=where_filter,
        )
        await retriever.ainvoke("query")
        kwargs = mock_async_flux_client.hybrid_search.call_args[1]
        assert kwargs["where"] == where_filter

    @pytest.mark.asyncio
    async def test_ainvoke_text_mode_with_offset_sort_where(
        self, mock_async_flux_client: AsyncMock
    ) -> None:
        """Async text mode passes offset, where, and sort into body."""
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            where={"status__eq": "published"},
            sort=["-title"],
            search_kwargs={"offset": 5},
        )
        await retriever.ainvoke("query")
        body = mock_async_flux_client.search.call_args[1]["body"]
        assert body["offset"] == 5
        assert body["where"] == {"status__eq": "published"}
        assert body["sort"] == ["-title"]

    @pytest.mark.asyncio
    async def test_ainvoke_boosted_without_custom_embeddings(
        self, mock_async_flux_client: AsyncMock
    ) -> None:
        """Async boosted mode without custom embeddings sends query as text."""
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector_boosted",
        )
        await retriever.ainvoke("boosted query")
        mock_async_flux_client.boosted_search.assert_called_once()
        kwargs = mock_async_flux_client.boosted_search.call_args[1]
        assert kwargs["query"] == "boosted query"
        assert "field" not in kwargs

    @pytest.mark.asyncio
    async def test_ainvoke_fallback_to_sync(
        self,
        mock_flux_client: MagicMock,
    ) -> None:
        """When only sync client is provided, ainvoke should fall back to sync via executor."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = await retriever.ainvoke("query")
        # Should have called sync client via executor fallback
        mock_flux_client.hybrid_search.assert_called_once()
        assert len(docs) == 3


class TestRetrieverAsyncEmbeddings:
    """Async retrieval with custom embeddings."""

    @pytest.mark.asyncio
    async def test_ainvoke_vector_field_search_with_embeddings(
        self, mock_async_flux_client: AsyncMock
    ) -> None:
        embeddings = _AsyncMockEmbeddings(vector=[0.1, 0.2, 0.3])

        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            embeddings=embeddings,
            vector_field="embedding",
        )
        docs = await retriever.ainvoke("test query")

        assert embeddings.aembed_query_calls == ["test query"]
        mock_async_flux_client.vector_field_search.assert_called_once()
        kwargs = mock_async_flux_client.vector_field_search.call_args[1]
        assert kwargs["field"] == "embedding"
        assert kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_ainvoke_vector_field_search_with_static_vector(
        self, mock_async_flux_client: AsyncMock
    ) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            query_vector=[0.5, 0.6],
            vector_field="embedding",
        )
        await retriever.ainvoke("ignored")

        mock_async_flux_client.vector_field_search.assert_called_once()
        kwargs = mock_async_flux_client.vector_field_search.call_args[1]
        assert kwargs["query_vector"] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_ainvoke_boosted_with_embeddings(self, mock_async_flux_client: AsyncMock) -> None:
        embeddings = _AsyncMockEmbeddings(vector=[0.1, 0.2])

        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector_boosted",
            embeddings=embeddings,
            vector_field="embedding",
        )
        await retriever.ainvoke("query")

        assert len(embeddings.aembed_query_calls) == 1
        mock_async_flux_client.boosted_search.assert_called_once()
        kwargs = mock_async_flux_client.boosted_search.call_args[1]
        assert kwargs["field"] == "embedding"
        assert kwargs["query_vector"] == [0.1, 0.2]
